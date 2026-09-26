# ================================================================================================================
# TEST 187 — L27 DISPLACEMENT REDISTRIBUTION X-RAY
# AkbasCore 3.2 / SEASC — TEST186 continuation — CAUSE↔EFFECT
#
# TEST185:
#   PRE27 @ORTH19 = +0.715652
#   RAW27 @ORTH19 = -0.297196
#   L27 Δ@ORTH19  = -1.012847
#
# TEST186:
#   L27 bypass restores terminal sign but does NOT rescue behavior.
#   Single-forward ORTH_RESCUE did not cancel TEST185 symmetric displacement.
#
# QUESTION
# Where does the symmetric steering displacement go through L27?
#
# MEASURE:
#   d_pre = .5*(PRE27+ - PRE27-)
#   d_raw = .5*(RAW27+ - RAW27-)
#   Δ27   = d_raw - d_pre
#
# Decompose d_raw / Δ27 against:
#   1) fixed ORTH19
#   2) d_pre direction
#   3) component parallel to d_pre
#   4) component orthogonal to d_pre
#   5) component orthogonal to ORTH19
#   6) PCA/SVD structure of per-prompt Δ27
#   7) pairwise transport cosine / angle / norm growth
#
# X-RAY ONLY — NO INTERVENTION / SEARCH / SELECTION / CONTROLLER / WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 187 — L27 DISPLACEMENT REDISTRIBUTION X-RAY")
print("CAUSE↔EFFECT — PRE27 → RAW27 — FIXED ORTH19 + TRANSPORT GEOMETRY")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=187
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST187_L27_REDISTRIBUTION_XRAY");ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(STEER_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
def cosv(a,b):return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(EPS)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
print(f"SEASC RSS={RSS:.9f} | scale={PRIMARY_SCALE}")

# ================================================================================================================
# [1/26] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test187";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""
#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d);
torch::Tensor seasc_batch(torch::Tensor h,torch::Tensor a,torch::Tensor d){
TORCH_CHECK(h.is_cuda()&&a.is_cuda()&&d.is_cuda(),"CUDA required");return seasc_batch_cuda(h,a,d);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch);}
"""
CUDA=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename scalar_t>
__global__ void k(scalar_t* h,const float* a,const float* d,int B,int S,int H){
int v=blockIdx.x,b=v/S;if(b>=B)return;extern __shared__ float sh[];
long long hb=(long long)v*H,ab=(long long)b*H;float ss=0.f;
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[hb+j];ss+=x*x;}
sh[threadIdx.x]=ss;__syncthreads();
for(unsigned s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
float z=d[b]*sqrtf(fmaxf(sh[0],1e-20f));
for(int j=threadIdx.x;j<H;j+=blockDim.x)h[hb+j]=(scalar_t)((float)h[hb+j]+z*a[ab+j]);}
torch::Tensor seasc_batch_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d){
auto o=h.contiguous().clone();auto aa=a.to(h.device(),torch::kFloat32).contiguous();
auto dd=d.to(h.device(),torch::kFloat32).contiguous();int B=o.size(0),S=o.size(1),H=o.size(2);
constexpr int T=256;cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,o.scalar_type(),"seasc",[&]{
k<scalar_t><<<B*S,T,T*sizeof(float),stream>>>(o.data_ptr<scalar_t>(),aa.data_ptr<float>(),dd.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
"""
print("\n[1/26] CUDA...")
seasc_ext=load_inline(name="akbascore_test187_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD,verbose=False)
print("       OK")

# ================================================================================================================
# [2/26] MODEL
# ================================================================================================================
print("\n[2/26] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},
 attn_implementation="sdpa",trust_remote_code=True)
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;FINAL_NORM=model.model.norm
if len(layers)!=TOTAL_LAYERS:raise RuntimeError(f"Expected {TOTAL_LAYERS} layers, got {len(layers)}")
print("       hidden:",H,"dtype:",next(model.parameters()).dtype,"layers:",len(layers))

def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],
 tokenize=False,add_generation_prompt=True)

@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt").to(DEVICE)
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);p=int(e.attention_mask[0].sum())-1
    return [o.hidden_states[L+1][0,p].float().detach() for L in range(STEER_LAYERS)]

def sequence(ctx,cont):
    p=tok(chat(ctx),return_tensors="pt",add_special_tokens=False);c=tok(cont,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p.input_ids,c.input_ids],1).to(DEVICE);return ids,torch.ones_like(ids),p.input_ids.shape[1]

def actgrad(ctx,cont,positive=False):
    ids,mask,plen=sequence(ctx,cont);emb=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    saved=[None]*STEER_LAYERS;hooks=[]
    for L in range(STEER_LAYERS):
        def mk(li):
            def hk(m,a,o):
                h=o[0] if isinstance(o,tuple) else o;h.retain_grad();saved[li]=h
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    model.zero_grad(set_to_none=True)
    try:
        o=model(inputs_embeds=emb,attention_mask=mask,use_cache=False,return_dict=True)
        z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:]
        sc=F.log_softmax(z,-1).gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean()
        (sc if positive else -sc).backward()
        g=[saved[L].grad[0].float().sum(0).detach().clone() for L in range(STEER_LAYERS)]
    finally:
        for h in hooks:h.remove()
        model.zero_grad(set_to_none=True)
    return g

# ================================================================================================================
# [3/26] FROZEN E AXIS
# ================================================================================================================
print("\n[3/26] Frozen E axis...")
PAIRS=[
("The spark is the cause of the fire.","The fire is the effect of the spark."),
("The impact is the cause of the crack.","The crack is the effect of the impact."),
("The rainfall is the cause of the flooding.","The flooding is the effect of the rainfall."),
("The heat is the cause of the melting.","The melting is the effect of the heat."),
("The collision is the cause of the damage.","The damage is the effect of the collision."),
("The pressure is the cause of the deformation.","The deformation is the effect of the pressure."),
("The infection is the cause of the fever.","The fever is the effect of the infection."),
("The vibration is the cause of the loosening.","The loosening is the effect of the vibration."),
("The frost is the cause of the cracking.","The cracking is the effect of the frost."),
("The blockage is the cause of the overflow.","The overflow is the effect of the blockage."),
("The friction is the cause of the heating.","The heating is the effect of the friction."),
("The force is the cause of the motion.","The motion is the effect of the force."),
("The outage is the cause of the shutdown.","The shutdown is the effect of the outage."),
("The leak is the cause of the pressure loss.","The pressure loss is the effect of the leak."),
("The acceleration is the cause of the displacement.","The displacement is the effect of the acceleration."),
("The trigger is the cause of the response.","The response is the effect of the trigger.")]

# ================================================================================================================
# [4/26] FISHER/BANK
# ================================================================================================================
print("\n[4/26] Fisher/Bank...")
PB=[[] for _ in range(STEER_LAYERS)];NB=[[] for _ in range(STEER_LAYERS)]
for i,(p,n) in enumerate(PAIRS,1):
    hp,hn=capture(p),capture(n)
    for L in range(STEER_LAYERS):PB[L].append(hp[L]);NB[L].append(hn[L])
    print(f"       {i:02d}/16")
PB=[torch.stack(x).float() for x in PB];NB=[torch.stack(x).float() for x in NB]
FISHER=[];BANK=[];FB=[]
for L in range(STEER_LAYERS):
    hp,hn=PB[L],NB[L];mp,mn=hp.mean(0),hn.mean(0);md=mp-mn;rr=torch.cat([hp-mp,hn-mn])
    _,s,vh=torch.linalg.svd(rr,full_matrices=False);eig=s.square()/max(rr.shape[0]-2,1)
    pe=eig[eig>FISHER_EPS];es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS);V=vh.T;pr=V.T@md
    f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float();dirs=norm(hp-hn)
    dirs*=torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
    b=norm(dirs.mean(0)).float();c=torch.dot(f,b)
    if c<0:b=-b;c=-c
    FISHER.append(f);BANK.append(b);FB.append(float(c))
FISHER=torch.stack(FISHER);BANK=torch.stack(BANK)
print(f"       mean cos(F,B)={np.mean(FB):+.6f}")

# ================================================================================================================
# [5/26] P8
# ================================================================================================================
print("\n[5/26] P8...")
RET=[
("A ceramic cup rests beside a folded newspaper."," The scene contains ordinary stationary objects."),
("A metal ruler lies beside a closed notebook."," The scene contains ordinary stationary objects."),
("A wooden bowl sits near a plain cloth."," The scene contains ordinary stationary objects."),
("A glass jar rests beside a cardboard package."," The scene contains ordinary stationary objects."),
("A calculator lies next to a paper envelope."," The objects remain stationary."),
("A closed umbrella rests beside a travel bag."," The objects remain stationary."),
("A spoon lies beside an empty plate."," The objects remain stationary."),
("A pencil rests next to a blank sheet of paper."," The objects remain stationary.")]
PG=[]
for i,x in enumerate(RET,1):PG.append(actgrad(*x));print(f"       {i}/8")
Q8=[];ranks=[]
for L in range(STEER_LAYERS):
    G=norm(torch.stack([g[L] for g in PG]));_,s,vh=torch.linalg.svd(G,full_matrices=False)
    r=int((s>max(float(s[0])*SVD_REL_TOL,EPS)).sum());Q8.append(vh[:r].T.contiguous());ranks.append(r)
print("       ranks:",ranks)
def guard(x):
    z=[]
    for L in range(STEER_LAYERS):
        q=Q8[L];v=x[L]-(q@(q.T@x[L]) if q.shape[1] else 0);z.append(norm(v))
    return torch.stack(z).contiguous()
FG=guard(FISHER)

# ================================================================================================================
# [6/26] FROZEN FORGE
# ================================================================================================================
print("\n[6/26] Frozen forge...")
FORGE={
"A":[
("A spark and a fire are related causally. Which item is the initiating event?"," The spark is the cause."," The fire is the effect."),
("An impact and a crack are related causally. Which item is the initiating event?"," The impact is the cause."," The crack is the effect."),
("Rainfall and flooding are related causally. Which item is the initiating event?"," The rainfall is the cause."," The flooding is the effect."),
("Heat and melting are related causally. Which item is the initiating event?"," The heat is the cause."," The melting is the effect.")],
"B":[
("A collision produces damage. Identify the causal role of the collision rather than its consequence."," The collision produces the damage."," The damage results from the collision."),
("Pressure produces deformation. Identify the causal role of the pressure rather than its consequence."," The pressure produces the deformation."," The deformation results from the pressure."),
("An infection produces fever. Identify the causal role of the infection rather than its consequence."," The infection produces the fever."," The fever results from the infection."),
("Vibration produces loosening. Identify the causal role of the vibration rather than its consequence."," The vibration produces the loosening."," The loosening results from the vibration.")],
"C":[
("Consider the directional causal relation between frost and cracking."," Causally, frost precedes and generates the cracking."," Causally, cracking follows from the frost."),
("Consider the directional causal relation between blockage and overflow."," Causally, blockage precedes and generates the overflow."," Causally, overflow follows from the blockage."),
("Consider the directional causal relation between friction and heating."," Causally, friction precedes and generates the heating."," Causally, heating follows from the friction."),
("Consider the directional causal relation between force and motion."," Causally, force precedes and generates the motion."," Causally, motion follows from the force.")],
"D":[
("An outage and a shutdown form a cause-effect pair. Select the upstream member."," The outage is upstream in the causal chain."," The shutdown is downstream in the causal chain."),
("A leak and pressure loss form a cause-effect pair. Select the upstream member."," The leak is upstream in the causal chain."," The pressure loss is downstream in the causal chain."),
("Acceleration and displacement form a cause-effect pair. Select the upstream member."," The acceleration is upstream in the causal chain."," The displacement is downstream in the causal chain."),
("A trigger and a response form a cause-effect pair. Select the upstream member."," The trigger is upstream in the causal chain."," The response is downstream in the causal chain.")]}
FAM={}
for fn,items in FORGE.items():
    acc=[torch.zeros(H,device=DEVICE) for _ in range(STEER_LAYERS)]
    for j,(ctx,p,n) in enumerate(items,1):
        gp,gn=actgrad(ctx,p,True),actgrad(ctx,n,True)
        for L in range(STEER_LAYERS):acc[L]+=gp[L]-gn[L]
        print(f"       {fn} {j}/4")
    v=torch.stack([norm(x) for x in acc])
    for L in range(STEER_LAYERS):
        if torch.dot(v[L],FISHER[L])<0:v[L]=-v[L]
    FAM[fn]=v
CAUSAL=torch.stack([norm(torch.stack([FAM[f][L] for f in FAM]).mean(0)) for L in range(STEER_LAYERS)])
CG=guard(CAUSAL);ORTH=[]
for L in range(STEER_LAYERS):ORTH.append(norm(CG[L]-torch.dot(CG[L],FG[L])*FG[L]))
ORTH=torch.stack(ORTH).contiguous();U=ORTH[19]
print("       ORTH ready | fixed coordinate=ORTH19")

# ================================================================================================================
# [7/26] FRESH XRAY
# ================================================================================================================
print("\n[7/26] Fresh X-ray...")
XRAY=[
"A fractured pipe and a pressure drop form a causal sequence.",
"A blocked air passage and rising temperature form a causal sequence.",
"A damaged seal and fluid loss form a causal sequence.",
"A current surge and a failed circuit form a causal sequence.",
"A stuck mechanism and restricted motion form a causal sequence.",
"A worn bearing and increased vibration form a causal sequence.",
"A clogged outlet and accumulated liquid form a causal sequence.",
"A strong impact and a deformed bracket form a causal sequence."]
print("       n=8")

# ================================================================================================================
# [8/26] EXACT PRE27 / RAW27 RECORDER
# ================================================================================================================
print("\n[8/26] Exact PRE27 / RAW27 recorder...")
def run_exact(prompt,sign):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    box={};hooks=[]
    for L in range(STEER_LAYERS):
        def mk(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out
                B=raw.shape[0];a=(float(sign)*ORTH[li][None].expand(B,-1)).float().contiguous()
                d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                new=seasc_ext.seasc_batch(raw,a,d)
                if li==19:box["post19"]=new[0,pos].float().detach().clone()
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    def pre27(m,args):box["pre27"]=args[0][0,pos].float().detach().clone()
    def raw27(m,args,out):
        h=out[0] if isinstance(out,tuple) else out
        box["raw27"]=h[0,pos].float().detach().clone()
    def npre(m,args):box["norm_in"]=args[0][0,pos].float().detach().clone()
    hooks.append(layers[27].register_forward_pre_hook(pre27))
    hooks.append(layers[27].register_forward_hook(raw27))
    hooks.append(FINAL_NORM.register_forward_pre_hook(npre))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    return box

# ================================================================================================================
# [9/26] PLUS / MINUS
# ================================================================================================================
print("\n[9/26] PLUS / MINUS...")
RUN={"PLUS":[],"MINUS":[]}
for i,p in enumerate(XRAY,1):
    RUN["PLUS"].append(run_exact(p,+1));RUN["MINUS"].append(run_exact(p,-1))
    print(f"       {i}/8")

# ================================================================================================================
# [10/26] SYMMETRIC DISPLACEMENTS
# ================================================================================================================
print("\n[10/26] Symmetric displacements...")
D={}
for k in ["post19","pre27","raw27","norm_in"]:
    P=torch.stack([x[k] for x in RUN["PLUS"]]);M=torch.stack([x[k] for x in RUN["MINUS"]])
    D[k]=.5*(P-M)
DPRE=D["pre27"];DRAW=D["raw27"];DELTA=DRAW-DPRE
print("       ready")

# ================================================================================================================
# [11/26] CONTINUITY
# ================================================================================================================
print("\n[11/26] RAW27→NORM_IN continuity...")
err=(DRAW-D["norm_in"]).norm(-1);rel=err/DRAW.norm(dim=-1).clamp_min(EPS)
print(f"       relerr={float(rel.mean()):.10f}")

# ================================================================================================================
# [12/26] ORTH19 LOSS
# ================================================================================================================
print("\n[12/26] Fixed ORTH19 projection...")
pre_u=DPRE@U;raw_u=DRAW@U;del_u=DELTA@U
print(f"       PRE27  ={float(pre_u.mean()):+.6f}")
print(f"       RAW27  ={float(raw_u.mean()):+.6f}")
print(f"       Δ27    ={float(del_u.mean()):+.6f}")
print(f"       P+ PRE ={int((pre_u>0).sum())}/8")
print(f"       P+ RAW ={int((raw_u>0).sum())}/8")

# ================================================================================================================
# [13/26] PRE→RAW TRANSPORT GEOMETRY
# ================================================================================================================
print("\n[13/26] PRE27→RAW27 transport geometry...")
transport_cos=cosv(DPRE,DRAW)
transport_ang=torch.rad2deg(torch.acos(transport_cos.clamp(-1,1)))
growth=DRAW.norm(dim=-1)/DPRE.norm(dim=-1).clamp_min(EPS)
print(f"       cos(PRE,RAW)={float(transport_cos.mean()):+.6f}")
print(f"       angle       ={float(transport_ang.mean()):.3f}°")
print(f"       norm PRE    ={float(DPRE.norm(dim=-1).mean()):.6f}")
print(f"       norm RAW    ={float(DRAW.norm(dim=-1).mean()):.6f}")
print(f"       growth      ={float(growth.mean()):.6f}x")

# ================================================================================================================
# [14/26] DECOMPOSE RAW AGAINST PRE DIRECTION
# ================================================================================================================
print("\n[14/26] RAW decomposition relative to PRE direction...")
PREH=norm(DPRE)
alpha=(DRAW*PREH).sum(-1)
RAW_PAR=alpha[:,None]*PREH
RAW_PERP=DRAW-RAW_PAR
frac_par=RAW_PAR.norm(dim=-1)/DRAW.norm(dim=-1).clamp_min(EPS)
frac_perp=RAW_PERP.norm(dim=-1)/DRAW.norm(dim=-1).clamp_min(EPS)
print(f"       parallel coefficient={float(alpha.mean()):+.6f}")
print(f"       parallel norm frac  ={float(frac_par.mean()):.6f}")
print(f"       orthogonal norm frac={float(frac_perp.mean()):.6f}")

# ================================================================================================================
# [15/26] DECOMPOSE Δ27 AGAINST PRE DIRECTION
# ================================================================================================================
print("\n[15/26] Δ27 decomposition relative to PRE direction...")
da=(DELTA*PREH).sum(-1)
DEL_PAR=da[:,None]*PREH
DEL_PERP=DELTA-DEL_PAR
print(f"       Δ parallel={float(da.mean()):+.6f}")
print(f"       ||Δparallel||={float(DEL_PAR.norm(dim=-1).mean()):.6f}")
print(f"       ||Δperp||    ={float(DEL_PERP.norm(dim=-1).mean()):.6f}")
print(f"       perp/Δ       ={float((DEL_PERP.norm(dim=-1)/DELTA.norm(dim=-1).clamp_min(EPS)).mean()):.6f}")

# ================================================================================================================
# [16/26] ORTH19-ORTHOGONAL REDISTRIBUTION
# ================================================================================================================
print("\n[16/26] ORTH19-orthogonal redistribution...")
PRE_U=pre_u[:,None]*U
RAW_U=raw_u[:,None]*U
DEL_U=del_u[:,None]*U
PRE_O=DPRE-PRE_U;RAW_O=DRAW-RAW_U;DEL_O=DELTA-DEL_U
print(f"       PRE ||orth||={float(PRE_O.norm(dim=-1).mean()):.6f}")
print(f"       RAW ||orth||={float(RAW_O.norm(dim=-1).mean()):.6f}")
print(f"       Δ   ||orth||={float(DEL_O.norm(dim=-1).mean()):.6f}")
print(f"       RAW orth frac={float((RAW_O.norm(dim=-1)/DRAW.norm(dim=-1).clamp_min(EPS)).mean()):.6f}")
print(f"       Δ orth frac  ={float((DEL_O.norm(dim=-1)/DELTA.norm(dim=-1).clamp_min(EPS)).mean()):.6f}")

# ================================================================================================================
# [17/26] ENERGY ACCOUNTING
# ================================================================================================================
print("\n[17/26] Energy accounting...")
pre2=DPRE.square().sum(-1);raw2=DRAW.square().sum(-1);del2=DELTA.square().sum(-1)
u_pre=pre_u.square();u_raw=raw_u.square();u_del=del_u.square()
print(f"       PRE energy={float(pre2.mean()):.6f} ORTH19 share={float((u_pre/pre2.clamp_min(EPS)).mean()):.8f}")
print(f"       RAW energy={float(raw2.mean()):.6f} ORTH19 share={float((u_raw/raw2.clamp_min(EPS)).mean()):.8f}")
print(f"       Δ27 energy={float(del2.mean()):.6f} ORTH19 share={float((u_del/del2.clamp_min(EPS)).mean()):.8f}")

# ================================================================================================================
# [18/26] PER-PROMPT
# ================================================================================================================
print("\n[18/26] Per-prompt transport...")
ROWS=[]
for i in range(8):
    r={"item":i+1,"pre_u":float(pre_u[i]),"raw_u":float(raw_u[i]),"delta_u":float(del_u[i]),
       "transport_cos":float(transport_cos[i]),"angle":float(transport_ang[i]),"growth":float(growth[i]),
       "delta_norm":float(DELTA[i].norm()),"delta_orth_norm":float(DEL_O[i].norm())}
    ROWS.append(r)
    print(f"       {i+1}: U {r['pre_u']:+.4f}→{r['raw_u']:+.4f} Δ={r['delta_u']:+.4f} cos={r['transport_cos']:+.4f} angle={r['angle']:.2f}°")

# ================================================================================================================
# [19/26] SVD / PCA OF Δ27
# ================================================================================================================
print("\n[19/26] Δ27 SVD/PCA...")
X=DELTA-DELTA.mean(0,keepdim=True)
_,S,Vh=torch.linalg.svd(X,full_matrices=False)
var=S.square();ratio=var/var.sum().clamp_min(EPS);cum=torch.cumsum(ratio,0)
for k in range(min(8,len(S))):
    print(f"       PC{k+1}: {float(ratio[k])*100:.3f}% cumulative={float(cum[k])*100:.3f}%")
PC1=Vh[0]
if torch.dot(PC1,DELTA.mean(0))<0:PC1=-PC1
print(f"       cos(PC1,ORTH19)={float(torch.dot(PC1,U)):+.6f}")
print(f"       cos(meanΔ,ORTH19)={float(torch.dot(norm(DELTA.mean(0)),U)):+.6f}")

# ================================================================================================================
# [20/26] MEAN Δ DIRECTION
# ================================================================================================================
print("\n[20/26] Mean L27 transformation direction...")
MEAN_DEL=DELTA.mean(0);MD=norm(MEAN_DEL)
print(f"       ||meanΔ||={float(MEAN_DEL.norm()):.6f}")
print(f"       meanΔ·ORTH19={float(torch.dot(MEAN_DEL,U)):+.6f}")
print(f"       cos(meanΔ,ORTH19)={float(torch.dot(MD,U)):+.6f}")
print(f"       angle(meanΔ,ORTH19)={float(torch.rad2deg(torch.acos(torch.dot(MD,U).clamp(-1,1)))):.3f}°")

# ================================================================================================================
# [21/26] BOOTSTRAP
# ================================================================================================================
print("\n[21/26] Bootstrap...")
def boot(x,n=20000,seed=187):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT=[]
tests=[
("ORTH19_DELTA",del_u.cpu().numpy()),
("TRANSPORT_COS",transport_cos.cpu().numpy()),
("GROWTH",growth.cpu().numpy()),
("DELTA_ORTH_FRAC",(DEL_O.norm(dim=-1)/DELTA.norm(dim=-1).clamp_min(EPS)).cpu().numpy())]
for j,(name,x) in enumerate(tests):
    m,lo,hi=boot(x,seed=SEED+j);BOOT.append({"metric":name,"mean":m,"low":lo,"high":hi})
    print(f"       {name:<18} {m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [22/26] REDISTRIBUTION TEST
# ================================================================================================================
print("\n[22/26] Redistribution test...")
ORTH_LOSS=float(del_u.mean())
ORTH_FRAC=float((DEL_O.norm(dim=-1)/DELTA.norm(dim=-1).clamp_min(EPS)).mean())
TC=float(transport_cos.mean())
print(f"       L27 Δ@ORTH19={ORTH_LOSS:+.6f}")
print(f"       Δ orthogonal fraction={ORTH_FRAC:.6f}")
print(f"       PRE→RAW cosine={TC:+.6f}")

# ================================================================================================================
# [23/26] DECISION
# ================================================================================================================
print("\n[23/26] TEST187 decision...")
ci=next(x for x in BOOT if x["metric"]=="ORTH19_DELTA")
if ci["high"]<0 and ORTH_FRAC>.80:
    DECISION="L27_ORTH19_LOSS_OCCURS_WITH_HIGH_DIMENSIONAL_REDISTRIBUTION"
elif ci["high"]<0 and TC<0:
    DECISION="L27_ORTH19_LOSS_WITH_GLOBAL_DISPLACEMENT_REVERSAL_SUPPORTED"
elif ci["high"]<0:
    DECISION="L27_ORTH19_LOSS_REPLICATED__REDISTRIBUTION_STRUCTURE_MIXED"
else:
    DECISION="L27_ORTH19_LOSS_NOT_REPLICATED"
print("       DECISION:",DECISION)

# ================================================================================================================
# [24/26] SAVE
# ================================================================================================================
print("\n[24/26] Saving...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("PER_PROMPT.csv",ROWS);savecsv("BOOTSTRAP.csv",BOOT)
np.savez_compressed(ROOT/"TEST187_L27_REDISTRIBUTION.npz",
 orth19=U.cpu().numpy(),pre27=DPRE.cpu().numpy(),raw27=DRAW.cpu().numpy(),
 delta27=DELTA.cpu().numpy(),delta_orth19_orthogonal=DEL_O.cpu().numpy(),
 pc_vectors=Vh.cpu().numpy(),pc_ratio=ratio.cpu().numpy())
REPORT={"test":"TEST187","parent":"TEST186","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
 "pre27_orth19":float(pre_u.mean()),"raw27_orth19":float(raw_u.mean()),"delta27_orth19":ORTH_LOSS,
 "transport_cos":TC,"transport_angle":float(transport_ang.mean()),"growth":float(growth.mean()),
 "delta_orthogonal_fraction":ORTH_FRAC,"pc1_ratio":float(ratio[0]),"pc12_ratio":float(ratio[:2].sum()),
 "pc1_orth19_cos":float(torch.dot(PC1,U)),"mean_delta_orth19_cos":float(torch.dot(MD,U)),
 "bootstrap":BOOT,
 "integrity":{"test185_186_lineage":True,"seasc_l0_l19_frozen":True,"l20_l27_untouched":True,
 "coordinate":"fixed ORTH19","intervention":False,"selection":False,"controller":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [25/26] SCIENTIFIC BOUNDARY
# ================================================================================================================
print("\n[25/26] Scientific boundary...")
print("       TEST187 measures the symmetric PLUS/MINUS displacement, not single-forward residual motion.")
print("       L27 is untouched; no bypass, rescue or projection intervention is applied.")
print("       ORTH19 loss and orthogonal displacement are geometric observations.")
print("       Orthogonal redistribution does not by itself identify where behavioral information is encoded.")
print("       PCA/SVD is descriptive only; no component is selected for steering.")

# ================================================================================================================
# [26/26] COMPLETE
# ================================================================================================================
print("\n[26/26] COMPLETE")
print("="*138)
print("TEST 187 — L27 DISPLACEMENT REDISTRIBUTION X-RAY — SUMMARY")
print("="*138)
print(f"PRE27 @ORTH19={float(pre_u.mean()):+.6f}")
print(f"RAW27 @ORTH19={float(raw_u.mean()):+.6f}")
print(f"L27 Δ @ORTH19={ORTH_LOSS:+.6f}")
print(f"PRE→RAW cos={TC:+.6f} angle={float(transport_ang.mean()):.3f}° growth={float(growth.mean()):.6f}x")
print(f"Δ27 orthogonal fraction={ORTH_FRAC:.6f}")
print(f"PC1={float(ratio[0])*100:.3f}% | PC1+PC2={float(ratio[:2].sum())*100:.3f}%")
print(f"cos(PC1,ORTH19)={float(torch.dot(PC1,U)):+.6f}")
print(f"cos(meanΔ,ORTH19)={float(torch.dot(MD,U)):+.6f}")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
