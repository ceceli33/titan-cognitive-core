# ================================================================================================================
# TEST 185 — UNSTEERED TAIL TRANSPORT X-RAY
# AkbasCore 3.2 / SEASC — TEST184 continuation — CAUSE↔EFFECT
#
# PURPOSE
# Localize TEST184 gap:
#   L19 POST @ ORTH19 > 0
#   final RMSNorm input @ ORTH19 < 0
#
# EXACT PATH:
#   POST19 → PRE20/RAW20 → ... → PRE27/RAW27 → NORM_IN → NORM_OUT
#
# IMPORTANT
#   SEASC remains active ONLY L0-L19.
#   L20-L27 are untouched model tail layers.
#   Tail is measured in ONE FIXED coordinate: ORTH19.
#   No moving-compass interpretation is used inside the tail.
#
# FROZEN: model / E axis / P8 / causal forge / SEASC / rho / scale=.50
# X-RAY ONLY — NO SEARCH / SELECTION / CONTROLLER / WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 185 — UNSTEERED TAIL TRANSPORT X-RAY")
print("CAUSE↔EFFECT — L19 POST → L20-L27 → FINAL RMSNORM")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=185
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST185_UNSTEERED_TAIL_XRAY");ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(STEER_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
print(f"SEASC RSS={RSS:.9f} | scale={PRIMARY_SCALE}")

# ================================================================================================================
# [1/24] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test185";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
print("\n[1/24] CUDA...")
seasc_ext=load_inline(name="akbascore_test185_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD,verbose=False)
print("       OK")

# ================================================================================================================
# [2/24] MODEL
# ================================================================================================================
print("\n[2/24] Model...")
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
    x=[o.hidden_states[L+1][0,p].float().detach() for L in range(STEER_LAYERS)]
    del e,o;return x

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
# [3/24] FROZEN E AXIS
# ================================================================================================================
print("\n[3/24] Frozen E axis...")
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
# [4/24] FISHER/BANK
# ================================================================================================================
print("\n[4/24] Fisher/Bank...")
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
# [5/24] P8
# ================================================================================================================
print("\n[5/24] P8...")
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
# [6/24] FROZEN FORGE
# ================================================================================================================
print("\n[6/24] Frozen forge...")
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
print("       ORTH ready | fixed tail coordinate=ORTH19")

# ================================================================================================================
# [7/24] FRESH XRAY
# ================================================================================================================
print("\n[7/24] Fresh X-ray...")
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
# [8/24] EXACT L19→L27 RECORDER
# ================================================================================================================
print("\n[8/24] Exact L19→L27 tail recorder...")
def run_exact(prompt,sign=0.0):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    BOX={"pre":{},"raw":{},"post19":None,"norm_in":None,"norm_out":None};hooks=[]
    for L in range(TOTAL_LAYERS):
        if L>=20:
            def prehook(li):
                def hk(m,args):BOX["pre"][li]=args[0][0,pos].float().detach().clone()
                return hk
            hooks.append(layers[L].register_forward_pre_hook(prehook(L)))
        def posthook(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out
                if li<STEER_LAYERS and sign!=0:
                    B=raw.shape[0];a=(float(sign)*ORTH[li][None].expand(B,-1)).float().contiguous()
                    d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                    new=seasc_ext.seasc_batch(raw,a,d)
                else:new=raw
                if li==19:BOX["post19"]=new[0,pos].float().detach().clone()
                if li>=20:BOX["raw"][li]=new[0,pos].float().detach().clone()
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_hook(posthook(L)))
    def npre(m,args):BOX["norm_in"]=args[0][0,pos].float().detach().clone()
    def npost(m,args,out):BOX["norm_out"]=out[0,pos].float().detach().clone()
    hooks.append(FINAL_NORM.register_forward_pre_hook(npre));hooks.append(FINAL_NORM.register_forward_hook(npost))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    return BOX

# ================================================================================================================
# [9/24] CAPTURE
# ================================================================================================================
print("\n[9/24] BASE / PLUS / MINUS...")
RUN={"BASE":[],"PLUS":[],"MINUS":[]}
for i,p in enumerate(XRAY,1):
    RUN["BASE"].append(run_exact(p,0));RUN["PLUS"].append(run_exact(p,+1));RUN["MINUS"].append(run_exact(p,-1))
    print(f"       {i}/8")

# ================================================================================================================
# [10/24] STACK
# ================================================================================================================
print("\n[10/24] Symmetric tail deltas...")
def stack_key(arm,key):
    if key=="post19":return torch.stack([r["post19"] for r in RUN[arm]])
    if key=="norm_in":return torch.stack([r["norm_in"] for r in RUN[arm]])
    if key=="norm_out":return torch.stack([r["norm_out"] for r in RUN[arm]])
    typ,L=key
    return torch.stack([r[typ][L] for r in RUN[arm]])
KEYS=["post19"]+[("pre",L) for L in range(20,28)]+[("raw",L) for L in range(20,28)]+["norm_in","norm_out"]
S={k:{a:stack_key(a,k) for a in RUN} for k in KEYS}
D={k:.5*(S[k]["PLUS"]-S[k]["MINUS"]) for k in KEYS}

# ================================================================================================================
# [11/24] PHYSICAL CONTINUITY
# ================================================================================================================
print("\n[11/24] Physical continuity...")
CONT=[]
pairs=[("POST19→PRE20","post19",("pre",20))]
for L in range(20,27):pairs.append((f"RAW{L}→PRE{L+1}",("raw",L),("pre",L+1)))
pairs.append(("RAW27→NORM_IN",("raw",27),"norm_in"))
for name,a,b in pairs:
    er=(D[a]-D[b]).norm(dim=-1);den=D[a].norm(dim=-1).clamp_min(EPS);rel=er/den
    r={"transition":name,"abs":float(er.mean()),"rel":float(rel.mean())};CONT.append(r)
    print(f"       {name:<16} relerr={r['rel']:.10f}")

# ================================================================================================================
# [12/24] FIXED ORTH19 TRAJECTORY
# ================================================================================================================
print("\n[12/24] Fixed ORTH19 tail trajectory...")
ORDER=[("POST19","post19")]
for L in range(20,28):ORDER.extend([(f"PRE{L}",("pre",L)),(f"RAW{L}",("raw",L))])
ORDER.extend([("NORM_IN","norm_in"),("NORM_OUT","norm_out")])
TRAJ=[]
for name,k in ORDER:
    x=D[k];tar=x@U;dn=x.norm(dim=-1).clamp_min(EPS);co=tar/dn
    r={"stage":name,"target":float(tar.mean()),"cos":float(co.mean()),"norm":float(dn.mean()),"positive":int((tar>0).sum())}
    TRAJ.append(r)
    print(f"       {name:<8} target={r['target']:+.6f} cos={r['cos']:+.6f} norm={r['norm']:.6f} P+={r['positive']}/8")

# ================================================================================================================
# [13/24] BLOCKWISE EFFECT
# ================================================================================================================
print("\n[13/24] Tail block contributions @ORTH19...")
BLOCK=[]
for L in range(20,28):
    pre=D[("pre",L)]@U;raw=D[("raw",L)]@U;delta=raw-pre
    r={"layer":L,"pre":float(pre.mean()),"raw":float(raw.mean()),"delta":float(delta.mean())}
    BLOCK.append(r)
    print(f"       L{L}: {r['pre']:+.6f} → {r['raw']:+.6f} Δ={r['delta']:+.6f}")

# ================================================================================================================
# [14/24] FIRST SIGN CROSSING
# ================================================================================================================
print("\n[14/24] First mean sign crossing...")
SEQ=[("POST19",D["post19"]@U)]+[(f"RAW{L}",D[("raw",L)]@U) for L in range(20,28)]+[("NORM_IN",D["norm_in"]@U)]
FIRST=None
for i in range(1,len(SEQ)):
    a=float(SEQ[i-1][1].mean());b=float(SEQ[i][1].mean())
    if a>=0 and b<0:FIRST=(SEQ[i-1][0],SEQ[i][0],a,b);break
if FIRST:print(f"       {FIRST[0]} {FIRST[2]:+.6f} → {FIRST[1]} {FIRST[3]:+.6f}")
else:print("       no positive→negative mean crossing")

# ================================================================================================================
# [15/24] PER-PROMPT CROSSING
# ================================================================================================================
print("\n[15/24] Per-prompt first crossing...")
PROMPT_CROSS=[]
for i in range(len(XRAY)):
    seq=[("POST19",float((D["post19"][i]@U)))]+[(f"RAW{L}",float(D[("raw",L)][i]@U)) for L in range(20,28)]
    hit="NONE"
    for j in range(1,len(seq)):
        if seq[j-1][1]>=0 and seq[j][1]<0:hit=f"{seq[j-1][0]}→{seq[j][0]}";break
    PROMPT_CROSS.append({"prompt":i+1,"crossing":hit})
    print(f"       {i+1}: {hit}")

# ================================================================================================================
# [16/24] CUMULATIVE CHANGE
# ================================================================================================================
print("\n[16/24] Cumulative ORTH19 change from POST19...")
START=D["post19"]@U
CUM=[]
for L in range(20,28):
    y=D[("raw",L)]@U;d=y-START
    r={"layer":L,"target":float(y.mean()),"change_from_post19":float(d.mean())};CUM.append(r)
    print(f"       POST19→RAW{L}: target={r['target']:+.6f} cumulative Δ={r['change_from_post19']:+.6f}")

# ================================================================================================================
# [17/24] RMSNORM SEPARATION
# ================================================================================================================
print("\n[17/24] Tail vs RMSNorm separation...")
p=float((D["post19"]@U).mean());r27=float((D[("raw",27)]@U).mean())
ni=float((D["norm_in"]@U).mean());no=float((D["norm_out"]@U).mean())
print(f"       POST19  ={p:+.6f}")
print(f"       RAW27   ={r27:+.6f} tail Δ={r27-p:+.6f}")
print(f"       NORM_IN ={ni:+.6f} continuity Δ={ni-r27:+.9f}")
print(f"       NORM_OUT={no:+.6f} RMSNorm Δ={no-ni:+.6f}")

# ================================================================================================================
# [18/24] BOOTSTRAP BLOCK EFFECTS
# ================================================================================================================
print("\n[18/24] Bootstrap tail block effects...")
def boot(x,n=20000,seed=185):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT=[]
for j,L in enumerate(range(20,28)):
    x=((D[("raw",L)]-D[("pre",L)])@U).cpu().numpy();m,lo,hi=boot(x,seed=SEED+j)
    BOOT.append({"layer":L,"mean":m,"low":lo,"high":hi})
    print(f"       L{L} Δ={m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [19/24] BOOTSTRAP TOTAL TAIL
# ================================================================================================================
print("\n[19/24] Bootstrap total tail transport...")
x=((D[("raw",27)]-D["post19"])@U).cpu().numpy();TM,TLO,THI=boot(x,seed=194)
print(f"       POST19→RAW27 Δ={TM:+.6f} [{TLO:+.6f},{THI:+.6f}]")
x=((D["norm_out"]-D["norm_in"])@U).cpu().numpy();NM,NLO,NHI=boot(x,seed=195)
print(f"       RMSNorm Δ       ={NM:+.6f} [{NLO:+.6f},{NHI:+.6f}]")

# ================================================================================================================
# [20/24] DOMINANT TAIL LAYER
# ================================================================================================================
print("\n[20/24] Dominant negative tail contribution...")
neg=[r for r in BLOCK if r["delta"]<0]
DOM=min(neg,key=lambda r:r["delta"]) if neg else None
if DOM:print(f"       L{DOM['layer']} Δ={DOM['delta']:+.6f}")
else:print("       none")

# ================================================================================================================
# [21/24] DECISION
# ================================================================================================================
print("\n[21/24] TEST185 decision...")
tail_cross=(p>0 and r27<0)
norm_cross=(ni>0 and no<0)
if tail_cross and FIRST:
    DECISION=f"TERMINAL_SIGN_REVERSAL_LOCALIZED_TO_UNSTEERED_TAIL__FIRST_CROSSING_{FIRST[0]}_TO_{FIRST[1]}"
elif p>0 and ni<0:
    DECISION="TERMINAL_SIGN_REVERSAL_LOCALIZED_TO_UNSTEERED_L20_L27_TAIL"
elif norm_cross:
    DECISION="TERMINAL_SIGN_REVERSAL_LOCALIZED_TO_FINAL_RMSNORM"
elif p>0 and no>0:
    DECISION="TERMINAL_SIGN_REVERSAL_NOT_REPLICATED_ON_FRESH_XRAY"
else:
    DECISION="TAIL_REVERSAL_MECHANISM_NOT_RESOLVED"
print("       POST19 positive:",p>0)
print("       RAW27 negative:",r27<0)
print("       NORM_IN negative:",ni<0)
print("       first crossing:",FIRST[:2] if FIRST else None)
print("       DECISION:",DECISION)

# ================================================================================================================
# [22/24] SAVE
# ================================================================================================================
print("\n[22/24] Saving...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("TAIL_TRAJECTORY.csv",TRAJ);savecsv("BLOCK_EFFECTS.csv",BLOCK);savecsv("CONTINUITY.csv",CONT)
savecsv("BOOTSTRAP.csv",BOOT);savecsv("PROMPT_CROSSINGS.csv",PROMPT_CROSS);savecsv("CUMULATIVE.csv",CUM)
np.savez_compressed(ROOT/"TEST185_TAIL_XRAY.npz",orth19=U.cpu().numpy(),post19=D["post19"].cpu().numpy(),
 **{f"pre{L}":D[("pre",L)].cpu().numpy() for L in range(20,28)},
 **{f"raw{L}":D[("raw",L)].cpu().numpy() for L in range(20,28)},
 norm_in=D["norm_in"].cpu().numpy(),norm_out=D["norm_out"].cpu().numpy())
REPORT={"test":"TEST185","parent":"TEST184","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
 "post19":p,"raw27":r27,"norm_in":ni,"norm_out":no,"total_tail_delta":TM,
 "total_tail_ci":[TLO,THI],"rmsnorm_delta":NM,"rmsnorm_ci":[NLO,NHI],
 "first_crossing":list(FIRST[:2]) if FIRST else None,
 "dominant_negative_layer":DOM["layer"] if DOM else None,
 "integrity":{"xray_only":True,"fixed_coordinate":"ORTH19","steered_layers":"L0-L19",
 "tail_layers":"L20-L27 untouched","vector_search":False,"rank_search":False,"layer_search":False,
 "envelope_search":False,"controller":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [23/24] SCIENTIFIC BOUNDARY
# ================================================================================================================
print("\n[23/24] Scientific boundary...")
print("       SEASC is applied only to frozen L0-L19 exactly as before.")
print("       L20-L27 receive no injection and no controller.")
print("       Tail transport is measured in one fixed ORTH19 coordinate.")
print("       POST19→PRE20 and RAW[L]→PRE[L+1] continuity is checked directly.")
print("       TEST185 localizes tail transport; it does not select a new steering schedule.")

# ================================================================================================================
# [24/24] COMPLETE
# ================================================================================================================
print("\n[24/24] COMPLETE")
print("="*138)
print("TEST 185 — UNSTEERED TAIL TRANSPORT X-RAY — SUMMARY")
print("="*138)
print(f"POST19   @ORTH19={p:+.6f}")
for L in range(20,28):
    rr=next(x for x in BLOCK if x["layer"]==L)
    print(f"L{L}      {rr['pre']:+.6f} → {rr['raw']:+.6f}  Δ={rr['delta']:+.6f}")
print(f"RAW27    @ORTH19={r27:+.6f}")
print(f"NORM_IN  @ORTH19={ni:+.6f}")
print(f"NORM_OUT @ORTH19={no:+.6f}")
print(f"TOTAL TAIL Δ={TM:+.6f} [{TLO:+.6f},{THI:+.6f}]")
print("FIRST CROSSING:",f"{FIRST[0]}→{FIRST[1]}" if FIRST else "NONE")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
