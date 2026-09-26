# ================================================================================================================
# TEST 184 — FINAL RMSNORM GEOMETRY X-RAY
# AkbasCore 3.2 / SEASC
# TEST183 continuation — CAUSE↔EFFECT
#
# PURPOSE
# Explain TEST183 terminal transition:
#   POST19 @ ORTH19 > 0  →  FINAL @ ORTH19 < 0
#
# DECOMPOSE FINAL RMSNORM:
#   X      = exact POST19 / final_norm input
#   SCALE  = X / RMS(X)
#   WEIGHT = SCALE * learned RMSNorm weight
#   FINAL  = actual model.model.norm output
#
# MEASURE under fixed ORTH18 / ORTH19:
#   X → SCALE → WEIGHT → FINAL
#   PLUS/MINUS separately + symmetric delta
#   RMS scalar asymmetry
#   learned-weight geometry
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
print("TEST 184 — FINAL RMSNORM GEOMETRY X-RAY")
print("CAUSE↔EFFECT — POST19 → RMS SCALE → LEARNED WEIGHT → FINAL")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=184
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST184_FINAL_RMSNORM_XRAY");ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(N_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
print(f"SEASC RSS={RSS:.9f} | scale={PRIMARY_SCALE}")

# ================================================================================================================
# [1/24] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test184";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
seasc_ext=load_inline(name="akbascore_test184_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
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
NORM_EPS=float(getattr(FINAL_NORM,"variance_epsilon",getattr(FINAL_NORM,"eps",1e-6)))
NORM_WEIGHT=FINAL_NORM.weight.detach().float()
print("       hidden:",H,"dtype:",next(model.parameters()).dtype,"norm_eps:",NORM_EPS)

def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],
 tokenize=False,add_generation_prompt=True)

@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt").to(DEVICE)
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);p=int(e.attention_mask[0].sum())-1
    x=[o.hidden_states[L+1][0,p].float().detach() for L in range(N_LAYERS)]
    del e,o;return x

def sequence(ctx,cont):
    p=tok(chat(ctx),return_tensors="pt",add_special_tokens=False);c=tok(cont,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p.input_ids,c.input_ids],1).to(DEVICE);return ids,torch.ones_like(ids),p.input_ids.shape[1]

def actgrad(ctx,cont,positive=False):
    ids,mask,plen=sequence(ctx,cont);emb=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    saved=[None]*N_LAYERS;hooks=[]
    for L in range(N_LAYERS):
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
        g=[saved[L].grad[0].float().sum(0).detach().clone() for L in range(N_LAYERS)]
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
PB=[[] for _ in range(N_LAYERS)];NB=[[] for _ in range(N_LAYERS)]
for i,(p,n) in enumerate(PAIRS,1):
    hp,hn=capture(p),capture(n)
    for L in range(N_LAYERS):PB[L].append(hp[L]);NB[L].append(hn[L])
    print(f"       {i:02d}/16")
PB=[torch.stack(x).float() for x in PB];NB=[torch.stack(x).float() for x in NB]
FISHER=[];BANK=[];FB=[]
for L in range(N_LAYERS):
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
for L in range(N_LAYERS):
    G=norm(torch.stack([g[L] for g in PG]));_,s,vh=torch.linalg.svd(G,full_matrices=False)
    r=int((s>max(float(s[0])*SVD_REL_TOL,EPS)).sum());Q8.append(vh[:r].T.contiguous());ranks.append(r)
print("       ranks:",ranks)
def guard(x):
    z=[]
    for L in range(N_LAYERS):
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
    acc=[torch.zeros(H,device=DEVICE) for _ in range(N_LAYERS)]
    for j,(ctx,p,n) in enumerate(items,1):
        gp,gn=actgrad(ctx,p,True),actgrad(ctx,n,True)
        for L in range(N_LAYERS):acc[L]+=gp[L]-gn[L]
        print(f"       {fn} {j}/4")
    v=torch.stack([norm(x) for x in acc])
    for L in range(N_LAYERS):
        if torch.dot(v[L],FISHER[L])<0:v[L]=-v[L]
    FAM[fn]=v
CAUSAL=torch.stack([norm(torch.stack([FAM[f][L] for f in FAM]).mean(0)) for L in range(N_LAYERS)])
CG=guard(CAUSAL);ORTH=[]
for L in range(N_LAYERS):ORTH.append(norm(CG[L]-torch.dot(CG[L],FG[L])*FG[L]))
ORTH=torch.stack(ORTH).contiguous()
print("       ORTH ready")

# ================================================================================================================
# [7/24] TERMINAL COMPASS GEOMETRY
# ================================================================================================================
print("\n[7/24] Terminal compass geometry...")
C1819=float(torch.dot(ORTH[18],ORTH[19]).clamp(-1,1))
A1819=math.degrees(math.acos(max(-1,min(1,C1819))))
print(f"       ORTH18→ORTH19 cos={C1819:+.6f} angle={A1819:.3f}°")

# ================================================================================================================
# [8/24] FRESH XRAY
# ================================================================================================================
print("\n[8/24] Fresh X-ray...")
XRAY=[
"A fractured cable and a lost signal form a causal sequence.",
"A blocked radiator and rising engine temperature form a causal sequence.",
"A punctured line and falling fluid pressure form a causal sequence.",
"A voltage spike and a failed controller form a causal sequence.",
"A jammed valve and reduced flow form a causal sequence.",
"A worn pulley and increased vibration form a causal sequence.",
"A clogged drain and accumulated water form a causal sequence.",
"A hard impact and a bent frame form a causal sequence."]
print("       n=8")

# ================================================================================================================
# [9/24] EXACT RMSNORM RECORDER
# ================================================================================================================
print("\n[9/24] Exact POST19 / RMSNorm recorder...")
def run_exact(prompt,sign=0.0):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    BOX={};hooks=[]
    for L in range(N_LAYERS):
        def posthook(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out
                if sign==0:new=raw
                else:
                    B=raw.shape[0];a=(float(sign)*ORTH[li][None].expand(B,-1)).float().contiguous()
                    d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                    new=seasc_ext.seasc_batch(raw,a,d)
                if li==19:BOX["post19"]=new[0,pos].float().detach().clone()
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_hook(posthook(L)))
    def norm_pre(m,args):BOX["norm_in"]=args[0][0,pos].float().detach().clone()
    def norm_post(m,args,out):BOX["norm_out"]=out[0,pos].float().detach().clone()
    hooks.append(FINAL_NORM.register_forward_pre_hook(norm_pre))
    hooks.append(FINAL_NORM.register_forward_hook(norm_post))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    x=BOX["norm_in"]
    rms=torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+NORM_EPS)
    scaled=x*rms
    weighted=scaled*NORM_WEIGHT
    BOX["rms"]=rms.squeeze().detach().clone()
    BOX["scaled"]=scaled.detach().clone()
    BOX["weighted"]=weighted.detach().clone()
    return BOX

# ================================================================================================================
# [10/24] CAPTURE
# ================================================================================================================
print("\n[10/24] BASE / PLUS / MINUS...")
RUN={"BASE":[],"PLUS":[],"MINUS":[]}
for i,p in enumerate(XRAY,1):
    RUN["BASE"].append(run_exact(p,0));RUN["PLUS"].append(run_exact(p,+1));RUN["MINUS"].append(run_exact(p,-1))
    print(f"       {i}/8")

# ================================================================================================================
# [11/24] EXACTNESS CHECK
# ================================================================================================================
print("\n[11/24] Exactness checks...")
for a in RUN:
    e1=[];e2=[]
    for r in RUN[a]:
        e1.append(float((r["post19"]-r["norm_in"]).norm()))
        e2.append(float((r["weighted"]-r["norm_out"]).norm()))
    print(f"       {a:<5} POST19→NORM_IN err={np.mean(e1):.9e} | manual→actual err={np.mean(e2):.9e}")

# ================================================================================================================
# [12/24] STACK / SYMMETRIC DELTA
# ================================================================================================================
print("\n[12/24] Symmetric deltas...")
STAGES=["post19","norm_in","scaled","weighted","norm_out"]
S={s:{a:torch.stack([r[s] for r in RUN[a]]) for a in RUN} for s in STAGES}
D={s:.5*(S[s]["PLUS"]-S[s]["MINUS"]) for s in STAGES}
RMS={a:torch.stack([r["rms"] for r in RUN[a]]) for a in RUN}

# ================================================================================================================
# [13/24] FIXED ORTH18 / ORTH19 PROJECTIONS
# ================================================================================================================
print("\n[13/24] Fixed-coordinate decomposition...")
ROWS=[]
for s in STAGES:
    r={"stage":s}
    for name,u in [("ORTH18",ORTH[18]),("ORTH19",ORTH[19])]:
        d=D[s];tar=d@u;dn=d.norm(dim=-1).clamp_min(EPS);co=tar/dn
        r[name+"_target"]=float(tar.mean());r[name+"_cos"]=float(co.mean())
    ROWS.append(r)
    print(f"       {s:<8} | O18={r['ORTH18_target']:+.6f} cos={r['ORTH18_cos']:+.6f} | "
          f"O19={r['ORTH19_target']:+.6f} cos={r['ORTH19_cos']:+.6f}")

# ================================================================================================================
# [14/24] PLUS/MINUS RMS SCALAR
# ================================================================================================================
print("\n[14/24] RMS scalar asymmetry...")
rp=RMS["PLUS"];rm=RMS["MINUS"];rb=RMS["BASE"]
print(f"       BASE  rsqrt(RMS²+eps)={float(rb.mean()):.8f}")
print(f"       PLUS  rsqrt(RMS²+eps)={float(rp.mean()):.8f}")
print(f"       MINUS rsqrt(RMS²+eps)={float(rm.mean()):.8f}")
print(f"       PLUS-MINUS Δscale    ={float((rp-rm).mean()):+.10f}")

# ================================================================================================================
# [15/24] SCALAR-NORM CONTRIBUTION
# ================================================================================================================
print("\n[15/24] Scalar RMS normalization contribution...")
for name,u in [("ORTH18",ORTH[18]),("ORTH19",ORTH[19])]:
    a=D["norm_in"]@u;b=D["scaled"]@u
    print(f"       {name}: INPUT={float(a.mean()):+.6f} → SCALE={float(b.mean()):+.6f} Δ={float((b-a).mean()):+.6f}")

# ================================================================================================================
# [16/24] LEARNED-WEIGHT CONTRIBUTION
# ================================================================================================================
print("\n[16/24] Learned RMSNorm-weight contribution...")
for name,u in [("ORTH18",ORTH[18]),("ORTH19",ORTH[19])]:
    a=D["scaled"]@u;b=D["weighted"]@u
    print(f"       {name}: SCALE={float(a.mean()):+.6f} → WEIGHT={float(b.mean()):+.6f} Δ={float((b-a).mean()):+.6f}")

# ================================================================================================================
# [17/24] ACTUAL FINAL CHECK
# ================================================================================================================
print("\n[17/24] Manual weighted → actual final...")
for name,u in [("ORTH18",ORTH[18]),("ORTH19",ORTH[19])]:
    a=D["weighted"]@u;b=D["norm_out"]@u
    print(f"       {name}: MANUAL={float(a.mean()):+.6f} → ACTUAL={float(b.mean()):+.6f} Δ={float((b-a).mean()):+.9f}")

# ================================================================================================================
# [18/24] WEIGHT GEOMETRY
# ================================================================================================================
print("\n[18/24] RMSNorm learned-weight geometry...")
W=NORM_WEIGHT
print(f"       weight min={float(W.min()):.6f} mean={float(W.mean()):.6f} max={float(W.max()):.6f} std={float(W.std()):.6f}")
for name,u in [("ORTH18",ORTH[18]),("ORTH19",ORTH[19])]:
    wu=W*u;c=float(torch.dot(norm(wu),u).clamp(-1,1));ang=math.degrees(math.acos(max(-1,min(1,c))))
    print(f"       {name}: cos(W⊙u,u)={c:+.6f} angle={ang:.3f}°")

# ================================================================================================================
# [19/24] PER-PROMPT SIGN TRACE
# ================================================================================================================
print("\n[19/24] Per-prompt ORTH19 sign trace...")
TRACE=[]
for i in range(len(XRAY)):
    vals={s:float(torch.dot(D[s][i],ORTH[19])) for s in STAGES}
    TRACE.append({"prompt":i+1,**vals})
    print(f"       {i+1}: IN={vals['norm_in']:+.4f} SCALE={vals['scaled']:+.4f} "
          f"WEIGHT={vals['weighted']:+.4f} FINAL={vals['norm_out']:+.4f}")

# ================================================================================================================
# [20/24] BOOTSTRAP TRANSITIONS
# ================================================================================================================
print("\n[20/24] Bootstrap ORTH19 transitions...")
def boot(x,n=20000,seed=184):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT=[]
pairs=[("INPUT→SCALE","norm_in","scaled"),("SCALE→WEIGHT","scaled","weighted"),("WEIGHT→FINAL","weighted","norm_out")]
for j,(name,a,b) in enumerate(pairs):
    x=((D[b]-D[a])@ORTH[19]).cpu().numpy();m,lo,hi=boot(x,seed=SEED+j)
    BOOT.append({"transition":name,"mean":m,"low":lo,"high":hi})
    print(f"       {name:<14} Δ={m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [21/24] DECOMPOSITION / DECISION
# ================================================================================================================
print("\n[21/24] TEST184 decision...")
TIN=float((D["norm_in"]@ORTH[19]).mean())
TSC=float((D["scaled"]@ORTH[19]).mean())
TWT=float((D["weighted"]@ORTH[19]).mean())
TOUT=float((D["norm_out"]@ORTH[19]).mean())
DS=TSC-TIN;DW=TWT-TSC;DFN=TOUT-TWT
print(f"       INPUT  ={TIN:+.6f}")
print(f"       SCALE  ={TSC:+.6f} Δ={DS:+.6f}")
print(f"       WEIGHT ={TWT:+.6f} Δ={DW:+.6f}")
print(f"       FINAL  ={TOUT:+.6f} Δ={DFN:+.9f}")

if TIN>0 and TSC<0:
    DECISION="FINAL_SIGN_REVERSAL_LOCALIZED_TO_RMS_SCALAR_NORMALIZATION"
elif TSC>0 and TWT<0:
    DECISION="FINAL_SIGN_REVERSAL_LOCALIZED_TO_LEARNED_RMSNORM_WEIGHT"
elif TIN>0 and TOUT<0 and TSC>=0 and TWT>=0:
    DECISION="FINAL_SIGN_REVERSAL_REQUIRES_COMBINED_RMSNORM_NONLINEARITY"
elif TIN>0 and TOUT>=0:
    DECISION="FINAL_SIGN_REVERSAL_NOT_REPLICATED_ON_FRESH_XRAY"
else:
    DECISION="FINAL_RMSNORM_MECHANISM_NOT_RESOLVED"
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
savecsv("RMSNORM_STAGES.csv",ROWS);savecsv("PROMPT_TRACE.csv",TRACE);savecsv("BOOTSTRAP.csv",BOOT)
np.savez_compressed(ROOT/"TEST184_RMSNORM_XRAY.npz",orth=ORTH.cpu().numpy(),weight=W.cpu().numpy(),
 post19=D["post19"].cpu().numpy(),norm_in=D["norm_in"].cpu().numpy(),scaled=D["scaled"].cpu().numpy(),
 weighted=D["weighted"].cpu().numpy(),norm_out=D["norm_out"].cpu().numpy(),
 rms_base=rb.cpu().numpy(),rms_plus=rp.cpu().numpy(),rms_minus=rm.cpu().numpy())
REPORT={"test":"TEST184","parent":"TEST183","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
 "orth18_19_cos":C1819,"orth18_19_angle":A1819,
 "orth19":{"input":TIN,"scaled":TSC,"weighted":TWT,"final":TOUT,
 "scalar_effect":DS,"weight_effect":DW,"manual_actual_effect":DFN},
 "integrity":{"xray_only":True,"fresh_prompts":True,"same_frozen_forge":True,"same_p8":True,
 "same_seasc":True,"vector_search":False,"rank_search":False,"layer_search":False,
 "envelope_search":False,"controller":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [23/24] SCIENTIFIC BOUNDARY
# ================================================================================================================
print("\n[23/24] Scientific boundary...")
print("       POST19 and final RMSNorm input are captured directly.")
print("       RMS scalar and learned element-wise weight are reconstructed separately in FP32.")
print("       PLUS/MINUS symmetric displacement is recomputed at every stage; this is not a linearized approximation.")
print("       No vector, rank, layer, envelope or controller is selected.")
print("       TEST184 localizes terminal normalization geometry; it does not establish behavioral causality.")

# ================================================================================================================
# [24/24] COMPLETE
# ================================================================================================================
print("\n[24/24] COMPLETE")
print("="*138)
print("TEST 184 — FINAL RMSNORM GEOMETRY X-RAY — SUMMARY")
print("="*138)
print(f"ORTH18→ORTH19 cos={C1819:+.6f} angle={A1819:.3f}°")
print(f"ORTH19 INPUT ={TIN:+.6f}")
print(f"       SCALE ={TSC:+.6f}  Δ={DS:+.6f}")
print(f"       WEIGHT={TWT:+.6f}  Δ={DW:+.6f}")
print(f"       FINAL ={TOUT:+.6f}  Δmanual={DFN:+.9f}")
print(f"RMS scale BASE={float(rb.mean()):.8f} PLUS={float(rp.mean()):.8f} MINUS={float(rm.mean()):.8f}")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
