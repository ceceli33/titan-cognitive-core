# ================================================================================================================
# TEST 188 — BEHAVIOR-SENSITIVE L27 TRANSPORT VECTOR FORGE — FIXED (LEAF-REPLACEMENT PRE-HOOK)
# AkbasCore 3.2 / SEASC — TEST187 continuation — CAUSE↔EFFECT
#
# TEST187: PRE27@ORTH19=+0.715652 | RAW27@ORTH19=-0.297196 | L27 Δ@ORTH19=-1.012847
#          PRE→RAW cos=+0.844819 | Δ27 orth frac=0.999771
#
# QUESTION: Does the high-dimensional L27 transport field contain a direction aligned with an
#           independent CAUSE↔EFFECT behavioral gradient?
#
# GRADIENT FIX:
#   Phase A (no_grad): normal SEASC forward, capture the full-sequence L27 input (PRE27) + reference score.
#   Phase B (grad):    identical SEASC forward; a layers[27] forward_pre_hook(with_kwargs=True) swaps the
#                      real L27 hidden_states input for an independent leaf (detach().clone().requires_grad_()).
#                      Qwen2Model itself supplies attention mask / position_ids / rotary position_embeddings,
#                      so L27 → final RMSNorm → lm_head is the untouched Transformers forward graph.
#   Integrity:         leaf identity at L27, replacement relerr, upstream recompute relerr,
#                      reference vs leaf score / logits error, finite + non-zero gradient. Any failure → RuntimeError.
#
# NO injection of new vectors | NO FINAL behavior | NO K winner | NO controller/rescue | NO weight change
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 188 — BEHAVIOR-SENSITIVE L27 TRANSPORT VECTOR FORGE — FIXED")
print("CAUSE↔EFFECT — Δ27 TRANSPORT SUBSPACE × LOCAL L27 BEHAVIOR GRADIENT (LEAF-REPLACEMENT PRE-HOOK)")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=188
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50
KS=[1,2,4,7]
# integrity tolerances
REPL_TOL=1e-6      # leaf vs captured PRE27 (must be exact copy)
UPSTREAM_TOL=1e-3  # recomputed L27 input (Phase B) vs captured PRE27 (Phase A)
SCORE_TOL=1e-2     # |reference score - leaf score|
LOGIT_TOL=1e-2     # ||reference logits - leaf logits|| / ||reference logits||
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST188_BEHAVIOR_SENSITIVE_L27_FORGE_FIXED");ROOT.mkdir(parents=True,exist_ok=True)

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
# [1/30] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test188_fixed";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
print("\n[1/30] CUDA...")
seasc_ext=load_inline(name="akbascore_test188_fixed_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD,verbose=False)
print("       OK")

# ================================================================================================================
# [2/30] MODEL
# ================================================================================================================
print("\n[2/30] Model...")
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
# [3/30] FROZEN E AXIS
# ================================================================================================================
print("\n[3/30] Frozen E axis...")
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
# [4/30] FISHER / BANK
# ================================================================================================================
print("\n[4/30] Fisher/Bank...")
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
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS);V0=vh.T;pr=V0.T@md
    f=norm(md/lam+V0@((1/(eig+lam)-1/lam)*pr)).float();dirs=norm(hp-hn)
    dirs*=torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
    b=norm(dirs.mean(0)).float();c=torch.dot(f,b)
    if c<0:b=-b;c=-c
    FISHER.append(f);BANK.append(b);FB.append(float(c))
FISHER=torch.stack(FISHER);BANK=torch.stack(BANK)
print(f"       mean cos(F,B)={np.mean(FB):+.6f}")

# ================================================================================================================
# [5/30] P8
# ================================================================================================================
print("\n[5/30] P8...")
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
    G0=norm(torch.stack([g[L] for g in PG]));_,s,vh=torch.linalg.svd(G0,full_matrices=False)
    r=int((s>max(float(s[0])*SVD_REL_TOL,EPS)).sum());Q8.append(vh[:r].T.contiguous());ranks.append(r)
print("       ranks:",ranks)
def guard(x):
    z=[]
    for L in range(STEER_LAYERS):
        q=Q8[L];v=x[L]-(q@(q.T@x[L]) if q.shape[1] else 0);z.append(norm(v))
    return torch.stack(z).contiguous()
FG=guard(FISHER)

# ================================================================================================================
# [6/30] FROZEN FORGE
# ================================================================================================================
print("\n[6/30] Frozen forge...")
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
# [7/30] FROZEN BUILD / GRAD / HOLDOUT SETS
# ================================================================================================================
print("\n[7/30] Frozen BUILD / GRAD / HOLDOUT sets...")
BUILD_X=[
"A fractured pipe and a pressure drop form a causal sequence.",
"A blocked air passage and rising temperature form a causal sequence.",
"A damaged seal and fluid loss form a causal sequence.",
"A current surge and a failed circuit form a causal sequence.",
"A stuck mechanism and restricted motion form a causal sequence.",
"A worn bearing and increased vibration form a causal sequence.",
"A clogged outlet and accumulated liquid form a causal sequence.",
"A strong impact and a deformed bracket form a causal sequence."]
GRAD_X=[
("A cracked hose produces a loss of pressure."," The cracked hose is the cause."," The pressure loss is the effect."),
("A failed fan produces overheating."," The failed fan is the cause."," The overheating is the effect."),
("A puncture produces loss of air pressure."," The puncture is the cause."," The pressure loss is the effect."),
("A short circuit produces a shutdown."," The short circuit is the cause."," The shutdown is the effect."),
("A blocked drain produces water accumulation."," The blocked drain is the cause."," The water accumulation is the effect."),
("A damaged bearing produces excessive vibration."," The bearing damage is the cause."," The vibration is the effect."),
("A clogged filter produces reduced flow."," The clogged filter is the cause."," The reduced flow is the effect."),
("A loose connection produces an intermittent signal."," The loose connection is the cause."," The intermittent signal is the effect.")]
HOLD_X=[
"A broken coupling and loss of transmitted motion form a causal sequence.",
"A blocked intake and reduced airflow form a causal sequence.",
"A damaged gasket and pressure leakage form a causal sequence.",
"A failed coolant pump and rising temperature form a causal sequence.",
"A seized bearing and restricted rotation form a causal sequence.",
"A severed cable and loss of electrical power form a causal sequence.",
"A clogged nozzle and reduced discharge form a causal sequence.",
"A fractured support and structural displacement form a causal sequence."]
print("       BUILD=8 GRAD=8 HOLDOUT=8 | K=",KS)

# ================================================================================================================
# [8/30] EXACT L27 TRANSPORT RECORDER (TEST187)
# ================================================================================================================
print("\n[8/30] Exact L27 transport recorder...")
def run_transport(prompt,sign):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    box={};hooks=[]
    for L in range(STEER_LAYERS):
        def mk(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out
                B=raw.shape[0];a=(float(sign)*ORTH[li][None].expand(B,-1)).float().contiguous()
                d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                new=seasc_ext.seasc_batch(raw,a,d)
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    def pre27(m,args):box["pre27"]=args[0][0,pos].float().detach().clone()
    def raw27(m,args,out):
        h=out[0] if isinstance(out,tuple) else out
        box["raw27"]=h[0,pos].float().detach().clone()
    hooks.append(layers[27].register_forward_pre_hook(pre27))
    hooks.append(layers[27].register_forward_hook(raw27))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    return box

# ================================================================================================================
# [9/30] BUILD TRANSPORT
# ================================================================================================================
print("\n[9/30] BUILD transport...")
BD=[]
for i,p in enumerate(BUILD_X,1):
    P=run_transport(p,+1);M=run_transport(p,-1)
    dp=.5*(P["pre27"]-M["pre27"]);dr=.5*(P["raw27"]-M["raw27"]);BD.append(dr-dp)
    print(f"       {i}/8")
BD=torch.stack(BD)

# ================================================================================================================
# [10/30] FROZEN Δ27 SVD
# ================================================================================================================
print("\n[10/30] Frozen Δ27 SVD...")
MU=BD.mean(0);XC=BD-MU
_,S,Vh=torch.linalg.svd(XC,full_matrices=False)
VAR=S.square();VR=VAR/VAR.sum().clamp_min(EPS);CUM=torch.cumsum(VR,0)
for i in range(7):print(f"       PC{i+1}: {float(VR[i])*100:.3f}% cumulative={float(CUM[i])*100:.3f}%")
V=Vh[:7].T.contiguous()
V_FROZEN=V.detach().clone()
print("       transport basis frozen before behavior gradients")

# ================================================================================================================
# [11/30] REFERENCE + LEAF-REPLACEMENT GRADIENT ENGINE
# ================================================================================================================
print("\n[11/30] Reference + leaf-replacement L27 gradient engine...")
def seasc_hooks(sign):
    hooks=[]
    for L in range(STEER_LAYERS):
        def mk(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out
                B=raw.shape[0];a=(float(sign)*ORTH[li][None].expand(B,-1)).float().contiguous()
                d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                new=seasc_ext.seasc_batch(raw,a,d)
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    return hooks

def _hs(args,kwargs):
    if len(args)>0:return args[0]
    if "hidden_states" in kwargs:return kwargs["hidden_states"]
    raise RuntimeError("L27 hidden_states input not found in args/kwargs.")

def cont_score(logits,ids,plen):
    z=logits[:,plen-1:-1,:].float();tar=ids[:,plen:]
    if z.shape[1]!=tar.shape[1] or tar.shape[1]<1:
        raise RuntimeError(f"Continuation alignment mismatch: logits={tuple(z.shape)} targets={tuple(tar.shape)}")
    return F.log_softmax(z,-1).gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean(),z

# Phase A: unmodified SEASC forward (no_grad, NOT inference_mode → captured tensor is autograd-compatible).
@torch.no_grad()
def reference_forward(ids,mask,plen,sign):
    box={};hooks=seasc_hooks(sign)
    def cap(m,args,kwargs):box["h"]=_hs(args,kwargs).detach().clone()
    hooks.append(layers[27].register_forward_pre_hook(cap,with_kwargs=True))
    try:o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    if "h" not in box:raise RuntimeError("PRE27 state was not captured.")
    sc,z=cont_score(o.logits,ids,plen)
    return box["h"],sc.detach(),z.detach()

# Phase B: identical SEASC forward; L27 hidden_states input swapped for the leaf. Mask / position_ids /
# rotary position_embeddings / cache kwargs are passed through untouched from Qwen2Model.forward.
def leaf_forward(ids,mask,plen,sign,leaf):
    box={"replaced":False,"identity":False,"out_rg":False};hooks=seasc_hooks(sign)
    def rep(m,args,kwargs):
        inc=_hs(args,kwargs)
        if inc.shape!=leaf.shape or inc.dtype!=leaf.dtype:
            raise RuntimeError(f"L27 input/leaf mismatch: {tuple(inc.shape)} {inc.dtype} vs {tuple(leaf.shape)} {leaf.dtype}")
        box["incoming"]=inc.detach().clone()
        if len(args)>0:args=(leaf,)+tuple(args[1:])
        else:kwargs=dict(kwargs);kwargs["hidden_states"]=leaf
        box["replaced"]=True
        return args,kwargs
    def ver(m,args,kwargs):box["identity"]=(_hs(args,kwargs) is leaf)
    def post(m,args,out):
        h=out[0] if isinstance(out,tuple) else out;box["out_rg"]=bool(h.requires_grad)
    hooks.append(layers[27].register_forward_pre_hook(rep,with_kwargs=True))
    hooks.append(layers[27].register_forward_pre_hook(ver,with_kwargs=True))
    hooks.append(layers[27].register_forward_hook(post))
    try:
        with torch.enable_grad():
            o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
            sc,z=cont_score(o.logits,ids,plen)
    finally:
        for h in hooks:h.remove()
    return sc,z,box

def local_l27_gradient(ctx,cont,sign):
    ids,mask,plen=sequence(ctx,cont)
    h0,ref_sc,ref_z=reference_forward(ids,mask,plen,sign)
    captured=h0 is not None and h0.dim()==3 and h0.shape[1]==ids.shape[1] and h0.shape[2]==H
    if not captured:raise RuntimeError(f"PRE27 capture invalid: {None if h0 is None else tuple(h0.shape)}")
    leaf=h0.detach().clone().requires_grad_(True)
    if not(leaf.is_leaf and leaf.requires_grad):raise RuntimeError("PRE27 leaf construction failed.")
    model.zero_grad(set_to_none=True)
    try:
        sc,z,box=leaf_forward(ids,mask,plen,sign,leaf)
        if not box["replaced"]:raise RuntimeError("Leaf replacement hook did not fire at L27.")
        if not box["identity"]:raise RuntimeError("L27 did not receive the leaf tensor.")
        if not box["out_rg"]:raise RuntimeError("L27 output does not depend on the leaf (requires_grad=False).")
        if not sc.requires_grad:raise RuntimeError("Continuation score is not connected to the leaf.")
        hc=h0.float();nh=hc.norm().clamp_min(EPS)
        repl=float((leaf.detach().float()-hc).norm()/nh)
        ups=float((box["incoming"].float()-hc).norm()/nh)
        serr=float((sc.detach()-ref_sc).abs())
        lerr=float((z.detach()-ref_z).norm()/ref_z.norm().clamp_min(EPS))
        if repl>REPL_TOL:raise RuntimeError(f"PRE27 replacement relerr too large: {repl:.3e}")
        if ups>UPSTREAM_TOL:raise RuntimeError(f"Upstream recompute vs captured PRE27 relerr too large: {ups:.3e}")
        if serr>SCORE_TOL:raise RuntimeError(f"Reference vs leaf score error too large: {serr:.3e}")
        if lerr>LOGIT_TOL:raise RuntimeError(f"Reference vs leaf logits relerr too large: {lerr:.3e}")
        gfull=torch.autograd.grad(sc,leaf,retain_graph=False,create_graph=False,allow_unused=False)[0]
        if gfull is None:raise RuntimeError("Leaf gradient is None.")
        g=gfull[0].float().sum(dim=0).detach().clone()
        fin=bool(torch.isfinite(gfull).all()) and bool(torch.isfinite(g).all())
        gn=float(g.norm())
        if not fin:raise RuntimeError("Non-finite L27 local gradient.")
        if gn<=EPS:raise RuntimeError("Zero L27 local gradient.")
    finally:
        model.zero_grad(set_to_none=True)
    diag={"captured":captured,"leaf_requires_grad":bool(leaf.requires_grad),"leaf_is_leaf":bool(leaf.is_leaf),
          "leaf_identity":bool(box["identity"]),"l27_out_requires_grad":bool(box["out_rg"]),
          "replacement_relerr":repl,"upstream_relerr":ups,"score_ref":float(ref_sc),"score_leaf":float(sc.detach()),
          "score_err":serr,"logits_relerr":lerr,"grad_finite":fin,"grad_norm":gn,"seq_len":int(ids.shape[1]),"plen":int(plen)}
    del gfull,leaf,h0,ref_z,z
    return g,diag
print("       ready | L27 input replaced via layers[27].register_forward_pre_hook(with_kwargs=True)")

# ================================================================================================================
# [12/30] GRADIENT INTEGRITY PREFLIGHT
# ================================================================================================================
print("\n[12/30] Gradient integrity preflight (GRAD item 1, CAUSE, PLUS)...")
_,PF=local_l27_gradient(GRAD_X[0][0],GRAD_X[0][1],+1)
print(f"       PRE27 captured: {PF['captured']} | seq_len={PF['seq_len']} plen={PF['plen']}")
print(f"       PRE27 leaf is_leaf: {PF['leaf_is_leaf']}")
print(f"       PRE27 leaf requires_grad: {PF['leaf_requires_grad']}")
print(f"       L27 received leaf (identity): {PF['leaf_identity']}")
print(f"       L27 output requires_grad: {PF['l27_out_requires_grad']}")
print(f"       PRE27 replacement relerr = {PF['replacement_relerr']:.3e}")
print(f"       upstream recompute relerr = {PF['upstream_relerr']:.3e}")
print(f"       reference score = {PF['score_ref']:+.8f} | leaf score = {PF['score_leaf']:+.8f}")
print(f"       reference vs leaf score error = {PF['score_err']:.3e}")
print(f"       reference vs leaf logits relerr = {PF['logits_relerr']:.3e}")
print(f"       gradient finite: {PF['grad_finite']}")
print(f"       gradient norm > 0: {PF['grad_norm']>EPS}")
print(f"       gradient norm = {PF['grad_norm']:.6e}")
print("       PREFLIGHT PASSED")

# ================================================================================================================
# [13/30] BEHAVIOR GRADIENTS
# ================================================================================================================
print("\n[13/30] Behavior gradients (J = logP(CAUSE) - logP(EFFECT), PLUS/MINUS symmetric)...")
GROWS=[];DIAG=[]
for i,(ctx,pos,neg) in enumerate(GRAD_X,1):
    gpP,d1=local_l27_gradient(ctx,pos,+1);gnP,d2=local_l27_gradient(ctx,neg,+1)
    gpM,d3=local_l27_gradient(ctx,pos,-1);gnM,d4=local_l27_gradient(ctx,neg,-1)
    for tag,d in zip(["CAUSE+","EFFECT+","CAUSE-","EFFECT-"],[d1,d2,d3,d4]):DIAG.append({"item":i,"cond":tag,**d})
    g=.5*((gpP-gnP)+(gpM-gnM))
    if not torch.isfinite(g).all() or float(g.norm())<=EPS:raise RuntimeError(f"Degenerate behavior gradient at item {i}.")
    GROWS.append(norm(g))
    ds=[d1,d2,d3,d4]
    print(f"       {i}/8 repl={max(d['replacement_relerr'] for d in ds):.1e} "
          f"ups={max(d['upstream_relerr'] for d in ds):.1e} "
          f"serr={max(d['score_err'] for d in ds):.1e} "
          f"lerr={max(d['logits_relerr'] for d in ds):.1e} "
          f"|g|min={min(d['grad_norm'] for d in ds):.3e} "
          f"J+={d1['score_ref']-d2['score_ref']:+.4f} J-={d3['score_ref']-d4['score_ref']:+.4f}")
GROWS=torch.stack(GROWS)
MAXD={k:max(d[k] for d in DIAG) for k in ["replacement_relerr","upstream_relerr","score_err","logits_relerr"]}
print(f"       max replacement relerr={MAXD['replacement_relerr']:.3e} | max upstream relerr={MAXD['upstream_relerr']:.3e}")
print(f"       max score err={MAXD['score_err']:.3e} | max logits relerr={MAXD['logits_relerr']:.3e}")
if not torch.equal(V,V_FROZEN):raise RuntimeError("Transport basis changed after freezing.")
print("       transport basis unchanged: True")

# ================================================================================================================
# [14/30] CONSENSUS BEHAVIOR GRADIENT
# ================================================================================================================
print("\n[14/30] Consensus behavior gradient...")
G=norm(GROWS.mean(0));signs=GROWS@G
GROWS=torch.where(signs[:,None]<0,-GROWS,GROWS);G=norm(GROWS.mean(0))
family_cos=GROWS@G
print(f"       sign flips={int((signs<0).sum())}/8")
print(f"       mean family cos={float(family_cos.mean()):+.6f}")
print(f"       min family cos ={float(family_cos.min()):+.6f}")
print(f"       cos(G,ORTH19)  ={float(torch.dot(G,U)):+.6f}")

# ================================================================================================================
# [15/30] BEHAVIOR GRADIENT × TRANSPORT PCs
# ================================================================================================================
print("\n[15/30] Behavior gradient × transport PCs...")
PCDOT=V.T@G
for i,x in enumerate(PCDOT,1):print(f"       PC{i}: cos={float(x):+.6f}")
FULL=float(PCDOT.square().sum().sqrt())
print(f"       full centered transport capture={FULL:.6f}")
print(f"       random-direction reference ≈ sqrt(7/{H}) = {math.sqrt(7/H):.6f}")

# ================================================================================================================
# [16/30] K-SUBSPACE CANDIDATES
# ================================================================================================================
print("\n[16/30] K-subspace behavior capture...")
KROWS=[];CANDS={}
for k in KS:
    q=V[:,:k];p=q@(q.T@G);cap=float(p.norm())
    cand=norm(p) if cap>EPS else torch.zeros_like(G)
    if cap>EPS and torch.dot(cand,G)<0:cand=-cand
    CANDS[k]=cand
    row={"K":k,"capture":cap,"cos_G":float(torch.dot(cand,G)) if cap>EPS else 0.,
         "cos_ORTH19":float(torch.dot(cand,U)) if cap>EPS else 0.}
    KROWS.append(row)
    print(f"       K={k}: capture={cap:.6f} cos(candidate,G)={row['cos_G']:+.6f} cos(candidate,ORTH19)={row['cos_ORTH19']:+.6f}")

# ================================================================================================================
# [17/30] PER-GRADIENT CAPTURE
# ================================================================================================================
print("\n[17/30] Per-gradient transport capture...")
for k in KS:
    caps=(GROWS@V[:,:k]).norm(dim=-1)
    print(f"       K={k}: mean={float(caps.mean()):.6f} min={float(caps.min()):.6f} max={float(caps.max()):.6f}")

# ================================================================================================================
# [18/30] CANDIDATE SIGN STABILITY
# ================================================================================================================
print("\n[18/30] Candidate sign stability...")
for k in KS:
    c=CANDS[k];dots=GROWS@c if float(c.norm())>EPS else torch.zeros(len(GROWS),device=DEVICE)
    print(f"       K={k}: P+={int((dots>0).sum())}/8 mean={float(dots.mean()):+.6f} min={float(dots.min()):+.6f}")

# ================================================================================================================
# [19/30] MEAN TRANSPORT RELATION
# ================================================================================================================
print("\n[19/30] Candidate relation to mean Δ27...")
MD=norm(MU)
print(f"       cos(meanΔ,G)={float(torch.dot(MD,G)):+.6f}")
for k in KS:print(f"       K={k}: cos(candidate,meanΔ)={float(torch.dot(CANDS[k],MD)):+.6f}")

# ================================================================================================================
# [20/30] HOLDOUT TRANSPORT
# ================================================================================================================
print("\n[20/30] HOLDOUT transport...")
HD=[]
for i,p in enumerate(HOLD_X,1):
    P=run_transport(p,+1);M=run_transport(p,-1)
    dp=.5*(P["pre27"]-M["pre27"]);dr=.5*(P["raw27"]-M["raw27"]);HD.append(dr-dp)
    print(f"       {i}/8")
HD=torch.stack(HD)

# ================================================================================================================
# [21/30] HOLDOUT SUBSPACE COVERAGE
# ================================================================================================================
print("\n[21/30] HOLDOUT transport coverage...")
HCOVER={}
for k in KS:
    frac=(HD@V[:,:k]).norm(dim=-1)/HD.norm(dim=-1).clamp_min(EPS);HCOVER[k]=frac
    print(f"       K={k}: mean norm coverage={float(frac.mean()):.6f} min={float(frac.min()):.6f}")
HC_C=HD-MU
for k in KS:
    fc=(HC_C@V[:,:k]).norm(dim=-1)/HC_C.norm(dim=-1).clamp_min(EPS)
    print(f"       K={k}: centered (BUILD mean) coverage={float(fc.mean()):.6f}")

# ================================================================================================================
# [22/30] HOLDOUT CANDIDATE ALIGNMENT
# ================================================================================================================
print("\n[22/30] HOLDOUT Δ27 alignment...")
for k in KS:
    c=CANDS[k]
    d=cosv(HD,c[None].expand_as(HD)) if float(c.norm())>EPS else torch.zeros(HD.shape[0],device=DEVICE)
    print(f"       K={k}: mean cos={float(d.mean()):+.6f} P+={int((d>0).sum())}/8")

# ================================================================================================================
# [23/30] ORTH19 COMPARISON
# ================================================================================================================
print("\n[23/30] ORTH19 comparison...")
dg=cosv(GROWS,U[None].expand_as(GROWS));dh=cosv(HD,U[None].expand_as(HD))
print(f"       behavior-grad cos ORTH19 mean={float(dg.mean()):+.6f}")
print(f"       HOLD Δ27 cos ORTH19 mean      ={float(dh.mean()):+.6f}")
for k in KS:print(f"       K={k} candidate cos ORTH19={float(torch.dot(CANDS[k],U)):+.6f}")

# ================================================================================================================
# [24/30] LOFO STABILITY
# ================================================================================================================
print("\n[24/30] Gradient LOFO stability...")
LOFO=[]
for leave in range(8):
    gg=norm(torch.cat([GROWS[:leave],GROWS[leave+1:]],dim=0).mean(0));row={"leave":leave+1}
    for k in KS:
        q=V[:,:k];p=q@(q.T@gg)
        if float(p.norm())>EPS:
            c=norm(p)
            if torch.dot(c,CANDS[k])<0:c=-c
            row[f"K{k}"]=float(torch.dot(c,CANDS[k]))
        else:row[f"K{k}"]=0.
    LOFO.append(row)
for k in KS:
    vals=[r[f"K{k}"] for r in LOFO]
    print(f"       K={k}: mean cos={np.mean(vals):+.6f} min={np.min(vals):+.6f}")

# ================================================================================================================
# [25/30] BOOTSTRAP
# ================================================================================================================
print("\n[25/30] Bootstrap candidate-gradient alignment...")
def boot(x,n=20000,seed=188):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT=[]
for j,k in enumerate(KS):
    x=(GROWS@CANDS[k]).detach().cpu().numpy();m,lo,hi=boot(x,seed=SEED+j)
    BOOT.append({"K":k,"mean":m,"low":lo,"high":hi})
    print(f"       K={k}: {m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [26/30] DESCRIPTIVE K SUMMARY
# ================================================================================================================
print("\n[26/30] Descriptive K summary...")
for r in KROWS:
    k=r["K"];lofo=np.min([x[f"K{k}"] for x in LOFO])
    print(f"       K={k}: Gcapture={r['capture']:.6f} LOFOmin={lofo:+.6f} HOLDcoverage={float(HCOVER[k].mean()):.6f}")
print("       No K selected in TEST188.")

# ================================================================================================================
# [27/30] NOVEL-DIRECTION GEOMETRY
# ================================================================================================================
print("\n[27/30] Novel-direction geometry...")
for k in KS:
    c=CANDS[k]
    print(f"       K={k}: cos(G)={float(torch.dot(c,G)):+.6f} cos(ORTH19)={float(torch.dot(c,U)):+.6f} "
          f"cos(FISHER19)={float(torch.dot(c,FISHER[19])):+.6f} cos(BANK19)={float(torch.dot(c,BANK[19])):+.6f}")

# ================================================================================================================
# [28/30] DECISION
# ================================================================================================================
print("\n[28/30] TEST188 decision...")
LOFO7=min(r["K7"] for r in LOFO);B7=next(x for x in BOOT if x["K"]==7)
if FULL>=.20 and LOFO7>=.80 and B7["low"]>0:
    DECISION="BEHAVIOR_SENSITIVE_DIRECTION_EXISTS_WITHIN_L27_TRANSPORT_SUBSPACE"
elif FULL>=.10 and B7["low"]>0:
    DECISION="WEAK_BEHAVIOR_SENSITIVE_L27_TRANSPORT_SIGNAL_DETECTED"
else:
    DECISION="BEHAVIOR_SENSITIVE_L27_TRANSPORT_DIRECTION_NOT_CONFIRMED"
print("       full transport capture:",f"{FULL:.6f}")
print("       K7 LOFO min:",f"{LOFO7:+.6f}")
print("       K7 bootstrap low:",f"{B7['low']:+.6f}")
print("       DECISION:",DECISION)
print("       (geometric/mechanistic result only — not a behavioral success claim)")

# ================================================================================================================
# [29/30] SAVE + INTEGRITY
# ================================================================================================================
print("\n[29/30] Saving / integrity...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("K_SUMMARY.csv",KROWS);savecsv("LOFO.csv",LOFO);savecsv("BOOTSTRAP.csv",BOOT);savecsv("GRAD_INTEGRITY.csv",DIAG)
np.savez_compressed(ROOT/"TEST188_TRANSPORT_FORGE.npz",
 orth19=U.detach().cpu().numpy(),behavior_gradient=G.detach().cpu().numpy(),grad_rows=GROWS.detach().cpu().numpy(),
 build_delta27=BD.detach().cpu().numpy(),hold_delta27=HD.detach().cpu().numpy(),
 transport_basis=V.detach().cpu().numpy(),transport_mean=MU.detach().cpu().numpy(),
 singular_values=S.detach().cpu().numpy(),variance_ratio=VR.detach().cpu().numpy(),
 **{f"candidate_K{k}":CANDS[k].detach().cpu().numpy() for k in KS})
REPORT={"test":"TEST188","parent":"TEST187","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
 "mean_fb":float(np.mean(FB)),"transport_pc_ratio":[float(x) for x in VR[:7]],
 "behavior_gradient_orth19":float(torch.dot(G,U)),"full_transport_capture":FULL,
 "K_summary":KROWS,"LOFO":LOFO,"bootstrap":BOOT,
 "gradient_integrity":{"preflight":PF,"max":MAXD,
  "tolerances":{"replacement":REPL_TOL,"upstream":UPSTREAM_TOL,"score":SCORE_TOL,"logits":LOGIT_TOL}},
 "integrity":{"test187_architecture_preserved":True,"seasc_l0_l19_frozen":True,"l20_l27_unmodified":True,
  "transport_build_separate_from_gradient_set":True,"holdout_transport_separate":True,
  "transport_basis_frozen_before_gradients":True,
  "local_gradient_source":"PRE27 leaf swapped into layers[27] via forward_pre_hook(with_kwargs) -> native Qwen2 L27 -> final norm -> lm_head",
  "native_mask_position_rotary_used":True,"new_vector_used_for_steering":False,"K_selected":False,
  "final_behavior_opened":False,"controller":False,"rescue":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)
print("       TEST187 SEASC L0-L19 preserved: True")
print("       L20-L27 weights untouched: True")
print("       Local gradient graph starts at PRE27 leaf inside native forward: True")
print("       New vector injected: False")
print("       K selected: False")
print("       FINAL behavioral test: False")
print("       Controller: False")
print("       Weight modification: False")

# ================================================================================================================
# [30/30] COMPLETE
# ================================================================================================================
print("\n[30/30] COMPLETE")
print("="*138)
print("TEST 188 — BEHAVIOR-SENSITIVE L27 TRANSPORT VECTOR FORGE — SUMMARY")
print("="*138)
print(f"mean cos(F,B)={np.mean(FB):+.6f}")
print(f"gradient integrity: max repl={MAXD['replacement_relerr']:.1e} ups={MAXD['upstream_relerr']:.1e} "
      f"score={MAXD['score_err']:.1e} logits={MAXD['logits_relerr']:.1e}")
print(f"full transport capture of behavior gradient={FULL:.6f} (random ≈ {math.sqrt(7/H):.6f})")
print(f"cos(G,ORTH19)={float(torch.dot(G,U)):+.6f}")
for r in KROWS:
    k=r["K"];lofo=min(x[f"K{k}"] for x in LOFO)
    print(f"K={k}: capture={r['capture']:.6f} cosG={r['cos_G']:+.6f} cosORTH19={r['cos_ORTH19']:+.6f} LOFOmin={lofo:+.6f}")
print("K selected: NONE")
print("New vector injected: False")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
