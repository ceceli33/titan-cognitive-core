
# ================================================================================================================
# TEST 189 — FULL-SPACE BEHAVIORAL TRANSPORT X-RAY
# AkbasCore 3.2 / SEASC — TEST187/188 continuation — CAUSE↔EFFECT
#
# TEST187:
#   L27 creates a predominantly ORTH19-orthogonal redistribution.
#
# TEST188:
#   Centered 7-PC transport basis captures only 0.064565 of independent behavior gradient G.
#   PCA transport-subspace hypothesis NOT confirmed.
#
# TEST189 QUESTION:
#   Does the FULL 3584D L27 displacement move along an independently-built behavior-sensitive direction?
#
# BUILD:
#   Build G_BEH from independent CAUSE↔EFFECT continuation gradients using TEST188's validated
#   PRE27 leaf-replacement native Qwen forward.
#
# XRAY:
#   d_pre = .5*(PRE27+ - PRE27-)
#   d_raw = .5*(RAW27+ - RAW27-)
#   Δ27   = d_raw - d_pre
#
#   B_pre = <d_pre,G_BEH>
#   B_raw = <d_raw,G_BEH>
#   ΔB27  = <Δ27,G_BEH> = B_raw-B_pre
#
# ALSO:
#   - cosine of PRE / RAW / Δ with G_BEH
#   - fraction of Δ norm carried along G_BEH
#   - behavior-sensitive component vs ORTH19 component
#   - per-prompt signs
#   - bootstrap CIs
#   - permutation null for mean ΔB27
#   - independent HOLDOUT replication
#
# NO PCA SELECTION | NO K | NO NEW VECTOR INJECTION | NO FINAL | NO CONTROLLER | NO WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 189 — FULL-SPACE BEHAVIORAL TRANSPORT X-RAY")
print("CAUSE↔EFFECT — PRE27 → RAW27 — FULL 3584D Δ27 × INDEPENDENT BEHAVIOR GRADIENT")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=189
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50
REPL_TOL=1e-6;UPSTREAM_TOL=1e-3;SCORE_TOL=1e-2;LOGIT_TOL=1e-2
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST189_FULLSPACE_BEHAVIOR_TRANSPORT");ROOT.mkdir(parents=True,exist_ok=True)

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
# [1/28] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test189";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
print("\n[1/28] CUDA...")
seasc_ext=load_inline(name="akbascore_test189_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD,verbose=False)
print("       OK")

# ================================================================================================================
# [2/28] MODEL
# ================================================================================================================
print("\n[2/28] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},
 attn_implementation="sdpa",trust_remote_code=True)
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
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
    p=tok(chat(ctx),return_tensors="pt",add_special_tokens=False)
    c=tok(cont,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p.input_ids,c.input_ids],1).to(DEVICE)
    return ids,torch.ones_like(ids),p.input_ids.shape[1]

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
# [3/28] FROZEN E AXIS
# ================================================================================================================
print("\n[3/28] Frozen E axis...")
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
# [4/28] FISHER / BANK
# ================================================================================================================
print("\n[4/28] Fisher/Bank...")
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
# [5/28] P8
# ================================================================================================================
print("\n[5/28] P8...")
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
# [6/28] FROZEN FORGE
# ================================================================================================================
print("\n[6/28] Frozen forge...")
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
# [7/28] FROZEN GRAD / XRAY / HOLDOUT SETS
# ================================================================================================================
print("\n[7/28] Frozen GRAD / XRAY / HOLDOUT sets...")
GRAD_X=[
("A cracked hose produces a loss of pressure."," The cracked hose is the cause."," The pressure loss is the effect."),
("A failed fan produces overheating."," The failed fan is the cause."," The overheating is the effect."),
("A puncture produces loss of air pressure."," The puncture is the cause."," The pressure loss is the effect."),
("A short circuit produces a shutdown."," The short circuit is the cause."," The shutdown is the effect."),
("A blocked drain produces water accumulation."," The blocked drain is the cause."," The water accumulation is the effect."),
("A damaged bearing produces excessive vibration."," The bearing damage is the cause."," The vibration is the effect."),
("A clogged filter produces reduced flow."," The clogged filter is the cause."," The reduced flow is the effect."),
("A loose connection produces an intermittent signal."," The loose connection is the cause."," The intermittent signal is the effect.")]
XRAY=[
"A fractured pipe and a pressure drop form a causal sequence.",
"A blocked air passage and rising temperature form a causal sequence.",
"A damaged seal and fluid loss form a causal sequence.",
"A current surge and a failed circuit form a causal sequence.",
"A stuck mechanism and restricted motion form a causal sequence.",
"A worn bearing and increased vibration form a causal sequence.",
"A clogged outlet and accumulated liquid form a causal sequence.",
"A strong impact and a deformed bracket form a causal sequence."]
HOLD=[
"A broken coupling and loss of transmitted motion form a causal sequence.",
"A blocked intake and reduced airflow form a causal sequence.",
"A damaged gasket and pressure leakage form a causal sequence.",
"A failed coolant pump and rising temperature form a causal sequence.",
"A seized bearing and restricted rotation form a causal sequence.",
"A severed cable and loss of electrical power form a causal sequence.",
"A clogged nozzle and reduced discharge form a causal sequence.",
"A fractured support and structural displacement form a causal sequence."]
print("       GRAD=8 XRAY=8 HOLDOUT=8")

# ================================================================================================================
# [8/28] SEASC HOOKS
# ================================================================================================================
print("\n[8/28] Frozen SEASC hooks...")
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
print("       ready")

# ================================================================================================================
# [9/28] EXACT PRE27 / RAW27 RECORDER
# ================================================================================================================
print("\n[9/28] Exact PRE27 / RAW27 recorder...")
def run_transport(prompt,sign):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    box={};hooks=seasc_hooks(sign)
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
    if "pre27" not in box or "raw27" not in box:raise RuntimeError("L27 transport capture failed.")
    return box
print("       ready")

# ================================================================================================================
# [10/28] VALIDATED LEAF-REPLACEMENT GRADIENT ENGINE
# ================================================================================================================
print("\n[10/28] Validated PRE27 leaf-replacement gradient engine...")
def _hs(args,kwargs):
    if len(args)>0:return args[0]
    if "hidden_states" in kwargs:return kwargs["hidden_states"]
    raise RuntimeError("L27 hidden_states input not found.")

def cont_score(logits,ids,plen):
    z=logits[:,plen-1:-1,:].float();tar=ids[:,plen:]
    if z.shape[1]!=tar.shape[1] or tar.shape[1]<1:
        raise RuntimeError(f"Continuation alignment mismatch: {tuple(z.shape)} vs {tuple(tar.shape)}")
    return F.log_softmax(z,-1).gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean(),z

@torch.no_grad()
def reference_forward(ids,mask,plen,sign):
    box={};hooks=seasc_hooks(sign)
    def cap(m,args,kwargs):box["h"]=_hs(args,kwargs).detach().clone()
    hooks.append(layers[27].register_forward_pre_hook(cap,with_kwargs=True))
    try:o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    if "h" not in box:raise RuntimeError("PRE27 capture failed.")
    sc,z=cont_score(o.logits,ids,plen)
    return box["h"],sc.detach(),z.detach()

def leaf_forward(ids,mask,plen,sign,leaf):
    box={"replaced":False,"identity":False,"out_rg":False};hooks=seasc_hooks(sign)
    def rep(m,args,kwargs):
        inc=_hs(args,kwargs);box["incoming"]=inc.detach().clone()
        if inc.shape!=leaf.shape or inc.dtype!=leaf.dtype:
            raise RuntimeError(f"L27 input/leaf mismatch: {tuple(inc.shape)} {inc.dtype} vs {tuple(leaf.shape)} {leaf.dtype}")
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
    leaf=h0.detach().clone().requires_grad_(True)
    if not leaf.is_leaf or not leaf.requires_grad:raise RuntimeError("PRE27 leaf construction failed.")
    model.zero_grad(set_to_none=True)
    try:
        sc,z,box=leaf_forward(ids,mask,plen,sign,leaf)
        if not box["replaced"] or not box["identity"] or not box["out_rg"]:
            raise RuntimeError("L27 leaf replacement integrity failure.")
        hc=h0.float();nh=hc.norm().clamp_min(EPS)
        repl=float((leaf.detach().float()-hc).norm()/nh)
        ups=float((box["incoming"].float()-hc).norm()/nh)
        serr=float((sc.detach()-ref_sc).abs())
        lerr=float((z.detach()-ref_z).norm()/ref_z.norm().clamp_min(EPS))
        if repl>REPL_TOL:raise RuntimeError(f"Replacement relerr={repl:.3e}")
        if ups>UPSTREAM_TOL:raise RuntimeError(f"Upstream relerr={ups:.3e}")
        if serr>SCORE_TOL:raise RuntimeError(f"Score error={serr:.3e}")
        if lerr>LOGIT_TOL:raise RuntimeError(f"Logit relerr={lerr:.3e}")
        gfull=torch.autograd.grad(sc,leaf,retain_graph=False,create_graph=False)[0]
        g=gfull[0].float().sum(0).detach().clone();gn=float(g.norm())
        if not torch.isfinite(g).all() or gn<=EPS:raise RuntimeError("Invalid L27 gradient.")
    finally:model.zero_grad(set_to_none=True)
    return g,{"replacement_relerr":repl,"upstream_relerr":ups,"score_err":serr,"logits_relerr":lerr,"grad_norm":gn}
print("       ready")

# ================================================================================================================
# [11/28] GRADIENT PREFLIGHT
# ================================================================================================================
print("\n[11/28] Gradient preflight...")
_,PF=local_l27_gradient(GRAD_X[0][0],GRAD_X[0][1],+1)
print(f"       replacement relerr={PF['replacement_relerr']:.3e}")
print(f"       upstream relerr   ={PF['upstream_relerr']:.3e}")
print(f"       score error       ={PF['score_err']:.3e}")
print(f"       logits relerr     ={PF['logits_relerr']:.3e}")
print(f"       gradient norm     ={PF['grad_norm']:.6e}")
print("       PREFLIGHT PASSED")

# ================================================================================================================
# [12/28] INDEPENDENT BEHAVIOR GRADIENT
# ================================================================================================================
print("\n[12/28] Independent behavior gradients...")
GROWS=[];DIAG=[]
for i,(ctx,pos,neg) in enumerate(GRAD_X,1):
    gpP,d1=local_l27_gradient(ctx,pos,+1);gnP,d2=local_l27_gradient(ctx,neg,+1)
    gpM,d3=local_l27_gradient(ctx,pos,-1);gnM,d4=local_l27_gradient(ctx,neg,-1)
    g=.5*((gpP-gnP)+(gpM-gnM))
    if not torch.isfinite(g).all() or float(g.norm())<=EPS:raise RuntimeError(f"Degenerate gradient item {i}")
    GROWS.append(norm(g))
    ds=[d1,d2,d3,d4]
    DIAG.append({"item":i,"max_repl":max(d["replacement_relerr"] for d in ds),
                 "max_upstream":max(d["upstream_relerr"] for d in ds),
                 "max_score":max(d["score_err"] for d in ds),
                 "max_logits":max(d["logits_relerr"] for d in ds)})
    print(f"       {i}/8 repl={DIAG[-1]['max_repl']:.1e} ups={DIAG[-1]['max_upstream']:.1e} "
          f"score={DIAG[-1]['max_score']:.1e} logits={DIAG[-1]['max_logits']:.1e}")
GROWS=torch.stack(GROWS)
G=norm(GROWS.mean(0));sg=GROWS@G
GROWS=torch.where(sg[:,None]<0,-GROWS,GROWS);G=norm(GROWS.mean(0))
GCOS=GROWS@G
print(f"       sign flips={int((sg<0).sum())}/8")
print(f"       mean consensus cos={float(GCOS.mean()):+.6f}")
print(f"       min consensus cos ={float(GCOS.min()):+.6f}")
print(f"       cos(G,ORTH19)     ={float(torch.dot(G,U)):+.6f}")
G=G.detach().clone()
print("       G_BEH frozen before XRAY/HOLDOUT")

# ================================================================================================================
# [13/28] XRAY FULL-SPACE TRANSPORT
# ================================================================================================================
print("\n[13/28] XRAY full-space transport...")
XP=[];XR=[];XD=[]
for i,p in enumerate(XRAY,1):
    P=run_transport(p,+1);M=run_transport(p,-1)
    dp=.5*(P["pre27"]-M["pre27"]);dr=.5*(P["raw27"]-M["raw27"])
    XP.append(dp);XR.append(dr);XD.append(dr-dp)
    print(f"       {i}/8")
XP=torch.stack(XP);XR=torch.stack(XR);XD=torch.stack(XD)

# ================================================================================================================
# [14/28] FULL-SPACE BEHAVIOR PROJECTION
# ================================================================================================================
print("\n[14/28] Full-space behavior projection...")
BPRE=XP@G;BRAW=XR@G;DB=XD@G
identity_err=float((DB-(BRAW-BPRE)).abs().max())
if identity_err>1e-5:raise RuntimeError(f"Projection identity failure: {identity_err:.3e}")
print(f"       B_PRE = {float(BPRE.mean()):+.6f}")
print(f"       B_RAW = {float(BRAW.mean()):+.6f}")
print(f"       ΔB27  = {float(DB.mean()):+.6f}")
print(f"       identity maxerr={identity_err:.3e}")
print(f"       P+(ΔB27)={int((DB>0).sum())}/8")

# ================================================================================================================
# [15/28] NORMALIZED BEHAVIOR ALIGNMENT
# ================================================================================================================
print("\n[15/28] Normalized behavior alignment...")
CP=cosv(XP,G[None].expand_as(XP))
CR=cosv(XR,G[None].expand_as(XR))
CD=cosv(XD,G[None].expand_as(XD))
print(f"       cos(PRE,G)={float(CP.mean()):+.6f}")
print(f"       cos(RAW,G)={float(CR.mean()):+.6f}")
print(f"       cos(Δ,G)  ={float(CD.mean()):+.6f}")
print(f"       P+ cosΔG  ={int((CD>0).sum())}/8")

# ================================================================================================================
# [16/28] BEHAVIOR COMPONENT FRACTION
# ================================================================================================================
print("\n[16/28] Behavior-sensitive Δ27 component...")
GCOMP=DB[:,None]*G
GFRAC=GCOMP.norm(dim=-1)/XD.norm(dim=-1).clamp_min(EPS)
print(f"       ||Δ_G||       ={float(GCOMP.norm(dim=-1).mean()):.6f}")
print(f"       ||Δ||         ={float(XD.norm(dim=-1).mean()):.6f}")
print(f"       Δ G-norm frac ={float(GFRAC.mean()):.6f}")
print(f"       Δ G-energy frac={float(GFRAC.square().mean()):.8f}")

# ================================================================================================================
# [17/28] ORTH19 VS BEHAVIOR COMPONENT
# ================================================================================================================
print("\n[17/28] ORTH19 vs behavior component...")
DU=XD@U
UFRAC=DU.abs()/XD.norm(dim=-1).clamp_min(EPS)
print(f"       mean Δ@ORTH19={float(DU.mean()):+.6f}")
print(f"       mean Δ@G     ={float(DB.mean()):+.6f}")
print(f"       |ORTH19| norm frac={float(UFRAC.mean()):.6f}")
print(f"       |G_BEH| norm frac ={float(GFRAC.mean()):.6f}")
print(f"       cos(G,ORTH19)={float(torch.dot(G,U)):+.6f}")

# ================================================================================================================
# [18/28] PER-PROMPT XRAY
# ================================================================================================================
print("\n[18/28] Per-prompt XRAY...")
ROWS=[]
for i in range(8):
    r={"set":"XRAY","item":i+1,"B_pre":float(BPRE[i]),"B_raw":float(BRAW[i]),"delta_B":float(DB[i]),
       "cos_pre_G":float(CP[i]),"cos_raw_G":float(CR[i]),"cos_delta_G":float(CD[i]),
       "delta_orth19":float(DU[i]),"delta_norm":float(XD[i].norm()),"G_fraction":float(GFRAC[i])}
    ROWS.append(r)
    print(f"       {i+1}: B {r['B_pre']:+.4f}→{r['B_raw']:+.4f} ΔB={r['delta_B']:+.4f} "
          f"cosΔG={r['cos_delta_G']:+.4f} ΔU={r['delta_orth19']:+.4f}")

# ================================================================================================================
# [19/28] XRAY BOOTSTRAP
# ================================================================================================================
print("\n[19/28] XRAY bootstrap...")
def boot(x,n=20000,seed=189):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT=[]
for j,(name,x) in enumerate([
    ("DELTA_B",DB.cpu().numpy()),("COS_DELTA_G",CD.cpu().numpy()),
    ("B_PRE",BPRE.cpu().numpy()),("B_RAW",BRAW.cpu().numpy()),
    ("G_NORM_FRAC",GFRAC.cpu().numpy()),("DELTA_ORTH19",DU.cpu().numpy())]):
    m,lo,hi=boot(x,seed=SEED+j);BOOT.append({"set":"XRAY","metric":name,"mean":m,"low":lo,"high":hi})
    print(f"       {name:<15} {m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [20/28] SIGN-FLIP PERMUTATION NULL
# ================================================================================================================
print("\n[20/28] XRAY sign-flip permutation null...")
obs=abs(float(DB.mean()));vals=DB.detach().cpu().numpy().astype(np.float64)
rng=np.random.default_rng(SEED);N_PERM=50000;cnt=0
for _ in range(N_PERM):
    s=rng.choice(np.array([-1.,1.]),size=len(vals))
    if abs(float((vals*s).mean()))>=obs-1e-15:cnt+=1
P_PERM=(cnt+1)/(N_PERM+1)
print(f"       observed |mean ΔB|={obs:.6f}")
print(f"       sign-flip p={P_PERM:.6f} ({N_PERM} permutations)")

# ================================================================================================================
# [21/28] HOLDOUT REPLICATION
# ================================================================================================================
print("\n[21/28] HOLDOUT replication...")
HP=[];HR=[];HD=[]
for i,p in enumerate(HOLD,1):
    P=run_transport(p,+1);M=run_transport(p,-1)
    dp=.5*(P["pre27"]-M["pre27"]);dr=.5*(P["raw27"]-M["raw27"])
    HP.append(dp);HR.append(dr);HD.append(dr-dp)
    print(f"       {i}/8")
HP=torch.stack(HP);HR=torch.stack(HR);HD=torch.stack(HD)
HBPRE=HP@G;HBRAW=HR@G;HDB=HD@G
HCP=cosv(HP,G[None].expand_as(HP));HCR=cosv(HR,G[None].expand_as(HR));HCD=cosv(HD,G[None].expand_as(HD))
HGFRAC=HDB.abs()/HD.norm(dim=-1).clamp_min(EPS)
HDU=HD@U
print(f"       B_PRE={float(HBPRE.mean()):+.6f}")
print(f"       B_RAW={float(HBRAW.mean()):+.6f}")
print(f"       ΔB27 ={float(HDB.mean()):+.6f}")
print(f"       cos(Δ,G)={float(HCD.mean()):+.6f}")
print(f"       P+(ΔB)={int((HDB>0).sum())}/8")

# ================================================================================================================
# [22/28] HOLDOUT PER-PROMPT
# ================================================================================================================
print("\n[22/28] HOLDOUT per-prompt...")
for i in range(8):
    r={"set":"HOLDOUT","item":i+1,"B_pre":float(HBPRE[i]),"B_raw":float(HBRAW[i]),"delta_B":float(HDB[i]),
       "cos_pre_G":float(HCP[i]),"cos_raw_G":float(HCR[i]),"cos_delta_G":float(HCD[i]),
       "delta_orth19":float(HDU[i]),"delta_norm":float(HD[i].norm()),"G_fraction":float(HGFRAC[i])}
    ROWS.append(r)
    print(f"       {i+1}: B {r['B_pre']:+.4f}→{r['B_raw']:+.4f} ΔB={r['delta_B']:+.4f} "
          f"cosΔG={r['cos_delta_G']:+.4f} ΔU={r['delta_orth19']:+.4f}")

# ================================================================================================================
# [23/28] HOLDOUT BOOTSTRAP + PERMUTATION
# ================================================================================================================
print("\n[23/28] HOLDOUT bootstrap + permutation...")
for j,(name,x) in enumerate([
    ("DELTA_B",HDB.cpu().numpy()),("COS_DELTA_G",HCD.cpu().numpy()),
    ("B_PRE",HBPRE.cpu().numpy()),("B_RAW",HBRAW.cpu().numpy()),
    ("G_NORM_FRAC",HGFRAC.cpu().numpy()),("DELTA_ORTH19",HDU.cpu().numpy())]):
    m,lo,hi=boot(x,seed=SEED+20+j);BOOT.append({"set":"HOLDOUT","metric":name,"mean":m,"low":lo,"high":hi})
    print(f"       {name:<15} {m:+.6f} [{lo:+.6f},{hi:+.6f}]")
hobs=abs(float(HDB.mean()));hvals=HDB.detach().cpu().numpy().astype(np.float64);cnt=0
rng=np.random.default_rng(SEED+99)
for _ in range(N_PERM):
    s=rng.choice(np.array([-1.,1.]),size=len(hvals))
    if abs(float((hvals*s).mean()))>=hobs-1e-15:cnt+=1
HP_PERM=(cnt+1)/(N_PERM+1)
print(f"       HOLD sign-flip p={HP_PERM:.6f}")

# ================================================================================================================
# [24/28] XRAY ↔ HOLDOUT CONSISTENCY
# ================================================================================================================
print("\n[24/28] XRAY ↔ HOLDOUT consistency...")
XDB=float(DB.mean());HDBM=float(HDB.mean())
XCOS=float(CD.mean());HCOS=float(HCD.mean())
same_sign=(XDB*HDBM)>0
print(f"       XRAY ΔB   ={XDB:+.6f}")
print(f"       HOLD ΔB   ={HDBM:+.6f}")
print(f"       same sign ={same_sign}")
print(f"       XRAY cosΔG={XCOS:+.6f}")
print(f"       HOLD cosΔG={HCOS:+.6f}")

# ================================================================================================================
# [25/28] MEAN-VECTOR GEOMETRY
# ================================================================================================================
print("\n[25/28] Mean-vector geometry...")
MX=norm(XD.mean(0));MH=norm(HD.mean(0))
print(f"       cos(mean XRAY Δ,G) ={float(torch.dot(MX,G)):+.6f}")
print(f"       cos(mean HOLD Δ,G) ={float(torch.dot(MH,G)):+.6f}")
print(f"       cos(mean XRAY Δ,mean HOLD Δ)={float(torch.dot(MX,MH)):+.6f}")
print(f"       cos(G,ORTH19)={float(torch.dot(G,U)):+.6f}")
print(f"       cos(mean XRAY Δ,ORTH19)={float(torch.dot(MX,U)):+.6f}")
print(f"       cos(mean HOLD Δ,ORTH19)={float(torch.dot(MH,U)):+.6f}")

# ================================================================================================================
# [26/28] DECISION
# ================================================================================================================
print("\n[26/28] TEST189 decision...")
XB=next(x for x in BOOT if x["set"]=="XRAY" and x["metric"]=="DELTA_B")
HB=next(x for x in BOOT if x["set"]=="HOLDOUT" and x["metric"]=="DELTA_B")
XPOS=XB["low"]>0 or XB["high"]<0
HPOS=HB["low"]>0 or HB["high"]<0
if XPOS and HPOS and same_sign:
    DECISION="FULLSPACE_L27_BEHAVIORAL_TRANSPORT_REPLICATED"
elif XPOS and same_sign:
    DECISION="FULLSPACE_L27_BEHAVIORAL_TRANSPORT_SIGNAL__HOLDOUT_NOT_CONFIRMED"
elif same_sign and (P_PERM<.05 or HP_PERM<.05):
    DECISION="FULLSPACE_L27_BEHAVIORAL_TRANSPORT_WEAK_SIGNAL"
else:
    DECISION="FULLSPACE_L27_BEHAVIORAL_TRANSPORT_NOT_CONFIRMED"
print("       XRAY ΔB CI   =",f"[{XB['low']:+.6f},{XB['high']:+.6f}]")
print("       HOLD ΔB CI   =",f"[{HB['low']:+.6f},{HB['high']:+.6f}]")
print("       XRAY perm p  =",f"{P_PERM:.6f}")
print("       HOLD perm p  =",f"{HP_PERM:.6f}")
print("       DECISION:",DECISION)
print("       Geometric/mechanistic result only; no behavioral intervention claim.")

# ================================================================================================================
# [27/28] SAVE
# ================================================================================================================
print("\n[27/28] Saving...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("PER_PROMPT.csv",ROWS);savecsv("BOOTSTRAP.csv",BOOT);savecsv("GRAD_INTEGRITY.csv",DIAG)
np.savez_compressed(ROOT/"TEST189_FULLSPACE_TRANSPORT.npz",
 behavior_gradient=G.cpu().numpy(),orth19=U.cpu().numpy(),
 xray_pre=XP.cpu().numpy(),xray_raw=XR.cpu().numpy(),xray_delta=XD.cpu().numpy(),
 hold_pre=HP.cpu().numpy(),hold_raw=HR.cpu().numpy(),hold_delta=HD.cpu().numpy(),
 grad_rows=GROWS.cpu().numpy())
REPORT={"test":"TEST189","parent":["TEST187","TEST188"],"decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
"mean_fb":float(np.mean(FB)),"behavior_gradient_orth19":float(torch.dot(G,U)),
"xray":{"B_pre":float(BPRE.mean()),"B_raw":float(BRAW.mean()),"delta_B":XDB,"cos_delta_G":XCOS,
         "Pplus":int((DB>0).sum()),"perm_p":P_PERM},
"holdout":{"B_pre":float(HBPRE.mean()),"B_raw":float(HBRAW.mean()),"delta_B":HDBM,"cos_delta_G":HCOS,
            "Pplus":int((HDB>0).sum()),"perm_p":HP_PERM},
"bootstrap":BOOT,
"integrity":{"test187_architecture_preserved":True,"seasc_l0_l19_frozen":True,"l20_l27_unmodified":True,
"behavior_gradient_frozen_before_xray":True,"full_3584d_test":True,"pca_used":False,"K_selection":False,
"new_vector_injected":False,"final_behavior_opened":False,"controller":False,"rescue":False,"weights_modified":False,
"gradient_engine":"validated PRE27 leaf replacement through native Qwen forward"}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [28/28] COMPLETE
# ================================================================================================================
print("\n[28/28] COMPLETE")
print("="*138)
print("TEST 189 — FULL-SPACE BEHAVIORAL TRANSPORT X-RAY — SUMMARY")
print("="*138)
print(f"mean cos(F,B)={np.mean(FB):+.6f}")
print(f"behavior consensus mean cos={float(GCOS.mean()):+.6f} min={float(GCOS.min()):+.6f}")
print(f"cos(G,ORTH19)={float(torch.dot(G,U)):+.6f}")
print(f"XRAY: B_PRE={float(BPRE.mean()):+.6f} B_RAW={float(BRAW.mean()):+.6f} ΔB={XDB:+.6f} P+={int((DB>0).sum())}/8")
print(f"XRAY: cos(Δ,G)={XCOS:+.6f} G-norm-frac={float(GFRAC.mean()):.6f} perm-p={P_PERM:.6f}")
print(f"HOLD: B_PRE={float(HBPRE.mean()):+.6f} B_RAW={float(HBRAW.mean()):+.6f} ΔB={HDBM:+.6f} P+={int((HDB>0).sum())}/8")
print(f"HOLD: cos(Δ,G)={HCOS:+.6f} G-norm-frac={float(HGFRAC.mean()):.6f} perm-p={HP_PERM:.6f}")
print(f"cos(mean XRAY Δ,mean HOLD Δ)={float(torch.dot(MX,MH)):+.6f}")
print("PCA used: False | K selected: NONE | New vector injected: False | FINAL opened: False")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
