# ======================================================================================================================
# TEST 171 — COLLATERAL-NULL CAUSAL VECTOR FORGE
# AkbasCore 3.2 / SEASC
# CAUSAL_ORTH -> NEUTRAL COLLATERAL NULLSPACE -> CAUSAL_SAFE
#
# PRIMARY QUESTION:
# Can the high-capacity TEST169 CAUSAL_ORTH direction preserve most of its causal effect
# after removing a neutral collateral-sensitive activation subspace?
#
# LOCKS:
# - Qwen/Qwen2.5-7B-Instruct BF16 / SDPA / A100
# - L0-L19
# - TEST169 CAUSAL_ORTH reconstruction unchanged
# - P8 retention guard unchanged for lineage comparability
# - SEASC frozen-norm CUDA injection unchanged
# - Collateral basis is built ONLY from SEARCH neutral probes
# - Collateral basis uses token-norm-weighted actuator gradients
# - Basis rank K selected on DEV only, frozen before FINAL
# - FINAL causal / neutral / retention sets never select K or vector
# - No runtime controller / gate / brake / cap / boost / FINAL adaptation
# ======================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,gc,importlib.util
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------
# BUILD DEPENDENCY GUARD
# ----------------------------------------------------------------------------------------------------------------------
if importlib.util.find_spec("ninja") is None:
    print("Installing required build dependency: ninja...")
    subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import ninja

import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154);print("TEST 171 — COLLATERAL-NULL CAUSAL VECTOR FORGE");print("CAUSAL_ORTH -> NEUTRAL COLLATERAL NULLSPACE -> CAUSAL_SAFE");print("="*154)
print("Ninja:",ninja.__version__)
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=171
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
P8_COUNT=8;SVD_REL_TOL=1e-5;PRIMARY_SCALE=.50
COLL_RANKS=[1,2,4,6,8];SCALES=[.125,.25,.50,.75,1.]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST171_COLLATERAL_NULL_CAUSAL_FORGE");ROOT.mkdir(parents=True,exist_ok=True)
print("GPU:",torch.cuda.get_device_name(0));print("Model:",MODEL_ID);print("Layers: L0-L19 | source=TEST169 CAUSAL_ORTH | primary dose=0.50")

# ======================================================================================================================
# LOCKED SEASC ENVELOPE
# ======================================================================================================================
def env(L):
    x=ZIRVE*math.exp(-SONUM*float(L))*(1+SONUM*float(L))+TABAN
    return x/(ZIRVE+TABAN)
ENVELOPE=np.asarray([env(L) for L in range(N_LAYERS)],dtype=np.float64)
RHO=IVME*ENVELOPE;RSS=float(np.sqrt(np.sum(RHO**2)))
print(f"Locked SEASC RSS dose budget: {RSS:.9f}")

# ======================================================================================================================
# CUDA SEASC — TEST169/170 INJECTION PHYSICS UNCHANGED
# ======================================================================================================================
BUILD="/tmp/akbascore_test171";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST171 SEASC");}"""
CUDA=r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename scalar_t>
__global__ void k(scalar_t* h,const float* a,const float* d,const int B,const int S,const int H){
int vec=blockIdx.x,b=vec/S;if(b>=B)return;extern __shared__ float sh[];
long long base=(long long)vec*H,ab=(long long)b*H;float ss=0.f;
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[base+j];ss+=x*x;}sh[threadIdx.x]=ss;__syncthreads();
for(unsigned s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
float n=sqrtf(fmaxf(sh[0],1e-20f)),scale=d[b]*n;__syncthreads();
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[base+j];h[base+j]=(scalar_t)(x+scale*a[ab+j]);}}
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
auto out=hidden.contiguous().clone();auto act=actuator.to(hidden.device(),torch::kFloat32).contiguous();auto d=dose.to(hidden.device(),torch::kFloat32).contiguous();
int B=out.size(0),S=out.size(1),H=out.size(2),nv=B*S;constexpr int T=256;size_t sb=T*sizeof(float);cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test171_seasc",[&]{k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/22] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test171_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
print("       OK")

# ======================================================================================================================
# MODEL
# ======================================================================================================================
print("\n[2/22] Loading frozen model...")
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa",trust_remote_code=True)
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;HIDDEN=model.config.hidden_size
def chat(x):return tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def norm(v):return v/v.norm(dim=-1,keepdim=True).clamp_min(EPS)
print("       hidden:",HIDDEN,"dtype:",next(model.parameters()).dtype)

# ======================================================================================================================
# TEST169/170 REFERENCE GEOMETRY
# ======================================================================================================================
SEARCH_PAIRS=[
("The lantern is above the stone arch.","The lantern is below the stone arch."),("The kite is above the old tower.","The kite is below the old tower."),
("The flag is above the wooden gate.","The flag is below the wooden gate."),("The bird is above the narrow bridge.","The bird is below the narrow bridge."),
("The sign is above the glass doorway.","The sign is below the glass doorway."),("The lamp is above the metal shelf.","The lamp is below the metal shelf."),
("The clock is above the brown cabinet.","The clock is below the brown cabinet."),("The balloon is above the green hill.","The balloon is below the green hill."),
("The camera is above the small desk.","The camera is below the small desk."),("The painting is above the blue sofa.","The painting is below the blue sofa."),
("The cloud is above the distant ridge.","The cloud is below the distant ridge."),("The marker is above the white board.","The marker is below the white board."),
("The drone is above the concrete wall.","The drone is below the concrete wall."),("The shelf is above the storage box.","The shelf is below the storage box."),
("The light is above the front window.","The light is below the front window."),("The symbol is above the horizontal line.","The symbol is below the horizontal line.")]

@torch.inference_mode()
def capture(text):
    e=tokenizer(chat(text),return_tensors="pt").to(DEVICE);o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);pos=int(e["attention_mask"][0].sum())-1
    hs=[o.hidden_states[L+1][0,pos].float().detach() for L in range(N_LAYERS)];del e,o;return hs

print("\n[3/22] Rebuilding TEST169 Fisher / Bank geometry...")
PB,NB=[[] for _ in range(N_LAYERS)],[[] for _ in range(N_LAYERS)]
for i,(p,n) in enumerate(SEARCH_PAIRS,1):
    hp,hn=capture(p),capture(n)
    for L in range(N_LAYERS):PB[L].append(hp[L]);NB[L].append(hn[L])
    print(f"       pair {i:02d}/16")
PB=[torch.stack(x).float() for x in PB];NB=[torch.stack(x).float() for x in NB]
FISHER,BANK,COS,RESN=[],[],[],[]
for L in range(N_LAYERS):
    Hp,Hn=PB[L],NB[L];mp,mn=Hp.mean(0),Hn.mean(0);md=mp-mn;RR=torch.cat([Hp-mp,Hn-mn],0).float();nu=max(RR.shape[0]-2,1)
    _,S,Vh=torch.linalg.svd(RR,full_matrices=False);eig=S.square()/float(nu);pe=eig[eig>FISHER_EPS];es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS);V=Vh.T;pr=V.T@md;f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float()
    dirs=norm(Hp-Hn);dirs=dirs*torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]));b=norm(dirs.mean(0)).float();c=torch.dot(f,b).clamp(-1,1)
    if c<0:b,c=-b,-c
    r=b-c*f;FISHER.append(f);BANK.append(b);COS.append(float(c));RESN.append(float(r.norm()))
    print(f"       L{L:02d} cos(F,B)={float(c):+.6f} residual={float(r.norm()):.6f}")
FISHER=torch.stack(FISHER).contiguous();BANK=torch.stack(BANK).contiguous()
print(f"       mean cos={np.mean(COS):+.6f} | mean residual={np.mean(RESN):.6f}")

# ======================================================================================================================
# P8 RETENTION BASIS — LINEAGE LOCK: SAME TEST169/170 CONSTRUCTION
# ======================================================================================================================
RETENTION_BUILD=[
("A ceramic cup rests beside a folded newspaper."," The scene contains ordinary stationary objects."),
("A metal ruler lies beside a closed notebook."," The scene contains ordinary stationary objects."),
("A wooden bowl sits near a plain cloth."," The scene contains ordinary stationary objects."),
("A glass jar rests beside a cardboard package."," The scene contains ordinary stationary objects."),
("A calculator lies next to a paper envelope."," The objects remain stationary."),
("A closed umbrella rests beside a travel bag."," The objects remain stationary."),
("A spoon lies beside an empty plate."," The objects remain stationary."),
("A pencil rests next to a blank sheet of paper."," The objects remain stationary.")]

def sequence(context,continuation):
    p=tokenizer(chat(context),return_tensors="pt",add_special_tokens=False);c=tokenizer(continuation,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p["input_ids"],c["input_ids"]],1).to(DEVICE);return ids,torch.ones_like(ids),p["input_ids"].shape[1]

def activation_gradient(context,continuation):
    ids,mask,plen=sequence(context,continuation);emb=model.get_input_embeddings()(ids).detach().requires_grad_(True);saved=[None]*N_LAYERS;hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(m,a,o):
                h=o[0] if isinstance(o,tuple) else o;h.retain_grad();saved[li]=h
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    model.zero_grad(set_to_none=True)
    try:
        o=model(inputs_embeds=emb,attention_mask=mask,use_cache=False,return_dict=True);z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];lp=F.log_softmax(z,-1)
        loss=-lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();loss.backward();g=[saved[L].grad[0].float().sum(0).detach().clone() for L in range(N_LAYERS)];lv=float(loss.detach())
    finally:
        for h in hs:h.remove()
        model.zero_grad(set_to_none=True)
    del ids,mask,emb,o,z,tar,lp,loss,saved;return g,lv

print("\n[4/22] Computing frozen P8 retention basis...")
PG=[]
for i,(ctx,cont) in enumerate(RETENTION_BUILD,1):
    g,l=activation_gradient(ctx,cont);PG.append(g);print(f"       probe {i}/8 | NLL={l:.6f} | mean ||g||={np.mean([float(x.norm()) for x in g]):.6e}")
Q8=[];RANKS=[]
for L in range(N_LAYERS):
    G=torch.stack([PG[i][L] for i in range(P8_COUNT)]).float();G=G[G.norm(dim=1)>EPS];G=norm(G);_,s,Vh=torch.linalg.svd(G,full_matrices=False)
    tol=max(float(s[0])*SVD_REL_TOL,EPS);rank=int((s>tol).sum());Q8.append(Vh[:rank].T.contiguous());RANKS.append(rank)
print("       P8 ranks:",RANKS)

def project_basis(raw,bases):
    out=[];tel=[]
    for L in range(N_LAYERS):
        u=raw[L].float();q=bases[L];p=q@(q.T@u) if q.shape[1] else torch.zeros_like(u);ug=u-p;rn=float(u.norm());rem=float(ug.norm())
        if rem<EPS:raise RuntimeError(f"Projection annihilated L{L}")
        ug=ug/rem;tel.append({"layer":L,"overlap_energy":float(p.square().sum()/u.square().sum().clamp_min(EPS)),"retained_norm_fraction":rem/max(rn,EPS),"post_Q":float((q.T@ug).norm()) if q.shape[1] else 0.});out.append(ug)
    return torch.stack(out).contiguous(),tel
FISHER_GUARD,FISHER_GT=project_basis(FISHER,Q8)

# ======================================================================================================================
# TEST169 CAUSAL_ORTH RECONSTRUCTION — UNCHANGED
# ======================================================================================================================
FORGE={
"A":[
{"context":"A weather instrument and a field station are visible, with no vertical relation specified.","pos":" The weather instrument is above the field station.","neg":" The weather instrument is below the field station."},
{"context":"A suspended sensor and a factory platform are visible, with no vertical relation specified.","pos":" The suspended sensor is above the factory platform.","neg":" The suspended sensor is below the factory platform."},
{"context":"A balloon and a village road are visible, with no vertical relation specified.","pos":" The balloon is above the village road.","neg":" The balloon is below the village road."},
{"context":"A hanging lamp and a workbench are visible, with no vertical relation specified.","pos":" The hanging lamp is above the workbench.","neg":" The hanging lamp is below the workbench."}],
"B":[
{"context":"A marker and a cabinet are shown. Their relative height is unspecified.","pos":" The marker is higher than the cabinet.","neg":" The marker is lower than the cabinet."},
{"context":"A sign and a doorway are shown. Their relative height is unspecified.","pos":" The sign is higher than the doorway.","neg":" The sign is lower than the doorway."},
{"context":"A camera and a desk are shown. Their relative height is unspecified.","pos":" The camera is higher than the desk.","neg":" The camera is lower than the desk."},
{"context":"A light and a shelf are shown. Their relative height is unspecified.","pos":" The light is higher than the shelf.","neg":" The light is lower than the shelf."}],
"C":[
{"context":"Consider the vertical axis containing a drone and a wall.","pos":" The drone occupies the upper position relative to the wall.","neg":" The drone occupies the lower position relative to the wall."},
{"context":"Consider the vertical axis containing a flag and a gate.","pos":" The flag occupies the upper position relative to the gate.","neg":" The flag occupies the lower position relative to the gate."},
{"context":"Consider the vertical axis containing a bird and a bridge.","pos":" The bird occupies the upper position relative to the bridge.","neg":" The bird occupies the lower position relative to the bridge."},
{"context":"Consider the vertical axis containing a cloud and a ridge.","pos":" The cloud occupies the upper position relative to the ridge.","neg":" The cloud occupies the lower position relative to the ridge."}],
"D":[
{"context":"Two objects are present: a kite and a tower. No spatial ordering is given.","pos":" Vertically, the kite is over the tower.","neg":" Vertically, the kite is under the tower."},
{"context":"Two objects are present: a clock and a cabinet. No spatial ordering is given.","pos":" Vertically, the clock is over the cabinet.","neg":" Vertically, the clock is under the cabinet."},
{"context":"Two objects are present: a painting and a sofa. No spatial ordering is given.","pos":" Vertically, the painting is over the sofa.","neg":" Vertically, the painting is under the sofa."},
{"context":"Two objects are present: a symbol and a line. No spatial ordering is given.","pos":" Vertically, the symbol is over the line.","neg":" Vertically, the symbol is under the line."}]}

def continuation_gradient(context,continuation):
    ids,mask,plen=sequence(context,continuation);emb=model.get_input_embeddings()(ids).detach().requires_grad_(True);saved=[None]*N_LAYERS;hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(m,a,o):
                h=o[0] if isinstance(o,tuple) else o;h.retain_grad();saved[li]=h
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    model.zero_grad(set_to_none=True)
    try:
        o=model(inputs_embeds=emb,attention_mask=mask,use_cache=False,return_dict=True);z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];lp=F.log_softmax(z,-1)
        score=lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();score.backward();g=[saved[L].grad[0].float().sum(0).detach().clone() for L in range(N_LAYERS)]
    finally:
        for h in hs:h.remove()
        model.zero_grad(set_to_none=True)
    del ids,mask,emb,o,z,tar,lp,score,saved;return g

print("\n[5/22] Reconstructing TEST169 CAUSAL_ORTH...")
FG={}
for fn,items in FORGE.items():
    acc=[torch.zeros(HIDDEN,device=DEVICE,dtype=torch.float32) for _ in range(N_LAYERS)]
    for j,x in enumerate(items,1):
        gp=continuation_gradient(x["context"],x["pos"]);gn=continuation_gradient(x["context"],x["neg"])
        for L in range(N_LAYERS):acc[L]+=gp[L]-gn[L]
        print(f"       family {fn} item {j}/4")
    v=torch.stack([norm(x) for x in acc]).contiguous()
    for L in range(N_LAYERS):
        if torch.dot(v[L],FISHER[L])<0:v[L]=-v[L]
    FG[fn]=v
CAUSAL=torch.stack([norm(torch.stack([FG[k][L] for k in FG]).mean(0)) for L in range(N_LAYERS)]).contiguous()
CAUSAL_GUARD,CGT=project_basis(CAUSAL,Q8);CAUSAL_ORTH=[];ORTH=[]
for L in range(N_LAYERS):
    f=FISHER_GUARD[L];g=CAUSAL_GUARD[L];r=g-torch.dot(g,f)*f;rn=r.norm()
    if rn<EPS:raise RuntimeError(f"CAUSAL_ORTH collapsed L{L}")
    CAUSAL_ORTH.append(r/rn);ORTH.append(float(rn))
CAUSAL_ORTH=torch.stack(CAUSAL_ORTH).contiguous()
print(f"       mean cos(Causal,Fisher)={np.mean([float(torch.dot(CAUSAL[L],FISHER[L])) for L in range(N_LAYERS)]):+.6f}")
print(f"       mean Fisher-orthogonal component={np.mean(ORTH):.6f}")
print(f"       mean P8 overlap after CAUSAL_ORTH={np.mean([float((Q8[L].T@CAUSAL_ORTH[L]).norm()) for L in range(N_LAYERS)]):.8e}")

# ======================================================================================================================
# COLLATERAL BUILD / DEV SETS — FINAL COMPLETELY SEPARATE
# ======================================================================================================================
COLL_BUILD=[
("A ceramic vase rests beside a folded cloth."," The objects remain stationary."),
("A closed box sits beside a plain notebook."," The objects remain stationary."),
("A metal key lies beside a paper card."," The objects remain stationary."),
("A wooden tray rests near an empty glass."," The objects remain stationary."),
("A paper folder lies beside a plastic ruler."," The objects remain stationary."),
("A clean bowl rests beside a cotton napkin."," The objects remain stationary."),
("A small mirror stands beside a closed case."," The objects remain stationary."),
("A metal spoon rests beside an empty cup."," The objects remain stationary.")]

DEV_NEUTRAL=[
("A wooden brush lies beside a sealed carton."," The objects remain stationary."),
("A plain bottle stands next to a folded cloth."," The objects remain stationary."),
("A notebook rests beside a small container."," The objects remain stationary."),
("A metal clip lies beside a clean sheet of paper."," The objects remain stationary."),
("A ceramic plate sits beside an empty jar."," The objects remain stationary."),
("A fabric pouch rests beside a wooden block."," The objects remain stationary.")]

DEV_RETENTION=[
("A plain cup rests beside a closed booklet."," The objects remain stationary."),
("A steel ruler lies next to a paper folder."," The objects remain stationary."),
("A wooden spoon rests beside an empty dish."," The objects remain stationary."),
("A glass container stands beside a folded towel."," The objects remain stationary."),
("A pencil lies beside a sealed envelope."," The objects remain stationary."),
("A small calculator rests next to a blank card."," The objects remain stationary.")]

# ======================================================================================================================
# TOKEN-NORM-WEIGHTED COLLATERAL ACTUATOR GRADIENT
# Exact direction derivative for common frozen-norm actuator, up to layer scalar rho.
# Loss is continuation NLL, nonzero at baseline.
# ======================================================================================================================
def actuator_gradient_weighted(context,continuation):
    ids,mask,plen=sequence(context,continuation);emb=model.get_input_embeddings()(ids).detach().requires_grad_(True);saved=[None]*N_LAYERS;hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(m,a,o):
                h=o[0] if isinstance(o,tuple) else o;h.retain_grad();saved[li]=h
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    model.zero_grad(set_to_none=True)
    try:
        o=model(inputs_embeds=emb,attention_mask=mask,use_cache=False,return_dict=True);z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];lp=F.log_softmax(z,-1)
        loss=-lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();loss.backward();gg=[]
        for L in range(N_LAYERS):
            h=saved[L].detach()[0].float();g=saved[L].grad[0].float();w=h.norm(dim=-1,keepdim=True);gg.append((w*g).sum(dim=0).detach().clone())
        lv=float(loss.detach())
    finally:
        for hnd in hs:hnd.remove()
        model.zero_grad(set_to_none=True)
    del ids,mask,emb,o,z,tar,lp,loss,saved;return gg,lv

print("\n[6/22] Building token-norm-weighted collateral gradient bank...")
COLL_G=[]
for i,(ctx,cont) in enumerate(COLL_BUILD,1):
    g,l=actuator_gradient_weighted(ctx,cont);COLL_G.append(g)
    print(f"       neutral {i}/8 | NLL={l:.6f} | mean ||g_act||={np.mean([float(x.norm()) for x in g]):.6e}")

# ======================================================================================================================
# NESTED COLLATERAL BASES K=1/2/4/6/8
# SVD is built once from all 8 BUILD probes; K is truncation rank only.
# ======================================================================================================================
print("\n[7/22] Constructing nested collateral subspaces...")
COLL_FULL=[];COLL_SVAL=[]
for L in range(N_LAYERS):
    G=torch.stack([COLL_G[i][L] for i in range(len(COLL_G))]).float();G=G[G.norm(dim=1)>EPS];G=norm(G);_,s,Vh=torch.linalg.svd(G,full_matrices=False)
    COLL_FULL.append(Vh.T.contiguous());COLL_SVAL.append(s.detach().cpu().numpy())
COLL_BASES={}
for K in COLL_RANKS:
    COLL_BASES[K]=[COLL_FULL[L][:,:min(K,COLL_FULL[L].shape[1])].contiguous() for L in range(N_LAYERS)]
    print(f"       K={K} ranks:",[q.shape[1] for q in COLL_BASES[K]])

# ======================================================================================================================
# COLLATERAL-NULL CANDIDATE VECTORS
# Projection order:
# CAUSAL_ORTH -> collateral null -> re-null P8 -> re-orthogonalize FisherGuard -> re-null P8 -> normalize
# This prevents numerical reintroduction of previously removed components.
# ======================================================================================================================
def make_safe(K):
    qcoll=COLL_BASES[K];out=[];tel=[]
    for L in range(N_LAYERS):
        u=CAUSAL_ORTH[L].float();qc=qcoll[L]
        pc=qc@(qc.T@u) if qc.shape[1] else torch.zeros_like(u);v=u-pc
        q=Q8[L];v=v-(q@(q.T@v) if q.shape[1] else 0)
        f=FISHER_GUARD[L];v=v-torch.dot(v,f)*f
        v=v-(q@(q.T@v) if q.shape[1] else 0)
        rn=v.norm()
        if rn<EPS:raise RuntimeError(f"CAUSAL_SAFE K={K} collapsed L{L}")
        v=v/rn
        tel.append({"layer":L,"K":K,"coll_overlap_energy":float(pc.square().sum()/u.square().sum().clamp_min(EPS)),"pre_norm_retained":float(rn),"post_coll_overlap":float((qc.T@v).norm()) if qc.shape[1] else 0.,"post_P8_overlap":float((q.T@v).norm()) if q.shape[1] else 0.,"post_fisher_cos":float(torch.dot(v,f))})
        out.append(v)
    return torch.stack(out).contiguous(),tel

SAFE_VECTORS={};SAFE_TEL={}
for K in COLL_RANKS:
    SAFE_VECTORS[K],SAFE_TEL[K]=make_safe(K)
    print(f"       K={K} overlap_energy={np.mean([x['coll_overlap_energy'] for x in SAFE_TEL[K]]):.6f} retained={np.mean([x['pre_norm_retained'] for x in SAFE_TEL[K]]):.6f} postColl={np.mean([x['post_coll_overlap'] for x in SAFE_TEL[K]]):.3e}")

# ======================================================================================================================
# SEASC ASSAYS — LOCKED TEST170 STRUCTURE
# ======================================================================================================================
def blank_tel():return [{"requested":[],"realized":[]} for _ in range(N_LAYERS)]
def hooks(vectors,sign,scale,tel,state=None):
    hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(module,args,out):
                old=out[0] if isinstance(out,tuple) else out;rest=out[1:] if isinstance(out,tuple) else None;B=old.shape[0]
                direction=(float(sign)*vectors[li][None,:].expand(B,-1)).float().contiguous();req=float(RHO[li])*float(scale);dose=torch.full((B,),req,device=DEVICE,dtype=torch.float32)
                oldlast=old[:,-1,:].float();new=seasc_ext.seasc_batch(old,direction,dose);newlast=new[:,-1,:].float()
                real=(newlast-oldlast).norm(dim=-1)/oldlast.norm(dim=-1).clamp_min(EPS)
                tel[li]["requested"].extend(dose.detach().cpu().tolist());tel[li]["realized"].extend(real.detach().cpu().tolist())
                if state is not None:state[li]=newlast.detach().float().cpu()
                return new if rest is None else (new,)+rest
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    return hs

@torch.inference_mode()
def logp(context,continuation,v=None,sign=0,scale=0):
    ids,mask,plen=sequence(context,continuation);tel=blank_tel();hs=hooks(v,sign,scale,tel) if v is not None else []
    try:o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];lp=F.log_softmax(z,-1);score=float(lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean())
    del ids,mask,o,z,tar,lp;return score,tel

def arm(name,v,scale,items):
    cs,pm,mm,req,real=[],[],[],[],[]
    for x in items:
        pp,t1=logp(x["context"],x["pos"],v,+1,scale);pn,t2=logp(x["context"],x["neg"],v,+1,scale)
        np_,t3=logp(x["context"],x["pos"],v,-1,scale);nn,t4=logp(x["context"],x["neg"],v,-1,scale)
        a,b=pp-pn,np_-nn;cs.append(.5*(a-b));pm.append(a);mm.append(b)
        for t in (t1,t2,t3,t4):
            for L in range(N_LAYERS):req+=t[L]["requested"];real+=t[L]["realized"]
    return {"method":name,"scale":float(scale),"Csym":float(np.mean(cs)),"Csym_sd":float(np.std(cs,ddof=1)),"plus_margin":float(np.mean(pm)),"minus_margin":float(np.mean(mm)),"positive_items":int(np.sum(np.asarray(cs)>0)),"requested_mean":float(np.mean(req)),"realized_mean":float(np.mean(real)),"item_csym":[float(x) for x in cs]}

@torch.inference_mode()
def collateral_one(ctx,cont,v,scale,sign):
    ids,mask,plen=sequence(ctx,cont);o0=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True);z0=o0.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];p0=F.log_softmax(z0,-1)
    n0=-p0.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();hs=hooks(v,sign,scale,blank_tel())
    try:o1=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    z1=o1.logits[:,plen-1:-1,:].float();p1=F.log_softmax(z1,-1);n1=-p1.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();kl=(p0.exp()*(p0-p1)).sum(-1).mean();ans=(float(n1-n0),float(kl))
    del ids,mask,o0,o1,z0,z1,tar,p0,p1,n0,n1,kl;return ans

def collateral(name,v,scale,probes):
    z=[]
    for ctx,cont in probes:z+=[collateral_one(ctx,cont,v,scale,+1),collateral_one(ctx,cont,v,scale,-1)]
    return {"method":name,"scale":float(scale),"kl_sym":float(np.mean([x[1] for x in z])),"abs_delta_nll_sym":float(np.mean([abs(x[0]) for x in z]))}

def retention(name,v,scale,probes):
    vals=[]
    for ctx,cont in probes:
        b,_=logp(ctx,cont,None,0,0)
        for s in (+1,-1):
            x,_=logp(ctx,cont,v,s,scale);vals.append(float(-x+b))
    return {"method":name,"scale":float(scale),"retention_abs_delta_nll_sym":float(np.mean(np.abs(vals))),"per_sign_delta":[float(x) for x in vals]}

@torch.inference_mode()
def states(ctx,v=None,scale=0,sign=0):
    e=tokenizer(chat(ctx),return_tensors="pt").to(DEVICE);cap=[None]*N_LAYERS;hs=[]
    if v is None:
        for L in range(N_LAYERS):
            def mk(li):
                def hk(m,a,o):cap[li]=(o[0] if isinstance(o,tuple) else o)[:,-1,:].detach().float().cpu()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
    else:hs=hooks(v,sign,scale,blank_tel(),cap)
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return cap

def canal(v,scale,items):
    dp,dt,tot=[],[],[]
    for x in items:
        h0=states(x["context"])
        for s in (+1,-1):
            h1=states(x["context"],v,scale,s)
            for L in range(N_LAYERS):
                a=h0[L][0];d=h1[L][0]-a;t=float(s)*v[L].detach().cpu().float();al=torch.dot(d,t);par=al*t;per=d-par;den=a.norm().clamp_min(EPS)
                tot.append(float(d.norm()/den));dt.append(float(par.norm()/den));dp.append(float(per.norm()/den))
    return {"D_perp":float(np.mean(dp)),"D_target":float(np.mean(dt)),"D_total":float(np.mean(tot)),"off_target_fraction":float(np.mean(dp)/max(np.mean(tot),EPS))}

# ======================================================================================================================
# DEV RANK ABLATION — K SELECTED BEFORE FINAL
# Baseline K=0 = untouched TEST169 CAUSAL_ORTH.
# ======================================================================================================================
print("\n[8/22] DEV collateral-rank ablation...")
DEV=[]
BASE_FAM=[arm("CAUSAL_ORTH",CAUSAL_ORTH,PRIMARY_SCALE,it)["Csym"] for it in FORGE.values()]
BASE_COLL=collateral("CAUSAL_ORTH",CAUSAL_ORTH,PRIMARY_SCALE,DEV_NEUTRAL)
BASE_RET=retention("CAUSAL_ORTH",CAUSAL_ORTH,PRIMARY_SCALE,DEV_RETENTION)
BASE_H=canal(CAUSAL_ORTH,PRIMARY_SCALE,FORGE["A"][:2])
BASE={"K":0,"name":"CAUSAL_ORTH","C_mean":float(np.mean(BASE_FAM)),"C_worst":float(np.min(BASE_FAM)),"C_sd":float(np.std(BASE_FAM,ddof=1)),"A":BASE_FAM[0],"B":BASE_FAM[1],"C":BASE_FAM[2],"D":BASE_FAM[3],"KL":BASE_COLL["kl_sym"],"Ret":BASE_RET["retention_abs_delta_nll_sym"],"Dperp":BASE_H["D_perp"],"off":BASE_H["off_target_fraction"],"removed":0.}
DEV.append(BASE)
print(f"       K=0 BASE C={BASE['C_mean']:+.6f} worst={BASE['C_worst']:+.6f} KL={BASE['KL']:.6f} Ret={BASE['Ret']:.6f} Dp={BASE['Dperp']:.6f}")
for K in COLL_RANKS:
    v=SAFE_VECTORS[K];fam=[arm(f"K{K}",v,PRIMARY_SCALE,it)["Csym"] for it in FORGE.values()];c=collateral(f"K{K}",v,PRIMARY_SCALE,DEV_NEUTRAL);r=retention(f"K{K}",v,PRIMARY_SCALE,DEV_RETENTION);h=canal(v,PRIMARY_SCALE,FORGE["A"][:2])
    removed=float(np.mean([x["coll_overlap_energy"] for x in SAFE_TEL[K]]))
    row={"K":K,"name":f"COLL_NULL_K{K}","C_mean":float(np.mean(fam)),"C_worst":float(np.min(fam)),"C_sd":float(np.std(fam,ddof=1)),"A":fam[0],"B":fam[1],"C":fam[2],"D":fam[3],"KL":c["kl_sym"],"Ret":r["retention_abs_delta_nll_sym"],"Dperp":h["D_perp"],"off":h["off_target_fraction"],"removed":removed};DEV.append(row)
    print(f"       K={K} C={row['C_mean']:+.6f} worst={row['C_worst']:+.6f} sd={row['C_sd']:.6f} KL={row['KL']:.6f} Ret={row['Ret']:.6f} Dp={row['Dperp']:.6f} removed={removed:.6f}")

# ======================================================================================================================
# VIABILITY / SAFE / PARETO
# Require >=90% causal mean and positive worst family.
# Safety: no >10% worsening in KL/Ret/Dperp AND at least one >=5% improvement.
# ======================================================================================================================
print("\n[9/22] Viability / safety / Pareto...")
VIABLE=[x for x in DEV if x["K"]>0 and x["C_mean"]>=.90*BASE["C_mean"] and x["C_worst"]>0]
SAFE=[x for x in VIABLE if x["KL"]<=1.10*BASE["KL"] and x["Ret"]<=1.10*BASE["Ret"] and x["Dperp"]<=1.10*BASE["Dperp"] and (x["KL"]<=.95*BASE["KL"] or x["Ret"]<=.95*BASE["Ret"] or x["Dperp"]<=.95*BASE["Dperp"])]
def dominates(a,b):
    nw=a["C_mean"]>=b["C_mean"] and a["C_worst"]>=b["C_worst"] and a["KL"]<=b["KL"] and a["Ret"]<=b["Ret"] and a["Dperp"]<=b["Dperp"] and a["C_sd"]<=b["C_sd"]
    st=a["C_mean"]>b["C_mean"] or a["C_worst"]>b["C_worst"] or a["KL"]<b["KL"] or a["Ret"]<b["Ret"] or a["Dperp"]<b["Dperp"] or a["C_sd"]<b["C_sd"];return nw and st
POOL=SAFE if SAFE else VIABLE
PARETO=[x for x in POOL if not any(dominates(y,x) for y in POOL if y is not x)]
print("       viable:",[x["K"] for x in VIABLE]);print("       safe:",[x["K"] for x in SAFE]);print("       Pareto:",[x["K"] for x in PARETO])

# ======================================================================================================================
# LOFO FAMILY STABILITY
# ======================================================================================================================
print("\n[10/22] LOFO family stability...")
LOFO=[];candidate_K=[x["K"] for x in (SAFE if SAFE else VIABLE)]
if not candidate_K:candidate_K=[0]
for held in FORGE:
    rows=[]
    for K in candidate_K:
        v=CAUSAL_ORTH if K==0 else SAFE_VECTORS[K];vals=[arm(f"K{K}",v,PRIMARY_SCALE,it)["Csym"] for fn,it in FORGE.items() if fn!=held]
        rows.append({"K":K,"C":float(np.mean(vals)),"worst":float(np.min(vals)),"sd":float(np.std(vals,ddof=1))})
    chosen=max(rows,key=lambda x:(x["worst"],x["C"],-x["sd"],-x["K"]));LOFO.append({"held":held,**chosen})
    print(f"       hold {held} -> K={chosen['K']} C={chosen['C']:+.6f} worst={chosen['worst']:+.6f}")

# ======================================================================================================================
# LOCAL RANK ROBUSTNESS
# ======================================================================================================================
print("\n[11/22] Rank robustness...")
ROBUST=[]
for row in (SAFE if SAFE else VIABLE):
    K=row["K"];neighbors=[k for k in COLL_RANKS if abs(k-K)<=2]
    vals=[]
    for k in neighbors:
        fam=[arm(f"K{k}",SAFE_VECTORS[k],PRIMARY_SCALE,it)["Csym"] for it in FORGE.values()];vals.append((k,float(np.mean(fam))))
    sd=float(np.std([x[1] for x in vals],ddof=1)) if len(vals)>1 else 0.
    ROBUST.append({**row,"neighbor_sd":sd,"neighbors":[[int(k),float(v)] for k,v in vals]})
    print(f"       K={K} neighbors={[(k,round(v,5)) for k,v in vals]} sd={sd:.6f}")

# ======================================================================================================================
# FREEZE — NO FINAL ACCESS
# Safety/collateral reduction is primary once causal >=90% is satisfied.
# ======================================================================================================================
print("\n[12/22] FREEZE...")
if SAFE:
    RP=[x for x in ROBUST if x["K"] in [y["K"] for y in SAFE]]
    def safety_score(x):
        return x["KL"]/max(BASE["KL"],EPS)+x["Ret"]/max(BASE["Ret"],EPS)+x["Dperp"]/max(BASE["Dperp"],EPS)
    WIN=min(RP,key=lambda x:(safety_score(x),x["neighbor_sd"],-x["C_worst"],-x["C_mean"],x["K"]))
    FROZEN_K=int(WIN["K"]);FALLBACK=False
else:
    FROZEN_K=0;WIN=BASE;FALLBACK=True
FROZEN_VECTOR=CAUSAL_ORTH if FROZEN_K==0 else SAFE_VECTORS[FROZEN_K]
print("       FROZEN K:",FROZEN_K);print("       fallback:",FALLBACK)
if FROZEN_K>0:
    print(f"       collateral overlap energy removed={np.mean([x['coll_overlap_energy'] for x in SAFE_TEL[FROZEN_K]]):.6f}")
    print(f"       mean post-collateral overlap={np.mean([x['post_coll_overlap'] for x in SAFE_TEL[FROZEN_K]]):.3e}")
print("       FINAL remains unopened.")

# ======================================================================================================================
# FINAL — FIRST ACCESS AFTER FREEZE
# ======================================================================================================================
FINAL=[
{"context":"A rescue drone and a coastal road are visible, with no vertical relation specified.","pos":" The rescue drone is above the coastal road.","neg":" The rescue drone is below the coastal road."},
{"context":"A hanging display and a service counter are visible, with no vertical relation specified.","pos":" The hanging display is above the service counter.","neg":" The hanging display is below the service counter."},
{"context":"A hawk and a cliff path are visible, with no vertical relation specified.","pos":" The hawk is above the cliff path.","neg":" The hawk is below the cliff path."},
{"context":"A suspended inspection camera and a machine table are visible, with no vertical relation specified.","pos":" The suspended inspection camera is above the machine table.","neg":" The suspended inspection camera is below the machine table."},
{"context":"A navigation beacon and a pier deck are visible, with no vertical relation specified.","pos":" The navigation beacon is above the pier deck.","neg":" The navigation beacon is below the pier deck."},
{"context":"A balloon and an orchard are visible, with no vertical relation specified.","pos":" The balloon is above the orchard.","neg":" The balloon is below the orchard."},
{"context":"A ceiling projector and a classroom table are visible, with no vertical relation specified.","pos":" The ceiling projector is above the classroom table.","neg":" The ceiling projector is below the classroom table."},
{"context":"A suspended detector and a laboratory bench are visible, with no vertical relation specified.","pos":" The suspended detector is above the laboratory bench.","neg":" The suspended detector is below the laboratory bench."},
{"context":"A radar antenna and an airport building are visible, with no vertical relation specified.","pos":" The radar antenna is above the airport building.","neg":" The radar antenna is below the airport building."},
{"context":"A hanging banner and a reception desk are visible, with no vertical relation specified.","pos":" The hanging banner is above the reception desk.","neg":" The hanging banner is below the reception desk."},
{"context":"A glider and a highway are visible, with no vertical relation specified.","pos":" The glider is above the highway.","neg":" The glider is below the highway."},
{"context":"A suspended microphone and a theatre floor are visible, with no vertical relation specified.","pos":" The suspended microphone is above the theatre floor.","neg":" The suspended microphone is below the theatre floor."},
{"context":"A warning lamp and a loading bay are visible, with no vertical relation specified.","pos":" The warning lamp is above the loading bay.","neg":" The warning lamp is below the loading bay."},
{"context":"A cloud and a canyon floor are visible, with no vertical relation specified.","pos":" The cloud is above the canyon floor.","neg":" The cloud is below the canyon floor."},
{"context":"A hanging sculpture and a museum bench are visible, with no vertical relation specified.","pos":" The hanging sculpture is above the museum bench.","neg":" The hanging sculpture is below the museum bench."},
{"context":"An observation balloon and a riverbank are visible, with no vertical relation specified.","pos":" The observation balloon is above the riverbank.","neg":" The observation balloon is below the riverbank."}]

FINAL_NEUTRAL=[
("A bottle stands beside a folded towel."," The scene contains ordinary stationary objects."),
("A tray rests beside a pair of gloves."," The scene contains ordinary stationary objects."),
("A plate sits beside a clean napkin."," The scene contains ordinary stationary objects."),
("A suitcase stands near a wooden chair."," The scene contains ordinary stationary objects."),
("A small radio rests beside a book."," The scene contains ordinary stationary objects."),
("A cardboard folder lies beside a metal pen."," The scene contains ordinary stationary objects.")]

RET_FINAL=[
("A plain mug stands beside a sealed magazine."," The objects remain stationary."),
("A steel clip lies next to a closed folder."," The objects remain stationary."),
("A wooden spoon rests beside a clean towel."," The objects remain stationary."),
("A glass bottle stands near an empty container."," The objects remain stationary."),
("A small brush lies beside a sheet of cardboard."," The objects remain stationary."),
("A fabric pouch rests next to a plastic case."," The objects remain stationary.")]

print("\n[13/22] FINAL — FIRST ACCESS AFTER FREEZE...")
ARMS={"CAUSAL_ORTH":CAUSAL_ORTH,"CAUSAL_SAFE":FROZEN_VECTOR}
FINAL_CAUSAL=[];FINAL_COLL=[];FINAL_RET=[]
for scale in SCALES:
    print(f"\n       DOSE={scale:.3f}")
    for name,v in ARMS.items():
        a=arm(name,v,scale,FINAL);c=collateral(name,v,scale,FINAL_NEUTRAL);r=retention(name,v,scale,RET_FINAL)
        FINAL_CAUSAL.append(a);FINAL_COLL.append(c);FINAL_RET.append(r)
        print(f"       {name:<14} C={a['Csym']:+.6f} P+={a['positive_items']:02d}/16 KL={c['kl_sym']:.6f} HeldRet={r['retention_abs_delta_nll_sym']:.6f}")

# ======================================================================================================================
# FINAL HIDDEN-STATE CANALIZATION
# ======================================================================================================================
print("\n[14/22] FINAL hidden-state canalization...")
FINAL_HIDDEN=[]
for scale in SCALES:
    for name,v in ARMS.items():
        h=canal(v,scale,FINAL[:4]);FINAL_HIDDEN.append({"method":name,"scale":scale,**h})
        print(f"       {name:<14} s={scale:.3f} Dperp={h['D_perp']:.6f} Dtarget={h['D_target']:.6f} off={h['off_target_fraction']:.4f}")

# ======================================================================================================================
# BOOTSTRAP — CAUSAL + HELD-OUT RETENTION
# ======================================================================================================================
print("\n[15/22] Paired bootstrap...")
def get(rows,name,scale):return next(x for x in rows if x["method"]==name and abs(x["scale"]-scale)<1e-9)
def bootdiff(a,b,n=20000,seed=171,absolute=False):
    aa=np.asarray(a,dtype=np.float64);bb=np.asarray(b,dtype=np.float64)
    if absolute:aa=np.abs(aa);bb=np.abs(bb)
    d=aa-bb;rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):
        ix=rng.integers(0,len(d),len(d));z[i]=d[ix].mean()
    return {"delta":float(d.mean()),"ci_low":float(np.quantile(z,.025)),"ci_high":float(np.quantile(z,.975))}
BOOT_CAUSAL=[];BOOT_RET=[]
for scale in SCALES:
    s=get(FINAL_CAUSAL,"CAUSAL_SAFE",scale);b=get(FINAL_CAUSAL,"CAUSAL_ORTH",scale);zc=bootdiff(s["item_csym"],b["item_csym"],seed=SEED+int(scale*1000))
    sr=get(FINAL_RET,"CAUSAL_SAFE",scale);br=get(FINAL_RET,"CAUSAL_ORTH",scale);zr=bootdiff(sr["per_sign_delta"],br["per_sign_delta"],seed=SEED+5000+int(scale*1000),absolute=True)
    BOOT_CAUSAL.append({"scale":scale,**zc});BOOT_RET.append({"scale":scale,**zr})
    print(f"       s={scale:.3f} SAFE-ORTH ΔC={zc['delta']:+.6f} [{zc['ci_low']:+.6f},{zc['ci_high']:+.6f}] | Δ|Ret|={zr['delta']:+.6f} [{zr['ci_low']:+.6f},{zr['ci_high']:+.6f}]")

# ======================================================================================================================
# PRIMARY MECHANISM
# ======================================================================================================================
print("\n[16/22] Primary mechanism @0.50...")
PRIMARY=[]
for name in ARMS:
    a=get(FINAL_CAUSAL,name,.5);c=get(FINAL_COLL,name,.5);r=get(FINAL_RET,name,.5);h=get(FINAL_HIDDEN,name,.5)
    x={"method":name,"C":a["Csym"],"KL":c["kl_sym"],"HeldRet":r["retention_abs_delta_nll_sym"],"Dperp":h["D_perp"],"Dtarget":h["D_target"],"off":h["off_target_fraction"],"P":a["positive_items"]};PRIMARY.append(x)
    print(f"       {name:<14} C={x['C']:+.6f} KL={x['KL']:.6f} HeldRet={x['HeldRet']:.6f} Dperp={x['Dperp']:.6f} off={x['off']:.4f} P+={x['P']}/16")

# ======================================================================================================================
# GEOMETRY / NULLSPACE VERIFICATION
# ======================================================================================================================
print("\n[17/22] Geometry / nullspace verification...")
GEOM=[]
for name,v in ARMS.items():
    coll=[];p8=[];fc=[]
    if FROZEN_K>0:
        qc=COLL_BASES[FROZEN_K]
        for L in range(N_LAYERS):coll.append(float((qc[L].T@v[L]).norm()))
    else:coll=[0.]*N_LAYERS
    for L in range(N_LAYERS):
        p8.append(float((Q8[L].T@v[L]).norm()));fc.append(float(torch.dot(v[L],FISHER_GUARD[L])))
    row={"method":name,"coll_overlap_mean":float(np.mean(coll)),"coll_overlap_max":float(np.max(coll)),"P8_overlap_mean":float(np.mean(p8)),"P8_overlap_max":float(np.max(p8)),"fisher_cos_mean":float(np.mean(fc))};GEOM.append(row)
    print(f"       {name:<14} CollQ={row['coll_overlap_mean']:.8e} max={row['coll_overlap_max']:.8e} P8={row['P8_overlap_mean']:.8e} FisherCos={row['fisher_cos_mean']:+.8f}")

# ======================================================================================================================
# CAUSAL RETENTION FRACTION / COLLATERAL REDUCTION
# ======================================================================================================================
print("\n[18/22] Effect-retention ratios...")
RATIOS=[]
for scale in SCALES:
    b=get(FINAL_CAUSAL,"CAUSAL_ORTH",scale);s=get(FINAL_CAUSAL,"CAUSAL_SAFE",scale)
    bc=get(FINAL_COLL,"CAUSAL_ORTH",scale);sc=get(FINAL_COLL,"CAUSAL_SAFE",scale)
    br=get(FINAL_RET,"CAUSAL_ORTH",scale);sr=get(FINAL_RET,"CAUSAL_SAFE",scale)
    row={"scale":scale,"causal_fraction":float(s["Csym"]/b["Csym"]) if abs(b["Csym"])>EPS else float("nan"),"KL_fraction":float(sc["kl_sym"]/bc["kl_sym"]) if bc["kl_sym"]>EPS else float("nan"),"Ret_fraction":float(sr["retention_abs_delta_nll_sym"]/br["retention_abs_delta_nll_sym"]) if br["retention_abs_delta_nll_sym"]>EPS else float("nan")}
    RATIOS.append(row);print(f"       s={scale:.3f} causal={row['causal_fraction']:.4f} KL={row['KL_fraction']:.4f} HeldRet={row['Ret_fraction']:.4f}")

# ======================================================================================================================
# DECISION — PRIMARY DOSE, FROZEN BEFORE FINAL
# ======================================================================================================================
print("\n[19/22] TEST171 decision...")
A=next(x for x in PRIMARY if x["method"]=="CAUSAL_SAFE");B=next(x for x in PRIMARY if x["method"]=="CAUSAL_ORTH")
BC=get(BOOT_CAUSAL,"" if False else "x",0) if False else next(x for x in BOOT_CAUSAL if abs(x["scale"]-.5)<1e-9)
BR=next(x for x in BOOT_RET if abs(x["scale"]-.5)<1e-9)
causal90=A["C"]>=.90*B["C"];causal80=A["C"]>=.80*B["C"];kl10=A["KL"]<=.90*B["KL"];ret10=A["HeldRet"]<=.90*B["HeldRet"];dp10=A["Dperp"]<=.90*B["Dperp"]
causal_not_catastrophic=BC["ci_low"]>-.25*B["C"];ret_boot_improved=BR["ci_high"]<0
if FROZEN_K==0:DECISION="NO_SAFE_COLLATERAL_NULL_VECTOR_FOUND"
elif causal90 and kl10 and ret10 and causal_not_catastrophic:DECISION="COLLATERAL_NULL_CAUSAL_VECTOR_SUPPORTED"
elif causal90 and (kl10 or ret10 or dp10):DECISION="PARTIAL_COLLATERAL_NULL_SIGNAL"
elif causal80 and (kl10 or ret10 or dp10):DECISION="WEAK_COLLATERAL_NULL_SIGNAL"
else:DECISION="COLLATERAL_NULL_ADVANTAGE_NOT_CONFIRMED"
print("       Frozen K:",FROZEN_K);print("       causal >=90%:",causal90);print("       KL >=10% reduction:",kl10);print("       held-ret >=10% reduction:",ret10);print("       Dperp >=10% reduction:",dp10);print("       retention bootstrap upper<0:",ret_boot_improved);print("       DECISION:",DECISION)

# ======================================================================================================================
# SAVE
# ======================================================================================================================
print("\n[20/22] Saving...")
def scalar(rows):return [{k:v for k,v in r.items() if not isinstance(v,(list,tuple,dict,np.ndarray))} for r in rows]
def savecsv(name,rows):
    rows=scalar(rows)
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

savecsv("DEV_RANK_ABLATION.csv",DEV);savecsv("LOFO.csv",LOFO);savecsv("ROBUSTNESS.csv",ROBUST);savecsv("FINAL_CAUSAL.csv",FINAL_CAUSAL);savecsv("FINAL_COLLATERAL.csv",FINAL_COLL);savecsv("FINAL_RETENTION_HELDOUT.csv",FINAL_RET);savecsv("FINAL_HIDDEN.csv",FINAL_HIDDEN);savecsv("BOOT_CAUSAL.csv",BOOT_CAUSAL);savecsv("BOOT_RETENTION.csv",BOOT_RET);savecsv("PRIMARY.csv",PRIMARY);savecsv("GEOMETRY.csv",GEOM);savecsv("RATIOS.csv",RATIOS)

np.savez_compressed(ROOT/"TEST171_COLLATERAL_NULL_CAUSAL_FORGE.npz",
causal_orth=CAUSAL_ORTH.detach().cpu().numpy(),
causal_safe=FROZEN_VECTOR.detach().cpu().numpy(),
fisher=FISHER.detach().cpu().numpy(),
fisher_guard=FISHER_GUARD.detach().cpu().numpy(),
rho=np.asarray(RHO,dtype=np.float32),
frozen_k=np.asarray([FROZEN_K],dtype=np.int32))

REPORT={
"test":"TEST 171",
"title":"COLLATERAL-NULL CAUSAL VECTOR FORGE",
"model":MODEL_ID,"seed":SEED,"frozen_k":FROZEN_K,"fallback":FALLBACK,"decision":DECISION,
"dev":DEV,"lofo":LOFO,"robustness":ROBUST,"final_causal":FINAL_CAUSAL,"final_collateral":FINAL_COLL,
"final_retention_heldout":FINAL_RET,"final_hidden":FINAL_HIDDEN,"bootstrap_causal":BOOT_CAUSAL,
"bootstrap_retention":BOOT_RET,"primary":PRIMARY,"geometry":GEOM,"ratios":RATIOS,
"contract":[
"TEST169 CAUSAL_ORTH is reconstructed before collateral-null synthesis.",
"P8 retention construction remains lineage-compatible with TEST169/170.",
"Collateral basis is activation-space and built only from COLL_BUILD neutral probes.",
"Collateral actuator gradients use token-norm weighting matching frozen-norm SEASC perturbation geometry.",
"DEV_NEUTRAL and DEV_RETENTION are distinct from collateral-build probes.",
"Collateral rank K is selected using DEV only and frozen before FINAL.",
"FINAL causal, FINAL neutral and RET_FINAL do not select K.",
"SEASC layer envelope and total dose remain locked; TEST170 envelope search is not repeated.",
"No runtime controller, gate, brake, cap, saturation, boost or FINAL adaptation.",
"A positive result supports removable collateral-sensitive geometry inside CAUSAL_ORTH; it does not imply all collateral is represented by the measured nullspace."
]}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)

# ======================================================================================================================
# INTEGRITY
# ======================================================================================================================
print("\n[21/22] Integrity...")
print("       P8 ranks:",RANKS);print("       Collateral rank frozen:",FROZEN_K);print("       SEASC envelope changed: NO")
print(f"       RSS dose budget: {RSS:.9f}");print("       FINAL selection access: NONE");print("       Model weights modified: NO");print("       Runtime controller: OFF")
if FROZEN_K>0:
    print(f"       mean final CollQ overlap: {next(x for x in GEOM if x['method']=='CAUSAL_SAFE')['coll_overlap_mean']:.8e}")
    print(f"       mean final P8 overlap: {next(x for x in GEOM if x['method']=='CAUSAL_SAFE')['P8_overlap_mean']:.8e}")

# ======================================================================================================================
# COMPLETE
# ======================================================================================================================
print("\n[22/22] COMPLETE")
print("="*154);print("TEST 171 — COLLATERAL-NULL CAUSAL VECTOR FORGE — SUMMARY");print("="*154)
print(f"Mean cos(Fisher,Bank): {np.mean(COS):+.6f}");print(f"Mean CAUSAL_ORTH Fisher-orthogonal component: {np.mean(ORTH):.6f}")
print("Frozen collateral rank K:",FROZEN_K);print("Fallback:",FALLBACK);print("DECISION:",DECISION);print("Saved:",ROOT);print("="*154)
