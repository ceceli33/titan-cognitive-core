# ======================================================================================================================
# TEST 170 — ENERGY-MATCHED ENVELOPE RELEASE ABLATION
# AkbasCore 3.2 / SEASC
# CAUSAL_ORTH + FOUR-ZONE DEPTH REDISTRIBUTION UNDER FIXED RSS DOSE
# Only layer-dose distribution is released. Direction, injection physics and total RSS dose remain fixed.
# ======================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,gc,importlib.util
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------
# BUILD DEPENDENCY GUARD — Colab/runtime may not ship with Ninja
# ----------------------------------------------------------------------------------------------------------------------
if importlib.util.find_spec("ninja") is None:
    print("Installing required build dependency: ninja...")
    subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import ninja

import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154);print("TEST 170 — ENERGY-MATCHED ENVELOPE RELEASE ABLATION");print("CAUSAL_ORTH + FOUR-ZONE DEPTH REDISTRIBUTION UNDER FIXED RSS DOSE");print("="*154)
print("Ninja:",ninja.__version__)
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=170
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
PRIMARY_PROBES=8;SVD_REL_TOL=1e-5;PRIMARY_SCALE=.50
SCALES=[.125,.25,.50,.75,1.]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST170_ENERGY_MATCHED_ENVELOPE_RELEASE");ROOT.mkdir(parents=True,exist_ok=True)
ZONES=[range(0,5),range(5,10),range(10,15),range(15,20)]
print("GPU:",torch.cuda.get_device_name(0));print("Model:",MODEL_ID);print("Layers: L0-L19 | direction=CAUSAL_ORTH | zones=[0-4,5-9,10-14,15-19]")

# ======================================================================================================================
# LOCKED SEASC ENVELOPE / ENERGY BUDGET
# ======================================================================================================================
def env(L):
    x=ZIRVE*math.exp(-SONUM*float(L))*(1+SONUM*float(L))+TABAN
    return x/(ZIRVE+TABAN)
ENVELOPE=np.asarray([env(L) for L in range(N_LAYERS)],dtype=np.float64)
BASE_RHO=IVME*ENVELOPE;BASE_RSS=float(np.sqrt(np.sum(BASE_RHO**2)))
def zone_to_layer(z):
    z=np.asarray(z,dtype=np.float64);m=np.ones(N_LAYERS,dtype=np.float64)
    for j,ix in enumerate(ZONES):
        for L in ix:m[L]=z[j]
    raw=BASE_RHO*m;rss=float(np.sqrt(np.sum(raw**2)))
    if rss<EPS:raise RuntimeError("Degenerate envelope.")
    k=BASE_RSS/rss;m*=k;rho=BASE_RHO*m
    return m,rho,float(np.sqrt(np.sum(rho**2))),float(k)
LOCK_M,LOCK_RHO,_,_=zone_to_layer([1,1,1,1])
print(f"Locked RSS dose budget: {BASE_RSS:.9f}")

CANDIDATES=[
("LOCKED_SEASC",[1.00,1.00,1.00,1.00]),
("EARLY_SOFT",[0.70,0.90,1.10,1.20]),
("EARLY_SOFT_2",[0.50,0.80,1.15,1.30]),
("LATE_SOFT",[1.20,1.10,0.90,0.70]),
("MIDDLE_FOCUS",[0.75,1.20,1.20,0.75]),
("MID_LATE",[0.65,0.95,1.20,1.25]),
("LATE_FOCUS",[0.60,0.80,1.05,1.40]),
("EARLY_FOCUS",[1.40,1.05,0.80,0.60]),
("FLAT",[1/ENVELOPE[L] for L in [2,7,12,17]])]
ENV_BANK={}
for name,z in CANDIDATES:
    m,rho,rss,k=zone_to_layer(z);ENV_BANK[name]={"name":name,"zone_raw":[float(x) for x in z],"layer_multiplier":m,"rho":rho,"rss":rss,"renorm":k}
print("Envelope candidates:",list(ENV_BANK))

# ======================================================================================================================
# CUDA SEASC — TEST169 INJECTION PHYSICS UNCHANGED
# ======================================================================================================================
BUILD="/tmp/akbascore_test170";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST170 SEASC");}"""
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
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test170_seasc",[&]{k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/20] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test170_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
print("       OK")

# ======================================================================================================================
# MODEL
# ======================================================================================================================
print("\n[2/20] Loading frozen model...")
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
# TEST169 REFERENCE GEOMETRY
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

print("\n[3/20] Rebuilding TEST169 Fisher / Bank geometry...")
PB,NB=[[] for _ in range(N_LAYERS)],[[] for _ in range(N_LAYERS)]
for i,(p,n) in enumerate(SEARCH_PAIRS,1):
    hp,hn=capture(p),capture(n)
    for L in range(N_LAYERS):PB[L].append(hp[L]);NB[L].append(hn[L])
    print(f"       pair {i:02d}/16")
PB=[torch.stack(x).float() for x in PB];NB=[torch.stack(x).float() for x in NB]
FISHER,BANK,COS,RESN=[],[],[],[]
for L in range(N_LAYERS):
    Hp,Hn=PB[L],NB[L];mp,mn=Hp.mean(0),Hn.mean(0);md=mp-mn;R=torch.cat([Hp-mp,Hn-mn],0).float();nu=max(R.shape[0]-2,1)
    _,S,Vh=torch.linalg.svd(R,full_matrices=False);eig=S.square()/float(nu);pe=eig[eig>FISHER_EPS];es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS);V=Vh.T;pr=V.T@md;f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float()
    dirs=norm(Hp-Hn);dirs=dirs*torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]));b=norm(dirs.mean(0)).float();c=torch.dot(f,b).clamp(-1,1)
    if c<0:b,c=-b,-c
    r=b-c*f;FISHER.append(f);BANK.append(b);COS.append(float(c));RESN.append(float(r.norm()))
    print(f"       L{L:02d} cos(F,B)={float(c):+.6f} residual={float(r.norm()):.6f}")
FISHER=torch.stack(FISHER).contiguous();BANK=torch.stack(BANK).contiguous()
print(f"       mean cos={np.mean(COS):+.6f} | mean residual={np.mean(RESN):.6f}")

# ======================================================================================================================
# P8 RETENTION BASIS
# ======================================================================================================================
RETENTION=[
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

print("\n[4/20] Computing frozen P8 retention basis...")
PG=[]
for i,(ctx,cont) in enumerate(RETENTION,1):
    g,l=activation_gradient(ctx,cont);PG.append(g);print(f"       probe {i}/8 | NLL={l:.6f} | mean ||g||={np.mean([float(x.norm()) for x in g]):.6e}")
Q8=[];RANKS=[]
for L in range(N_LAYERS):
    G=torch.stack([PG[i][L] for i in range(PRIMARY_PROBES)]).float();G=G[G.norm(dim=1)>EPS];G=norm(G);_,s,Vh=torch.linalg.svd(G,full_matrices=False)
    tol=max(float(s[0])*SVD_REL_TOL,EPS);rank=int((s>tol).sum());Q8.append(Vh[:rank].T.contiguous());RANKS.append(rank)
print("       P8 ranks:",RANKS)

def guard_vectors(raw):
    out=[];tel=[]
    for L in range(N_LAYERS):
        u=raw[L].float();q=Q8[L];p=q@(q.T@u) if q.shape[1] else torch.zeros_like(u);ug=u-p;rn=float(u.norm());rem=float(ug.norm())
        if rem<EPS:raise RuntimeError(f"Guard annihilated L{L}")
        ug=ug/rem;tel.append({"layer":L,"overlap_energy":float(p.square().sum()/u.square().sum().clamp_min(EPS)),"retained_norm_fraction":rem/max(rn,EPS),"post_Q":float((q.T@ug).norm())});out.append(ug)
    return torch.stack(out).contiguous(),tel
FISHER_GUARD,FISHER_GT=guard_vectors(FISHER)

# ======================================================================================================================
# TEST169 CAUSAL_ORTH RECONSTRUCTION
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
FORGE_NEUTRAL=[
("A ceramic vase rests beside a folded cloth."," The objects remain stationary."),
("A closed box sits beside a plain notebook."," The objects remain stationary."),
("A metal key lies beside a paper card."," The objects remain stationary."),
("A wooden tray rests near an empty glass."," The objects remain stationary.")]

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

print("\n[5/20] Reconstructing TEST169 CAUSAL_ORTH...")
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
CAUSAL_GUARD,CGT=guard_vectors(CAUSAL);CAUSAL_ORTH=[];ORTH=[]
for L in range(N_LAYERS):
    f=FISHER_GUARD[L];g=CAUSAL_GUARD[L];r=g-torch.dot(g,f)*f;rn=r.norm()
    if rn<EPS:raise RuntimeError(f"CAUSAL_ORTH collapsed L{L}")
    CAUSAL_ORTH.append(r/rn);ORTH.append(float(rn))
CAUSAL_ORTH=torch.stack(CAUSAL_ORTH).contiguous()
print(f"       mean cos(Causal,Fisher)={np.mean([float(torch.dot(CAUSAL[L],FISHER[L])) for L in range(N_LAYERS)]):+.6f}")
print(f"       mean orthogonal component={np.mean(ORTH):.6f}")
print(f"       mean post-Q overlap={np.mean([float((Q8[L].T@CAUSAL_ORTH[L]).norm()) for L in range(N_LAYERS)]):.8e}")

# ======================================================================================================================
# ENVELOPE-AWARE SEASC
# ======================================================================================================================
def blank_tel():return [{"requested":[],"realized":[]} for _ in range(N_LAYERS)]
def hooks(vectors,sign,scale,tel,rho,state=None):
    hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(module,args,out):
                old=out[0] if isinstance(out,tuple) else out;rest=out[1:] if isinstance(out,tuple) else None;B=old.shape[0]
                direction=(float(sign)*vectors[li][None,:].expand(B,-1)).float().contiguous();req=float(rho[li])*float(scale);dose=torch.full((B,),req,device=DEVICE,dtype=torch.float32)
                oldlast=old[:,-1,:].float();new=seasc_ext.seasc_batch(old,direction,dose);newlast=new[:,-1,:].float()
                real=(newlast-oldlast).norm(dim=-1)/oldlast.norm(dim=-1).clamp_min(EPS)
                tel[li]["requested"].extend(dose.detach().cpu().tolist());tel[li]["realized"].extend(real.detach().cpu().tolist())
                if state is not None:state[li]=newlast.detach().float().cpu()
                return new if rest is None else (new,)+rest
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    return hs

@torch.inference_mode()
def logp(context,continuation,v=None,sign=0,scale=0,rho=None):
    ids,mask,plen=sequence(context,continuation);tel=blank_tel();hs=hooks(v,sign,scale,tel,rho) if v is not None else []
    try:o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];lp=F.log_softmax(z,-1);score=float(lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean())
    del ids,mask,o,z,tar,lp;return score,tel

def arm(name,v,scale,items,rho):
    cs,pm,mm,req,real=[],[],[],[],[]
    for x in items:
        pp,t1=logp(x["context"],x["pos"],v,+1,scale,rho);pn,t2=logp(x["context"],x["neg"],v,+1,scale,rho)
        np_,t3=logp(x["context"],x["pos"],v,-1,scale,rho);nn,t4=logp(x["context"],x["neg"],v,-1,scale,rho)
        a,b=pp-pn,np_-nn;cs.append(.5*(a-b));pm.append(a);mm.append(b)
        for t in (t1,t2,t3,t4):
            for L in range(N_LAYERS):req+=t[L]["requested"];real+=t[L]["realized"]
    return {"method":name,"scale":float(scale),"Csym":float(np.mean(cs)),"Csym_sd":float(np.std(cs,ddof=1)),"plus_margin":float(np.mean(pm)),"minus_margin":float(np.mean(mm)),"positive_items":int(np.sum(np.asarray(cs)>0)),"requested_mean":float(np.mean(req)),"realized_mean":float(np.mean(real)),"item_csym":[float(x) for x in cs]}

@torch.inference_mode()
def collateral_one(ctx,cont,v,scale,sign,rho):
    ids,mask,plen=sequence(ctx,cont);o0=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True);z0=o0.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];p0=F.log_softmax(z0,-1)
    n0=-p0.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();hs=hooks(v,sign,scale,blank_tel(),rho)
    try:o1=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    z1=o1.logits[:,plen-1:-1,:].float();p1=F.log_softmax(z1,-1);n1=-p1.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();kl=(p0.exp()*(p0-p1)).sum(-1).mean();ans=(float(n1-n0),float(kl))
    del ids,mask,o0,o1,z0,z1,tar,p0,p1,n0,n1,kl;return ans

def collateral(name,v,scale,probes,rho):
    z=[]
    for ctx,cont in probes:z+=[collateral_one(ctx,cont,v,scale,+1,rho),collateral_one(ctx,cont,v,scale,-1,rho)]
    return {"method":name,"scale":float(scale),"kl_sym":float(np.mean([x[1] for x in z])),"abs_delta_nll_sym":float(np.mean([abs(x[0]) for x in z]))}

def retention(name,v,scale,probes,rho):
    vals=[]
    for ctx,cont in probes:
        b,_=logp(ctx,cont,None,0,0,None)
        for s in (+1,-1):
            x,_=logp(ctx,cont,v,s,scale,rho);vals.append(float(-x+b))
    return {"method":name,"scale":float(scale),"retention_abs_delta_nll_sym":float(np.mean(np.abs(vals))),"per_sign_delta":[float(x) for x in vals]}

@torch.inference_mode()
def states(ctx,v=None,scale=0,sign=0,rho=None):
    e=tokenizer(chat(ctx),return_tensors="pt").to(DEVICE);cap=[None]*N_LAYERS;hs=[]
    if v is None:
        for L in range(N_LAYERS):
            def mk(li):
                def hk(m,a,o):cap[li]=(o[0] if isinstance(o,tuple) else o)[:,-1,:].detach().float().cpu()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
    else:hs=hooks(v,sign,scale,blank_tel(),rho,cap)
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return cap

def canal(v,scale,items,rho):
    dp,dt,tot=[],[],[]
    for x in items:
        h0=states(x["context"])
        for s in (+1,-1):
            h1=states(x["context"],v,scale,s,rho)
            for L in range(N_LAYERS):
                a=h0[L][0];d=h1[L][0]-a;t=float(s)*v[L].detach().cpu().float();al=torch.dot(d,t);par=al*t;per=d-par;den=a.norm().clamp_min(EPS)
                tot.append(float(d.norm()/den));dt.append(float(par.norm()/den));dp.append(float(per.norm()/den))
    return {"D_perp":float(np.mean(dp)),"D_target":float(np.mean(dt)),"D_total":float(np.mean(tot)),"off_target_fraction":float(np.mean(dp)/max(np.mean(tot),EPS))}

# ======================================================================================================================
# DEVELOPMENT ENVELOPE SEARCH
# ======================================================================================================================
print("\n[6/20] Energy-matched envelope search...")
DEV=[]
for name,e in ENV_BANK.items():
    fam=[arm(name,CAUSAL_ORTH,PRIMARY_SCALE,it,e["rho"])["Csym"] for it in FORGE.values()]
    c=collateral(name,CAUSAL_ORTH,PRIMARY_SCALE,FORGE_NEUTRAL,e["rho"]);r=retention(name,CAUSAL_ORTH,PRIMARY_SCALE,RETENTION,e["rho"]);h=canal(CAUSAL_ORTH,PRIMARY_SCALE,FORGE["A"][:2],e["rho"])
    row={"name":name,"C_mean":float(np.mean(fam)),"C_worst":float(np.min(fam)),"C_sd":float(np.std(fam,ddof=1)),"A":fam[0],"B":fam[1],"C":fam[2],"D":fam[3],"KL":c["kl_sym"],"Ret":r["retention_abs_delta_nll_sym"],"Dperp":h["D_perp"],"off":h["off_target_fraction"],"rss":e["rss"],"renorm":e["renorm"]};DEV.append(row)
    print(f"       {name:<14} C={row['C_mean']:+.6f} worst={row['C_worst']:+.6f} sd={row['C_sd']:.6f} KL={row['KL']:.6f} Ret={row['Ret']:.6f} Dp={row['Dperp']:.6f} RSS={row['rss']:.9f}")

print("\n[7/20] Viability / Pareto...")
LOCK=next(x for x in DEV if x["name"]=="LOCKED_SEASC")
VIABLE=[x for x in DEV if x["C_mean"]>=.90*LOCK["C_mean"] and x["C_worst"]>0]
SAFE=[x for x in VIABLE if x["KL"]<=1.10*LOCK["KL"] and x["Ret"]<=1.10*LOCK["Ret"] and x["Dperp"]<=1.10*LOCK["Dperp"] and (x["KL"]<LOCK["KL"] or x["Ret"]<LOCK["Ret"] or x["Dperp"]<LOCK["Dperp"])]
def dominates(a,b):
    nw=a["C_mean"]>=b["C_mean"] and a["C_worst"]>=b["C_worst"] and a["KL"]<=b["KL"] and a["Ret"]<=b["Ret"] and a["Dperp"]<=b["Dperp"] and a["C_sd"]<=b["C_sd"]
    st=a["C_mean"]>b["C_mean"] or a["C_worst"]>b["C_worst"] or a["KL"]<b["KL"] or a["Ret"]<b["Ret"] or a["Dperp"]<b["Dperp"] or a["C_sd"]<b["C_sd"];return nw and st
POOL=SAFE if SAFE else VIABLE
PARETO=[x for x in POOL if not any(dominates(y,x) for y in POOL if y is not x)]
print("       locked C:",f"{LOCK['C_mean']:+.6f}","| viable:",len(VIABLE),"| safe:",len(SAFE),"| Pareto:",[x["name"] for x in PARETO])

# ======================================================================================================================
# LOFO
# ======================================================================================================================
print("\n[8/20] LOFO envelope stability...")
LOFO=[];candidate_names=[x["name"] for x in (SAFE if SAFE else VIABLE)]
if not candidate_names:candidate_names=["LOCKED_SEASC"]
for held in FORGE:
    rows=[]
    for name in candidate_names:
        e=ENV_BANK[name];vals=[arm(name,CAUSAL_ORTH,PRIMARY_SCALE,it,e["rho"])["Csym"] for fn,it in FORGE.items() if fn!=held]
        rows.append({"name":name,"C":float(np.mean(vals)),"worst":float(np.min(vals)),"sd":float(np.std(vals,ddof=1))})
    chosen=max(rows,key=lambda x:(x["worst"],x["C"],-x["sd"]));LOFO.append({"held":held,**chosen});print(f"       hold {held} -> {chosen['name']:<14} C={chosen['C']:+.6f} worst={chosen['worst']:+.6f}")

# ======================================================================================================================
# MICRO ROBUSTNESS
# ======================================================================================================================
print("\n[9/20] Micro envelope robustness...")
MICRO=[];micro_source=SAFE if SAFE else VIABLE
if not micro_source:micro_source=[LOCK]
for row in micro_source:
    name=row["name"];base=np.asarray(ENV_BANK[name]["zone_raw"],dtype=np.float64);vals=[]
    for q in (.90,1.00,1.10):
        z=1.+q*(base-1.);m,rho,rss,k=zone_to_layer(z);fam=[arm("micro",CAUSAL_ORTH,PRIMARY_SCALE,it,rho)["Csym"] for it in FORGE.values()];vals.append(float(np.mean(fam)))
    sd=float(np.std(vals,ddof=1));curv=float(vals[0]-2*vals[1]+vals[2]);MICRO.append({**row,"micro_minus":vals[0],"micro_center":vals[1],"micro_plus":vals[2],"micro_sd":sd,"curvature":curv})
    print(f"       {name:<14} micro={vals[0]:+.5f}/{vals[1]:+.5f}/{vals[2]:+.5f} sd={sd:.6f} curv={curv:+.6f}")

# ======================================================================================================================
# FREEZE
# ======================================================================================================================
print("\n[10/20] FREEZE...")
if SAFE:
    MP=[x for x in MICRO if x["name"] in [y["name"] for y in SAFE]]
    WIN=min(MP,key=lambda x:(-x["C_worst"],x["micro_sd"],x["KL"]/max(LOCK["KL"],EPS)+x["Ret"]/max(LOCK["Ret"],EPS)+x["Dperp"]/max(LOCK["Dperp"],EPS),-x["C_mean"],x["C_sd"]))
    FROZEN_NAME=WIN["name"];FALLBACK=False
else:FROZEN_NAME="LOCKED_SEASC";WIN=LOCK;FALLBACK=True
FROZEN=ENV_BANK[FROZEN_NAME]
print("       FROZEN envelope:",FROZEN_NAME);print("       raw zones:",FROZEN["zone_raw"]);print("       normalized layer multipliers:",[round(float(FROZEN["layer_multiplier"][L]),4) for L in (0,4,5,9,10,14,15,19)])
print(f"       RSS={FROZEN['rss']:.9f} | baseline RSS={BASE_RSS:.9f} | fallback={FALLBACK}");print("       FINAL remains unopened.")

# ======================================================================================================================
# FINAL — FIRST ACCESS ONLY AFTER FREEZE
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

print("\n[11/20] FINAL — FIRST ACCESS AFTER FREEZE...")
ARMS={"LOCKED_SEASC":LOCK_RHO,"RELEASED":FROZEN["rho"]}
FINAL_CAUSAL=[];FINAL_COLL=[];FINAL_RET=[]
for scale in SCALES:
    print(f"\n       DOSE={scale:.3f}")
    for name,rho in ARMS.items():
        a=arm(name,CAUSAL_ORTH,scale,FINAL,rho);c=collateral(name,CAUSAL_ORTH,scale,FINAL_NEUTRAL,rho);r=retention(name,CAUSAL_ORTH,scale,RET_FINAL,rho)
        FINAL_CAUSAL.append(a);FINAL_COLL.append(c);FINAL_RET.append(r)
        print(f"       {name:<14} C={a['Csym']:+.6f} P+={a['positive_items']:02d}/16 KL={c['kl_sym']:.6f} HeldRet={r['retention_abs_delta_nll_sym']:.6f}")

print("\n[12/20] FINAL hidden-state canalization...")
FINAL_HIDDEN=[]
for scale in SCALES:
    for name,rho in ARMS.items():
        h=canal(CAUSAL_ORTH,scale,FINAL[:4],rho);FINAL_HIDDEN.append({"method":name,"scale":scale,**h})
        print(f"       {name:<14} s={scale:.3f} Dperp={h['D_perp']:.6f} Dtarget={h['D_target']:.6f} off={h['off_target_fraction']:.4f}")

# ======================================================================================================================
# BOOTSTRAP
# ======================================================================================================================
print("\n[13/20] Paired causal bootstrap...")
def get(rows,name,scale):return next(x for x in rows if x["method"]==name and abs(x["scale"]-scale)<1e-9)
def boot(a,b,n=20000,seed=170):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):
        ix=rng.integers(0,len(d),len(d));z[i]=d[ix].mean()
    return {"delta":float(d.mean()),"ci_low":float(np.quantile(z,.025)),"ci_high":float(np.quantile(z,.975))}
BOOT=[]
for scale in SCALES:
    a=get(FINAL_CAUSAL,"RELEASED",scale);b=get(FINAL_CAUSAL,"LOCKED_SEASC",scale);z=boot(a["item_csym"],b["item_csym"],seed=SEED+int(scale*1000));BOOT.append({"scale":scale,**z})
    print(f"       s={scale:.3f} RELEASED-LOCKED ΔC={z['delta']:+.6f} [{z['ci_low']:+.6f},{z['ci_high']:+.6f}]")

print("\n[14/20] Primary mechanism @0.50...")
PRIMARY=[]
for name in ARMS:
    a=get(FINAL_CAUSAL,name,.5);c=get(FINAL_COLL,name,.5);r=get(FINAL_RET,name,.5);h=get(FINAL_HIDDEN,name,.5)
    x={"method":name,"C":a["Csym"],"KL":c["kl_sym"],"HeldRet":r["retention_abs_delta_nll_sym"],"Dperp":h["D_perp"],"Dtarget":h["D_target"],"off":h["off_target_fraction"],"P":a["positive_items"]};PRIMARY.append(x)
    print(f"       {name:<14} C={x['C']:+.6f} KL={x['KL']:.6f} HeldRet={x['HeldRet']:.6f} Dperp={x['Dperp']:.6f} off={x['off']:.4f} P+={x['P']}/16")

# ======================================================================================================================
# DOSE ACCOUNTING
# ======================================================================================================================
print("\n[15/20] Dose-budget verification...")
DOSE_ROWS=[]
for name,rho in ARMS.items():
    rss=float(np.sqrt(np.sum(np.asarray(rho)**2)));sm=float(np.sum(rho));mx=float(np.max(rho));DOSE_ROWS.append({"method":name,"rss":rss,"sum":sm,"max":mx})
    print(f"       {name:<14} RSS={rss:.9f} SUM={sm:.9f} MAX={mx:.9f}")
print(f"       RSS absolute difference={abs(DOSE_ROWS[0]['rss']-DOSE_ROWS[1]['rss']):.12e}")

# ======================================================================================================================
# LAYER PROFILE
# ======================================================================================================================
print("\n[16/20] Frozen layer profile...")
PROFILE=[]
for L in range(N_LAYERS):
    x={"layer":L,"base_E":float(ENVELOPE[L]),"locked_rho":float(LOCK_RHO[L]),"released_multiplier":float(FROZEN["layer_multiplier"][L]),"released_rho":float(FROZEN["rho"][L]),"ratio":float(FROZEN["rho"][L]/LOCK_RHO[L])}
    PROFILE.append(x);print(f"       L{L:02d} locked={x['locked_rho']:.6f} m={x['released_multiplier']:.4f} released={x['released_rho']:.6f}")

# ======================================================================================================================
# DECISION
# ======================================================================================================================
print("\n[17/20] TEST170 decision...")
A=next(x for x in PRIMARY if x["method"]=="RELEASED");B=next(x for x in PRIMARY if x["method"]=="LOCKED_SEASC")
causal_preserved=A["C"]>=.90*B["C"];kl_improved=A["KL"]<B["KL"];ret_improved=A["HeldRet"]<B["HeldRet"];dp_improved=A["Dperp"]<B["Dperp"]
if FROZEN_NAME=="LOCKED_SEASC":DECISION="NO_SAFE_ENVELOPE_RELEASE_FOUND"
elif causal_preserved and kl_improved and ret_improved and dp_improved:DECISION="ENVELOPE_CONSTRAINT_MECHANISM_SUPPORTED"
elif causal_preserved and sum([kl_improved,ret_improved,dp_improved])>=2:DECISION="PARTIAL_ENVELOPE_RELEASE_SIGNAL"
elif causal_preserved and sum([kl_improved,ret_improved,dp_improved])>=1:DECISION="WEAK_ENVELOPE_RELEASE_SIGNAL"
else:DECISION="ENVELOPE_RELEASE_NOT_SUPPORTED"
print("       Frozen:",FROZEN_NAME);print("       causal >=90%:",causal_preserved);print("       held-out KL improved:",kl_improved);print("       held-out retention improved:",ret_improved);print("       Dperp improved:",dp_improved);print("       DECISION:",DECISION)

# ======================================================================================================================
# SAVE
# ======================================================================================================================
print("\n[18/20] Saving...")
def scalar(rows):return [{k:v for k,v in r.items() if not isinstance(v,(list,tuple,dict,np.ndarray))} for r in rows]
def savecsv(name,rows):
    rows=scalar(rows)
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

savecsv("DEV_ENVELOPES.csv",DEV);savecsv("LOFO.csv",LOFO);savecsv("MICRO.csv",MICRO);savecsv("FINAL_CAUSAL.csv",FINAL_CAUSAL);savecsv("FINAL_COLLATERAL.csv",FINAL_COLL);savecsv("FINAL_RETENTION_HELDOUT.csv",FINAL_RET);savecsv("FINAL_HIDDEN.csv",FINAL_HIDDEN);savecsv("BOOTSTRAP.csv",BOOT);savecsv("PRIMARY.csv",PRIMARY);savecsv("DOSE_BUDGET.csv",DOSE_ROWS);savecsv("FROZEN_PROFILE.csv",PROFILE)

np.savez_compressed(ROOT/"TEST170_ENVELOPE_RELEASE.npz",
causal_orth=CAUSAL_ORTH.detach().cpu().numpy(),
fisher=FISHER.detach().cpu().numpy(),
fisher_guard=FISHER_GUARD.detach().cpu().numpy(),
locked_rho=np.asarray(LOCK_RHO,dtype=np.float32),
released_rho=np.asarray(FROZEN["rho"],dtype=np.float32),
released_multiplier=np.asarray(FROZEN["layer_multiplier"],dtype=np.float32))

REPORT={
"test":"TEST 170",
"title":"ENERGY-MATCHED ENVELOPE RELEASE ABLATION",
"model":MODEL_ID,
"seed":SEED,
"base_rss":BASE_RSS,
"frozen_envelope":FROZEN_NAME,
"fallback":FALLBACK,
"decision":DECISION,
"dev":DEV,
"lofo":LOFO,
"micro":MICRO,
"final_causal":FINAL_CAUSAL,
"final_collateral":FINAL_COLL,
"final_retention_heldout":FINAL_RET,
"final_hidden":FINAL_HIDDEN,
"bootstrap":BOOT,
"primary":PRIMARY,
"profile":PROFILE,
"contract":[
"TEST169 CAUSAL_ORTH construction reproduced without FINAL access.",
"Only layerwise dose distribution is released.",
"SEASC frozen-norm CUDA injection physics is unchanged.",
"All candidate envelopes have exactly matched RSS dose budget.",
"Four-zone envelope family is fixed before FINAL.",
"Envelope is frozen before FINAL.",
"FINAL causal, neutral and held-out retention sets do not select the envelope.",
"RET_FINAL is distinct from P8 guard-building probes.",
"No runtime controller, gate, brake, cap, saturation, boost or FINAL adaptation.",
"A positive result means depth allocation contributed to the trade-off; it does not prove the locked envelope was the only cause."
]}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)

print("\n[19/20] Integrity...")
print("       P8 ranks:",RANKS);print("       Direction changed during envelope search: NO");print("       Frozen envelope:",FROZEN_NAME)
print(f"       RSS match error: {abs(float(np.sqrt(np.sum(FROZEN['rho']**2)))-BASE_RSS):.12e}")
print("       FINAL selection access: NONE");print("       Model weights modified: NO");print("       Runtime controller: OFF")

print("\n[20/20] COMPLETE")
print("="*154);print("TEST 170 — ENERGY-MATCHED ENVELOPE RELEASE ABLATION — SUMMARY");print("="*154)
print(f"Mean cos(Fisher,Bank): {np.mean(COS):+.6f}");print(f"Mean CAUSAL_ORTH retained component: {np.mean(ORTH):.6f}")
print("Frozen envelope:",FROZEN_NAME);print("Raw zones:",FROZEN["zone_raw"]);print(f"Locked RSS : {BASE_RSS:.9f}");print(f"Release RSS: {float(np.sqrt(np.sum(FROZEN['rho']**2))):.9f}")
print("DECISION:",DECISION);print("Saved:",ROOT);print("="*154)
