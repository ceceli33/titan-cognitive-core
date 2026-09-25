# ======================================================================================================================
# TEST 172 — COLLATERAL COMPONENT TRANSPORT X-RAY
# AkbasCore 3.2 / SEASC
# CAUSAL_ORTH vs CAUSAL_SAFE(K=8) vs REMOVED_COMPONENT
#
# PURPOSE:
# TEST171 showed that removing ~1% collateral-sensitive vector energy reduced held-out collateral while preserving /
# increasing causal effect. TEST172 asks WHERE that small removed component propagates, rotates and amplifies downstream.
#
# LOCKS:
# - Qwen/Qwen2.5-7B-Instruct BF16 / SDPA / A100
# - L0-L19
# - TEST169 CAUSAL_ORTH reconstruction unchanged
# - TEST171 collateral basis construction unchanged
# - TEST171 K=8 is PREDECLARED / FROZEN; NO rank search
# - Locked SEASC envelope / same RSS dose
# - No FINAL171 reuse for selection
# - New X-RAY prompts only
# - No runtime controller / gate / brake / cap / boost
# - Model weights unchanged
# ======================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,gc,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:
    print("Installing required build dependency: ninja...")
    subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import ninja
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154);print("TEST 172 — COLLATERAL COMPONENT TRANSPORT X-RAY");print("CAUSAL_ORTH vs CAUSAL_SAFE(K=8) vs REMOVED_COMPONENT");print("="*154)
print("Ninja:",ninja.__version__)
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=172
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
P8_COUNT=8;SVD_REL_TOL=1e-5;FROZEN_K=8;PRIMARY_SCALE=.50
SCALES=[.125,.25,.50,.75,1.]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST172_COLLATERAL_COMPONENT_TRANSPORT_XRAY");ROOT.mkdir(parents=True,exist_ok=True)
print("GPU:",torch.cuda.get_device_name(0));print("Model:",MODEL_ID);print("Layers: L0-L19 | frozen K=8 | X-ray primary dose=0.50")

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
# CUDA SEASC — TEST171 PHYSICS UNCHANGED
# ======================================================================================================================
BUILD="/tmp/akbascore_test172";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST172 SEASC");}"""
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
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test172_seasc",[&]{k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/22] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test172_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
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
# TEST169/171 REFERENCE GEOMETRY
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
# P8 RETENTION BASIS — TEST171 LINEAGE LOCK
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
        u=raw[L].float();q=bases[L];p=q@(q.T@u) if q.shape[1] else torch.zeros_like(u);v=u-p;rn=float(u.norm());rem=float(v.norm())
        if rem<EPS:raise RuntimeError(f"Projection annihilated L{L}")
        v=v/rem;tel.append({"layer":L,"overlap_energy":float(p.square().sum()/u.square().sum().clamp_min(EPS)),"retained_norm_fraction":rem/max(rn,EPS),"post_Q":float((q.T@v).norm())});out.append(v)
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
print(f"       mean P8 overlap={np.mean([float((Q8[L].T@CAUSAL_ORTH[L]).norm()) for L in range(N_LAYERS)]):.8e}")

# ======================================================================================================================
# TEST171 COLLATERAL BASIS — EXACT BUILD SET / WEIGHTED ACTUATOR GRADIENT
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

print("\n[6/22] Rebuilding TEST171 collateral basis...")
COLL_G=[]
for i,(ctx,cont) in enumerate(COLL_BUILD,1):
    g,l=actuator_gradient_weighted(ctx,cont);COLL_G.append(g)
    print(f"       neutral {i}/8 | NLL={l:.6f} | mean ||g_act||={np.mean([float(x.norm()) for x in g]):.6e}")
QCOLL=[]
for L in range(N_LAYERS):
    G=torch.stack([COLL_G[i][L] for i in range(8)]).float();G=G[G.norm(dim=1)>EPS];G=norm(G);_,s,Vh=torch.linalg.svd(G,full_matrices=False);QCOLL.append(Vh[:FROZEN_K].T.contiguous())
print("       Frozen collateral ranks:",[q.shape[1] for q in QCOLL])

# ======================================================================================================================
# REBUILD TEST171 CAUSAL_SAFE + EXPLICIT REMOVED COMPONENT
# IMPORTANT:
# SAFE reconstruction exactly follows TEST171 order.
# REMOVED_COMPONENT = normalized difference CAUSAL_ORTH - CAUSAL_SAFE, sign-aligned layerwise.
# RAW_COLL_COMPONENT = direct Qcoll projection of CAUSAL_ORTH, separately retained for mechanism comparison.
# ======================================================================================================================
print("\n[7/22] Reconstructing TEST171 CAUSAL_SAFE and removed component...")
SAFE=[];RAW_COLL=[];REMOVED=[];GEOM=[]
for L in range(N_LAYERS):
    u=CAUSAL_ORTH[L].float();qc=QCOLL[L];pc=qc@(qc.T@u);v=u-pc
    q=Q8[L];v=v-q@(q.T@v);f=FISHER_GUARD[L];v=v-torch.dot(v,f)*f;v=v-q@(q.T@v);v=norm(v)
    raw=norm(pc) if pc.norm()>EPS else torch.zeros_like(pc)
    d=u-v
    if d.norm()>EPS:d=norm(d)
    else:d=torch.zeros_like(d)
    if d.norm()>EPS and pc.norm()>EPS and torch.dot(d,pc)<0:d=-d
    SAFE.append(v);RAW_COLL.append(raw);REMOVED.append(d)
    GEOM.append({"layer":L,"raw_coll_energy":float(pc.square().sum()/u.square().sum().clamp_min(EPS)),"safe_cos_orth":float(torch.dot(v,u)),"removed_norm_raw":float((u-v).norm()),"removed_vs_rawcoll":float(torch.dot(d,raw)) if d.norm()>EPS and raw.norm()>EPS else 0.,"safe_collQ":float((qc.T@v).norm()),"safe_P8":float((q.T@v).norm()),"safe_fisher":float(torch.dot(v,f))})
SAFE=torch.stack(SAFE).contiguous();RAW_COLL=torch.stack(RAW_COLL).contiguous();REMOVED=torch.stack(REMOVED).contiguous()
print(f"       mean raw collateral energy={np.mean([x['raw_coll_energy'] for x in GEOM]):.6f}")
print(f"       mean cos(SAFE,ORTH)={np.mean([x['safe_cos_orth'] for x in GEOM]):.6f}")
print(f"       mean ||ORTH-SAFE||={np.mean([x['removed_norm_raw'] for x in GEOM]):.6f}")
print(f"       mean cos(REMOVED,RAW_COLL)={np.mean([x['removed_vs_rawcoll'] for x in GEOM]):+.6f}")
print(f"       mean SAFE CollQ={np.mean([x['safe_collQ'] for x in GEOM]):.8e}")

# ======================================================================================================================
# NEW X-RAY SET — NO TEST171 FINAL REUSE
# ======================================================================================================================
XRAY_CAUSAL=[
{"context":"A survey drone and a mountain road are visible, with no vertical relation specified.","pos":" The survey drone is above the mountain road.","neg":" The survey drone is below the mountain road."},
{"context":"A suspended camera and an assembly table are visible, with no vertical relation specified.","pos":" The suspended camera is above the assembly table.","neg":" The suspended camera is below the assembly table."},
{"context":"A weather balloon and a railway bridge are visible, with no vertical relation specified.","pos":" The weather balloon is above the railway bridge.","neg":" The weather balloon is below the railway bridge."},
{"context":"A hanging sensor and a laboratory desk are visible, with no vertical relation specified.","pos":" The hanging sensor is above the laboratory desk.","neg":" The hanging sensor is below the laboratory desk."},
{"context":"A bird and a stone path are visible, with no vertical relation specified.","pos":" The bird is above the stone path.","neg":" The bird is below the stone path."},
{"context":"A warning beacon and a loading platform are visible, with no vertical relation specified.","pos":" The warning beacon is above the loading platform.","neg":" The warning beacon is below the loading platform."},
{"context":"A glider and a valley road are visible, with no vertical relation specified.","pos":" The glider is above the valley road.","neg":" The glider is below the valley road."},
{"context":"A hanging screen and a conference table are visible, with no vertical relation specified.","pos":" The hanging screen is above the conference table.","neg":" The hanging screen is below the conference table."}]

XRAY_NEUTRAL=[
("A ceramic bottle stands beside a folded sheet."," The objects remain stationary."),
("A wooden ruler lies beside a sealed notebook."," The objects remain stationary."),
("A glass cup rests next to an empty tray."," The objects remain stationary."),
("A metal brush lies beside a cardboard folder."," The objects remain stationary."),
("A fabric bag rests next to a plastic container."," The objects remain stationary."),
("A plain plate sits beside a closed book."," The objects remain stationary."),
("A small box stands beside a clean towel."," The objects remain stationary."),
("A steel spoon rests beside an empty bowl."," The objects remain stationary.")]

# ======================================================================================================================
# SEASC HOOKS + FULL-LAYER STATE CAPTURE
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

@torch.inference_mode()
def logp(context,continuation,v=None,sign=0,scale=0):
    ids,mask,plen=sequence(context,continuation);tel=blank_tel();hs=hooks(v,sign,scale,tel) if v is not None else []
    try:o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];lp=F.log_softmax(z,-1);score=float(lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean())
    del ids,mask,o,z,tar,lp;return score

def causal_arm(name,v,scale,items):
    cs=[]
    for x in items:
        pp=logp(x["context"],x["pos"],v,+1,scale);pn=logp(x["context"],x["neg"],v,+1,scale)
        np_=logp(x["context"],x["pos"],v,-1,scale);nn=logp(x["context"],x["neg"],v,-1,scale)
        cs.append(.5*((pp-pn)-(np_-nn)))
    return {"method":name,"scale":scale,"Csym":float(np.mean(cs)),"sd":float(np.std(cs,ddof=1)),"positive":int(np.sum(np.asarray(cs)>0)),"items":[float(x) for x in cs]}

@torch.inference_mode()
def neutral_one(ctx,cont,v,scale,sign):
    ids,mask,plen=sequence(ctx,cont);o0=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True);z0=o0.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];p0=F.log_softmax(z0,-1)
    n0=-p0.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();hs=hooks(v,sign,scale,blank_tel())
    try:o1=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    z1=o1.logits[:,plen-1:-1,:].float();p1=F.log_softmax(z1,-1);n1=-p1.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();kl=(p0.exp()*(p0-p1)).sum(-1).mean()
    ans=(float(n1-n0),float(kl));del ids,mask,o0,o1,z0,z1,tar,p0,p1,n0,n1,kl;return ans

def neutral_assay(name,v,scale):
    z=[]
    for ctx,cont in XRAY_NEUTRAL:z+=[neutral_one(ctx,cont,v,scale,+1),neutral_one(ctx,cont,v,scale,-1)]
    return {"method":name,"scale":scale,"KL":float(np.mean([x[1] for x in z])),"abs_dNLL":float(np.mean([abs(x[0]) for x in z]))}

# ======================================================================================================================
# LAYERWISE TRANSPORT X-RAY
# delta_L = steered hidden after layer L - baseline hidden after layer L
# Measures:
# rel       = ||delta|| / ||baseline||
# target    = projection on injected vector at that layer / ||baseline||
# perp      = orthogonal displacement / ||baseline||
# cos_act   = cos(delta, actuator_L)
# angle     = arccos(cos_act)
# collQ     = fraction of delta norm lying in TEST171 collateral subspace
# P8        = fraction lying in retention subspace
# growth    = rel_L / rel_(L-1)
# ======================================================================================================================
def xray_vector(name,v,scale,prompts):
    rows=[]
    for pi,x in enumerate(prompts):
        ctx=x["context"];h0=states(ctx)
        for sign in (+1,-1):
            h1=states(ctx,v,scale,sign);prev=None
            for L in range(N_LAYERS):
                a=h0[L][0].float();b=h1[L][0].float();d=b-a;dn=d.norm().clamp_min(EPS);bn=a.norm().clamp_min(EPS);t=float(sign)*v[L].detach().cpu().float()
                al=torch.dot(d,t);par=al*t;per=d-par;cos=float(torch.dot(d,t)/(dn*t.norm().clamp_min(EPS)));cos=max(-1.,min(1.,cos))
                qc=QCOLL[L].detach().cpu().float();qr=Q8[L].detach().cpu().float()
                coll=float((qc.T@d).norm()/dn) if qc.shape[1] else 0.;ret=float((qr.T@d).norm()/dn) if qr.shape[1] else 0.
                rel=float(dn/bn);growth=float(rel/max(prev,EPS)) if prev is not None else float("nan");prev=rel
                rows.append({"method":name,"prompt":pi,"sign":sign,"layer":L,"rel":rel,"target":float(par.norm()/bn),"perp":float(per.norm()/bn),"cos_act":cos,"angle_deg":float(math.degrees(math.acos(cos))),"collQ_frac":coll,"P8_frac":ret,"growth":growth})
    return rows

print("\n[8/22] Primary layerwise transport X-ray @0.50...")
XRAY_ROWS=[]
for name,v in [("CAUSAL_ORTH",CAUSAL_ORTH),("CAUSAL_SAFE",SAFE),("REMOVED_COMPONENT",REMOVED),("RAW_COLL_COMPONENT",RAW_COLL)]:
    r=xray_vector(name,v,PRIMARY_SCALE,XRAY_CAUSAL);XRAY_ROWS+=r
    print(f"       {name:<18} rows={len(r)}")

# ======================================================================================================================
# LAYER AGGREGATION
# ======================================================================================================================
print("\n[9/22] Aggregating transport by layer...")
LAYER_SUM=[]
for name in ["CAUSAL_ORTH","CAUSAL_SAFE","REMOVED_COMPONENT","RAW_COLL_COMPONENT"]:
    for L in range(N_LAYERS):
        rr=[x for x in XRAY_ROWS if x["method"]==name and x["layer"]==L]
        row={"method":name,"layer":L}
        for k in ["rel","target","perp","cos_act","angle_deg","collQ_frac","P8_frac"]:
            row[k]=float(np.mean([x[k] for x in rr]))
        gg=[x["growth"] for x in rr if np.isfinite(x["growth"])];row["growth"]=float(np.mean(gg)) if gg else float("nan")
        LAYER_SUM.append(row)
        if name in ("CAUSAL_ORTH","CAUSAL_SAFE"):
            print(f"       {name:<12} L{L:02d} rel={row['rel']:.6f} target={row['target']:.6f} perp={row['perp']:.6f} cos={row['cos_act']:+.4f} CollQ={row['collQ_frac']:.4f}")

# ======================================================================================================================
# ORTH vs SAFE DELTA — WHERE DOES CLEANING CHANGE DOWNSTREAM TRAJECTORY?
# ======================================================================================================================
print("\n[10/22] ORTH vs SAFE trajectory divergence...")
DIFF_ROWS=[]
for pi,x in enumerate(XRAY_CAUSAL):
    ctx=x["context"];h0=states(ctx)
    for sign in (+1,-1):
        ho=states(ctx,CAUSAL_ORTH,PRIMARY_SCALE,sign);hs=states(ctx,SAFE,PRIMARY_SCALE,sign)
        for L in range(N_LAYERS):
            base=h0[L][0].float();do=ho[L][0].float()-base;ds=hs[L][0].float()-base;dd=do-ds;den=base.norm().clamp_min(EPS)
            qc=QCOLL[L].detach().cpu().float();dn=dd.norm().clamp_min(EPS)
            DIFF_ROWS.append({"prompt":pi,"sign":sign,"layer":L,"trajectory_gap":float(dd.norm()/den),"gap_collQ_fraction":float((qc.T@dd).norm()/dn),"cos_delta_orth_safe":float(torch.dot(do,ds)/(do.norm().clamp_min(EPS)*ds.norm().clamp_min(EPS)))})
DIFF_SUM=[]
for L in range(N_LAYERS):
    rr=[x for x in DIFF_ROWS if x["layer"]==L]
    row={"layer":L,"trajectory_gap":float(np.mean([x["trajectory_gap"] for x in rr])),"gap_collQ_fraction":float(np.mean([x["gap_collQ_fraction"] for x in rr])),"cos_delta_orth_safe":float(np.mean([x["cos_delta_orth_safe"] for x in rr]))};DIFF_SUM.append(row)
    print(f"       L{L:02d} gap={row['trajectory_gap']:.6f} gapCollQ={row['gap_collQ_fraction']:.4f} cos(ΔO,ΔS)={row['cos_delta_orth_safe']:+.6f}")

# ======================================================================================================================
# REMOVED COMPONENT LAYER FINGERPRINT
# ======================================================================================================================
print("\n[11/22] Removed-component fingerprint...")
REMOVED_LAYER=[]
for L in range(N_LAYERS):
    g=GEOM[L];r=next(x for x in LAYER_SUM if x["method"]=="REMOVED_COMPONENT" and x["layer"]==L);raw=next(x for x in LAYER_SUM if x["method"]=="RAW_COLL_COMPONENT" and x["layer"]==L)
    row={"layer":L,"removed_vector_norm":g["removed_norm_raw"],"raw_coll_energy":g["raw_coll_energy"],"removed_vs_rawcoll":g["removed_vs_rawcoll"],"removed_rel":r["rel"],"removed_perp":r["perp"],"removed_cos_act":r["cos_act"],"removed_collQ_frac":r["collQ_frac"],"rawcoll_rel":raw["rel"],"trajectory_gap":DIFF_SUM[L]["trajectory_gap"]};REMOVED_LAYER.append(row)
    print(f"       L{L:02d} vecΔ={row['removed_vector_norm']:.6f} energy={row['raw_coll_energy']:.6f} transport={row['removed_rel']:.6f} gap={row['trajectory_gap']:.6f} CollQ={row['removed_collQ_frac']:.4f}")

# ======================================================================================================================
# CAUSAL / NEUTRAL BEHAVIOR OF REMOVED COMPONENT
# If removed component itself has high collateral but weak desired causal effect, mechanism is directly supported.
# ======================================================================================================================
print("\n[12/22] Behavioral assay of isolated components...")
BEHAVIOR=[]
for scale in SCALES:
    print(f"\n       DOSE={scale:.3f}")
    for name,v in [("CAUSAL_ORTH",CAUSAL_ORTH),("CAUSAL_SAFE",SAFE),("REMOVED_COMPONENT",REMOVED),("RAW_COLL_COMPONENT",RAW_COLL)]:
        a=causal_arm(name,v,scale,XRAY_CAUSAL);c=neutral_assay(name,v,scale);BEHAVIOR.append({**a,"KL":c["KL"],"abs_dNLL":c["abs_dNLL"]})
        print(f"       {name:<18} C={a['Csym']:+.6f} P+={a['positive']}/8 KL={c['KL']:.6f} |dNLL|={c['abs_dNLL']:.6f}")

# ======================================================================================================================
# DOSE RESPONSE X-RAY — ORTH / SAFE ONLY
# ======================================================================================================================
print("\n[13/22] Dose-response transport summary...")
DOSE_XRAY=[]
for scale in SCALES:
    for name,v in [("CAUSAL_ORTH",CAUSAL_ORTH),("CAUSAL_SAFE",SAFE)]:
        rr=xray_vector(name,v,scale,XRAY_CAUSAL[:4])
        for L in range(N_LAYERS):
            z=[x for x in rr if x["layer"]==L]
            DOSE_XRAY.append({"method":name,"scale":scale,"layer":L,"rel":float(np.mean([x["rel"] for x in z])),"perp":float(np.mean([x["perp"] for x in z])),"target":float(np.mean([x["target"] for x in z])),"collQ_frac":float(np.mean([x["collQ_frac"] for x in z]))})
    print(f"       scale={scale:.3f} complete")

# ======================================================================================================================
# GROWTH / ROTATION HOTSPOTS
# ======================================================================================================================
print("\n[14/22] Growth / rotation hotspots...")
HOTSPOTS=[]
for name in ["CAUSAL_ORTH","CAUSAL_SAFE","REMOVED_COMPONENT"]:
    z=[x for x in LAYER_SUM if x["method"]==name]
    rel=np.asarray([x["rel"] for x in z]);per=np.asarray([x["perp"] for x in z]);ang=np.asarray([x["angle_deg"] for x in z])
    growth=np.asarray([np.nan if not np.isfinite(x["growth"]) else x["growth"] for x in z])
    peak_rel=int(np.argmax(rel));peak_perp=int(np.argmax(per));peak_angle=int(np.argmax(ang));valid=np.where(np.isfinite(growth))[0];peak_growth=int(valid[np.argmax(growth[valid])]) if len(valid) else -1
    row={"method":name,"peak_rel_layer":peak_rel,"peak_rel":float(rel[peak_rel]),"peak_perp_layer":peak_perp,"peak_perp":float(per[peak_perp]),"peak_angle_layer":peak_angle,"peak_angle":float(ang[peak_angle]),"peak_growth_layer":peak_growth,"peak_growth":float(growth[peak_growth]) if peak_growth>=0 else float("nan")};HOTSPOTS.append(row)
    print(f"       {name:<18} peakRel=L{peak_rel:02d} {row['peak_rel']:.6f} | peakPerp=L{peak_perp:02d} {row['peak_perp']:.6f} | peakAngle=L{peak_angle:02d} {row['peak_angle']:.2f}° | peakGrowth=L{peak_growth:02d} {row['peak_growth']:.4f}")

# ======================================================================================================================
# TRAJECTORY GAP HOTSPOT
# ======================================================================================================================
print("\n[15/22] ORTH-SAFE divergence hotspot...")
gap=np.asarray([x["trajectory_gap"] for x in DIFF_SUM]);gc=np.asarray([x["gap_collQ_fraction"] for x in DIFF_SUM])
peak_gap=int(np.argmax(gap));peak_gc=int(np.argmax(gc))
print(f"       max trajectory gap: L{peak_gap:02d} = {gap[peak_gap]:.6f}")
print(f"       max gap CollQ fraction: L{peak_gc:02d} = {gc[peak_gc]:.6f}")

# ======================================================================================================================
# PCA OF DOWNSTREAM DELTAS
# Separate question: is removed-component transport low-dimensional?
# ======================================================================================================================
print("\n[16/22] PCA of downstream displacement...")
PCA_ROWS=[]
for name in ["CAUSAL_ORTH","CAUSAL_SAFE","REMOVED_COMPONENT"]:
    mats=[]
    v={"CAUSAL_ORTH":CAUSAL_ORTH,"CAUSAL_SAFE":SAFE,"REMOVED_COMPONENT":REMOVED}[name]
    for x in XRAY_CAUSAL:
        h0=states(x["context"])
        for sign in (+1,-1):
            h1=states(x["context"],v,PRIMARY_SCALE,sign)
            for L in range(N_LAYERS):
                d=(h1[L][0]-h0[L][0]).float();mats.append(d/d.norm().clamp_min(EPS))
    M=torch.stack(mats).float();M=M-M.mean(0,keepdim=True);_,S,_=torch.linalg.svd(M,full_matrices=False);var=S.square();frac=var/var.sum().clamp_min(EPS)
    row={"method":name,"PC1":float(frac[0]),"PC1_PC2":float(frac[:2].sum()),"PC1_PC5":float(frac[:5].sum()),"effective_rank":float(torch.exp(-(frac*torch.log(frac.clamp_min(EPS))).sum()))};PCA_ROWS.append(row)
    print(f"       {name:<18} PC1={row['PC1']:.4f} PC1+2={row['PC1_PC2']:.4f} PC1+5={row['PC1_PC5']:.4f} effRank={row['effective_rank']:.3f}")

# ======================================================================================================================
# BOOTSTRAP BEHAVIOR — SAFE vs ORTH / REMOVED vs ORTH
# ======================================================================================================================
print("\n[17/22] Paired causal bootstrap...")
def getb(name,scale):return next(x for x in BEHAVIOR if x["method"]==name and abs(x["scale"]-scale)<1e-9)
def boot(a,b,n=20000,seed=172):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):
        ix=rng.integers(0,len(d),len(d));z[i]=d[ix].mean()
    return {"delta":float(d.mean()),"ci_low":float(np.quantile(z,.025)),"ci_high":float(np.quantile(z,.975))}
BOOT=[]
for scale in SCALES:
    s,o,r=getb("CAUSAL_SAFE",scale),getb("CAUSAL_ORTH",scale),getb("REMOVED_COMPONENT",scale)
    z1=boot(s["items"],o["items"],seed=SEED+int(scale*1000));z2=boot(r["items"],o["items"],seed=SEED+5000+int(scale*1000))
    BOOT.append({"scale":scale,"comparison":"SAFE_MINUS_ORTH",**z1});BOOT.append({"scale":scale,"comparison":"REMOVED_MINUS_ORTH",**z2})
    print(f"       s={scale:.3f} SAFE-ORTH={z1['delta']:+.6f}[{z1['ci_low']:+.6f},{z1['ci_high']:+.6f}] | REMOVED-ORTH={z2['delta']:+.6f}[{z2['ci_low']:+.6f},{z2['ci_high']:+.6f}]")

# ======================================================================================================================
# PRIMARY MECHANISM TABLE
# ======================================================================================================================
print("\n[18/22] Primary mechanism @0.50...")
PRIMARY=[]
for name in ["CAUSAL_ORTH","CAUSAL_SAFE","REMOVED_COMPONENT","RAW_COLL_COMPONENT"]:
    b=getb(name,.5);l=next(x for x in LAYER_SUM if x["method"]==name and x["layer"]==19)
    row={"method":name,"C":b["Csym"],"KL":b["KL"],"abs_dNLL":b["abs_dNLL"],"L19_rel":l["rel"],"L19_perp":l["perp"],"L19_target":l["target"],"L19_collQ":l["collQ_frac"]};PRIMARY.append(row)
    print(f"       {name:<18} C={row['C']:+.6f} KL={row['KL']:.6f} |dNLL|={row['abs_dNLL']:.6f} L19rel={row['L19_rel']:.6f} L19perp={row['L19_perp']:.6f} CollQ={row['L19_collQ']:.4f}")

# ======================================================================================================================
# MECHANISM DECISION
# This is an X-ray / mechanistic test, not a new vector-selection test.
# ======================================================================================================================
print("\n[19/22] TEST172 mechanism decision...")
O=next(x for x in PRIMARY if x["method"]=="CAUSAL_ORTH");S=next(x for x in PRIMARY if x["method"]=="CAUSAL_SAFE");R=next(x for x in PRIMARY if x["method"]=="REMOVED_COMPONENT")
safe_causal=S["C"]>=.90*O["C"];safe_kl=S["KL"]<O["KL"];safe_perp=S["L19_perp"]<O["L19_perp"]
removed_collateral_density=(R["KL"]/max(abs(R["C"]),.05))>(O["KL"]/max(abs(O["C"]),.05))
gap_grows=DIFF_SUM[-1]["trajectory_gap"]>DIFF_SUM[0]["trajectory_gap"]
if safe_causal and safe_kl and safe_perp and removed_collateral_density and gap_grows:DECISION="SMALL_COMPONENT_DOWNSTREAM_AMPLIFICATION_SUPPORTED"
elif safe_causal and safe_kl and (removed_collateral_density or gap_grows):DECISION="PARTIAL_COLLATERAL_TRANSPORT_MECHANISM"
elif safe_causal and safe_kl:DECISION="COLLATERAL_SUPPRESSION_REPLICATED__TRANSPORT_MECHANISM_UNRESOLVED"
else:DECISION="COLLATERAL_TRANSPORT_MECHANISM_NOT_CONFIRMED"
print("       SAFE causal >=90% ORTH:",safe_causal);print("       SAFE KL < ORTH:",safe_kl);print("       SAFE L19 perp < ORTH:",safe_perp)
print("       REMOVED collateral-density > ORTH:",removed_collateral_density);print("       ORTH-SAFE trajectory gap grows L0->L19:",gap_grows);print("       DECISION:",DECISION)

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

savecsv("VECTOR_GEOMETRY.csv",GEOM);savecsv("XRAY_RAW.csv",XRAY_ROWS);savecsv("LAYER_SUMMARY.csv",LAYER_SUM)
savecsv("ORTH_SAFE_TRAJECTORY_RAW.csv",DIFF_ROWS);savecsv("ORTH_SAFE_TRAJECTORY_SUMMARY.csv",DIFF_SUM)
savecsv("REMOVED_COMPONENT_FINGERPRINT.csv",REMOVED_LAYER);savecsv("BEHAVIOR.csv",BEHAVIOR)
savecsv("DOSE_XRAY.csv",DOSE_XRAY);savecsv("HOTSPOTS.csv",HOTSPOTS);savecsv("PCA.csv",PCA_ROWS)
savecsv("BOOTSTRAP.csv",BOOT);savecsv("PRIMARY.csv",PRIMARY)

np.savez_compressed(ROOT/"TEST172_COLLATERAL_COMPONENT_TRANSPORT_XRAY.npz",
causal_orth=CAUSAL_ORTH.detach().cpu().numpy(),
causal_safe=SAFE.detach().cpu().numpy(),
removed_component=REMOVED.detach().cpu().numpy(),
raw_coll_component=RAW_COLL.detach().cpu().numpy(),
fisher_guard=FISHER_GUARD.detach().cpu().numpy(),
rho=np.asarray(RHO,dtype=np.float32))

REPORT={
"test":"TEST 172","title":"COLLATERAL COMPONENT TRANSPORT X-RAY","model":MODEL_ID,"seed":SEED,
"frozen_k":FROZEN_K,"primary_scale":PRIMARY_SCALE,"decision":DECISION,
"vector_geometry":GEOM,"layer_summary":LAYER_SUM,"trajectory_summary":DIFF_SUM,
"removed_component":REMOVED_LAYER,"behavior":BEHAVIOR,"hotspots":HOTSPOTS,"pca":PCA_ROWS,
"bootstrap":BOOT,"primary":PRIMARY,
"contract":[
"TEST172 is mechanistic X-ray only; it does not select a new steering vector.",
"TEST169 CAUSAL_ORTH is reconstructed unchanged.",
"TEST171 collateral basis build set and token-norm-weighted actuator-gradient construction are reproduced.",
"Collateral rank K=8 is predeclared from TEST171 and is not searched in TEST172.",
"TEST171 FINAL prompts are not reused for vector or mechanism selection.",
"TEST172 uses a new X-ray causal set and new neutral set.",
"SEASC frozen-norm CUDA injection and locked layer envelope are unchanged.",
"CAUSAL_SAFE follows TEST171 projection order.",
"REMOVED_COMPONENT is the normalized layerwise difference between CAUSAL_ORTH and CAUSAL_SAFE.",
"RAW_COLL_COMPONENT is the direct Qcoll projection and is analyzed separately.",
"No runtime controller, gate, brake, cap, saturation, boost or weight modification.",
"A positive result supports downstream amplification/transport of the removed component; it does not establish that this component is the unique source of collateral behavior."
]}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)

# ======================================================================================================================
# INTEGRITY
# ======================================================================================================================
print("\n[21/22] Integrity...")
print("       P8 ranks:",RANKS);print("       Frozen collateral K:",FROZEN_K);print("       SEASC envelope changed: NO")
print(f"       RSS dose budget: {RSS:.9f}");print("       TEST171 FINAL used for selection: NO");print("       New vector selected in TEST172: NO")
print("       Model weights modified: NO");print("       Runtime controller: OFF")
print(f"       mean SAFE CollQ overlap: {np.mean([x['safe_collQ'] for x in GEOM]):.8e}")

# ======================================================================================================================
# COMPLETE
# ======================================================================================================================
print("\n[22/22] COMPLETE")
print("="*154);print("TEST 172 — COLLATERAL COMPONENT TRANSPORT X-RAY — SUMMARY");print("="*154)
print(f"Mean cos(Fisher,Bank): {np.mean(COS):+.6f}");print(f"Mean raw collateral energy in CAUSAL_ORTH: {np.mean([x['raw_coll_energy'] for x in GEOM]):.6f}")
print(f"Mean cos(REMOVED,RAW_COLL): {np.mean([x['removed_vs_rawcoll'] for x in GEOM]):+.6f}")
print(f"ORTH->SAFE gap L00/L19: {DIFF_SUM[0]['trajectory_gap']:.6f} / {DIFF_SUM[-1]['trajectory_gap']:.6f}")
print("DECISION:",DECISION);print("Saved:",ROOT);print("="*154)
