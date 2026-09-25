# ======================================================================================================================
# TEST 173 — SOURCE-LAYER COLLATERAL NULLING ABLATION
# AkbasCore 3.2 / SEASC
# WHERE IS TEST171 COLLATERAL CLEANING ACTUALLY NEEDED?
#
# PURPOSE:
# TEST171 found a small collateral-sensitive component inside CAUSAL_ORTH.
# TEST172 showed that this small component can amplify strongly downstream.
# TEST173 localizes the SOURCE layers responsible for the useful cleanup:
# keep the SEASC envelope fixed and change ONLY which layers use CAUSAL_SAFE instead of CAUSAL_ORTH.
#
# LOCKS:
# - Qwen/Qwen2.5-7B-Instruct BF16 / SDPA / A100
# - L0-L19
# - TEST169 CAUSAL_ORTH reconstruction unchanged
# - TEST171 collateral basis construction unchanged
# - TEST171 K=8 PREDECLARED / FROZEN
# - Locked SEASC envelope / same RSS dose
# - TEST171/172 FINAL not used for selection
# - DEV selects source-layer mask; FINAL opened only after mask freeze
# - No layer-dose redistribution
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

print("="*154);print("TEST 173 — SOURCE-LAYER COLLATERAL NULLING ABLATION");print("CAUSAL_ORTH vs CAUSAL_SAFE(K=8) vs LAYER-LOCAL HYBRIDS");print("="*154)
print("Ninja:",ninja.__version__)
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=173
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
P8_COUNT=8;SVD_REL_TOL=1e-5;FROZEN_K=8;PRIMARY_SCALE=.50
SCALES=[.125,.25,.50,.75,1.]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST173_SOURCE_LAYER_COLLATERAL_NULLING_ABLATION");ROOT.mkdir(parents=True,exist_ok=True)
print("GPU:",torch.cuda.get_device_name(0));print("Model:",MODEL_ID);print("Layers: L0-L19 | frozen K=8 | primary dose=0.50")

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
# CUDA SEASC — TEST172 PHYSICS UNCHANGED
# ======================================================================================================================
BUILD="/tmp/akbascore_test173";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST173 SEASC");}"""
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
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test173_seasc",[&]{k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/24] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test173_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
print("       OK")

# ======================================================================================================================
# MODEL
# ======================================================================================================================
print("\n[2/24] Loading frozen model...")
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

print("\n[3/24] Rebuilding TEST169 Fisher / Bank geometry...")
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
# P8 — TEST171 LINEAGE LOCK
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

def activation_gradient(context,continuation,weighted=False,positive_score=False):
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
        score=lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean()
        objective=score if positive_score else -score;objective.backward();gg=[]
        for L in range(N_LAYERS):
            g=saved[L].grad[0].float()
            if weighted:
                h=saved[L].detach()[0].float();g=(h.norm(dim=-1,keepdim=True)*g).sum(dim=0)
            else:g=g.sum(dim=0)
            gg.append(g.detach().clone())
        val=float(objective.detach())
    finally:
        for h in hs:h.remove()
        model.zero_grad(set_to_none=True)
    del ids,mask,emb,o,z,tar,lp,score,objective,saved;return gg,val

print("\n[4/24] Computing frozen P8 retention basis...")
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
# TEST169 CAUSAL_ORTH — UNCHANGED
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

print("\n[5/24] Reconstructing TEST169 CAUSAL_ORTH...")
FG={}
for fn,items in FORGE.items():
    acc=[torch.zeros(HIDDEN,device=DEVICE,dtype=torch.float32) for _ in range(N_LAYERS)]
    for j,x in enumerate(items,1):
        gp,_=activation_gradient(x["context"],x["pos"],positive_score=True);gn,_=activation_gradient(x["context"],x["neg"],positive_score=True)
        for L in range(N_LAYERS):acc[L]+=gp[L]-gn[L]
        print(f"       family {fn} item {j}/4")
    v=torch.stack([norm(x) for x in acc]).contiguous()
    for L in range(N_LAYERS):
        if torch.dot(v[L],FISHER[L])<0:v[L]=-v[L]
    FG[fn]=v
CAUSAL=torch.stack([norm(torch.stack([FG[k][L] for k in FG]).mean(0)) for L in range(N_LAYERS)]).contiguous()
CAUSAL_GUARD,_=project_basis(CAUSAL,Q8);CO=[];ORTH=[]
for L in range(N_LAYERS):
    f=FISHER_GUARD[L];g=CAUSAL_GUARD[L];r=g-torch.dot(g,f)*f;rn=r.norm()
    if rn<EPS:raise RuntimeError(f"CAUSAL_ORTH collapsed L{L}")
    CO.append(r/rn);ORTH.append(float(rn))
CAUSAL_ORTH=torch.stack(CO).contiguous()
print(f"       mean cos(Causal,Fisher)={np.mean([float(torch.dot(CAUSAL[L],FISHER[L])) for L in range(N_LAYERS)]):+.6f}")
print(f"       mean Fisher-orthogonal component={np.mean(ORTH):.6f}")
print(f"       mean P8 overlap={np.mean([float((Q8[L].T@CAUSAL_ORTH[L]).norm()) for L in range(N_LAYERS)]):.8e}")

# ======================================================================================================================
# TEST171 COLLATERAL BASIS — EXACT BUILD
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

print("\n[6/24] Rebuilding TEST171 collateral basis...")
COLL_G=[]
for i,(ctx,cont) in enumerate(COLL_BUILD,1):
    g,l=activation_gradient(ctx,cont,weighted=True);COLL_G.append(g)
    print(f"       neutral {i}/8 | NLL={l:.6f} | mean ||g_act||={np.mean([float(x.norm()) for x in g]):.6e}")
QCOLL=[]
for L in range(N_LAYERS):
    G=torch.stack([COLL_G[i][L] for i in range(8)]).float();G=G[G.norm(dim=1)>EPS];G=norm(G);_,s,Vh=torch.linalg.svd(G,full_matrices=False);QCOLL.append(Vh[:FROZEN_K].T.contiguous())
print("       Frozen collateral ranks:",[q.shape[1] for q in QCOLL])

# ======================================================================================================================
# TEST171 CAUSAL_SAFE — EXACT PROJECTION ORDER
# ======================================================================================================================
print("\n[7/24] Reconstructing TEST171 CAUSAL_SAFE...")
SAFE=[];GEOM=[]
for L in range(N_LAYERS):
    u=CAUSAL_ORTH[L].float();qc=QCOLL[L];pc=qc@(qc.T@u);v=u-pc
    q=Q8[L];v=v-q@(q.T@v);f=FISHER_GUARD[L];v=v-torch.dot(v,f)*f;v=v-q@(q.T@v);v=norm(v)
    SAFE.append(v);GEOM.append({"layer":L,"raw_coll_energy":float(pc.square().sum()),"safe_cos_orth":float(torch.dot(v,u)),"removed_norm":float((u-v).norm()),"safe_collQ":float((qc.T@v).norm())})
SAFE=torch.stack(SAFE).contiguous()
print(f"       mean raw collateral energy={np.mean([x['raw_coll_energy'] for x in GEOM]):.6f}")
print(f"       mean cos(SAFE,ORTH)={np.mean([x['safe_cos_orth'] for x in GEOM]):.6f}")
print(f"       mean ||ORTH-SAFE||={np.mean([x['removed_norm'] for x in GEOM]):.6f}")
print(f"       mean SAFE CollQ={np.mean([x['safe_collQ'] for x in GEOM]):.8e}")

# ======================================================================================================================
# TEST173 DATA — NEW DEV / FINAL, NO TEST171/172 FINAL REUSE
# ======================================================================================================================
DEV_CAUSAL=[
{"context":"A monitoring drone and a service road are visible, with no vertical relation specified.","pos":" The monitoring drone is above the service road.","neg":" The monitoring drone is below the service road."},
{"context":"A suspended microphone and a studio table are visible, with no vertical relation specified.","pos":" The suspended microphone is above the studio table.","neg":" The suspended microphone is below the studio table."},
{"context":"A weather balloon and a river crossing are visible, with no vertical relation specified.","pos":" The weather balloon is above the river crossing.","neg":" The weather balloon is below the river crossing."},
{"context":"A ceiling sensor and a laboratory bench are visible, with no vertical relation specified.","pos":" The ceiling sensor is above the laboratory bench.","neg":" The ceiling sensor is below the laboratory bench."},
{"context":"A gull and a coastal path are visible, with no vertical relation specified.","pos":" The gull is above the coastal path.","neg":" The gull is below the coastal path."},
{"context":"A signal lamp and a cargo platform are visible, with no vertical relation specified.","pos":" The signal lamp is above the cargo platform.","neg":" The signal lamp is below the cargo platform."},
{"context":"A small aircraft and a desert track are visible, with no vertical relation specified.","pos":" The small aircraft is above the desert track.","neg":" The small aircraft is below the desert track."},
{"context":"A projector and a meeting table are visible, with no vertical relation specified.","pos":" The projector is above the meeting table.","neg":" The projector is below the meeting table."}]
DEV_NEUTRAL=[
("A ceramic mug rests beside a folded map."," The objects remain stationary."),
("A wooden pencil lies beside a sealed folder."," The objects remain stationary."),
("A glass bottle rests next to an empty dish."," The objects remain stationary."),
("A metal clip lies beside a cardboard card."," The objects remain stationary."),
("A fabric pouch rests next to a plastic case."," The objects remain stationary."),
("A plain saucer sits beside a closed journal."," The objects remain stationary."),
("A small carton stands beside a clean cloth."," The objects remain stationary."),
("A steel fork rests beside an empty plate."," The objects remain stationary.")]

FINAL_CAUSAL=[
{"context":"An inspection drone and a forest trail are visible, with no vertical relation specified.","pos":" The inspection drone is above the forest trail.","neg":" The inspection drone is below the forest trail."},
{"context":"A hanging camera and a workshop counter are visible, with no vertical relation specified.","pos":" The hanging camera is above the workshop counter.","neg":" The hanging camera is below the workshop counter."},
{"context":"A research balloon and a highway bridge are visible, with no vertical relation specified.","pos":" The research balloon is above the highway bridge.","neg":" The research balloon is below the highway bridge."},
{"context":"A suspended detector and a technical desk are visible, with no vertical relation specified.","pos":" The suspended detector is above the technical desk.","neg":" The suspended detector is below the technical desk."},
{"context":"A hawk and a gravel path are visible, with no vertical relation specified.","pos":" The hawk is above the gravel path.","neg":" The hawk is below the gravel path."},
{"context":"A warning light and a freight platform are visible, with no vertical relation specified.","pos":" The warning light is above the freight platform.","neg":" The warning light is below the freight platform."},
{"context":"A glider and a rural track are visible, with no vertical relation specified.","pos":" The glider is above the rural track.","neg":" The glider is below the rural track."},
{"context":"A hanging display and a classroom table are visible, with no vertical relation specified.","pos":" The hanging display is above the classroom table.","neg":" The hanging display is below the classroom table."}]
FINAL_NEUTRAL=[
("A ceramic pitcher rests beside a folded towel."," The objects remain stationary."),
("A wooden marker lies beside a closed binder."," The objects remain stationary."),
("A glass bowl rests next to an empty board."," The objects remain stationary."),
("A metal token lies beside a paper folder."," The objects remain stationary."),
("A cloth bag rests next to a plastic tray."," The objects remain stationary."),
("A plain cup sits beside a closed manual."," The objects remain stationary."),
("A small package stands beside a clean napkin."," The objects remain stationary."),
("A steel knife rests beside an empty dish."," The objects remain stationary.")]

# ======================================================================================================================
# SOURCE-LAYER MASKS
# SAFE means use TEST171 cleaned vector at that source layer.
# ORTH means leave TEST169 CAUSAL_ORTH unchanged.
# All 20 SEASC doses remain active and unchanged.
# ======================================================================================================================
MASKS={
"ORTH_ALL":[],
"SAFE_ALL":list(range(20)),
"SAFE_L00_04":list(range(0,5)),
"SAFE_L05_09":list(range(5,10)),
"SAFE_L10_14":list(range(10,15)),
"SAFE_L15_19":list(range(15,20)),
"SAFE_L00_09":list(range(0,10)),
"SAFE_L10_19":list(range(10,20)),
"SAFE_L00_14":list(range(0,15)),
"SAFE_L05_19":list(range(5,20))}
def hybrid(mask):
    idx=set(mask);return torch.stack([SAFE[L] if L in idx else CAUSAL_ORTH[L] for L in range(N_LAYERS)]).contiguous()
VECTORS={k:hybrid(v) for k,v in MASKS.items()}

print("\n[8/24] Building source-layer ablation arms...")
for k,m in MASKS.items():print(f"       {k:<12} cleaned={len(m):02d}/20 layers={m if m else 'NONE'}")

# ======================================================================================================================
# SEASC HOOKS / ASSAYS
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
    ids,mask,plen=sequence(context,continuation);hs=hooks(v,sign,scale,blank_tel()) if v is not None else []
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

def neutral_arm(name,v,scale,items):
    z=[]
    for ctx,cont in items:z+=[neutral_one(ctx,cont,v,scale,+1),neutral_one(ctx,cont,v,scale,-1)]
    return {"method":name,"scale":scale,"KL":float(np.mean([x[1] for x in z])),"abs_dNLL":float(np.mean([abs(x[0]) for x in z])),"ret_items":[float(abs(x[0])) for x in z],"kl_items":[float(x[1]) for x in z]}

# ======================================================================================================================
# DEV BASELINES
# ======================================================================================================================
print("\n[9/24] DEV baselines @0.50...")
DEV=[]
for name in ["ORTH_ALL","SAFE_ALL"]:
    a=causal_arm(name,VECTORS[name],PRIMARY_SCALE,DEV_CAUSAL);n=neutral_arm(name,VECTORS[name],PRIMARY_SCALE,DEV_NEUTRAL);r={**a,**{k:v for k,v in n.items() if k not in ("method","scale")}};DEV.append(r)
    print(f"       {name:<12} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")

# ======================================================================================================================
# DEV 5-LAYER BLOCK ABLATION
# ======================================================================================================================
print("\n[10/24] DEV 5-layer source blocks...")
for name in ["SAFE_L00_04","SAFE_L05_09","SAFE_L10_14","SAFE_L15_19"]:
    a=causal_arm(name,VECTORS[name],PRIMARY_SCALE,DEV_CAUSAL);n=neutral_arm(name,VECTORS[name],PRIMARY_SCALE,DEV_NEUTRAL);r={**a,**{k:v for k,v in n.items() if k not in ("method","scale")}};DEV.append(r)
    print(f"       {name:<12} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")

# ======================================================================================================================
# DEV PREFIX / SUFFIX ABLATION
# ======================================================================================================================
print("\n[11/24] DEV prefix / suffix masks...")
for name in ["SAFE_L00_09","SAFE_L10_19","SAFE_L00_14","SAFE_L05_19"]:
    a=causal_arm(name,VECTORS[name],PRIMARY_SCALE,DEV_CAUSAL);n=neutral_arm(name,VECTORS[name],PRIMARY_SCALE,DEV_NEUTRAL);r={**a,**{k:v for k,v in n.items() if k not in ("method","scale")}};DEV.append(r)
    print(f"       {name:<12} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")

# ======================================================================================================================
# DEV NORMALIZED BURDEN
# Select smallest cleaned-layer mask satisfying causal preservation and collateral improvement.
# No FINAL information is used.
# ======================================================================================================================
print("\n[12/24] DEV viability / burden...")
def row(name):return next(x for x in DEV if x["method"]==name)
O=row("ORTH_ALL");SA=row("SAFE_ALL")
C_FLOOR=.90*O["Csym"];KL_TARGET=O["KL"]*.95;RET_TARGET=O["abs_dNLL"]*1.10
CAND=[]
for name in MASKS:
    if name in ("ORTH_ALL","SAFE_ALL"):continue
    r=row(name);viable=r["Csym"]>=C_FLOOR and r["positive"]>=7
    improved=r["KL"]<=KL_TARGET
    retention_ok=r["abs_dNLL"]<=RET_TARGET
    kl_ratio=r["KL"]/max(O["KL"],EPS);ret_ratio=r["abs_dNLL"]/max(O["abs_dNLL"],EPS);c_ratio=r["Csym"]/max(O["Csym"],EPS)
    burden=.60*kl_ratio+.25*ret_ratio+.15*(1/max(c_ratio,EPS))
    z={"method":name,"cleaned_layers":len(MASKS[name]),"viable":viable,"KL_improved_5pct":improved,"retention_ok":retention_ok,"causal_ratio":c_ratio,"KL_ratio":kl_ratio,"ret_ratio":ret_ratio,"burden":burden};CAND.append(z)
    print(f"       {name:<12} layers={len(MASKS[name]):02d} viable={viable} KL5={improved} RetOK={retention_ok} Cx={c_ratio:.4f} KLx={kl_ratio:.4f} Rx={ret_ratio:.4f} burden={burden:.4f}")

# ======================================================================================================================
# PREDECLARED SELECTION
# 1) viable + >=5% KL improvement + retention guard
# 2) fewest cleaned source layers
# 3) lowest burden
# fallback SAFE_ALL if no partial mask qualifies
# ======================================================================================================================
print("\n[13/24] Freezing source-layer mask...")
SAFE_CAND=[x for x in CAND if x["viable"] and x["KL_improved_5pct"] and x["retention_ok"]]
if SAFE_CAND:
    min_layers=min(x["cleaned_layers"] for x in SAFE_CAND);POOL=[x for x in SAFE_CAND if x["cleaned_layers"]==min_layers]
    WIN=min(POOL,key=lambda x:x["burden"]);FROZEN_MASK=WIN["method"];FALLBACK=False
else:FROZEN_MASK="SAFE_ALL";FALLBACK=True
FROZEN_LAYERS=MASKS[FROZEN_MASK];FROZEN_VECTOR=VECTORS[FROZEN_MASK]
print("       FROZEN:",FROZEN_MASK);print("       cleaned source layers:",FROZEN_LAYERS if FROZEN_LAYERS else "NONE");print("       fallback:",FALLBACK)
print("       FINAL remains unopened.")

# ======================================================================================================================
# LEAVE-ONE-DEV-ITEM-OUT ROBUSTNESS — FROZEN CANDIDATE ONLY
# ======================================================================================================================
print("\n[14/24] DEV leave-one-item-out robustness...")
R=row(FROZEN_MASK);LOO=[]
o_items=np.asarray(O["items"]);r_items=np.asarray(R["items"])
for i in range(len(o_items)):
    ix=np.arange(len(o_items))!=i;co=float(r_items[ix].mean());bo=float(o_items[ix].mean());ratio=co/max(bo,EPS)
    LOO.append({"held_out":i,"candidate_C":co,"orth_C":bo,"causal_ratio":ratio})
    print(f"       hold {i+1}/8 C={co:+.6f} ORTH={bo:+.6f} ratio={ratio:.4f}")
LOO_MIN=min(x["causal_ratio"] for x in LOO);print(f"       minimum LOO causal ratio={LOO_MIN:.4f}")

# ======================================================================================================================
# MASK GEOMETRY
# ======================================================================================================================
print("\n[15/24] Frozen-mask geometry...")
MASK_GEOM=[]
for L in range(N_LAYERS):
    v=FROZEN_VECTOR[L];qc=QCOLL[L];q=Q8[L]
    z={"layer":L,"cleaned":int(L in FROZEN_LAYERS),"cos_to_orth":float(torch.dot(v,CAUSAL_ORTH[L])),"cos_to_safe":float(torch.dot(v,SAFE[L])),"collQ":float((qc.T@v).norm()),"P8":float((q.T@v).norm()),"rho":float(RHO[L])}
    MASK_GEOM.append(z);print(f"       L{L:02d} clean={z['cleaned']} cosO={z['cos_to_orth']:+.6f} cosS={z['cos_to_safe']:+.6f} CollQ={z['collQ']:.6f}")

# ======================================================================================================================
# FINAL OPEN — ONLY AFTER MASK FREEZE
# ======================================================================================================================
print("\n[16/24] FINAL opened — frozen mask evaluation...")
FINAL=[]
for scale in SCALES:
    print(f"\n       DOSE={scale:.3f}")
    for name,v in [("ORTH_ALL",VECTORS["ORTH_ALL"]),("SAFE_ALL",VECTORS["SAFE_ALL"]),("FROZEN_PARTIAL",FROZEN_VECTOR)]:
        a=causal_arm(name,v,scale,FINAL_CAUSAL);n=neutral_arm(name,v,scale,FINAL_NEUTRAL);r={**a,**{k:v for k,v in n.items() if k not in ("method","scale")}};FINAL.append(r)
        print(f"       {name:<14} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")

# ======================================================================================================================
# FINAL PAIRED CAUSAL BOOTSTRAP
# ======================================================================================================================
print("\n[17/24] FINAL paired causal bootstrap...")
def frow(name,scale):return next(x for x in FINAL if x["method"]==name and abs(x["scale"]-scale)<1e-9)
def bootdiff(a,b,n=20000,seed=173):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n,dtype=np.float64)
    for i in range(n):
        ix=rng.integers(0,len(d),len(d));z[i]=d[ix].mean()
    return {"delta":float(d.mean()),"ci_low":float(np.quantile(z,.025)),"ci_high":float(np.quantile(z,.975))}
BOOT=[]
for s in SCALES:
    p,o,sa=frow("FROZEN_PARTIAL",s),frow("ORTH_ALL",s),frow("SAFE_ALL",s)
    a=bootdiff(p["items"],o["items"],seed=SEED+int(s*1000));b=bootdiff(p["items"],sa["items"],seed=SEED+5000+int(s*1000))
    BOOT+=[{"scale":s,"comparison":"PARTIAL_MINUS_ORTH",**a},{"scale":s,"comparison":"PARTIAL_MINUS_SAFE",**b}]
    print(f"       s={s:.3f} P-O={a['delta']:+.6f}[{a['ci_low']:+.6f},{a['ci_high']:+.6f}] | P-S={b['delta']:+.6f}[{b['ci_low']:+.6f},{b['ci_high']:+.6f}]")

# ======================================================================================================================
# FINAL NEUTRAL PAIRED BOOTSTRAP
# 16 observations = 8 prompts x +/- steering signs.
# ======================================================================================================================
print("\n[18/24] FINAL neutral bootstrap...")
NBOOT=[]
for s in SCALES:
    p,o,sa=frow("FROZEN_PARTIAL",s),frow("ORTH_ALL",s),frow("SAFE_ALL",s)
    for metric,key in [("KL","kl_items"),("RET","ret_items")]:
        a=bootdiff(p[key],o[key],seed=SEED+10000+int(s*1000)+(0 if metric=="KL" else 1))
        b=bootdiff(p[key],sa[key],seed=SEED+20000+int(s*1000)+(0 if metric=="KL" else 1))
        NBOOT+=[{"scale":s,"metric":metric,"comparison":"PARTIAL_MINUS_ORTH",**a},{"scale":s,"metric":metric,"comparison":"PARTIAL_MINUS_SAFE",**b}]
        print(f"       s={s:.3f} {metric:<3} P-O={a['delta']:+.6f}[{a['ci_low']:+.6f},{a['ci_high']:+.6f}] | P-S={b['delta']:+.6f}[{b['ci_low']:+.6f},{b['ci_high']:+.6f}]")

# ======================================================================================================================
# PRIMARY FINAL TABLE @0.50
# ======================================================================================================================
print("\n[19/24] Primary FINAL @0.50...")
PRIMARY=[]
for name in ["ORTH_ALL","SAFE_ALL","FROZEN_PARTIAL"]:
    r=frow(name,.5);z={"method":name,"C":r["Csym"],"positive":r["positive"],"KL":r["KL"],"abs_dNLL":r["abs_dNLL"]};PRIMARY.append(z)
    print(f"       {name:<14} C={z['C']:+.6f} P+={z['positive']}/8 KL={z['KL']:.6f} |dNLL|={z['abs_dNLL']:.6f}")
PO=next(x for x in PRIMARY if x["method"]=="ORTH_ALL");PS=next(x for x in PRIMARY if x["method"]=="SAFE_ALL");PP=next(x for x in PRIMARY if x["method"]=="FROZEN_PARTIAL")
print(f"       PARTIAL/ORTH causal={PP['C']/max(PO['C'],EPS):.4f} KL={PP['KL']/max(PO['KL'],EPS):.4f} Ret={PP['abs_dNLL']/max(PO['abs_dNLL'],EPS):.4f}")
print(f"       PARTIAL/SAFE causal={PP['C']/max(PS['C'],EPS):.4f} KL={PP['KL']/max(PS['KL'],EPS):.4f} Ret={PP['abs_dNLL']/max(PS['abs_dNLL'],EPS):.4f}")

# ======================================================================================================================
# SOURCE-LOCALIZATION MAP
# DEV-only diagnostic: benefit per cleaned source layer.
# ======================================================================================================================
print("\n[20/24] Source-layer localization map...")
LOCAL=[]
for name in ["SAFE_L00_04","SAFE_L05_09","SAFE_L10_14","SAFE_L15_19"]:
    r=row(name);layers_clean=MASKS[name];kl_gain=O["KL"]-r["KL"];ret_gain=O["abs_dNLL"]-r["abs_dNLL"];c_delta=r["Csym"]-O["Csym"]
    z={"method":name,"start":min(layers_clean),"end":max(layers_clean),"causal_delta":c_delta,"KL_reduction":kl_gain,"retention_reduction":ret_gain,"KL_reduction_per_layer":kl_gain/len(layers_clean)}
    LOCAL.append(z);print(f"       {name:<12} ΔC={c_delta:+.6f} ΔKL={-kl_gain:+.6f} Δ|dNLL|={-ret_gain:+.6f} KLgain/layer={z['KL_reduction_per_layer']:+.6f}")
BEST_BLOCK=max(LOCAL,key=lambda x:x["KL_reduction_per_layer"])
print(f"       strongest DEV KL cleanup per layer: {BEST_BLOCK['method']}")

# ======================================================================================================================
# DECISION
# TEST173 asks whether cleanup can be source-localized without changing envelope.
# FINAL does NOT re-select the mask.
# ======================================================================================================================
print("\n[21/24] TEST173 decision...")
primary_boot=next(x for x in BOOT if x["scale"]==.5 and x["comparison"]=="PARTIAL_MINUS_ORTH")
primary_kl=next(x for x in NBOOT if x["scale"]==.5 and x["metric"]=="KL" and x["comparison"]=="PARTIAL_MINUS_ORTH")
partial_mask=FROZEN_MASK!="SAFE_ALL"
causal_preserved=PP["C"]>=.90*PO["C"] and PP["positive"]>=7
kl_better=PP["KL"]<PO["KL"]
causal_ci_ok=primary_boot["ci_low"]>-.10*abs(PO["C"])
localized=partial_mask and causal_preserved and kl_better
if localized and primary_kl["ci_high"]<0:DECISION="SOURCE_LOCALIZED_COLLATERAL_CLEANUP_SUPPORTED"
elif localized:DECISION="SOURCE_LOCALIZATION_SUPPORTED_BY_POINT_ESTIMATE"
elif FROZEN_MASK=="SAFE_ALL" and causal_preserved and kl_better:DECISION="FULL_DEPTH_CLEANUP_REMAINS_NECESSARY"
else:DECISION="SOURCE_LAYER_LOCALIZATION_NOT_CONFIRMED"
print("       partial mask frozen:",partial_mask)
print("       FINAL causal >=90% ORTH:",causal_preserved)
print("       FINAL KL < ORTH:",kl_better)
print("       causal bootstrap guard:",causal_ci_ok)
print("       primary KL bootstrap upper < 0:",primary_kl["ci_high"]<0)
print("       DECISION:",DECISION)

# ======================================================================================================================
# SAVE
# ======================================================================================================================
print("\n[22/24] Saving...")
def scalar(rows):return [{k:v for k,v in r.items() if not isinstance(v,(list,tuple,dict,np.ndarray))} for r in rows]
def savecsv(name,rows):
    rows=scalar(rows)
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("VECTOR_GEOMETRY.csv",GEOM);savecsv("DEV.csv",DEV);savecsv("CANDIDATES.csv",CAND);savecsv("LOO.csv",LOO)
savecsv("MASK_GEOMETRY.csv",MASK_GEOM);savecsv("FINAL.csv",FINAL);savecsv("CAUSAL_BOOTSTRAP.csv",BOOT)
savecsv("NEUTRAL_BOOTSTRAP.csv",NBOOT);savecsv("PRIMARY.csv",PRIMARY);savecsv("LOCALIZATION.csv",LOCAL)
np.savez_compressed(ROOT/"TEST173_SOURCE_LAYER_COLLATERAL_NULLING_ABLATION.npz",
causal_orth=CAUSAL_ORTH.detach().cpu().numpy(),causal_safe=SAFE.detach().cpu().numpy(),
frozen_partial=FROZEN_VECTOR.detach().cpu().numpy(),frozen_mask=np.asarray(FROZEN_LAYERS,dtype=np.int32),
fisher_guard=FISHER_GUARD.detach().cpu().numpy(),rho=np.asarray(RHO,dtype=np.float32))
REPORT={
"test":"TEST 173","title":"SOURCE-LAYER COLLATERAL NULLING ABLATION","model":MODEL_ID,"seed":SEED,
"frozen_k":FROZEN_K,"primary_scale":PRIMARY_SCALE,"rss":RSS,"frozen_mask":FROZEN_MASK,"frozen_layers":FROZEN_LAYERS,
"fallback":FALLBACK,"decision":DECISION,"dev":DEV,"candidates":CAND,"loo":LOO,"mask_geometry":MASK_GEOM,
"final":FINAL,"causal_bootstrap":BOOT,"neutral_bootstrap":NBOOT,"primary":PRIMARY,"localization":LOCAL,
"contract":[
"TEST173 changes only which source layers use TEST171 CAUSAL_SAFE versus TEST169 CAUSAL_ORTH.",
"All L0-L19 SEASC injections remain active and retain the locked SEASC dose envelope.",
"TEST169 CAUSAL_ORTH reconstruction is unchanged.",
"TEST171 collateral basis construction and frozen K=8 are unchanged.",
"Source-layer mask selection uses TEST173 DEV only.",
"TEST173 FINAL remains unopened until the source-layer mask is frozen.",
"TEST171 and TEST172 FINAL results are not used to select the TEST173 mask.",
"No dose redistribution, runtime controller, gate, brake, cap, saturation, boost or weight modification is introduced.",
"A positive result supports source-layer localization of collateral cleanup; it does not prove that other layers have zero collateral contribution."
]}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)

# ======================================================================================================================
# INTEGRITY
# ======================================================================================================================
print("\n[23/24] Integrity...")
print("       P8 ranks:",RANKS);print("       Frozen collateral K:",FROZEN_K)
print("       Frozen source mask:",FROZEN_MASK);print("       Frozen source layers:",FROZEN_LAYERS if FROZEN_LAYERS else "NONE")
print("       Mask frozen before FINAL: YES");print("       TEST171/172 FINAL used for selection: NO")
print("       SEASC envelope changed: NO");print(f"       RSS dose budget: {RSS:.9f}")
print("       All L0-L19 injections active: YES");print("       Model weights modified: NO");print("       Runtime controller: OFF")

# ======================================================================================================================
# COMPLETE
# ======================================================================================================================
print("\n[24/24] COMPLETE")
print("="*154);print("TEST 173 — SOURCE-LAYER COLLATERAL NULLING ABLATION — SUMMARY");print("="*154)
print(f"Mean cos(Fisher,Bank): {np.mean(COS):+.6f}")
print(f"Mean raw collateral energy in CAUSAL_ORTH: {np.mean([x['raw_coll_energy'] for x in GEOM]):.6f}")
print("Frozen mask:",FROZEN_MASK);print("Frozen cleaned layers:",FROZEN_LAYERS if FROZEN_LAYERS else "NONE")
print(f"Primary FINAL ORTH    : C={PO['C']:+.6f} KL={PO['KL']:.6f} |dNLL|={PO['abs_dNLL']:.6f}")
print(f"Primary FINAL SAFE    : C={PS['C']:+.6f} KL={PS['KL']:.6f} |dNLL|={PS['abs_dNLL']:.6f}")
print(f"Primary FINAL PARTIAL : C={PP['C']:+.6f} KL={PP['KL']:.6f} |dNLL|={PP['abs_dNLL']:.6f}")
print("Strongest DEV block:",BEST_BLOCK["method"]);print("DECISION:",DECISION);print("Saved:",ROOT);print("="*154)
