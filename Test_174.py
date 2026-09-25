# ======================================================================================================================
# TEST 174 — EARLY SOURCE-LAYER MICRO-LOCALIZATION
# AkbasCore 3.2 / SEASC
# TEST173 L00-L04 -> SINGLE-LAYER / PAIR SOURCE LOCALIZATION
#
# PURPOSE:
# TEST173 froze SAFE_L00_04 and showed that collateral cleanup can be localized to early source layers.
# TEST174 asks whether that 5-layer region can be reduced further to one or two specific source layers.
#
# LOCKS:
# - Qwen/Qwen2.5-7B-Instruct BF16 / SDPA / A100
# - L0-L19; all injections remain active
# - TEST169 CAUSAL_ORTH unchanged
# - TEST171 QCOLL construction unchanged; K=8 frozen
# - TEST173 parent region L00-L04 PREDECLARED / FROZEN
# - Locked SEASC envelope / same RSS dose
# - Search only inside L00-L04
# - Hierarchical DEV: singles first; pairs only if no single qualifies
# - FINAL opened only after exact layer mask is frozen
# - TEST171/172/173 FINAL never used for selection
# - No envelope search / controller / gate / brake / cap / boost
# - Model weights unchanged
# ======================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:
    print("Installing required build dependency: ninja...")
    subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import ninja
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154);print("TEST 174 — EARLY SOURCE-LAYER MICRO-LOCALIZATION");print("TEST173 L00-L04 -> SINGLE-LAYER / PAIR SOURCE LOCALIZATION");print("="*154)
print("Ninja:",ninja.__version__)
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=174
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
P8_COUNT=8;SVD_REL_TOL=1e-5;FROZEN_K=8;PRIMARY_SCALE=.50
PARENT_REGION=[0,1,2,3,4];SCALES=[.125,.25,.50,.75,1.]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST174_EARLY_SOURCE_LAYER_MICRO_LOCALIZATION");ROOT.mkdir(parents=True,exist_ok=True)
print("GPU:",torch.cuda.get_device_name(0));print("Model:",MODEL_ID)
print("Layers: L0-L19 | frozen K=8 | TEST173 parent=L00-L04 | primary dose=0.50")

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
# CUDA SEASC — TEST173 PHYSICS UNCHANGED
# ======================================================================================================================
BUILD="/tmp/akbascore_test174";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST174 SEASC");}"""
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
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test174_seasc",[&]{k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/24] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test174_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
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
        score=lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean();objective=score if positive_score else -score;objective.backward();gg=[]
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
FISHER_GUARD,_=project_basis(FISHER,Q8)

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
CAUSAL_GUARD,_=project_basis(CAUSAL,Q8);CO=[];ORTH_FRAC=[]
for L in range(N_LAYERS):
    f=FISHER_GUARD[L];g=CAUSAL_GUARD[L];r=g-torch.dot(g,f)*f;rn=r.norm()
    if rn<EPS:raise RuntimeError(f"CAUSAL_ORTH collapsed L{L}")
    CO.append(r/rn);ORTH_FRAC.append(float(rn))
CAUSAL_ORTH=torch.stack(CO).contiguous()
print(f"       mean cos(Causal,Fisher)={np.mean([float(torch.dot(CAUSAL[L],FISHER[L])) for L in range(N_LAYERS)]):+.6f}")
print(f"       mean Fisher-orthogonal component={np.mean(ORTH_FRAC):.6f}")
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
# TEST171 CAUSAL_SAFE — EXACT ORDER
# ======================================================================================================================
print("\n[7/24] Reconstructing TEST171 CAUSAL_SAFE...")
SAFE=[];GEOM=[]
for L in range(N_LAYERS):
    u=CAUSAL_ORTH[L].float();qc=QCOLL[L];pc=qc@(qc.T@u);v=u-pc
    q=Q8[L];v=v-q@(q.T@v);f=FISHER_GUARD[L];v=v-torch.dot(v,f)*f;v=v-q@(q.T@v);v=norm(v)
    SAFE.append(v);GEOM.append({"layer":L,"raw_coll_energy":float(pc.square().sum()/u.square().sum().clamp_min(EPS)),"safe_cos_orth":float(torch.dot(v,u)),"removed_norm":float((u-v).norm()),"safe_collQ":float((qc.T@v).norm())})
SAFE=torch.stack(SAFE).contiguous()
print(f"       mean raw collateral energy={np.mean([x['raw_coll_energy'] for x in GEOM]):.6f}")
print(f"       mean cos(SAFE,ORTH)={np.mean([x['safe_cos_orth'] for x in GEOM]):.6f}")
print(f"       mean ||ORTH-SAFE||={np.mean([x['removed_norm'] for x in GEOM]):.6f}")
print(f"       mean SAFE CollQ={np.mean([x['safe_collQ'] for x in GEOM]):.8e}")

# ======================================================================================================================
# TEST174 NEW DEV / FINAL — SEALED FROM TEST173
# ======================================================================================================================
DEV_CAUSAL=[
{"context":"An aerial sensor and a maintenance lane are visible, with no vertical relation specified.","pos":" The aerial sensor is above the maintenance lane.","neg":" The aerial sensor is below the maintenance lane."},
{"context":"A suspended recorder and an engineering bench are visible, with no vertical relation specified.","pos":" The suspended recorder is above the engineering bench.","neg":" The suspended recorder is below the engineering bench."},
{"context":"A sounding balloon and a canal bridge are visible, with no vertical relation specified.","pos":" The sounding balloon is above the canal bridge.","neg":" The sounding balloon is below the canal bridge."},
{"context":"A ceiling detector and a testing table are visible, with no vertical relation specified.","pos":" The ceiling detector is above the testing table.","neg":" The ceiling detector is below the testing table."},
{"context":"A falcon and a country path are visible, with no vertical relation specified.","pos":" The falcon is above the country path.","neg":" The falcon is below the country path."},
{"context":"A signal beacon and a storage deck are visible, with no vertical relation specified.","pos":" The signal beacon is above the storage deck.","neg":" The signal beacon is below the storage deck."},
{"context":"A light aircraft and a farm road are visible, with no vertical relation specified.","pos":" The light aircraft is above the farm road.","neg":" The light aircraft is below the farm road."},
{"context":"A suspended monitor and a training desk are visible, with no vertical relation specified.","pos":" The suspended monitor is above the training desk.","neg":" The suspended monitor is below the training desk."}]
DEV_NEUTRAL=[
("A porcelain cup rests beside a folded leaflet."," The objects remain stationary."),
("A wooden pen lies beside a sealed workbook."," The objects remain stationary."),
("A glass jar rests next to an empty saucer."," The objects remain stationary."),
("A metal tag lies beside a paper packet."," The objects remain stationary."),
("A fabric case rests next to a plastic box."," The objects remain stationary."),
("A plain bowl sits beside a closed catalogue."," The objects remain stationary."),
("A small parcel stands beside a clean cloth."," The objects remain stationary."),
("A steel spoon lies beside an empty tray."," The objects remain stationary.")]

FINAL_CAUSAL=[
{"context":"A mapping drone and a woodland track are visible, with no vertical relation specified.","pos":" The mapping drone is above the woodland track.","neg":" The mapping drone is below the woodland track."},
{"context":"A suspended lens and a production bench are visible, with no vertical relation specified.","pos":" The suspended lens is above the production bench.","neg":" The suspended lens is below the production bench."},
{"context":"An observation balloon and a concrete bridge are visible, with no vertical relation specified.","pos":" The observation balloon is above the concrete bridge.","neg":" The observation balloon is below the concrete bridge."},
{"context":"A hanging detector and a research table are visible, with no vertical relation specified.","pos":" The hanging detector is above the research table.","neg":" The hanging detector is below the research table."},
{"context":"An eagle and a narrow trail are visible, with no vertical relation specified.","pos":" The eagle is above the narrow trail.","neg":" The eagle is below the narrow trail."},
{"context":"A warning beacon and a transport deck are visible, with no vertical relation specified.","pos":" The warning beacon is above the transport deck.","neg":" The warning beacon is below the transport deck."},
{"context":"A sailplane and a country road are visible, with no vertical relation specified.","pos":" The sailplane is above the country road.","neg":" The sailplane is below the country road."},
{"context":"A hanging display and an office desk are visible, with no vertical relation specified.","pos":" The hanging display is above the office desk.","neg":" The hanging display is below the office desk."}]
FINAL_NEUTRAL=[
("A ceramic bottle rests beside a folded brochure."," The objects remain stationary."),
("A wooden stylus lies beside a closed ledger."," The objects remain stationary."),
("A glass plate rests next to an empty container."," The objects remain stationary."),
("A metal token lies beside a paper envelope."," The objects remain stationary."),
("A cloth pouch rests next to a plastic holder."," The objects remain stationary."),
("A plain mug sits beside a closed handbook."," The objects remain stationary."),
("A small carton stands beside a clean towel."," The objects remain stationary."),
("A steel fork lies beside an empty bowl."," The objects remain stationary.")]

# ======================================================================================================================
# MASK CONSTRUCTION
# Only TEST173 parent region L00-L04 is searched.
# ======================================================================================================================
def hybrid(mask):
    s=set(mask);return torch.stack([SAFE[L] if L in s else CAUSAL_ORTH[L] for L in range(N_LAYERS)]).contiguous()

SINGLE_MASKS={f"SAFE_L{L:02d}":[L] for L in PARENT_REGION}
PAIR_MASKS={}
for i in range(len(PARENT_REGION)):
    for j in range(i+1,len(PARENT_REGION)):
        a,b=PARENT_REGION[i],PARENT_REGION[j];PAIR_MASKS[f"SAFE_L{a:02d}_L{b:02d}"]=[a,b]

MASKS={"ORTH_ALL":[],"SAFE_PARENT":PARENT_REGION,"SAFE_ALL":list(range(20)),**SINGLE_MASKS,**PAIR_MASKS}
VECTORS={k:hybrid(v) for k,v in MASKS.items()}
print("\n[8/24] Building TEST174 micro-localization arms...")
print("       Parent region frozen from TEST173:",PARENT_REGION)
print("       Singles:",list(SINGLE_MASKS))
print("       Pairs available only if no single qualifies:",list(PAIR_MASKS))

# ======================================================================================================================
# SEASC HOOKS / ASSAYS — TEST173 UNCHANGED
# ======================================================================================================================
def blank_tel():return [{"requested":[],"realized":[]} for _ in range(N_LAYERS)]
def hooks(vectors,sign,scale,tel):
    hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(module,args,out):
                old=out[0] if isinstance(out,tuple) else out;rest=out[1:] if isinstance(out,tuple) else None;B=old.shape[0]
                direction=(float(sign)*vectors[li][None,:].expand(B,-1)).float().contiguous();req=float(RHO[li])*float(scale);dose=torch.full((B,),req,device=DEVICE,dtype=torch.float32)
                oldlast=old[:,-1,:].float();new=seasc_ext.seasc_batch(old,direction,dose);newlast=new[:,-1,:].float()
                real=(newlast-oldlast).norm(dim=-1)/oldlast.norm(dim=-1).clamp_min(EPS)
                tel[li]["requested"].extend(dose.detach().cpu().tolist());tel[li]["realized"].extend(real.detach().cpu().tolist())
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
    return {"method":name,"scale":scale,"KL":float(np.mean([x[1] for x in z])),"abs_dNLL":float(np.mean([abs(x[0]) for x in z])),"kl_items":[float(x[1]) for x in z],"ret_items":[float(abs(x[0])) for x in z]}

def assay(name,v,items_c,items_n):
    a=causal_arm(name,v,PRIMARY_SCALE,items_c);n=neutral_arm(name,v,PRIMARY_SCALE,items_n)
    return {**a,**{k:v for k,v in n.items() if k not in ("method","scale")}}

# ======================================================================================================================
# DEV BASELINES
# ======================================================================================================================
print("\n[9/24] DEV baselines @0.50...")
DEV=[]
for name in ["ORTH_ALL","SAFE_PARENT","SAFE_ALL"]:
    r=assay(name,VECTORS[name],DEV_CAUSAL,DEV_NEUTRAL);DEV.append(r)
    print(f"       {name:<12} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")
def drow(name):return next(x for x in DEV if x["method"]==name)
O=drow("ORTH_ALL");P=drow("SAFE_PARENT")

# ======================================================================================================================
# STAGE 1 — SINGLE-LAYER SEARCH
# ======================================================================================================================
print("\n[10/24] DEV single-layer localization...")
for name in SINGLE_MASKS:
    r=assay(name,VECTORS[name],DEV_CAUSAL,DEV_NEUTRAL);DEV.append(r)
    print(f"       {name:<12} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")

# ======================================================================================================================
# PREDECLARED QUALIFICATION
# Candidate must preserve causal signal and improve BOTH collateral measures relative to ORTH.
# 5% KL improvement retained from TEST173 criterion.
# Retention only required not to worsen (>10%) and preferably improve.
# ======================================================================================================================
print("\n[11/24] Single-layer qualification...")
def qualify(name):
    r=drow(name);cr=r["Csym"]/max(O["Csym"],EPS);kr=r["KL"]/max(O["KL"],EPS);rr=r["abs_dNLL"]/max(O["abs_dNLL"],EPS)
    viable=r["Csym"]>=.90*O["Csym"] and r["positive"]>=7
    kl5=kr<=.95;retok=rr<=1.10
    burden=.60*kr+.25*rr+.15*(1/max(cr,EPS))
    return {"method":name,"layers":MASKS[name],"n_layers":len(MASKS[name]),"viable":viable,"KL5":kl5,"retention_ok":retok,"causal_ratio":cr,"KL_ratio":kr,"ret_ratio":rr,"burden":burden,"qualified":viable and kl5 and retok}
SINGLE_SCORE=[qualify(x) for x in SINGLE_MASKS]
for z in SINGLE_SCORE:
    print(f"       {z['method']:<12} Q={z['qualified']} Cx={z['causal_ratio']:.4f} KLx={z['KL_ratio']:.4f} Rx={z['ret_ratio']:.4f} burden={z['burden']:.4f}")

# ======================================================================================================================
# STAGE 2 — PAIRS ONLY IF NO SINGLE QUALIFIES
# This avoids searching 15 candidates and then retrospectively preferring complexity.
# ======================================================================================================================
SINGLE_OK=[x for x in SINGLE_SCORE if x["qualified"]]
PAIR_SCORE=[]
if SINGLE_OK:
    print("\n[12/24] Pair search skipped — at least one single layer qualifies.")
else:
    print("\n[12/24] No single qualified — evaluating predeclared pairs...")
    for name in PAIR_MASKS:
        r=assay(name,VECTORS[name],DEV_CAUSAL,DEV_NEUTRAL);DEV.append(r)
        print(f"       {name:<12} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")
    PAIR_SCORE=[qualify(x) for x in PAIR_MASKS]
    for z in PAIR_SCORE:
        print(f"       {z['method']:<12} Q={z['qualified']} Cx={z['causal_ratio']:.4f} KLx={z['KL_ratio']:.4f} Rx={z['ret_ratio']:.4f} burden={z['burden']:.4f}")

# ======================================================================================================================
# FREEZE
# Hierarchy:
# 1 qualifying single -> lowest burden
# 2 else qualifying pair -> lowest burden
# 3 else TEST173 parent L00-L04 fallback
# ======================================================================================================================
print("\n[13/24] Freezing exact source-layer mask...")
if SINGLE_OK:
    WIN=min(SINGLE_OK,key=lambda x:x["burden"]);FROZEN_MASK=WIN["method"];SEARCH_LEVEL="SINGLE";FALLBACK=False
else:
    PAIR_OK=[x for x in PAIR_SCORE if x["qualified"]]
    if PAIR_OK:
        WIN=min(PAIR_OK,key=lambda x:x["burden"]);FROZEN_MASK=WIN["method"];SEARCH_LEVEL="PAIR";FALLBACK=False
    else:FROZEN_MASK="SAFE_PARENT";SEARCH_LEVEL="PARENT_FALLBACK";FALLBACK=True
FROZEN_LAYERS=MASKS[FROZEN_MASK];FROZEN_VECTOR=VECTORS[FROZEN_MASK]
print("       FROZEN:",FROZEN_MASK);print("       layers:",FROZEN_LAYERS);print("       search level:",SEARCH_LEVEL);print("       fallback:",FALLBACK);print("       FINAL remains unopened.")

# ======================================================================================================================
# DEV LOO ROBUSTNESS OF FROZEN MASK
# ======================================================================================================================
print("\n[14/24] DEV leave-one-item-out causal robustness...")
R=drow(FROZEN_MASK);LOO=[];oi=np.asarray(O["items"]);ri=np.asarray(R["items"])
for i in range(len(oi)):
    ix=np.arange(len(oi))!=i;rc=float(ri[ix].mean());oc=float(oi[ix].mean());ratio=rc/max(oc,EPS)
    LOO.append({"held_out":i,"candidate_C":rc,"orth_C":oc,"causal_ratio":ratio})
    print(f"       hold {i+1}/8 C={rc:+.6f} ORTH={oc:+.6f} ratio={ratio:.4f}")
LOO_MIN=min(x["causal_ratio"] for x in LOO);print(f"       minimum LOO causal ratio={LOO_MIN:.4f}")

# ======================================================================================================================
# MICRO GEOMETRY — L00-L04
# ======================================================================================================================
print("\n[15/24] Early-layer micro geometry...")
MICRO=[]
for L in PARENT_REGION:
    u=CAUSAL_ORTH[L];s=SAFE[L];qc=QCOLL[L];removed=u-s
    z={"layer":L,"selected":int(L in FROZEN_LAYERS),"rho":float(RHO[L]),"raw_coll_energy":GEOM[L]["raw_coll_energy"],"removed_norm":float(removed.norm()),"cos_safe_orth":float(torch.dot(s,u)),"orth_collQ":float((qc.T@u).norm()),"safe_collQ":float((qc.T@s).norm())}
    MICRO.append(z)
    print(f"       L{L:02d} selected={z['selected']} rho={z['rho']:.6f} collEnergy={z['raw_coll_energy']:.6f} vecΔ={z['removed_norm']:.6f} CollQ O/S={z['orth_collQ']:.6f}/{z['safe_collQ']:.6f}")

# ======================================================================================================================
# FINAL OPEN — MASK ALREADY FROZEN
# ======================================================================================================================
print("\n[16/24] FINAL opened — frozen micro-mask evaluation...")
FINAL=[]
for scale in SCALES:
    print(f"\n       DOSE={scale:.3f}")
    for name,v in [("ORTH_ALL",VECTORS["ORTH_ALL"]),("SAFE_PARENT",VECTORS["SAFE_PARENT"]),("SAFE_ALL",VECTORS["SAFE_ALL"]),("FROZEN_MICRO",FROZEN_VECTOR)]:
        a=causal_arm(name,v,scale,FINAL_CAUSAL);n=neutral_arm(name,v,scale,FINAL_NEUTRAL)
        r={**a,**{k:v for k,v in n.items() if k not in ("method","scale")}};FINAL.append(r)
        print(f"       {name:<13} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")
def frow(name,scale):return next(x for x in FINAL if x["method"]==name and abs(x["scale"]-scale)<1e-9)

# ======================================================================================================================
# FINAL CAUSAL BOOTSTRAP
# ======================================================================================================================
print("\n[17/24] FINAL paired causal bootstrap...")
def bootdiff(a,b,n=20000,seed=174):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n,dtype=np.float64)
    for i in range(n):
        ix=rng.integers(0,len(d),len(d));z[i]=d[ix].mean()
    return {"delta":float(d.mean()),"ci_low":float(np.quantile(z,.025)),"ci_high":float(np.quantile(z,.975))}
BOOT=[]
for s in SCALES:
    m,o,p=frow("FROZEN_MICRO",s),frow("ORTH_ALL",s),frow("SAFE_PARENT",s)
    a=bootdiff(m["items"],o["items"],seed=SEED+int(s*1000));b=bootdiff(m["items"],p["items"],seed=SEED+5000+int(s*1000))
    BOOT+=[{"scale":s,"comparison":"MICRO_MINUS_ORTH",**a},{"scale":s,"comparison":"MICRO_MINUS_PARENT",**b}]
    print(f"       s={s:.3f} M-O={a['delta']:+.6f}[{a['ci_low']:+.6f},{a['ci_high']:+.6f}] | M-P={b['delta']:+.6f}[{b['ci_low']:+.6f},{b['ci_high']:+.6f}]")

# ======================================================================================================================
# FINAL NEUTRAL BOOTSTRAP
# ======================================================================================================================
print("\n[18/24] FINAL neutral bootstrap...")
NBOOT=[]
for s in SCALES:
    m,o,p=frow("FROZEN_MICRO",s),frow("ORTH_ALL",s),frow("SAFE_PARENT",s)
    for metric,key in [("KL","kl_items"),("RET","ret_items")]:
        a=bootdiff(m[key],o[key],seed=SEED+10000+int(s*1000)+(0 if metric=="KL" else 1))
        b=bootdiff(m[key],p[key],seed=SEED+20000+int(s*1000)+(0 if metric=="KL" else 1))
        NBOOT+=[{"scale":s,"metric":metric,"comparison":"MICRO_MINUS_ORTH",**a},{"scale":s,"metric":metric,"comparison":"MICRO_MINUS_PARENT",**b}]
        print(f"       s={s:.3f} {metric:<3} M-O={a['delta']:+.6f}[{a['ci_low']:+.6f},{a['ci_high']:+.6f}] | M-P={b['delta']:+.6f}[{b['ci_low']:+.6f},{b['ci_high']:+.6f}]")

# ======================================================================================================================
# PRIMARY FINAL
# ======================================================================================================================
print("\n[19/24] Primary FINAL @0.50...")
PRIMARY=[]
for name in ["ORTH_ALL","SAFE_PARENT","SAFE_ALL","FROZEN_MICRO"]:
    r=frow(name,.5);z={"method":name,"C":r["Csym"],"positive":r["positive"],"KL":r["KL"],"abs_dNLL":r["abs_dNLL"]};PRIMARY.append(z)
    print(f"       {name:<13} C={z['C']:+.6f} P+={z['positive']}/8 KL={z['KL']:.6f} |dNLL|={z['abs_dNLL']:.6f}")
PO=next(x for x in PRIMARY if x["method"]=="ORTH_ALL");PP=next(x for x in PRIMARY if x["method"]=="SAFE_PARENT");PM=next(x for x in PRIMARY if x["method"]=="FROZEN_MICRO")
print(f"       MICRO/ORTH  causal={PM['C']/max(PO['C'],EPS):.4f} KL={PM['KL']/max(PO['KL'],EPS):.4f} Ret={PM['abs_dNLL']/max(PO['abs_dNLL'],EPS):.4f}")
print(f"       MICRO/PARENT causal={PM['C']/max(PP['C'],EPS):.4f} KL={PM['KL']/max(PP['KL'],EPS):.4f} Ret={PM['abs_dNLL']/max(PP['abs_dNLL'],EPS):.4f}")

# ======================================================================================================================
# DEV SINGLE-LAYER FINGERPRINT
# ======================================================================================================================
print("\n[20/24] Single-layer DEV fingerprint...")
FINGER=[]
for name in SINGLE_MASKS:
    r=drow(name);L=MASKS[name][0]
    z={"method":name,"layer":L,"C":r["Csym"],"causal_delta":r["Csym"]-O["Csym"],"KL":r["KL"],"KL_reduction":O["KL"]-r["KL"],"abs_dNLL":r["abs_dNLL"],"ret_reduction":O["abs_dNLL"]-r["abs_dNLL"],"raw_coll_energy":GEOM[L]["raw_coll_energy"],"rho":float(RHO[L])};FINGER.append(z)
    print(f"       L{L:02d} ΔC={z['causal_delta']:+.6f} KLgain={z['KL_reduction']:+.6f} RetGain={z['ret_reduction']:+.6f} collEnergy={z['raw_coll_energy']:.6f} rho={z['rho']:.6f}")

# ======================================================================================================================
# DECISION
# ======================================================================================================================
print("\n[21/24] TEST174 decision...")
cb=next(x for x in BOOT if x["scale"]==.5 and x["comparison"]=="MICRO_MINUS_ORTH")
kb=next(x for x in NBOOT if x["scale"]==.5 and x["metric"]=="KL" and x["comparison"]=="MICRO_MINUS_ORTH")
rb=next(x for x in NBOOT if x["scale"]==.5 and x["metric"]=="RET" and x["comparison"]=="MICRO_MINUS_ORTH")
micro_reduced=len(FROZEN_LAYERS)<len(PARENT_REGION)
causal_preserved=PM["C"]>=.90*PO["C"] and PM["positive"]>=7
kl_better=PM["KL"]<PO["KL"];ret_better=PM["abs_dNLL"]<PO["abs_dNLL"]
kl_supported=kb["ci_high"]<0;ret_supported=rb["ci_high"]<0
if micro_reduced and causal_preserved and kl_better and kl_supported:
    DECISION="EARLY_COLLATERAL_SOURCE_MICRO_LOCALIZED"
elif micro_reduced and causal_preserved and kl_better:
    DECISION="EARLY_MICRO_LOCALIZATION_SUPPORTED_BY_POINT_ESTIMATE"
elif FALLBACK and causal_preserved and kl_better:
    DECISION="TEST173_L00_L04_REGION_REMAINS_MINIMAL"
else:
    DECISION="EARLY_MICRO_LOCALIZATION_NOT_CONFIRMED"
print("       reduced below TEST173 5-layer parent:",micro_reduced)
print("       FINAL causal >=90% ORTH:",causal_preserved)
print("       FINAL KL < ORTH:",kl_better)
print("       FINAL retention disturbance < ORTH:",ret_better)
print("       primary KL bootstrap upper < 0:",kl_supported)
print("       primary retention bootstrap upper < 0:",ret_supported)
print("       primary causal MICRO-ORTH:",f"{cb['delta']:+.6f}[{cb['ci_low']:+.6f},{cb['ci_high']:+.6f}]")
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
savecsv("VECTOR_GEOMETRY.csv",GEOM);savecsv("DEV.csv",DEV);savecsv("SINGLE_SCORE.csv",SINGLE_SCORE)
savecsv("PAIR_SCORE.csv",PAIR_SCORE);savecsv("LOO.csv",LOO);savecsv("MICRO_GEOMETRY.csv",MICRO)
savecsv("FINAL.csv",FINAL);savecsv("CAUSAL_BOOTSTRAP.csv",BOOT);savecsv("NEUTRAL_BOOTSTRAP.csv",NBOOT)
savecsv("PRIMARY.csv",PRIMARY);savecsv("SINGLE_LAYER_FINGERPRINT.csv",FINGER)
np.savez_compressed(ROOT/"TEST174_EARLY_SOURCE_LAYER_MICRO_LOCALIZATION.npz",
causal_orth=CAUSAL_ORTH.detach().cpu().numpy(),causal_safe=SAFE.detach().cpu().numpy(),
frozen_micro=FROZEN_VECTOR.detach().cpu().numpy(),frozen_layers=np.asarray(FROZEN_LAYERS,dtype=np.int32),
fisher_guard=FISHER_GUARD.detach().cpu().numpy(),rho=np.asarray(RHO,dtype=np.float32))
REPORT={
"test":"TEST 174","title":"EARLY SOURCE-LAYER MICRO-LOCALIZATION","model":MODEL_ID,"seed":SEED,
"frozen_k":FROZEN_K,"parent_region":PARENT_REGION,"primary_scale":PRIMARY_SCALE,"rss":RSS,
"frozen_mask":FROZEN_MASK,"frozen_layers":FROZEN_LAYERS,"search_level":SEARCH_LEVEL,"fallback":FALLBACK,
"decision":DECISION,"dev":DEV,"single_score":SINGLE_SCORE,"pair_score":PAIR_SCORE,"loo":LOO,
"micro_geometry":MICRO,"final":FINAL,"causal_bootstrap":BOOT,"neutral_bootstrap":NBOOT,
"primary":PRIMARY,"fingerprint":FINGER,
"contract":[
"TEST174 searches only inside the TEST173 frozen parent region L00-L04.",
"TEST169 CAUSAL_ORTH reconstruction is unchanged.",
"TEST171 collateral basis construction and K=8 are unchanged.",
"All L0-L19 SEASC injections remain active with the locked envelope and dose.",
"Single-layer candidates are evaluated first.",
"Pair candidates are evaluated only if no single-layer candidate satisfies the predeclared DEV criteria.",
"The exact source-layer mask is frozen before TEST174 FINAL is opened.",
"TEST171, TEST172 and TEST173 FINAL results are not used for TEST174 selection.",
"No envelope search, runtime controller, gate, brake, cap, saturation, boost or model-weight modification is introduced.",
"A positive result localizes useful collateral cleanup within the early source region; it does not establish that unselected layers have zero collateral influence."
]}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)

# ======================================================================================================================
# INTEGRITY
# ======================================================================================================================
print("\n[23/24] Integrity...")
print("       P8 ranks:",RANKS);print("       Frozen collateral K:",FROZEN_K)
print("       TEST173 parent region:",PARENT_REGION);print("       Frozen TEST174 mask:",FROZEN_MASK)
print("       Frozen source layers:",FROZEN_LAYERS);print("       Search level:",SEARCH_LEVEL)
print("       Mask frozen before FINAL: YES");print("       TEST171/172/173 FINAL used for selection: NO")
print("       SEASC envelope changed: NO");print(f"       RSS dose budget: {RSS:.9f}")
print("       All L0-L19 injections active: YES");print("       Model weights modified: NO");print("       Runtime controller: OFF")

# ======================================================================================================================
# COMPLETE
# ======================================================================================================================
print("\n[24/24] COMPLETE")
print("="*154);print("TEST 174 — EARLY SOURCE-LAYER MICRO-LOCALIZATION — SUMMARY");print("="*154)
print(f"Mean cos(Fisher,Bank): {np.mean(COS):+.6f}")
print(f"Mean raw collateral energy in CAUSAL_ORTH: {np.mean([x['raw_coll_energy'] for x in GEOM]):.6f}")
print("TEST173 parent:",PARENT_REGION);print("Frozen TEST174 mask:",FROZEN_MASK);print("Frozen layers:",FROZEN_LAYERS)
print("Search level:",SEARCH_LEVEL)
print(f"Primary FINAL ORTH   : C={PO['C']:+.6f} KL={PO['KL']:.6f} |dNLL|={PO['abs_dNLL']:.6f}")
print(f"Primary FINAL PARENT : C={PP['C']:+.6f} KL={PP['KL']:.6f} |dNLL|={PP['abs_dNLL']:.6f}")
print(f"Primary FINAL MICRO  : C={PM['C']:+.6f} KL={PM['KL']:.6f} |dNLL|={PM['abs_dNLL']:.6f}")
print("DECISION:",DECISION);print("Saved:",ROOT);print("="*154)
