# ======================================================================================================================
# TEST 176 — THIRD-CONCEPT CAUSAL VECTOR FORGE
# AkbasCore 3.2 / SEASC
# INDEPENDENT INSIDE↔OUTSIDE REPLICATION OF TEST175 PIPELINE
#
# QUESTION:
# Can the frozen causal-vector construction recipe generalize to a THIRD
# semantic axis, INSIDE↔OUTSIDE, without reusing A/B-axis steering vectors?
#
# LOCKS:
# - Qwen/Qwen2.5-7B-Instruct BF16 / SDPA / A100
# - L0-L19, locked SEASC envelope
# - Same causal-gradient forge logic
# - Same P8 retention-null logic
# - Same weighted collateral-gradient logic
# - K=8 frozen
# - L00-L04 early cleanup transferred WITHOUT C-axis search
# - C-axis vector built independently from INSIDE↔OUTSIDE families
# - DEV diagnostic only; no layer/rank/envelope search
# - FINAL opened only after C_CAUSAL_ORTH / C_SAFE / C_EARLY_SAFE frozen
# - No ABOVE↔BELOW or BEFORE↔AFTER steering vector used
# - No controller / gate / brake / cap / boost / weight modification
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

print("="*154);print("TEST 176 — THIRD-CONCEPT CAUSAL VECTOR FORGE");print("INDEPENDENT INSIDE↔OUTSIDE REPLICATION — FROZEN TEST175 RECIPE");print("="*154)
print("Ninja:",ninja.__version__)
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=176
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
P8_COUNT=8;FROZEN_K=8;SVD_REL_TOL=1e-5;PRIMARY_SCALE=.50
EARLY_SAFE=[0,1,2,3,4];SCALES=[.125,.25,.50,.75,1.]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST176_THIRD_CONCEPT_CAUSAL_VECTOR_FORGE");ROOT.mkdir(parents=True,exist_ok=True)
print("GPU:",torch.cuda.get_device_name(0));print("Model:",MODEL_ID)
print("Target axis: INSIDE↔OUTSIDE | L0-L19 | K=8 | frozen early-safe=L00-L04")
def env(L):
    x=ZIRVE*math.exp(-SONUM*float(L))*(1+SONUM*float(L))+TABAN
    return x/(ZIRVE+TABAN)
ENVELOPE=np.asarray([env(L) for L in range(N_LAYERS)],dtype=np.float64);RHO=IVME*ENVELOPE;RSS=float(np.sqrt(np.sum(RHO**2)))
print(f"Locked SEASC RSS dose budget: {RSS:.9f}")

# ======================================================================================================================
# CUDA SEASC — TEST175 PHYSICS UNCHANGED
# ======================================================================================================================
BUILD="/tmp/akbascore_test176";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST176 SEASC");}"""
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
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test176_seasc",[&]{k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/24] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test176_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
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
# C-AXIS INSIDE↔OUTSIDE FISHER/BANK — INDEPENDENT
# ======================================================================================================================
SEARCH_PAIRS=[
("The key is inside the wooden box.","The key is outside the wooden box."),
("The coin is inside the glass jar.","The coin is outside the glass jar."),
("The ball is inside the fabric bag.","The ball is outside the fabric bag."),
("The card is inside the paper envelope.","The card is outside the paper envelope."),
("The tool is inside the metal cabinet.","The tool is outside the metal cabinet."),
("The toy is inside the storage basket.","The toy is outside the storage basket."),
("The bottle is inside the refrigerator.","The bottle is outside the refrigerator."),
("The notebook is inside the desk drawer.","The notebook is outside the desk drawer."),
("The package is inside the delivery crate.","The package is outside the delivery crate."),
("The apple is inside the shopping bag.","The apple is outside the shopping bag."),
("The document is inside the blue folder.","The document is outside the blue folder."),
("The cable is inside the equipment case.","The cable is outside the equipment case."),
("The brush is inside the plastic container.","The brush is outside the plastic container."),
("The camera is inside the travel case.","The camera is outside the travel case."),
("The book is inside the cardboard carton.","The book is outside the cardboard carton."),
("The sensor is inside the protective housing.","The sensor is outside the protective housing.")]

@torch.inference_mode()
def capture(text):
    e=tokenizer(chat(text),return_tensors="pt").to(DEVICE);o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);pos=int(e["attention_mask"][0].sum())-1
    hs=[o.hidden_states[L+1][0,pos].float().detach() for L in range(N_LAYERS)];del e,o;return hs

print("\n[3/24] Building independent INSIDE↔OUTSIDE Fisher / Bank geometry...")
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
# SHARED P8 RETENTION BASIS — RECIPE UNCHANGED
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
# C-AXIS CAUSAL FORGE — SAME FOUR-FAMILY RECIPE, THIRD SEMANTIC AXIS
# ======================================================================================================================
FORGE={
"A":[
{"context":"A key and a wooden box are present, but their containment relation is unspecified.","pos":" The key is inside the wooden box.","neg":" The key is outside the wooden box."},
{"context":"A coin and a glass jar are present, but their containment relation is unspecified.","pos":" The coin is inside the glass jar.","neg":" The coin is outside the glass jar."},
{"context":"A ball and a fabric bag are present, but their containment relation is unspecified.","pos":" The ball is inside the fabric bag.","neg":" The ball is outside the fabric bag."},
{"context":"A card and a paper envelope are present, but their containment relation is unspecified.","pos":" The card is inside the paper envelope.","neg":" The card is outside the paper envelope."}],
"B":[
{"context":"A tool and a cabinet are present. Their containment relation is unspecified.","pos":" The tool is contained within the cabinet.","neg":" The tool is located outside the cabinet."},
{"context":"A toy and a basket are present. Their containment relation is unspecified.","pos":" The toy is contained within the basket.","neg":" The toy is located outside the basket."},
{"context":"A bottle and a refrigerator are present. Their containment relation is unspecified.","pos":" The bottle is contained within the refrigerator.","neg":" The bottle is located outside the refrigerator."},
{"context":"A notebook and a drawer are present. Their containment relation is unspecified.","pos":" The notebook is contained within the drawer.","neg":" The notebook is located outside the drawer."}],
"C":[
{"context":"Consider a package relative to the boundary of a delivery crate.","pos":" The package occupies the interior region of the crate.","neg":" The package occupies the exterior region of the crate."},
{"context":"Consider an apple relative to the boundary of a shopping bag.","pos":" The apple occupies the interior region of the bag.","neg":" The apple occupies the exterior region of the bag."},
{"context":"Consider a document relative to the boundary of a folder.","pos":" The document occupies the interior region of the folder.","neg":" The document occupies the exterior region of the folder."},
{"context":"Consider a cable relative to the boundary of an equipment case.","pos":" The cable occupies the interior region of the case.","neg":" The cable occupies the exterior region of the case."}],
"D":[
{"context":"A brush and a container are present. No containment state is given.","pos":" Spatially, the container encloses the brush.","neg":" Spatially, the brush remains outside the container."},
{"context":"A camera and a travel case are present. No containment state is given.","pos":" Spatially, the travel case encloses the camera.","neg":" Spatially, the camera remains outside the travel case."},
{"context":"A book and a carton are present. No containment state is given.","pos":" Spatially, the carton encloses the book.","neg":" Spatially, the book remains outside the carton."},
{"context":"A sensor and a protective housing are present. No containment state is given.","pos":" Spatially, the housing encloses the sensor.","neg":" Spatially, the sensor remains outside the housing."}]}

print("\n[5/24] Forging independent INSIDE↔OUTSIDE causal direction...")
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
    if rn<EPS:raise RuntimeError(f"C_CAUSAL_ORTH collapsed L{L}")
    CO.append(r/rn);ORTH_FRAC.append(float(rn))
C_CAUSAL_ORTH=torch.stack(CO).contiguous()
print(f"       mean cos(Causal,C-Fisher)={np.mean([float(torch.dot(CAUSAL[L],FISHER[L])) for L in range(N_LAYERS)]):+.6f}")
print(f"       mean C-Fisher-orthogonal component={np.mean(ORTH_FRAC):.6f}")
print(f"       mean P8 overlap={np.mean([float((Q8[L].T@C_CAUSAL_ORTH[L]).norm()) for L in range(N_LAYERS)]):.8e}")

# ======================================================================================================================
# COLLATERAL BASIS — SAME TEST171/175 RECIPE
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

print("\n[6/24] Rebuilding frozen K=8 collateral basis...")
COLL_G=[]
for i,(ctx,cont) in enumerate(COLL_BUILD,1):
    g,l=activation_gradient(ctx,cont,weighted=True);COLL_G.append(g)
    print(f"       neutral {i}/8 | NLL={l:.6f} | mean ||g_act||={np.mean([float(x.norm()) for x in g]):.6e}")
QCOLL=[]
for L in range(N_LAYERS):
    G=torch.stack([COLL_G[i][L] for i in range(8)]).float();G=G[G.norm(dim=1)>EPS];G=norm(G);_,s,Vh=torch.linalg.svd(G,full_matrices=False);QCOLL.append(Vh[:FROZEN_K].T.contiguous())
print("       Frozen collateral ranks:",[q.shape[1] for q in QCOLL])

# ======================================================================================================================
# C_CAUSAL_SAFE — SAME PROJECTION ORDER
# ======================================================================================================================
print("\n[7/24] Constructing C_CAUSAL_SAFE...")
SAFE=[];GEOM=[]
for L in range(N_LAYERS):
    u=C_CAUSAL_ORTH[L].float();qc=QCOLL[L];pc=qc@(qc.T@u);v=u-pc
    q=Q8[L];v=v-q@(q.T@v);f=FISHER_GUARD[L];v=v-torch.dot(v,f)*f;v=v-q@(q.T@v);v=norm(v)
    SAFE.append(v);GEOM.append({"layer":L,"raw_coll_energy":float(pc.square().sum()/u.square().sum().clamp_min(EPS)),"safe_cos_orth":float(torch.dot(v,u)),"removed_norm":float((u-v).norm()),"safe_collQ":float((qc.T@v).norm())})
C_CAUSAL_SAFE=torch.stack(SAFE).contiguous()
C_EARLY_SAFE=torch.stack([C_CAUSAL_SAFE[L] if L in EARLY_SAFE else C_CAUSAL_ORTH[L] for L in range(N_LAYERS)]).contiguous()
print(f"       mean raw collateral energy={np.mean([x['raw_coll_energy'] for x in GEOM]):.6f}")
print(f"       mean cos(SAFE,ORTH)={np.mean([x['safe_cos_orth'] for x in GEOM]):.6f}")
print(f"       mean ||ORTH-SAFE||={np.mean([x['removed_norm'] for x in GEOM]):.6f}")
print("       EARLY_SAFE frozen at L00-L04 BEFORE evaluation.")

# ======================================================================================================================
# NEW C-AXIS DEV / FINAL
# ======================================================================================================================
DEV_CAUSAL=[
{"context":"A token and a small pouch are present, but their containment relation is unspecified.","pos":" The token is inside the pouch.","neg":" The token is outside the pouch."},
{"context":"A lens and a protective case are present, but their containment relation is unspecified.","pos":" The lens is inside the protective case.","neg":" The lens is outside the protective case."},
{"context":"A battery and a storage compartment are present, but their containment relation is unspecified.","pos":" The battery is inside the compartment.","neg":" The battery is outside the compartment."},
{"context":"A photograph and a paper sleeve are present, but their containment relation is unspecified.","pos":" The photograph is inside the sleeve.","neg":" The photograph is outside the sleeve."},
{"context":"A parcel and a cargo bin are present, but their containment relation is unspecified.","pos":" The parcel is inside the cargo bin.","neg":" The parcel is outside the cargo bin."},
{"context":"A marker and a pencil case are present, but their containment relation is unspecified.","pos":" The marker is inside the pencil case.","neg":" The marker is outside the pencil case."},
{"context":"A sample and a laboratory container are present, but their containment relation is unspecified.","pos":" The sample is inside the container.","neg":" The sample is outside the container."},
{"context":"A memory card and a camera case are present, but their containment relation is unspecified.","pos":" The memory card is inside the camera case.","neg":" The memory card is outside the camera case."}]
DEV_NEUTRAL=[
("A porcelain mug rests beside a folded map."," The objects remain stationary."),
("A wooden pencil lies beside a sealed folder."," The objects remain stationary."),
("A glass bottle rests next to an empty dish."," The objects remain stationary."),
("A metal clip lies beside a cardboard card."," The objects remain stationary."),
("A fabric pouch rests next to a plastic case."," The objects remain stationary."),
("A plain saucer sits beside a closed journal."," The objects remain stationary."),
("A small carton stands beside a clean cloth."," The objects remain stationary."),
("A steel fork rests beside an empty plate."," The objects remain stationary.")]

FINAL_CAUSAL=[
{"context":"A badge and a leather wallet are present, but their containment relation is unspecified.","pos":" The badge is inside the wallet.","neg":" The badge is outside the wallet."},
{"context":"A filter and an equipment housing are present, but their containment relation is unspecified.","pos":" The filter is inside the equipment housing.","neg":" The filter is outside the equipment housing."},
{"context":"A medicine bottle and a travel bag are present, but their containment relation is unspecified.","pos":" The medicine bottle is inside the travel bag.","neg":" The medicine bottle is outside the travel bag."},
{"context":"A receipt and a document sleeve are present, but their containment relation is unspecified.","pos":" The receipt is inside the document sleeve.","neg":" The receipt is outside the document sleeve."},
{"context":"A component and a transport crate are present, but their containment relation is unspecified.","pos":" The component is inside the transport crate.","neg":" The component is outside the transport crate."},
{"context":"A pen and a carrying case are present, but their containment relation is unspecified.","pos":" The pen is inside the carrying case.","neg":" The pen is outside the carrying case."},
{"context":"A specimen and a laboratory vessel are present, but their containment relation is unspecified.","pos":" The specimen is inside the laboratory vessel.","neg":" The specimen is outside the laboratory vessel."},
{"context":"A memory device and a protective case are present, but their containment relation is unspecified.","pos":" The memory device is inside the protective case.","neg":" The memory device is outside the protective case."}]
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
# SEASC ASSAYS — TEST175 UNCHANGED
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

# ======================================================================================================================
# DEV — DIAGNOSTIC ONLY, NO SELECTION
# ======================================================================================================================
print("\n[8/24] C-axis vectors frozen. DEV diagnostic begins...")
ARMS={"C_ORTH":C_CAUSAL_ORTH,"C_EARLY_SAFE":C_EARLY_SAFE,"C_SAFE_ALL":C_CAUSAL_SAFE}
DEV=[]
for name,v in ARMS.items():
    a=causal_arm(name,v,.5,DEV_CAUSAL);n=neutral_arm(name,v,.5,DEV_NEUTRAL);r={**a,**{k:v for k,v in n.items() if k not in ("method","scale")}};DEV.append(r)
    print(f"       {name:<13} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")

# ======================================================================================================================
# FAMILY CONSENSUS GEOMETRY
# ======================================================================================================================
print("\n[9/24] Cross-family causal consensus...")
FAMILY_GEOM=[]
for a in FG:
    vals=[]
    for b in FG:
        if a<b:vals.append((b,float(torch.stack([torch.dot(FG[a][L],FG[b][L]) for L in range(N_LAYERS)]).mean())))
    for b,c in vals:FAMILY_GEOM.append({"a":a,"b":b,"mean_cos":c});print(f"       {a}-{b}: mean cos={c:+.6f}")

# ======================================================================================================================
# VECTOR GEOMETRY
# ======================================================================================================================
print("\n[10/24] C-axis vector geometry...")
for L in range(N_LAYERS):
    print(f"       L{L:02d} collE={GEOM[L]['raw_coll_energy']:.6f} cosSafeOrth={GEOM[L]['safe_cos_orth']:+.6f} early={int(L in EARLY_SAFE)}")

# ======================================================================================================================
# FREEZE DECLARATION — NO DEV-BASED CHOICE
# ======================================================================================================================
print("\n[11/24] Freeze declaration...")
print("       C_CAUSAL_ORTH: FROZEN")
print("       C_EARLY_SAFE : FROZEN L00-L04 transferred from prior result")
print("       C_SAFE_ALL   : reference arm")
print("       DEV selected nothing.")
print("       FINAL remains unopened.")

# ======================================================================================================================
# FINAL
# ======================================================================================================================
print("\n[12/24] FINAL opened...")
FINAL=[]
for scale in SCALES:
    print(f"\n       DOSE={scale:.3f}")
    for name,v in ARMS.items():
        a=causal_arm(name,v,scale,FINAL_CAUSAL);n=neutral_arm(name,v,scale,FINAL_NEUTRAL);r={**a,**{k:v for k,v in n.items() if k not in ("method","scale")}};FINAL.append(r)
        print(f"       {name:<13} C={r['Csym']:+.6f} P+={r['positive']}/8 KL={r['KL']:.6f} |dNLL|={r['abs_dNLL']:.6f}")
def frow(name,scale):return next(x for x in FINAL if x["method"]==name and abs(x["scale"]-scale)<1e-9)

# ======================================================================================================================
# BOOTSTRAP
# ======================================================================================================================
print("\n[13/24] FINAL causal bootstrap...")
def bootdiff(a,b,n=20000,seed=176):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n,dtype=np.float64)
    for i in range(n):
        ix=rng.integers(0,len(d),len(d));z[i]=d[ix].mean()
    return {"delta":float(d.mean()),"ci_low":float(np.quantile(z,.025)),"ci_high":float(np.quantile(z,.975))}
BOOT=[]
for s in SCALES:
    e,o,sa=frow("C_EARLY_SAFE",s),frow("C_ORTH",s),frow("C_SAFE_ALL",s)
    a=bootdiff(e["items"],o["items"],seed=SEED+int(s*1000));b=bootdiff(e["items"],sa["items"],seed=SEED+5000+int(s*1000))
    BOOT+=[{"scale":s,"comparison":"EARLY_MINUS_ORTH",**a},{"scale":s,"comparison":"EARLY_MINUS_SAFEALL",**b}]
    print(f"       s={s:.3f} E-O={a['delta']:+.6f}[{a['ci_low']:+.6f},{a['ci_high']:+.6f}] | E-S={b['delta']:+.6f}[{b['ci_low']:+.6f},{b['ci_high']:+.6f}]")

print("\n[14/24] FINAL neutral bootstrap...")
NBOOT=[]
for s in SCALES:
    e,o,sa=frow("C_EARLY_SAFE",s),frow("C_ORTH",s),frow("C_SAFE_ALL",s)
    for metric,key in [("KL","kl_items"),("RET","ret_items")]:
        a=bootdiff(e[key],o[key],seed=SEED+10000+int(s*1000)+(metric=="RET"))
        b=bootdiff(e[key],sa[key],seed=SEED+20000+int(s*1000)+(metric=="RET"))
        NBOOT+=[{"scale":s,"metric":metric,"comparison":"EARLY_MINUS_ORTH",**a},{"scale":s,"metric":metric,"comparison":"EARLY_MINUS_SAFEALL",**b}]
        print(f"       s={s:.3f} {metric:<3} E-O={a['delta']:+.6f}[{a['ci_low']:+.6f},{a['ci_high']:+.6f}] | E-S={b['delta']:+.6f}[{b['ci_low']:+.6f},{b['ci_high']:+.6f}]")

# ======================================================================================================================
# PRIMARY
# ======================================================================================================================
print("\n[15/24] Primary FINAL @0.50...")
PRIMARY=[]
for name in ARMS:
    r=frow(name,.5);z={"method":name,"C":r["Csym"],"positive":r["positive"],"KL":r["KL"],"abs_dNLL":r["abs_dNLL"]};PRIMARY.append(z)
    print(f"       {name:<13} C={z['C']:+.6f} P+={z['positive']}/8 KL={z['KL']:.6f} |dNLL|={z['abs_dNLL']:.6f}")
PO=next(x for x in PRIMARY if x["method"]=="C_ORTH");PE=next(x for x in PRIMARY if x["method"]=="C_EARLY_SAFE");PS=next(x for x in PRIMARY if x["method"]=="C_SAFE_ALL")
print(f"       EARLY/ORTH causal={PE['C']/max(PO['C'],EPS):.4f} KL={PE['KL']/max(PO['KL'],EPS):.4f} Ret={PE['abs_dNLL']/max(PO['abs_dNLL'],EPS):.4f}")

# ======================================================================================================================
# DOSE CONSISTENCY
# ======================================================================================================================
print("\n[16/24] Dose consistency...")
DOSE_SUM=[]
for name in ARMS:
    rr=[frow(name,s) for s in SCALES];pos=sum(x["Csym"]>0 for x in rr);pitems=sum(x["positive"] for x in rr)
    z={"method":name,"positive_doses":pos,"positive_items_total":pitems,"mean_C":float(np.mean([x["Csym"] for x in rr]))};DOSE_SUM.append(z)
    print(f"       {name:<13} positive doses={pos}/5 P+ total={pitems}/40 meanC={z['mean_C']:+.6f}")

# ======================================================================================================================
# EARLY-SAFE EFFECT PROFILE
# ======================================================================================================================
print("\n[17/24] Early-safe effect profile...")
PROFILE=[]
for s in SCALES:
    e,o=frow("C_EARLY_SAFE",s),frow("C_ORTH",s)
    z={"scale":s,"dC":e["Csym"]-o["Csym"],"dKL":e["KL"]-o["KL"],"dRet":e["abs_dNLL"]-o["abs_dNLL"]};PROFILE.append(z)
    print(f"       s={s:.3f} ΔC={z['dC']:+.6f} ΔKL={z['dKL']:+.6f} ΔRet={z['dRet']:+.6f}")

# ======================================================================================================================
# THIRD-CONCEPT REPLICATION CHECK
# ======================================================================================================================
print("\n[18/24] Third-concept replication check...")
cb=next(x for x in BOOT if x["scale"]==.5 and x["comparison"]=="EARLY_MINUS_ORTH")
kb=next(x for x in NBOOT if x["scale"]==.5 and x["metric"]=="KL" and x["comparison"]=="EARLY_MINUS_ORTH")
rb=next(x for x in NBOOT if x["scale"]==.5 and x["metric"]=="RET" and x["comparison"]=="EARLY_MINUS_ORTH")
causal_exists=PO["C"]>0 and PO["positive"]==8
early_causal=PE["C"]>0 and PE["positive"]==8
early_kl=PE["KL"]<PO["KL"];early_ret=PE["abs_dNLL"]<PO["abs_dNLL"]
print("       independent C causal vector positive 8/8:",causal_exists)
print("       C early-safe positive 8/8:",early_causal)
print("       C early-safe KL < C_ORTH:",early_kl)
print("       C early-safe retention disturbance < C_ORTH:",early_ret)
print("       primary ΔC CI:",f"[{cb['ci_low']:+.6f},{cb['ci_high']:+.6f}]")
print("       primary ΔKL CI:",f"[{kb['ci_low']:+.6f},{kb['ci_high']:+.6f}]")
print("       primary ΔRet CI:",f"[{rb['ci_low']:+.6f},{rb['ci_high']:+.6f}]")

# ======================================================================================================================
# DECISION
# ======================================================================================================================
print("\n[19/24] TEST176 decision...")
if causal_exists and early_causal and early_kl and early_ret and kb["ci_high"]<0:
    DECISION="THIRD_CONCEPT_CAUSAL_FORGE_REPLICATED__EARLY_CLEANING_SUPPORTED"
elif causal_exists and early_causal and (early_kl or early_ret):
    DECISION="THIRD_CONCEPT_CAUSAL_FORGE_REPLICATED__EARLY_CLEANING_PARTIAL"
elif causal_exists:
    DECISION="THIRD_CONCEPT_CAUSAL_FORGE_REPLICATED__CLEANING_NOT_REPLICATED"
else:
    DECISION="THIRD_CONCEPT_CAUSAL_FORGE_NOT_CONFIRMED"
print("       DECISION:",DECISION)

# ======================================================================================================================
# CONSERVATIVE INTERPRETATION FLAGS
# ======================================================================================================================
print("\n[20/24] Interpretation flags...")
FLAGS={
"third_semantic_axis":True,
"axis":"INSIDE_OUTSIDE",
"above_below_vector_used_for_C_construction":False,
"before_after_vector_used_for_C_construction":False,
"recipe_transferred":True,
"early_region_transferred_without_C_search":True,
"final_used_for_selection":False,
"universal_algorithm_proven":False}
for k,v in FLAGS.items():print(f"       {k}: {v}")

# ======================================================================================================================
# SAVE
# ======================================================================================================================
print("\n[21/24] Saving...")
def scalar(rows):return [{k:v for k,v in r.items() if not isinstance(v,(list,tuple,dict,np.ndarray))} for r in rows]
def savecsv(name,rows):
    rows=scalar(rows)
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("VECTOR_GEOMETRY.csv",GEOM);savecsv("FAMILY_GEOMETRY.csv",FAMILY_GEOM);savecsv("DEV.csv",DEV)
savecsv("FINAL.csv",FINAL);savecsv("CAUSAL_BOOTSTRAP.csv",BOOT);savecsv("NEUTRAL_BOOTSTRAP.csv",NBOOT)
savecsv("PRIMARY.csv",PRIMARY);savecsv("DOSE_SUMMARY.csv",DOSE_SUM);savecsv("EARLY_SAFE_PROFILE.csv",PROFILE)
np.savez_compressed(ROOT/"TEST176_THIRD_CONCEPT_CAUSAL_VECTOR_FORGE.npz",
c_fisher=FISHER.detach().cpu().numpy(),c_causal=C_CAUSAL_ORTH.detach().cpu().numpy(),
c_safe=C_CAUSAL_SAFE.detach().cpu().numpy(),c_early_safe=C_EARLY_SAFE.detach().cpu().numpy(),
rho=np.asarray(RHO,dtype=np.float32),early_safe=np.asarray(EARLY_SAFE,dtype=np.int32))
REPORT={"test":"TEST 176","title":"THIRD-CONCEPT CAUSAL VECTOR FORGE","axis":"INSIDE_OUTSIDE","model":MODEL_ID,"seed":SEED,
"frozen_k":FROZEN_K,"early_safe":EARLY_SAFE,"primary_scale":PRIMARY_SCALE,"rss":RSS,"decision":DECISION,
"geometry":GEOM,"family_geometry":FAMILY_GEOM,"dev":DEV,"final":FINAL,"causal_bootstrap":BOOT,
"neutral_bootstrap":NBOOT,"primary":PRIMARY,"dose_summary":DOSE_SUM,"profile":PROFILE,"flags":FLAGS,
"contract":[
"INSIDE↔OUTSIDE is constructed independently from new containment training families.",
"No ABOVE↔BELOW or BEFORE↔AFTER steering vector is used to construct or select the C-axis direction.",
"The causal-gradient forge recipe is transferred unchanged in structure.",
"P8 retention-null and weighted K=8 collateral construction are transferred unchanged.",
"L00-L04 early cleanup is transferred without C-axis layer search.",
"DEV is diagnostic only and selects no C-axis hyperparameter.",
"FINAL is opened only after C_CAUSAL_ORTH, C_CAUSAL_SAFE and C_EARLY_SAFE are frozen.",
"A positive result supports third-concept replication; it does not establish universal semantic generality."
]}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)

# ======================================================================================================================
# INTEGRITY
# ======================================================================================================================
print("\n[22/24] Integrity...")
print("       P8 ranks:",RANKS);print("       Frozen collateral K:",FROZEN_K)
print("       Semantic axis: INSIDE↔OUTSIDE")
print("       ABOVE↔BELOW vector used to build C-axis: NO")
print("       BEFORE↔AFTER vector used to build C-axis: NO")
print("       Early-safe region:",EARLY_SAFE,"(transferred, not searched)")
print("       DEV selected hyperparameters: NO");print("       FINAL used for selection: NO")
print("       SEASC envelope changed: NO");print(f"       RSS dose budget: {RSS:.9f}")
print("       All L0-L19 injections active: YES");print("       Model weights modified: NO");print("       Runtime controller: OFF")

# ======================================================================================================================
# SCIENTIFIC BOUNDARY
# ======================================================================================================================
print("\n[23/24] Scientific boundary...")
print("       This test evaluates independent replication on a third semantic axis.")
print("       Success strengthens cross-concept replication evidence.")
print("       It does not by itself establish universal semantic generality.")
print("       Early L00-L04 cleanup is transferred, not C-axis-selected.")

# ======================================================================================================================
# COMPLETE
# ======================================================================================================================
print("\n[24/24] COMPLETE")
print("="*154);print("TEST 176 — THIRD-CONCEPT CAUSAL VECTOR FORGE — SUMMARY");print("="*154)
print("Axis: INSIDE↔OUTSIDE")
print(f"Mean cos(C-Fisher,C-Bank): {np.mean(COS):+.6f}")
print(f"Mean causal Fisher-orthogonal component: {np.mean(ORTH_FRAC):.6f}")
print(f"Mean raw collateral energy: {np.mean([x['raw_coll_energy'] for x in GEOM]):.6f}")
print("Early-safe layers:",EARLY_SAFE)
print(f"Primary C_ORTH       : C={PO['C']:+.6f} KL={PO['KL']:.6f} |dNLL|={PO['abs_dNLL']:.6f}")
print(f"Primary C_EARLY_SAFE : C={PE['C']:+.6f} KL={PE['KL']:.6f} |dNLL|={PE['abs_dNLL']:.6f}")
print(f"Primary C_SAFE_ALL   : C={PS['C']:+.6f} KL={PS['KL']:.6f} |dNLL|={PS['abs_dNLL']:.6f}")
print("DECISION:",DECISION);print("Saved:",ROOT);print("="*154)
