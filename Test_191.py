# ================================================================================================================
# TEST 191 — CROSS-AXIS L27 COMMON TRANSPORT GEOMETRY
# AkbasCore 3.2 / SEASC — TEST190 continuation
#
# TEST190:
#   E/CAUSE↔EFFECT common L27 transport axis replicated across two independent holdouts.
#   BUILD mean cos(Δ,T27)=+0.895821
#   BUILD↔H1 mean=+0.973703 | BUILD↔H2 mean=+0.969289 | H1↔H2=+0.974272
#
# QUESTION:
#   Is the replicated L27 transport axis a general cross-axis property, or axis-specific?
#
# AXES:
#   A ABOVE↔BELOW
#   B BEFORE↔AFTER
#   C INSIDE↔OUTSIDE
#   D OPEN↔CLOSED
#   E CAUSE↔EFFECT
#
# METHOD:
#   For each axis independently:
#   1) build frozen layer-local compass from 8 POS/NEG pairs
#   2) apply identical SEASC L0-L19
#   3) measure symmetric PRE27→RAW27 Δ27 on BUILD prompts
#   4) freeze T27_axis = normalize(mean BUILD Δ27)
#   5) replicate on independent HOLDOUT
#   6) compare all T27 axes with a 5×5 cosine matrix
#
# NO PCA | NO K | NO BEHAVIOR GRADIENT | NO NEW VECTOR | NO FINAL | NO CONTROLLER | NO WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 191 — CROSS-AXIS L27 COMMON TRANSPORT GEOMETRY")
print("A/B/C/D/E — BUILD-FROZEN T27 AXES × INDEPENDENT HOLDOUT × 5×5 CROSS-AXIS MATRIX")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=191
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
PRIMARY_SCALE=.50;EPS=1e-10;N_BOOT=20000
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST191_CROSS_AXIS_L27_GEOMETRY");ROOT.mkdir(parents=True,exist_ok=True)

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
# [1/22] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test191";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
print("\n[1/22] CUDA...")
seasc_ext=load_inline(name="akbascore_test191_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD,verbose=False)
print("       OK")

# ================================================================================================================
# [2/22] MODEL
# ================================================================================================================
print("\n[2/22] Model...")
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
    return torch.stack([o.hidden_states[L+1][0,p].float().detach() for L in range(STEER_LAYERS)])

# ================================================================================================================
# [3/22] LOCKED AXIS DEFINITIONS
# ================================================================================================================
print("\n[3/22] Locked axis definitions...")
AXIS_PAIRS={
"A":[
("The lamp is above the table.","The lamp is below the table."),
("The bird is above the branch.","The bird is below the branch."),
("The sign is above the doorway.","The sign is below the doorway."),
("The shelf is above the cabinet.","The shelf is below the cabinet."),
("The cloud is above the hill.","The cloud is below the hill."),
("The clock is above the window.","The clock is below the window."),
("The bridge is above the river.","The bridge is below the river."),
("The picture is above the sofa.","The picture is below the sofa.")],
"B":[
("The alarm occurs before the evacuation.","The alarm occurs after the evacuation."),
("The flash occurs before the thunder.","The flash occurs after the thunder."),
("The ignition occurs before the motion.","The ignition occurs after the motion."),
("The warning occurs before the shutdown.","The warning occurs after the shutdown."),
("The sunrise occurs before noon.","The sunrise occurs after noon."),
("The opening occurs before the closing.","The opening occurs after the closing."),
("The question occurs before the answer.","The question occurs after the answer."),
("The launch occurs before the landing.","The launch occurs after the landing.")],
"C":[
("The key is inside the box.","The key is outside the box."),
("The coin is inside the jar.","The coin is outside the jar."),
("The book is inside the bag.","The book is outside the bag."),
("The ball is inside the basket.","The ball is outside the basket."),
("The tool is inside the drawer.","The tool is outside the drawer."),
("The letter is inside the envelope.","The letter is outside the envelope."),
("The bottle is inside the cabinet.","The bottle is outside the cabinet."),
("The card is inside the folder.","The card is outside the folder.")],
"D":[
("The door is open.","The door is closed."),
("The window is open.","The window is closed."),
("The gate is open.","The gate is closed."),
("The box is open.","The box is closed."),
("The drawer is open.","The drawer is closed."),
("The lid is open.","The lid is closed."),
("The cabinet is open.","The cabinet is closed."),
("The hatch is open.","The hatch is closed.")],
"E":[
("The spark is the cause of the fire.","The fire is the effect of the spark."),
("The impact is the cause of the crack.","The crack is the effect of the impact."),
("The rainfall is the cause of the flooding.","The flooding is the effect of the rainfall."),
("The heat is the cause of the melting.","The melting is the effect of the heat."),
("The collision is the cause of the damage.","The damage is the effect of the collision."),
("The pressure is the cause of the deformation.","The deformation is the effect of the pressure."),
("The infection is the cause of the fever.","The fever is the effect of the infection."),
("The vibration is the cause of the loosening.","The loosening is the effect of the vibration.")]}
AXIS_NAME={"A":"ABOVE↔BELOW","B":"BEFORE↔AFTER","C":"INSIDE↔OUTSIDE","D":"OPEN↔CLOSED","E":"CAUSE↔EFFECT"}
print("       A/B/C/D/E ready | 8 pairs each")

# ================================================================================================================
# [4/22] BUILD LAYER-LOCAL COMPASSES
# ================================================================================================================
print("\n[4/22] Layer-local compasses...")
COMPASS={}
for ax in "ABCDE":
    P=[];N=[]
    for i,(p,n) in enumerate(AXIS_PAIRS[ax],1):
        P.append(capture(p));N.append(capture(n))
    P=torch.stack(P);N=torch.stack(N)
    C=norm(P.mean(0)-N.mean(0)).contiguous()
    COMPASS[ax]=C
    print(f"       {ax} {AXIS_NAME[ax]} ready | cos L19 self=+1.000000")

# ================================================================================================================
# [5/22] CROSS-COMPASS L19 MATRIX
# ================================================================================================================
print("\n[5/22] L19 compass cosine matrix...")
AXES=list("ABCDE")
CM=np.zeros((5,5),dtype=np.float64)
for i,a in enumerate(AXES):
    for j,b in enumerate(AXES):CM[i,j]=float(torch.dot(COMPASS[a][19],COMPASS[b][19]))
print("             A         B         C         D         E")
for i,a in enumerate(AXES):print(f"       {a} "+" ".join(f"{CM[i,j]:+9.5f}" for j in range(5)))

# ================================================================================================================
# [6/22] FROZEN BUILD / HOLDOUT PROMPTS
# ================================================================================================================
print("\n[6/22] Frozen BUILD / HOLDOUT prompts...")
PROMPTS={
"A":{"B":[
"The red marker and blue marker are vertically separated.","Two lamps occupy different vertical positions.",
"A sign and a doorway have a vertical relation.","A bird and branch have a vertical relation.",
"A shelf and cabinet occupy different heights.","A cloud and hill have a vertical relation.",
"A clock and window occupy different heights.","A bridge and river have a vertical relation."],
"H":[
"A picture and sofa occupy different vertical positions.","A flag and roof have a vertical relation.",
"A drone and building occupy different heights.","A balloon and tree have a vertical relation.",
"A ceiling light and desk occupy different heights.","A kite and field have a vertical relation.",
"A camera and tripod base occupy different heights.","A mountain peak and valley have a vertical relation."]},
"B":{"B":[
"An alarm and evacuation occur at different times.","A flash and thunder occur in temporal sequence.",
"Ignition and motion occur in temporal sequence.","A warning and shutdown occur at different times.",
"Sunrise and noon occur in temporal sequence.","Opening and closing occur at different times.",
"A question and answer occur in temporal sequence.","Launch and landing occur at different times."],
"H":[
"A signal and response occur in temporal sequence.","A start and finish occur at different times.",
"Departure and arrival occur in temporal sequence.","A trigger and reaction occur at different times.",
"Heating and cooling occur in temporal sequence.","Entry and exit occur at different times.",
"Detection and reporting occur in temporal sequence.","Loading and unloading occur at different times."]},
"C":{"B":[
"A key and box have a containment relation.","A coin and jar have a containment relation.",
"A book and bag have a containment relation.","A ball and basket have a containment relation.",
"A tool and drawer have a containment relation.","A letter and envelope have a containment relation.",
"A bottle and cabinet have a containment relation.","A card and folder have a containment relation."],
"H":[
"A toy and container have a containment relation.","A document and case have a containment relation.",
"A cable and conduit have a containment relation.","A package and vehicle have a containment relation.",
"A component and housing have a containment relation.","A seed and shell have a containment relation.",
"A file and archive have a containment relation.","A capsule and chamber have a containment relation."]},
"D":{"B":[
"A door has an open-closed state.","A window has an open-closed state.",
"A gate has an open-closed state.","A box has an open-closed state.",
"A drawer has an open-closed state.","A lid has an open-closed state.",
"A cabinet has an open-closed state.","A hatch has an open-closed state."],
"H":[
"A valve has an open-closed state.","A shutter has an open-closed state.",
"A container has an open-closed state.","A cover has an open-closed state.",
"A barrier has an open-closed state.","A vent has an open-closed state.",
"A panel has an open-closed state.","A compartment has an open-closed state."]},
"E":{"B":[
"A fractured pipe and a pressure drop form a causal sequence.","A blocked air passage and rising temperature form a causal sequence.",
"A damaged seal and fluid loss form a causal sequence.","A current surge and a failed circuit form a causal sequence.",
"A stuck mechanism and restricted motion form a causal sequence.","A worn bearing and increased vibration form a causal sequence.",
"A clogged outlet and accumulated liquid form a causal sequence.","A strong impact and a deformed bracket form a causal sequence."],
"H":[
"A broken coupling and loss of transmitted motion form a causal sequence.","A blocked intake and reduced airflow form a causal sequence.",
"A damaged gasket and pressure leakage form a causal sequence.","A failed coolant pump and rising temperature form a causal sequence.",
"A seized bearing and restricted rotation form a causal sequence.","A severed cable and loss of electrical power form a causal sequence.",
"A clogged nozzle and reduced discharge form a causal sequence.","A fractured support and structural displacement form a causal sequence."]}}
print("       5 axes × BUILD8 × HOLDOUT8")

# ================================================================================================================
# [7/22] AXIS-SPECIFIC SEASC
# ================================================================================================================
print("\n[7/22] Axis-specific SEASC hooks...")
def seasc_hooks(axis,sign):
    C=COMPASS[axis];hooks=[]
    for L in range(STEER_LAYERS):
        def mk(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out;B=raw.shape[0]
                a=(float(sign)*C[li][None].expand(B,-1)).float().contiguous()
                d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                new=seasc_ext.seasc_batch(raw,a,d)
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    return hooks
print("       ready")

# ================================================================================================================
# [8/22] EXACT PRE27 / RAW27 RECORDER
# ================================================================================================================
print("\n[8/22] Exact PRE27 / RAW27 recorder...")
def run_transport(prompt,axis,sign):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    box={};hooks=seasc_hooks(axis,sign)
    def pre27(m,args):box["pre"]=args[0][0,pos].float().detach().clone()
    def raw27(m,args,out):
        h=out[0] if isinstance(out,tuple) else out;box["raw"]=h[0,pos].float().detach().clone()
    hooks.append(layers[27].register_forward_pre_hook(pre27))
    hooks.append(layers[27].register_forward_hook(raw27))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    if "pre" not in box or "raw" not in box:raise RuntimeError("L27 capture failed.")
    return box

def collect(axis,prompts,label):
    D=[]
    for i,p in enumerate(prompts,1):
        P=run_transport(p,axis,+1);M=run_transport(p,axis,-1)
        dp=.5*(P["pre"]-M["pre"]);dr=.5*(P["raw"]-M["raw"]);D.append(dr-dp)
        print(f"       {axis} {label} {i}/8")
    return torch.stack(D)
print("       ready")

# ================================================================================================================
# [9/22] BUILD ALL AXES
# ================================================================================================================
print("\n[9/22] BUILD transport all axes...")
BD={}
for ax in AXES:BD[ax]=collect(ax,PROMPTS[ax]["B"],"BUILD")

# ================================================================================================================
# [10/22] FREEZE T27 AXES
# ================================================================================================================
print("\n[10/22] Freeze T27 axes...")
T27={};BUILD_MET={}
for ax in AXES:
    mu=BD[ax].mean(0)
    if float(mu.norm())<=EPS:raise RuntimeError(f"Degenerate mean Δ27 for {ax}")
    t=norm(mu).detach().clone();T27[ax]=t
    c=cosv(BD[ax],t[None].expand_as(BD[ax]));p=BD[ax]@t
    BUILD_MET[ax]={"mean_norm":float(mu.norm()),"mean_cos":float(c.mean()),"Pplus":int((p>0).sum())}
    print(f"       {ax}: ||meanΔ||={float(mu.norm()):.6f} mean cos={float(c.mean()):+.6f} P+={int((p>0).sum())}/8")
print("       all T27 axes frozen before HOLDOUT")

# ================================================================================================================
# [11/22] BUILD LOFO
# ================================================================================================================
print("\n[11/22] BUILD LOFO...")
LOFO={}
for ax in AXES:
    vals=[];held=[]
    for leave in range(8):
        idx=[i for i in range(8) if i!=leave];t=norm(BD[ax][idx].mean(0))
        vals.append(float(torch.dot(t,T27[ax])));held.append(float(cosv(BD[ax][leave],t)))
    LOFO[ax]={"axis_mean":float(np.mean(vals)),"axis_min":float(np.min(vals)),
              "held_mean":float(np.mean(held)),"held_min":float(np.min(held)),"held_Pplus":int(np.sum(np.asarray(held)>0))}
    print(f"       {ax}: axis min={min(vals):+.6f} held mean={np.mean(held):+.6f} held min={min(held):+.6f} P+={sum(x>0 for x in held)}/8")

# ================================================================================================================
# [12/22] HOLDOUT ALL AXES
# ================================================================================================================
print("\n[12/22] HOLDOUT transport all axes...")
HD={}
for ax in AXES:HD[ax]=collect(ax,PROMPTS[ax]["H"],"HOLD")

# ================================================================================================================
# [13/22] HOLDOUT REPLICATION
# ================================================================================================================
print("\n[13/22] HOLDOUT replication...")
HOLD_MET={}
for ax in AXES:
    t=T27[ax];c=cosv(HD[ax],t[None].expand_as(HD[ax]));p=HD[ax]@t
    bm=norm(BD[ax].mean(0));hm=norm(HD[ax].mean(0));mc=float(torch.dot(bm,hm))
    HOLD_MET[ax]={"mean_cos":float(c.mean()),"min_cos":float(c.min()),"Pplus":int((p>0).sum()),"mean_axis_cos":mc}
    print(f"       {ax}: mean cos={float(c.mean()):+.6f} min={float(c.min()):+.6f} P+={int((p>0).sum())}/8 BUILD↔HOLD={mc:+.6f}")

# ================================================================================================================
# [14/22] BOOTSTRAP HOLDOUT
# ================================================================================================================
print("\n[14/22] Bootstrap HOLDOUT...")
def boot(x,n=N_BOOT,seed=191):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT={}
for j,ax in enumerate(AXES):
    c=cosv(HD[ax],T27[ax][None].expand_as(HD[ax])).cpu().numpy()
    m,lo,hi=boot(c,seed=SEED+j);BOOT[ax]={"mean":m,"low":lo,"high":hi}
    print(f"       {ax}: {m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [15/22] 5×5 T27 COSINE MATRIX
# ================================================================================================================
print("\n[15/22] Frozen T27 5×5 cosine matrix...")
TM=np.zeros((5,5),dtype=np.float64)
for i,a in enumerate(AXES):
    for j,b in enumerate(AXES):TM[i,j]=float(torch.dot(T27[a],T27[b]))
print("             A         B         C         D         E")
for i,a in enumerate(AXES):print(f"       {a} "+" ".join(f"{TM[i,j]:+9.5f}" for j in range(5)))

# ================================================================================================================
# [16/22] HOLDOUT-MEAN 5×5 MATRIX
# ================================================================================================================
print("\n[16/22] HOLDOUT mean Δ27 5×5 cosine matrix...")
HM={ax:norm(HD[ax].mean(0)) for ax in AXES}
HMM=np.zeros((5,5),dtype=np.float64)
for i,a in enumerate(AXES):
    for j,b in enumerate(AXES):HMM[i,j]=float(torch.dot(HM[a],HM[b]))
print("             A         B         C         D         E")
for i,a in enumerate(AXES):print(f"       {a} "+" ".join(f"{HMM[i,j]:+9.5f}" for j in range(5)))

# ================================================================================================================
# [17/22] CROSS-AXIS GENERALIZATION MATRIX
# ================================================================================================================
print("\n[17/22] BUILD T27 → foreign-axis HOLDOUT mean alignment...")
XMAT=np.zeros((5,5),dtype=np.float64)
for i,a in enumerate(AXES):
    for j,b in enumerate(AXES):XMAT[i,j]=float(torch.dot(T27[a],HM[b]))
print("             HOLD-A    HOLD-B    HOLD-C    HOLD-D    HOLD-E")
for i,a in enumerate(AXES):print(f"       T-{a} "+" ".join(f"{XMAT[i,j]:+9.5f}" for j in range(5)))

# ================================================================================================================
# [18/22] COMMON-AXIS SUMMARY
# ================================================================================================================
print("\n[18/22] Common-axis summary...")
OFF=[TM[i,j] for i in range(5) for j in range(i+1,5)]
SUCCESS=[TM[i,j] for i in range(4) for j in range(i+1,4)]
E_CROSS=[TM[i,4] for i in range(4)]
print(f"       all off-diagonal mean={np.mean(OFF):+.6f} min={np.min(OFF):+.6f} max={np.max(OFF):+.6f}")
print(f"       A-D pair mean        ={np.mean(SUCCESS):+.6f} min={np.min(SUCCESS):+.6f}")
print(f"       E↔A-D mean           ={np.mean(E_CROSS):+.6f} min={np.min(E_CROSS):+.6f} max={np.max(E_CROSS):+.6f}")

# ================================================================================================================
# [19/22] ORTH/COMPASS RELATION
# ================================================================================================================
print("\n[19/22] T27 relation to source L19 compass...")
for ax in AXES:
    own=float(torch.dot(T27[ax],COMPASS[ax][19]))
    print(f"       {ax}: cos(T27_{ax},COMPASS19_{ax})={own:+.6f}")

# ================================================================================================================
# [20/22] DECISION
# ================================================================================================================
print("\n[20/22] TEST191 decision...")
REP={ax:(BOOT[ax]["low"]>0 and HOLD_MET[ax]["Pplus"]>=7 and HOLD_MET[ax]["mean_axis_cos"]>0) for ax in AXES}
ALL_REP=all(REP.values())
COMMON=np.mean(OFF)>=.80 and np.min(OFF)>=.60
ESEP=np.mean(SUCCESS)>=.80 and np.mean(E_CROSS)<.60
if ALL_REP and COMMON:
    DECISION="GENERAL_CROSS_AXIS_L27_TRANSPORT_CHANNEL_SUPPORTED"
elif ALL_REP and ESEP:
    DECISION="L27_TRANSPORT_REPLICATES_PER_AXIS__E_GEOMETRICALLY_DIVERGES_FROM_A_D"
elif ALL_REP:
    DECISION="L27_TRANSPORT_REPLICATES_PER_AXIS__NO_SINGLE_COMMON_CROSS_AXIS_CHANNEL"
else:
    DECISION="CROSS_AXIS_L27_TRANSPORT_GENERALIZATION_NOT_CONFIRMED"
for ax in AXES:print(f"       {ax} replicated={REP[ax]}")
print("       DECISION:",DECISION)
print("       Geometric/mechanistic result only; no behavioral intervention claim.")

# ================================================================================================================
# [21/22] SAVE
# ================================================================================================================
print("\n[21/22] Saving...")
ROWS=[]
for ax in AXES:
    c=cosv(HD[ax],T27[ax][None].expand_as(HD[ax]));p=HD[ax]@T27[ax]
    for i in range(8):ROWS.append({"axis":ax,"item":i+1,"projection":float(p[i]),"cos_T27":float(c[i]),"delta_norm":float(HD[ax][i].norm())})
with open(ROOT/"HOLDOUT.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
np.savez_compressed(ROOT/"TEST191_CROSS_AXIS_L27.npz",
 compass_matrix=CM,T27_matrix=TM,holdout_mean_matrix=HMM,cross_generalization_matrix=XMAT,
 **{f"T27_{a}":T27[a].cpu().numpy() for a in AXES},
 **{f"BUILD_{a}":BD[a].cpu().numpy() for a in AXES},
 **{f"HOLD_{a}":HD[a].cpu().numpy() for a in AXES})
REPORT={"test":"TEST191","parent":"TEST190","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
"axis_names":AXIS_NAME,"build":BUILD_MET,"lofo":LOFO,"holdout":HOLD_MET,"bootstrap":BOOT,
"compass_matrix":CM.tolist(),"T27_matrix":TM.tolist(),"holdout_mean_matrix":HMM.tolist(),
"cross_generalization_matrix":XMAT.tolist(),"offdiag_mean":float(np.mean(OFF)),
"A_D_pair_mean":float(np.mean(SUCCESS)),"E_cross_mean":float(np.mean(E_CROSS)),
"replicated":REP,"integrity":{"seasc_l0_l19_frozen":True,"l20_l27_unmodified":True,
"layer_local_compasses":True,"T27_frozen_from_BUILD_only":True,"holdouts_separate":True,
"pca_used":False,"K_selected":False,"behavior_gradient_used":False,"new_vector_injected":False,
"final_opened":False,"controller":False,"rescue":False,"weights_modified":False,
"historical_prompt_identity_claimed":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [22/22] COMPLETE
# ================================================================================================================
print("\n[22/22] COMPLETE")
print("="*138)
print("TEST 191 — CROSS-AXIS L27 COMMON TRANSPORT GEOMETRY — SUMMARY")
print("="*138)
for ax in AXES:
    print(f"{ax} {AXIS_NAME[ax]}: BUILDcos={BUILD_MET[ax]['mean_cos']:+.6f} "
          f"LOFOmin={LOFO[ax]['axis_min']:+.6f} HOLDcos={HOLD_MET[ax]['mean_cos']:+.6f} "
          f"P+={HOLD_MET[ax]['Pplus']}/8 BUILD↔HOLD={HOLD_MET[ax]['mean_axis_cos']:+.6f}")
print(f"T27 offdiag mean={np.mean(OFF):+.6f} min={np.min(OFF):+.6f}")
print(f"A-D T27 pair mean={np.mean(SUCCESS):+.6f}")
print(f"E↔A-D T27 mean={np.mean(E_CROSS):+.6f}")
print("PCA used: False | K selected: NONE | Behavior gradient: False | New vector injected: False | FINAL opened: False")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
