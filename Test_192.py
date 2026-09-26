# ================================================================================================================
# TEST 192 — CROSS-AXIS TAIL TRANSPORT CHANNEL FORMATION X-RAY
# AkbasCore 3.2 / SEASC — TEST191 continuation
#
# QUESTION:
#   Where across the unsteered L20-L27 tail does the cross-axis transport geometry form?
#
# METHOD:
#   A/B/C/D/E locked layer-local compasses from TEST191.
#   Identical SEASC L0-L19.
#   For each axis and prompt, capture PRE20 and RAW20...RAW27.
#   D_L=.5[(H_L+ - H_L-) - (PRE20+ - PRE20-)]
#   Freeze T_L^axis from BUILD only.
#   Replicate each layer on independent HOLDOUT.
#   Measure within-axis BUILD↔HOLD and cross-axis 5×5 T_L cosine matrices at L20-L27.
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
print("TEST 192 — CROSS-AXIS TAIL TRANSPORT CHANNEL FORMATION X-RAY")
print("A/B/C/D/E — L20→L27 UNSTEERED TAIL — BUILD-FROZEN TRANSPORT AXES × INDEPENDENT HOLDOUT")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=192
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28;TAIL=list(range(20,28))
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
PRIMARY_SCALE=.50;EPS=1e-10;N_BOOT=20000
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST192_CROSS_AXIS_TAIL_FORMATION");ROOT.mkdir(parents=True,exist_ok=True)

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
# [1/24] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test192";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
seasc_ext=load_inline(name="akbascore_test192_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
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
# [3/24] LOCKED TEST191 AXES
# ================================================================================================================
print("\n[3/24] Locked TEST191 axis definitions...")
AXIS_PAIRS={
"A":[("The lamp is above the table.","The lamp is below the table."),("The bird is above the branch.","The bird is below the branch."),
("The sign is above the doorway.","The sign is below the doorway."),("The shelf is above the cabinet.","The shelf is below the cabinet."),
("The cloud is above the hill.","The cloud is below the hill."),("The clock is above the window.","The clock is below the window."),
("The bridge is above the river.","The bridge is below the river."),("The picture is above the sofa.","The picture is below the sofa.")],
"B":[("The alarm occurs before the evacuation.","The alarm occurs after the evacuation."),("The flash occurs before the thunder.","The flash occurs after the thunder."),
("The ignition occurs before the motion.","The ignition occurs after the motion."),("The warning occurs before the shutdown.","The warning occurs after the shutdown."),
("The sunrise occurs before noon.","The sunrise occurs after noon."),("The opening occurs before the closing.","The opening occurs after the closing."),
("The question occurs before the answer.","The question occurs after the answer."),("The launch occurs before the landing.","The launch occurs after the landing.")],
"C":[("The key is inside the box.","The key is outside the box."),("The coin is inside the jar.","The coin is outside the jar."),
("The book is inside the bag.","The book is outside the bag."),("The ball is inside the basket.","The ball is outside the basket."),
("The tool is inside the drawer.","The tool is outside the drawer."),("The letter is inside the envelope.","The letter is outside the envelope."),
("The bottle is inside the cabinet.","The bottle is outside the cabinet."),("The card is inside the folder.","The card is outside the folder.")],
"D":[("The door is open.","The door is closed."),("The window is open.","The window is closed."),
("The gate is open.","The gate is closed."),("The box is open.","The box is closed."),
("The drawer is open.","The drawer is closed."),("The lid is open.","The lid is closed."),
("The cabinet is open.","The cabinet is closed."),("The hatch is open.","The hatch is closed.")],
"E":[("The spark is the cause of the fire.","The fire is the effect of the spark."),("The impact is the cause of the crack.","The crack is the effect of the impact."),
("The rainfall is the cause of the flooding.","The flooding is the effect of the rainfall."),("The heat is the cause of the melting.","The melting is the effect of the heat."),
("The collision is the cause of the damage.","The damage is the effect of the collision."),("The pressure is the cause of the deformation.","The deformation is the effect of the pressure."),
("The infection is the cause of the fever.","The fever is the effect of the infection."),("The vibration is the cause of the loosening.","The loosening is the effect of the vibration.")]}
AXES=list("ABCDE");AXIS_NAME={"A":"ABOVE↔BELOW","B":"BEFORE↔AFTER","C":"INSIDE↔OUTSIDE","D":"OPEN↔CLOSED","E":"CAUSE↔EFFECT"}
print("       A/B/C/D/E ready")

# ================================================================================================================
# [4/24] LAYER-LOCAL COMPASSES
# ================================================================================================================
print("\n[4/24] Layer-local compasses...")
COMPASS={}
for ax in AXES:
    P=[];N=[]
    for p,n in AXIS_PAIRS[ax]:P.append(capture(p));N.append(capture(n))
    COMPASS[ax]=norm(torch.stack(P).mean(0)-torch.stack(N).mean(0)).contiguous()
    print(f"       {ax} {AXIS_NAME[ax]} ready")

# ================================================================================================================
# [5/24] TEST191 BUILD / HOLDOUT LOCK
# ================================================================================================================
print("\n[5/24] TEST191 BUILD / HOLDOUT prompts...")
PROMPTS={
"A":{"B":["The red marker and blue marker are vertically separated.","Two lamps occupy different vertical positions.","A sign and a doorway have a vertical relation.","A bird and branch have a vertical relation.","A shelf and cabinet occupy different heights.","A cloud and hill have a vertical relation.","A clock and window occupy different heights.","A bridge and river have a vertical relation."],
"H":["A picture and sofa occupy different vertical positions.","A flag and roof have a vertical relation.","A drone and building occupy different heights.","A balloon and tree have a vertical relation.","A ceiling light and desk occupy different heights.","A kite and field have a vertical relation.","A camera and tripod base occupy different heights.","A mountain peak and valley have a vertical relation."]},
"B":{"B":["An alarm and evacuation occur at different times.","A flash and thunder occur in temporal sequence.","Ignition and motion occur in temporal sequence.","A warning and shutdown occur at different times.","Sunrise and noon occur in temporal sequence.","Opening and closing occur at different times.","A question and answer occur in temporal sequence.","Launch and landing occur at different times."],
"H":["A signal and response occur in temporal sequence.","A start and finish occur at different times.","Departure and arrival occur in temporal sequence.","A trigger and reaction occur at different times.","Heating and cooling occur in temporal sequence.","Entry and exit occur at different times.","Detection and reporting occur in temporal sequence.","Loading and unloading occur at different times."]},
"C":{"B":["A key and box have a containment relation.","A coin and jar have a containment relation.","A book and bag have a containment relation.","A ball and basket have a containment relation.","A tool and drawer have a containment relation.","A letter and envelope have a containment relation.","A bottle and cabinet have a containment relation.","A card and folder have a containment relation."],
"H":["A toy and container have a containment relation.","A document and case have a containment relation.","A cable and conduit have a containment relation.","A package and vehicle have a containment relation.","A component and housing have a containment relation.","A seed and shell have a containment relation.","A file and archive have a containment relation.","A capsule and chamber have a containment relation."]},
"D":{"B":["A door has an open-closed state.","A window has an open-closed state.","A gate has an open-closed state.","A box has an open-closed state.","A drawer has an open-closed state.","A lid has an open-closed state.","A cabinet has an open-closed state.","A hatch has an open-closed state."],
"H":["A valve has an open-closed state.","A shutter has an open-closed state.","A container has an open-closed state.","A cover has an open-closed state.","A barrier has an open-closed state.","A vent has an open-closed state.","A panel has an open-closed state.","A compartment has an open-closed state."]},
"E":{"B":["A fractured pipe and a pressure drop form a causal sequence.","A blocked air passage and rising temperature form a causal sequence.","A damaged seal and fluid loss form a causal sequence.","A current surge and a failed circuit form a causal sequence.","A stuck mechanism and restricted motion form a causal sequence.","A worn bearing and increased vibration form a causal sequence.","A clogged outlet and accumulated liquid form a causal sequence.","A strong impact and a deformed bracket form a causal sequence."],
"H":["A broken coupling and loss of transmitted motion form a causal sequence.","A blocked intake and reduced airflow form a causal sequence.","A damaged gasket and pressure leakage form a causal sequence.","A failed coolant pump and rising temperature form a causal sequence.","A seized bearing and restricted rotation form a causal sequence.","A severed cable and loss of electrical power form a causal sequence.","A clogged nozzle and reduced discharge form a causal sequence.","A fractured support and structural displacement form a causal sequence."]}}
print("       5 axes × BUILD8 × HOLDOUT8")

# ================================================================================================================
# [6/24] SEASC
# ================================================================================================================
print("\n[6/24] Axis-specific SEASC...")
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
# [7/24] FULL TAIL RECORDER
# ================================================================================================================
print("\n[7/24] PRE20 + RAW20-L27 recorder...")
def run_tail(prompt,axis,sign):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    box={"raw":{}};hooks=seasc_hooks(axis,sign)
    def pre20(m,args):box["pre20"]=args[0][0,pos].float().detach().clone()
    hooks.append(layers[20].register_forward_pre_hook(pre20))
    for L in TAIL:
        def mk(li):
            def hk(m,args,out):
                h=out[0] if isinstance(out,tuple) else out;box["raw"][li]=h[0,pos].float().detach().clone()
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    if "pre20" not in box or len(box["raw"])!=8:raise RuntimeError("Tail capture failed.")
    return box

def collect(axis,prompts,label):
    D={L:[] for L in TAIL}
    for i,p in enumerate(prompts,1):
        P=run_tail(p,axis,+1);M=run_tail(p,axis,-1)
        base=.5*(P["pre20"]-M["pre20"])
        for L in TAIL:D[L].append(.5*(P["raw"][L]-M["raw"][L])-base)
        print(f"       {axis} {label} {i}/8")
    return {L:torch.stack(D[L]) for L in TAIL}
print("       ready")

# ================================================================================================================
# [8/24] BUILD ALL AXES
# ================================================================================================================
print("\n[8/24] BUILD tail transport...")
BD={}
for ax in AXES:BD[ax]=collect(ax,PROMPTS[ax]["B"],"BUILD")

# ================================================================================================================
# [9/24] FREEZE LAYERWISE TRANSPORT AXES
# ================================================================================================================
print("\n[9/24] Freeze BUILD T_L axes...")
TL={ax:{} for ax in AXES};BM={}
for ax in AXES:
    BM[ax]={}
    for L in TAIL:
        mu=BD[ax][L].mean(0)
        if float(mu.norm())<=EPS:raise RuntimeError(f"Degenerate mean: {ax} L{L}")
        TL[ax][L]=norm(mu).detach().clone()
        c=cosv(BD[ax][L],TL[ax][L][None].expand_as(BD[ax][L]))
        BM[ax][L]={"norm":float(mu.norm()),"cos":float(c.mean()),"Pplus":int((c>0).sum())}
    print(f"       {ax}: "+" ".join(f"L{L}={BM[ax][L]['cos']:+.3f}" for L in TAIL))
print("       all T_L frozen before HOLDOUT")

# ================================================================================================================
# [10/24] BUILD LOFO
# ================================================================================================================
print("\n[10/24] BUILD LOFO...")
LOFO={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:
        st=[];held=[]
        for leave in range(8):
            idx=[i for i in range(8) if i!=leave];t=norm(BD[ax][L][idx].mean(0))
            st.append(float(torch.dot(t,TL[ax][L])));held.append(float(cosv(BD[ax][L][leave],t)))
        LOFO[ax][L]={"axis_min":float(np.min(st)),"held_mean":float(np.mean(held)),"held_min":float(np.min(held))}
    print(f"       {ax}: L20 held={LOFO[ax][20]['held_mean']:+.3f} → L27 held={LOFO[ax][27]['held_mean']:+.3f}")

# ================================================================================================================
# [11/24] HOLDOUT ALL AXES
# ================================================================================================================
print("\n[11/24] HOLDOUT tail transport...")
HD={}
for ax in AXES:HD[ax]=collect(ax,PROMPTS[ax]["H"],"HOLD")

# ================================================================================================================
# [12/24] WITHIN-AXIS REPLICATION BY LAYER
# ================================================================================================================
print("\n[12/24] Within-axis HOLDOUT replication...")
HMET={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:
        c=cosv(HD[ax][L],TL[ax][L][None].expand_as(HD[ax][L]))
        bc=norm(BD[ax][L].mean(0));hc=norm(HD[ax][L].mean(0))
        HMET[ax][L]={"mean_cos":float(c.mean()),"min_cos":float(c.min()),"Pplus":int((c>0).sum()),
                     "build_hold":float(torch.dot(bc,hc))}
    print(f"       {ax}: "+" ".join(f"L{L}={HMET[ax][L]['mean_cos']:+.3f}" for L in TAIL))

# ================================================================================================================
# [13/24] BOOTSTRAP
# ================================================================================================================
print("\n[13/24] Bootstrap HOLDOUT cosine...")
def boot(x,n=N_BOOT,seed=192):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT={ax:{} for ax in AXES}
for ai,ax in enumerate(AXES):
    for L in TAIL:
        c=cosv(HD[ax][L],TL[ax][L][None].expand_as(HD[ax][L])).cpu().numpy()
        m,lo,hi=boot(c,seed=SEED+ai*20+L);BOOT[ax][L]={"mean":m,"low":lo,"high":hi}
    print(f"       {ax}: L20 {BOOT[ax][20]['mean']:+.3f}[{BOOT[ax][20]['low']:+.3f},{BOOT[ax][20]['high']:+.3f}] "
          f"L27 {BOOT[ax][27]['mean']:+.3f}[{BOOT[ax][27]['low']:+.3f},{BOOT[ax][27]['high']:+.3f}]")

# ================================================================================================================
# [14/24] CROSS-AXIS T_L MATRICES
# ================================================================================================================
print("\n[14/24] Cross-axis T_L matrices...")
TM={}
for L in TAIL:
    M=np.zeros((5,5))
    for i,a in enumerate(AXES):
        for j,b in enumerate(AXES):M[i,j]=float(torch.dot(TL[a][L],TL[b][L]))
    TM[L]=M
    off=[M[i,j] for i in range(5) for j in range(i+1,5)]
    print(f"       L{L}: offdiag mean={np.mean(off):+.6f} min={np.min(off):+.6f} max={np.max(off):+.6f}")

# ================================================================================================================
# [15/24] A-D VS E CROSS-AXIS TRAJECTORY
# ================================================================================================================
print("\n[15/24] A-D vs E cross-axis trajectory...")
CROSS={}
for L in TAIL:
    M=TM[L];ad=[M[i,j] for i in range(4) for j in range(i+1,4)];ec=[M[i,4] for i in range(4)]
    CROSS[L]={"AD_mean":float(np.mean(ad)),"AD_min":float(np.min(ad)),
              "E_mean":float(np.mean(ec)),"E_min":float(np.min(ec)),"all_mean":float(np.mean([M[i,j] for i in range(5) for j in range(i+1,5)]))}
    print(f"       L{L}: A-D={CROSS[L]['AD_mean']:+.6f} E↔A-D={CROSS[L]['E_mean']:+.6f} all={CROSS[L]['all_mean']:+.6f}")

# ================================================================================================================
# [16/24] HOLDOUT-MEAN CROSS-AXIS MATRICES
# ================================================================================================================
print("\n[16/24] HOLDOUT-mean cross-axis matrices...")
HMM={}
for L in TAIL:
    hm={ax:norm(HD[ax][L].mean(0)) for ax in AXES};M=np.zeros((5,5))
    for i,a in enumerate(AXES):
        for j,b in enumerate(AXES):M[i,j]=float(torch.dot(hm[a],hm[b]))
    HMM[L]=M;off=[M[i,j] for i in range(5) for j in range(i+1,5)]
    print(f"       L{L}: offdiag mean={np.mean(off):+.6f} min={np.min(off):+.6f}")

# ================================================================================================================
# [17/24] LAYER-TO-LAYER AXIS ROTATION
# ================================================================================================================
print("\n[17/24] Layer-to-layer T_L rotation...")
ROT={ax:{} for ax in AXES}
for ax in AXES:
    vals=[]
    for L in range(21,28):
        c=float(torch.dot(TL[ax][L-1],TL[ax][L]));ROT[ax][L]=c;vals.append(c)
    print(f"       {ax}: "+" ".join(f"{L-1}→{L}={ROT[ax][L]:+.3f}" for L in range(21,28)))

# ================================================================================================================
# [18/24] L20→L27 NET ROTATION
# ================================================================================================================
print("\n[18/24] L20→L27 net rotation...")
NET={}
for ax in AXES:
    c=float(torch.dot(TL[ax][20],TL[ax][27]));NET[ax]=c
    print(f"       {ax}: cos(T20,T27)={c:+.6f} angle={math.degrees(math.acos(max(-1,min(1,c)))):.3f}°")

# ================================================================================================================
# [19/24] L27 TEST191 CONSISTENCY
# ================================================================================================================
print("\n[19/24] L27 TEST191-equivalent summary...")
for ax in AXES:
    print(f"       {ax}: BUILDcos={BM[ax][27]['cos']:+.6f} LOFOmin={LOFO[ax][27]['axis_min']:+.6f} "
          f"HOLDcos={HMET[ax][27]['mean_cos']:+.6f} BUILD↔HOLD={HMET[ax][27]['build_hold']:+.6f}")

# ================================================================================================================
# [20/24] FORMATION LOCALIZATION
# ================================================================================================================
print("\n[20/24] Formation localization...")
FORM={}
for ax in AXES:
    qualified=[L for L in TAIL if BOOT[ax][L]["low"]>0 and HMET[ax][L]["Pplus"]>=7]
    FORM[ax]=qualified[0] if qualified else None
    print(f"       {ax}: first replicated layer={FORM[ax] if FORM[ax] is not None else 'NONE'}")
COMMON_LAYERS=[]
for L in TAIL:
    off=[TM[L][i,j] for i in range(5) for j in range(i+1,5)]
    if np.mean(off)>=.80 and np.min(off)>=.60:COMMON_LAYERS.append(L)
FIRST_COMMON=COMMON_LAYERS[0] if COMMON_LAYERS else None
print(f"       first general cross-axis common layer={FIRST_COMMON if FIRST_COMMON is not None else 'NONE'}")

# ================================================================================================================
# [21/24] DECISION
# ================================================================================================================
print("\n[21/24] TEST192 decision...")
ALL_L27=all(BOOT[a][27]["low"]>0 and HMET[a][27]["Pplus"]>=7 for a in AXES)
if FIRST_COMMON is not None and ALL_L27:
    DECISION=f"GENERAL_CROSS_AXIS_TAIL_CHANNEL_FORMS_BY_L{FIRST_COMMON}"
elif ALL_L27:
    DECISION="AXIS_SPECIFIC_TAIL_TRANSPORT_REPLICATES__NO_SINGLE_GENERAL_CHANNEL"
else:
    DECISION="TAIL_TRANSPORT_CHANNEL_FORMATION_NOT_CONFIRMED"
print("       DECISION:",DECISION)
print("       Layer-resolved geometric/mechanistic result only.")

# ================================================================================================================
# [22/24] SAVE CSV
# ================================================================================================================
print("\n[22/24] Saving CSV...")
ROWS=[]
for ax in AXES:
    for L in TAIL:
        ROWS.append({"axis":ax,"layer":L,"build_mean_norm":BM[ax][L]["norm"],"build_mean_cos":BM[ax][L]["cos"],
                     "lofo_axis_min":LOFO[ax][L]["axis_min"],"lofo_held_mean":LOFO[ax][L]["held_mean"],
                     "hold_mean_cos":HMET[ax][L]["mean_cos"],"hold_min_cos":HMET[ax][L]["min_cos"],
                     "hold_Pplus":HMET[ax][L]["Pplus"],"build_hold_cos":HMET[ax][L]["build_hold"],
                     "boot_low":BOOT[ax][L]["low"],"boot_high":BOOT[ax][L]["high"]})
with open(ROOT/"LAYERWISE.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
print("       CSV ready")

# ================================================================================================================
# [23/24] SAVE NPZ / REPORT
# ================================================================================================================
print("\n[23/24] Saving NPZ / report...")
ARR={}
for ax in AXES:
    for L in TAIL:
        ARR[f"T_{ax}_L{L}"]=TL[ax][L].cpu().numpy()
        ARR[f"BUILD_{ax}_L{L}"]=BD[ax][L].cpu().numpy()
        ARR[f"HOLD_{ax}_L{L}"]=HD[ax][L].cpu().numpy()
for L in TAIL:
    ARR[f"T_MATRIX_L{L}"]=TM[L];ARR[f"HOLD_MATRIX_L{L}"]=HMM[L]
np.savez_compressed(ROOT/"TEST192_TAIL_FORMATION.npz",**ARR)
REPORT={"test":"TEST192","parent":"TEST191","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
"formation":FORM,"first_general_common_layer":FIRST_COMMON,"cross_layer":CROSS,"net_rotation":NET,
"build":BM,"lofo":LOFO,"holdout":HMET,"bootstrap":BOOT,
"T_matrices":{str(L):TM[L].tolist() for L in TAIL},
"holdout_matrices":{str(L):HMM[L].tolist() for L in TAIL},
"integrity":{"test191_axis_definitions_preserved":True,"test191_prompts_preserved":True,
"seasc_l0_l19_frozen":True,"l20_l27_unmodified":True,"build_frozen_before_holdout":True,
"pca_used":False,"K_selected":False,"behavior_gradient_used":False,"new_vector_injected":False,
"final_opened":False,"controller":False,"rescue":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [24/24] COMPLETE
# ================================================================================================================
print("\n[24/24] COMPLETE")
print("="*138)
print("TEST 192 — CROSS-AXIS TAIL TRANSPORT CHANNEL FORMATION — SUMMARY")
print("="*138)
for L in TAIL:
    print(f"L{L}: cross-axis={CROSS[L]['all_mean']:+.6f} A-D={CROSS[L]['AD_mean']:+.6f} E↔A-D={CROSS[L]['E_mean']:+.6f}")
for ax in AXES:
    print(f"{ax}: first={FORM[ax]} T20→T27={NET[ax]:+.6f} L27 HOLD={HMET[ax][27]['mean_cos']:+.6f} "
          f"BUILD↔HOLD={HMET[ax][27]['build_hold']:+.6f}")
print("First general common layer:",FIRST_COMMON)
print("PCA: False | K: NONE | Behavior gradient: False | New vector: False | FINAL: False")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
