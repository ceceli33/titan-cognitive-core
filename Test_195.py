# ================================================================================================================
# TEST 195 — NOVEL-SURFACE TAIL TRANSPORT TRANSFER
# AkbasCore 3.2 / SEASC — TEST194 continuation
#
# QUESTION:
#   Do TEST194 BUILD-frozen axis-specific composed tail laws generalize to a third,
#   lexically and structurally novel prompt set never used for fitting?
#
# METHOD:
#   Preserve TEST194 model/axes/compasses/SEASC/BUILD/HOLDOUT.
#   Fit M20...M27 from BUILD only.
#   Freeze all operators.
#   Open NEW SURFACE8 only after freeze.
#   Start from true SURFACE PRE20 only; compose M20→...→M27 without true intermediate states.
#   Compare terminal RAW27 against identity and foreign-axis composed controls.
#
# NO REFIT | NO SELECTION | NO PCA | NO K | NO BEHAVIOR GRADIENT | NO NEW VECTOR | NO FINAL | NO WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 195 — NOVEL-SURFACE TAIL TRANSPORT TRANSFER")
print("A/B/C/D/E — BUILD-FROZEN M20→M27 × THIRD INDEPENDENT SURFACE8")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=195
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28;TAIL=list(range(20,28))
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
PRIMARY_SCALE=.50;EPS=1e-10;RIDGE_REL=1e-4;N_BOOT=20000
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST195_NOVEL_SURFACE_TRANSFER");ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(STEER_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
def cosv(a,b):return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(EPS)
def rerr(a,b):return (a-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(EPS)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
print(f"SEASC RSS={RSS:.9f} | scale={PRIMARY_SCALE} | ridge_rel={RIDGE_REL}")

# ================================================================================================================
# [1/27] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test195";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
print("\n[1/27] CUDA...")
seasc_ext=load_inline(name="akbascore_test195_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD,verbose=False)
print("       OK")

# ================================================================================================================
# [2/27] MODEL
# ================================================================================================================
print("\n[2/27] Model...")
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
# [3/27] TEST194 AXIS LOCK
# ================================================================================================================
print("\n[3/27] TEST194 axis lock...")
AXIS_PAIRS={
"A":[("The lamp is above the table.","The lamp is below the table."),("The bird is above the branch.","The bird is below the branch."),("The sign is above the doorway.","The sign is below the doorway."),("The shelf is above the cabinet.","The shelf is below the cabinet."),("The cloud is above the hill.","The cloud is below the hill."),("The clock is above the window.","The clock is below the window."),("The bridge is above the river.","The bridge is below the river."),("The picture is above the sofa.","The picture is below the sofa.")],
"B":[("The alarm occurs before the evacuation.","The alarm occurs after the evacuation."),("The flash occurs before the thunder.","The flash occurs after the thunder."),("The ignition occurs before the motion.","The ignition occurs after the motion."),("The warning occurs before the shutdown.","The warning occurs after the shutdown."),("The sunrise occurs before noon.","The sunrise occurs after noon."),("The opening occurs before the closing.","The opening occurs after the closing."),("The question occurs before the answer.","The question occurs after the answer."),("The launch occurs before the landing.","The launch occurs after the landing.")],
"C":[("The key is inside the box.","The key is outside the box."),("The coin is inside the jar.","The coin is outside the jar."),("The book is inside the bag.","The book is outside the bag."),("The ball is inside the basket.","The ball is outside the basket."),("The tool is inside the drawer.","The tool is outside the drawer."),("The letter is inside the envelope.","The letter is outside the envelope."),("The bottle is inside the cabinet.","The bottle is outside the cabinet."),("The card is inside the folder.","The card is outside the folder.")],
"D":[("The door is open.","The door is closed."),("The window is open.","The window is closed."),("The gate is open.","The gate is closed."),("The box is open.","The box is closed."),("The drawer is open.","The drawer is closed."),("The lid is open.","The lid is closed."),("The cabinet is open.","The cabinet is closed."),("The hatch is open.","The hatch is closed.")],
"E":[("The spark is the cause of the fire.","The fire is the effect of the spark."),("The impact is the cause of the crack.","The crack is the effect of the impact."),("The rainfall is the cause of the flooding.","The flooding is the effect of the rainfall."),("The heat is the cause of the melting.","The melting is the effect of the heat."),("The collision is the cause of the damage.","The damage is the effect of the collision."),("The pressure is the cause of the deformation.","The deformation is the effect of the pressure."),("The infection is the cause of the fever.","The fever is the effect of the infection."),("The vibration is the cause of the loosening.","The loosening is the effect of the vibration.")]}
AXES=list("ABCDE");AXIS_NAME={"A":"ABOVE↔BELOW","B":"BEFORE↔AFTER","C":"INSIDE↔OUTSIDE","D":"OPEN↔CLOSED","E":"CAUSE↔EFFECT"}
print("       A/B/C/D/E ready")

# ================================================================================================================
# [4/27] COMPASSES
# ================================================================================================================
print("\n[4/27] Layer-local compasses...")
COMPASS={}
for ax in AXES:
    P=[];N=[]
    for p,n in AXIS_PAIRS[ax]:P.append(capture(p));N.append(capture(n))
    COMPASS[ax]=norm(torch.stack(P).mean(0)-torch.stack(N).mean(0)).contiguous()
    print(f"       {ax} {AXIS_NAME[ax]} ready")

# ================================================================================================================
# [5/27] TEST194 BUILD/HOLDOUT LOCK
# ================================================================================================================
print("\n[5/27] TEST194 BUILD/HOLDOUT lock...")
PROMPTS={
"A":{"B":["The red marker and blue marker are vertically separated.","Two lamps occupy different vertical positions.","A sign and a doorway have a vertical relation.","A bird and branch have a vertical relation.","A shelf and cabinet occupy different heights.","A cloud and hill have a vertical relation.","A clock and window occupy different heights.","A bridge and river have a vertical relation."],"H":["A picture and sofa occupy different vertical positions.","A flag and roof have a vertical relation.","A drone and building occupy different heights.","A balloon and tree have a vertical relation.","A ceiling light and desk occupy different heights.","A kite and field have a vertical relation.","A camera and tripod base occupy different heights.","A mountain peak and valley have a vertical relation."]},
"B":{"B":["An alarm and evacuation occur at different times.","A flash and thunder occur in temporal sequence.","Ignition and motion occur in temporal sequence.","A warning and shutdown occur at different times.","Sunrise and noon occur in temporal sequence.","Opening and closing occur at different times.","A question and answer occur in temporal sequence.","Launch and landing occur at different times."],"H":["A signal and response occur in temporal sequence.","A start and finish occur at different times.","Departure and arrival occur in temporal sequence.","A trigger and reaction occur at different times.","Heating and cooling occur in temporal sequence.","Entry and exit occur at different times.","Detection and reporting occur in temporal sequence.","Loading and unloading occur at different times."]},
"C":{"B":["A key and box have a containment relation.","A coin and jar have a containment relation.","A book and bag have a containment relation.","A ball and basket have a containment relation.","A tool and drawer have a containment relation.","A letter and envelope have a containment relation.","A bottle and cabinet have a containment relation.","A card and folder have a containment relation."],"H":["A toy and container have a containment relation.","A document and case have a containment relation.","A cable and conduit have a containment relation.","A package and vehicle have a containment relation.","A component and housing have a containment relation.","A seed and shell have a containment relation.","A file and archive have a containment relation.","A capsule and chamber have a containment relation."]},
"D":{"B":["A door has an open-closed state.","A window has an open-closed state.","A gate has an open-closed state.","A box has an open-closed state.","A drawer has an open-closed state.","A lid has an open-closed state.","A cabinet has an open-closed state.","A hatch has an open-closed state."],"H":["A valve has an open-closed state.","A shutter has an open-closed state.","A container has an open-closed state.","A cover has an open-closed state.","A barrier has an open-closed state.","A vent has an open-closed state.","A panel has an open-closed state.","A compartment has an open-closed state."]},
"E":{"B":["A fractured pipe and a pressure drop form a causal sequence.","A blocked air passage and rising temperature form a causal sequence.","A damaged seal and fluid loss form a causal sequence.","A current surge and a failed circuit form a causal sequence.","A stuck mechanism and restricted motion form a causal sequence.","A worn bearing and increased vibration form a causal sequence.","A clogged outlet and accumulated liquid form a causal sequence.","A strong impact and a deformed bracket form a causal sequence."],"H":["A broken coupling and loss of transmitted motion form a causal sequence.","A blocked intake and reduced airflow form a causal sequence.","A damaged gasket and pressure leakage form a causal sequence.","A failed coolant pump and rising temperature form a causal sequence.","A seized bearing and restricted rotation form a causal sequence.","A severed cable and loss of electrical power form a causal sequence.","A clogged nozzle and reduced discharge form a causal sequence.","A fractured support and structural displacement form a causal sequence."]}}
print("       BUILD/HOLDOUT locked")

# ================================================================================================================
# [6/27] NEW SURFACE8 — NEVER USED FOR FITTING
# ================================================================================================================
SURFACE={
"A":["Compare the altitude of a helicopter with the road beneath it.","Consider where a chandelier sits relative to the dining table.","Describe the vertical placement of a satellite relative to Earth's surface.","Compare the height of a rooftop antenna with the street.","Consider the position of an aircraft relative to the runway.","Describe where a canopy sits relative to the chairs underneath.","Compare a lighthouse lamp with the shoreline.","Consider the vertical relation between a balcony and the pavement."],
"B":["Consider the ordering of registration and admission.","Compare when a diagnosis occurs relative to treatment.","Describe the sequence connecting payment and shipment.","Consider the ordering of rehearsal and performance.","Compare inspection with subsequent certification.","Describe the temporal relation between planting and harvesting.","Consider the sequence of login and logout.","Compare data collection with later analysis."],
"C":["Consider whether a passenger occupies a vehicle cabin.","Describe the containment relation between medicine and a vial.","Consider a photograph relative to a frame.","Describe a battery relative to a device housing.","Consider luggage relative to an aircraft cargo hold.","Describe a cartridge relative to a printer.","Consider food relative to a sealed package.","Describe a specimen relative to a laboratory tube."],
"D":["Consider the state of a lockable safe with respect to access.","Describe whether a garage entrance permits passage or blocks it.","Consider the state of a bottle cap with respect to access to the contents.","Describe whether a tunnel barrier permits passage.","Consider the state of a laptop lid.","Describe whether a faucet valve permits flow or prevents it.","Consider the state of a shipping crate before or after sealing.","Describe whether an access port permits entry."],
"E":["A depleted battery and loss of device power are causally related.","A loose electrical connection and intermittent operation are causally related.","A frozen water line and interrupted flow are causally related.","Excessive load and structural bending are causally related.","A contaminated filter and reduced ventilation are causally related.","A software fault and an unexpected shutdown are causally related.","A punctured tire and falling air pressure are causally related.","A disconnected sensor and missing measurements are causally related."]}
print("\n[6/27] NEW SURFACE8 defined but unopened...")
print("       5 axes × 8 novel lexical/structural prompts")

# ================================================================================================================
# [7/27] SEASC
# ================================================================================================================
print("\n[7/27] SEASC...")
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
# [8/27] TAIL RECORDER
# ================================================================================================================
print("\n[8/27] PRE/RAW tail recorder...")
def run_tail(prompt,axis,sign):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    box={"pre":{},"raw":{}};hooks=seasc_hooks(axis,sign)
    for L in TAIL:
        def mkp(li):
            def hk(m,args):box["pre"][li]=args[0][0,pos].float().detach().clone()
            return hk
        def mkr(li):
            def hk(m,args,out):
                h=out[0] if isinstance(out,tuple) else out;box["raw"][li]=h[0,pos].float().detach().clone()
            return hk
        hooks.append(layers[L].register_forward_pre_hook(mkp(L)))
        hooks.append(layers[L].register_forward_hook(mkr(L)))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    return box

def collect(axis,prompts,label):
    PRE={L:[] for L in TAIL};RAW={L:[] for L in TAIL}
    for i,p in enumerate(prompts,1):
        P=run_tail(p,axis,+1);M=run_tail(p,axis,-1)
        for L in TAIL:
            PRE[L].append(.5*(P["pre"][L]-M["pre"][L]))
            RAW[L].append(.5*(P["raw"][L]-M["raw"][L]))
        print(f"       {axis} {label} {i}/8")
    return {L:torch.stack(PRE[L]) for L in TAIL},{L:torch.stack(RAW[L]) for L in TAIL}
print("       ready")

# ================================================================================================================
# [9/27] BUILD ONLY
# ================================================================================================================
print("\n[9/27] BUILD collection...")
BPRE={};BRAW={}
for ax in AXES:BPRE[ax],BRAW[ax]=collect(ax,PROMPTS[ax]["B"],"BUILD")

# ================================================================================================================
# [10/27] CONTINUITY
# ================================================================================================================
print("\n[10/27] BUILD continuity...")
for ax in AXES:
    m=max(float((BRAW[ax][L]-BPRE[ax][L+1]).norm()/BPRE[ax][L+1].norm().clamp_min(EPS)) for L in range(20,27))
    print(f"       {ax}: max relerr={m:.10f}")
    if m>1e-6:raise RuntimeError("Continuity failed.")

# ================================================================================================================
# [11/27] FIT + FREEZE OPERATORS
# ================================================================================================================
print("\n[11/27] Fit BUILD-only M20→M27...")
def fit_op(X,Y):
    X=X.float();Y=Y.float();G=X@X.T
    lam=RIDGE_REL*float(torch.trace(G)/G.shape[0])
    A=torch.linalg.solve(G+lam*torch.eye(G.shape[0],device=G.device),Y)
    return {"X":X.detach().clone(),"A":A.detach().clone(),"lambda":lam}
def apply_op(op,x):return (x@op["X"].T)@op["A"]

OPS={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:OPS[ax][L]=fit_op(BPRE[ax][L],BRAW[ax][L])
    print(f"       {ax} frozen")
print("       ALL OPERATORS FROZEN — SURFACE8 MAY NOW OPEN")

# ================================================================================================================
# [12/27] HISTORICAL HOLDOUT REFERENCE
# ================================================================================================================
print("\n[12/27] TEST194 HOLDOUT reference...")
HPRE={};HRAW={}
for ax in AXES:HPRE[ax],HRAW[ax]=collect(ax,PROMPTS[ax]["H"],"HOLD")

HREF={}
for ax in AXES:
    x=HPRE[ax][20].clone()
    for L in TAIL:x=apply_op(OPS[ax][L],x)
    y=HRAW[ax][27]
    HREF[ax]={"cos":float(cosv(x,y).mean()),"err":float(rerr(x,y).mean())}
    print(f"       {ax}: terminal cos={HREF[ax]['cos']:+.6f} err={HREF[ax]['err']:.6f}")

# ================================================================================================================
# [13/27] OPEN THIRD SURFACE
# ================================================================================================================
print("\n[13/27] Opening NEW SURFACE8 after freeze...")
SPRE={};SRAW={}
for ax in AXES:SPRE[ax],SRAW[ax]=collect(ax,SURFACE[ax],"SURFACE")

# ================================================================================================================
# [14/27] SURFACE CONTINUITY
# ================================================================================================================
print("\n[14/27] SURFACE continuity...")
for ax in AXES:
    m=max(float((SRAW[ax][L]-SPRE[ax][L+1]).norm()/SPRE[ax][L+1].norm().clamp_min(EPS)) for L in range(20,27))
    print(f"       {ax}: max relerr={m:.10f}")
    if m>1e-6:raise RuntimeError("SURFACE continuity failed.")

# ================================================================================================================
# [15/27] TEACHER-FORCED SURFACE
# ================================================================================================================
print("\n[15/27] SURFACE teacher-forced...")
TEACH={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:
        p=apply_op(OPS[ax][L],SPRE[ax][L]);y=SRAW[ax][L]
        TEACH[ax][L]={"cos":float(cosv(p,y).mean()),"err":float(rerr(p,y).mean())}
    print(f"       {ax}: L20={TEACH[ax][20]['cos']:+.6f} L27={TEACH[ax][27]['cos']:+.6f}")

# ================================================================================================================
# [16/27] FREE-RUN SURFACE
# ================================================================================================================
print("\n[16/27] SURFACE free-run composition...")
PRED={ax:{} for ax in AXES};MET={ax:{} for ax in AXES}
for ax in AXES:
    x=SPRE[ax][20].clone();x0=x.clone()
    for L in TAIL:
        x=apply_op(OPS[ax][L],x);PRED[ax][L]=x.detach().clone()
        y=SRAW[ax][L];cp=cosv(x,y);ci=cosv(x0,y);ep=rerr(x,y);ei=rerr(x0,y)
        MET[ax][L]={"cos":float(cp.mean()),"identity_cos":float(ci.mean()),
                    "gain":float((cp-ci).mean()),"err":float(ep.mean()),
                    "identity_err":float(ei.mean()),"err_gain":float((ei-ep).mean()),
                    "Pplus":int(((cp-ci)>0).sum())}
    print(f"       {ax}: "+" ".join(f"L{L}={MET[ax][L]['cos']:+.3f}" for L in TAIL))

# ================================================================================================================
# [17/27] TERMINAL TRANSFER
# ================================================================================================================
print("\n[17/27] SURFACE RAW27 terminal transfer...")
TERM={}
for ax in AXES:
    m=MET[ax][27]
    TERM[ax]={"surface_cos":m["cos"],"identity_cos":m["identity_cos"],"gain":m["gain"],
              "surface_err":m["err"],"identity_err":m["identity_err"],"err_gain":m["err_gain"],
              "teacher_cos":TEACH[ax][27]["cos"],"free_teacher_gap":m["cos"]-TEACH[ax][27]["cos"],
              "Pplus":m["Pplus"],"historical_hold_cos":HREF[ax]["cos"],
              "surface_minus_hold":m["cos"]-HREF[ax]["cos"]}
    print(f"       {ax}: surface={m['cos']:+.6f} identity={m['identity_cos']:+.6f} gain={m['gain']:+.6f} "
          f"teacher={TEACH[ax][27]['cos']:+.6f} vs TEST194={HREF[ax]['cos']:+.6f}")

# ================================================================================================================
# [18/27] BOOTSTRAP TERMINAL GAIN
# ================================================================================================================
print("\n[18/27] Bootstrap SURFACE terminal Δcos...")
def boot(x,n=N_BOOT,seed=195):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT={}
for ai,ax in enumerate(AXES):
    g=(cosv(PRED[ax][27],SRAW[ax][27])-cosv(SPRE[ax][20],SRAW[ax][27])).cpu().numpy()
    m,lo,hi=boot(g,seed=SEED+ai);BOOT[ax]={"mean":m,"low":lo,"high":hi}
    print(f"       {ax}: {m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [19/27] EXACT SIGN-FLIP
# ================================================================================================================
print("\n[19/27] Exact n=8 sign-flip...")
def exact_signflip(x):
    x=np.asarray(x,dtype=np.float64);obs=abs(x.mean());hit=0;n=len(x)
    for mask in range(1<<n):
        s=np.asarray([1. if (mask>>i)&1 else -1. for i in range(n)])
        if abs(np.mean(x*s))>=obs-1e-15:hit+=1
    return hit/(1<<n)
PVAL={}
for ax in AXES:
    g=(cosv(PRED[ax][27],SRAW[ax][27])-cosv(SPRE[ax][20],SRAW[ax][27])).cpu().numpy()
    PVAL[ax]=exact_signflip(g)
    print(f"       {ax}: p={PVAL[ax]:.6f}")

# ================================================================================================================
# [20/27] FOREIGN-AXIS SURFACE CONTROL
# ================================================================================================================
print("\n[20/27] Foreign-axis composed SURFACE control...")
FOREIGN=np.zeros((5,5),dtype=np.float64);FERR=np.zeros((5,5),dtype=np.float64)
for ai,a in enumerate(AXES):
    for bi,b in enumerate(AXES):
        x=SPRE[b][20].clone()
        for L in TAIL:x=apply_op(OPS[a][L],x)
        FOREIGN[ai,bi]=float(cosv(x,SRAW[b][27]).mean())
        FERR[ai,bi]=float(rerr(x,SRAW[b][27]).mean())
print("             SURF-A    SURF-B    SURF-C    SURF-D    SURF-E")
for i,a in enumerate(AXES):print(f"       OP-{a} "+" ".join(f"{FOREIGN[i,j]:+9.5f}" for j in range(5)))

# ================================================================================================================
# [21/27] SPECIFICITY
# ================================================================================================================
print("\n[21/27] SURFACE axis specificity...")
SPEC={}
for i,a in enumerate(AXES):
    own=float(FOREIGN[i,i]);foreign=float(np.mean([FOREIGN[i,j] for j in range(5) if j!=i]))
    SPEC[a]={"own":own,"foreign":foreign,"gap":own-foreign}
    print(f"       {a}: own={own:+.6f} foreign={foreign:+.6f} gap={own-foreign:+.6f}")

# ================================================================================================================
# [22/27] TRANSFER RETENTION VS TEST194
# ================================================================================================================
print("\n[22/27] Novel-surface retention vs TEST194 HOLDOUT...")
RET={}
for ax in AXES:
    ratio=TERM[ax]["surface_cos"]/max(HREF[ax]["cos"],EPS)
    RET[ax]=ratio
    print(f"       {ax}: HOLD={HREF[ax]['cos']:+.6f} SURFACE={TERM[ax]['surface_cos']:+.6f} ratio={ratio:.6f}")

# ================================================================================================================
# [23/27] GLOBAL LAYER TRAJECTORY
# ================================================================================================================
print("\n[23/27] Global SURFACE trajectory...")
LSUM={}
for L in TAIL:
    cc=float(np.mean([MET[a][L]["cos"] for a in AXES]))
    tc=float(np.mean([TEACH[a][L]["cos"] for a in AXES]))
    ic=float(np.mean([MET[a][L]["identity_cos"] for a in AXES]))
    er=float(np.mean([MET[a][L]["err"] for a in AXES]))
    LSUM[L]={"composed":cc,"teacher":tc,"identity":ic,"err":er}
    print(f"       L{L}: composed={cc:+.6f} teacher={tc:+.6f} identity={ic:+.6f} err={er:.6f}")

# ================================================================================================================
# [24/27] DECISION
# ================================================================================================================
print("\n[24/27] TEST195 decision...")
REP={}
for ax in AXES:
    REP[ax]=(BOOT[ax]["low"]>0 and TERM[ax]["Pplus"]>=7 and SPEC[ax]["gap"]>0)
NREP=sum(REP.values());MEAN_RET=float(np.mean(list(RET.values())));MEAN_SPEC=float(np.mean([SPEC[a]["gap"] for a in AXES]))
if NREP==5:
    DECISION="NOVEL_SURFACE_AXIS_SPECIFIC_COMPOSED_TRANSPORT_REPLICATED"
elif NREP>=3:
    DECISION="NOVEL_SURFACE_COMPOSED_TRANSPORT_PARTIALLY_REPLICATED"
else:
    DECISION="NOVEL_SURFACE_TRANSPORT_GENERALIZATION_NOT_CONFIRMED"
for ax in AXES:print(f"       {ax}: replicated={REP[ax]}")
print("       DECISION:",DECISION)
print("       Third-set transfer only; no model-global/full-3584D dynamics claim.")

# ================================================================================================================
# [25/27] SAVE CSV
# ================================================================================================================
print("\n[25/27] Saving CSV...")
ROWS=[]
for ax in AXES:
    for L in TAIL:
        ROWS.append({"axis":ax,"layer":L,"surface_composed_cos":MET[ax][L]["cos"],
        "surface_teacher_cos":TEACH[ax][L]["cos"],"surface_identity_cos":MET[ax][L]["identity_cos"],
        "surface_gain":MET[ax][L]["gain"],"surface_err":MET[ax][L]["err"],
        "surface_identity_err":MET[ax][L]["identity_err"],"surface_err_gain":MET[ax][L]["err_gain"]})
with open(ROOT/"SURFACE_TRAJECTORY.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
print("       CSV ready")

# ================================================================================================================
# [26/27] SAVE NPZ / REPORT
# ================================================================================================================
print("\n[26/27] Saving NPZ / report...")
ARR={"foreign_surface_cos":FOREIGN,"foreign_surface_err":FERR}
for ax in AXES:
    for L in TAIL:
        ARR[f"SURFACE_PRE_{ax}_L{L}"]=SPRE[ax][L].cpu().numpy()
        ARR[f"SURFACE_RAW_{ax}_L{L}"]=SRAW[ax][L].cpu().numpy()
        ARR[f"SURFACE_PRED_{ax}_L{L}"]=PRED[ax][L].cpu().numpy()
        ARR[f"OP_X_{ax}_L{L}"]=OPS[ax][L]["X"].cpu().numpy()
        ARR[f"OP_A_{ax}_L{L}"]=OPS[ax][L]["A"].cpu().numpy()
np.savez_compressed(ROOT/"TEST195_NOVEL_SURFACE.npz",**ARR)
REPORT={"test":"TEST195","parent":"TEST194","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
"ridge_rel":RIDGE_REL,"historical_holdout_reference":HREF,"surface_teacher":TEACH,"surface_free_run":MET,
"terminal":TERM,"bootstrap":BOOT,"exact_signflip_p":PVAL,"specificity":SPEC,"retention_vs_holdout":RET,
"mean_retention":MEAN_RET,"mean_specificity":MEAN_SPEC,"replicated":REP,"layer_summary":LSUM,
"foreign_surface_cos":FOREIGN.tolist(),"foreign_surface_err":FERR.tolist(),
"integrity":{"test194_axis_lock_preserved":True,"test194_build_holdout_preserved":True,
"operators_fit_BUILD_only":True,"operators_frozen_before_SURFACE":True,"surface_used_for_fit":False,
"surface_used_for_selection":False,"surface_free_run_starts_true_PRE20_only":True,
"true_surface_intermediate_states_not_used_in_free_run":True,"seasc_l0_l19_frozen":True,
"tail_l20_l27_unmodified":True,"pca":False,"K_selection":False,"behavior_gradient":False,
"new_vector":False,"final":False,"controller":False,"weights_modified":False,
"full_3584D_dynamics_claim":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [27/27] COMPLETE
# ================================================================================================================
print("\n[27/27] COMPLETE")
print("="*138)
print("TEST 195 — NOVEL-SURFACE TAIL TRANSPORT TRANSFER — SUMMARY")
print("="*138)
for ax in AXES:
    print(f"{ax}: SURFACE={TERM[ax]['surface_cos']:+.6f} HOLD={HREF[ax]['cos']:+.6f} "
          f"retention={RET[ax]:.6f} identity={TERM[ax]['identity_cos']:+.6f} "
          f"gain={TERM[ax]['gain']:+.6f} BOOT=[{BOOT[ax]['low']:+.6f},{BOOT[ax]['high']:+.6f}] "
          f"p={PVAL[ax]:.6f} specificity={SPEC[ax]['gap']:+.6f}")
print(f"Replicated axes={NREP}/5 | mean transfer retention={MEAN_RET:.6f} | mean specificity={MEAN_SPEC:+.6f}")
for L in TAIL:
    print(f"L{L}: composed={LSUM[L]['composed']:+.6f} teacher={LSUM[L]['teacher']:+.6f} "
          f"identity={LSUM[L]['identity']:+.6f}")
print("SURFACE8 opened only after BUILD operators frozen.")
print("No SURFACE refit/selection | Full 3584D dynamics claim: False | New vector: False | FINAL: False")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
