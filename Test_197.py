# ================================================================================================================
# TEST 197 — BUILD-SPAN CAPACITY SCALING
# AkbasCore 3.2 / SEASC — TEST196 continuation
#
# QUESTION:
#   Does controlled expansion of BUILD support increase novel-surface span coverage
#   and frozen-operator prediction quality?
#
# DESIGN:
#   Nested BUILD sizes: 8 → 16 → 24 → 32
#   Same SEASC L0-L19, same axes, same SURFACE8, same RIDGE_REL.
#   BUILD32 is fixed before SURFACE evaluation.
#   For every N and L20-L27:
#     fit BUILD-only operator M_L^(N)
#     measure rank and SURFACE BUILD-span coverage
#     teacher-forced prediction
#     free-run PRE20→RAW27 composition
#   Expected geometric chain:
#     BUILD support ↑ → span coverage ↑ → prediction ↑
#
# NO HYPERPARAMETER TUNING | NO SURFACE FIT | NO SELECTION | NO FINAL | NO WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 197 — BUILD-SPAN CAPACITY SCALING")
print("A/B/C/D/E — NESTED BUILD 8→16→24→32 × FIXED NOVEL SURFACE8")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=197
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28;TAIL=list(range(20,28))
BUILD_SIZES=[8,16,24,32]
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
PRIMARY_SCALE=.50;RIDGE_REL=1e-4;EPS=1e-10;N_BOOT=20000
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST197_BUILD_SPAN_CAPACITY");ROOT.mkdir(parents=True,exist_ok=True)

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
BUILD_DIR="/tmp/akbascore_test197";shutil.rmtree(BUILD_DIR,ignore_errors=True);os.makedirs(BUILD_DIR,exist_ok=True)
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
seasc_ext=load_inline(name="akbascore_test197_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD_DIR,verbose=False)
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
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True)
    p=int(e.attention_mask[0].sum())-1
    return torch.stack([o.hidden_states[L+1][0,p].float().detach() for L in range(STEER_LAYERS)])

# ================================================================================================================
# [3/27] AXIS LOCK
# ================================================================================================================
print("\n[3/27] TEST196 axis lock...")
AXIS_PAIRS={
"A":[("The lamp is above the table.","The lamp is below the table."),("The bird is above the branch.","The bird is below the branch."),("The sign is above the doorway.","The sign is below the doorway."),("The shelf is above the cabinet.","The shelf is below the cabinet."),("The cloud is above the hill.","The cloud is below the hill."),("The clock is above the window.","The clock is below the window."),("The bridge is above the river.","The bridge is below the river."),("The picture is above the sofa.","The picture is below the sofa.")],
"B":[("The alarm occurs before the evacuation.","The alarm occurs after the evacuation."),("The flash occurs before the thunder.","The flash occurs after the thunder."),("The ignition occurs before the motion.","The ignition occurs after the motion."),("The warning occurs before the shutdown.","The warning occurs after the shutdown."),("The sunrise occurs before noon.","The sunrise occurs after noon."),("The opening occurs before the closing.","The opening occurs after the closing."),("The question occurs before the answer.","The question occurs after the answer."),("The launch occurs before the landing.","The launch occurs after the landing.")],
"C":[("The key is inside the box.","The key is outside the box."),("The coin is inside the jar.","The coin is outside the jar."),("The book is inside the bag.","The book is outside the bag."),("The ball is inside the basket.","The ball is outside the basket."),("The tool is inside the drawer.","The tool is outside the drawer."),("The letter is inside the envelope.","The letter is outside the envelope."),("The bottle is inside the cabinet.","The bottle is outside the cabinet."),("The card is inside the folder.","The card is outside the folder.")],
"D":[("The door is open.","The door is closed."),("The window is open.","The window is closed."),("The gate is open.","The gate is closed."),("The box is open.","The box is closed."),("The drawer is open.","The drawer is closed."),("The lid is open.","The lid is closed."),("The cabinet is open.","The cabinet is closed."),("The hatch is open.","The hatch is closed.")],
"E":[("The spark is the cause of the fire.","The fire is the effect of the spark."),("The impact is the cause of the crack.","The crack is the effect of the impact."),("The rainfall is the cause of the flooding.","The flooding is the effect of the rainfall."),("The heat is the cause of the melting.","The melting is the effect of the heat."),("The collision is the cause of the damage.","The damage is the effect of the collision."),("The pressure is the cause of the deformation.","The deformation is the effect of the pressure."),("The infection is the cause of the fever.","The fever is the effect of the infection."),("The vibration is the cause of the loosening.","The loosening is the effect of the vibration.")]}
AXES=list("ABCDE")
AXIS_NAME={"A":"ABOVE↔BELOW","B":"BEFORE↔AFTER","C":"INSIDE↔OUTSIDE","D":"OPEN↔CLOSED","E":"CAUSE↔EFFECT"}
print("       ready")

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
# [5/27] NESTED BUILD32 + TEST195 SURFACE8
# ================================================================================================================
print("\n[5/27] Nested BUILD32 / SURFACE8 lock...")
BUILD32={
"A":[
"The red marker and blue marker are vertically separated.","Two lamps occupy different vertical positions.","A sign and a doorway have a vertical relation.","A bird and branch have a vertical relation.","A shelf and cabinet occupy different heights.","A cloud and hill have a vertical relation.","A clock and window occupy different heights.","A bridge and river have a vertical relation.",
"A flagpole tip and its base occupy different heights.","A ceiling and floor have a vertical relation.","A tower top and entrance occupy different heights.","A tree crown and roots have a vertical relation.","A rooftop and sidewalk occupy different heights.","A crane hook and ground have a vertical relation.","A mast top and deck occupy different heights.","A streetlight and curb have a vertical relation.",
"A chimney top and roof edge occupy different heights.","A cliff edge and beach have a vertical relation.","A bookshelf top and floor occupy different heights.","A stadium light and field have a vertical relation.","A monument peak and plaza occupy different heights.","A waterfall source and pool have a vertical relation.","A pole-mounted camera and road occupy different heights.","A hilltop and plain have a vertical relation.",
"A radio tower antenna and foundation occupy different heights.","A hanging sign and pavement have a vertical relation.","A skylight and room floor occupy different heights.","A castle turret and courtyard have a vertical relation.","A wind turbine hub and field occupy different heights.","A suspended cable and roadway have a vertical relation.","A balcony railing and garden occupy different heights.","A church spire and entrance have a vertical relation."],
"B":[
"An alarm and evacuation occur at different times.","A flash and thunder occur in temporal sequence.","Ignition and motion occur in temporal sequence.","A warning and shutdown occur at different times.","Sunrise and noon occur in temporal sequence.","Opening and closing occur at different times.","A question and answer occur in temporal sequence.","Launch and landing occur at different times.",
"Preparation and execution occur at different times.","An invitation and attendance occur in temporal sequence.","Charging and use occur at different times.","A request and approval occur in temporal sequence.","Construction and occupancy occur at different times.","A forecast and observation occur in temporal sequence.","A countdown and departure occur at different times.","An announcement and response occur in temporal sequence.",
"Enrollment and graduation occur at different times.","A command and completion occur in temporal sequence.","Assembly and operation occur at different times.","A reservation and arrival occur in temporal sequence.","A notification and acknowledgement occur at different times.","Calibration and measurement occur in temporal sequence.","A purchase and delivery occur at different times.","An application and decision occur in temporal sequence.",
"Initialization and termination occur at different times.","A proposal and acceptance occur in temporal sequence.","A scan and diagnosis occur at different times.","A rehearsal and presentation occur in temporal sequence.","A deposit and withdrawal occur at different times.","A transmission and reception occur in temporal sequence.","A booking and check-in occur at different times.","A draft and publication occur in temporal sequence."],
"C":[
"A key and box have a containment relation.","A coin and jar have a containment relation.","A book and bag have a containment relation.","A ball and basket have a containment relation.","A tool and drawer have a containment relation.","A letter and envelope have a containment relation.","A bottle and cabinet have a containment relation.","A card and folder have a containment relation.",
"A pencil and case have a containment relation.","A device and enclosure have a containment relation.","A parcel and locker have a containment relation.","A cable and sleeve have a containment relation.","A photograph and album have a containment relation.","A sample and container have a containment relation.","A disk and drive bay have a containment relation.","A garment and suitcase have a containment relation.",
"A component and chassis have a containment relation.","A document and binder have a containment relation.","A utensil and cupboard have a containment relation.","A tool and toolbox have a containment relation.","A chip and package have a containment relation.","A liquid and vessel have a containment relation.","A note and notebook have a containment relation.","A module and rack have a containment relation.",
"A sensor and housing have a containment relation.","A parcel and crate have a containment relation.","A lens and camera body have a containment relation.","A cartridge and holder have a containment relation.","A specimen and chamber have a containment relation.","A cable and channel have a containment relation.","A token and pouch have a containment relation.","A part and storage bin have a containment relation."],
"D":[
"A door has an open-closed state.","A window has an open-closed state.","A gate has an open-closed state.","A box has an open-closed state.","A drawer has an open-closed state.","A lid has an open-closed state.","A cabinet has an open-closed state.","A hatch has an open-closed state.",
"A locker has an open-closed state.","A suitcase has an open-closed state.","A garage door has an open-closed state.","A trapdoor has an open-closed state.","A case has an open-closed state.","A chest has an open-closed state.","A cupboard has an open-closed state.","A skylight has an open-closed state.",
"A mailbox has an open-closed state.","A storage bin has an open-closed state.","A roof vent has an open-closed state.","A screen panel has an open-closed state.","A sliding door has an open-closed state.","A service hatch has an open-closed state.","A cargo door has an open-closed state.","A tool chest has an open-closed state.",
"A fence gate has an open-closed state.","A roof hatch has an open-closed state.","A compartment door has an open-closed state.","A protective cover has an open-closed state.","A display case has an open-closed state.","A storage cabinet has an open-closed state.","A maintenance panel has an open-closed state.","A loading door has an open-closed state."],
"E":[
"A fractured pipe and a pressure drop form a causal sequence.","A blocked air passage and rising temperature form a causal sequence.","A damaged seal and fluid loss form a causal sequence.","A current surge and a failed circuit form a causal sequence.","A stuck mechanism and restricted motion form a causal sequence.","A worn bearing and increased vibration form a causal sequence.","A clogged outlet and accumulated liquid form a causal sequence.","A strong impact and a deformed bracket form a causal sequence.",
"A leaking hose and reduced pressure form a causal sequence.","A blocked drain and rising water level form a causal sequence.","A broken fan and increased temperature form a causal sequence.","A loose connector and intermittent power form a causal sequence.","A cracked tank and fluid loss form a causal sequence.","A jammed gear and stopped rotation form a causal sequence.","A dirty filter and reduced airflow form a causal sequence.","A bent shaft and increased vibration form a causal sequence.",
"A failed valve and uncontrolled flow form a causal sequence.","A damaged wire and loss of current form a causal sequence.","A blocked radiator and overheating form a causal sequence.","A worn belt and reduced drive form a causal sequence.","A punctured hose and pressure loss form a causal sequence.","A seized joint and restricted movement form a causal sequence.","A clogged pipe and reduced flow form a causal sequence.","A fractured mount and structural movement form a causal sequence.",
"A short circuit and power interruption form a causal sequence.","A blocked vent and heat accumulation form a causal sequence.","A damaged pump and reduced circulation form a causal sequence.","A loose fastener and mechanical play form a causal sequence.","A ruptured membrane and pressure loss form a causal sequence.","A contaminated bearing and increased friction form a causal sequence.","A blocked nozzle and reduced output form a causal sequence.","A broken support and component displacement form a causal sequence."]}

SURFACE={
"A":["Compare the altitude of a helicopter with the road beneath it.","Consider where a chandelier sits relative to the dining table.","Describe the vertical placement of a satellite relative to Earth's surface.","Compare the height of a rooftop antenna with the street.","Consider the position of an aircraft relative to the runway.","Describe where a canopy sits relative to the chairs underneath.","Compare a lighthouse lamp with the shoreline.","Consider the vertical relation between a balcony and the pavement."],
"B":["Consider the ordering of registration and admission.","Compare when a diagnosis occurs relative to treatment.","Describe the sequence connecting payment and shipment.","Consider the ordering of rehearsal and performance.","Compare inspection with subsequent certification.","Describe the temporal relation between planting and harvesting.","Consider the sequence of login and logout.","Compare data collection with later analysis."],
"C":["Consider whether a passenger occupies a vehicle cabin.","Describe the containment relation between medicine and a vial.","Consider a photograph relative to a frame.","Describe a battery relative to a device housing.","Consider luggage relative to an aircraft cargo hold.","Describe a cartridge relative to a printer.","Consider food relative to a sealed package.","Describe a specimen relative to a laboratory tube."],
"D":["Consider the state of a lockable safe with respect to access.","Describe whether a garage entrance permits passage or blocks it.","Consider the state of a bottle cap with respect to access to the contents.","Describe whether a tunnel barrier permits passage.","Consider the state of a laptop lid.","Describe whether a faucet valve permits flow or prevents it.","Consider the state of a shipping crate before or after sealing.","Describe whether an access port permits entry."],
"E":["A depleted battery and loss of device power are causally related.","A loose electrical connection and intermittent operation are causally related.","A frozen water line and interrupted flow are causally related.","Excessive load and structural bending are causally related.","A contaminated filter and reduced ventilation are causally related.","A software fault and an unexpected shutdown are causally related.","A punctured tire and falling air pressure are causally related.","A disconnected sensor and missing measurements are causally related."]}

for a in AXES:
    if len(BUILD32[a])!=32 or len(SURFACE[a])!=8:raise RuntimeError(f"Prompt count error: {a}")
print("       Nested BUILD sizes:",BUILD_SIZES)
print("       SURFACE8 fixed and excluded from BUILD")

# ================================================================================================================
# [6/27] SEASC
# ================================================================================================================
print("\n[6/27] SEASC...")
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
# [7/27] TAIL RECORDER
# ================================================================================================================
print("\n[7/27] Tail recorder...")
def run_tail(prompt,axis,sign):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    box={"pre":{},"raw":{}};hooks=seasc_hooks(axis,sign)
    for L in TAIL:
        def mkp(li):
            def hk(m,args):box["pre"][li]=args[0][0,pos].float().detach().clone()
            return hk
        def mkr(li):
            def hk(m,args,out):
                h=out[0] if isinstance(out,tuple) else out
                box["raw"][li]=h[0,pos].float().detach().clone()
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
        print(f"       {axis} {label} {i}/{len(prompts)}")
    return {L:torch.stack(PRE[L]) for L in TAIL},{L:torch.stack(RAW[L]) for L in TAIL}
print("       ready")

# ================================================================================================================
# [8/27] BUILD32 COLLECTION
# ================================================================================================================
print("\n[8/27] BUILD32 collection...")
BPRE={};BRAW={}
for ax in AXES:BPRE[ax],BRAW[ax]=collect(ax,BUILD32[ax],"BUILD32")

# ================================================================================================================
# [9/27] BUILD CONTINUITY
# ================================================================================================================
print("\n[9/27] BUILD continuity...")
for ax in AXES:
    m=max(float((BRAW[ax][L]-BPRE[ax][L+1]).norm()/BPRE[ax][L+1].norm().clamp_min(EPS)) for L in range(20,27))
    print(f"       {ax}: max relerr={m:.10f}")
    if m>1e-6:raise RuntimeError("BUILD continuity failed.")

# ================================================================================================================
# [10/27] SURFACE COLLECTION
# ================================================================================================================
print("\n[10/27] Fixed SURFACE8 collection...")
SPRE={};SRAW={}
for ax in AXES:SPRE[ax],SRAW[ax]=collect(ax,SURFACE[ax],"SURFACE")

# ================================================================================================================
# [11/27] SURFACE CONTINUITY
# ================================================================================================================
print("\n[11/27] SURFACE continuity...")
for ax in AXES:
    m=max(float((SRAW[ax][L]-SPRE[ax][L+1]).norm()/SPRE[ax][L+1].norm().clamp_min(EPS)) for L in range(20,27))
    print(f"       {ax}: max relerr={m:.10f}")
    if m>1e-6:raise RuntimeError("SURFACE continuity failed.")

# ================================================================================================================
# [12/27] OPERATOR / SPAN FUNCTIONS
# ================================================================================================================
print("\n[12/27] Operator/span functions...")
def fit_op(X,Y):
    X=X.float();Y=Y.float();G=X@X.T
    lam=RIDGE_REL*float(torch.trace(G)/G.shape[0])
    A=torch.linalg.solve(G+lam*torch.eye(G.shape[0],device=G.device),Y)
    return {"X":X.detach().clone(),"A":A.detach().clone(),"lambda":lam}

def apply_op(op,x):return (x@op["X"].T)@op["A"]

def row_basis(X):
    _,S,Vh=torch.linalg.svd(X.float(),full_matrices=False)
    tol=max(X.shape)*torch.finfo(torch.float32).eps*float(S.max())
    r=int((S>tol).sum())
    return Vh[:r].T.contiguous(),r

def project(Q,x):return (x@Q)@Q.T
print("       ready")

# ================================================================================================================
# [13/27] FIT ALL NESTED CAPACITIES
# ================================================================================================================
print("\n[13/27] Fit nested BUILD capacities...")
OPS={};BASIS={};RANK={}
for N in BUILD_SIZES:
    OPS[N]={};BASIS[N]={};RANK[N]={}
    for ax in AXES:
        OPS[N][ax]={};BASIS[N][ax]={};RANK[N][ax]={}
        for L in TAIL:
            X=BPRE[ax][L][:N];Y=BRAW[ax][L][:N]
            OPS[N][ax][L]=fit_op(X,Y)
            Q,r=row_basis(X);BASIS[N][ax][L]=Q;RANK[N][ax][L]=r
        print(f"       N={N:02d} {ax}: ranks "+" ".join(str(RANK[N][ax][L]) for L in TAIL))
print("       all capacities frozen")

# ================================================================================================================
# [14/27] SURFACE COVERAGE SCALING
# ================================================================================================================
print("\n[14/27] SURFACE coverage scaling...")
COVER={}
for N in BUILD_SIZES:
    COVER[N]={}
    for ax in AXES:
        COVER[N][ax]={}
        for L in TAIL:
            x=SPRE[ax][L];xp=project(BASIS[N][ax][L],x)
            c=xp.norm(dim=-1)/x.norm(dim=-1).clamp_min(EPS)
            e=xp.square().sum(-1)/x.square().sum(-1).clamp_min(EPS)
            COVER[N][ax][L]={"values":c.detach(),"mean":float(c.mean()),"energy":float(e.mean())}
        print(f"       N={N:02d} {ax}: mean={np.mean([COVER[N][ax][L]['mean'] for L in TAIL]):.6f}")

# ================================================================================================================
# [15/27] TEACHER-FORCED PREDICTION SCALING
# ================================================================================================================
print("\n[15/27] Teacher-forced prediction scaling...")
TEACH={}
for N in BUILD_SIZES:
    TEACH[N]={}
    for ax in AXES:
        TEACH[N][ax]={}
        for L in TAIL:
            p=apply_op(OPS[N][ax][L],SPRE[ax][L]);y=SRAW[ax][L]
            TEACH[N][ax][L]={"cos_values":cosv(p,y).detach(),
                             "cos":float(cosv(p,y).mean()),
                             "err":float(rerr(p,y).mean())}
        print(f"       N={N:02d} {ax}: L20={TEACH[N][ax][20]['cos']:+.4f} L27={TEACH[N][ax][27]['cos']:+.4f}")

# ================================================================================================================
# [16/27] FREE-RUN COMPOSITION SCALING
# ================================================================================================================
print("\n[16/27] Free-run PRE20→RAW27 scaling...")
FREE={}
for N in BUILD_SIZES:
    FREE[N]={}
    for ax in AXES:
        x=SPRE[ax][20].clone();traj={}
        for L in TAIL:
            x=apply_op(OPS[N][ax][L],x)
            traj[L]={"cos":float(cosv(x,SRAW[ax][L]).mean()),"err":float(rerr(x,SRAW[ax][L]).mean())}
        y=SRAW[ax][27];base=SPRE[ax][20]
        FREE[N][ax]={"traj":traj,"pred":x.detach(),
                     "cos_values":cosv(x,y).detach(),
                     "cos":float(cosv(x,y).mean()),
                     "err":float(rerr(x,y).mean()),
                     "identity":float(cosv(base,y).mean()),
                     "gain":float((cosv(x,y)-cosv(base,y)).mean())}
        print(f"       N={N:02d} {ax}: L27={FREE[N][ax]['cos']:+.6f} gain={FREE[N][ax]['gain']:+.6f}")

# ================================================================================================================
# [17/27] GLOBAL CAPACITY CURVE
# ================================================================================================================
print("\n[17/27] Global capacity curve...")
GLOBAL={}
for N in BUILD_SIZES:
    cov=float(np.mean([COVER[N][a][L]["mean"] for a in AXES for L in TAIL]))
    tc=float(np.mean([TEACH[N][a][L]["cos"] for a in AXES for L in TAIL]))
    fc=float(np.mean([FREE[N][a]["cos"] for a in AXES]))
    fe=float(np.mean([FREE[N][a]["err"] for a in AXES]))
    GLOBAL[N]={"coverage":cov,"teacher":tc,"free_terminal":fc,"free_err":fe}
    print(f"       N={N:02d}: coverage={cov:.6f} teacher={tc:+.6f} free_L27={fc:+.6f} free_err={fe:.6f}")

# ================================================================================================================
# [18/27] AXIS CAPACITY CURVES
# ================================================================================================================
print("\n[18/27] Axis capacity curves...")
AXCURVE={}
for ax in AXES:
    AXCURVE[ax]={}
    print(f"       {ax}:")
    for N in BUILD_SIZES:
        c=float(np.mean([COVER[N][ax][L]["mean"] for L in TAIL]))
        t=float(np.mean([TEACH[N][ax][L]["cos"] for L in TAIL]))
        f=FREE[N][ax]["cos"]
        AXCURVE[ax][N]={"coverage":c,"teacher":t,"free":f}
        print(f"          N={N:02d} cov={c:.6f} teacher={t:+.6f} free={f:+.6f}")

# ================================================================================================================
# [19/27] MONOTONICITY
# ================================================================================================================
print("\n[19/27] Monotonicity...")
MONO={}
for ax in AXES:
    cv=[AXCURVE[ax][N]["coverage"] for N in BUILD_SIZES]
    tv=[AXCURVE[ax][N]["teacher"] for N in BUILD_SIZES]
    fv=[AXCURVE[ax][N]["free"] for N in BUILD_SIZES]
    MONO[ax]={"coverage_steps":sum(cv[i+1]>=cv[i]-1e-8 for i in range(3)),
              "teacher_steps":sum(tv[i+1]>=tv[i]-1e-8 for i in range(3)),
              "free_steps":sum(fv[i+1]>=fv[i]-1e-8 for i in range(3)),
              "coverage_8_32":cv[-1]-cv[0],"teacher_8_32":tv[-1]-tv[0],"free_8_32":fv[-1]-fv[0]}
    print(f"       {ax}: cov {MONO[ax]['coverage_steps']}/3 Δ={MONO[ax]['coverage_8_32']:+.6f} | "
          f"teacher {MONO[ax]['teacher_steps']}/3 Δ={MONO[ax]['teacher_8_32']:+.6f} | "
          f"free {MONO[ax]['free_steps']}/3 Δ={MONO[ax]['free_8_32']:+.6f}")

# ================================================================================================================
# [20/27] SAMPLE-LEVEL ΔCOVERAGE → ΔPREDICTION
# ================================================================================================================
print("\n[20/27] N=8→32 sample-level Δcoverage→Δprediction...")
DC=[];DP=[]
for ax in AXES:
    for L in TAIL:
        c8=COVER[8][ax][L]["values"].cpu().numpy()
        c32=COVER[32][ax][L]["values"].cpu().numpy()
        p8=TEACH[8][ax][L]["cos_values"].cpu().numpy()
        p32=TEACH[32][ax][L]["cos_values"].cpu().numpy()
        DC.extend((c32-c8).tolist());DP.extend((p32-p8).tolist())

def pearson(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if np.std(x)<1e-12 or np.std(y)<1e-12:return float("nan")
    return float(np.corrcoef(x,y)[0,1])

def spearman(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float);ry=np.argsort(np.argsort(y)).astype(float)
    return pearson(rx,ry)

DELTA_CORR={"pearson":pearson(DC,DP),"spearman":spearman(DC,DP)}
print(f"       N={len(DC)} r={DELTA_CORR['pearson']:+.6f} rho={DELTA_CORR['spearman']:+.6f}")

# ================================================================================================================
# [21/27] BOOTSTRAP N=32 vs N=8 TERMINAL FREE-RUN
# ================================================================================================================
print("\n[21/27] Bootstrap N32−N8 terminal free-run...")
def boot(x,n=N_BOOT,seed=197):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))

BOOT={}
for i,ax in enumerate(AXES):
    d=(FREE[32][ax]["cos_values"]-FREE[8][ax]["cos_values"]).cpu().numpy()
    m,lo,hi=boot(d,seed=SEED+i);BOOT[ax]={"mean":m,"low":lo,"high":hi}
    print(f"       {ax}: Δ={m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [22/27] EXACT SIGN-FLIP N32−N8
# ================================================================================================================
print("\n[22/27] Exact n=8 sign-flip...")
def exact_signflip(x):
    x=np.asarray(x,float);obs=abs(x.mean());hit=0;n=len(x)
    for mask in range(1<<n):
        s=np.asarray([1. if (mask>>i)&1 else -1. for i in range(n)])
        if abs(np.mean(x*s))>=obs-1e-15:hit+=1
    return hit/(1<<n)

PVAL={}
for ax in AXES:
    d=(FREE[32][ax]["cos_values"]-FREE[8][ax]["cos_values"]).cpu().numpy()
    PVAL[ax]=exact_signflip(d)
    print(f"       {ax}: p={PVAL[ax]:.6f}")

# ================================================================================================================
# [23/27] LAYERWISE CAPACITY EFFECT
# ================================================================================================================
print("\n[23/27] Layerwise N8→N32 effect...")
LAYER={}
for L in TAIL:
    c8=np.mean([COVER[8][a][L]["mean"] for a in AXES])
    c32=np.mean([COVER[32][a][L]["mean"] for a in AXES])
    p8=np.mean([TEACH[8][a][L]["cos"] for a in AXES])
    p32=np.mean([TEACH[32][a][L]["cos"] for a in AXES])
    LAYER[L]={"cov8":float(c8),"cov32":float(c32),"dcov":float(c32-c8),
              "pred8":float(p8),"pred32":float(p32),"dpred":float(p32-p8)}
    print(f"       L{L}: coverage {c8:.4f}→{c32:.4f} Δ={c32-c8:+.4f} | "
          f"prediction {p8:.4f}→{p32:.4f} Δ={p32-p8:+.4f}")

# ================================================================================================================
# [24/27] DECISION
# ================================================================================================================
print("\n[24/27] TEST197 decision...")
AX_PASS={}
for ax in AXES:
    AX_PASS[ax]=(MONO[ax]["coverage_8_32"]>0 and MONO[ax]["teacher_8_32"]>0 and
                 MONO[ax]["free_8_32"]>0 and BOOT[ax]["low"]>0)
NPASS=sum(AX_PASS.values())
GLOBAL_COV_GAIN=GLOBAL[32]["coverage"]-GLOBAL[8]["coverage"]
GLOBAL_TEACH_GAIN=GLOBAL[32]["teacher"]-GLOBAL[8]["teacher"]
GLOBAL_FREE_GAIN=GLOBAL[32]["free_terminal"]-GLOBAL[8]["free_terminal"]
if NPASS==5 and GLOBAL_COV_GAIN>0 and GLOBAL_TEACH_GAIN>0 and GLOBAL_FREE_GAIN>0:
    DECISION="BUILD_SPAN_CAPACITY_SCALING_REPLICATED_ACROSS_ALL_AXES"
elif NPASS>=3 and GLOBAL_COV_GAIN>0 and GLOBAL_FREE_GAIN>0:
    DECISION="BUILD_SPAN_CAPACITY_SCALING_PARTIALLY_REPLICATED"
else:
    DECISION="BUILD_SPAN_CAPACITY_SCALING_NOT_CONFIRMED"
for ax in AXES:print(f"       {ax}: pass={AX_PASS[ax]}")
print("       DECISION:",DECISION)
print("       Capacity intervention is on measurement support only; transformer weights/tail remain untouched.")

# ================================================================================================================
# [25/27] SAVE CSV
# ================================================================================================================
print("\n[25/27] Saving CSV...")
ROWS=[]
for N in BUILD_SIZES:
    for ax in AXES:
        for L in TAIL:
            ROWS.append({"build_n":N,"axis":ax,"layer":L,"rank":RANK[N][ax][L],
                         "coverage":COVER[N][ax][L]["mean"],
                         "energy_coverage":COVER[N][ax][L]["energy"],
                         "teacher_cos":TEACH[N][ax][L]["cos"],
                         "teacher_err":TEACH[N][ax][L]["err"],
                         "free_terminal_cos":FREE[N][ax]["cos"],
                         "free_terminal_err":FREE[N][ax]["err"]})
with open(ROOT/"CAPACITY_SCALING.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
print("       CSV ready")

# ================================================================================================================
# [26/27] SAVE REPORT
# ================================================================================================================
print("\n[26/27] Saving report...")
REPORT={
"test":"TEST197","parent":"TEST196","decision":DECISION,"build_sizes":BUILD_SIZES,
"rss":RSS,"scale":PRIMARY_SCALE,"ridge_rel":RIDGE_REL,"global":GLOBAL,"axis_curves":AXCURVE,
"monotonicity":MONO,"bootstrap_32_minus_8":BOOT,"exact_signflip_p":PVAL,
"delta_coverage_prediction_correlation":DELTA_CORR,"layer_effect":LAYER,
"axis_pass":AX_PASS,"global_gains":{"coverage":GLOBAL_COV_GAIN,"teacher":GLOBAL_TEACH_GAIN,"free":GLOBAL_FREE_GAIN},
"rank":RANK,
"integrity":{"nested_build":True,"build8_exact_test196_seed_set":True,
"surface8_exact_test195_test196":True,"surface_excluded_from_build":True,
"same_surface_used_for_all_capacities":True,"ridge_rel_fixed":True,
"seasc_l0_l19_only":True,"tail_l20_l27_motor_off":True,
"no_surface_fit":True,"no_surface_selection":True,"no_hyperparameter_tuning":True,
"new_vector":False,"final":False,"weights_modified":False,
"full_3584D_operator_claim":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [27/27] COMPLETE
# ================================================================================================================
print("\n[27/27] COMPLETE")
print("="*138)
print("TEST 197 — BUILD-SPAN CAPACITY SCALING — SUMMARY")
print("="*138)
for N in BUILD_SIZES:
    print(f"N={N:02d}: coverage={GLOBAL[N]['coverage']:.6f} "
          f"teacher={GLOBAL[N]['teacher']:+.6f} "
          f"free_L27={GLOBAL[N]['free_terminal']:+.6f} "
          f"free_err={GLOBAL[N]['free_err']:.6f}")
print("-"*138)
for ax in AXES:
    print(f"{ax}: Δcoverage={MONO[ax]['coverage_8_32']:+.6f} "
          f"Δteacher={MONO[ax]['teacher_8_32']:+.6f} "
          f"Δfree={MONO[ax]['free_8_32']:+.6f} "
          f"BOOT=[{BOOT[ax]['low']:+.6f},{BOOT[ax]['high']:+.6f}] "
          f"p={PVAL[ax]:.6f}")
print("-"*138)
print(f"GLOBAL N8→N32 coverage gain={GLOBAL_COV_GAIN:+.6f}")
print(f"GLOBAL N8→N32 teacher gain={GLOBAL_TEACH_GAIN:+.6f}")
print(f"GLOBAL N8→N32 free-run gain={GLOBAL_FREE_GAIN:+.6f}")
print(f"Sample Δcoverage→Δprediction: r={DELTA_CORR['pearson']:+.6f} rho={DELTA_CORR['spearman']:+.6f}")
print("Nested BUILD only | Same SURFACE8 | Same ridge | L20-L27 motor OFF | No FINAL | No weight change")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
