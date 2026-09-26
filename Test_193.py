# ================================================================================================================
# TEST 193 — CROSS-AXIS TAIL LOCAL TRANSPORT OPERATOR ASSAY
# AkbasCore 3.2 / SEASC — TEST192 continuation
#
# QUESTION:
#   TEST192 found stable axis-specific rotating tail trajectories.
#   Does each unsteered block implement a prompt-general local transport law?
#
# METHOD:
#   For A/B/C/D/E:
#   1) preserve TEST192 compass + BUILD/HOLDOUT locks
#   2) capture PRE_L and RAW_L for L20-L27 under symmetric +/- SEASC
#   3) dPRE_L=.5*(PRE_L+ - PRE_L-)
#   4) dRAW_L=.5*(RAW_L+ - RAW_L-)
#   5) BUILD-only low-rank ridge operator predicts dRAW_L from dPRE_L
#   6) freeze operator before HOLDOUT
#   7) compare HOLDOUT prediction vs true RAW displacement
#   8) compare against identity baseline dRAW≈dPRE
#
# Operator is defined only on BUILD span; no claim of full 3584×3584 identification.
# NO PCA | NO K SELECTION | NO BEHAVIOR GRADIENT | NO NEW VECTOR | NO FINAL | NO CONTROLLER | NO WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 193 — CROSS-AXIS TAIL LOCAL TRANSPORT OPERATOR ASSAY")
print("A/B/C/D/E — L20→L27 — BUILD-FROZEN LOW-RANK LOCAL OPERATORS × INDEPENDENT HOLDOUT")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=193
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28;TAIL=list(range(20,28))
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
PRIMARY_SCALE=.50;EPS=1e-10;RIDGE_REL=1e-4;N_BOOT=20000
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST193_TAIL_LOCAL_OPERATOR");ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(STEER_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
def cosv(a,b):return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(EPS)
def relerr(a,b):return float((a-b).norm()/b.norm().clamp_min(EPS))
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
print(f"SEASC RSS={RSS:.9f} | scale={PRIMARY_SCALE} | ridge_rel={RIDGE_REL}")

# ================================================================================================================
# [1/26] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test193";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
print("\n[1/26] CUDA...")
seasc_ext=load_inline(name="akbascore_test193_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD,verbose=False)
print("       OK")

# ================================================================================================================
# [2/26] MODEL
# ================================================================================================================
print("\n[2/26] Model...")
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
# [3/26] TEST192 AXIS LOCK
# ================================================================================================================
print("\n[3/26] TEST192 axis lock...")
AXIS_PAIRS={
"A":[("The lamp is above the table.","The lamp is below the table."),("The bird is above the branch.","The bird is below the branch."),("The sign is above the doorway.","The sign is below the doorway."),("The shelf is above the cabinet.","The shelf is below the cabinet."),("The cloud is above the hill.","The cloud is below the hill."),("The clock is above the window.","The clock is below the window."),("The bridge is above the river.","The bridge is below the river."),("The picture is above the sofa.","The picture is below the sofa.")],
"B":[("The alarm occurs before the evacuation.","The alarm occurs after the evacuation."),("The flash occurs before the thunder.","The flash occurs after the thunder."),("The ignition occurs before the motion.","The ignition occurs after the motion."),("The warning occurs before the shutdown.","The warning occurs after the shutdown."),("The sunrise occurs before noon.","The sunrise occurs after noon."),("The opening occurs before the closing.","The opening occurs after the closing."),("The question occurs before the answer.","The question occurs after the answer."),("The launch occurs before the landing.","The launch occurs after the landing.")],
"C":[("The key is inside the box.","The key is outside the box."),("The coin is inside the jar.","The coin is outside the jar."),("The book is inside the bag.","The book is outside the bag."),("The ball is inside the basket.","The ball is outside the basket."),("The tool is inside the drawer.","The tool is outside the drawer."),("The letter is inside the envelope.","The letter is outside the envelope."),("The bottle is inside the cabinet.","The bottle is outside the cabinet."),("The card is inside the folder.","The card is outside the folder.")],
"D":[("The door is open.","The door is closed."),("The window is open.","The window is closed."),("The gate is open.","The gate is closed."),("The box is open.","The box is closed."),("The drawer is open.","The drawer is closed."),("The lid is open.","The lid is closed."),("The cabinet is open.","The cabinet is closed."),("The hatch is open.","The hatch is closed.")],
"E":[("The spark is the cause of the fire.","The fire is the effect of the spark."),("The impact is the cause of the crack.","The crack is the effect of the impact."),("The rainfall is the cause of the flooding.","The flooding is the effect of the rainfall."),("The heat is the cause of the melting.","The melting is the effect of the heat."),("The collision is the cause of the damage.","The damage is the effect of the collision."),("The pressure is the cause of the deformation.","The deformation is the effect of the pressure."),("The infection is the cause of the fever.","The fever is the effect of the infection."),("The vibration is the cause of the loosening.","The loosening is the effect of the vibration.")]}
AXES=list("ABCDE");AXIS_NAME={"A":"ABOVE↔BELOW","B":"BEFORE↔AFTER","C":"INSIDE↔OUTSIDE","D":"OPEN↔CLOSED","E":"CAUSE↔EFFECT"}
print("       A/B/C/D/E ready")

# ================================================================================================================
# [4/26] COMPASSES
# ================================================================================================================
print("\n[4/26] Layer-local compasses...")
COMPASS={}
for ax in AXES:
    P=[];N=[]
    for p,n in AXIS_PAIRS[ax]:P.append(capture(p));N.append(capture(n))
    COMPASS[ax]=norm(torch.stack(P).mean(0)-torch.stack(N).mean(0)).contiguous()
    print(f"       {ax} {AXIS_NAME[ax]} ready")

# ================================================================================================================
# [5/26] TEST192 PROMPT LOCK
# ================================================================================================================
print("\n[5/26] TEST192 BUILD/HOLDOUT lock...")
PROMPTS={
"A":{"B":["The red marker and blue marker are vertically separated.","Two lamps occupy different vertical positions.","A sign and a doorway have a vertical relation.","A bird and branch have a vertical relation.","A shelf and cabinet occupy different heights.","A cloud and hill have a vertical relation.","A clock and window occupy different heights.","A bridge and river have a vertical relation."],"H":["A picture and sofa occupy different vertical positions.","A flag and roof have a vertical relation.","A drone and building occupy different heights.","A balloon and tree have a vertical relation.","A ceiling light and desk occupy different heights.","A kite and field have a vertical relation.","A camera and tripod base occupy different heights.","A mountain peak and valley have a vertical relation."]},
"B":{"B":["An alarm and evacuation occur at different times.","A flash and thunder occur in temporal sequence.","Ignition and motion occur in temporal sequence.","A warning and shutdown occur at different times.","Sunrise and noon occur in temporal sequence.","Opening and closing occur at different times.","A question and answer occur in temporal sequence.","Launch and landing occur at different times."],"H":["A signal and response occur in temporal sequence.","A start and finish occur at different times.","Departure and arrival occur in temporal sequence.","A trigger and reaction occur at different times.","Heating and cooling occur in temporal sequence.","Entry and exit occur at different times.","Detection and reporting occur in temporal sequence.","Loading and unloading occur at different times."]},
"C":{"B":["A key and box have a containment relation.","A coin and jar have a containment relation.","A book and bag have a containment relation.","A ball and basket have a containment relation.","A tool and drawer have a containment relation.","A letter and envelope have a containment relation.","A bottle and cabinet have a containment relation.","A card and folder have a containment relation."],"H":["A toy and container have a containment relation.","A document and case have a containment relation.","A cable and conduit have a containment relation.","A package and vehicle have a containment relation.","A component and housing have a containment relation.","A seed and shell have a containment relation.","A file and archive have a containment relation.","A capsule and chamber have a containment relation."]},
"D":{"B":["A door has an open-closed state.","A window has an open-closed state.","A gate has an open-closed state.","A box has an open-closed state.","A drawer has an open-closed state.","A lid has an open-closed state.","A cabinet has an open-closed state.","A hatch has an open-closed state."],"H":["A valve has an open-closed state.","A shutter has an open-closed state.","A container has an open-closed state.","A cover has an open-closed state.","A barrier has an open-closed state.","A vent has an open-closed state.","A panel has an open-closed state.","A compartment has an open-closed state."]},
"E":{"B":["A fractured pipe and a pressure drop form a causal sequence.","A blocked air passage and rising temperature form a causal sequence.","A damaged seal and fluid loss form a causal sequence.","A current surge and a failed circuit form a causal sequence.","A stuck mechanism and restricted motion form a causal sequence.","A worn bearing and increased vibration form a causal sequence.","A clogged outlet and accumulated liquid form a causal sequence.","A strong impact and a deformed bracket form a causal sequence."],"H":["A broken coupling and loss of transmitted motion form a causal sequence.","A blocked intake and reduced airflow form a causal sequence.","A damaged gasket and pressure leakage form a causal sequence.","A failed coolant pump and rising temperature form a causal sequence.","A seized bearing and restricted rotation form a causal sequence.","A severed cable and loss of electrical power form a causal sequence.","A clogged nozzle and reduced discharge form a causal sequence.","A fractured support and structural displacement form a causal sequence."]}}
print("       5 axes × BUILD8 × HOLDOUT8")

# ================================================================================================================
# [6/26] SEASC
# ================================================================================================================
print("\n[6/26] SEASC...")
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
# [7/26] PRE/RAW TAIL RECORDER
# ================================================================================================================
print("\n[7/26] PRE/RAW L20-L27 recorder...")
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
    if len(box["pre"])!=8 or len(box["raw"])!=8:raise RuntimeError("Tail capture failed.")
    return box
print("       ready")

# ================================================================================================================
# [8/26] COLLECT
# ================================================================================================================
def collect(axis,prompts,label):
    PRE={L:[] for L in TAIL};RAW={L:[] for L in TAIL}
    for i,p in enumerate(prompts,1):
        P=run_tail(p,axis,+1);M=run_tail(p,axis,-1)
        for L in TAIL:
            PRE[L].append(.5*(P["pre"][L]-M["pre"][L]))
            RAW[L].append(.5*(P["raw"][L]-M["raw"][L]))
        print(f"       {axis} {label} {i}/8")
    return {L:torch.stack(PRE[L]) for L in TAIL},{L:torch.stack(RAW[L]) for L in TAIL}

print("\n[8/26] BUILD collection...")
BPRE={};BRAW={}
for ax in AXES:BPRE[ax],BRAW[ax]=collect(ax,PROMPTS[ax]["B"],"BUILD")

# ================================================================================================================
# [9/26] CONTINUITY
# ================================================================================================================
print("\n[9/26] Tail continuity...")
CONT={}
for ax in AXES:
    vals=[]
    for L in range(20,27):vals.append(relerr(BRAW[ax][L],BPRE[ax][L+1]))
    CONT[ax]=max(vals)
    print(f"       {ax}: max RAW[L]→PRE[L+1] relerr={CONT[ax]:.10f}")
if max(CONT.values())>1e-6:raise RuntimeError("Tail continuity failed.")

# ================================================================================================================
# [10/26] BUILD-ONLY LOW-RANK RIDGE OPERATORS
# ================================================================================================================
print("\n[10/26] Fit BUILD-only local operators...")
OPS={ax:{} for ax in AXES}
def fit_op(X,Y):
    X=X.float();Y=Y.float()
    G=X@X.T
    lam=RIDGE_REL*float(torch.trace(G)/G.shape[0])
    A=torch.linalg.solve(G+lam*torch.eye(G.shape[0],device=G.device),Y)
    return {"X":X.detach().clone(),"A":A.detach().clone(),"lambda":lam}
def apply_op(op,x):
    X,A=op["X"],op["A"];return (x@X.T)@A
for ax in AXES:
    for L in TAIL:OPS[ax][L]=fit_op(BPRE[ax][L],BRAW[ax][L])
    print(f"       {ax}: "+" ".join(f"L{L} λ={OPS[ax][L]['lambda']:.3e}" for L in TAIL))
print("       operators frozen before HOLDOUT")

# ================================================================================================================
# [11/26] BUILD FIT SANITY
# ================================================================================================================
print("\n[11/26] BUILD fit sanity...")
BFIT={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:
        pred=apply_op(OPS[ax][L],BPRE[ax][L]);true=BRAW[ax][L]
        c=cosv(pred,true);err=(pred-true).norm(dim=1)/true.norm(dim=1).clamp_min(EPS)
        BFIT[ax][L]={"cos":float(c.mean()),"relerr":float(err.mean())}
    print(f"       {ax}: L20 cos={BFIT[ax][20]['cos']:+.4f} err={BFIT[ax][20]['relerr']:.4f} | "
          f"L27 cos={BFIT[ax][27]['cos']:+.4f} err={BFIT[ax][27]['relerr']:.4f}")

# ================================================================================================================
# [12/26] LOFO OPERATOR VALIDATION
# ================================================================================================================
print("\n[12/26] BUILD LOFO operator validation...")
LOFO={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:
        cs=[];base=[];errs=[]
        for leave in range(8):
            idx=[i for i in range(8) if i!=leave]
            op=fit_op(BPRE[ax][L][idx],BRAW[ax][L][idx])
            x=BPRE[ax][L][leave:leave+1];y=BRAW[ax][L][leave:leave+1];p=apply_op(op,x)
            cs.append(float(cosv(p,y)));base.append(float(cosv(x,y)))
            errs.append(float((p-y).norm()/y.norm().clamp_min(EPS)))
        LOFO[ax][L]={"pred_cos":float(np.mean(cs)),"identity_cos":float(np.mean(base)),
                     "gain":float(np.mean(cs)-np.mean(base)),"relerr":float(np.mean(errs))}
    print(f"       {ax}: L20 gain={LOFO[ax][20]['gain']:+.4f} L27 gain={LOFO[ax][27]['gain']:+.4f}")

# ================================================================================================================
# [13/26] HOLDOUT COLLECTION
# ================================================================================================================
print("\n[13/26] HOLDOUT collection...")
HPRE={};HRAW={}
for ax in AXES:HPRE[ax],HRAW[ax]=collect(ax,PROMPTS[ax]["H"],"HOLD")

# ================================================================================================================
# [14/26] HOLDOUT OPERATOR TEST
# ================================================================================================================
print("\n[14/26] HOLDOUT operator prediction...")
HMET={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:
        x=HPRE[ax][L];y=HRAW[ax][L];p=apply_op(OPS[ax][L],x)
        cp=cosv(p,y);ci=cosv(x,y)
        ep=(p-y).norm(dim=1)/y.norm(dim=1).clamp_min(EPS)
        ei=(x-y).norm(dim=1)/y.norm(dim=1).clamp_min(EPS)
        HMET[ax][L]={"pred_cos":float(cp.mean()),"identity_cos":float(ci.mean()),
                     "cos_gain":float((cp-ci).mean()),"pred_relerr":float(ep.mean()),
                     "identity_relerr":float(ei.mean()),"err_gain":float((ei-ep).mean()),
                     "P_cos_gain":int(((cp-ci)>0).sum()),"P_err_gain":int(((ei-ep)>0).sum())}
    print(f"       {ax}: "+" ".join(f"L{L} Δcos={HMET[ax][L]['cos_gain']:+.3f}" for L in TAIL))

# ================================================================================================================
# [15/26] PER-LAYER HOLDOUT SUMMARY
# ================================================================================================================
print("\n[15/26] Layer summary...")
LAYER={}
for L in TAIL:
    pc=np.mean([HMET[a][L]["pred_cos"] for a in AXES])
    ic=np.mean([HMET[a][L]["identity_cos"] for a in AXES])
    cg=np.mean([HMET[a][L]["cos_gain"] for a in AXES])
    pe=np.mean([HMET[a][L]["pred_relerr"] for a in AXES])
    ie=np.mean([HMET[a][L]["identity_relerr"] for a in AXES])
    LAYER[L]={"pred_cos":float(pc),"identity_cos":float(ic),"cos_gain":float(cg),
              "pred_err":float(pe),"identity_err":float(ie),"err_gain":float(ie-pe)}
    print(f"       L{L}: pred cos={pc:+.6f} identity={ic:+.6f} Δ={cg:+.6f} pred err={pe:.6f} identity={ie:.6f}")

# ================================================================================================================
# [16/26] BOOTSTRAP HOLDOUT GAINS
# ================================================================================================================
print("\n[16/26] Bootstrap HOLDOUT Δcos...")
def boot(x,n=N_BOOT,seed=193):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT={ax:{} for ax in AXES}
for ai,ax in enumerate(AXES):
    for L in TAIL:
        x=HPRE[ax][L];y=HRAW[ax][L];p=apply_op(OPS[ax][L],x)
        g=(cosv(p,y)-cosv(x,y)).cpu().numpy()
        m,lo,hi=boot(g,seed=SEED+ai*20+L);BOOT[ax][L]={"mean":m,"low":lo,"high":hi}
    print(f"       {ax}: L20 {BOOT[ax][20]['mean']:+.4f}[{BOOT[ax][20]['low']:+.4f},{BOOT[ax][20]['high']:+.4f}] "
          f"L27 {BOOT[ax][27]['mean']:+.4f}[{BOOT[ax][27]['low']:+.4f},{BOOT[ax][27]['high']:+.4f}]")

# ================================================================================================================
# [17/26] EXACT SIGN-FLIP TEST
# ================================================================================================================
print("\n[17/26] Exact n=8 sign-flip tests...")
def exact_signflip(x):
    x=np.asarray(x,dtype=np.float64);obs=abs(x.mean());vals=[]
    for mask in range(1<<len(x)):
        s=np.asarray([1. if (mask>>i)&1 else -1. for i in range(len(x))])
        vals.append(abs(np.mean(x*s)))
    vals=np.asarray(vals);return float(np.mean(vals>=obs-1e-15))
PVAL={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:
        x=HPRE[ax][L];y=HRAW[ax][L];p=apply_op(OPS[ax][L],x)
        g=(cosv(p,y)-cosv(x,y)).cpu().numpy()
        PVAL[ax][L]=exact_signflip(g)
    print(f"       {ax}: L20 p={PVAL[ax][20]:.6f} L27 p={PVAL[ax][27]:.6f}")

# ================================================================================================================
# [18/26] FOREIGN-AXIS OPERATOR CONTROL
# ================================================================================================================
print("\n[18/26] Foreign-axis operator control...")
FOREIGN=np.zeros((5,5,8),dtype=np.float64)
for ai,a in enumerate(AXES):
    for bi,b in enumerate(AXES):
        vals=[]
        for L in TAIL:
            p=apply_op(OPS[a][L],HPRE[b][L]);y=HRAW[b][L]
            vals.append(float(cosv(p,y).mean()))
        FOREIGN[ai,bi]=np.asarray(vals)
print("       Mean L20-L27 prediction cosine:")
print("             HOLD-A    HOLD-B    HOLD-C    HOLD-D    HOLD-E")
for i,a in enumerate(AXES):print(f"       OP-{a} "+" ".join(f"{FOREIGN[i,j].mean():+9.5f}" for j in range(5)))

# ================================================================================================================
# [19/26] OWN VS FOREIGN SPECIFICITY
# ================================================================================================================
print("\n[19/26] Operator axis specificity...")
SPEC={}
for i,a in enumerate(AXES):
    own=float(FOREIGN[i,i].mean())
    foreign=float(np.mean([FOREIGN[i,j].mean() for j in range(5) if j!=i]))
    SPEC[a]={"own":own,"foreign":foreign,"gap":own-foreign}
    print(f"       {a}: own={own:+.6f} foreign={foreign:+.6f} gap={own-foreign:+.6f}")

# ================================================================================================================
# [20/26] TRAJECTORY LAW SCORE
# ================================================================================================================
print("\n[20/26] Trajectory-law replication...")
REP={ax:{} for ax in AXES}
for ax in AXES:
    for L in TAIL:
        h=HMET[ax][L];REP[ax][L]=(h["cos_gain"]>0 and h["err_gain"]>0 and h["P_cos_gain"]>=6)
    n=sum(REP[ax].values())
    print(f"       {ax}: replicated layers={n}/8 | "+",".join(f"L{L}" for L in TAIL if REP[ax][L]))

# ================================================================================================================
# [21/26] STRONGEST / WEAKEST LAYERS — DESCRIPTIVE ONLY
# ================================================================================================================
print("\n[21/26] Descriptive layer extremes...")
EXT={}
for ax in AXES:
    vals={L:HMET[ax][L]["cos_gain"] for L in TAIL}
    hi=max(vals,key=vals.get);lo=min(vals,key=vals.get)
    EXT[ax]={"max_layer":hi,"max_gain":vals[hi],"min_layer":lo,"min_gain":vals[lo]}
    print(f"       {ax}: max L{hi} {vals[hi]:+.6f} | min L{lo} {vals[lo]:+.6f}")
print("       descriptive only; no layer selected for intervention")

# ================================================================================================================
# [22/26] DECISION
# ================================================================================================================
print("\n[22/26] TEST193 decision...")
TOTAL=sum(sum(REP[a].values()) for a in AXES)
AXIS_REP={a:sum(REP[a].values())>=6 for a in AXES}
if all(AXIS_REP.values()) and np.mean([SPEC[a]["gap"] for a in AXES])>0:
    DECISION="AXIS_SPECIFIC_PROMPT_GENERAL_TAIL_TRANSPORT_LAWS_SUPPORTED"
elif TOTAL>=20:
    DECISION="TAIL_LOCAL_TRANSPORT_PREDICTABILITY_PARTIALLY_SUPPORTED"
else:
    DECISION="PROMPT_GENERAL_LOCAL_TRANSPORT_OPERATOR_NOT_CONFIRMED"
print(f"       replicated layer-axis cells={TOTAL}/40")
for a in AXES:print(f"       {a} axis-law replicated={AXIS_REP[a]}")
print("       DECISION:",DECISION)
print("       Operator is BUILD-span predictive map, not a full 3584D Jacobian.")

# ================================================================================================================
# [23/26] SAVE CSV
# ================================================================================================================
print("\n[23/26] Saving CSV...")
ROWS=[]
for ax in AXES:
    for L in TAIL:
        h=HMET[ax][L]
        ROWS.append({"axis":ax,"layer":L,"build_fit_cos":BFIT[ax][L]["cos"],
        "lofo_pred_cos":LOFO[ax][L]["pred_cos"],"lofo_identity_cos":LOFO[ax][L]["identity_cos"],
        "hold_pred_cos":h["pred_cos"],"hold_identity_cos":h["identity_cos"],"hold_cos_gain":h["cos_gain"],
        "hold_pred_relerr":h["pred_relerr"],"hold_identity_relerr":h["identity_relerr"],"hold_err_gain":h["err_gain"],
        "P_cos_gain":h["P_cos_gain"],"P_err_gain":h["P_err_gain"],"boot_low":BOOT[ax][L]["low"],
        "boot_high":BOOT[ax][L]["high"],"exact_signflip_p":PVAL[ax][L],"replicated":REP[ax][L]})
with open(ROOT/"LAYERWISE_OPERATOR.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
print("       CSV ready")

# ================================================================================================================
# [24/26] SAVE NPZ
# ================================================================================================================
print("\n[24/26] Saving NPZ...")
ARR={"foreign_operator_matrix":FOREIGN}
for ax in AXES:
    for L in TAIL:
        ARR[f"BPRE_{ax}_L{L}"]=BPRE[ax][L].cpu().numpy()
        ARR[f"BRAW_{ax}_L{L}"]=BRAW[ax][L].cpu().numpy()
        ARR[f"HPRE_{ax}_L{L}"]=HPRE[ax][L].cpu().numpy()
        ARR[f"HRAW_{ax}_L{L}"]=HRAW[ax][L].cpu().numpy()
        ARR[f"OP_X_{ax}_L{L}"]=OPS[ax][L]["X"].cpu().numpy()
        ARR[f"OP_A_{ax}_L{L}"]=OPS[ax][L]["A"].cpu().numpy()
np.savez_compressed(ROOT/"TEST193_LOCAL_OPERATORS.npz",**ARR)
print("       NPZ ready")

# ================================================================================================================
# [25/26] REPORT
# ================================================================================================================
print("\n[25/26] Saving report...")
REPORT={"test":"TEST193","parent":"TEST192","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
"ridge_rel":RIDGE_REL,"continuity":CONT,"build_fit":BFIT,"lofo":LOFO,"holdout":HMET,"bootstrap":BOOT,
"exact_signflip_p":PVAL,"specificity":SPEC,"replicated":REP,"axis_replicated":AXIS_REP,
"replicated_cells":TOTAL,"layer_summary":LAYER,"extremes_descriptive":EXT,
"foreign_matrix_mean_over_layers":FOREIGN.mean(axis=2).tolist(),
"integrity":{"test192_axis_definitions_preserved":True,"test192_prompts_preserved":True,
"seasc_l0_l19_frozen":True,"l20_l27_unmodified":True,"operators_fit_BUILD_only":True,
"operators_frozen_before_HOLDOUT":True,"operator_full_3584D_claim":False,
"pca_used":False,"K_selected":False,"behavior_gradient_used":False,"new_vector_injected":False,
"final_opened":False,"controller":False,"rescue":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [26/26] COMPLETE
# ================================================================================================================
print("\n[26/26] COMPLETE")
print("="*138)
print("TEST 193 — CROSS-AXIS TAIL LOCAL TRANSPORT OPERATOR — SUMMARY")
print("="*138)
for ax in AXES:
    print(f"{ax}: replicated={sum(REP[ax].values())}/8 own={SPEC[ax]['own']:+.6f} foreign={SPEC[ax]['foreign']:+.6f} "
          f"gap={SPEC[ax]['gap']:+.6f} L20Δcos={HMET[ax][20]['cos_gain']:+.6f} L27Δcos={HMET[ax][27]['cos_gain']:+.6f}")
print(f"Replicated layer-axis cells={TOTAL}/40")
for L in TAIL:
    print(f"L{L}: pred={LAYER[L]['pred_cos']:+.6f} identity={LAYER[L]['identity_cos']:+.6f} "
          f"Δcos={LAYER[L]['cos_gain']:+.6f} Δerr={LAYER[L]['err_gain']:+.6f}")
print("Operator: BUILD-span low-rank predictive map | Full 3584D Jacobian claim: False")
print("PCA: False | K selection: NONE | Behavior gradient: False | New vector: False | FINAL: False")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
