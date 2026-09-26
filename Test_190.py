# ================================================================================================================
# TEST 190 — REPLICATED COMMON L27 TRANSPORT AXIS
# AkbasCore 3.2 / SEASC — TEST189 continuation — CAUSE↔EFFECT
#
# TEST189:
#   XRAY  mean ΔB27=-0.226119 | HOLD mean ΔB27=-0.483730
#   cos(mean XRAY Δ,mean HOLD Δ)=+0.973703
#   Behavior-sensitive component small (~1% norm); full-space behavioral transport only WEAK SIGNAL.
#
# QUESTION:
#   Does L27 contain a prompt-general COMMON TRANSPORT AXIS that replicates independently?
#
# BUILD:
#   Δ27_i = .5[(RAW27+ - RAW27-) - (PRE27+ - PRE27-)]
#   T27 = normalize(mean_i Δ27_i)
#
# TEST:
#   Freeze T27 using BUILD only.
#   Measure cos(Δ27,T27), projection Δ27·T27 and norm fraction on HOLDOUT-1 and independent HOLDOUT-2.
#   Also test BUILD LOFO stability, bootstrap CIs, sign consistency and isotropic random-direction null.
#
# NO PCA | NO K | NO GRADIENT SELECTION | NO NEW VECTOR INJECTION | NO FINAL | NO CONTROLLER | NO WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 190 — REPLICATED COMMON L27 TRANSPORT AXIS")
print("CAUSE↔EFFECT — BUILD-FROZEN MEAN Δ27 AXIS × TWO INDEPENDENT HOLDOUTS")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=190
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50;N_BOOT=20000;N_NULL=20000
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST190_COMMON_L27_TRANSPORT_AXIS");ROOT.mkdir(parents=True,exist_ok=True)

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
BUILD="/tmp/akbascore_test190";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
seasc_ext=load_inline(name="akbascore_test190_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
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
    return [o.hidden_states[L+1][0,p].float().detach() for L in range(STEER_LAYERS)]

def sequence(ctx,cont):
    p=tok(chat(ctx),return_tensors="pt",add_special_tokens=False)
    c=tok(cont,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p.input_ids,c.input_ids],1).to(DEVICE)
    return ids,torch.ones_like(ids),p.input_ids.shape[1]

def actgrad(ctx,cont,positive=False):
    ids,mask,plen=sequence(ctx,cont);emb=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    saved=[None]*STEER_LAYERS;hooks=[]
    for L in range(STEER_LAYERS):
        def mk(li):
            def hk(m,a,o):
                h=o[0] if isinstance(o,tuple) else o;h.retain_grad();saved[li]=h
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    model.zero_grad(set_to_none=True)
    try:
        o=model(inputs_embeds=emb,attention_mask=mask,use_cache=False,return_dict=True)
        z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:]
        sc=F.log_softmax(z,-1).gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean()
        (sc if positive else -sc).backward()
        g=[saved[L].grad[0].float().sum(0).detach().clone() for L in range(STEER_LAYERS)]
    finally:
        for h in hooks:h.remove()
        model.zero_grad(set_to_none=True)
    return g

# ================================================================================================================
# [3/24] FROZEN E AXIS
# ================================================================================================================
print("\n[3/24] Frozen E axis...")
PAIRS=[
("The spark is the cause of the fire.","The fire is the effect of the spark."),
("The impact is the cause of the crack.","The crack is the effect of the impact."),
("The rainfall is the cause of the flooding.","The flooding is the effect of the rainfall."),
("The heat is the cause of the melting.","The melting is the effect of the heat."),
("The collision is the cause of the damage.","The damage is the effect of the collision."),
("The pressure is the cause of the deformation.","The deformation is the effect of the pressure."),
("The infection is the cause of the fever.","The fever is the effect of the infection."),
("The vibration is the cause of the loosening.","The loosening is the effect of the vibration."),
("The frost is the cause of the cracking.","The cracking is the effect of the frost."),
("The blockage is the cause of the overflow.","The overflow is the effect of the blockage."),
("The friction is the cause of the heating.","The heating is the effect of the friction."),
("The force is the cause of the motion.","The motion is the effect of the force."),
("The outage is the cause of the shutdown.","The shutdown is the effect of the outage."),
("The leak is the cause of the pressure loss.","The pressure loss is the effect of the leak."),
("The acceleration is the cause of the displacement.","The displacement is the effect of the acceleration."),
("The trigger is the cause of the response.","The response is the effect of the trigger.")]

# ================================================================================================================
# [4/24] FISHER / BANK
# ================================================================================================================
print("\n[4/24] Fisher/Bank...")
PB=[[] for _ in range(STEER_LAYERS)];NB=[[] for _ in range(STEER_LAYERS)]
for i,(p,n) in enumerate(PAIRS,1):
    hp,hn=capture(p),capture(n)
    for L in range(STEER_LAYERS):PB[L].append(hp[L]);NB[L].append(hn[L])
    print(f"       {i:02d}/16")
PB=[torch.stack(x).float() for x in PB];NB=[torch.stack(x).float() for x in NB]
FISHER=[];BANK=[];FB=[]
for L in range(STEER_LAYERS):
    hp,hn=PB[L],NB[L];mp,mn=hp.mean(0),hn.mean(0);md=mp-mn;rr=torch.cat([hp-mp,hn-mn])
    _,s,vh=torch.linalg.svd(rr,full_matrices=False);eig=s.square()/max(rr.shape[0]-2,1)
    pe=eig[eig>FISHER_EPS];es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS);V0=vh.T;pr=V0.T@md
    f=norm(md/lam+V0@((1/(eig+lam)-1/lam)*pr)).float();dirs=norm(hp-hn)
    dirs*=torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
    b=norm(dirs.mean(0)).float();c=torch.dot(f,b)
    if c<0:b=-b;c=-c
    FISHER.append(f);BANK.append(b);FB.append(float(c))
FISHER=torch.stack(FISHER);BANK=torch.stack(BANK)
print(f"       mean cos(F,B)={np.mean(FB):+.6f}")

# ================================================================================================================
# [5/24] P8
# ================================================================================================================
print("\n[5/24] P8...")
RET=[
("A ceramic cup rests beside a folded newspaper."," The scene contains ordinary stationary objects."),
("A metal ruler lies beside a closed notebook."," The scene contains ordinary stationary objects."),
("A wooden bowl sits near a plain cloth."," The scene contains ordinary stationary objects."),
("A glass jar rests beside a cardboard package."," The scene contains ordinary stationary objects."),
("A calculator lies next to a paper envelope."," The objects remain stationary."),
("A closed umbrella rests beside a travel bag."," The objects remain stationary."),
("A spoon lies beside an empty plate."," The objects remain stationary."),
("A pencil rests next to a blank sheet of paper."," The objects remain stationary.")]
PG=[]
for i,x in enumerate(RET,1):PG.append(actgrad(*x));print(f"       {i}/8")
Q8=[];ranks=[]
for L in range(STEER_LAYERS):
    G0=norm(torch.stack([g[L] for g in PG]));_,s,vh=torch.linalg.svd(G0,full_matrices=False)
    r=int((s>max(float(s[0])*SVD_REL_TOL,EPS)).sum());Q8.append(vh[:r].T.contiguous());ranks.append(r)
print("       ranks:",ranks)
def guard(x):
    z=[]
    for L in range(STEER_LAYERS):
        q=Q8[L];v=x[L]-(q@(q.T@x[L]) if q.shape[1] else 0);z.append(norm(v))
    return torch.stack(z).contiguous()
FG=guard(FISHER)

# ================================================================================================================
# [6/24] FROZEN FORGE
# ================================================================================================================
print("\n[6/24] Frozen forge...")
FORGE={
"A":[
("A spark and a fire are related causally. Which item is the initiating event?"," The spark is the cause."," The fire is the effect."),
("An impact and a crack are related causally. Which item is the initiating event?"," The impact is the cause."," The crack is the effect."),
("Rainfall and flooding are related causally. Which item is the initiating event?"," The rainfall is the cause."," The flooding is the effect."),
("Heat and melting are related causally. Which item is the initiating event?"," The heat is the cause."," The melting is the effect.")],
"B":[
("A collision produces damage. Identify the causal role of the collision rather than its consequence."," The collision produces the damage."," The damage results from the collision."),
("Pressure produces deformation. Identify the causal role of the pressure rather than its consequence."," The pressure produces the deformation."," The deformation results from the pressure."),
("An infection produces fever. Identify the causal role of the infection rather than its consequence."," The infection produces the fever."," The fever results from the infection."),
("Vibration produces loosening. Identify the causal role of the vibration rather than its consequence."," The vibration produces the loosening."," The loosening results from the vibration.")],
"C":[
("Consider the directional causal relation between frost and cracking."," Causally, frost precedes and generates the cracking."," Causally, cracking follows from the frost."),
("Consider the directional causal relation between blockage and overflow."," Causally, blockage precedes and generates the overflow."," Causally, overflow follows from the blockage."),
("Consider the directional causal relation between friction and heating."," Causally, friction precedes and generates the heating."," Causally, heating follows from the friction."),
("Consider the directional causal relation between force and motion."," Causally, force precedes and generates the motion."," Causally, motion follows from the force.")],
"D":[
("An outage and a shutdown form a cause-effect pair. Select the upstream member."," The outage is upstream in the causal chain."," The shutdown is downstream in the causal chain."),
("A leak and pressure loss form a cause-effect pair. Select the upstream member."," The leak is upstream in the causal chain."," The pressure loss is downstream in the causal chain."),
("Acceleration and displacement form a cause-effect pair. Select the upstream member."," The acceleration is upstream in the causal chain."," The displacement is downstream in the causal chain."),
("A trigger and a response form a cause-effect pair. Select the upstream member."," The trigger is upstream in the causal chain."," The response is downstream in the causal chain.")]}
FAM={}
for fn,items in FORGE.items():
    acc=[torch.zeros(H,device=DEVICE) for _ in range(STEER_LAYERS)]
    for j,(ctx,p,n) in enumerate(items,1):
        gp,gn=actgrad(ctx,p,True),actgrad(ctx,n,True)
        for L in range(STEER_LAYERS):acc[L]+=gp[L]-gn[L]
        print(f"       {fn} {j}/4")
    v=torch.stack([norm(x) for x in acc])
    for L in range(STEER_LAYERS):
        if torch.dot(v[L],FISHER[L])<0:v[L]=-v[L]
    FAM[fn]=v
CAUSAL=torch.stack([norm(torch.stack([FAM[f][L] for f in FAM]).mean(0)) for L in range(STEER_LAYERS)])
CG=guard(CAUSAL);ORTH=[]
for L in range(STEER_LAYERS):ORTH.append(norm(CG[L]-torch.dot(CG[L],FG[L])*FG[L]))
ORTH=torch.stack(ORTH).contiguous();U=ORTH[19]
print("       ORTH ready | fixed coordinate=ORTH19")

# ================================================================================================================
# [7/24] FROZEN BUILD / HOLDOUT SETS
# ================================================================================================================
print("\n[7/24] Frozen BUILD / HOLDOUT sets...")
BUILD_X=[
"A fractured pipe and a pressure drop form a causal sequence.",
"A blocked air passage and rising temperature form a causal sequence.",
"A damaged seal and fluid loss form a causal sequence.",
"A current surge and a failed circuit form a causal sequence.",
"A stuck mechanism and restricted motion form a causal sequence.",
"A worn bearing and increased vibration form a causal sequence.",
"A clogged outlet and accumulated liquid form a causal sequence.",
"A strong impact and a deformed bracket form a causal sequence."]
HOLD1=[
"A broken coupling and loss of transmitted motion form a causal sequence.",
"A blocked intake and reduced airflow form a causal sequence.",
"A damaged gasket and pressure leakage form a causal sequence.",
"A failed coolant pump and rising temperature form a causal sequence.",
"A seized bearing and restricted rotation form a causal sequence.",
"A severed cable and loss of electrical power form a causal sequence.",
"A clogged nozzle and reduced discharge form a causal sequence.",
"A fractured support and structural displacement form a causal sequence."]
HOLD2=[
"A cracked reservoir and escaping fluid form a causal sequence.",
"A jammed valve and reduced flow form a causal sequence.",
"A failed heater and falling temperature form a causal sequence.",
"A broken gear and loss of rotation form a causal sequence.",
"A corroded contact and interrupted current form a causal sequence.",
"A blocked vent and increasing pressure form a causal sequence.",
"A loose fastener and mechanical vibration form a causal sequence.",
"A damaged insulation layer and heat leakage form a causal sequence."]
print("       BUILD=8 HOLDOUT1=8 HOLDOUT2=8")

# ================================================================================================================
# [8/24] SEASC HOOKS
# ================================================================================================================
print("\n[8/24] Frozen SEASC hooks...")
def seasc_hooks(sign):
    hooks=[]
    for L in range(STEER_LAYERS):
        def mk(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out
                B=raw.shape[0];a=(float(sign)*ORTH[li][None].expand(B,-1)).float().contiguous()
                d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                new=seasc_ext.seasc_batch(raw,a,d)
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    return hooks
print("       ready")

# ================================================================================================================
# [9/24] EXACT L27 TRANSPORT
# ================================================================================================================
print("\n[9/24] Exact L27 transport recorder...")
def run_transport(prompt,sign):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    box={};hooks=seasc_hooks(sign)
    def pre27(m,args):box["pre27"]=args[0][0,pos].float().detach().clone()
    def raw27(m,args,out):
        h=out[0] if isinstance(out,tuple) else out
        box["raw27"]=h[0,pos].float().detach().clone()
    hooks.append(layers[27].register_forward_pre_hook(pre27))
    hooks.append(layers[27].register_forward_hook(raw27))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    if "pre27" not in box or "raw27" not in box:raise RuntimeError("L27 transport capture failed.")
    return box

def collect(prompts,label):
    out=[]
    for i,p in enumerate(prompts,1):
        P=run_transport(p,+1);M=run_transport(p,-1)
        dp=.5*(P["pre27"]-M["pre27"]);dr=.5*(P["raw27"]-M["raw27"]);out.append(dr-dp)
        print(f"       {label} {i}/8")
    return torch.stack(out)
print("       ready")

# ================================================================================================================
# [10/24] BUILD TRANSPORT
# ================================================================================================================
print("\n[10/24] BUILD transport...")
BD=collect(BUILD_X,"BUILD")

# ================================================================================================================
# [11/24] FREEZE COMMON TRANSPORT AXIS
# ================================================================================================================
print("\n[11/24] Freeze common transport axis...")
BM=BD.mean(0);BMN=float(BM.norm())
if BMN<=EPS:raise RuntimeError("Degenerate BUILD mean transport.")
T27=norm(BM).detach().clone()
BPROJ=BD@T27;BCOS=cosv(BD,T27[None].expand_as(BD));BFRAC=BPROJ.abs()/BD.norm(dim=-1).clamp_min(EPS)
print(f"       ||mean Δ27||={BMN:.6f}")
print(f"       mean Δ·T27={float(BPROJ.mean()):+.6f}")
print(f"       mean cos(Δ,T27)={float(BCOS.mean()):+.6f}")
print(f"       P+={int((BPROJ>0).sum())}/8")
print(f"       mean axis norm frac={float(BFRAC.mean()):.6f}")
print(f"       cos(T27,ORTH19)={float(torch.dot(T27,U)):+.6f}")
print("       T27 FROZEN before HOLDOUTS")

# ================================================================================================================
# [12/24] BUILD LOFO
# ================================================================================================================
print("\n[12/24] BUILD LOFO...")
LOFO=[];LOFO_HELD=[]
for leave in range(8):
    idx=[i for i in range(8) if i!=leave]
    t=norm(BD[idx].mean(0))
    stab=float(torch.dot(t,T27));held=float(cosv(BD[leave],t))
    LOFO.append(stab);LOFO_HELD.append(held)
    print(f"       leave {leave+1}: axis cos={stab:+.6f} held cos={held:+.6f}")
print(f"       axis LOFO mean={np.mean(LOFO):+.6f} min={np.min(LOFO):+.6f}")
print(f"       held LOFO mean={np.mean(LOFO_HELD):+.6f} P+={sum(x>0 for x in LOFO_HELD)}/8")

# ================================================================================================================
# [13/24] HOLDOUT-1
# ================================================================================================================
print("\n[13/24] HOLDOUT-1...")
H1=collect(HOLD1,"H1")
H1P=H1@T27;H1C=cosv(H1,T27[None].expand_as(H1));H1F=H1P.abs()/H1.norm(dim=-1).clamp_min(EPS)
print(f"       mean Δ·T27={float(H1P.mean()):+.6f}")
print(f"       mean cos={float(H1C.mean()):+.6f}")
print(f"       P+={int((H1P>0).sum())}/8")
print(f"       mean axis norm frac={float(H1F.mean()):.6f}")

# ================================================================================================================
# [14/24] HOLDOUT-2
# ================================================================================================================
print("\n[14/24] HOLDOUT-2...")
H2=collect(HOLD2,"H2")
H2P=H2@T27;H2C=cosv(H2,T27[None].expand_as(H2));H2F=H2P.abs()/H2.norm(dim=-1).clamp_min(EPS)
print(f"       mean Δ·T27={float(H2P.mean()):+.6f}")
print(f"       mean cos={float(H2C.mean()):+.6f}")
print(f"       P+={int((H2P>0).sum())}/8")
print(f"       mean axis norm frac={float(H2F.mean()):.6f}")

# ================================================================================================================
# [15/24] CROSS-SET MEAN GEOMETRY
# ================================================================================================================
print("\n[15/24] Cross-set mean geometry...")
M1=norm(H1.mean(0));M2=norm(H2.mean(0))
C_BH1=float(torch.dot(T27,M1));C_BH2=float(torch.dot(T27,M2));C_H12=float(torch.dot(M1,M2))
print(f"       cos(BUILD mean,H1 mean)={C_BH1:+.6f}")
print(f"       cos(BUILD mean,H2 mean)={C_BH2:+.6f}")
print(f"       cos(H1 mean,H2 mean)   ={C_H12:+.6f}")
print(f"       H1 mean cos ORTH19={float(torch.dot(M1,U)):+.6f}")
print(f"       H2 mean cos ORTH19={float(torch.dot(M2,U)):+.6f}")

# ================================================================================================================
# [16/24] PER-PROMPT
# ================================================================================================================
print("\n[16/24] Per-prompt alignment...")
ROWS=[]
for label,X,P,C,F in [("BUILD",BD,BPROJ,BCOS,BFRAC),("HOLDOUT1",H1,H1P,H1C,H1F),("HOLDOUT2",H2,H2P,H2C,H2F)]:
    for i in range(8):
        r={"set":label,"item":i+1,"projection":float(P[i]),"cos_T27":float(C[i]),
           "axis_norm_fraction":float(F[i]),"delta_norm":float(X[i].norm()),"delta_orth19":float(X[i]@U)}
        ROWS.append(r)
        print(f"       {label:<8} {i+1}: proj={r['projection']:+.4f} cos={r['cos_T27']:+.4f} frac={r['axis_norm_fraction']:.4f}")

# ================================================================================================================
# [17/24] BOOTSTRAP
# ================================================================================================================
print("\n[17/24] Bootstrap...")
def boot(x,n=N_BOOT,seed=190):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT=[]
sets=[("BUILD",BPROJ,BCOS),("HOLDOUT1",H1P,H1C),("HOLDOUT2",H2P,H2C)]
for j,(label,p,c) in enumerate(sets):
    for k,(name,x) in enumerate([("PROJECTION",p.cpu().numpy()),("COS_T27",c.cpu().numpy())]):
        m,lo,hi=boot(x,seed=SEED+j*10+k);BOOT.append({"set":label,"metric":name,"mean":m,"low":lo,"high":hi})
        print(f"       {label:<8} {name:<10} {m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [18/24] ISOTROPIC RANDOM-DIRECTION NULL
# ================================================================================================================
print("\n[18/24] Isotropic random-direction null...")
def random_axis_null(X,observed,n=N_NULL,seed=190):
    rng=torch.Generator(device=DEVICE);rng.manual_seed(seed)
    cnt=0;vals=[];batch=500
    for start in range(0,n,batch):
        m=min(batch,n-start)
        R=torch.randn((m,H),device=DEVICE,dtype=torch.float32,generator=rng);R=norm(R)
        v=cosv(X.mean(0)[None].expand(m,-1),R)
        cnt+=int((v.abs()>=abs(observed)-1e-12).sum());vals.append(v.detach().cpu())
    vals=torch.cat(vals).numpy()
    return (cnt+1)/(n+1),float(np.quantile(np.abs(vals),.95)),float(np.quantile(np.abs(vals),.99))
P1,N195,N199=random_axis_null(H1,C_BH1,seed=SEED+1)
P2,N295,N299=random_axis_null(H2,C_BH2,seed=SEED+2)
print(f"       H1 observed mean-axis cos={C_BH1:+.6f} null-p={P1:.6f} | |cos|95={N195:.6f} 99={N199:.6f}")
print(f"       H2 observed mean-axis cos={C_BH2:+.6f} null-p={P2:.6f} | |cos|95={N295:.6f} 99={N299:.6f}")

# ================================================================================================================
# [19/24] SIGN-FLIP NULL
# ================================================================================================================
print("\n[19/24] Sign-flip null...")
def signflip(x,n=50000,seed=190):
    x=np.asarray(x,dtype=np.float64);obs=abs(x.mean());rng=np.random.default_rng(seed);cnt=0
    for _ in range(n):
        s=rng.choice(np.array([-1.,1.]),size=len(x))
        if abs(float((x*s).mean()))>=obs-1e-15:cnt+=1
    return (cnt+1)/(n+1)
S1=signflip(H1P.cpu().numpy(),seed=SEED+10);S2=signflip(H2P.cpu().numpy(),seed=SEED+20)
print(f"       HOLDOUT1 projection sign-flip p={S1:.6f}")
print(f"       HOLDOUT2 projection sign-flip p={S2:.6f}")

# ================================================================================================================
# [20/24] ORTH19 COMPARISON
# ================================================================================================================
print("\n[20/24] ORTH19 comparison...")
for label,X in [("BUILD",BD),("HOLDOUT1",H1),("HOLDOUT2",H2)]:
    du=X@U;cu=cosv(X,U[None].expand_as(X))
    print(f"       {label:<8} mean Δ@ORTH19={float(du.mean()):+.6f} mean cos={float(cu.mean()):+.6f}")
print(f"       cos(T27,ORTH19)={float(torch.dot(T27,U)):+.6f}")

# ================================================================================================================
# [21/24] REPLICATION CRITERIA
# ================================================================================================================
print("\n[21/24] Replication criteria...")
B1=next(x for x in BOOT if x["set"]=="HOLDOUT1" and x["metric"]=="COS_T27")
B2=next(x for x in BOOT if x["set"]=="HOLDOUT2" and x["metric"]=="COS_T27")
R1=B1["low"]>0 and int((H1C>0).sum())>=7 and C_BH1>0
R2=B2["low"]>0 and int((H2C>0).sum())>=7 and C_BH2>0
print(f"       H1 bootstrap cos low>0: {B1['low']>0} | P+={int((H1C>0).sum())}/8 | mean-axis={C_BH1:+.6f}")
print(f"       H2 bootstrap cos low>0: {B2['low']>0} | P+={int((H2C>0).sum())}/8 | mean-axis={C_BH2:+.6f}")
print(f"       H1 replicated={R1}")
print(f"       H2 replicated={R2}")

# ================================================================================================================
# [22/24] DECISION
# ================================================================================================================
print("\n[22/24] TEST190 decision...")
if R1 and R2:
    DECISION="COMMON_L27_TRANSPORT_AXIS_REPLICATED_ACROSS_TWO_INDEPENDENT_HOLDOUTS"
elif R1 or R2:
    DECISION="COMMON_L27_TRANSPORT_AXIS_PARTIAL_REPLICATION"
else:
    DECISION="COMMON_L27_TRANSPORT_AXIS_NOT_CONFIRMED"
print("       DECISION:",DECISION)
print("       Mechanistic transport result only; no behavioral or causal-intervention claim.")

# ================================================================================================================
# [23/24] SAVE
# ================================================================================================================
print("\n[23/24] Saving...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("PER_PROMPT.csv",ROWS);savecsv("BOOTSTRAP.csv",BOOT)
np.savez_compressed(ROOT/"TEST190_COMMON_TRANSPORT_AXIS.npz",
 orth19=U.cpu().numpy(),T27=T27.cpu().numpy(),build_delta=BD.cpu().numpy(),
 holdout1_delta=H1.cpu().numpy(),holdout2_delta=H2.cpu().numpy())
REPORT={"test":"TEST190","parent":"TEST189","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
"mean_fb":float(np.mean(FB)),"build":{"mean_norm":BMN,"mean_cos":float(BCOS.mean()),"Pplus":int((BCOS>0).sum()),
"lofo_axis_mean":float(np.mean(LOFO)),"lofo_axis_min":float(np.min(LOFO)),"lofo_held_mean":float(np.mean(LOFO_HELD))},
"holdout1":{"mean_projection":float(H1P.mean()),"mean_cos":float(H1C.mean()),"Pplus":int((H1C>0).sum()),
"mean_axis_cos":C_BH1,"random_null_p":P1,"signflip_p":S1},
"holdout2":{"mean_projection":float(H2P.mean()),"mean_cos":float(H2C.mean()),"Pplus":int((H2C>0).sum()),
"mean_axis_cos":C_BH2,"random_null_p":P2,"signflip_p":S2},
"cross":{"H1_H2_mean_cos":C_H12,"T27_ORTH19":float(torch.dot(T27,U))},"bootstrap":BOOT,
"integrity":{"test189_architecture_preserved":True,"seasc_l0_l19_frozen":True,"l20_l27_unmodified":True,
"T27_built_from_BUILD_only":True,"T27_frozen_before_holdouts":True,"pca_used":False,"K_selection":False,
"gradient_selection":False,"new_vector_injected":False,"final_opened":False,"controller":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [24/24] COMPLETE
# ================================================================================================================
print("\n[24/24] COMPLETE")
print("="*138)
print("TEST 190 — REPLICATED COMMON L27 TRANSPORT AXIS — SUMMARY")
print("="*138)
print(f"mean cos(F,B)={np.mean(FB):+.6f}")
print(f"BUILD: mean cos(Δ,T27)={float(BCOS.mean()):+.6f} LOFO-axis-min={np.min(LOFO):+.6f} LOFO-held-mean={np.mean(LOFO_HELD):+.6f}")
print(f"HOLD1: mean cos={float(H1C.mean()):+.6f} P+={int((H1C>0).sum())}/8 mean-axis={C_BH1:+.6f} signflip-p={S1:.6f}")
print(f"HOLD2: mean cos={float(H2C.mean()):+.6f} P+={int((H2C>0).sum())}/8 mean-axis={C_BH2:+.6f} signflip-p={S2:.6f}")
print(f"cos(H1 mean,H2 mean)={C_H12:+.6f}")
print(f"cos(T27,ORTH19)={float(torch.dot(T27,U)):+.6f}")
print("PCA used: False | K selected: NONE | Gradient selection: False | New vector injected: False | FINAL opened: False")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
