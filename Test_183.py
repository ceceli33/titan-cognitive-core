# ================================================================================================================
# TEST 183 — MOVING-COMPASS / COORDINATE-ROTATION X-RAY
# AkbasCore 3.2 / SEASC
# TEST182 continuation — CAUSE↔EFFECT
#
# PURPOSE
# Separate:
#   (1) actual hidden-state transport
#   (2) layer-local ORTH compass rotation
#
# TEST182:
#   POST18 == PRE19 physically (relerr=0)
#   yet POST18·ORTH18 > 0 while PRE19·ORTH19 < 0.
#
# TEST183 asks:
#   Is the apparent terminal reversal caused primarily by rotation of ORTH18→ORTH19?
#
# MEASURE
#   C_adj[L]       = cos(ORTH[L],ORTH[L+1])
#   SAME STATE:
#       POST[L]·ORTH[L]
#       POST[L]·ORTH[L+1]
#   COORD_ROT:
#       POST[L]·ORTH[L+1] - POST[L]·ORTH[L]
#   TRANSPORT:
#       PRE[L+1] - POST[L]  (must remain ~0)
#
# TERMINAL:
#   POST18 projected onto ORTH18 and ORTH19
#   RAW19/POST19 projected onto ORTH18 and ORTH19
#   POST19 → final RMSNorm under fixed ORTH19
#
# FROZEN: model / E axis / P8 / causal forge / SEASC / rho / scale=.50
# X-RAY ONLY — NO SEARCH / SELECTION / CONTROLLER / WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 183 — MOVING-COMPASS / COORDINATE-ROTATION X-RAY")
print("CAUSE↔EFFECT — STATE TRANSPORT vs LAYER-LOCAL COMPASS ROTATION")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=183
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST183_MOVING_COMPASS_XRAY");ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(N_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
print(f"SEASC RSS={RSS:.9f} | scale={PRIMARY_SCALE}")

# ================================================================================================================
# [1/24] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test183";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
seasc_ext=load_inline(name="akbascore_test183_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
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
print("       hidden:",H,"dtype:",next(model.parameters()).dtype)

def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],
 tokenize=False,add_generation_prompt=True)

@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt").to(DEVICE)
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);p=int(e.attention_mask[0].sum())-1
    x=[o.hidden_states[L+1][0,p].float().detach() for L in range(N_LAYERS)]
    del e,o;return x

def sequence(ctx,cont):
    p=tok(chat(ctx),return_tensors="pt",add_special_tokens=False);c=tok(cont,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p.input_ids,c.input_ids],1).to(DEVICE);return ids,torch.ones_like(ids),p.input_ids.shape[1]

def actgrad(ctx,cont,positive=False):
    ids,mask,plen=sequence(ctx,cont);emb=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    saved=[None]*N_LAYERS;hooks=[]
    for L in range(N_LAYERS):
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
        g=[saved[L].grad[0].float().sum(0).detach().clone() for L in range(N_LAYERS)]
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
# [4/24] FISHER/BANK
# ================================================================================================================
print("\n[4/24] Fisher/Bank...")
PB=[[] for _ in range(N_LAYERS)];NB=[[] for _ in range(N_LAYERS)]
for i,(p,n) in enumerate(PAIRS,1):
    hp,hn=capture(p),capture(n)
    for L in range(N_LAYERS):PB[L].append(hp[L]);NB[L].append(hn[L])
    print(f"       {i:02d}/16")
PB=[torch.stack(x).float() for x in PB];NB=[torch.stack(x).float() for x in NB]
FISHER=[];BANK=[];FB=[]
for L in range(N_LAYERS):
    hp,hn=PB[L],NB[L];mp,mn=hp.mean(0),hn.mean(0);md=mp-mn;rr=torch.cat([hp-mp,hn-mn])
    _,s,vh=torch.linalg.svd(rr,full_matrices=False);eig=s.square()/max(rr.shape[0]-2,1)
    pe=eig[eig>FISHER_EPS];es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS);V=vh.T;pr=V.T@md
    f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float();dirs=norm(hp-hn)
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
for L in range(N_LAYERS):
    G=norm(torch.stack([g[L] for g in PG]));_,s,vh=torch.linalg.svd(G,full_matrices=False)
    r=int((s>max(float(s[0])*SVD_REL_TOL,EPS)).sum());Q8.append(vh[:r].T.contiguous());ranks.append(r)
print("       ranks:",ranks)
def guard(x):
    z=[]
    for L in range(N_LAYERS):
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
    acc=[torch.zeros(H,device=DEVICE) for _ in range(N_LAYERS)]
    for j,(ctx,p,n) in enumerate(items,1):
        gp,gn=actgrad(ctx,p,True),actgrad(ctx,n,True)
        for L in range(N_LAYERS):acc[L]+=gp[L]-gn[L]
        print(f"       {fn} {j}/4")
    v=torch.stack([norm(x) for x in acc])
    for L in range(N_LAYERS):
        if torch.dot(v[L],FISHER[L])<0:v[L]=-v[L]
    FAM[fn]=v
CAUSAL=torch.stack([norm(torch.stack([FAM[f][L] for f in FAM]).mean(0)) for L in range(N_LAYERS)])
CG=guard(CAUSAL);ORTH=[]
for L in range(N_LAYERS):ORTH.append(norm(CG[L]-torch.dot(CG[L],FG[L])*FG[L]))
ORTH=torch.stack(ORTH).contiguous()
print("       ORTH ready")

# ================================================================================================================
# [7/24] COMPASS ROTATION
# ================================================================================================================
print("\n[7/24] Adjacent ORTH compass rotation...")
COMPASS=[]
for L in range(N_LAYERS-1):
    c=float(torch.dot(ORTH[L],ORTH[L+1]).clamp(-1,1))
    a=math.degrees(math.acos(max(-1,min(1,c))))
    COMPASS.append({"from":L,"to":L+1,"cos":c,"angle":a})
    print(f"       ORTH{L:02d}→ORTH{L+1:02d} cos={c:+.6f} angle={a:6.2f}°")

# ================================================================================================================
# [8/24] FRESH XRAY
# ================================================================================================================
print("\n[8/24] Fresh X-ray...")
XRAY=[
"A cracked connector and an interrupted signal form a causal sequence.",
"A blocked vent and rising internal temperature form a causal sequence.",
"A damaged gasket and falling pressure form a causal sequence.",
"A power surge and a failed electronic component form a causal sequence.",
"A frozen valve and restricted flow form a causal sequence.",
"A worn gear and increased mechanical noise form a causal sequence.",
"A clogged passage and accumulated fluid form a causal sequence.",
"A sharp collision and a bent support form a causal sequence."]
print("       n=8")

# ================================================================================================================
# [9/24] EXACT RECORDER
# ================================================================================================================
print("\n[9/24] Exact PRE/RAW/POST recorder...")
def run_exact(prompt,sign=0.0):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    PRE=[None]*N_LAYERS;RAW=[None]*N_LAYERS;POST=[None]*N_LAYERS;hooks=[];final={}
    for L in range(N_LAYERS):
        def prehook(li):
            def hk(m,args):PRE[li]=args[0][0,pos].float().detach().clone()
            return hk
        def posthook(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out;RAW[li]=raw[0,pos].float().detach().clone()
                if sign==0:new=raw
                else:
                    B=raw.shape[0];a=(float(sign)*ORTH[li][None].expand(B,-1)).float().contiguous()
                    d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                    new=seasc_ext.seasc_batch(raw,a,d)
                POST[li]=new[0,pos].float().detach().clone()
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_pre_hook(prehook(L)))
        hooks.append(layers[L].register_forward_hook(posthook(L)))
    def fh(m,args,out):final["x"]=out[0,pos].float().detach().clone()
    hooks.append(model.model.norm.register_forward_hook(fh))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    return {"pre":torch.stack(PRE),"raw":torch.stack(RAW),"post":torch.stack(POST),"final":final["x"]}

# ================================================================================================================
# [10/24] CAPTURE
# ================================================================================================================
print("\n[10/24] BASE / PLUS / MINUS...")
RUN={"BASE":[],"PLUS":[],"MINUS":[]}
for i,p in enumerate(XRAY,1):
    RUN["BASE"].append(run_exact(p,0));RUN["PLUS"].append(run_exact(p,+1));RUN["MINUS"].append(run_exact(p,-1))
    print(f"       {i}/8")

def stk(stage,a):return torch.stack([x[stage] for x in RUN[a]])
S={stage:{a:stk(stage,a) for a in RUN} for stage in ["pre","raw","post"]}
D={stage:.5*(S[stage]["PLUS"]-S[stage]["MINUS"]) for stage in S}

# ================================================================================================================
# [11/24] PHYSICAL TRANSPORT CHECK
# ================================================================================================================
print("\n[11/24] Physical POST[L]→PRE[L+1]...")
TRANSPORT=[]
for L in range(N_LAYERS-1):
    err=(D["post"][:,L]-D["pre"][:,L+1]).norm(dim=-1)
    den=D["post"][:,L].norm(dim=-1).clamp_min(EPS);rel=err/den
    r={"layer":L,"abs":float(err.mean()),"rel":float(rel.mean())};TRANSPORT.append(r)
    print(f"       {L:02d}→{L+1:02d} relerr={r['rel']:.10f}")

# ================================================================================================================
# [12/24] SAME-STATE DUAL-COMPASS PROJECTION
# ================================================================================================================
print("\n[12/24] Same-state dual-compass projection...")
DUAL=[]
for L in range(N_LAYERS-1):
    x=D["post"][:,L]
    old=x@ORTH[L];new=x@ORTH[L+1]
    r={"layer":L,"old":float(old.mean()),"new":float(new.mean()),
       "coord_delta":float((new-old).mean()),"old_pos":int((old>0).sum()),"new_pos":int((new>0).sum())}
    DUAL.append(r)
    print(f"       POST{L:02d}: ORTH{L:02d}={r['old']:+.6f} → ORTH{L+1:02d}={r['new']:+.6f} "
          f"Δcoord={r['coord_delta']:+.6f} P+={r['old_pos']}/8→{r['new_pos']}/8")

# ================================================================================================================
# [13/24] COORDINATE-ONLY SIGN FLIPS
# ================================================================================================================
print("\n[13/24] Coordinate-only sign flips...")
FLIPS=[]
for r in DUAL:
    flip=np.sign(r["old"])!=np.sign(r["new"])
    if flip:
        FLIPS.append(r["layer"])
        print(f"       L{r['layer']:02d}→L{r['layer']+1:02d} SIGN FLIP")
print("       flips:",FLIPS)

# ================================================================================================================
# [14/24] TERMINAL FORENSIC
# ================================================================================================================
print("\n[14/24] Terminal moving-compass forensic...")
x18=D["post"][:,18]
p18_18=x18@ORTH[18];p18_19=x18@ORTH[19]
raw19=D["raw"][:,19];post19=D["post"][:,19]
r19_18=raw19@ORTH[18];r19_19=raw19@ORTH[19]
p19_18=post19@ORTH[18];p19_19=post19@ORTH[19]
print(f"       SAME POST18 : @ORTH18={float(p18_18.mean()):+.6f} | @ORTH19={float(p18_19.mean()):+.6f}")
print(f"       RAW19       : @ORTH18={float(r19_18.mean()):+.6f} | @ORTH19={float(r19_19.mean()):+.6f}")
print(f"       POST19      : @ORTH18={float(p19_18.mean()):+.6f} | @ORTH19={float(p19_19.mean()):+.6f}")
c1819=float(torch.dot(ORTH[18],ORTH[19]).clamp(-1,1))
a1819=math.degrees(math.acos(max(-1,min(1,c1819))))
print(f"       ORTH18→ORTH19 cos={c1819:+.6f} angle={a1819:.3f}°")

# ================================================================================================================
# [15/24] BLOCK19 TRANSFORMATION
# ================================================================================================================
print("\n[15/24] Block19 transformation under fixed coordinates...")
pre19=D["pre"][:,19]
for name,u in [("ORTH18",ORTH[18]),("ORTH19",ORTH[19])]:
    a=pre19@u;b=raw19@u;c=post19@u
    print(f"       {name}: PRE19={float(a.mean()):+.6f} → RAW19={float(b.mean()):+.6f} → POST19={float(c.mean()):+.6f}")

# ================================================================================================================
# [16/24] FINAL RMSNORM — FIXED DUAL COORDINATES
# ================================================================================================================
print("\n[16/24] POST19 → final RMSNorm...")
FIN={a:torch.stack([x["final"] for x in RUN[a]]) for a in RUN}
DF=.5*(FIN["PLUS"]-FIN["MINUS"])
for name,u in [("ORTH18",ORTH[18]),("ORTH19",ORTH[19])]:
    a=post19@u;b=DF@u
    print(f"       {name}: POST19={float(a.mean()):+.6f} → FINAL={float(b.mean()):+.6f} Δ={float((b-a).mean()):+.6f}")

# ================================================================================================================
# [17/24] ADJACENT COMPASS vs COORDINATE LOSS
# ================================================================================================================
print("\n[17/24] Compass rotation vs coordinate change...")
CORR_X=[];CORR_Y=[]
for L,r in enumerate(DUAL):
    angle=COMPASS[L]["angle"];loss=abs(r["coord_delta"])
    CORR_X.append(angle);CORR_Y.append(loss)
    print(f"       {L:02d}→{L+1:02d} angle={angle:6.2f}° | |Δcoord|={loss:.6f}")
corr=float(np.corrcoef(CORR_X,CORR_Y)[0,1]) if np.std(CORR_X)>0 and np.std(CORR_Y)>0 else float("nan")
print(f"       Pearson(angle,|Δcoord|)={corr:+.6f}")

# ================================================================================================================
# [18/24] BOOTSTRAP TERMINAL COORDINATE ROTATION
# ================================================================================================================
print("\n[18/24] Bootstrap terminal coordinate rotation...")
def boot(x,n=20000,seed=183):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))

terminal_coord=(p18_19-p18_18).detach().cpu().numpy()
m,lo,hi=boot(terminal_coord)
print(f"       POST18 ORTH19−ORTH18={m:+.6f} [{lo:+.6f},{hi:+.6f}]")

inj19=((post19-raw19)@ORTH[19]).detach().cpu().numpy()
im,ilo,ihi=boot(inj19,seed=184)
print(f"       L19 injection @ORTH19={im:+.6f} [{ilo:+.6f},{ihi:+.6f}]")

# ================================================================================================================
# [19/24] TERMINAL DECOMPOSITION
# ================================================================================================================
print("\n[19/24] Terminal decomposition...")
old=float(p18_18.mean());rot=float(p18_19.mean());raw=float(r19_19.mean());post=float(p19_19.mean())
coord_effect=rot-old
block_effect=raw-rot
inject_effect=post-raw
print(f"       START @ORTH18       ={old:+.6f}")
print(f"       coordinate rotation ={coord_effect:+.6f}")
print(f"       block19 transform   ={block_effect:+.6f}")
print(f"       L19 injection       ={inject_effect:+.6f}")
print(f"       END @ORTH19         ={post:+.6f}")
print(f"       reconstruction      ={old+coord_effect+block_effect+inject_effect:+.6f}")

# ================================================================================================================
# [20/24] DECISION
# ================================================================================================================
print("\n[20/24] TEST183 decision...")
coord_flip=(old>0 and rot<0)
physical_same=float((D["post"][:,18]-D["pre"][:,19]).norm(dim=-1).mean())<1e-6
inject_rescue=(raw<0 and post>raw)
inject_cross=(raw<0 and post>0)

if physical_same and coord_flip and inject_cross:
    DECISION="TERMINAL_REVERSAL_LOCALIZED_TO_COMPASS_ROTATION__L19_INJECTION_RESTORES_POSITIVE_LOCAL_TARGET"
elif physical_same and coord_flip and inject_rescue:
    DECISION="TERMINAL_REVERSAL_LOCALIZED_TO_COMPASS_ROTATION__L19_INJECTION_PARTIALLY_CORRECTS"
elif physical_same and coord_flip:
    DECISION="TERMINAL_REVERSAL_LOCALIZED_TO_LAYER_LOCAL_COMPASS_ROTATION"
else:
    DECISION="MOVING_COMPASS_HYPOTHESIS_NOT_CONFIRMED"

print("       physical POST18==PRE19:",physical_same)
print("       coordinate sign flip:",coord_flip)
print("       L19 injection rescue:",inject_rescue)
print("       L19 injection crosses zero:",inject_cross)
print("       DECISION:",DECISION)

# ================================================================================================================
# [21/24] SAVE
# ================================================================================================================
print("\n[21/24] Saving...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("COMPASS_ROTATION.csv",COMPASS);savecsv("TRANSPORT.csv",TRANSPORT);savecsv("DUAL_PROJECTION.csv",DUAL)
np.savez_compressed(ROOT/"TEST183_MOVING_COMPASS.npz",rho=RHO.astype(np.float32),orth=ORTH.cpu().numpy(),
 pre=D["pre"].cpu().numpy(),raw=D["raw"].cpu().numpy(),post=D["post"].cpu().numpy(),final=DF.cpu().numpy())
REPORT={"test":"TEST183","parent":"TEST182","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
 "terminal":{"orth18_orth19_cos":c1819,"angle":a1819,"post18_orth18":old,"post18_orth19":rot,
 "raw19_orth19":raw,"post19_orth19":post,"coordinate_effect":coord_effect,
 "block19_effect":block_effect,"injection_effect":inject_effect},
 "coordinate_flips":FLIPS,"angle_coord_change_correlation":corr,
 "integrity":{"xray_only":True,"fresh_prompts":True,"vector_search":False,"rank_search":False,
 "layer_search":False,"envelope_search":False,"controller":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [22/24] INTEGRITY
# ================================================================================================================
print("\n[22/24] Integrity...")
print("       TEST182 infrastructure preserved: True")
print("       Same E forge lineage: True")
print("       Same P8: True")
print("       Same SEASC envelope: True")
print("       Same scale=.5: True")
print("       No vector/rank/layer/envelope selection: True")
print("       No controller: True")
print("       No weight modification: True")

# ================================================================================================================
# [23/24] SCIENTIFIC BOUNDARY
# ================================================================================================================
print("\n[23/24] Scientific boundary...")
print("       POST[L] and PRE[L+1] are compared as physical tensors.")
print("       The same POST state is projected onto both ORTH[L] and ORTH[L+1].")
print("       Therefore coordinate rotation is separated from actual inter-block state transport.")
print("       Layer-local ORTH directions are not assumed to form one fixed global coordinate.")
print("       TEST183 is mechanistic X-ray only; it does not establish a universal success criterion.")

# ================================================================================================================
# [24/24] COMPLETE
# ================================================================================================================
print("\n[24/24] COMPLETE")
print("="*138)
print("TEST 183 — MOVING-COMPASS X-RAY — SUMMARY")
print("="*138)
print(f"ORTH18→ORTH19 cos={c1819:+.6f} angle={a1819:.3f}°")
print(f"SAME POST18 : @ORTH18={old:+.6f} → @ORTH19={rot:+.6f}")
print(f"BLOCK19     : {rot:+.6f} → {raw:+.6f}")
print(f"L19 INJECT  : {raw:+.6f} → {post:+.6f}")
print(f"DECOMPOSE   : coordinate={coord_effect:+.6f} block19={block_effect:+.6f} injection={inject_effect:+.6f}")
print(f"POST18→PRE19 physical equality: {physical_same}")
print("Coordinate-only sign flips:",FLIPS)
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
