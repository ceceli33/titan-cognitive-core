# ================================================================================================================
# TEST 182 — PRE/POST INJECTION HOOK-TIMING X-RAY
# AkbasCore 3.2 / SEASC
# TEST181 continuation — CAUSE↔EFFECT
#
# PURPOSE:
# Locate where TEST180/181 terminal target reversal actually appears.
#
# FROZEN:
# model / E axis / P8 / causal forge / SEASC / rho / scale=.50
#
# MEASURE:
# PRE[L]  = exact block input before transformer block L
# RAW[L]  = exact block output before SEASC injection
# POST[L] = exact tensor immediately after SEASC injection
# NEXT    = PRE[L+1]
# FINAL   = model final RMSNorm output
#
# ARMS:
# BASE
# PLUS  (+E_ORTH)
# MINUS (-E_ORTH)
#
# X-RAY ONLY — NO SEARCH / SELECTION / CONTROLLER / WEIGHT CHANGE
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*132)
print("TEST 182 — PRE/POST INJECTION HOOK-TIMING X-RAY")
print("CAUSE↔EFFECT — EXACT BLOCK BOUNDARY LOCALIZATION")
print("="*132)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=182
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST182_PRE_POST_HOOK_TIMING_XRAY");ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(N_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
print(f"SEASC RSS={RSS:.9f} | scale={PRIMARY_SCALE}")

# ================================================================================================================
# [1/22] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test182";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""
#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d);
torch::Tensor seasc_batch(torch::Tensor h,torch::Tensor a,torch::Tensor d){
TORCH_CHECK(h.is_cuda()&&a.is_cuda()&&d.is_cuda(),"CUDA required");
return seasc_batch_cuda(h,a,d);}
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
for(int j=threadIdx.x;j<H;j+=blockDim.x)h[hb+j]=(scalar_t)((float)h[hb+j]+z*a[ab+j]);
}
torch::Tensor seasc_batch_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d){
auto o=h.contiguous().clone();auto aa=a.to(h.device(),torch::kFloat32).contiguous();
auto dd=d.to(h.device(),torch::kFloat32).contiguous();int B=o.size(0),S=o.size(1),H=o.size(2);
constexpr int T=256;cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,o.scalar_type(),"seasc",[&]{
k<scalar_t><<<B*S,T,T*sizeof(float),stream>>>(o.data_ptr<scalar_t>(),aa.data_ptr<float>(),dd.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
"""
print("\n[1/22] CUDA compile...")
seasc_ext=load_inline(name="akbascore_test182_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
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
print("       hidden:",H,"dtype:",next(model.parameters()).dtype)

def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],
 tokenize=False,add_generation_prompt=True)
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)

@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt").to(DEVICE)
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);p=int(e["attention_mask"][0].sum())-1
    x=[o.hidden_states[L+1][0,p].float().detach() for L in range(N_LAYERS)]
    del e,o;return x

def sequence(ctx,cont):
    p=tok(chat(ctx),return_tensors="pt",add_special_tokens=False)
    c=tok(cont,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p.input_ids,c.input_ids],1).to(DEVICE)
    return ids,torch.ones_like(ids),p.input_ids.shape[1]

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
        score=F.log_softmax(z,-1).gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean()
        (score if positive else -score).backward()
        g=[saved[L].grad[0].float().sum(0).detach().clone() for L in range(N_LAYERS)]
    finally:
        for h in hooks:h.remove()
        model.zero_grad(set_to_none=True)
    return g

# ================================================================================================================
# [3/22] E AXIS
# ================================================================================================================
print("\n[3/22] Frozen E axis...")
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
# [4/22] FISHER
# ================================================================================================================
print("\n[4/22] Fisher...")
PB=[[] for _ in range(N_LAYERS)];NB=[[] for _ in range(N_LAYERS)]
for i,(p,n) in enumerate(PAIRS,1):
    hp,hn=capture(p),capture(n)
    for L in range(N_LAYERS):PB[L].append(hp[L]);NB[L].append(hn[L])
    print(f"       {i:02d}/16")
PB=[torch.stack(x).float() for x in PB];NB=[torch.stack(x).float() for x in NB]
FISHER=[];BANK=[];FB=[]
for L in range(N_LAYERS):
    hp,hn=PB[L],NB[L];mp,mn=hp.mean(0),hn.mean(0);md=mp-mn
    rr=torch.cat([hp-mp,hn-mn]);_,s,vh=torch.linalg.svd(rr,full_matrices=False)
    eig=s.square()/max(rr.shape[0]-2,1);pe=eig[eig>FISHER_EPS]
    es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE);lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS)
    V=vh.T;pr=V.T@md;f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float()
    dirs=norm(hp-hn);dirs*=torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
    b=norm(dirs.mean(0)).float();c=torch.dot(f,b)
    if c<0:b=-b;c=-c
    FISHER.append(f);BANK.append(b);FB.append(float(c))
FISHER=torch.stack(FISHER);BANK=torch.stack(BANK)
print(f"       mean cos(F,B)={np.mean(FB):+.6f}")

# ================================================================================================================
# [5/22] P8
# ================================================================================================================
print("\n[5/22] P8...")
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
    out=[]
    for L in range(N_LAYERS):
        q=Q8[L];v=x[L]-(q@(q.T@x[L]) if q.shape[1] else 0)
        out.append(norm(v))
    return torch.stack(out).contiguous()
FG=guard(FISHER)

# ================================================================================================================
# [6/22] E FORGE
# ================================================================================================================
print("\n[6/22] Frozen causal forge...")
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
for L in range(N_LAYERS):
    v=CG[L]-torch.dot(CG[L],FG[L])*FG[L];ORTH.append(norm(v))
ORTH=torch.stack(ORTH).contiguous()
print(f"       mean orth={np.mean([float((CG[L]-torch.dot(CG[L],FG[L])*FG[L]).norm()) for L in range(N_LAYERS)]):.6f}")

# ================================================================================================================
# [7/22] FRESH XRAY
# ================================================================================================================
print("\n[7/22] Fresh X-ray prompts...")
XRAY=[
"A blocked air intake and an overheated motor form a causal sequence.",
"A ruptured hose and falling pressure form a causal sequence.",
"A voltage surge and a damaged circuit form a causal sequence.",
"A frozen pipe and restricted water flow form a causal sequence.",
"A worn bearing and increased vibration form a causal sequence.",
"A clogged outlet and rising liquid level form a causal sequence.",
"A broken wire and loss of electrical power form a causal sequence.",
"A strong impact and a deformed panel form a causal sequence."]
print("       n=8")

# ================================================================================================================
# [8/22] EXACT PRE/RAW/POST RECORDER
# ================================================================================================================
print("\n[8/22] Exact block-boundary recorder...")

def run_exact(prompt,sign=0.0):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    PRE=[None]*N_LAYERS;RAW=[None]*N_LAYERS;POST=[None]*N_LAYERS;hooks=[]
    for L in range(N_LAYERS):
        def prehook(li):
            def hk(m,args):
                PRE[li]=args[0][0,pos].float().detach().clone()
            return hk
        def posthook(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out
                RAW[li]=raw[0,pos].float().detach().clone()
                if sign==0.0:
                    new=raw
                else:
                    B=raw.shape[0];a=(float(sign)*ORTH[li][None,:].expand(B,-1)).float().contiguous()
                    d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                    new=seasc_ext.seasc_batch(raw,a,d)
                POST[li]=new[0,pos].float().detach().clone()
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_pre_hook(prehook(L)))
        hooks.append(layers[L].register_forward_hook(posthook(L)))
    final_box={}
    def final_hook(m,args,out):final_box["x"]=out[0,pos].float().detach().clone()
    hooks.append(model.model.norm.register_forward_hook(final_hook))
    try:
        with torch.inference_mode():o=model(**e,use_cache=False,return_dict=True)
        logits=o.logits[0,-1].float().detach().clone()
    finally:
        for h in hooks:h.remove()
    return {"pre":torch.stack(PRE),"raw":torch.stack(RAW),"post":torch.stack(POST),
            "final":final_box["x"],"logits":logits}

# ================================================================================================================
# [9/22] CAPTURE
# ================================================================================================================
print("\n[9/22] Capturing BASE / PLUS / MINUS...")
RUN={"BASE":[],"PLUS":[],"MINUS":[]}
for i,p in enumerate(XRAY,1):
    RUN["BASE"].append(run_exact(p,0.0));RUN["PLUS"].append(run_exact(p,+1.0));RUN["MINUS"].append(run_exact(p,-1.0))
    print(f"       {i}/8")

# ================================================================================================================
# [10/22] SYMMETRIC DELTAS
# ================================================================================================================
print("\n[10/22] Symmetric PRE/RAW/POST deltas...")
def stack(stage):
    return {a:torch.stack([x[stage] for x in RUN[a]]) for a in RUN}
S={x:stack(x) for x in ["pre","raw","post"]}
D={}
for stage in S:D[stage]=.5*(S[stage]["PLUS"]-S[stage]["MINUS"])

# ================================================================================================================
# [11/22] METRICS
# ================================================================================================================
print("\n[11/22] Layer metrics...")
ROWS=[]
def met(d,b,u):
    dn=d.norm(dim=-1).clamp_min(EPS);bn=b.norm(dim=-1).clamp_min(EPS)
    tar=d@u;co=tar/dn
    return float((dn/bn).mean()),float(tar.mean()),float(co.mean()),float(torch.rad2deg(torch.acos(co.clamp(-1,1))).mean())

for L in range(N_LAYERS):
    row={"layer":L}
    for stage in ["pre","raw","post"]:
        rel,tar,co,ang=met(D[stage][:,L],S[stage]["BASE"][:,L],ORTH[L])
        row[f"{stage}_rel"]=rel;row[f"{stage}_target"]=tar;row[f"{stage}_cos"]=co;row[f"{stage}_angle"]=ang
    ROWS.append(row)
    print(f"       L{L:02d} PRE tar={row['pre_target']:+.4f} cos={row['pre_cos']:+.4f} | "
          f"RAW tar={row['raw_target']:+.4f} cos={row['raw_cos']:+.4f} | "
          f"POST tar={row['post_target']:+.4f} cos={row['post_cos']:+.4f}")

# ================================================================================================================
# [12/22] DIRECT INJECTION EFFECT
# ================================================================================================================
print("\n[12/22] Direct injection contribution RAW→POST...")
INJ=[]
for L in range(N_LAYERS):
    dd=D["post"][:,L]-D["raw"][:,L]
    tar=(dd@ORTH[L]).detach().cpu().numpy()
    n=dd.norm(dim=-1).detach().cpu().numpy()
    r={"layer":L,"inj_target":float(tar.mean()),"inj_norm":float(n.mean())}
    INJ.append(r)
    print(f"       L{L:02d} Δtarget={r['inj_target']:+.6f} Δnorm={r['inj_norm']:.6f}")

# ================================================================================================================
# [13/22] TRANSPORT POST[L] → PRE[L+1]
# ================================================================================================================
print("\n[13/22] Inter-block transport...")
TRANSPORT=[]
for L in range(N_LAYERS-1):
    post=D["post"][:,L];nxt=D["pre"][:,L+1]
    # same physical tensor should match closely if no hidden operation exists between blocks
    err=(nxt-post).norm(dim=-1)
    rel=err/post.norm(dim=-1).clamp_min(EPS)
    r={"from_layer":L,"to_layer":L+1,"abs_error":float(err.mean()),"rel_error":float(rel.mean())}
    TRANSPORT.append(r)
    print(f"       POST{L:02d}→PRE{L+1:02d} relerr={r['rel_error']:.8f}")

# ================================================================================================================
# [14/22] L18/L19 FORENSIC
# ================================================================================================================
print("\n[14/22] L18/L19 forensic...")
for L in [17,18,19]:
    r=ROWS[L]
    print(f"       L{L:02d}: PRE {r['pre_target']:+.6f} → RAW {r['raw_target']:+.6f} → POST {r['post_target']:+.6f} "
          f"| injection Δ={INJ[L]['inj_target']:+.6f}")

# ================================================================================================================
# [15/22] FINAL RMSNORM
# ================================================================================================================
print("\n[15/22] Final RMSNorm...")
FINAL={a:torch.stack([x["final"] for x in RUN[a]]) for a in RUN}
DF=.5*(FINAL["PLUS"]-FINAL["MINUS"])
# L19 direction is used only as the fixed terminal reference direction.
frel,ftar,fcos,fang=met(DF,FINAL["BASE"],ORTH[19])
print(f"       FINAL rel={frel:.6f} target@ORTH19={ftar:+.6f} cos={fcos:+.6f} angle={fang:.2f}°")

# ================================================================================================================
# [16/22] TERMINAL LOGIT DIVERGENCE
# ================================================================================================================
print("\n[16/22] Terminal logit-space displacement...")
LOG={a:torch.stack([x["logits"] for x in RUN[a]]) for a in RUN}
DL=.5*(LOG["PLUS"]-LOG["MINUS"])
log_rel=DL.norm(dim=-1)/LOG["BASE"].norm(dim=-1).clamp_min(EPS)
print(f"       symmetric logit rel={float(log_rel.mean()):.8f}")
print(f"       symmetric logit norm={float(DL.norm(dim=-1).mean()):.6f}")

# ================================================================================================================
# [17/22] BOOTSTRAP KEY TRANSITIONS
# ================================================================================================================
print("\n[17/22] Bootstrap key transitions...")
def boot(x,n=20000,seed=182):
    x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))

BOOT=[]
for L in [17,18,19]:
    for name,x in [
        ("PRE_TARGET",(D["pre"][:,L]@ORTH[L]).cpu().numpy()),
        ("RAW_TARGET",(D["raw"][:,L]@ORTH[L]).cpu().numpy()),
        ("POST_TARGET",(D["post"][:,L]@ORTH[L]).cpu().numpy()),
        ("INJECTION_DELTA",((D["post"][:,L]-D["raw"][:,L])@ORTH[L]).cpu().numpy())]:
        m,lo,hi=boot(x,seed=SEED+L*10+len(BOOT))
        BOOT.append({"layer":L,"metric":name,"mean":m,"ci_low":lo,"ci_high":hi})
        print(f"       L{L:02d} {name:<15} {m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [18/22] REVERSAL LOCALIZATION
# ================================================================================================================
print("\n[18/22] Reversal localization...")
# Important: vectors are layer-local, so sign comparisons are made within each measurement layer.
# We distinguish reversal already present at PRE19/RAW19 from a reversal created by injection19.
pre19=ROWS[19]["pre_target"];raw19=ROWS[19]["raw_target"];post19=ROWS[19]["post_target"]
inj19=INJ[19]["inj_target"]

if raw19<0 and post19>raw19 and inj19>0:
    DECISION="REVERSAL_PREEXISTS_L19_INJECTION__L19_INJECTION_PARTIALLY_CORRECTS"
elif raw19>=0 and post19<0 and inj19<0:
    DECISION="L19_INJECTION_CREATES_TERMINAL_REVERSAL"
elif raw19<0 and post19<0 and abs(post19)<abs(raw19) and inj19>0:
    DECISION="REVERSAL_PREEXISTS_L19__L19_INJECTION_REDUCES_BUT_DOES_NOT_REMOVE_IT"
elif raw19<0 and post19<=raw19:
    DECISION="REVERSAL_PREEXISTS_L19__L19_INJECTION_DOES_NOT_CORRECT"
else:
    DECISION="TERMINAL_REVERSAL_LOCATION_NOT_RESOLVED"

print(f"       PRE19  ={pre19:+.6f}")
print(f"       RAW19  ={raw19:+.6f}")
print(f"       POST19 ={post19:+.6f}")
print(f"       INJ19 Δ={inj19:+.6f}")
print("       DECISION:",DECISION)

# ================================================================================================================
# [19/22] IMPORTANT INDEXING CHECK
# ================================================================================================================
print("\n[19/22] Hidden-state indexing check...")
# TEST181 showed changing L19 altered behavior but not hidden_states[L19+1].
# Here POST is captured directly after our injection and therefore is the authoritative injection measurement.
l19_changed=float((D["post"][:,19]-D["raw"][:,19]).norm(dim=-1).mean())
print(f"       direct L19 RAW→POST delta norm={l19_changed:.6f}")
if l19_changed>1e-6:
    print("       CONFIRMED: direct hook recorder sees L19 injection.")
else:
    print("       WARNING: direct hook recorder did not detect L19 injection.")

# ================================================================================================================
# [20/22] SAVE
# ================================================================================================================
print("\n[20/22] Saving...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

savecsv("LAYERWISE_PRE_RAW_POST.csv",ROWS)
savecsv("DIRECT_INJECTION.csv",INJ)
savecsv("INTERBLOCK_TRANSPORT.csv",TRANSPORT)
savecsv("BOOTSTRAP.csv",BOOT)

np.savez_compressed(ROOT/"TEST182_XRAY.npz",
 rho=RHO.astype(np.float32),orth=ORTH.cpu().numpy(),
 base_pre=S["pre"]["BASE"].cpu().numpy(),plus_pre=S["pre"]["PLUS"].cpu().numpy(),minus_pre=S["pre"]["MINUS"].cpu().numpy(),
 base_raw=S["raw"]["BASE"].cpu().numpy(),plus_raw=S["raw"]["PLUS"].cpu().numpy(),minus_raw=S["raw"]["MINUS"].cpu().numpy(),
 base_post=S["post"]["BASE"].cpu().numpy(),plus_post=S["post"]["PLUS"].cpu().numpy(),minus_post=S["post"]["MINUS"].cpu().numpy(),
 base_final=FINAL["BASE"].cpu().numpy(),plus_final=FINAL["PLUS"].cpu().numpy(),minus_final=FINAL["MINUS"].cpu().numpy())

REPORT={"test":"TEST182","title":"PRE/POST INJECTION HOOK-TIMING X-RAY","parent":"TEST181",
 "model":MODEL_ID,"scale":PRIMARY_SCALE,"rss":RSS,"decision":DECISION,
 "terminal":{"pre19":pre19,"raw19":raw19,"post19":post19,"inj19_delta":inj19,
 "final_target":ftar,"final_cos":fcos,"final_angle":fang},
 "integrity":{"xray_only":True,"vector_search":False,"rank_search":False,"layer_search":False,
 "envelope_search":False,"controller":False,"weights_modified":False,"fresh_prompts":True,
 "direct_hook_post_injection_capture":True}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [21/22] SCIENTIFIC BOUNDARY
# ================================================================================================================
print("\n[21/22] Scientific boundary...")
print("       PRE/RAW/POST are captured directly at transformer-block boundaries.")
print("       POST is the authoritative state immediately after SEASC injection.")
print("       No vector, dose, rank, layer or controller is selected in this test.")
print("       TEST182 localizes the terminal reversal; it does not by itself prove behavioral causality.")
print("       Layer-local directions differ across depth; cross-layer target magnitudes are mechanistic diagnostics, not one common coordinate.")

# ================================================================================================================
# [22/22] COMPLETE
# ================================================================================================================
print("\n[22/22] COMPLETE")
print("="*132)
print("TEST 182 — TERMINAL FORENSIC SUMMARY")
print("="*132)
for L in [17,18,19]:
    r=ROWS[L]
    print(f"L{L:02d} PRE={r['pre_target']:+.6f} RAW={r['raw_target']:+.6f} POST={r['post_target']:+.6f} "
          f"INJECT={INJ[L]['inj_target']:+.6f} POSTcos={r['post_cos']:+.6f}")
print("-"*132)
print(f"FINAL RMSNorm target@ORTH19={ftar:+.6f} cos={fcos:+.6f} angle={fang:.2f}°")
print(f"L19 direct RAW→POST delta norm={l19_changed:.6f}")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*132)
