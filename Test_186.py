# ================================================================================================================
# TEST 186 — L27 CAUSAL RESCUE ABLATION
# AkbasCore 3.2 / SEASC — TEST185 continuation — CAUSE↔EFFECT
#
# TEST185:
#   POST19 @ORTH19 = +0.464281
#   RAW26  @ORTH19 = +0.715652
#   RAW27  @ORTH19 = -0.297196
#   L27 contribution = -1.012847 [-1.314393,-0.705642]
#
# QUESTION
# Is L27 merely correlated with terminal sign loss, or does its transformation
# causally suppress the behavioral CAUSE↔EFFECT steering effect?
#
# PREDECLARED ARMS
#   NORMAL          : frozen SEASC L0-L19; untouched L20-L27
#   L27_BYPASS      : replace L27 output hidden-state with its exact input
#   L27_ORTH_RESCUE : preserve L27 output except cancel its negative change along fixed ORTH19:
#                     delta27 = RAW27 - PRE27
#                     if <delta27,ORTH19> < 0:
#                         RAW27' = RAW27 - <delta27,ORTH19>*ORTH19
#
# No tunable rescue coefficient. No L20-L26 modification.
# Fresh DEV = diagnostic only. FINAL opened after arms frozen.
# FINAL causal + neutral collateral. X-ray verifies intervention.
# ================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*138)
print("TEST 186 — L27 CAUSAL RESCUE ABLATION")
print("CAUSE↔EFFECT — NORMAL vs L27_BYPASS vs L27_ORTH_RESCUE")
print("="*138)

os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=186
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS,SVD_REL_TOL=.10,1e-8,1e-10,1e-5
PRIMARY_SCALE=.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST186_L27_CAUSAL_RESCUE");ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(STEER_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
print(f"SEASC RSS={RSS:.9f} | scale={PRIMARY_SCALE}")

# ================================================================================================================
# [1/28] CUDA
# ================================================================================================================
BUILD="/tmp/akbascore_test186";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
print("\n[1/28] CUDA...")
seasc_ext=load_inline(name="akbascore_test186_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,
 build_directory=BUILD,verbose=False)
print("       OK")

# ================================================================================================================
# [2/28] MODEL
# ================================================================================================================
print("\n[2/28] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},
 attn_implementation="sdpa",trust_remote_code=True)
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;FINAL_NORM=model.model.norm
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
    p=tok(chat(ctx),return_tensors="pt",add_special_tokens=False);c=tok(cont,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p.input_ids,c.input_ids],1).to(DEVICE);return ids,torch.ones_like(ids),p.input_ids.shape[1]

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
# [3/28] FROZEN E AXIS
# ================================================================================================================
print("\n[3/28] Frozen E axis...")
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
# [4/28] FISHER/BANK
# ================================================================================================================
print("\n[4/28] Fisher/Bank...")
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
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS);V=vh.T;pr=V.T@md
    f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float();dirs=norm(hp-hn)
    dirs*=torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
    b=norm(dirs.mean(0)).float();c=torch.dot(f,b)
    if c<0:b=-b;c=-c
    FISHER.append(f);BANK.append(b);FB.append(float(c))
FISHER=torch.stack(FISHER);BANK=torch.stack(BANK)
print(f"       mean cos(F,B)={np.mean(FB):+.6f}")

# ================================================================================================================
# [5/28] P8
# ================================================================================================================
print("\n[5/28] P8...")
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
    G=norm(torch.stack([g[L] for g in PG]));_,s,vh=torch.linalg.svd(G,full_matrices=False)
    r=int((s>max(float(s[0])*SVD_REL_TOL,EPS)).sum());Q8.append(vh[:r].T.contiguous());ranks.append(r)
print("       ranks:",ranks)
def guard(x):
    z=[]
    for L in range(STEER_LAYERS):
        q=Q8[L];v=x[L]-(q@(q.T@x[L]) if q.shape[1] else 0);z.append(norm(v))
    return torch.stack(z).contiguous()
FG=guard(FISHER)

# ================================================================================================================
# [6/28] FROZEN FORGE
# ================================================================================================================
print("\n[6/28] Frozen forge...")
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
print("       ORTH ready | rescue coordinate=ORTH19")

# ================================================================================================================
# [7/28] FROZEN ARMS / DATA
# ================================================================================================================
print("\n[7/28] Frozen arms / data...")
ARMS=["NORMAL","L27_BYPASS","L27_ORTH_RESCUE"]
DEV=[
("A damaged pump causes a reduction in pressure."," The damaged pump is the cause."," The pressure reduction is the effect."),
("A blocked vent causes heat accumulation."," The blocked vent is the cause."," The heat accumulation is the effect."),
("A broken wire causes loss of power."," The broken wire is the cause."," The power loss is the effect."),
("A leaking pipe causes a drop in fluid level."," The leaking pipe is the cause."," The fluid-level drop is the effect."),
("A jammed gear causes motion to stop."," The jammed gear is the cause."," The stopped motion is the effect."),
("A voltage surge causes a component failure."," The voltage surge is the cause."," The component failure is the effect."),
("A clogged filter causes reduced flow."," The clogged filter is the cause."," The reduced flow is the effect."),
("A loose connection causes an intermittent signal."," The loose connection is the cause."," The intermittent signal is the effect.")]
FINAL=[
("A cracked hose produces a loss of pressure."," The cracked hose is the cause."," The pressure loss is the effect."),
("A frozen pipe produces a blockage."," The freezing is the cause."," The blockage is the effect."),
("A short circuit produces a system shutdown."," The short circuit is the cause."," The shutdown is the effect."),
("A damaged bearing produces excessive vibration."," The bearing damage is the cause."," The vibration is the effect."),
("A blocked drain produces water accumulation."," The blockage is the cause."," The accumulated water is the effect."),
("A failed fan produces overheating."," The fan failure is the cause."," The overheating is the effect."),
("A puncture produces loss of air pressure."," The puncture is the cause."," The pressure loss is the effect."),
("A hard collision produces structural deformation."," The collision is the cause."," The deformation is the effect.")]
NEUTRAL=[
("A blue notebook lies beside a metal pen."," The notebook and pen are ordinary objects."),
("A ceramic plate rests beside a folded towel."," The plate and towel are stationary objects."),
("A small box sits next to a glass bottle."," The box and bottle are ordinary objects."),
("A wooden ruler lies across a blank sheet."," The ruler and sheet are stationary objects."),
("A closed book rests beside a desk lamp."," The book and lamp are ordinary objects."),
("A paper folder lies next to a calculator."," The folder and calculator are stationary objects."),
("A spoon rests beside an empty bowl."," The spoon and bowl are ordinary objects."),
("A plain envelope lies beside a pencil."," The envelope and pencil are stationary objects.")]
print("       arms:",ARMS)
print("       DEV=8 FINAL=8 NEUTRAL=8")

# ================================================================================================================
# [8/28] FORWARD WITH FROZEN L27 INTERVENTION
# ================================================================================================================
print("\n[8/28] Intervention engine...")
def install_hooks(sign,arm,box=None,pos=None):
    hooks=[]
    for L in range(STEER_LAYERS):
        def mk(li):
            def hk(m,args,out):
                raw=out[0] if isinstance(out,tuple) else out
                if sign!=0:
                    B=raw.shape[0];a=(float(sign)*ORTH[li][None].expand(B,-1)).float().contiguous()
                    d=torch.full((B,),float(RHO[li])*PRIMARY_SCALE,device=DEVICE,dtype=torch.float32)
                    new=seasc_ext.seasc_batch(raw,a,d)
                else:new=raw
                if box is not None and li==19:box["post19"]=new[0,pos].float().detach().clone()
                return new if not isinstance(out,tuple) else (new,)+out[1:]
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    if arm!="NORMAL":
        cache={}
        def pre27(m,args):
            cache["pre"]=args[0]
            if box is not None:box["pre27"]=args[0][0,pos].float().detach().clone()
        def post27(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out;pre=cache["pre"]
            if box is not None:box["raw27"]=raw[0,pos].float().detach().clone()
            if arm=="L27_BYPASS":new=pre
            elif arm=="L27_ORTH_RESCUE":
                delta=(raw.float()-pre.float())
                proj=torch.einsum("bsh,h->bs",delta,U.float())
                corr=torch.clamp(-proj,min=0.0)[...,None]*U.float()[None,None,:]
                new=(raw.float()+corr).to(raw.dtype)
            else:new=raw
            if box is not None:box["out27"]=new[0,pos].float().detach().clone()
            return new if not isinstance(out,tuple) else (new,)+out[1:]
        hooks.append(layers[27].register_forward_pre_hook(pre27))
        hooks.append(layers[27].register_forward_hook(post27))
    elif box is not None:
        def pre27n(m,args):box["pre27"]=args[0][0,pos].float().detach().clone()
        def post27n(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out
            box["raw27"]=raw[0,pos].float().detach().clone();box["out27"]=box["raw27"].clone()
        hooks.append(layers[27].register_forward_pre_hook(pre27n))
        hooks.append(layers[27].register_forward_hook(post27n))
    return hooks

def score_cont(ctx,cont,sign,arm):
    ids,mask,plen=sequence(ctx,cont);hooks=install_hooks(sign,arm)
    try:
        with torch.inference_mode():
            o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
            z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:]
            return float(F.log_softmax(z,-1).gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean())
    finally:
        for h in hooks:h.remove()

def logits_prompt(prompt,sign,arm):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);hooks=install_hooks(sign,arm)
    try:
        with torch.inference_mode():return model(**e,use_cache=False,return_dict=True).logits[0,-1].float().detach()
    finally:
        for h in hooks:h.remove()

# ================================================================================================================
# [9/28] CAUSAL METRIC
# ================================================================================================================
print("\n[9/28] Causal metric...")
def causal_eval(data,arm):
    rows=[]
    for i,(ctx,p,n) in enumerate(data,1):
        pp=score_cont(ctx,p,+1,arm);pn=score_cont(ctx,n,+1,arm)
        mp=score_cont(ctx,p,-1,arm);mn=score_cont(ctx,n,-1,arm)
        c=.5*((pp-pn)-(mp-mn))
        rows.append({"item":i,"C":c,"plus_margin":pp-pn,"minus_margin":mp-mn})
    return rows
def summary(rows):
    x=np.asarray([r["C"] for r in rows]);return float(x.mean()),int((x>0).sum())

# ================================================================================================================
# [10/28] DEV DIAGNOSTIC — NO SELECTION
# ================================================================================================================
print("\n[10/28] DEV diagnostic — arms already frozen...")
DEVRES={}
for a in ARMS:
    DEVRES[a]=causal_eval(DEV,a);m,p=summary(DEVRES[a])
    print(f"       {a:<16} C={m:+.6f} P+={p}/8")
print("       No arm selected from DEV.")

# ================================================================================================================
# [11/28] FINAL OPEN
# ================================================================================================================
print("\n[11/28] FINAL causal...")
FINALRES={}
for a in ARMS:
    FINALRES[a]=causal_eval(FINAL,a);m,p=summary(FINALRES[a])
    print(f"       {a:<16} C={m:+.6f} P+={p}/8")

# ================================================================================================================
# [12/28] NEUTRAL KL
# ================================================================================================================
print("\n[12/28] FINAL neutral KL...")
def kl_eval(data,arm,topk=256):
    out=[]
    for i,(ctx,_) in enumerate(data,1):
        b=logits_prompt(ctx,0,"NORMAL")
        p=logits_prompt(ctx,+1,arm);m=logits_prompt(ctx,-1,arm)
        idx=torch.topk(b,topk).indices;lb=b[idx];lp=p[idx];lm=m[idx]
        qb=F.softmax(lb,-1);lqb=F.log_softmax(lb,-1)
        kp=float((qb*(lqb-F.log_softmax(lp,-1))).sum())
        km=float((qb*(lqb-F.log_softmax(lm,-1))).sum())
        out.append({"item":i,"KL":.5*(kp+km)})
    return out
KLRES={}
for a in ARMS:
    KLRES[a]=kl_eval(NEUTRAL,a);v=np.mean([x["KL"] for x in KLRES[a]])
    print(f"       {a:<16} KL={v:.6f}")

# ================================================================================================================
# [13/28] NEUTRAL |ΔNLL|
# ================================================================================================================
print("\n[13/28] FINAL neutral |ΔNLL|...")
def retention_eval(data,arm):
    out=[]
    for i,(ctx,cont) in enumerate(data,1):
        b=score_cont(ctx,cont,0,"NORMAL");p=score_cont(ctx,cont,+1,arm);m=score_cont(ctx,cont,-1,arm)
        out.append({"item":i,"Ret":.5*(abs(p-b)+abs(m-b))})
    return out
RETRES={}
for a in ARMS:
    RETRES[a]=retention_eval(NEUTRAL,a);v=np.mean([x["Ret"] for x in RETRES[a]])
    print(f"       {a:<16} |ΔNLL|={v:.6f}")

# ================================================================================================================
# [14/28] XRAY SET
# ================================================================================================================
print("\n[14/28] Fresh intervention X-ray...")
XRAY=[
"A fractured pipe and a pressure drop form a causal sequence.",
"A blocked air passage and rising temperature form a causal sequence.",
"A damaged seal and fluid loss form a causal sequence.",
"A current surge and a failed circuit form a causal sequence.",
"A stuck mechanism and restricted motion form a causal sequence.",
"A worn bearing and increased vibration form a causal sequence.",
"A clogged outlet and accumulated liquid form a causal sequence.",
"A strong impact and a deformed bracket form a causal sequence."]
print("       n=8")

# ================================================================================================================
# [15/28] XRAY RECORDER
# ================================================================================================================
print("\n[15/28] L27 intervention recorder...")
def xrun(prompt,sign,arm):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;box={}
    hooks=install_hooks(sign,arm,box,pos)
    def npre(m,args):box["norm_in"]=args[0][0,pos].float().detach().clone()
    hooks.append(FINAL_NORM.register_forward_pre_hook(npre))
    try:
        with torch.inference_mode():model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hooks:h.remove()
    return box
XR={}
for a in ARMS:
    XR[a]={"PLUS":[],"MINUS":[]}
    for i,p in enumerate(XRAY,1):
        XR[a]["PLUS"].append(xrun(p,+1,a));XR[a]["MINUS"].append(xrun(p,-1,a))
    print("      ",a,"done")

# ================================================================================================================
# [16/28] XRAY SYMMETRIC DELTAS
# ================================================================================================================
print("\n[16/28] X-ray symmetric deltas @ORTH19...")
XMET={}
for a in ARMS:
    XMET[a]={}
    for k in ["post19","pre27","raw27","out27","norm_in"]:
        P=torch.stack([x[k] for x in XR[a]["PLUS"]]);M=torch.stack([x[k] for x in XR[a]["MINUS"]]);d=.5*(P-M)
        XMET[a][k]=d
    vals=[]
    for k in ["post19","pre27","raw27","out27","norm_in"]:
        v=float((XMET[a][k]@U).mean());vals.append(v)
    print(f"       {a:<16} POST19={vals[0]:+.6f} PRE27={vals[1]:+.6f} RAW27={vals[2]:+.6f} OUT27={vals[3]:+.6f} NORM_IN={vals[4]:+.6f}")

# ================================================================================================================
# [17/28] INTERVENTION INTEGRITY
# ================================================================================================================
print("\n[17/28] Intervention integrity...")
for a in ARMS:
    pre=XMET[a]["pre27"];raw=XMET[a]["raw27"];out=XMET[a]["out27"]
    print(f"       {a:<16} rawΔ@U={float(((raw-pre)@U).mean()):+.6f} interventionΔ@U={float(((out-raw)@U).mean()):+.6f}")

# ================================================================================================================
# [18/28] BOOTSTRAP
# ================================================================================================================
print("\n[18/28] Bootstrap FINAL causal...")
def bootdiff(x,y,n=20000,seed=186):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);d=x-y
    rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=d[rng.integers(0,len(d),len(d))].mean()
    return float(d.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
BOOT=[]
base=[r["C"] for r in FINALRES["NORMAL"]]
for j,a in enumerate(["L27_BYPASS","L27_ORTH_RESCUE"]):
    x=[r["C"] for r in FINALRES[a]];m,lo,hi=bootdiff(x,base,seed=SEED+j)
    BOOT.append({"metric":"C","arm":a,"mean":m,"low":lo,"high":hi})
    print(f"       {a:<16} ΔC={m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [19/28] BOOTSTRAP NEUTRAL KL
# ================================================================================================================
print("\n[19/28] Bootstrap neutral KL...")
base=[r["KL"] for r in KLRES["NORMAL"]]
for j,a in enumerate(["L27_BYPASS","L27_ORTH_RESCUE"]):
    x=[r["KL"] for r in KLRES[a]];m,lo,hi=bootdiff(x,base,seed=196+j)
    BOOT.append({"metric":"KL","arm":a,"mean":m,"low":lo,"high":hi})
    print(f"       {a:<16} ΔKL={m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [20/28] BOOTSTRAP RETENTION
# ================================================================================================================
print("\n[20/28] Bootstrap neutral |ΔNLL|...")
base=[r["Ret"] for r in RETRES["NORMAL"]]
for j,a in enumerate(["L27_BYPASS","L27_ORTH_RESCUE"]):
    x=[r["Ret"] for r in RETRES[a]];m,lo,hi=bootdiff(x,base,seed=206+j)
    BOOT.append({"metric":"Ret","arm":a,"mean":m,"low":lo,"high":hi})
    print(f"       {a:<16} ΔRet={m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ================================================================================================================
# [21/28] PRIMARY TABLE
# ================================================================================================================
print("\n[21/28] Primary FINAL table...")
TABLE=[]
for a in ARMS:
    C,P=summary(FINALRES[a]);K=float(np.mean([x["KL"] for x in KLRES[a]]));R=float(np.mean([x["Ret"] for x in RETRES[a]]))
    X=float((XMET[a]["norm_in"]@U).mean())
    row={"arm":a,"C":C,"P":P,"KL":K,"Ret":R,"terminal":X};TABLE.append(row)
    print(f"       {a:<16} C={C:+.6f} P+={P}/8 KL={K:.6f} Ret={R:.6f} terminal={X:+.6f}")

# ================================================================================================================
# [22/28] CAUSAL RESCUE TEST
# ================================================================================================================
print("\n[22/28] Causal rescue test...")
B={r["arm"]:r for r in TABLE};N=B["NORMAL"];BP=B["L27_BYPASS"];RS=B["L27_ORTH_RESCUE"]
bc=next(x for x in BOOT if x["metric"]=="C" and x["arm"]=="L27_BYPASS")
rc=next(x for x in BOOT if x["metric"]=="C" and x["arm"]=="L27_ORTH_RESCUE")
print(f"       NORMAL terminal={N['terminal']:+.6f} C={N['C']:+.6f}")
print(f"       BYPASS terminal={BP['terminal']:+.6f} C={BP['C']:+.6f}")
print(f"       RESCUE terminal={RS['terminal']:+.6f} C={RS['C']:+.6f}")
print("       BYPASS causal CI > 0:",bc["low"]>0)
print("       RESCUE causal CI > 0:",rc["low"]>0)

# ================================================================================================================
# [23/28] DECISION
# ================================================================================================================
print("\n[23/28] TEST186 decision...")
terminal_rescue=(RS["terminal"]>N["terminal"])
causal_rescue=(rc["low"]>0)
bypass_rescue=(bc["low"]>0)
if terminal_rescue and causal_rescue:
    DECISION="L27_NEGATIVE_ORTH19_COMPONENT_CAUSALLY_SUPPRESSES_EFFECT__TARGETED_RESCUE_SUPPORTED"
elif bypass_rescue and not causal_rescue:
    DECISION="L27_BLOCK_CAUSAL_SUPPRESSION_SUPPORTED__ORTH19_COMPONENT_ALONE_NOT_SUFFICIENT"
elif terminal_rescue and not causal_rescue:
    DECISION="L27_TERMINAL_GEOMETRY_RESCUED__BEHAVIORAL_CAUSAL_RESCUE_NOT_CONFIRMED"
else:
    DECISION="L27_CAUSAL_SUPPRESSION_NOT_CONFIRMED"
print("       DECISION:",DECISION)

# ================================================================================================================
# [24/28] EFFECT RATIOS
# ================================================================================================================
print("\n[24/28] Effect ratios...")
for a in ["L27_BYPASS","L27_ORTH_RESCUE"]:
    r=B[a]
    print(f"       {a:<16} C/N={r['C']/max(abs(N['C']),EPS):.4f} KL/N={r['KL']/max(N['KL'],EPS):.4f} Ret/N={r['Ret']/max(N['Ret'],EPS):.4f}")

# ================================================================================================================
# [25/28] SAVE
# ================================================================================================================
print("\n[25/28] Saving...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("FINAL_TABLE.csv",TABLE);savecsv("BOOTSTRAP.csv",BOOT)
for a in ARMS:
    savecsv(f"FINAL_CAUSAL_{a}.csv",FINALRES[a]);savecsv(f"NEUTRAL_KL_{a}.csv",KLRES[a]);savecsv(f"NEUTRAL_RET_{a}.csv",RETRES[a])
REPORT={"test":"TEST186","parent":"TEST185","decision":DECISION,"rss":RSS,"scale":PRIMARY_SCALE,
 "arms":ARMS,"final":TABLE,"bootstrap":BOOT,
 "integrity":{"l0_l19_seasc_frozen":True,"l20_l26_untouched":True,"l27_only_intervention":True,
 "rescue_coordinate":"ORTH19","rescue_coefficient":"none; exact negative L27 projection cancellation",
 "dev_used_for_selection":False,"final_used_for_selection":False,"controller":False,"weights_modified":False}}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ================================================================================================================
# [26/28] INTEGRITY
# ================================================================================================================
print("\n[26/28] Integrity...")
print("       TEST185 frozen SEASC L0-L19 preserved: True")
print("       L20-L26 untouched: True")
print("       Only L27 intervention: True")
print("       Fixed rescue coordinate ORTH19: True")
print("       No rescue coefficient search: True")
print("       DEV selection: False")
print("       FINAL selection: False")
print("       Controller: False")
print("       Weight modification: False")

# ================================================================================================================
# [27/28] SCIENTIFIC BOUNDARY
# ================================================================================================================
print("\n[27/28] Scientific boundary...")
print("       BYPASS tests the causal role of the whole L27 transformation.")
print("       ORTH_RESCUE tests only the negative L27 displacement component along fixed ORTH19.")
print("       ORTH_RESCUE does not inject a tuned magnitude; it cancels only the observed negative within-pass L27 component.")
print("       A geometric terminal rescue alone is not sufficient evidence of behavioral causal rescue.")
print("       Behavioral rescue requires FINAL causal improvement; bootstrap CI is reported without FINAL-based tuning.")

# ================================================================================================================
# [28/28] COMPLETE
# ================================================================================================================
print("\n[28/28] COMPLETE")
print("="*138)
print("TEST 186 — L27 CAUSAL RESCUE ABLATION — SUMMARY")
print("="*138)
for r in TABLE:
    print(f"{r['arm']:<16} C={r['C']:+.6f} P+={r['P']}/8 KL={r['KL']:.6f} Ret={r['Ret']:.6f} terminal={r['terminal']:+.6f}")
for r in BOOT:
    if r["metric"]=="C":print(f"{r['arm']} ΔC={r['mean']:+.6f} [{r['low']:+.6f},{r['high']:+.6f}]")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*138)
