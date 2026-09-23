# =============================================================================
# TEST 147 — AKBASCORE PARAMETER-SPACE RETENTION GUARD
# A=ABOVE<->BELOW | B=BEFORE<->AFTER | Qwen2.5-7B-Instruct | A100
# TEST146 follow-up: control harmful optimizer displacement g_retain·DeltaTheta
# =============================================================================

import sys,os,gc,math,json,random,shutil,subprocess,warnings,re
import importlib.metadata as md
from datetime import datetime,timezone

# ---- environment -------------------------------------------------------------
try: ta=md.version("torchao")
except md.PackageNotFoundError: ta=None
if ta:
    print("Removing incompatible torchao:",ta)
    subprocess.run([sys.executable,"-m","pip","uninstall","-y","torchao"],
                   stdout=subprocess.DEVNULL)
    for k in list(sys.modules):
        if k=="torchao" or k.startswith("torchao."): del sys.modules[k]

import numpy as np,pandas as pd,torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transformers,peft
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,PeftModel

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required.")

# =============================================================================
# CONFIG
# =============================================================================

SEED=42
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
DEVICE=torch.device("cuda")
GB=1024**3
N_LAYERS=20
HARD_MAX=2000

IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
LR=2e-5
LAMBDA_PRESERVE=.05
GRAD_CLIP=1.
LORA_R,LORA_ALPHA=8,16
MAX_TOKENS=128
TOP_K=256

N_TRAIN,N_VAL,N_FINAL=64,24,24
EVAL_EVERY=10
MIN_UPDATES=50
PATIENCE=5
OVERSHOOT=.05
ALIGN_FLOOR=0.

# Guard: one fixed held-out retention probe.
# Applied every optimizer step to actual DeltaTheta.
GUARD_PROBES=1
GUARD_EPS=1e-12

MODULES=("q_proj","k_proj","v_proj","o_proj")
SYSTEM="You are a concise reasoning assistant. Describe the requested relationship clearly and precisely."

OUT="/content/test147_retention_guard"
CSV=f"{OUT}/CSV"; JSOND=f"{OUT}/JSON"; NPZD=f"{OUT}/NPZ"
FIG=f"{OUT}/FIGURES"; REPORT=f"{OUT}/REPORT"; ADAPTER=f"{OUT}/ADAPTERS"

shutil.rmtree(OUT,ignore_errors=True)
for d in (CSV,JSOND,NPZD,FIG,REPORT,ADAPTER): os.makedirs(d,exist_ok=True)

START=datetime.now(timezone.utc).isoformat()

def seed(x=0):
    s=SEED+x
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

seed();torch.backends.cuda.matmul.allow_tf32=True

print("="*110)
print("TEST 147 — PARAMETER-SPACE RETENTION GUARD")
print("="*110)
print("GPU:",torch.cuda.get_device_name(0),
      "| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("Controller: actual optimizer displacement projection | probes:",GUARD_PROBES)

# =============================================================================
# DATA / SEASC
# =============================================================================

def envelope(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)

ENV=np.array([envelope(i) for i in range(N_LAYERS)])
DOSE=IVME*ENV

A_POS=[
"a person standing above a bridge","a person positioned above a tower",
"an object located above the roof","a bird flying above the mountain",
"a light appearing above the building","a person looking from above the valley",
"a structure extending above the wall","a signal coming from above the surface"]
A_NEG=[x.replace("above","below") for x in A_POS]

B_POS=[
"an event occurring before a meeting","a signal appearing before an alarm",
"a person arriving before a ceremony","a light turning on before a sound",
"a message arriving before a response","a vehicle stopping before a departure",
"a door opening before an announcement","a bell ringing before a performance"]
B_NEG=[x.replace("before","after") for x in B_POS]

SA=["red apple","blue cup","small lamp","wooden box",
    "green pear","silver key","yellow ball","black book"]
SB=["white plate","glass bottle","brown chair","metal tray",
    "purple vase","orange cone","grey stone","blue notebook"]
SURF=["table","desk","shelf","platform","counter"]

EA=["red light turns on","small bell rings","door opens","vehicle stops",
    "message appears","camera flashes","alarm sounds","screen changes"]
EB=["meeting begins","music starts","announcement plays","train departs",
    "reply arrives","recording starts","worker enters","timer ends"]
CTX=["during the recorded sequence","in the observed timeline",
     "during the experiment","in the event log","during the demonstration"]

A_ALL=[f"A {b} is on the {s}. The {a} is positioned {r} the {b}. "
       f"Describe their relative spatial positions precisely."
       for a in SA for b in SB for s in SURF for r in ("above","below")]

B_ALL=[f"{c.capitalize()}, the {a} {r} the {b}. "
       f"Describe the relative temporal order of the two events precisely."
       for a in EA for b in EB for c in CTX for r in ("before","after")]

random.Random(SEED).shuffle(A_ALL)
random.Random(SEED+1000).shuffle(B_ALL)

def split(x):
    return (x[:N_TRAIN],
            x[N_TRAIN:N_TRAIN+N_VAL],
            x[N_TRAIN+N_VAL:N_TRAIN+N_VAL+N_FINAL])

A_TRAIN,A_VAL,A_FINAL=split(A_ALL)
B_TRAIN,B_VAL,B_FINAL=split(B_ALL)

# =============================================================================
# MODEL
# =============================================================================

tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token

def enc(p):
    t=tok.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":p}],
        tokenize=False,add_generation_prompt=True)
    x=tok(t,return_tensors="pt",truncation=True,max_length=MAX_TOKENS)
    return {k:v.to(DEVICE) for k,v in x.items()}

def clean():
    gc.collect();torch.cuda.empty_cache()
    try:torch.cuda.ipc_collect()
    except:pass

def load_base():
    clean()
    m=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,dtype=torch.bfloat16,device_map={"":0},
        attn_implementation="sdpa",trust_remote_code=True)
    m.config.use_cache=False
    return m

def layers(m):
    for f in (
        lambda:m.model.layers,
        lambda:m.model.model.layers,
        lambda:m.base_model.model.model.layers,
        lambda:m.base_model.model.model.model.layers):
        try:
            x=f()
            if len(x)>=N_LAYERS:return x
        except:pass
    raise RuntimeError("Decoder layers unresolved.")

def lc():
    return LoraConfig(
        r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=0.,
        bias="none",task_type="CAUSAL_LM",target_modules=list(MODULES))

def new_student():return get_peft_model(load_base(),lc())
def add_lora(m):return get_peft_model(m,lc())

def merge_base(path):
    return PeftModel.from_pretrained(
        load_base(),path,is_trainable=False).merge_and_unload()

def merge_seq(first,second):
    m=merge_base(first)
    return PeftModel.from_pretrained(
        m,second,is_trainable=False).merge_and_unload()

# =============================================================================
# GEOMETRY
# =============================================================================

HIDDEN=None

def cosine(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return 0. if n<1e-12 else float(np.clip(np.dot(a,b)/n,-1,1))

@torch.inference_mode()
def compass(m,pos,neg):
    was=m.training;m.eval()
    P=[torch.zeros(HIDDEN,device=DEVICE) for _ in range(N_LAYERS)]
    N=[torch.zeros(HIDDEN,device=DEVICE) for _ in range(N_LAYERS)]
    for texts,acc in ((pos,P),(neg,N)):
        for t in texts:
            x=tok(t,return_tensors="pt",add_special_tokens=True)
            x={k:v.to(DEVICE) for k,v in x.items()}
            o=m(**x,output_hidden_states=True,use_cache=False,return_dict=True)
            q=int(x["attention_mask"][0].sum())-1
            for L in range(N_LAYERS):
                acc[L]+=o.hidden_states[L+1][0,q].float()
    out=[]
    for L in range(N_LAYERS):
        d=P[L]/len(pos)-N[L]/len(neg)
        out.append((d/d.norm().clamp_min(1e-12)).detach().clone())
    if was:m.train()
    return out

def steer(h,v,rho):
    n=h.float().norm(dim=-1,keepdim=True)
    return (h.float()+rho*n*v.to(h.device).float().view(1,1,-1)).to(h.dtype)

@torch.inference_mode()
def capture(m,p,c=None):
    was=m.training;m.eval();x=enc(p);q=int(x["attention_mask"][0].sum())-1
    if c is None:
        o=m(**x,output_hidden_states=True,use_cache=False,return_dict=True)
        z=np.stack([o.hidden_states[L+1][0,q].float().cpu().numpy()
                    for L in range(N_LAYERS)])
    else:
        cap=[None]*N_LAYERS;hs=[]
        for L in range(N_LAYERS):
            def hook(mod,args,out,L=L):
                h=out[0] if isinstance(out,tuple) else out
                s=steer(h,c[L],DOSE[L]);cap[L]=s[0,q].float().cpu().numpy().copy()
                return (s,)+out[1:] if isinstance(out,tuple) else s
            hs.append(layers(m)[L].register_forward_hook(hook))
        try:m(**x,use_cache=False,return_dict=True)
        finally:
            for h in hs:h.remove()
        z=np.stack(cap)
    if was:m.train()
    return z

def capset(m,ps):return np.stack([capture(m,p) for p in ps])

def refs(m,ps,c):
    B=[];T=[]
    for i,p in enumerate(ps,1):
        B.append(capture(m,p));T.append(capture(m,p,c))
        print(f"reference {i:02d}/{len(ps)}")
    return np.stack(B),np.stack(T)

def metrics(S,B,T):
    d=np.linalg.norm(S-T,axis=2)
    d0=np.maximum(np.linalg.norm(T-B,axis=2),1e-12)
    P=1-d/d0
    A=np.zeros_like(P)
    for i in range(len(S)):
        for L in range(N_LAYERS):
            A[i,L]=cosine(S[i,L]-B[i,L],T[i,L]-B[i,L])
    return dict(progress=float(P.mean()),distance=float(d.mean()),
                alignment=float(A.mean()),layer_progress=P.mean(0))

# =============================================================================
# IMMUTABLE BASE
# =============================================================================

base=load_base();base.eval()
for p in base.parameters():p.requires_grad_(False)
HIDDEN=int(base.config.hidden_size)

A0=[x.cpu() for x in compass(base,A_POS,A_NEG)]
B0=[x.cpu() for x in compass(base,B_POS,B_NEG)]

A_VB,A_VT=refs(base,A_VAL,A0)
B_VB,B_VT=refs(base,B_VAL,B0)
A_FB,A_FT=refs(base,A_FINAL,A0)
B_FB,B_FT=refs(base,B_FINAL,B0)

@torch.inference_mode()
def topk_base(m,p):
    x=enc(p);o=m(**x,use_cache=False,return_dict=True)
    q=int(x["attention_mask"][0].sum())-1
    v,i=torch.topk(o.logits[0,q].float(),min(TOP_K,o.logits.shape[-1]))
    return i.cpu().int(),v.cpu().half()

CACHE={p:topk_base(base,p) for p in dict.fromkeys(A_TRAIN+B_TRAIN)}
del base;clean()

# =============================================================================
# TRAINING
# =============================================================================

@torch.no_grad()
def teacher(m,p,c):
    was=m.training;m.eval();x=enc(p);q=int(x["attention_mask"][0].sum())-1
    ls=layers(m);cap=[None]*N_LAYERS;hs=[]
    for L in range(N_LAYERS):
        def hook(mod,args,out,L=L):
            h=out[0] if isinstance(out,tuple) else out
            s=steer(h,c[L],DOSE[L])
            cap[L]=s[0,q].detach().float().cpu().half()
            return (s,)+out[1:] if isinstance(out,tuple) else s
        hs.append(ls[L].register_forward_hook(hook))
    try:m(**x,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    if was:m.train()
    return torch.stack(cap)

def train_loss(m,p,target):
    x=enc(p);q=int(x["attention_mask"][0].sum())-1
    o=m(**x,output_hidden_states=True,use_cache=False,return_dict=True)
    geo=torch.stack([
        (o.hidden_states[L+1][0,q].float()-target[L].to(DEVICE).float())
        .square().sum()/target[L].to(DEVICE).float().square().sum().detach().clamp_min(1e-6)
        for L in range(N_LAYERS)]).mean()
    ii,bl=CACHE[p];ii=ii.to(DEVICE).long();bl=bl.to(DEVICE).float()
    kl=F.kl_div(F.log_softmax(o.logits[0,q].float()[ii],-1),
                F.softmax(bl,-1),reduction="sum")
    return geo+LAMBDA_PRESERVE*kl,geo.detach(),kl.detach()

# =============================================================================
# RETENTION GUARD
# =============================================================================

GUARD=[]

def trainable_lora(m):
    return {n:p for n,p in m.named_parameters()
            if p.requires_grad and "lora_" in n}

def retain_loss(m,ps,B,T):
    vals=[]
    for i,p in enumerate(ps[:GUARD_PROBES]):
        x=enc(p);q=int(x["attention_mask"][0].sum())-1
        o=m(**x,output_hidden_states=True,use_cache=False,return_dict=True)
        ls=[]
        for L in range(N_LAYERS):
            h=o.hidden_states[L+1][0,q].float()
            t=torch.as_tensor(T[i,L],device=DEVICE,dtype=torch.float32)
            b=torch.as_tensor(B[i,L],device=DEVICE,dtype=torch.float32)
            ls.append((h-t).square().sum()/(t-b).square().sum().clamp_min(1e-6))
        vals.append(torch.stack(ls).mean())
    return torch.stack(vals).mean()

def guard_pre(m,ps,B,T):
    P=trainable_lora(m)
    was=m.training;m.eval()
    loss=retain_loss(m,ps,B,T)
    gs=torch.autograd.grad(loss,list(P.values()),allow_unused=True)
    G={n:(torch.zeros_like(p) if g is None else g.detach())
       for (n,p),g in zip(P.items(),gs)}
    before={n:p.detach().clone() for n,p in P.items()}
    lb=float(loss.detach())
    if was:m.train()
    return P,G,before,lb

@torch.no_grad()
def apply_guard(P,G,before):
    dot=norm=raw_norm=0.
    for n,p in P.items():
        d=p.detach()-before[n]
        g=G[n]
        dot+=float((g.float()*d.float()).sum())
        norm+=float(g.float().square().sum())
        raw_norm+=float(d.float().square().sum())

    corrected=False
    alpha=0.
    if dot>0 and norm>GUARD_EPS:
        alpha=dot/norm
        for n,p in P.items():
            p.sub_(alpha*G[n].to(p.dtype))
        corrected=True

    final_dot=0.;final_norm=0.
    for n,p in P.items():
        d=p.detach()-before[n]
        final_dot+=float((G[n].float()*d.float()).sum())
        final_norm+=float(d.float().square().sum())

    return dot,final_dot,math.sqrt(raw_norm),math.sqrt(final_norm),alpha,corrected

def guard_post(m,branch,step,X):
    P,G,before,lb=X
    raw_dot,final_dot,raw_n,final_n,alpha,corrected=apply_guard(P,G,before)

    was=m.training;m.eval()
    with torch.no_grad():
        la=float(retain_loss(m,
            GUARD_CONTEXT[branch][0],
            GUARD_CONTEXT[branch][1],
            GUARD_CONTEXT[branch][2]).cpu())
    if was:m.train()

    GUARD.append(dict(
        branch=branch,step=step,
        raw_retain_step_dot=raw_dot,
        guarded_retain_step_dot=final_dot,
        raw_step_norm=raw_n,guarded_step_norm=final_n,
        projection_alpha=alpha,guard_triggered=int(corrected),
        retain_loss_before=lb,retain_loss_after=la,
        actual_delta_loss=la-lb))

    del G,before
    return raw_dot,final_dot,la-lb,corrected

# =============================================================================
# CRYSTALLIZE
# =============================================================================

TRAIN=[];EVAL=[]

def crystallize(m,branch,train_ps,pos,neg,
                val_ps,val_B,val_T,
                sec_ps,sec_B,sec_T,
                sec_baseline=None,seed_offset=0,guard=False):

    print("\n"+"="*110+"\n",branch,"\n"+"="*110)
    seed(seed_offset);cur=compass(m,pos,neg)
    params=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=LR,betas=(.9,.95),weight_decay=.01)

    best=f"{ADAPTER}/{branch}_BEST"
    shutil.rmtree(best,ignore_errors=True)

    P0=metrics(capset(m,val_ps),val_B,val_T)
    S0=metrics(capset(m,sec_ps),sec_B,sec_T)
    if sec_baseline is None:sec_baseline=S0

    bestP=P0["progress"];best_step=0;pat=0;stop=None
    m.save_pretrained(best);tok.save_pretrained(best)

    rng=random.Random(SEED+seed_offset);step=0

    while step<HARD_MAX and stop is None:
        order=list(range(len(train_ps)));rng.shuffle(order)
        for idx in order:
            if step>=HARD_MAX:break
            step+=1;p=train_ps[idx]

            target=teacher(m,p,cur)
            opt.zero_grad(set_to_none=True)
            loss,geo,kl=train_loss(m,p,target)
            if not torch.isfinite(loss):raise RuntimeError("Non-finite loss.")
            loss.backward()

            G=None
            if guard:G=guard_pre(m,*GUARD_CONTEXT[branch])

            gn=torch.nn.utils.clip_grad_norm_(params,GRAD_CLIP)
            if not torch.isfinite(torch.as_tensor(gn)):
                raise RuntimeError("Non-finite gradient.")
            opt.step()

            if guard:
                rd,gd,dl,trig=guard_post(m,branch,step,G)
                flag="*" if trig else "-"
                print(f"{branch:10s} {step:04d} L={float(loss):.5f} "
                      f"raw={rd:+.2e} guard={gd:+.2e} ΔR={dl:+.2e} {flag}")
            else:
                print(f"{branch:10s} {step:04d} L={float(loss):.5f}")

            cur=compass(m,pos,neg)

            TRAIN.append(dict(branch=branch,step=step,loss=float(loss),
                              geometry=float(geo),preserve=float(kl),
                              grad_norm=float(torch.as_tensor(gn))))

            del target,loss,geo,kl

            if step%EVAL_EVERY==0:
                P=metrics(capset(m,val_ps),val_B,val_T)
                S=metrics(capset(m,sec_ps),sec_B,sec_T)
                I=S["progress"]-sec_baseline["progress"]

                EVAL.append(dict(
                    branch=branch,step=step,
                    primary_progress=P["progress"],
                    primary_alignment=P["alignment"],
                    secondary_progress=S["progress"],
                    secondary_interference=I))

                print(f"EVAL P={100*P['progress']:+.4f}% "
                      f"S={100*S['progress']:+.4f}% I={100*I:+.4f}pp")

                improved=P["progress"]>bestP and P["alignment"]>=ALIGN_FLOOR
                if improved:
                    bestP=P["progress"];best_step=step;pat=0
                    shutil.rmtree(best,ignore_errors=True)
                    m.save_pretrained(best);tok.save_pretrained(best)
                else:pat+=1

                if step>=MIN_UPDATES and bestP-P["progress"]>=OVERSHOOT:
                    stop="OVERSHOOT"
                elif step>=MIN_UPDATES and pat>=PATIENCE:
                    stop="HELDOUT_EQUILIBRIUM"
                if stop:break

    if stop is None:stop="HARD_MAX"
    del opt,params,cur;clean()
    return dict(best_dir=best,best_step=best_step,
                best_progress=bestP,stop_reason=stop)

# =============================================================================
# EXPERIMENT
# =============================================================================

# Single-crystal references
m=new_student()
AR=crystallize(m,"A_ONLY",A_TRAIN,A_POS,A_NEG,
               A_VAL,A_VB,A_VT,B_VAL,B_VB,B_VT,seed_offset=10)
del m;clean()

AM=merge_base(AR["best_dir"]);AM.eval()
A_REF=metrics(capset(AM,A_VAL),A_VB,A_VT)

m=new_student()
BR=crystallize(m,"B_ONLY",B_TRAIN,B_POS,B_NEG,
               B_VAL,B_VB,B_VT,A_VAL,A_VB,A_VT,seed_offset=30)
del m;clean()

BM=merge_base(BR["best_dir"]);BM.eval()
B_REF=metrics(capset(BM,B_VAL),B_VB,B_VT)

# Fixed held-out retention contexts
GUARD_CONTEXT={
    "A_TO_B_G":(A_VAL,A_VB,A_VT),
    "B_TO_A_G":(B_VAL,B_VB,B_VT)
}

# A -> B protected
m=add_lora(AM);AM=None
ABR=crystallize(m,"A_TO_B_G",B_TRAIN,B_POS,B_NEG,
                B_VAL,B_VB,B_VT,A_VAL,A_VB,A_VT,
                A_REF,20,True)
del m;clean()

# B -> A protected
m=add_lora(BM);BM=None
BAR=crystallize(m,"B_TO_A_G",A_TRAIN,A_POS,A_NEG,
                A_VAL,A_VB,A_VT,B_VAL,B_VB,B_VT,
                B_REF,40,True)
del m;clean()

# =============================================================================
# FINAL
# =============================================================================

FINAL={}

def final(m,label):
    A=metrics(capset(m,A_FINAL),A_FB,A_FT)
    B=metrics(capset(m,B_FINAL),B_FB,B_FT)
    FINAL[label]=dict(A=A,B=B)
    print(label,"A=",100*A["progress"],"B=",100*B["progress"])

m=merge_base(AR["best_dir"]);final(m,"A_ONLY");del m;clean()
m=merge_base(BR["best_dir"]);final(m,"B_ONLY");del m;clean()
m=merge_seq(AR["best_dir"],ABR["best_dir"]);final(m,"A_TO_B_G");del m;clean()
m=merge_seq(BR["best_dir"],BAR["best_dir"]);final(m,"B_TO_A_G");del m;clean()

IA=float(FINAL["A_TO_B_G"]["A"]["progress"]-FINAL["A_ONLY"]["A"]["progress"])
IB=float(FINAL["B_TO_A_G"]["B"]["progress"]-FINAL["B_ONLY"]["B"]["progress"])

gdf=pd.DataFrame(GUARD)
tdf=pd.DataFrame(TRAIN)
edf=pd.DataFrame(EVAL)

gdf.to_csv(f"{CSV}/TEST147_guard.csv",index=False)
tdf.to_csv(f"{CSV}/TEST147_training.csv",index=False)
edf.to_csv(f"{CSV}/TEST147_validation.csv",index=False)

GS={}
for b,g in gdf.groupby("branch"):
    GS[b]=dict(
        steps=len(g),
        trigger_fraction=float(g.guard_triggered.mean()),
        mean_raw_dot=float(g.raw_retain_step_dot.mean()),
        mean_guarded_dot=float(g.guarded_retain_step_dot.mean()),
        mean_actual_delta_loss=float(g.actual_delta_loss.mean()),
        positive_actual_damage_fraction=float((g.actual_delta_loss>0).mean()),
        mean_step_norm_ratio=float(
            (g.guarded_step_norm/np.maximum(g.raw_step_norm,1e-12)).mean()))

def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

SUMMARY={
    "experiment":"TEST 147 — Parameter-Space Retention Guard",
    "model":MODEL_ID,
    "seed":SEED,
    "controller":{
        "type":"post-optimizer parameter displacement projection",
        "rule":"if g_retain·DeltaTheta > 0, remove the component of DeltaTheta parallel to g_retain",
        "equation":"DeltaTheta_guard = DeltaTheta - (gR·DeltaTheta / ||gR||^2) gR",
        "probes":GUARD_PROBES,
        "uses_final_test":False},
    "results":{
        "I_A_from_B_guarded":IA,
        "I_B_from_A_guarded":IB,
        "A_only_A":FINAL["A_ONLY"]["A"]["progress"],
        "B_only_B":FINAL["B_ONLY"]["B"]["progress"],
        "AB_guard_A":FINAL["A_TO_B_G"]["A"]["progress"],
        "AB_guard_B":FINAL["A_TO_B_G"]["B"]["progress"],
        "BA_guard_A":FINAL["B_TO_A_G"]["A"]["progress"],
        "BA_guard_B":FINAL["B_TO_A_G"]["B"]["progress"]},
    "best_steps":{
        "A_ONLY":AR["best_step"],"B_ONLY":BR["best_step"],
        "A_TO_B_G":ABR["best_step"],"B_TO_A_G":BAR["best_step"]},
    "guard_summary":GS,
    "start_utc":START,
    "end_utc":datetime.now(timezone.utc).isoformat()}

with open(f"{JSOND}/TEST147_summary.json","w",encoding="utf-8") as f:
    json.dump(safe(SUMMARY),f,ensure_ascii=False,indent=2,allow_nan=False)

np.savez_compressed(
    f"{NPZD}/TEST147_geometry.npz",
    envelope=ENV,dose=DOSE,
    A0=np.stack([x.numpy() for x in A0]),
    B0=np.stack([x.numpy() for x in B0]))

# =============================================================================
# FIGURE / REPORT
# =============================================================================

if len(gdf):
    fig=plt.figure(figsize=(11,6))
    for b,g in gdf.groupby("branch"):
        plt.plot(g.step,g.raw_retain_step_dot,label=f"{b} raw")
        plt.plot(g.step,g.guarded_retain_step_dot,label=f"{b} guarded")
    plt.axhline(0,linewidth=.8)
    plt.xlabel("Update");plt.ylabel("g_retain · DeltaTheta")
    plt.title("TEST 147 — Raw vs Guarded Optimizer Displacement")
    plt.legend();plt.tight_layout()
    fig.savefig(f"{FIG}/TEST147_RETENTION_GUARD.png",dpi=180)
    plt.close(fig)

report=f"""TEST 147 — PARAMETER-SPACE RETENTION GUARD
======================================================================

Model: {MODEL_ID}

CONTROLLER
If g_retain · DeltaTheta > 0:

DeltaTheta_guard =
DeltaTheta - (g_retain·DeltaTheta / ||g_retain||^2) g_retain

The controller acts on the ACTUAL AdamW parameter displacement,
not on raw training-gradient conflict.

FINAL GUARDED INTERFERENCE
I[A <- B] = {100*IA:+.6f} pp
I[B <- A] = {100*IB:+.6f} pp

GUARD SUMMARY
{json.dumps(safe(GS),indent=2)}

Final-test examples are not used by training, guard, or model selection.
"""

with open(f"{REPORT}/TEST147_REPORT.txt","w") as f:f.write(report)

print("\n"+"="*110)
print("TEST 147 — FINAL")
print("="*110)
print(f"I[A <- B] GUARDED : {100*IA:+.6f} pp")
print(f"I[B <- A] GUARDED : {100*IB:+.6f} pp")

for b,s in GS.items():
    print(f"\n{b}")
    print(f"Guard trigger       : {100*s['trigger_fraction']:.2f}%")
    print(f"Mean raw gR·Δtheta  : {s['mean_raw_dot']:+.6e}")
    print(f"Mean guard gR·Δtheta: {s['mean_guarded_dot']:+.6e}")
    print(f"Actual harm steps   : {100*s['positive_actual_damage_fraction']:.2f}%")
    print(f"Mean actual Δloss   : {s['mean_actual_delta_loss']:+.6e}")
    print(f"Step norm retained  : {100*s['mean_step_norm_ratio']:.2f}%")

ZIP=shutil.make_archive(
    "/content/TEST147_AKBASCORE_RETENTION_GUARD_COMPLETE","zip",OUT)

print("\nMotor: OFF | SEASC: OFF | Hooks: NONE | LoRA: MERGED")
print("ZIP:",ZIP)

from google.colab import files
files.download(ZIP)

print("="*110)
print("TEST 147 COMPLETE")
print("="*110)
