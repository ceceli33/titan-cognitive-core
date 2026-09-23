# =============================================================================
# TEST 146 — AKBASCORE PARAMETER-SPACE INTERFERENCE X-RAY
# A=ABOVE<->BELOW | B=BEFORE<->AFTER | Qwen2.5-7B-Instruct | A100
# =============================================================================

import sys,os,gc,math,time,json,random,shutil,subprocess,warnings,re
import importlib.metadata as md
from datetime import datetime,timezone

# ---- environment: MUST precede transformers/peft imports ---------------------
try: ta=md.version("torchao")
except md.PackageNotFoundError: ta=None
if ta:
    print("Removing incompatible torchao:",ta)
    r=subprocess.run([sys.executable,"-m","pip","uninstall","-y","torchao"],
                     text=True,capture_output=True)
    print(r.stdout)
    for k in list(sys.modules):
        if k=="torchao" or k.startswith("torchao."): del sys.modules[k]
try:
    md.version("torchao")
    raise RuntimeError("torchao removal failed. Restart runtime and rerun.")
except md.PackageNotFoundError: pass

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
HARD_MAX_UPDATES=2000

IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
LR=2e-5
LAMBDA_GEOMETRY=1.
LAMBDA_PRESERVE=.05
GRAD_CLIP=1.
LORA_R,LORA_ALPHA,LORA_DROPOUT=8,16,0.
MAX_TOKENS=128
PRESERVE_TOP_K=256

N_TRAIN,N_VALIDATION,N_FINAL=64,24,24
EVAL_EVERY=10
MIN_UPDATES_BEFORE_STOP=50
PATIENCE_EVALS=5
OVERSHOOT_PROGRESS_DROP=.05
ALIGNMENT_FLOOR=0.

XRAY_EVERY=10
XRAY_N_PROBES=4
XRAY_MODULES=("q_proj","k_proj","v_proj","o_proj")

SYSTEM_PROMPT="You are a concise reasoning assistant. Describe the requested relationship clearly and precisely."

OUT="/content/test146_parameter_xray"
CSV=f"{OUT}/CSV"; JSOND=f"{OUT}/JSON"; NPZD=f"{OUT}/NPZ"
FIG=f"{OUT}/FIGURES"; REPORT=f"{OUT}/REPORT"; ADAPTER=f"{OUT}/ADAPTERS"

shutil.rmtree(OUT,ignore_errors=True)
for d in (OUT,CSV,JSOND,NPZD,FIG,REPORT,ADAPTER): os.makedirs(d,exist_ok=True)

START=datetime.now(timezone.utc).isoformat()

def seed(x=0):
    s=SEED+x
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed()
torch.backends.cuda.matmul.allow_tf32=True

print("="*120)
print("TEST 146 — PARAMETER-SPACE INTERFERENCE X-RAY")
print("="*120)
print("GPU:",torch.cuda.get_device_name(0),"| PyTorch:",torch.__version__,
      "| Transformers:",transformers.__version__,"| PEFT:",peft.__version__)
print("X-ray: diagnostic only | probes:",XRAY_N_PROBES,
      "| every:",XRAY_EVERY,"| modules:",XRAY_MODULES)

# =============================================================================
# DATA / MOTOR
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
    return x[:N_TRAIN],x[N_TRAIN:N_TRAIN+N_VALIDATION],x[N_TRAIN+N_VALIDATION:N_TRAIN+N_VALIDATION+N_FINAL]

A_TRAIN,A_VAL,A_FINAL=split(A_ALL)
B_TRAIN,B_VAL,B_FINAL=split(B_ALL)

# =============================================================================
# MODEL
# =============================================================================

tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token

def enc(prompt):
    text=tok.apply_chat_template(
        [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        tokenize=False,add_generation_prompt=True)
    x=tok(text,return_tensors="pt",truncation=True,max_length=MAX_TOKENS)
    return {k:v.to(DEVICE) for k,v in x.items()}

def clean():
    gc.collect(); torch.cuda.empty_cache()
    try: torch.cuda.ipc_collect()
    except: pass

def load_base():
    clean()
    print(f"[VRAM] before model load: {torch.cuda.memory_allocated()/GB:.3f} GB allocated")
    m=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,dtype=torch.bfloat16,device_map={"":0},
        attn_implementation="sdpa",trust_remote_code=True)
    m.config.use_cache=False
    return m

def layers(m):
    for fn in (
        lambda:m.model.layers,
        lambda:m.model.model.layers,
        lambda:m.base_model.model.model.layers,
        lambda:m.base_model.model.model.model.layers):
        try:
            x=fn()
            if len(x)>=N_LAYERS:return x
        except: pass
    raise RuntimeError("Decoder layers unresolved.")

def lc():
    return LoraConfig(r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=LORA_DROPOUT,
        bias="none",task_type="CAUSAL_LM",target_modules=list(XRAY_MODULES))

def new_student(): return get_peft_model(load_base(),lc())
def add_lora(m): return get_peft_model(m,lc())

def merge_base(path):
    m=load_base()
    p=PeftModel.from_pretrained(m,path,is_trainable=False)
    return p.merge_and_unload()

def merge_seq(first,second):
    m=merge_base(first); m.eval()
    p=PeftModel.from_pretrained(m,second,is_trainable=False)
    return p.merge_and_unload()

# =============================================================================
# GEOMETRY
# =============================================================================

HIDDEN=None

def cos(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
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
            for L in range(N_LAYERS): acc[L]+=o.hidden_states[L+1][0,q].float()
    out=[]; raw=[]
    for L in range(N_LAYERS):
        d=P[L]/len(pos)-N[L]/len(neg); n=d.norm().clamp_min(1e-12)
        out.append((d/n).detach().clone()); raw.append(float(n))
    if was:m.train()
    return out,raw

def steer(h,v,rho):
    n=h.float().norm(dim=-1,keepdim=True)
    return (h.float()+rho*n*v.to(h.device).float().view(1,1,-1)).to(h.dtype)

@torch.inference_mode()
def capture(m,prompt,steer_vec=None):
    was=m.training;m.eval(); x=enc(prompt)
    if steer_vec is None:
        o=m(**x,output_hidden_states=True,use_cache=False,return_dict=True)
        q=int(x["attention_mask"][0].sum())-1
        z=np.stack([o.hidden_states[L+1][0,q].float().cpu().numpy() for L in range(N_LAYERS)])
    else:
        ls=layers(m); q=int(x["attention_mask"][0].sum())-1
        cap=[None]*N_LAYERS; hs=[]
        for L in range(N_LAYERS):
            def hook(mod,args,out,L=L):
                h=out[0] if isinstance(out,tuple) else out
                s=steer(h,steer_vec[L],DOSE[L])
                cap[L]=s[0,q].float().cpu().numpy().copy()
                return (s,)+out[1:] if isinstance(out,tuple) else s
            hs.append(ls[L].register_forward_hook(hook))
        try:m(**x,use_cache=False,return_dict=True)
        finally:
            for h in hs:h.remove()
        z=np.stack(cap)
    if was:m.train()
    return z

def capset(m,ps): return np.stack([capture(m,p) for p in ps])

def refs(m,ps,c):
    b=[];t=[]
    for i,p in enumerate(ps,1):
        b.append(capture(m,p)); t.append(capture(m,p,c))
        print(f"reference {i:02d}/{len(ps)}")
    return np.stack(b),np.stack(t)

def metrics(S,B,T):
    d=np.linalg.norm(S-T,axis=2)
    d0=np.maximum(np.linalg.norm(T-B,axis=2),1e-12)
    P=1-d/d0
    A=np.zeros_like(P)
    for i in range(len(S)):
        for L in range(N_LAYERS): A[i,L]=cos(S[i,L]-B[i,L],T[i,L]-B[i,L])
    return dict(progress=float(P.mean()),distance=float(d.mean()),
                alignment=float(A.mean()),layer_progress=P.mean(0),
                layer_alignment=A.mean(0))

# =============================================================================
# IMMUTABLE BASE
# =============================================================================

base=load_base(); base.eval()
for p in base.parameters():p.requires_grad_(False)
HIDDEN=int(base.config.hidden_size)

A0,_=compass(base,A_POS,A_NEG); B0,_=compass(base,B_POS,B_NEG)
A0=[x.float().cpu() for x in A0]; B0=[x.float().cpu() for x in B0]

A_VAL_B,A_VAL_T=refs(base,A_VAL,A0)
B_VAL_B,B_VAL_T=refs(base,B_VAL,B0)
A_FIN_B,A_FIN_T=refs(base,A_FINAL,A0)
B_FIN_B,B_FIN_T=refs(base,B_FINAL,B0)

@torch.inference_mode()
def topk_base(m,p):
    x=enc(p);o=m(**x,use_cache=False,return_dict=True)
    q=int(x["attention_mask"][0].sum())-1
    v,i=torch.topk(o.logits[0,q].float(),min(PRESERVE_TOP_K,o.logits.shape[-1]))
    return i.cpu().int(),v.cpu().half()

CACHE={}
for p in dict.fromkeys(A_TRAIN+B_TRAIN): CACHE[p]=topk_base(base,p)

del base;clean()
print(f"[VRAM] immutable base released: {torch.cuda.memory_allocated()/GB:.3f} GB allocated")

# =============================================================================
# TRAINING OBJECTIVE
# =============================================================================

@torch.no_grad()
def teacher(m,prompt,c):
    was=m.training;m.eval(); x=enc(prompt); q=int(x["attention_mask"][0].sum())-1
    ls=layers(m); cap=[None]*N_LAYERS; hs=[]
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

def train_loss(m,prompt,target):
    x=enc(prompt); q=int(x["attention_mask"][0].sum())-1
    o=m(**x,output_hidden_states=True,use_cache=False,return_dict=True)
    geo=[]
    for L in range(N_LAYERS):
        h=o.hidden_states[L+1][0,q].float()
        t=target[L].to(DEVICE).float()
        geo.append((h-t).square().sum()/t.square().sum().detach().clamp_min(1e-6))
    geo=torch.stack(geo).mean()
    ii,bl=CACHE[prompt]; ii=ii.to(DEVICE).long(); bl=bl.to(DEVICE).float()
    kl=F.kl_div(F.log_softmax(o.logits[0,q].float()[ii],-1),
                F.softmax(bl,-1),reduction="sum")
    return LAMBDA_GEOMETRY*geo+LAMBDA_PRESERVE*kl,geo.detach(),kl.detach()

# =============================================================================
# X-RAY
# =============================================================================

XRAY=[]

def lora_params(m):
    out={}
    for n,p in m.named_parameters():
        if p.requires_grad and "lora_" in n:
            mt=re.search(r"layers\.(\d+)\.",n)
            mod=next((x for x in XRAY_MODULES if f".{x}." in n),None)
            if mt and mod and int(mt.group(1))<N_LAYERS:
                out[n]=(p,int(mt.group(1)),mod)
    if not out: raise RuntimeError("No LoRA parameters found for X-ray.")
    return out

def probe_loss(m,prompt,B,T):
    x=enc(prompt);q=int(x["attention_mask"][0].sum())-1
    o=m(**x,output_hidden_states=True,use_cache=False,return_dict=True)
    z=[]
    for L in range(N_LAYERS):
        h=o.hidden_states[L+1][0,q].float()
        t=torch.as_tensor(T[L],device=DEVICE,dtype=torch.float32)
        b=torch.as_tensor(B[L],device=DEVICE,dtype=torch.float32)
        scale=(t-b).square().sum().clamp_min(1e-6)
        z.append((h-t).square().sum()/scale)
    return torch.stack(z).mean()

def probe_objective(m,ps,B,T):
    return torch.stack([probe_loss(m,p,B[i],T[i]) for i,p in enumerate(ps)]).mean()

def grad_map(loss,P,retain_graph=False):
    names=list(P)
    gs=torch.autograd.grad(loss,[P[n][0] for n in names],
                           retain_graph=retain_graph,allow_unused=True)
    return {n:(torch.zeros_like(P[n][0]) if g is None else g.detach().clone())
            for n,g in zip(names,gs)}

def pair_stats(a,b,names):
    dot=na=nb=0.
    for n in names:
        x=a[n].float();y=b[n].float()
        dot+=float((x*y).sum())
        na+=float(x.square().sum());nb+=float(y.square().sum())
    den=math.sqrt(na*nb)
    return dot,(dot/den if den>1e-30 else 0.),math.sqrt(na),math.sqrt(nb)

def grouped_rows(branch,step,gR,gN,before,after,P,retain_before):
    groups={}
    for n,(_,L,M) in P.items():groups.setdefault((L,M),[]).append(n)
    rows=[]
    for (L,M),ns in groups.items():
        dot,co,nr,nn=pair_stats(gR,gN,ns)
        delta={n:(after[n]-before[n]) for n in ns}
        sd,sc,_,dn=pair_stats(gR,delta,ns)
        rows.append(dict(branch=branch,step=step,layer=L,module=M,
            grad_dot=dot,grad_cosine=co,retain_grad_norm=nr,new_grad_norm=nn,
            raw_conflict=int(dot<0),retain_step_dot=sd,
            retain_step_cosine=sc,actual_step_norm=dn,
            predicted_harm=int(sd>0),retain_loss_before=retain_before))
    return rows

def xpre(m,branch,step,retain_ps,retain_B,retain_T,new_ps,new_B,new_T):
    P=lora_params(m)
    rps=retain_ps[:XRAY_N_PROBES]; nps=new_ps[:XRAY_N_PROBES]
    rB,rT=retain_B[:XRAY_N_PROBES],retain_T[:XRAY_N_PROBES]
    nB,nT=new_B[:XRAY_N_PROBES],new_T[:XRAY_N_PROBES]

    was=m.training;m.eval()
    LR=probe_objective(m,rps,rB,rT)
    LN=probe_objective(m,nps,nB,nT)
    gR=grad_map(LR,P,retain_graph=True)
    gN=grad_map(LN,P,retain_graph=False)
    before={n:p.detach().clone() for n,(p,_,_) in P.items()}
    rb=float(LR.detach().cpu())
    if was:m.train()

    return dict(P=P,gR=gR,gN=gN,before=before,retain_before=rb,
                rps=rps,rB=rB,rT=rT)

def xpost(m,branch,step,X):
    after={n:p.detach().clone() for n,(p,_,_) in X["P"].items()}
    rows=grouped_rows(branch,step,X["gR"],X["gN"],X["before"],after,
                      X["P"],X["retain_before"])

    was=m.training;m.eval()
    with torch.no_grad():
        ra=float(probe_objective(m,X["rps"],X["rB"],X["rT"]).cpu())
    if was:m.train()

    actual=ra-X["retain_before"]
    for r in rows:
        r["retain_loss_after"]=ra
        r["actual_delta_loss"]=actual
        r["sign_match"]=int((r["retain_step_dot"]>0)==(actual>0))

    XRAY.extend(rows)

    gdot=sum(r["grad_dot"] for r in rows)
    sdot=sum(r["retain_step_dot"] for r in rows)
    print(f"XRAY {branch} step={step:04d} | gR·gN={gdot:+.4e} | "
          f"gR·Δθ={sdot:+.4e} | ΔLretain={actual:+.4e}")

    del after
    for k in ("gR","gN","before"): X[k].clear()
    del X
    clean()

# =============================================================================
# TRAIN
# =============================================================================

TRAIN=[];EVAL=[]

def crystallize(m,branch,crystal,train_ps,pos,neg,immutable,
                val_ps,val_B,val_T,secondary_ps,sec_B,sec_T,
                sec_baseline=None,seed_offset=0,
                xray=None):

    print("\n"+"="*120+"\n",branch,"\n"+"="*120)
    seed(seed_offset)
    cur,_=compass(m,pos,neg)
    params=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=LR,betas=(.9,.95),weight_decay=.01)

    best_dir=f"{ADAPTER}/{branch}_BEST"
    shutil.rmtree(best_dir,ignore_errors=True);os.makedirs(best_dir,exist_ok=True)

    P0=metrics(capset(m,val_ps),val_B,val_T)
    S0=metrics(capset(m,secondary_ps),sec_B,sec_T)
    if sec_baseline is None:sec_baseline=S0

    bestP,bestD,bestA=P0["progress"],P0["distance"],P0["alignment"]
    best_step=0;pat=0;stop=None
    m.save_pretrained(best_dir);tok.save_pretrained(best_dir)

    rng=random.Random(SEED+seed_offset);step=0;epoch=0

    while step<HARD_MAX_UPDATES and stop is None:
        epoch+=1;order=list(range(len(train_ps)));rng.shuffle(order)
        for idx in order:
            if step>=HARD_MAX_UPDATES:break
            step+=1;p=train_ps[idx]

            target=teacher(m,p,cur)
            opt.zero_grad(set_to_none=True)
            loss,geo,kl=train_loss(m,p,target)
            if not torch.isfinite(loss):raise RuntimeError("Non-finite loss.")
            loss.backward()

            X=None
            if xray and step%XRAY_EVERY==0:
                X=xpre(m,branch,step,*xray)

            gn=torch.nn.utils.clip_grad_norm_(params,GRAD_CLIP)
            if not torch.isfinite(torch.as_tensor(gn)):raise RuntimeError("Non-finite gradient.")
            opt.step()

            if X is not None:xpost(m,branch,step,X)

            new,_=compass(m,pos,neg)
            del cur
            cur=[x.detach().clone() for x in new];del new

            TRAIN.append(dict(branch=branch,step=step,epoch=epoch,
                loss=float(loss.detach().cpu()),geometry=float(geo.cpu()),
                preserve=float(kl.cpu()),grad_norm=float(torch.as_tensor(gn).cpu())))

            print(f"{branch:8s} step={step:04d} loss={float(loss):.6f} "
                  f"geo={float(geo):.6f} pres={float(kl):.6f}")

            del target,loss,geo,kl

            if step%EVAL_EVERY==0:
                P=metrics(capset(m,val_ps),val_B,val_T)
                S=metrics(capset(m,secondary_ps),sec_B,sec_T)
                interference=S["progress"]-sec_baseline["progress"]

                EVAL.append(dict(branch=branch,step=step,
                    primary_progress=P["progress"],primary_distance=P["distance"],
                    primary_alignment=P["alignment"],
                    secondary_progress=S["progress"],
                    secondary_interference=interference))

                print(f"EVAL P={100*P['progress']:+.4f}% "
                      f"S={100*S['progress']:+.4f}% "
                      f"I={100*interference:+.4f}pp")

                improved=P["progress"]>bestP and P["alignment"]>=ALIGNMENT_FLOOR

                if improved:
                    bestP,bestD,bestA=P["progress"],P["distance"],P["alignment"]
                    best_step=step;pat=0
                    shutil.rmtree(best_dir,ignore_errors=True)
                    m.save_pretrained(best_dir);tok.save_pretrained(best_dir)
                else:pat+=1

                if step>=MIN_UPDATES_BEFORE_STOP and bestP-P["progress"]>=OVERSHOOT_PROGRESS_DROP:
                    stop="OVERSHOOT"
                elif step>=MIN_UPDATES_BEFORE_STOP and pat>=PATIENCE_EVALS:
                    stop="HELDOUT_EQUILIBRIUM"
                if stop:break

    if stop is None:stop="HARD_MAX"
    opt.zero_grad(set_to_none=True)
    del opt,params,cur
    clean()

    return dict(best_dir=best_dir,best_step=best_step,stop_reason=stop,
                best_progress=bestP,initial_primary=P0,initial_secondary=S0,
                secondary_baseline=sec_baseline)

# =============================================================================
# MERGE DRIFT — SAME LIVE MODEL
# =============================================================================

MERGE=[]

def drift_base(path,ps,label):
    parent=load_base()
    p=PeftModel.from_pretrained(parent,path,is_trainable=False);p.eval()
    before=capset(p,ps)
    m=p.merge_and_unload();m.eval();m.config.use_cache=False
    after=capset(m,ps)
    per=np.array([np.linalg.norm(after[:,L]-before[:,L])/
                  max(np.linalg.norm(before[:,L]),1e-12) for L in range(N_LAYERS)])
    d=float(np.linalg.norm(after-before)/max(np.linalg.norm(before),1e-12))
    MERGE.append(dict(stage=label,relative_hidden_drift=d,
                      max_layer_drift=float(per.max()),status="MEASURED_ONLY_NO_THRESHOLD"))
    del before,after,p,parent;clean()
    return m

def drift_seq(first,second,ps,label):
    parent=merge_base(first);parent.eval()
    p=PeftModel.from_pretrained(parent,second,is_trainable=False);p.eval()
    before=capset(p,ps)
    m=p.merge_and_unload();m.eval();m.config.use_cache=False
    after=capset(m,ps)
    per=np.array([np.linalg.norm(after[:,L]-before[:,L])/
                  max(np.linalg.norm(before[:,L]),1e-12) for L in range(N_LAYERS)])
    d=float(np.linalg.norm(after-before)/max(np.linalg.norm(before),1e-12))
    MERGE.append(dict(stage=label,relative_hidden_drift=d,
                      max_layer_drift=float(per.max()),status="MEASURED_ONLY_NO_THRESHOLD"))
    del before,after,p,parent;clean()
    return m

# =============================================================================
# FOUR BRANCHES
# =============================================================================

# BASE -> A
m=new_student()
AR=crystallize(m,"A_ONLY","A",A_TRAIN,A_POS,A_NEG,A0,
               A_VAL,A_VAL_B,A_VAL_T,B_VAL,B_VAL_B,B_VAL_T,seed_offset=10)
del m;clean()

AM=drift_base(AR["best_dir"],A_VAL,"A_ONLY")
A_ONLY_A=metrics(capset(AM,A_VAL),A_VAL_B,A_VAL_T)
A_ONLY_B=metrics(capset(AM,B_VAL),B_VAL_B,B_VAL_T)

# A -> B : X-ray retention=A, new=B
m=add_lora(AM);AM=None
ABR=crystallize(
    m,"A_TO_B","B",B_TRAIN,B_POS,B_NEG,B0,
    B_VAL,B_VAL_B,B_VAL_T,A_VAL,A_VAL_B,A_VAL_T,A_ONLY_A,20,
    xray=(A_VAL,A_VAL_B,A_VAL_T,B_VAL,B_VAL_B,B_VAL_T))
del m;clean()

ABM=drift_seq(AR["best_dir"],ABR["best_dir"],B_VAL,"A_TO_B")
del ABM;clean()

# BASE -> B
m=new_student()
BR=crystallize(m,"B_ONLY","B",B_TRAIN,B_POS,B_NEG,B0,
               B_VAL,B_VAL_B,B_VAL_T,A_VAL,A_VAL_B,A_VAL_T,seed_offset=30)
del m;clean()

BM=drift_base(BR["best_dir"],B_VAL,"B_ONLY")
B_ONLY_B=metrics(capset(BM,B_VAL),B_VAL_B,B_VAL_T)
B_ONLY_A=metrics(capset(BM,A_VAL),A_VAL_B,A_VAL_T)

# B -> A : X-ray retention=B, new=A
m=add_lora(BM);BM=None
BAR=crystallize(
    m,"B_TO_A","A",A_TRAIN,A_POS,A_NEG,A0,
    A_VAL,A_VAL_B,A_VAL_T,B_VAL,B_VAL_B,B_VAL_T,B_ONLY_B,40,
    xray=(B_VAL,B_VAL_B,B_VAL_T,A_VAL,A_VAL_B,A_VAL_T))
del m;clean()

BAM=drift_seq(BR["best_dir"],BAR["best_dir"],A_VAL,"B_TO_A")
del BAM;clean()

# =============================================================================
# FINAL EXAM
# =============================================================================

FINAL={}

def final(m,label):
    AS=capset(m,A_FINAL);BS=capset(m,B_FINAL)
    A=metrics(AS,A_FIN_B,A_FIN_T);B=metrics(BS,B_FIN_B,B_FIN_T)
    FINAL[label]=dict(A=A,B=B,A_states=AS,B_states=BS)
    print(label,"A=",100*A["progress"],"B=",100*B["progress"])

m=merge_base(AR["best_dir"]);final(m,"A_ONLY");del m;clean()
m=merge_base(BR["best_dir"]);final(m,"B_ONLY");del m;clean()
m=merge_seq(AR["best_dir"],ABR["best_dir"]);final(m,"A_TO_B");del m;clean()
m=merge_seq(BR["best_dir"],BAR["best_dir"]);final(m,"B_TO_A");del m;clean()

A_INT=FINAL["A_TO_B"]["A"]["layer_progress"]-FINAL["A_ONLY"]["A"]["layer_progress"]
B_INT=FINAL["B_TO_A"]["B"]["layer_progress"]-FINAL["B_ONLY"]["B"]["layer_progress"]
IA=float(FINAL["A_TO_B"]["A"]["progress"]-FINAL["A_ONLY"]["A"]["progress"])
IB=float(FINAL["B_TO_A"]["B"]["progress"]-FINAL["B_ONLY"]["B"]["progress"])

# =============================================================================
# X-RAY AGGREGATION — BRANCH × LAYER, NOT STEP × LAYER
# =============================================================================

xdf=pd.DataFrame(XRAY)

if len(xdf):
    layer=xdf.groupby(["branch","layer"],as_index=False).agg(
        mean_grad_cosine=("grad_cosine","mean"),
        mean_grad_dot=("grad_dot","mean"),
        raw_conflict_fraction=("raw_conflict","mean"),
        mean_retain_step_dot=("retain_step_dot","mean"),
        mean_retain_step_cosine=("retain_step_cosine","mean"),
        mean_actual_step_norm=("actual_step_norm","mean"),
        predicted_harm_fraction=("predicted_harm","mean"))

    layer["final_representation_interference"]=np.nan
    for i,r in layer.iterrows():
        v=A_INT if r.branch=="A_TO_B" else B_INT
        layer.loc[i,"final_representation_interference"]=v[int(r.layer)]
    layer["final_representation_damage"]=-layer["final_representation_interference"]

    step=xdf.groupby(["branch","step"],as_index=False).agg(
        predicted_delta_loss=("retain_step_dot","sum"),
        actual_delta_loss=("actual_delta_loss","first"))
    step["predicted_harm"]=step.predicted_delta_loss>0
    step["actual_harm"]=step.actual_delta_loss>0
    step["sign_match"]=step.predicted_harm==step.actual_harm

    module=xdf.groupby(["branch","module"],as_index=False).agg(
        mean_grad_cosine=("grad_cosine","mean"),
        raw_conflict_fraction=("raw_conflict","mean"),
        mean_retain_step_dot=("retain_step_dot","mean"),
        predicted_harm_fraction=("predicted_harm","mean"))

    CORR=[]
    for branch,g in layer.groupby("branch"):
        for metric in ("mean_grad_cosine","mean_grad_dot","raw_conflict_fraction",
                       "mean_retain_step_dot","mean_retain_step_cosine",
                       "mean_actual_step_norm","predicted_harm_fraction"):
            x=g[metric].to_numpy(float);y=g.final_representation_damage.to_numpy(float)
            c=np.nan if np.std(x)<1e-15 or np.std(y)<1e-15 else np.corrcoef(x,y)[0,1]
            CORR.append(dict(branch=branch,metric=metric,
                             correlation_with_final_damage=float(c)))
    corr=pd.DataFrame(CORR)
else:
    layer=step=module=corr=pd.DataFrame()

# =============================================================================
# SAVE
# =============================================================================

pd.DataFrame(TRAIN).to_csv(f"{CSV}/TEST146_training.csv",index=False)
pd.DataFrame(EVAL).to_csv(f"{CSV}/TEST146_validation.csv",index=False)
xdf.to_csv(f"{CSV}/TEST146_xray_raw.csv",index=False)
layer.to_csv(f"{CSV}/TEST146_xray_layer_summary.csv",index=False)
step.to_csv(f"{CSV}/TEST146_xray_step_summary.csv",index=False)
module.to_csv(f"{CSV}/TEST146_xray_module_summary.csv",index=False)
corr.to_csv(f"{CSV}/TEST146_xray_layer_correlations.csv",index=False)
pd.DataFrame(MERGE).to_csv(f"{CSV}/TEST146_merge_drift.csv",index=False)

np.savez_compressed(
    f"{NPZD}/TEST146_geometry.npz",
    A0=np.stack([x.numpy() for x in A0]),
    B0=np.stack([x.numpy() for x in B0]),
    envelope=ENV,dose=DOSE,
    A_interference=A_INT,B_interference=B_INT,
    A_only_A=FINAL["A_ONLY"]["A_states"],
    B_only_B=FINAL["B_ONLY"]["B_states"],
    AB_A=FINAL["A_TO_B"]["A_states"],
    AB_B=FINAL["A_TO_B"]["B_states"],
    BA_A=FINAL["B_TO_A"]["A_states"],
    BA_B=FINAL["B_TO_A"]["B_states"])

def safejson(x):
    if isinstance(x,dict):return {str(k):safejson(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safejson(v) for v in x]
    if isinstance(x,np.ndarray):return safejson(x.tolist())
    if isinstance(x,np.generic):return safejson(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

STEP_SUM={}
if len(step):
    for b,g in step.groupby("branch"):
        STEP_SUM[b]={
            "n_xray_steps":len(g),
            "predicted_harm_fraction":float(g.predicted_harm.mean()),
            "actual_harm_fraction":float(g.actual_harm.mean()),
            "sign_match_fraction":float(g.sign_match.mean()),
            "mean_predicted_delta_loss":float(g.predicted_delta_loss.mean()),
            "mean_actual_delta_loss":float(g.actual_delta_loss.mean())}

SUMMARY={
"experiment":"TEST 146 — Parameter-Space Interference X-Ray",
"model":MODEL_ID,"seed":SEED,
"xray":{
 "diagnostic_only":True,"every_updates":XRAY_EVERY,"probes":XRAY_N_PROBES,
 "modules":XRAY_MODULES,
 "raw_gradient_interpretation":
 "g_retain·g_new < 0 = held-out immutable probe-gradient conflict under simple gradient descent.",
 "optimizer_step_interpretation":
 "g_retain·Δtheta > 0 = first-order predicted retention-loss increase.",
 "aggregation":"X-ray statistics aggregated to branch×layer before final layer correlation.",
 "step_summary":STEP_SUM},
"results":{
 "I_A_from_B":IA,"I_B_from_A":IB,
 "A_only_A":FINAL["A_ONLY"]["A"]["progress"],
 "B_only_B":FINAL["B_ONLY"]["B"]["progress"],
 "AB_A":FINAL["A_TO_B"]["A"]["progress"],
 "AB_B":FINAL["A_TO_B"]["B"]["progress"],
 "BA_A":FINAL["B_TO_A"]["A"]["progress"],
 "BA_B":FINAL["B_TO_A"]["B"]["progress"]},
"best_steps":{"A_ONLY":AR["best_step"],"A_TO_B":ABR["best_step"],
              "B_ONLY":BR["best_step"],"B_TO_A":BAR["best_step"]},
"merge_drift_policy":"measured only; no pass/fail threshold",
"merge_drift":MERGE,
"runtime":{"motor":"OFF","seasc":"OFF","hooks":"NONE","lora":"MERGED"},
"start_utc":START,"end_utc":datetime.now(timezone.utc).isoformat()}

with open(f"{JSOND}/TEST146_summary.json","w",encoding="utf-8") as f:
    json.dump(safejson(SUMMARY),f,ensure_ascii=False,indent=2,allow_nan=False)

# =============================================================================
# FIGURES / REPORT
# =============================================================================

if len(layer):
    for branch,g in layer.groupby("branch"):
        g=g.sort_values("layer")
        fig=plt.figure(figsize=(12,6))
        plt.plot(g.layer,g.mean_retain_step_dot,marker="o",label="mean gR·Δθ")
        plt.axhline(0,linewidth=.8);plt.xticks(range(N_LAYERS))
        plt.xlabel("Layer");plt.ylabel("First-order retention step effect")
        plt.title(f"TEST 146 — {branch} Parameter-Space Interference")
        plt.legend();plt.tight_layout()
        fig.savefig(f"{FIG}/TEST146_{branch}_XRAY.png",dpi=180);plt.close(fig)

report=f"""TEST 146 — PARAMETER-SPACE INTERFERENCE X-RAY
======================================================================

Model: {MODEL_ID}
A: ABOVE <-> BELOW
B: BEFORE <-> AFTER

FINAL REPRESENTATION INTERFERENCE
I[A <- B] = {100*IA:+.6f} pp
I[B <- A] = {100*IB:+.6f} pp

X-RAY SEMANTICS
g_retain·g_new < 0:
held-out immutable probe-gradient conflict under simple gradient descent.

g_retain·DeltaTheta > 0:
actual optimizer step is first-order predicted to increase retention loss.

Actual probe DeltaLoss > 0:
measured immediate retention-probe damage after the optimizer step.

Final layer correlations use 20 branch×layer observations per branch.
Step×layer rows are NOT treated as independent final observations.

X-ray is diagnostic only.
No gradient projection.
No anti-interference controller.
No gradient modification.
No X-ray at step 0.
No final-test information used for training or model selection.

MERGE DRIFT
{json.dumps(safejson(MERGE),indent=2)}

FINAL RUNTIME
Motor: OFF
SEASC: OFF
Hooks: NONE
LoRA: MERGED
"""

with open(f"{REPORT}/TEST146_REPORT.txt","w",encoding="utf-8") as f:f.write(report)

print("\n"+"="*120)
print("TEST 146 — FINAL")
print("="*120)
print(f"I[A <- B] : {100*IA:+.6f} pp")
print(f"I[B <- A] : {100*IB:+.6f} pp")

if len(step):
    for b,g in step.groupby("branch"):
        print(f"\n{b}")
        print(f"X-ray steps        : {len(g)}")
        print(f"Predicted harm     : {100*g.predicted_harm.mean():.2f}%")
        print(f"Actual harm        : {100*g.actual_harm.mean():.2f}%")
        print(f"Sign agreement     : {100*g.sign_match.mean():.2f}%")
        print(f"Mean gR·Δtheta     : {g.predicted_delta_loss.mean():+.6e}")
        print(f"Mean actual Δloss  : {g.actual_delta_loss.mean():+.6e}")

if len(corr):
    print("\nLAYER CORRELATIONS WITH FINAL DAMAGE")
    print(corr.to_string(index=False))

print("\nMERGE DRIFT")
for r in MERGE:
    print(r["stage"],f"{r['relative_hidden_drift']:.8e}",
          "|",r["status"])

print("\nMotor: OFF | SEASC: OFF | Hooks: NONE | LoRA: MERGED")
print(f"VRAM: {torch.cuda.memory_allocated()/GB:.3f} GB allocated")

ZIP="/content/TEST146_AKBASCORE_PARAMETER_XRAY_COMPLETE"
ZIP_PATH=shutil.make_archive(ZIP,"zip",OUT)
print("\nZIP:",ZIP_PATH)

from google.colab import files
files.download(ZIP_PATH)

print("="*120)
print("TEST 146 COMPLETE")
print("="*120)
