# =============================================================================
# TEST 150 — AKBASCORE COMPASS FOUNDRY
# ON-THE-GO COMPASS REFINEMENT — 4-ARM CAUSAL TOURNAMENT
#
# CONTROL : MEAN + ORTHO
# ARM A   : DIAG_WHITENED + ORTHO
# ARM B   : SUBSPACE_FISHER + ORTHO
# ARM C   : DIAG_WHITENED + INVARIANCE + ORTHO
#
# MODEL : Qwen/Qwen2.5-1.5B-Instruct
# MOTOR : AkbasCore SEASC — FROZEN
# TRAIN : NONE
# FINAL : UNTOUCHED UNTIL LOCK
# =============================================================================

import os, math, json, random, shutil, hashlib, warnings
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"]="false"

# =============================================================================
# 0. CONFIG
# =============================================================================

SEED=42
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE=torch.bfloat16 if DEVICE.type=="cuda" else torch.float32
N_LAYERS=20
MAX_LENGTH=128

IVME,SONUM,ZIRVE,TABAN=0.10,0.30,0.70,0.20
THETAS=(0.5,1.0,2.0)

NUISANCE_RANK=3
NUISANCE_RETENTION_MIN=0.20
DIAG_ALPHA=0.10
FISHER_ALPHA=0.10
INVARIANCE_GAMMA=1.0

K_SHUFFLE=8
K_RANDOM=3
Z_LOCK=2.0
DELTA_STD_FRAC=0.05
DELTA_FLOOR=1e-4
TAU_FRAC=0.25
SMOOTH_KERNEL=np.array([0.25,0.50,0.25],dtype=np.float64)
ALPHA_NULL=1.0
BETA_COLLATERAL=0.20
EPS=1e-8

SCREEN_N=6
VALIDATION_N=12
FINAL_N=16

OUT="/content/TEST150_COMPASS_FOUNDRY"
shutil.rmtree(OUT,ignore_errors=True)
os.makedirs(OUT,exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=True

def stable_seed(*x):
    return SEED+int.from_bytes(hashlib.sha256("|".join(map(str,x)).encode()).digest()[:4],"little")%1_000_000

def unit(v):
    v=v.float()
    return v/v.norm().clamp_min(EPS)

def cosine(a,b):
    return float(torch.clamp(torch.dot(unit(a),unit(b)),-1,1))

# =============================================================================
# 1. SEASC — FROZEN
# =============================================================================

def env(L):
    return (ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN)

SEASC_ENV=np.array([env(L) for L in range(N_LAYERS)],dtype=np.float64)
SEASC_DOSE=IVME*SEASC_ENV

# =============================================================================
# 2. MODEL
# =============================================================================

print("="*110)
print("TEST 150 — AKBASCORE COMPASS FOUNDRY — ON-THE-GO 4-ARM TOURNAMENT")
print("="*110)
if DEVICE.type=="cuda": print("GPU  :",torch.cuda.get_device_name(0))
print("Model:",MODEL_ID)

tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token

kw=dict(pretrained_model_name_or_path=MODEL_ID,dtype=DTYPE,trust_remote_code=True)
if DEVICE.type=="cuda":
    kw["device_map"]={"":0}
    kw["attn_implementation"]="sdpa"

model=AutoModelForCausalLM.from_pretrained(**kw)
if DEVICE.type!="cuda": model.to(DEVICE)
model.eval(); model.config.use_cache=False
for p in model.parameters(): p.requires_grad_(False)

HIDDEN=int(model.config.hidden_size)
try: LAYERS=model.model.layers
except: LAYERS=model.model.model.layers
assert len(LAYERS)>=N_LAYERS
print("Hidden:",HIDDEN,"| Layers:",len(LAYERS),"| Foundry:",N_LAYERS)

# =============================================================================
# 3. MATCHED COUNTERFACTUAL DATA
# =============================================================================

SUBJECTS=[
"the laboratory evidence","the recorded measurement","the available observations",
"the experimental record","the supplied evidence","the documented result",
"the collected measurements","the observed pattern","the available data",
"the recorded findings"
]
POS=[
"clearly supports the conclusion",
"provides sufficient evidence for the conclusion",
"is enough to justify the conclusion",
"strongly supports the stated conclusion"
]
NEG=[
"does not support the conclusion",
"provides insufficient evidence for the conclusion",
"is not enough to justify the conclusion",
"does not justify the stated conclusion"
]

PAIRS=[]
for si,s in enumerate(SUBJECTS):
    for fi,(p,n) in enumerate(zip(POS,NEG)):
        PAIRS.append(dict(family=fi,subject=si,pos=f"{s.capitalize()} {p}.",neg=f"{s.capitalize()} {n}."))
random.Random(SEED).shuffle(PAIRS)
CAL=PAIRS[:24]

NEUTRAL=[
"appears in the written report","was recorded during the procedure",
"is mentioned in the document","appears in the supplied material",
"was included in the summary","is described in the record"
]
NUISANCE=[f"{s.capitalize()} {p}." for s in SUBJECTS for p in NEUTRAL]

# =============================================================================
# 4. LAST-TOKEN ACTIVATIONS
# =============================================================================

@torch.inference_mode()
def hidden_last(text):
    x=tok(text,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)
    x={k:v.to(DEVICE) for k,v in x.items()}
    o=model(**x,output_hidden_states=True,use_cache=False,return_dict=True)
    q=int(x["attention_mask"][0].sum())-1
    return torch.stack([o.hidden_states[L+1][0,q].float().cpu() for L in range(N_LAYERS)])

print("\n[1/9] Extracting matched activations")
P=[]; N=[]
for i,r in enumerate(CAL,1):
    P.append(hidden_last(r["pos"])); N.append(hidden_last(r["neg"]))
    if i%4==0: print(f"  {i:02d}/{len(CAL)}")
P=torch.stack(P); N=torch.stack(N)                     # [N,L,d]
D=P-N

print("\n[2/9] Nuisance basis")
NX=torch.stack([hidden_last(x) for x in NUISANCE])     # [M,L,d]
NB=[]
for L in range(N_LAYERS):
    X=NX[:,L]-NX[:,L].mean(0,keepdim=True)
    _,_,Vh=torch.linalg.svd(X,full_matrices=False)
    NB.append(Vh[:min(NUISANCE_RANK,Vh.shape[0])].T.contiguous())

def ortho(v,L):
    v=unit(v); B=NB[L]
    z=v-B@(B.T@v)
    r=float(z.norm()/v.norm().clamp_min(EPS))
    return (unit(z),r) if r>=NUISANCE_RETENTION_MIN else (None,r)

# =============================================================================
# 5. FOUR COMPASS ARMS
# =============================================================================

def mean_compass(L):
    return unit(D[:,L].mean(0))

def diag_compass(L):
    X=D[:,L].float()
    mu=X.mean(0)
    var=X.var(0,unbiased=True)
    lam=DIAG_ALPHA*float(torch.median(var).clamp_min(EPS))
    return unit(mu/torch.sqrt(var+lam))

def fisher_compass(L):
    Hp=P[:,L].float(); Hn=N[:,L].float()
    md=Hp.mean(0)-Hn.mean(0)
    R=torch.cat([Hp-Hp.mean(0,keepdim=True),Hn-Hn.mean(0,keepdim=True)],0)
    _,s,Vh=torch.linalg.svd(R,full_matrices=False)

    # Sigma_within ~= R^T R / (2N-2)
    # (Sigma + lambda I)^-1 md, evaluated without any dxd matrix.
    denom=max(R.shape[0]-2,1)
    eig=(s*s)/denom
    lam=FISHER_ALPHA*float(torch.median(eig).clamp_min(EPS))

    V=Vh.T
    proj=V.T@md

    # exact ridge inverse action:
    # md/lambda + V[(1/(eig+lambda)-1/lambda)*(V^T md)]
    a=md/lam + V@((1.0/(eig+lam)-1.0/lam)*proj)
    return unit(a)

def invariance_mask(L):
    fam=[]
    for k in sorted(set(r["family"] for r in CAL)):
        idx=[i for i,r in enumerate(CAL) if r["family"]==k]
        X=D[idx,L].float()
        mu=X.mean(0); var=X.var(0,unbiased=True)
        lam=DIAG_ALPHA*float(torch.median(var).clamp_min(EPS))
        fam.append(unit(mu/torch.sqrt(var+lam)))
    Fm=torch.stack(fam)
    return torch.abs(torch.sign(Fm).mean(0)).pow(INVARIANCE_GAMMA)

ARM_NAMES=("CONTROL_MEAN_ORTHO","DIAG_WHITENED_ORTHO","SUBSPACE_FISHER_ORTHO","DIAG_INVARIANCE_ORTHO")
RAW={"CONTROL_MEAN_ORTHO":[],"DIAG_WHITENED_ORTHO":[],"SUBSPACE_FISHER_ORTHO":[],"DIAG_INVARIANCE_ORTHO":[]}

for L in range(N_LAYERS):
    m=mean_compass(L)
    d=diag_compass(L)
    f=fisher_compass(L)
    inv=unit(d*invariance_mask(L))
    RAW["CONTROL_MEAN_ORTHO"].append(m)
    RAW["DIAG_WHITENED_ORTHO"].append(d)
    RAW["SUBSPACE_FISHER_ORTHO"].append(f)
    RAW["DIAG_INVARIANCE_ORTHO"].append(inv)

ARMS={}; RET={}
for name in ARM_NAMES:
    ARMS[name]=[]; RET[name]=[]
    for L,v in enumerate(RAW[name]):
        z,r=ortho(v,L)
        ARMS[name].append(unit(v) if z is None else z)
        RET[name].append(r)

# Invalid ORTHO layer is not allowed to kill an entire arm; layer scan masks it.
VALID_MASK={n:np.array([r>=NUISANCE_RETENTION_MIN for r in RET[n]],dtype=np.float64) for n in ARM_NAMES}

print("\nCOMPASS GEOMETRY")
for name in ARM_NAMES:
    cc=np.array([cosine(ARMS["CONTROL_MEAN_ORTHO"][L],ARMS[name][L]) for L in range(N_LAYERS)])
    print(f"{name:28s} retention={np.mean(RET[name]):.4f}  cos(control)={cc.mean():+.4f}")

# =============================================================================
# 6. BEHAVIORAL SETS
# =============================================================================

SCREEN_FACTS=[
"Three independent measurements agree with the proposed conclusion.",
"Repeated observations consistently support the proposed conclusion.",
"Several independent records all point toward the proposed conclusion.",
"The measurements repeatedly match the proposed conclusion.",
"Independent observations converge on the same conclusion.",
"Multiple recorded measurements agree with the proposed conclusion."
]
VAL_FACTS=[
"Replicated measurements from separate trials agree with the interpretation.",
"Several independent observations consistently favor the interpretation.",
"Repeated measurements provide converging evidence for the interpretation.",
"Independent trials repeatedly produce evidence consistent with the claim.",
"The recorded observations converge strongly on the proposed interpretation.",
"Separate measurements repeatedly agree with the stated interpretation.",
"Multiple independent observations all support the proposed explanation.",
"Repeated trials provide consistent evidence favoring the conclusion.",
"Several separately recorded measurements converge on the conclusion.",
"Independent measurements repeatedly align with the proposed explanation.",
"Multiple observations consistently agree with the stated conclusion.",
"Replicated observations provide convergent evidence for the claim."
]
FINAL_FACTS=[
"Separate replicated observations consistently favor the stated inference.",
"Evidence collected independently across trials converges on the inference.",
"Several separately obtained measurements consistently agree with the claim.",
"Repeated independent observations converge on the proposed inference.",
"Measurements obtained in separate runs consistently support the inference.",
"Multiple independent records converge on the same interpretation.",
"Repeated observations obtained separately agree with the stated inference.",
"Several replicated measurements consistently favor the proposed claim.",
"Independent evidence gathered across trials converges on the conclusion.",
"Separate observations repeatedly agree with the proposed interpretation.",
"Replicated evidence from independent runs consistently favors the claim.",
"Multiple separately collected observations converge on the same inference.",
"Repeated independent measurements agree with the proposed conclusion.",
"Evidence obtained in separate trials consistently supports the inference.",
"Several independent records converge on the proposed conclusion.",
"Replicated observations consistently agree with the stated inference."
]

def prompt(s):
    return f"Read the evidence statement.\nEvidence: {s}\nQuestion: Is the evidence described as sufficient to support the proposed conclusion?\nAnswer only YES or NO:"

SCREEN=[prompt(x) for x in SCREEN_FACTS[:SCREEN_N]]
VAL=[prompt(x) for x in VAL_FACTS[:VALIDATION_N]]
FINAL=[prompt(x) for x in FINAL_FACTS[:FINAL_N]]
assert not(set(SCREEN)&set(VAL)) and not(set(SCREEN)&set(FINAL)) and not(set(VAL)&set(FINAL))

COLLATERAL=[
"The wooden box remained on the table near the window.",
"A blue notebook was placed on the shelf in the afternoon.",
"The train reached the station shortly after noon.",
"The researcher recorded the temperature at regular intervals.",
"The lamp was turned off before the room was closed.",
"A glass bottle stood beside a metal tray."
]

SYSTEM="Follow the user's requested answer format exactly."

def chat(prompt):
    t=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    x=tok(t,return_tensors="pt",truncation=True,max_length=MAX_LENGTH)
    return {k:v.to(DEVICE) for k,v in x.items()}

def one_id(word):
    for s in (word," "+word,word.lower()," "+word.lower()):
        ids=tok(s,add_special_tokens=False).input_ids
        if len(ids)==1:return int(ids[0])
    return None

YES_ID,NO_ID=one_id("YES"),one_id("NO")
if YES_ID is None or NO_ID is None: raise RuntimeError("YES/NO must be single tokens for this compact TEST150.")
print("YES:",YES_ID,"NO:",NO_ID)

# =============================================================================
# 7. SEASC INJECTION + SCORING
# =============================================================================

def hooks(vectors,weights,theta,direction):
    hs=[]
    for L in range(N_LAYERS):
        w=float(weights[L])
        if abs(w)<EPS: continue
        A=unit(vectors[L]).to(DEVICE)
        dose=float(direction)*float(theta)*float(SEASC_DOSE[L])*w
        def hk(mod,args,out,A=A,dose=dose):
            h=out[0] if isinstance(out,tuple) else out
            hf=h.float(); n=hf.norm(dim=-1,keepdim=True)
            s=(hf+dose*n*A.view(1,1,-1)).to(h.dtype)
            return (s,)+out[1:] if isinstance(out,tuple) else s
        hs.append(LAYERS[L].register_forward_hook(hk))
    return hs

@torch.inference_mode()
def logits(prompt,v=None,w=None,theta=1,direction=1):
    x=chat(prompt); hs=[] if v is None else hooks(v,w,theta,direction)
    try:o=model(**x,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    q=int(x["attention_mask"][0].sum())-1
    return o.logits[0,q].float()

def scores(prompts,v=None,w=None,theta=1,direction=1):
    return np.array([float((lambda z:z[YES_ID]-z[NO_ID])(logits(p,v,w,theta,direction))) for p in prompts],dtype=np.float64)

def cstats(base,plus,minus):
    b,p,m=map(lambda x:float(np.mean(x)),(base,plus,minus))
    sd=float(np.std(base,ddof=1)) if len(base)>1 else 0.
    delta=max(DELTA_STD_FRAC*sd,DELTA_FLOOR)
    dp,dm=p-b,b-m
    mono=p>=b+delta and m<=b-delta
    anti=.5*(dp+dm)
    sym=float(np.clip(1-abs(dp-dm)/(abs(dp)+abs(dm)+EPS),0,1))
    return dict(base=b,plus=p,minus=m,baseline_std=sd,delta=delta,dplus=dp,dminus=dm,anti=anti,csym=sym,monotonic=bool(mono),causal=max(0.,anti)*sym if mono else 0.)

SCREEN_BASE=scores(SCREEN)
VAL_BASE=scores(VAL)

def scan(v,valid):
    M=np.zeros((len(THETAS),N_LAYERS))
    for ti,t in enumerate(THETAS):
        for L in range(N_LAYERS):
            if valid[L]==0: continue
            w=np.zeros(N_LAYERS); w[L]=1
            M[ti,L]=cstats(SCREEN_BASE,scores(SCREEN,v,w,t,+1),scores(SCREEN,v,w,t,-1))["causal"]
    totals=M.sum(1); bi=int(np.argmax(totals))
    return THETAS[bi],M[bi],M,totals

def envelope(signal,valid):
    s=np.asarray(signal)*valid
    if s.max()<=0:return np.zeros(N_LAYERS),0.
    tau=TAU_FRAC*s.max(); x=np.maximum(0,s-tau)
    sm=np.convolve(np.pad(x,(1,1),mode="edge"),SMOOTH_KERNEL,mode="valid")*valid
    if sm.max()>0:sm/=sm.max()
    return sm,float(tau)

def validate(v,w,t,base=VAL_BASE):
    return cstats(base,scores(VAL,v,w,t,+1),scores(VAL,v,w,t,-1))

# =============================================================================
# 8. SHUFFLE / RANDOM / NLL
# =============================================================================

def build_arm_from(Px,Nx,name):
    Dx=Px-Nx; out=[]
    for L in range(N_LAYERS):
        if name=="CONTROL_MEAN_ORTHO": a=unit(Dx[:,L].mean(0))
        elif name in ("DIAG_WHITENED_ORTHO","DIAG_INVARIANCE_ORTHO"):
            X=Dx[:,L]; mu=X.mean(0); var=X.var(0,unbiased=True)
            lam=DIAG_ALPHA*float(torch.median(var).clamp_min(EPS))
            a=unit(mu/torch.sqrt(var+lam))
            # shuffled labels destroy the original family mapping, so C's
            # invariance mask is fixed from CAL; the label signal itself is shuffled.
            if name=="DIAG_INVARIANCE_ORTHO": a=unit(a*invariance_mask(L))
        else:
            Hp,Hn=Px[:,L],Nx[:,L]; md=Hp.mean(0)-Hn.mean(0)
            R=torch.cat([Hp-Hp.mean(0,keepdim=True),Hn-Hn.mean(0,keepdim=True)],0)
            _,s,Vh=torch.linalg.svd(R,full_matrices=False)
            eig=s.square()/max(R.shape[0]-2,1)
            lam=FISHER_ALPHA*float(torch.median(eig).clamp_min(EPS))
            V=Vh.T; q=V.T@md
            a=unit(md/lam+V@((1/(eig+lam)-1/lam)*q))
        z,r=ortho(a,L)
        out.append(unit(a) if z is None else z)
    return out

def shuffled(name,k):
    pool=torch.cat([P,N],0); n=P.shape[0]
    rng=np.random.default_rng(stable_seed(name,"shuffle",k)); ix=rng.permutation(2*n)
    return build_arm_from(pool[ix[:n]],pool[ix[n:]],name)

def random_vectors(k):
    g=torch.Generator().manual_seed(stable_seed("random",k))
    return [unit(torch.randn(HIDDEN,generator=g)) for _ in range(N_LAYERS)]

@torch.inference_mode()
def nll(texts,v=None,w=None,t=1,direction=1):
    ls=[]
    for text in texts:
        x=tok(text,return_tensors="pt",truncation=True,max_length=MAX_LENGTH); x={k:z.to(DEVICE) for k,z in x.items()}
        hs=[] if v is None else hooks(v,w,t,direction)
        try:o=model(**x,use_cache=False,return_dict=True)
        finally:
            for h in hs:h.remove()
        lg=o.logits[:,:-1].float(); y=x["input_ids"][:,1:]
        ls.append(float(F.cross_entropy(lg.reshape(-1,lg.shape[-1]),y.reshape(-1))))
    return float(np.mean(ls))

BASE_NLL=nll(COLLATERAL)

# =============================================================================
# 9. FOUR-ARM TOURNAMENT
# =============================================================================

print("\n[3/9] Four-arm causal tournament")
RESULTS=[]

for name in ARM_NAMES:
    print("\n"+"-"*110)
    print(name)
    t,sig,M,tot=scan(ARMS[name],VALID_MASK[name])
    w,tau=envelope(sig,VALID_MASK[name])
    real=validate(ARMS[name],w,t)

    sh=np.array([validate(shuffled(name,k),w,t)["causal"] for k in range(K_SHUFFLE)])
    sm=float(sh.mean()); ss=float(sh.std(ddof=1)) if len(sh)>1 else 0.
    z=(real["causal"]-sm)/max(ss,1e-6)

    rnd=np.array([validate(random_vectors(stable_seed(name,k)),w,t)["causal"] for k in range(K_RANDOM)])
    rm=float(rnd.mean())

    pn=nll(COLLATERAL,ARMS[name],w,t,+1)
    mn=nll(COLLATERAL,ARMS[name],w,t,-1)
    collateral=.5*(max(0,pn-BASE_NLL)+max(0,mn-BASE_NLL))
    cvs=real["causal"]-ALPHA_NULL*max(sm,rm)-BETA_COLLATERAL*collateral

    row=dict(name=name,theta=t,signal=sig,matrix=M,theta_totals=tot,envelope=w,tau=tau,real=real,
             shuffle=sh,shuffle_mean=sm,shuffle_std=ss,random=rnd,random_mean=rm,
             base_nll=BASE_NLL,plus_nll=pn,minus_nll=mn,collateral=collateral,z=z,cvs=cvs)
    RESULTS.append(row)

    print("theta*       :",t)
    print("active layers:",np.where(w>1e-6)[0].tolist())
    print("causal       :",f"{real['causal']:+.8f}")
    print("C_sym        :",f"{real['csym']:.6f}")
    print("shuffle      :",f"{sm:+.8f} ± {ss:.8f}")
    print("random       :",f"{rm:+.8f}")
    print("ΔNLL         :",f"{collateral:+.8f}")
    print("Z            :",f"{z:+.6f}")
    print("CVS          :",f"{cvs:+.8f}")

# =============================================================================
# 10. COMPARATIVE DECISION — CONTROL IS FROZEN REFERENCE
# =============================================================================

control=next(x for x in RESULTS if x["name"]=="CONTROL_MEAN_ORTHO")
C0=control["real"]["causal"]; N0=control["collateral"]

for x in RESULTS:
    x["causal_ratio"]=x["real"]["causal"]/max(C0,EPS)
    x["nll_ratio"]=x["collateral"]/max(N0,EPS)
    x["efficiency"]=x["real"]["causal"]/max(x["collateral"],EPS)
    x["pareto_vs_control"]=bool(x["real"]["causal"]>=C0 and x["collateral"]<N0)

# Selection remains validation-only.
LOCKABLE=[x for x in RESULTS if x["real"]["monotonic"] and x["z"]>=Z_LOCK and np.any(x["envelope"]>1e-6)]

if LOCKABLE:
    # Prefer strict causal>=control + lower collateral; otherwise causal efficiency,
    # with causal strength and Z as deterministic tie-breakers.
    pareto=[x for x in LOCKABLE if x["pareto_vs_control"]]
    pool=pareto if pareto else LOCKABLE
    pool.sort(key=lambda x:(x["efficiency"],x["real"]["causal"],x["z"]),reverse=True)
    WIN=pool[0]; LOCK=True
else:
    WIN=None; LOCK=False

print("\n"+"="*110)
print("COMPARATIVE MATRIX")
print("="*110)
for x in RESULTS:
    print(f"{x['name']:28s} C={x['real']['causal']:+.6f} ΔNLL={x['collateral']:+.6f} "
          f"Rc={x['causal_ratio']:.3f} Rn={x['nll_ratio']:.3f} Z={x['z']:+.3f} "
          f"Pareto={x['pareto_vs_control']}")

print("\nLAYERWISE COSINE vs CONTROL")
for name in ARM_NAMES:
    cs=[cosine(ARMS["CONTROL_MEAN_ORTHO"][L],ARMS[name][L]) for L in range(N_LAYERS)]
    print(f"{name:28s}",np.round(cs,4).tolist())

# =============================================================================
# 11. UNTOUCHED FINAL
# =============================================================================

FINAL_RESULT=None
if LOCK:
    print("\n[4/9] LOCK:",WIN["name"])
    FINAL_BASE=scores(FINAL)
    FINAL_RESULT=cstats(
        FINAL_BASE,
        scores(FINAL,ARMS[WIN["name"]],WIN["envelope"],WIN["theta"],+1),
        scores(FINAL,ARMS[WIN["name"]],WIN["envelope"],WIN["theta"],-1)
    )
    print("FINAL BASE :",f"{FINAL_RESULT['base']:+.8f}")
    print("FINAL +A   :",f"{FINAL_RESULT['plus']:+.8f}")
    print("FINAL -A   :",f"{FINAL_RESULT['minus']:+.8f}")
    print("FINAL C    :",f"{FINAL_RESULT['causal']:+.8f}")
    print("FINAL Csym :",f"{FINAL_RESULT['csym']:.6f}")
    print("FINAL mono :",FINAL_RESULT["monotonic"])
else:
    print("\nNO_LOCK — FINAL NOT ACCESSED")

# =============================================================================
# 12. SAVE
# =============================================================================

def safe(x):
    if isinstance(x,dict): return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe(v) for v in x]
    if isinstance(x,np.ndarray): return safe(x.tolist())
    if isinstance(x,np.generic): return safe(x.item())
    if isinstance(x,float): return x if math.isfinite(x) else None
    return x

summary={
"experiment":"TEST 150 — AkbasCore Compass Foundry On-the-Go",
"model":MODEL_ID,"seed":SEED,"hidden":HIDDEN,
"motor":{"IVME":IVME,"SONUM":SONUM,"ZIRVE":ZIRVE,"TABAN":TABAN,"seasc_envelope":SEASC_ENV,"seasc_dose":SEASC_DOSE},
"arms":{},"lock":LOCK,"winner":WIN["name"] if LOCK else None,"final_accessed":LOCK,"final":FINAL_RESULT
}

for x in RESULTS:
    summary["arms"][x["name"]]={
        "theta":x["theta"],"theta_totals":x["theta_totals"],"signal":x["signal"],"envelope":x["envelope"],
        "real":x["real"],"shuffle_scores":x["shuffle"],"shuffle_mean":x["shuffle_mean"],"shuffle_std":x["shuffle_std"],
        "random_scores":x["random"],"random_mean":x["random_mean"],"collateral_delta_nll":x["collateral"],
        "z_causal":x["z"],"cvs":x["cvs"],"causal_ratio":x["causal_ratio"],"nll_ratio":x["nll_ratio"],
        "efficiency":x["efficiency"],"pareto_vs_control":x["pareto_vs_control"],
        "retention":RET[x["name"]],"valid_mask":VALID_MASK[x["name"]]
    }

JSON_PATH=f"{OUT}/TEST150_SUMMARY.json"
with open(JSON_PATH,"w",encoding="utf-8") as f: json.dump(safe(summary),f,ensure_ascii=False,indent=2,allow_nan=False)

NPZ_PATH=None
if LOCK:
    name=WIN["name"]
    A=np.stack([unit(v).numpy() for v in ARMS[name]]).astype(np.float32)
    assert np.allclose(np.linalg.norm(A,axis=1),1,atol=1e-5)
    NPZ_PATH=f"{OUT}/TEST150_COMPASS_LOCK.npz"
    np.savez_compressed(NPZ_PATH,compass=A,layer_envelope=WIN["envelope"].astype(np.float32),
                        layer_signal=WIN["signal"].astype(np.float32),seasc_envelope=SEASC_ENV.astype(np.float32),
                        seasc_dose=SEASC_DOSE.astype(np.float32),theta=np.array([WIN["theta"]],np.float32))

REPORT=f"{OUT}/TEST150_REPORT.txt"
with open(REPORT,"w",encoding="utf-8") as f:
    f.write("TEST 150 — AKBASCORE COMPASS FOUNDRY — ON-THE-GO\n"+"="*72+"\n")
    f.write(f"STATUS: {'LOCK' if LOCK else 'NO_LOCK'}\n\n")
    for x in RESULTS:
        f.write(f"{x['name']}\nCausal={x['real']['causal']:+.8f}  DeltaNLL={x['collateral']:+.8f}  "
                f"Rc={x['causal_ratio']:.4f}  Rn={x['nll_ratio']:.4f}  Z={x['z']:+.4f}  "
                f"CVS={x['cvs']:+.8f}  Pareto={x['pareto_vs_control']}\n\n")
    if LOCK:
        f.write(f"WINNER={WIN['name']}\nTHETA={WIN['theta']}\n")
        f.write(f"FINAL={FINAL_RESULT}\n")
    else:f.write("FINAL WAS NOT ACCESSED.\n")

ZIP=shutil.make_archive("/content/TEST150_COMPASS_FOUNDRY_COMPLETE","zip",OUT)

print("\n"+"="*110)
print("TEST 150 COMPLETE")
print("STATUS :", "LOCK" if LOCK else "NO_LOCK")
if LOCK: print("WINNER :",WIN["name"])
print("JSON   :",JSON_PATH)
print("REPORT :",REPORT)
if NPZ_PATH: print("NPZ    :",NPZ_PATH)
print("ZIP    :",ZIP)
print("="*110)

try:
    from google.colab import files
    files.download(ZIP)
except: pass
