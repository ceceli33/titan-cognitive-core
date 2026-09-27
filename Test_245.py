# ==================================================================================================
# TEST 245 — HELD-OUT CONTEXT TRAIN-ONLY GATE/UP SHARED-SUBSPACE X-RAY
# WORKING BASELINE: TEST244
# FIX: REMOVE PER-SAMPLE BISECTOR ARTIFACT
# NATIVE BF16 | TRAIN-ONLY SVD SHARED SUBSPACE | HELD-OUT CONTEXT TEST | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=245
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
TOTAL=28;H_EXPECT=3584;KVH=0;EPS=1e-8;PRIMARY=.04
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere","the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
C=len(CONTEXTS);O=len(OBJECTS)

print("="*116)
print("TEST 245 — HELD-OUT CONTEXT TRAIN-ONLY GATE/UP SHARED-SUBSPACE X-RAY")
print("="*116)
print("Baseline: TEST244 | native BF16 | train-only shared subspace | held-out context")

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/12] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers
H=model.config.hidden_size;NH=model.config.num_attention_heads;NKV=model.config.num_key_value_heads
HD=H//NH;GROUP=NH//NKV;ACT=layers[19].mlp.act_fn
if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:raise RuntimeError("Architecture mismatch.")

FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.gate_proj.weight,layers[19].mlp.up_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()

def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):
    na=a.norm();nb=b.norm()
    if float(na)<EPS or float(nb)<EPS:return 0.
    return float(torch.dot(a,b)/(na*nb))
def relerr(a,b):return float((a-b).norm()/b.norm().clamp_min(EPS))
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def ids(x):return tok(x,add_special_tokens=False).input_ids
def subseq(hay,needle):return [list(range(i,i+len(needle))) for i in range(len(hay)-len(needle)+1) if hay[i:i+len(needle)]==needle] if needle else []
def last_span(full,text):
    a=subseq(full,ids(text))
    if a:return a[-1]
    a=subseq(full,ids(" "+text))
    return a[-1] if a else []
def fact_text(s,r,o):return f"Fact: {s} {r} {o}."
def qform(s,r):return f"What does {s} "+{"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r]+"?"
def remove(hs):
    for h in hs:h.remove()

print("[2/12] Factorial + source map...")
FMAP={};QENC={}
for c,(s,r) in enumerate(CONTEXTS):
    QENC[c]=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    for o,obj in enumerate(OBJECTS):
        e=tok(chat(fact_text(s,r,obj)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
        full=e.input_ids[0].tolist();ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,obj)
        if not ss or not rs or not os_:raise RuntimeError("Token map failed.")
        FMAP[(c,o)]=(e,ss,rs,os_)

CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj);CANDS.append(b if len(b)<=len(a) else a)
COMMON=CANDS[0][0]
if not all(x[0]==COMMON for x in CANDS):raise RuntimeError("Common prefix failed.")

print("[3/12] RoPE + source Q/K/V...")
rotary=model.model.rotary_emb
MAXSEQ=max(max(v[0].input_ids.shape[1] for v in FMAP.values()),max(v.input_ids.shape[1] for v in QENC.values()))+4
dummy=torch.zeros(1,MAXSEQ,H,device=DEVICE,dtype=model.dtype)
pos=torch.arange(MAXSEQ,device=DEVICE).unsqueeze(0)
with torch.inference_mode():COS,SIN=rotary(dummy,pos)
COS=COS[0].float();SIN=SIN[0].float()

def rotate_half(x):
    n=x.shape[-1]//2
    return torch.cat((-x[...,n:],x[...,:n]),dim=-1)
def rope(x,p):return x*COS[p]+rotate_half(x)*SIN[p]

@torch.inference_mode()
def capture_source(e):
    S={};hs=[]
    for name,mod in [("Q",layers[8].self_attn.q_proj),("K",layers[8].self_attn.k_proj),("V",layers[8].self_attn.v_proj)]:
        def mk(n):
            def hk(m,args,out):S[n]=out[0].float().detach().clone()
            return hk
        hs.append(mod.register_forward_hook(mk(name)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S

SRC={(c,o):capture_source(FMAP[(c,o)][0]) for c in range(C) for o in range(O)}

print("[4/12] TEST222 forge + TEST230 object-main...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    kh=qhead//GROUP;q=rope(qh(S["Q"],qpos,qhead),qpos)
    K=torch.stack([rope(kvh(S["K"],p,kh),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)

RAW={}
for c in range(C):
    for o in range(O):
        S=SRC[(c,o)];oe=FMAP[(c,o)][3][-1];rows=[]
        for h in QGROUP:
            for qp in range(oe,FMAP[(c,o)][0].input_ids.shape[1]):
                a=attn_row(S,qp,h);rows.append((float(a[oe]),h))
        v=kvh(S["V"],oe,KVH)
        p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32)
        for h in QGROUP:p[h]=max(x[0] for x in rows if x[1]==h)*v
        RAW[(c,o)]=layers[8].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()

GRAND=torch.stack(list(RAW.values())).mean(0)
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
print("ObjectMain norms:"," ".join(f"{OBJ[o].norm():.4f}" for o in range(O)))

print("[5/12] Prompt injection + L19 capture...")
@torch.inference_mode()
def prefill(e,packet=None):
    calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float()
        d=unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY
        y[:,-1,:]=(z+d).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    h=layers[8].register_forward_hook(inject) if packet is not None else None
    try:r=model(**e,use_cache=True,return_dict=True)
    finally:
        if h is not None:h.remove()
    if packet is not None and calls!=1:raise RuntimeError("Injection count mismatch.")
    return r.past_key_values

@torch.inference_mode()
def continuation(past):
    S={};mlp=layers[19].mlp
    def gh(m,args,out):S["G"]=out[0,-1].detach().clone()
    def uh(m,args,out):S["U"]=out[0,-1].detach().clone()
    def ph(m,args):S["P"]=args[0][0,-1].detach().clone()
    hs=[mlp.gate_proj.register_forward_hook(gh),mlp.up_proj.register_forward_hook(uh),mlp.down_proj.register_forward_pre_hook(ph)]
    try:model(input_ids=torch.tensor([[COMMON]],device=DEVICE),past_key_values=past,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S

VAN={};INJ={}
for c in range(C):
    VAN[c]=continuation(prefill(QENC[c]))
    for o in range(O):INJ[(c,o)]=continuation(prefill(QENC[c],OBJ[o]))

print("[6/12] Native BF16 lock...")
errs=[]
for c in range(C):
    errs.append(relerr((ACT(VAN[c]["G"])*VAN[c]["U"]).float(),VAN[c]["P"].float()))
    for o in range(O):
        s=INJ[(c,o)];errs.append(relerr((ACT(s["G"])*s["U"]).float(),s["P"].float()))
MAXERR=max(errs)
print(f"maxNativeProductError={MAXERR*100:.9f}%")
if MAXERR>1e-7:raise RuntimeError("Native BF16 reconstruction failed.")

print("[7/12] Build centered GATE/UP carriers...")
D={}
for c in range(C):
    g0=VAN[c]["G"];u0=VAN[c]["U"];p0=ACT(g0)*u0
    for o in range(O):
        g=INJ[(c,o)]["G"];u=INJ[(c,o)]["U"]
        D[(c,o,"GATE")]=(ACT(g)*u0-p0).float()
        D[(c,o,"UP")]=(ACT(g0)*u-p0).float()
        D[(c,o,"MATCHED")]=(ACT(g)*u-p0).float()

Z={}
for name in ["GATE","UP","MATCHED"]:
    for c in range(C):
        m=torch.stack([D[(c,o,name)] for o in range(O)]).mean(0)
        for o in range(O):Z[(c,o,name)]=D[(c,o,name)]-m

print("[8/12] Train-only shared subspaces...")
# Shared subspace for each held-out context:
# train rows = normalized GATE+UP paired identity directions from the other 3 contexts.
# SVD is fitted on train contexts only. Rank is selected by >=90% train energy, capped at 14.
SUB={};RANK={}
for hold in range(C):
    tr=[c for c in range(C) if c!=hold]
    rows=[]
    for c in tr:
        for o in range(O):
            g=unit(Z[(c,o,"GATE")]);u=unit(Z[(c,o,"UP")])
            rows.append(unit(g+u))
    X=torch.stack(rows)
    _,S,Vh=torch.linalg.svd(X,full_matrices=False)
    e=S.square();cum=torch.cumsum(e,0)/e.sum().clamp_min(EPS)
    idx=torch.where(cum>=.90)[0]
    r=int(idx[0].item()+1) if len(idx) else int(len(S))
    r=max(1,min(r,14))
    SUB[hold]=Vh[:r].T.contiguous()
    RANK[hold]=r
print("trainOnlyRanks:"," ".join(f"C{c+1}={RANK[c]}" for c in range(C)))

def proj(x,B):return B@(B.T@x)

print("[9/12] Held-out shared/private decoding...")
PART={}
for hold in range(C):
    B=SUB[hold]
    for c in range(C):
        for o in range(O):
            for name in ["GATE","UP"]:
                x=Z[(c,o,name)]
                s=proj(x,B);p=x-s
                PART[(hold,c,o,name+"_SHARED")]=s
                PART[(hold,c,o,name+"_PRIVATE")]=p
            PART[(hold,c,o,"SHARED_SUM")]=PART[(hold,c,o,"GATE_SHARED")]+PART[(hold,c,o,"UP_SHARED")]
            PART[(hold,c,o,"PRIVATE_SUM")]=PART[(hold,c,o,"GATE_PRIVATE")]+PART[(hold,c,o,"UP_PRIVATE")]
            PART[(hold,c,o,"MAIN")]=Z[(c,o,"GATE")]+Z[(c,o,"UP")]

def decode_part(name):
    hit=0;ranks=[];marg=[]
    for hold in range(C):
        tr=[c for c in range(C) if c!=hold]
        cent=[torch.stack([PART[(hold,c,o,name)] for c in tr]).mean(0) for o in range(O)]
        for o in range(O):
            q=PART[(hold,hold,o,name)]
            sc=[cos(q,cent[j]) for j in range(O)]
            rank=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rank==1;ranks.append(rank)
            marg.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
    return hit/(C*O),float(np.mean(ranks)),float(np.mean(marg))

PNAMES=["GATE_SHARED","UP_SHARED","SHARED_SUM","GATE_PRIVATE","UP_PRIVATE","PRIVATE_SUM","MAIN"]
RES={n:decode_part(n) for n in PNAMES}
for n in PNAMES:
    a=RES[n];print(f"{n:14s} top1={a[0]*100:5.1f}% rank={a[1]:.3f} margin={a[2]:+.4f}")

print("[10/12] Held-out geometry...")
GU=[];GSH=[];USH=[];GPR=[];UPR=[];PCOS=[]
for hold in range(C):
    for o in range(O):
        g=Z[(hold,o,"GATE")];u=Z[(hold,o,"UP")]
        gs=PART[(hold,hold,o,"GATE_SHARED")];us=PART[(hold,hold,o,"UP_SHARED")]
        gp=PART[(hold,hold,o,"GATE_PRIVATE")];up=PART[(hold,hold,o,"UP_PRIVATE")]
        GU.append(cos(g,u))
        GSH.append(float(gs.norm()/g.norm().clamp_min(EPS)));USH.append(float(us.norm()/u.norm().clamp_min(EPS)))
        GPR.append(float(gp.norm()/g.norm().clamp_min(EPS)));UPR.append(float(up.norm()/u.norm().clamp_min(EPS)))
        PCOS.append(cos(gp,up))
print(f"cos(GATE,UP)={np.mean(GU):+.4f}")
print(f"GATE shared/private norm={np.mean(GSH)*100:.2f}%/{np.mean(GPR)*100:.2f}%")
print(f"UP   shared/private norm={np.mean(USH)*100:.2f}%/{np.mean(UPR)*100:.2f}%")
print(f"cos(GATE_PRIVATE,UP_PRIVATE)={np.mean(PCOS):+.4f}")

print("[11/12] Matched-vs-crossed fixed-subspace audit...")
MR=[];CR=[];MS=[];MP=[];CS=[];CP=[]
for hold in range(C):
    B=SUB[hold]
    for o in range(O):
        j=(o+1)%O
        g=Z[(hold,o,"GATE")];u=Z[(hold,o,"UP")];uw=Z[(hold,j,"UP")]
        m=g+u;c=g+uw
        ms=proj(m,B);mp=m-ms;cs=proj(c,B);cp=c-cs
        MS.append(float(ms.norm()));MP.append(float(mp.norm()))
        CS.append(float(cs.norm()));CP.append(float(cp.norm()))
        MR.append(float(mp.norm()/ms.norm().clamp_min(EPS)))
        CR.append(float(cp.norm()/cs.norm().clamp_min(EPS)))
print(f"MATCH shared={np.mean(MS):.4f} private={np.mean(MP):.4f} private/shared={np.mean(MR):.4f}")
print(f"CROSS shared={np.mean(CS):.4f} private={np.mean(CP):.4f} private/shared={np.mean(CR):.4f}")
print(f"CROSS-MATCH ratio Δ={(np.mean(CR)-np.mean(MR))/(np.mean(MR)+EPS)*100:+.2f}%")

if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")

print("[12/12] RESULTS")
print("="*116)
print("TEST 245 RESULTS — TRAIN-ONLY SHARED SUBSPACE")
print("Ranks: "+" ".join(str(RANK[c]) for c in range(C)))
for n in PNAMES:
    a=RES[n];print(f"{n:14s}: {a[0]*100:5.1f}% | rank {a[1]:.3f} | margin {a[2]:+.4f}")
print("-"*116)
print(f"Private cosine={np.mean(PCOS):+.4f}")
print(f"MATCH private/shared={np.mean(MR):.4f}")
print(f"CROSS private/shared={np.mean(CR):.4f}")
print(f"Cross ratio change={(np.mean(CR)-np.mean(MR))/(np.mean(MR)+EPS)*100:+.2f}%")
print("Weights: PASS")
if np.mean(PCOS)<-.95:
    print("RESULT: PRIVATE_ANTAGONISM_PERSISTS_UNDER_TRAIN_ONLY_SUBSPACE")
elif np.mean(CR)>np.mean(MR)*1.20:
    print("RESULT: CROSS_PAIRING_INCREASES_HELD_OUT_PRIVATE_GEOMETRY")
elif RES["SHARED_SUM"][0]>=RES["PRIVATE_SUM"][0]+.10:
    print("RESULT: IDENTITY_DOMINATED_BY_TRAIN_ONLY_SHARED_SUBSPACE")
elif RES["PRIVATE_SUM"][0]>=RES["SHARED_SUM"][0]+.10:
    print("RESULT: IDENTITY_DOMINATED_BY_HELD_OUT_PRIVATE_COMPLEMENT")
else:
    print("RESULT: DISTRIBUTED_TRAIN_ONLY_SHARED_PRIVATE_GEOMETRY")
print("Per-sample bisector removed; subspace fitted only on training contexts.")
print("Offline geometry only; no counterfactual tensor injected.")
print("="*116)
print("TEST 245 COMPLETE")



