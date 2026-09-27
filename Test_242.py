# ==================================================================================================
# TEST 242 — FULL 8x8 GATE×UP CROSSING MATRIX
# WORKING BASELINE: TEST241
# QUESTION: IS WRONG-PAIR ADVANTAGE SYSTEMATIC OR A CYCLIC-PERMUTATION ACCIDENT?
# --------------------------------------------------------------------------------------------------
# SAME TEST241 PIPELINE
# EXACT NATIVE BF16 PRODUCT: ACT(GATE_BF16) * UP_BF16
# FOR EACH TARGET OBJECT: ALL 8 GATE DONORS × ALL 8 UP DONORS
# PRIMARY: DIAGONAL vs OFF-DIAGONAL IDENTITY DECODING / MARGINS / DONOR EFFECTS
# NO NEW MODEL INTERVENTION | NO GUARD | NO TRANSPORT | NO CONTROLLER | WEIGHTS FROZEN
# ONE RUN — COMPACT OUTPUT
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=242
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
TOTAL=28;H_EXPECT=3584;KVH=0;EPS=1e-8;PRIMARY=.04
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=[
    "the amber compass","the silver lantern","the violet key","the bronze sphere",
    "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"
]
CONTEXTS=[
    ("Rovan Tesk","keeps"),("Mira Veln","carries"),
    ("Dalen Quor","owns"),("Sorin Kelm","guards")
]
C=len(CONTEXTS);O=len(OBJECTS)

print("="*116)
print("TEST 242 — FULL 8x8 GATE×UP CROSSING MATRIX")
print("="*116)
print("Baseline: TEST241 | exact native BF16 | compact 8x8 crossing audit")

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0)
          for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/12] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}
)
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers
H=model.config.hidden_size;NH=model.config.num_attention_heads
NKV=model.config.num_key_value_heads;HD=H//NH;GROUP=NH//NKV
I=layers[19].mlp.gate_proj.out_features;ACT=layers[19].mlp.act_fn
if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:
    raise RuntimeError("Architecture mismatch.")

FP_T=[
    layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,
    layers[19].mlp.gate_proj.weight,layers[19].mlp.up_proj.weight,
    layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,
    model.model.norm.weight,model.lm_head.weight
]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()

def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):
    na=a.norm();nb=b.norm()
    if float(na)<EPS or float(nb)<EPS:return 0.
    return float(torch.dot(a,b)/(na*nb))
def relerr(a,b):return float((a-b).norm()/b.norm().clamp_min(EPS))
def chat(x):
    return tok.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":x}],
        tokenize=False,add_generation_prompt=True
    )
def ids(x):return tok(x,add_special_tokens=False).input_ids
def subseq(hay,needle):
    return [list(range(i,i+len(needle))) for i in range(len(hay)-len(needle)+1)
            if hay[i:i+len(needle)]==needle] if needle else []
def last_span(full,text):
    a=subseq(full,ids(text))
    if a:return a[-1]
    a=subseq(full,ids(" "+text))
    return a[-1] if a else []
def fact_text(s,r,o):return f"Fact: {s} {r} {o}."
def qform(s,r):
    return f"What does {s} "+{"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r]+"?"
def remove(hs):
    for h in hs:h.remove()

print("[2/12] Factorial + source map...")
FMAP={};QENC={}
for c,(s,r) in enumerate(CONTEXTS):
    QENC[c]=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    for o,obj in enumerate(OBJECTS):
        e=tok(chat(fact_text(s,r,obj)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
        full=e.input_ids[0].tolist()
        ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,obj)
        if not ss or not rs or not os_:raise RuntimeError("Token map failed.")
        FMAP[(c,o)]=(e,ss,rs,os_)

CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj)
    CANDS.append(b if len(b)<=len(a) else a)
COMMON=CANDS[0][0]
if not all(x[0]==COMMON for x in CANDS):raise RuntimeError("Common prefix failed.")

print("[3/12] RoPE + source Q/K/V...")
rotary=model.model.rotary_emb
MAXSEQ=max(
    max(v[0].input_ids.shape[1] for v in FMAP.values()),
    max(v.input_ids.shape[1] for v in QENC.values())
)+4
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
    for name,mod in [
        ("Q",layers[8].self_attn.q_proj),
        ("K",layers[8].self_attn.k_proj),
        ("V",layers[8].self_attn.v_proj)
    ]:
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
    kh=qhead//GROUP
    q=rope(qh(S["Q"],qpos,qhead),qpos)
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
        for h in QGROUP:
            p[h]=max(x[0] for x in rows if x[1]==h)*v
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
    S={};hs=[];mlp=layers[19].mlp
    def gh(m,args,out):S["G"]=out[0,-1].detach().clone()
    def uh(m,args,out):S["U"]=out[0,-1].detach().clone()
    def ph(m,args):S["P"]=args[0][0,-1].detach().clone()
    hs=[
        mlp.gate_proj.register_forward_hook(gh),
        mlp.up_proj.register_forward_hook(uh),
        mlp.down_proj.register_forward_pre_hook(ph)
    ]
    try:
        model(
            input_ids=torch.tensor([[COMMON]],device=DEVICE),
            past_key_values=past,use_cache=False,return_dict=True
        )
    finally:remove(hs)
    return S

VAN={};INJ={}
for c in range(C):
    VAN[c]=continuation(prefill(QENC[c]))
    for o in range(O):
        INJ[(c,o)]=continuation(prefill(QENC[c],OBJ[o]))

print("[6/12] Native BF16 lock...")
errs=[]
for c in range(C):
    errs.append(relerr(
        (ACT(VAN[c]["G"])*VAN[c]["U"]).float(),
        VAN[c]["P"].float()
    ))
    for o in range(O):
        s=INJ[(c,o)]
        errs.append(relerr(
            (ACT(s["G"])*s["U"]).float(),
            s["P"].float()
        ))
MAXERR=max(errs)
if MAXERR>1e-7:raise RuntimeError("Native BF16 reconstruction failed.")
print(f"maxNativeProductError={MAXERR*100:.9f}%")

print("[7/12] Full 8x8 native-BF16 crossing...")
# CROSS[(context, gate_donor, up_donor)]
# Every branch is measured relative to that context's vanilla product.
CROSS={}
for c in range(C):
    p0=ACT(VAN[c]["G"])*VAN[c]["U"]
    for g in range(O):
        for u in range(O):
            CROSS[(c,g,u)]=(ACT(INJ[(c,g)]["G"])*INJ[(c,u)]["U"]-p0).float()

# Exact diagonal sanity against real captured PRODUCT delta.
diagerr=[]
for c in range(C):
    for o in range(O):
        real=INJ[(c,o)]["P"].float()-VAN[c]["P"].float()
        diagerr.append(relerr(CROSS[(c,o,o)],real))
print(f"diagonal-vs-realDelta error={np.mean(diagerr)*100:.6f}%")

print("[8/12] LOO decoder for every GATE×UP offset...")
# To compare pairing structure without printing 64 large blocks:
# offset d=0 => matched; d=1..7 => UP donor=(target+d)%8 while GATE remains target.
# Reverse family swaps the role: UP target, GATE donor=(target+d)%8.

def build_family(mode,d):
    X={}
    for c in range(C):
        for o in range(O):
            if mode=="UP":
                g=o;u=(o+d)%O
            elif mode=="GATE":
                g=(o+d)%O;u=o
            elif mode=="BOTH":
                g=(o+d)%O;u=(o+d)%O
            else:
                raise ValueError(mode)
            X[(c,o)]=CROSS[(c,g,u)]
    return X

def center(X):
    Z={}
    for c in range(C):
        m=torch.stack([X[(c,o)] for o in range(O)]).mean(0)
        for o in range(O):Z[(c,o)]=X[(c,o)]-m
    return Z

def decode(X):
    Z=center(X);hit=0;ranks=[];marg=[]
    for hold in range(C):
        tr=[c for c in range(C) if c!=hold]
        cent=[torch.stack([Z[(c,o)] for c in tr]).mean(0) for o in range(O)]
        for o in range(O):
            q=Z[(hold,o)]
            sc=[cos(q,cent[j]) for j in range(O)]
            rank=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rank==1;ranks.append(rank)
            marg.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
    return hit/(C*O),float(np.mean(ranks)),float(np.mean(marg))

UPRES=[];GRES=[];BRES=[]
for d in range(O):
    UPRES.append(decode(build_family("UP",d)))
    GRES.append(decode(build_family("GATE",d)))
    BRES.append(decode(build_family("BOTH",d)))

print("offset | MATCH/UP-cross        | GATE-cross            | BOTH-shift")
for d in range(O):
    a=UPRES[d];b=GRES[d];q=BRES[d]
    print(
        f"  {d}    | {a[0]*100:5.1f}% {a[2]:+.4f} "
        f"| {b[0]*100:5.1f}% {b[2]:+.4f} "
        f"| {q[0]*100:5.1f}% {q[2]:+.4f}"
    )

print("[9/12] All 56 off-diagonal target-preserving crossings...")
# For every target o, use every other donor j.
# UP-CROSS: GATE target, UP donor j.
# GATE-CROSS: GATE donor j, UP target.
# Decode each donor offset independently, then aggregate.
up_off=np.array([UPRES[d][0] for d in range(1,O)])
ga_off=np.array([GRES[d][0] for d in range(1,O)])
up_margin=np.array([UPRES[d][2] for d in range(1,O)])
ga_margin=np.array([GRES[d][2] for d in range(1,O)])

matched=UPRES[0]
print(f"MATCHED       top1={matched[0]*100:.1f}% margin={matched[2]:+.4f}")
print(f"UP-CROSS mean top1={up_off.mean()*100:.1f}% range={up_off.min()*100:.1f}-{up_off.max()*100:.1f}% margin={up_margin.mean():+.4f}")
print(f"GATE-X mean   top1={ga_off.mean()*100:.1f}% range={ga_off.min()*100:.1f}-{ga_off.max()*100:.1f}% margin={ga_margin.mean():+.4f}")
print(f"UP-cross > matched:   {int(np.sum(up_off>matched[0]))}/7")
print(f"GATE-cross > matched: {int(np.sum(ga_off>matched[0]))}/7")

print("[10/12] Pairwise relief matrix...")
# Matrix cell = mean held-out-context target-vs-wrong centroid margin for a
# fixed donor offset relation. Compact 8x8 descriptive geometry:
# rows = target GATE object, columns = UP donor object.
#
# Here each cell uses the four context responses and compares that response
# to the target object's matched LOO centroid. This is descriptive only.
MATCHED_Z=center({(c,o):CROSS[(c,o,o)] for c in range(C) for o in range(O)})
M=np.zeros((O,O),dtype=np.float64)

for target in range(O):
    for donor in range(O):
        vals=[]
        for hold in range(C):
            tr=[c for c in range(C) if c!=hold]
            cents=[
                torch.stack([MATCHED_Z[(c,j)] for c in tr]).mean(0)
                for j in range(O)
            ]
            # Center donor-cross responses within the held context using
            # the corresponding UP donor offset for every target label.
            X={}
            d=(donor-target)%O
            for o in range(O):
                X[o]=CROSS[(hold,o,(o+d)%O)]
            xm=torch.stack(list(X.values())).mean(0)
            q=X[target]-xm
            sc=[cos(q,cents[j]) for j in range(O)]
            vals.append(sc[target]-max(sc[j] for j in range(O) if j!=target))
        M[target,donor]=np.mean(vals)

print("UP donor columns O1..O8 | rows=target GATE object")
for i in range(O):
    print("O%d "%(i+1)+" ".join(f"{M[i,j]:+.3f}" for j in range(O)))

diag=float(np.mean(np.diag(M)))
off=float(np.mean(M[~np.eye(O,dtype=bool)]))
print(f"matrix mean diagonalMargin={diag:+.4f} offDiagonalMargin={off:+.4f}")

print("[11/12] Matched-vs-cross summary + weights...")
best_up=int(np.argmax([x[0] for x in UPRES[1:]]))+1
best_ga=int(np.argmax([x[0] for x in GRES[1:]]))+1
print(f"best UP-cross offset={best_up} top1={UPRES[best_up][0]*100:.1f}% margin={UPRES[best_up][2]:+.4f}")
print(f"best GATE-cross offset={best_ga} top1={GRES[best_ga][0]*100:.1f}% margin={GRES[best_ga][2]:+.4f}")

if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")

print("[12/12] RESULTS")
print("="*116)
print("TEST 242 RESULTS — FULL NATIVE-BF16 CROSSING")
print(f"MATCHED        : {matched[0]*100:.1f}% | rank {matched[1]:.3f} | margin {matched[2]:+.4f}")
print(f"UP-CROSS mean  : {up_off.mean()*100:.1f}% | mean margin {up_margin.mean():+.4f}")
print(f"GATE-CROSS mean: {ga_off.mean()*100:.1f}% | mean margin {ga_margin.mean():+.4f}")
print(f"UP better      : {int(np.sum(up_off>matched[0]))}/7")
print(f"GATE better    : {int(np.sum(ga_off>matched[0]))}/7")
print("-"*116)

n_better=int(np.sum(up_off>matched[0])+np.sum(ga_off>matched[0]))
mean_cross=float(np.mean(np.concatenate([up_off,ga_off])))

if n_better>=10 and mean_cross>matched[0]:
    print("RESULT: SYSTEMATIC_CROSS_OBJECT_RELIEF")
elif n_better>=7:
    print("RESULT: PARTIAL_CROSS_OBJECT_RELIEF")
elif n_better<=3 and matched[0]>mean_cross:
    print("RESULT: MATCHED_PAIR_ADVANTAGE")
else:
    print("RESULT: MIXED_PAIRING_EFFECT")

print("Exact native BF16 arithmetic preserved from TEST241.")
print("No crossed product was fed back into the model; crossing analysis is offline.")
print("="*116)
print("TEST 242 COMPLETE")
