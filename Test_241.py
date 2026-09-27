# ==================================================================================================
# TEST 241 — NATIVE-BF16 SWIGLU GATE×UP BINDING RETEST
# WORKING BASELINE: TEST240 — FIXED
# --------------------------------------------------------------------------------------------------
# TEST239 HYPOTHESIS RETESTED AFTER TEST240 NUMERICAL AUDIT
# ALL OFFLINE PRODUCTS USE THE EXACT MODEL-NATIVE BF16 PATH:
#       ACT(GATE_BF16) * UP_BF16
#
# SAME MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / TEST230 OBJECT MAIN
# SINGLE L08 PROMPT INJECTION | CONTINUATION " the"
# NO GUARD | NO TRANSPORT | NO CONTROLLER | NO RE-INJECTION | WEIGHTS FROZEN
#
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
DEVICE=torch.device("cuda");SEED=241
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
PERM=np.array([1,2,3,4,5,6,7,0],dtype=int)

print("="*116)
print("TEST 241 — NATIVE-BF16 SWIGLU GATE×UP BINDING RETEST")
print("="*116)
print("Baseline: TEST240 | TEST239 retest | exact native BF16 product path")

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

print("[3/12] RoPE + Q/K/V source capture...")
rotary=model.model.rotary_emb
MAXSEQ=max(max(v[0].input_ids.shape[1] for v in FMAP.values()),
           max(v.input_ids.shape[1] for v in QENC.values()))+4
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
    for name,mod in [("Q",layers[8].self_attn.q_proj),
                     ("K",layers[8].self_attn.k_proj),
                     ("V",layers[8].self_attn.v_proj)]:
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

print("[5/12] Prompt injection + exact L19 capture...")
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
    hs+=[
        mlp.gate_proj.register_forward_hook(gh),
        mlp.up_proj.register_forward_hook(uh),
        mlp.down_proj.register_forward_pre_hook(ph)
    ]
    try:
        model(input_ids=torch.tensor([[COMMON]],device=DEVICE),
              past_key_values=past,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S

VAN={};INJ={}
for c in range(C):
    VAN[c]=continuation(prefill(QENC[c]))
    for o in range(O):INJ[(c,o)]=continuation(prefill(QENC[c],OBJ[o]))

print("[6/12] Native BF16 reconstruction lock...")
errs=[]
for c in range(C):
    errs.append(relerr((ACT(VAN[c]["G"])*VAN[c]["U"]).float(),VAN[c]["P"].float()))
    for o in range(O):
        s=INJ[(c,o)]
        errs.append(relerr((ACT(s["G"])*s["U"]).float(),s["P"].float()))
MAXERR=max(errs)
print(f"maxNativeBF16ProductError={MAXERR*100:.9f}%")
if MAXERR>1e-7:raise RuntimeError("Native BF16 reconstruction lock failed.")

print("[7/12] Build BF16 matched/crossed products...")
BR=["MATCHED","WRONG_UP","WRONG_GATE","WRONG_BOTH"]
D={}

for c in range(C):
    g0=VAN[c]["G"];u0=VAN[c]["U"]
    p0=ACT(g0)*u0

    for o in range(O):
        w=int(PERM[o])
        g=INJ[(c,o)]["G"];u=INJ[(c,o)]["U"]
        gw=INJ[(c,w)]["G"];uw=INJ[(c,w)]["U"]

        D[(c,o,"MATCHED")]=(ACT(g)*u-p0).float()
        D[(c,o,"WRONG_UP")]=(ACT(g)*uw-p0).float()
        D[(c,o,"WRONG_GATE")]=(ACT(gw)*u-p0).float()
        D[(c,o,"WRONG_BOTH")]=(ACT(gw)*uw-p0).float()

# Real PRODUCT delta must now equal MATCHED exactly.
realerr=[]
for c in range(C):
    for o in range(O):
        real=INJ[(c,o)]["P"].float()-VAN[c]["P"].float()
        realerr.append(relerr(D[(c,o,"MATCHED")],real))
print(f"MATCHED-vs-realProductDelta error={np.mean(realerr)*100:.9f}%")

print("[8/12] Exact BF16 main/interaction decomposition...")
PART={}
for c in range(C):
    g0=VAN[c]["G"];u0=VAN[c]["U"];p0=ACT(g0)*u0
    for o in range(O):
        g=INJ[(c,o)]["G"];u=INJ[(c,o)]["U"]
        full=ACT(g)*u-p0
        gate=ACT(g)*u0-p0
        up=ACT(g0)*u-p0
        interaction=full-gate-up
        PART[(c,o,"GATE_ONLY")]=gate.float()
        PART[(c,o,"UP_ONLY")]=up.float()
        PART[(c,o,"INTERACTION")]=interaction.float()

def centered(src,names):
    z={}
    for name in names:
        for c in range(C):
            m=torch.stack([src[(c,o,name)] for o in range(O)]).mean(0)
            for o in range(O):z[(c,o,name)]=src[(c,o,name)]-m
    return z

Z=centered(D,BR)
PZ=centered(PART,["GATE_ONLY","UP_ONLY","INTERACTION"])

def decode(fetch):
    hit=0;ranks=[];marg=[]
    for hold in range(C):
        tr=[c for c in range(C) if c!=hold]
        cent=[torch.stack([fetch(c,o) for c in tr]).mean(0) for o in range(O)]
        for o in range(O):
            q=fetch(hold,o)
            sc=[cos(q,cent[j]) for j in range(O)]
            rank=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rank==1;ranks.append(rank)
            marg.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
    return hit/(C*O),float(np.mean(ranks)),float(np.mean(marg))

print("[9/12] Identity results...")
RES={}
for name in BR:
    RES[name]=decode(lambda c,o,n=name:Z[(c,o,n)])
for name in ["GATE_ONLY","UP_ONLY","INTERACTION"]:
    RES[name]=decode(lambda c,o,n=name:PZ[(c,o,n)])

for name in ["MATCHED","WRONG_UP","WRONG_GATE","WRONG_BOTH",
             "GATE_ONLY","UP_ONLY","INTERACTION"]:
    a=RES[name]
    print(f"{name:12s} top1={a[0]*100:5.1f}% rank={a[1]:.3f} margin={a[2]:+.4f}")

print("[10/12] Interaction energy + pairing diagnostics...")
rat=[]
for c in range(C):
    for o in range(O):
        rat.append(float(
            PZ[(c,o,"INTERACTION")].norm()/
            Z[(c,o,"MATCHED")].norm().clamp_min(EPS)
        ))
print(f"interaction/matched centered norm={np.mean(rat)*100:.3f}%")

for name in ["WRONG_UP","WRONG_GATE","WRONG_BOTH"]:
    vals=[cos(Z[(c,o,"MATCHED")],Z[(c,o,name)]) for c in range(C) for o in range(O)]
    print(f"MATCHED->{name:10s} meanCos={np.mean(vals):+.4f}")

print("[11/12] Compact permutation validation...")
rng=np.random.default_rng(SEED);NPERM=1000
for name,source in [
    ("MATCHED",Z),("WRONG_UP",Z),("WRONG_GATE",Z),
    ("GATE_ONLY",PZ),("UP_ONLY",PZ),("INTERACTION",PZ)
]:
    rows=[]
    for hold in range(C):
        tr=[c for c in range(C) if c!=hold]
        cent=[torch.stack([source[(c,o,name)] for c in tr]).mean(0) for o in range(O)]
        for o in range(O):
            q=source[(hold,o,name)]
            rows.append([cos(q,cent[j]) for j in range(O)])

    arr=np.asarray(rows)
    obs=float(np.mean(np.argmax(arr,axis=1)==np.tile(np.arange(O),C)))
    null=[]
    for _ in range(NPERM):
        # Same object-label permutation applied within every context.
        perm=rng.permutation(O)
        labels=np.concatenate([perm for _ in range(C)])
        null.append(float(np.mean(np.argmax(arr,axis=1)==labels)))
    mu=float(np.mean(null));sd=float(np.std(null)+1e-12)
    p=(1+sum(v>=obs for v in null))/(NPERM+1)
    print(f"{name:12s} obs={obs:.4f} null={mu:.4f}±{sd:.4f} p={p:.4f}")

if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")

print("[12/12] RESULTS")
print("="*116)
print("TEST 241 RESULTS — NATIVE BF16 RETEST")
print(f"Native product max error : {MAXERR*100:.9f}%")
print(f"Matched-real delta error : {np.mean(realerr)*100:.9f}%")
print("-"*116)
for name in ["MATCHED","WRONG_UP","WRONG_GATE","WRONG_BOTH",
             "GATE_ONLY","UP_ONLY","INTERACTION"]:
    a=RES[name]
    print(f"{name:12s}: {a[0]*100:5.1f}% | rank {a[1]:.3f} | margin {a[2]:+.4f}")
print(f"Interaction norm fraction: {np.mean(rat)*100:.3f}%")
print("Weights: PASS")
print("-"*116)

M=RES["MATCHED"][0];WU=RES["WRONG_UP"][0];WG=RES["WRONG_GATE"][0]
GI=RES["INTERACTION"][0]

if M>=WU+.15 and M>=WG+.15 and GI>=.50:
    print("RESULT: MATCHED_GATE_UP_MULTIPLICATIVE_BINDING_SUPPORTED")
elif GI>=.50 and GI>max(RES["GATE_ONLY"][0],RES["UP_ONLY"][0]):
    print("RESULT: MULTIPLICATIVE_INTERACTION_IDENTITY_ENRICHED")
elif M>WU+.10 or M>WG+.10:
    print("RESULT: PARTIAL_GATE_UP_PAIRING_DEPENDENCE")
else:
    print("RESULT: NO_STRONG_MATCHED_GATE_UP_BINDING_EFFECT")

print("TEST239 dtype artefact removed; all product branches use native BF16 arithmetic.")
print("="*116)
print("TEST 241 COMPLETE")
