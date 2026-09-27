# ==================================================================================================
# TEST 239 — L19 SWIGLU GATE×UP BINDING X-RAY
# WORKING BASELINE: TEST238
# QUESTION: DOES OBJECT IDENTITY DEPEND ON MATCHED GATE×UP MULTIPLICATIVE BINDING?
# --------------------------------------------------------------------------------------------------
# SAME MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / TEST230 OBJECT MAIN EFFECT
# SINGLE L08 PROMPT INJECTION | CONTINUATION " the" | ZERO CONTINUATION INTERVENTION
# OBSERVATIONAL REAL FORWARD: MLP_IN / GATE / UP / PRODUCT / DOWN / BLOCK_OUT
# COUNTERFACTUAL MEASUREMENT ONLY:
#   MATCHED_PRODUCT = ACT(GATE_RESPONSE) × UP_RESPONSE
#   WRONG_UP        = ACT(GATE_RESPONSE[o]) × UP_RESPONSE[wrong_object]
#   WRONG_GATE      = ACT(GATE_RESPONSE[wrong_object]) × UP_RESPONSE[o]
#   SHUFFLED_PAIR   = deterministic object permutation
# NO COUNTERFACTUAL PRODUCT IS FED BACK INTO THE MODEL
# NO GUARD | NO TRANSPORT | NO CONTROLLER | NO RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=239
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
TOTAL=28;H_EXPECT=3584;KVH=0;EPS=1e-8;PRIMARY=.04
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."

OBJECTS=[
    "the amber compass","the silver lantern","the violet key","the bronze sphere",
    "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"
]
CONTEXTS=[
    ("Rovan Tesk","keeps"),
    ("Mira Veln","carries"),
    ("Dalen Quor","owns"),
    ("Sorin Kelm","guards")
]
C=len(CONTEXTS);O=len(OBJECTS)
PERM=np.array([1,2,3,4,5,6,7,0],dtype=int)

print("="*128)
print("TEST 239 — L19 SWIGLU GATE×UP BINDING X-RAY")
print("="*128)
print("WORKING BASELINE: TEST238 | OBSERVATIONAL + OFFLINE COUNTERFACTUAL GEOMETRY")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0)
          for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/31] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token

model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"":0},
    attn_implementation="sdpa",
    **{DT:torch.bfloat16}
)
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers
H=model.config.hidden_size
NH=model.config.num_attention_heads
NKV=model.config.num_key_value_heads
HD=H//NH
GROUP=NH//NKV
I=layers[19].mlp.gate_proj.out_features
ACT=model.model.layers[19].mlp.act_fn

if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:
    raise RuntimeError("Architecture mismatch.")

print(f"hidden={H} intermediate={I} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")

FP_T=[
    layers[0].self_attn.q_proj.weight,
    layers[8].self_attn.o_proj.weight,
    layers[19].mlp.gate_proj.weight,
    layers[19].mlp.up_proj.weight,
    layers[19].mlp.down_proj.weight,
    layers[27].mlp.down_proj.weight,
    model.model.norm.weight,
    model.lm_head.weight
]

@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()

def unit(x):return x/x.norm().clamp_min(EPS)

def cos(a,b):
    na=a.norm();nb=b.norm()
    if float(na)<EPS or float(nb)<EPS:return 0.
    return float(torch.dot(a,b)/(na*nb))

def chat(x):
    return tok.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":x}],
        tokenize=False,add_generation_prompt=True
    )

def ids(x):return tok(x,add_special_tokens=False).input_ids

def subseq(hay,needle):
    if not needle:return []
    return [list(range(i,i+len(needle))) for i in range(len(hay)-len(needle)+1)
            if hay[i:i+len(needle)]==needle]

def last_span(full,text):
    a=subseq(full,ids(text))
    if a:return a[-1]
    a=subseq(full,ids(" "+text))
    return a[-1] if a else []

def fact_text(s,r,o):return f"Fact: {s} {r} {o}."

def qform(s,r):
    return f"What does {s} "+{
        "keeps":"keep","carries":"carry","owns":"own","guards":"guard"
    }[r]+"?"

def remove(hs):
    for h in hs:h.remove()

print("[2/31] Build TEST238 factorial set...")
FMAP={};QENC={}
for c,(s,r) in enumerate(CONTEXTS):
    qe=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    QENC[c]=qe
    for o,obj in enumerate(OBJECTS):
        fi=tok(chat(fact_text(s,r,obj)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
        full=fi.input_ids[0].tolist()
        ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,obj)
        if not ss or not rs or not os_:raise RuntimeError(f"Token map fail C{c+1} O{o+1}")
        if obj.lower() in qform(s,r).lower():raise RuntimeError("Target leakage.")
        FMAP[(c,o)]=(fi,ss,rs,os_)
    print(f"C{c+1} {s} {r} | blindSlot={qe.input_ids.shape[1]-1}")

print("[3/31] Shared-prefix token...")
CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj)
    CANDS.append(b if len(b)<=len(a) else a)
COMMON=CANDS[0][0]
if not all(x[0]==COMMON for x in CANDS):raise RuntimeError("No common first token.")
print(f"sharedPrefixToken={COMMON} text={tok.decode([COMMON])!r}")

print("[4/31] RoPE...")
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

print("[5/31] Capture source Q/K/V...")
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

SRC={}
for c in range(C):
    for o in range(O):SRC[(c,o)]=capture_source(FMAP[(c,o)][0])
    print(f"C{c+1}: source captures complete")

print("[6/31] Reconstruct TEST222 RAW packets...")
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
        S=SRC[(c,o)]
        oe=FMAP[(c,o)][3][-1]
        rows=[]
        for h in QGROUP:
            for qp in range(oe,FMAP[(c,o)][0].input_ids.shape[1]):
                a=attn_row(S,qp,h)
                rows.append((float(a[oe]),h,qp))
        v=kvh(S["V"],oe,KVH)
        p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32)
        for h in QGROUP:
            w=max(x[0] for x in rows if x[1]==h)
            p[h]=w*v
        RAW[(c,o)]=layers[8].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"C{c+1}: RAW packets complete")

print("[7/31] TEST230 object-main packets...")
GRAND=torch.stack(list(RAW.values())).mean(0)
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
for o in range(O):print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")

print("[8/31] Prompt prefill...")
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
    if packet is not None and calls!=1:raise RuntimeError(f"Injection calls={calls}")
    return r.past_key_values

print("[9/31] Capture L19 SwiGLU internals...")
@torch.inference_mode()
def continuation(past):
    S={};hs=[];mlp=layers[19].mlp

    def mlp_in(m,args):S["MLP_IN"]=args[0][0,-1].float().detach().clone()
    def gate_out(m,args,out):S["GATE"]=out[0,-1].float().detach().clone()
    def up_out(m,args,out):S["UP"]=out[0,-1].float().detach().clone()
    def product_in(m,args):S["PRODUCT"]=args[0][0,-1].float().detach().clone()
    def down_out(m,args,out):S["DOWN"]=out[0,-1].float().detach().clone()
    def block_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["BLOCK_OUT"]=x[0,-1].float().detach().clone()

    hs.append(mlp.register_forward_pre_hook(mlp_in))
    hs.append(mlp.gate_proj.register_forward_hook(gate_out))
    hs.append(mlp.up_proj.register_forward_hook(up_out))
    hs.append(mlp.down_proj.register_forward_pre_hook(product_in))
    hs.append(mlp.down_proj.register_forward_hook(down_out))
    hs.append(layers[19].register_forward_hook(block_out))
    try:
        model(
            input_ids=torch.tensor([[COMMON]],device=DEVICE),
            past_key_values=past,use_cache=False,return_dict=True
        )
    finally:remove(hs)
    return S

VAN={};INJ={}
for c in range(C):
    VAN[c]=continuation(prefill(QENC[c],None))
    for o in range(O):INJ[(c,o)]=continuation(prefill(QENC[c],OBJ[o]))
    print(f"C{c+1}: captures complete")

print("[10/31] Validate real PRODUCT...")
errs=[]
for c in range(C):
    for o in range(O):
        s=INJ[(c,o)]
        p=ACT(s["GATE"].to(model.dtype)).float()*s["UP"]
        errs.append(float((p-s["PRODUCT"]).norm()/s["PRODUCT"].norm().clamp_min(EPS)))
print(f"ACT(GATE)*UP reconstruction relative error={np.mean(errs)*100:.4f}%")

print("[11/31] Build real centered responses...")
REAL={};CENTER={}
for st in ["MLP_IN","GATE","UP","PRODUCT","DOWN","BLOCK_OUT"]:
    for c in range(C):
        arr=[]
        for o in range(O):
            x=INJ[(c,o)][st]-VAN[c][st]
            REAL[(c,o,st)]=x;arr.append(x)
        m=torch.stack(arr).mean(0)
        for o in range(O):CENTER[(c,o,st)]=REAL[(c,o,st)]-m

print("[12/31] Construct exact offline matched/crossed products...")
CF={}
for c in range(C):
    vg=VAN[c]["GATE"]
    vu=VAN[c]["UP"]
    vp=ACT(vg.to(model.dtype)).float()*vu

    for o in range(O):
        g=INJ[(c,o)]["GATE"]
        u=INJ[(c,o)]["UP"]
        w=int(PERM[o])
        gw=INJ[(c,w)]["GATE"]
        uw=INJ[(c,w)]["UP"]

        CF[(c,o,"MATCHED_ABS")]=ACT(g.to(model.dtype)).float()*u
        CF[(c,o,"WRONG_UP_ABS")]=ACT(g.to(model.dtype)).float()*uw
        CF[(c,o,"WRONG_GATE_ABS")]=ACT(gw.to(model.dtype)).float()*u
        CF[(c,o,"WRONG_BOTH_ABS")]=ACT(gw.to(model.dtype)).float()*uw

        # Branch deltas against the exact same vanilla product.
        CF[(c,o,"MATCHED")]=CF[(c,o,"MATCHED_ABS")]-vp
        CF[(c,o,"WRONG_UP")]=CF[(c,o,"WRONG_UP_ABS")]-vp
        CF[(c,o,"WRONG_GATE")]=CF[(c,o,"WRONG_GATE_ABS")]-vp
        CF[(c,o,"WRONG_BOTH")]=CF[(c,o,"WRONG_BOTH_ABS")]-vp

print("[13/31] Matched-product consistency...")
errs=[]
for c in range(C):
    for o in range(O):
        a=CF[(c,o,"MATCHED")]
        b=REAL[(c,o,"PRODUCT")]
        errs.append(float((a-b).norm()/b.norm().clamp_min(EPS)))
print(f"offlineMatched vs realProductDelta relative error={np.mean(errs)*100:.4f}%")

print("[14/31] Center counterfactual branches...")
CFC={}
BRANCHES=["MATCHED","WRONG_UP","WRONG_GATE","WRONG_BOTH"]
for br in BRANCHES:
    for c in range(C):
        m=torch.stack([CF[(c,o,br)] for o in range(O)]).mean(0)
        for o in range(O):CFC[(c,o,br)]=CF[(c,o,br)]-m

def decode(fetch):
    hit=0;ranks=[];margins=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([fetch(c,o) for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            q=fetch(hold,o)
            sc=[cos(q,cent[j]) for j in range(O)]
            rr=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rr==1;ranks.append(rr)
            margins.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
    return hit/(C*O),float(np.mean(ranks)),float(np.mean(margins))

print("[15/31] Real branch identity baseline...")
for st in ["MLP_IN","GATE","UP","PRODUCT","DOWN","BLOCK_OUT"]:
    x=decode(lambda c,o,st=st:CENTER[(c,o,st)])
    print(f"{st:10s} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")

print("[16/31] Counterfactual product identity...")
DEC={}
for br in BRANCHES:
    DEC[br]=decode(lambda c,o,br=br:CFC[(c,o,br)])
    x=DEC[br]
    print(f"{br:10s} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")

print("[17/31] Correct-label vs donor-label decoding...")
def donor_decode(br,donor_from_perm):
    correct=0;donor=0;other=0
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CFC[(c,o,br)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            q=CFC[(hold,o,br)]
            sc=[cos(q,cent[j]) for j in range(O)]
            pred=int(np.argmax(sc))
            d=int(PERM[o]) if donor_from_perm else o
            if pred==o:correct+=1
            elif pred==d:donor+=1
            else:other+=1
    return correct/(C*O),donor/(C*O),other/(C*O)

for br in ["WRONG_UP","WRONG_GATE"]:
    a,b,d=donor_decode(br,True)
    print(f"{br:10s} targetLabel={a*100:5.1f}% donorLabel={b*100:5.1f}% other={d*100:5.1f}%")

print("[18/31] Pairwise matched-vs-crossed cosine...")
for br in ["WRONG_UP","WRONG_GATE","WRONG_BOTH"]:
    vals=[cos(CFC[(c,o,"MATCHED")],CFC[(c,o,br)]) for c in range(C) for o in range(O)]
    print(f"MATCHED->{br:10s} cosine={np.mean(vals):+.4f}")

print("[19/31] Counterfactual response norms...")
for br in BRANCHES:
    vals=[float(CF[(c,o,br)].norm()) for c in range(C) for o in range(O)]
    print(f"{br:10s} meanDeltaNorm={np.mean(vals):.4f}")

print("[20/31] Norm-matched crossed-product decoding...")
NM={}
for br in ["WRONG_UP","WRONG_GATE","WRONG_BOTH"]:
    for c in range(C):
        for o in range(O):
            x=CFC[(c,o,br)]
            ref=CFC[(c,o,"MATCHED")].norm()
            NM[(c,o,br)]=unit(x)*ref

for br in ["WRONG_UP","WRONG_GATE","WRONG_BOTH"]:
    x=decode(lambda c,o,br=br:NM[(c,o,br)])
    print(f"{br:10s} normMatched top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")

print("[21/31] Same-object cross-context invariance...")
for br in BRANCHES:
    vals=[]
    for o in range(O):
        for a in range(C):
            for b in range(a+1,C):
                vals.append(cos(CFC[(a,o,br)],CFC[(b,o,br)]))
    print(f"{br:10s} invariance={np.mean(vals):+.4f}")

print("[22/31] Correct-vs-wrong separation...")
for br in BRANCHES:
    cor=[];wrong=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CFC[(c,o,br)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            q=CFC[(hold,o,br)]
            cor.append(cos(q,cent[o]))
            wrong.extend(cos(q,cent[j]) for j in range(O) if j!=o)
    print(f"{br:10s} gap={np.mean(cor)-np.mean(wrong):+.4f}")

print("[23/31] Geometry preservation...")
def geom(fetch):
    cent=[torch.stack([fetch(c,o) for c in range(C)]).mean(0) for o in range(O)]
    G=np.zeros((O,O))
    for i in range(O):
        for j in range(O):G[i,j]=cos(cent[i],cent[j])
    return G

def rankdata(x):
    order=np.argsort(x);r=np.empty(len(x),dtype=float);r[order]=np.arange(len(x))
    return r

def spear(a,b):
    a=rankdata(np.asarray(a));b=rankdata(np.asarray(b))
    if np.std(a)<EPS or np.std(b)<EPS:return 0.
    return float(np.corrcoef(a,b)[0,1])

GM=geom(lambda c,o:CFC[(c,o,"MATCHED")])
tri=np.triu_indices(O,1)
for br in ["WRONG_UP","WRONG_GATE","WRONG_BOTH"]:
    G=geom(lambda c,o,br=br:CFC[(c,o,br)])
    sp=spear(GM[tri],G[tri])
    rm=float(np.sqrt(np.mean((GM[tri]-G[tri])**2)))
    print(f"MATCHED->{br:10s} geometrySpearman={sp:+.4f} RMSE={rm:.4f}")

print("[24/31] Pure branch-response interaction decomposition...")
# ΔP = P(g0+Δg,u0+Δu)-P(g0,u0)
# Gate-only = P(g0+Δg,u0)-P0
# Up-only   = P(g0,u0+Δu)-P0
# Interaction = ΔP-GateOnly-UpOnly
PART={}
for c in range(C):
    g0=VAN[c]["GATE"];u0=VAN[c]["UP"]
    p0=ACT(g0.to(model.dtype)).float()*u0
    for o in range(O):
        g=INJ[(c,o)]["GATE"];u=INJ[(c,o)]["UP"]
        gate_only=ACT(g.to(model.dtype)).float()*u0-p0
        up_only=ACT(g0.to(model.dtype)).float()*u-p0
        matched=CF[(c,o,"MATCHED")]
        interaction=matched-gate_only-up_only
        PART[(c,o,"GATE_ONLY")]=gate_only
        PART[(c,o,"UP_ONLY")]=up_only
        PART[(c,o,"INTERACTION")]=interaction

PARTC={}
for br in ["GATE_ONLY","UP_ONLY","INTERACTION"]:
    for c in range(C):
        m=torch.stack([PART[(c,o,br)] for o in range(O)]).mean(0)
        for o in range(O):PARTC[(c,o,br)]=PART[(c,o,br)]-m

for br in ["GATE_ONLY","UP_ONLY","INTERACTION"]:
    x=decode(lambda c,o,br=br:PARTC[(c,o,br)])
    print(f"{br:12s} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")

print("[25/31] Interaction energy...")
for br in ["GATE_ONLY","UP_ONLY","INTERACTION"]:
    vals=[float(PARTC[(c,o,br)].norm()) for c in range(C) for o in range(O)]
    print(f"{br:12s} meanCenteredNorm={np.mean(vals):.4f}")

rat=[]
for c in range(C):
    for o in range(O):
        rat.append(float(
            PARTC[(c,o,"INTERACTION")].norm()/
            CFC[(c,o,"MATCHED")].norm().clamp_min(EPS)
        ))
print(f"interaction/matched centered norm ratio={np.mean(rat):.4f}")

print("[26/31] Interaction geometry...")
GI=geom(lambda c,o:PARTC[(c,o,"INTERACTION")])
sp=spear(GM[tri],GI[tri])
rm=float(np.sqrt(np.mean((GM[tri]-GI[tri])**2)))
print(f"MATCHED->INTERACTION geometrySpearman={sp:+.4f} RMSE={rm:.4f}")

print("[27/31] Channel-wise multiplicative interaction concentration...")
X=torch.stack([
    PARTC[(c,o,"INTERACTION")]
    for c in range(C) for o in range(O)
]).float()
en=X.square().mean(0)
order=torch.argsort(en,descending=True)
cum=torch.cumsum(en[order],0)/en.sum().clamp_min(EPS)
for q in [.50,.80,.90,.95]:
    n=int(torch.where(cum>=q)[0][0])+1
    print(f"interactionEnergy{int(q*100)}={n}/{I} channels")

print("[28/31] Permutation null...")
rng=np.random.default_rng(SEED);PERMS=2000
for br in ["MATCHED","WRONG_UP","WRONG_GATE","INTERACTION"]:
    if br=="INTERACTION":
        fetch=lambda c,o:PARTC[(c,o,"INTERACTION")]
        obs=decode(fetch)[0]
    else:
        fetch=lambda c,o,br=br:CFC[(c,o,br)]
        obs=DEC[br][0]

    rows=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([fetch(c,o) for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            q=fetch(hold,o)
            rows.append([cos(q,cent[j]) for j in range(O)])

    arr=np.asarray(rows);labels=np.tile(np.arange(O),C);null=[]
    for _ in range(PERMS):
        lab=rng.permutation(labels)
        null.append(float(np.mean(np.argmax(arr,axis=1)==lab)))
    mu=float(np.mean(null));sd=float(np.std(null)+1e-12)
    z=(obs-mu)/sd;p=(1+sum(x>=obs for x in null))/(PERMS+1)
    print(f"{br:12s} obs={obs:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={p:.4f}")

print("[29/31] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")

print("[30/31] Decision metrics...")
matched=DEC["MATCHED"][0]
wrong_up=DEC["WRONG_UP"][0]
wrong_gate=DEC["WRONG_GATE"][0]
wrong_both=DEC["WRONG_BOTH"][0]
gate_only=decode(lambda c,o:PARTC[(c,o,"GATE_ONLY")])[0]
up_only=decode(lambda c,o:PARTC[(c,o,"UP_ONLY")])[0]
interaction=decode(lambda c,o:PARTC[(c,o,"INTERACTION")])[0]

print(f"MATCHED={matched*100:.1f}%")
print(f"WRONG_UP={wrong_up*100:.1f}%")
print(f"WRONG_GATE={wrong_gate*100:.1f}%")
print(f"WRONG_BOTH={wrong_both*100:.1f}%")
print(f"GATE_ONLY={gate_only*100:.1f}%")
print(f"UP_ONLY={up_only*100:.1f}%")
print(f"INTERACTION={interaction*100:.1f}%")

print("[31/31] RESULTS")
print("="*128)
print("TEST 239 RESULTS")
print("="*128)
print("BASELINE: TEST238 | SINGLE L08 PROMPT INJECTION | CONTINUATION ' the'")
print("TARGET: L19 SWIGLU MATCHED GATE×UP IDENTITY BINDING")
print(f"MATCHED={matched*100:.1f}% | WRONG_UP={wrong_up*100:.1f}% | WRONG_GATE={wrong_gate*100:.1f}% | WRONG_BOTH={wrong_both*100:.1f}%")
print(f"GATE_ONLY={gate_only*100:.1f}% | UP_ONLY={up_only*100:.1f}% | INTERACTION={interaction*100:.1f}%")
print("-"*128)

if matched>=wrong_up+.15 and matched>=wrong_gate+.15 and interaction>=.50:
    print("RESULT: MATCHED_GATE_UP_MULTIPLICATIVE_BINDING_SUPPORTED")
elif interaction>=.50 and interaction>max(gate_only,up_only):
    print("RESULT: MULTIPLICATIVE_INTERACTION_IDENTITY_ENRICHED")
elif matched>wrong_up+.10 or matched>wrong_gate+.10:
    print("RESULT: PARTIAL_GATE_UP_PAIRING_DEPENDENCE")
else:
    print("RESULT: NO_STRONG_MATCHED_GATE_UP_BINDING_EFFECT")

print("Counterfactual crossed products were measured offline only and were never injected into the model.")
print("No causal intervention inside L19 MLP is claimed.")
print("No guard, transport, controller or continuation re-injection was used.")
print("="*128)
print("TEST 239 COMPLETE")
