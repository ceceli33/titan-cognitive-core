# ==================================================================================================
# TEST 238 — L19 MLP INTERNAL CARRIER X-RAY
# WORKING BASELINE: TEST237
# QUESTION: WHERE INSIDE L19 MLP DOES THE DISTRIBUTED OBJECT CARRIER TRANSFORM?
# --------------------------------------------------------------------------------------------------
# SAME MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / TEST230 OBJECT MAIN EFFECT
# SINGLE L08 PROMPT INJECTION | CONTINUATION " the" | ZERO CONTINUATION INTERVENTION
# X-RAY: MLP_INPUT -> GATE_PROJ / UP_PROJ -> ACT(GATE)*UP -> DOWN_PROJ -> BLOCK_OUT
# PRIMARY: LEAVE-ONE-CONTEXT-OUT OBJECT IDENTITY AT EACH INTERNAL STAGE
# SECONDARY: CROSS-CONTEXT INVARIANCE / CORRECT-WRONG GAP / SPECTRUM / ENERGY / GEOMETRY
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
DEVICE=torch.device("cuda");SEED=238
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

print("="*128)
print("TEST 238 — L19 MLP INTERNAL CARRIER X-RAY")
print("="*128)
print("WORKING BASELINE: TEST237 | INTERNAL MLP LOCALIZATION")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0)
          for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/30] Model...")
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

print("[2/30] Build TEST237 factorial set...")
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

print("[3/30] Shared-prefix token...")
CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj)
    CANDS.append(b if len(b)<=len(a) else a)
COMMON=CANDS[0][0]
if not all(x[0]==COMMON for x in CANDS):raise RuntimeError("No common first token.")
print(f"sharedPrefixToken={COMMON} text={tok.decode([COMMON])!r}")

print("[4/30] RoPE...")
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

print("[5/30] Capture source Q/K/V...")
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

print("[6/30] Reconstruct TEST222 RAW packets...")
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

print("[7/30] TEST230 object-main packets...")
GRAND=torch.stack(list(RAW.values())).mean(0)
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
for o in range(O):print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")

print("[8/30] Prompt prefill...")
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

print("[9/30] Capture L19 MLP internals...")
@torch.inference_mode()
def continuation(past):
    S={};hs=[]
    mlp=layers[19].mlp

    def block_in(m,args):
        S["BLOCK_IN"]=args[0][0,-1].float().detach().clone()

    def attn_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["ATTN"]=x[0,-1].float().detach().clone()

    def mlp_in(m,args):
        S["MLP_IN"]=args[0][0,-1].float().detach().clone()

    def gate_out(m,args,out):
        S["GATE"]=out[0,-1].float().detach().clone()

    def up_out(m,args,out):
        S["UP"]=out[0,-1].float().detach().clone()

    def down_in(m,args):
        S["PRODUCT"]=args[0][0,-1].float().detach().clone()

    def down_out(m,args,out):
        S["DOWN"]=out[0,-1].float().detach().clone()

    def block_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["BLOCK_OUT"]=x[0,-1].float().detach().clone()

    def l27_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["L27_OUT"]=x[0,-1].float().detach().clone()

    hs.append(layers[19].register_forward_pre_hook(block_in))
    hs.append(layers[19].self_attn.register_forward_hook(attn_out))
    hs.append(mlp.register_forward_pre_hook(mlp_in))
    hs.append(mlp.gate_proj.register_forward_hook(gate_out))
    hs.append(mlp.up_proj.register_forward_hook(up_out))
    hs.append(mlp.down_proj.register_forward_pre_hook(down_in))
    hs.append(mlp.down_proj.register_forward_hook(down_out))
    hs.append(layers[19].register_forward_hook(block_out))
    hs.append(layers[27].register_forward_hook(l27_out))

    try:
        r=model(
            input_ids=torch.tensor([[COMMON]],device=DEVICE),
            past_key_values=past,
            use_cache=False,
            return_dict=True
        )
    finally:remove(hs)

    S["POST_ATTN"]=S["BLOCK_IN"]+S["ATTN"]
    return S,r.logits[0,-1].float().detach().clone()

VAN={};INJ={}
for c in range(C):
    VAN[c],_=continuation(prefill(QENC[c],None))
    for o in range(O):
        INJ[(c,o)],_=continuation(prefill(QENC[c],OBJ[o]))
    print(f"C{c+1}: internal captures complete")

print("[10/30] Validate MLP reconstruction...")
errs=[]
for c in range(C):
    for o in range(O):
        S=INJ[(c,o)]
        prod=(model.model.layers[19].mlp.act_fn(S["GATE"].to(model.dtype)).float()
              *S["UP"])
        errs.append(float((prod-S["PRODUCT"]).norm()/S["PRODUCT"].norm().clamp_min(EPS)))
print(f"ACT(GATE)*UP reconstruction relative error={np.mean(errs)*100:.4f}%")

print("[11/30] Build centered intervention responses...")
STAGES=["BLOCK_IN","POST_ATTN","MLP_IN","GATE","UP","PRODUCT","DOWN","BLOCK_OUT","L27_OUT"]
D={};CENTER={}
for st in STAGES:
    for c in range(C):
        arr=[]
        for o in range(O):
            x=INJ[(c,o)][st]-VAN[c][st]
            D[(c,o,st)]=x
            arr.append(x)
        mean=torch.stack(arr).mean(0)
        for o in range(O):CENTER[(c,o,st)]=D[(c,o,st)]-mean

print("[12/30] Leave-one-context-out identity decoding...")
DEC={}
def decode_stage(st):
    hit=0;ranks=[];margins=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,st)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            q=CENTER[(hold,o,st)]
            sc=[cos(q,cent[j]) for j in range(O)]
            rr=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rr==1;ranks.append(rr)
            margins.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
    return hit/(C*O),float(np.mean(ranks)),float(np.mean(margins))

for st in STAGES:
    DEC[st]=decode_stage(st)
    x=DEC[st]
    print(f"{st:10s} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")

print("[13/30] Same-object cross-context invariance...")
INV={}
for st in STAGES:
    vals=[]
    for o in range(O):
        for a in range(C):
            for b in range(a+1,C):
                vals.append(cos(CENTER[(a,o,st)],CENTER[(b,o,st)]))
    INV[st]=float(np.mean(vals))
    print(f"{st:10s} invariance={INV[st]:+.4f}")

print("[14/30] Correct-vs-wrong separation...")
GAP={}
for st in STAGES:
    cor=[];wrong=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,st)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            q=CENTER[(hold,o,st)]
            cor.append(cos(q,cent[o]))
            wrong.extend(cos(q,cent[j]) for j in range(O) if j!=o)
    GAP[st]=float(np.mean(cor)-np.mean(wrong))
    print(f"{st:10s} gap={GAP[st]:+.4f}")

print("[15/30] Physical response magnitude...")
MAG={}
for st in STAGES:
    vals=[]
    for c in range(C):
        for o in range(O):
            vals.append(float(D[(c,o,st)].norm()/VAN[c][st].norm().clamp_min(EPS))*100)
    MAG[st]=float(np.mean(vals))
    print(f"{st:10s} displacement={MAG[st]:.4f}%")

print("[16/30] Stage-to-stage carrier cosine...")
PAIRS=[
    ("BLOCK_IN","POST_ATTN"),
    ("POST_ATTN","MLP_IN"),
    ("GATE","PRODUCT"),
    ("UP","PRODUCT"),
    ("MLP_IN","DOWN"),
    ("POST_ATTN","DOWN"),
    ("POST_ATTN","BLOCK_OUT"),
    ("DOWN","BLOCK_OUT"),
    ("BLOCK_OUT","L27_OUT")
]
for a,b in PAIRS:
    if CENTER[(0,0,a)].numel()!=CENTER[(0,0,b)].numel():
        print(f"{a}->{b} cosine=N/A dimensionality {CENTER[(0,0,a)].numel()}->{CENTER[(0,0,b)].numel()}")
        continue
    vals=[cos(CENTER[(c,o,a)],CENTER[(c,o,b)]) for c in range(C) for o in range(O)]
    print(f"{a}->{b} cosine={np.mean(vals):+.4f}")

print("[17/30] Internal identity transition deltas...")
for a,b in [
    ("BLOCK_IN","POST_ATTN"),
    ("POST_ATTN","MLP_IN"),
    ("MLP_IN","GATE"),
    ("GATE","PRODUCT"),
    ("UP","PRODUCT"),
    ("PRODUCT","DOWN"),
    ("DOWN","BLOCK_OUT"),
    ("BLOCK_OUT","L27_OUT")
]:
    print(f"{a}->{b} Δtop1={(DEC[b][0]-DEC[a][0])*100:+.1f}pp "
          f"Δmargin={DEC[b][2]-DEC[a][2]:+.4f}")

print("[18/30] SVD spectra by internal stage...")
SPEC={}
for st in STAGES:
    X=torch.stack([CENTER[(c,o,st)] for c in range(C) for o in range(O)]).float()
    _,s,_=torch.linalg.svd(X,full_matrices=False)
    e=s.square();e=e/e.sum().clamp_min(EPS)
    cum=torch.cumsum(e,0)
    p=e[e>0]
    eff=math.exp(float(-(p*torch.log(p)).sum()))
    pr=float(1.0/p.square().sum())
    r90=int(torch.where(cum>=.90)[0][0])+1
    r95=int(torch.where(cum>=.95)[0][0])+1
    SPEC[st]=(eff,pr,r90,r95,float(e[0]),float(cum[min(6,len(cum)-1)]))
    print(f"{st:10s} effRank={eff:6.2f} PR={pr:6.2f} r90={r90:2d} r95={r95:2d} "
          f"PC1={float(e[0])*100:5.1f}% PC1-7={float(cum[min(6,len(cum)-1)])*100:5.1f}%")

print("[19/30] Internal geometry preservation...")
def geometry(st):
    cent=[torch.stack([CENTER[(c,o,st)] for c in range(C)]).mean(0) for o in range(O)]
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

GEO={st:geometry(st) for st in STAGES}
tri=np.triu_indices(O,1)
for a,b in [
    ("BLOCK_IN","POST_ATTN"),
    ("POST_ATTN","MLP_IN"),
    ("GATE","PRODUCT"),
    ("UP","PRODUCT"),
    ("POST_ATTN","DOWN"),
    ("POST_ATTN","BLOCK_OUT"),
    ("BLOCK_OUT","L27_OUT")
]:
    sp=spear(GEO[a][tri],GEO[b][tri])
    rm=float(np.sqrt(np.mean((GEO[a][tri]-GEO[b][tri])**2)))
    print(f"{a}->{b} geometrySpearman={sp:+.4f} RMSE={rm:.4f}")

print("[20/30] Gate vs Up identity contribution...")
print(f"GATE    top1={DEC['GATE'][0]*100:.1f}% margin={DEC['GATE'][2]:+.4f}")
print(f"UP      top1={DEC['UP'][0]*100:.1f}% margin={DEC['UP'][2]:+.4f}")
print(f"PRODUCT top1={DEC['PRODUCT'][0]*100:.1f}% margin={DEC['PRODUCT'][2]:+.4f}")
print(f"DOWN    top1={DEC['DOWN'][0]*100:.1f}% margin={DEC['DOWN'][2]:+.4f}")

print("[21/30] Product identity relative to its parents...")
print(f"PRODUCT-GATE Δtop1={(DEC['PRODUCT'][0]-DEC['GATE'][0])*100:+.1f}pp")
print(f"PRODUCT-UP   Δtop1={(DEC['PRODUCT'][0]-DEC['UP'][0])*100:+.1f}pp")
print(f"DOWN-PRODUCT Δtop1={(DEC['DOWN'][0]-DEC['PRODUCT'][0])*100:+.1f}pp")
print(f"BLOCK-DOWN   Δtop1={(DEC['BLOCK_OUT'][0]-DEC['DOWN'][0])*100:+.1f}pp")

print("[22/30] Intermediate channel response concentration...")
for st in ["GATE","UP","PRODUCT"]:
    X=torch.stack([CENTER[(c,o,st)] for c in range(C) for o in range(O)]).float()
    channel_energy=X.square().mean(0)
    order=torch.argsort(channel_energy,descending=True)
    total=channel_energy.sum().clamp_min(EPS)
    cum=torch.cumsum(channel_energy[order],0)/total
    n50=int(torch.where(cum>=.50)[0][0])+1
    n80=int(torch.where(cum>=.80)[0][0])+1
    n90=int(torch.where(cum>=.90)[0][0])+1
    n95=int(torch.where(cum>=.95)[0][0])+1
    print(f"{st:7s} channels50={n50}/{I} channels80={n80}/{I} "
          f"channels90={n90}/{I} channels95={n95}/{I}")

print("[23/30] Held-out decoding using top-energy intermediate channels...")
CHANNEL_RESULTS={}
for st in ["GATE","UP","PRODUCT"]:
    for n in [32,64,128,256,512,1024]:
        hit=0;ranks=[]
        for hold in range(C):
            train=[c for c in range(C) if c!=hold]
            X=torch.stack([CENTER[(c,o,st)] for c in train for o in range(O)]).float()
            en=X.square().mean(0)
            idx=torch.argsort(en,descending=True)[:min(n,X.shape[1])]
            cent=[torch.stack([CENTER[(c,o,st)][idx] for c in train]).mean(0) for o in range(O)]
            for o in range(O):
                q=CENTER[(hold,o,st)][idx]
                sc=[cos(q,cent[j]) for j in range(O)]
                rr=np.argsort(sc)[::-1].tolist().index(o)+1
                hit+=rr==1;ranks.append(rr)
        CHANNEL_RESULTS[(st,n)]=(hit/(C*O),float(np.mean(ranks)))
        x=CHANNEL_RESULTS[(st,n)]
        print(f"{st:7s} topChannels={n:4d} top1={x[0]*100:5.1f}% rank={x[1]:.3f}")

print("[24/30] Complement after removing top-energy channels...")
for st in ["GATE","UP","PRODUCT"]:
    for n in [32,128,512,1024]:
        hit=0;ranks=[]
        for hold in range(C):
            train=[c for c in range(C) if c!=hold]
            X=torch.stack([CENTER[(c,o,st)] for c in train for o in range(O)]).float()
            en=X.square().mean(0)
            idx=torch.argsort(en,descending=True)
            keep=idx[min(n,len(idx)):]
            cent=[torch.stack([CENTER[(c,o,st)][keep] for c in train]).mean(0) for o in range(O)]
            for o in range(O):
                q=CENTER[(hold,o,st)][keep]
                sc=[cos(q,cent[j]) for j in range(O)]
                rr=np.argsort(sc)[::-1].tolist().index(o)+1
                hit+=rr==1;ranks.append(rr)
        print(f"{st:7s} removeTop={n:4d} top1={hit/(C*O)*100:5.1f}% rank={np.mean(ranks):.3f}")

print("[25/30] Product sign/sparsity telemetry...")
for label,source in [("VAN",VAN),("INJ",INJ)]:
    vals=[];zeros=[];positive=[]
    if label=="VAN":
        iterable=[source[c]["PRODUCT"] for c in range(C)]
    else:
        iterable=[source[(c,o)]["PRODUCT"] for c in range(C) for o in range(O)]
    for x in iterable:
        vals.append(float(x.norm()))
        zeros.append(float((x.abs()<1e-6).float().mean()))
        positive.append(float((x>0).float().mean()))
    print(f"{label} productNorm={np.mean(vals):.4f} nearZero={np.mean(zeros)*100:.2f}% positive={np.mean(positive)*100:.2f}%")

print("[26/30] Permutation null at PRODUCT and DOWN...")
rng=np.random.default_rng(SEED);PERMS=2000
for st in ["PRODUCT","DOWN"]:
    rows=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,st)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            q=CENTER[(hold,o,st)]
            rows.append([cos(q,cent[j]) for j in range(O)])
    arr=np.asarray(rows);labels=np.tile(np.arange(O),C)
    obs=DEC[st][0];null=[]
    for _ in range(PERMS):
        lab=rng.permutation(labels)
        null.append(float(np.mean(np.argmax(arr,axis=1)==lab)))
    mu=float(np.mean(null));sd=float(np.std(null)+1e-12)
    z=(obs-mu)/sd;p=(1+sum(x>=obs for x in null))/(PERMS+1)
    print(f"{st} obs={obs:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={p:.4f}")

print("[27/30] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")

print("[28/30] Localization metrics...")
pre=DEC["POST_ATTN"][0]
mlpin=DEC["MLP_IN"][0]
gate=DEC["GATE"][0]
up=DEC["UP"][0]
product=DEC["PRODUCT"][0]
down=DEC["DOWN"][0]
out=DEC["BLOCK_OUT"][0]

print(f"POST_ATTN={pre*100:.1f}%")
print(f"MLP_IN   ={mlpin*100:.1f}%")
print(f"GATE     ={gate*100:.1f}%")
print(f"UP       ={up*100:.1f}%")
print(f"PRODUCT  ={product*100:.1f}%")
print(f"DOWN     ={down*100:.1f}%")
print(f"BLOCK_OUT={out*100:.1f}%")

drops={
    "GATE_PROJECTION":gate-mlpin,
    "UP_PROJECTION":up-mlpin,
    "GATED_PRODUCT":product-max(gate,up),
    "DOWN_PROJECTION":down-product,
    "RESIDUAL_COMBINATION":out-down
}
for k,v in drops.items():print(f"{k:20s} Δ={v*100:+.1f}pp")

print("[29/30] Decision...")
largest=min(drops,key=drops.get)
largest_drop=drops[largest]
print(f"largestIdentityDrop={largest} {largest_drop*100:+.1f}pp")

print("[30/30] RESULTS")
print("="*128)
print("TEST 238 RESULTS")
print("="*128)
print("BASELINE: TEST237 | SINGLE L08 PROMPT INJECTION | CONTINUATION ' the'")
print("TARGET: INTERNAL LOCALIZATION OF L19 MLP DISTRIBUTED CARRIER TRANSFORMATION")
print(f"POST_ATTN={pre*100:.1f}% | MLP_IN={mlpin*100:.1f}% | GATE={gate*100:.1f}% | UP={up*100:.1f}%")
print(f"PRODUCT={product*100:.1f}% | DOWN={down*100:.1f}% | BLOCK_OUT={out*100:.1f}%")
print("-"*128)

if down<product-.15:
    print("RESULT: DOWN_PROJECTION_DOMINANT_IDENTITY_TRANSFORMATION")
elif product<min(gate,up)-.15:
    print("RESULT: GATED_PRODUCT_DOMINANT_IDENTITY_TRANSFORMATION")
elif min(gate,up)<mlpin-.15:
    print("RESULT: INPUT_PROJECTION_DOMINANT_IDENTITY_TRANSFORMATION")
elif out<down-.15:
    print("RESULT: RESIDUAL_COMBINATION_DOMINANT_IDENTITY_TRANSFORMATION")
else:
    print("RESULT: DISTRIBUTED_MLP_INTERNAL_TRANSFORMATION")

print("Observational localization only; no internal MLP component was causally ablated.")
print("No guard, transport, controller or continuation re-injection was used.")
print("="*128)
print("TEST 238 COMPLETE")
