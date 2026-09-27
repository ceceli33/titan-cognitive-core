# ==================================================================================================
# TEST 237 — L19 MLP RECODING SUBSPACE SPECTRUM
# WORKING BASELINE: TEST236
# QUESTION: HOW MANY INDEPENDENT DIMENSIONS CARRY THE L19 MLP RECODED OBJECT SIGNAL?
# --------------------------------------------------------------------------------------------------
# SAME MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / TEST230 OBJECT MAIN EFFECT
# SINGLE L08 PROMPT INJECTION | CONTINUATION " the" | ZERO CONTINUATION INTERVENTION
# PRIMARY: L19 POST_ATTN -> L19 OUT RECODING-DELTA SVD
# HELD-OUT CONTEXT ONLY FOR DECODING / RANK SELECTION
# TEST: TOP SUBSPACE / COMPLEMENT / CUMULATIVE RANK / LEAVE-ONE-COMPONENT-OUT
# NO GUARD | NO TRANSPORT MAP | NO CONTROLLER | NO RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=237
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
RANKS=[1,2,3,4,5,6,7]

print("="*128)
print("TEST 237 — L19 MLP RECODING SUBSPACE SPECTRUM")
print("="*128)
print("WORKING BASELINE: TEST236 | OBSERVATIONAL SUBSPACE FOLLOW-UP")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0)
          for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/29] Model...")
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

if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:
    raise RuntimeError("Architecture mismatch.")

print(f"hidden={H} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")

FP_T=[
    layers[0].self_attn.q_proj.weight,
    layers[8].self_attn.o_proj.weight,
    layers[18].mlp.down_proj.weight,
    layers[19].mlp.down_proj.weight,
    layers[20].mlp.down_proj.weight,
    layers[27].mlp.down_proj.weight,
    model.model.norm.weight,
    model.lm_head.weight
]

@torch.inference_mode()
def fp():
    return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)

FP0=fp()

def unit(x):
    return x/x.norm().clamp_min(EPS)

def cos(a,b):
    na=a.norm();nb=b.norm()
    if float(na)<EPS or float(nb)<EPS:return 0.
    return float(torch.dot(a,b)/(na*nb))

def chat(x):
    return tok.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":x}],
        tokenize=False,
        add_generation_prompt=True
    )

def ids(x):
    return tok(x,add_special_tokens=False).input_ids

def subseq(hay,needle):
    if not needle:return []
    return [
        list(range(i,i+len(needle)))
        for i in range(len(hay)-len(needle)+1)
        if hay[i:i+len(needle)]==needle
    ]

def last_span(full,text):
    a=subseq(full,ids(text))
    if a:return a[-1]
    a=subseq(full,ids(" "+text))
    return a[-1] if a else []

def fact_text(s,r,o):
    return f"Fact: {s} {r} {o}."

def qform(s,r):
    return f"What does {s} "+{
        "keeps":"keep",
        "carries":"carry",
        "owns":"own",
        "guards":"guard"
    }[r]+"?"

def remove(hs):
    for h in hs:h.remove()

print("[2/29] Build TEST236 factorial set...")
FMAP={};QENC={}

for c,(s,r) in enumerate(CONTEXTS):
    qe=tok(
        chat(qform(s,r)),
        return_tensors="pt",
        add_special_tokens=False
    ).to(DEVICE)
    QENC[c]=qe

    for o,obj in enumerate(OBJECTS):
        fi=tok(
            chat(fact_text(s,r,obj)),
            return_tensors="pt",
            add_special_tokens=False
        ).to(DEVICE)

        full=fi.input_ids[0].tolist()
        ss=last_span(full,s)
        rs=last_span(full,r)
        os_=last_span(full,obj)

        if not ss or not rs or not os_:
            raise RuntimeError(f"Token map fail C{c+1} O{o+1}")

        if obj.lower() in qform(s,r).lower():
            raise RuntimeError("Target leakage.")

        FMAP[(c,o)]=(fi,ss,rs,os_)

    print(f"C{c+1} {s} {r} | blindSlot={qe.input_ids.shape[1]-1}")

print("[3/29] Shared-prefix tokens...")
CANDS=[]

for obj in OBJECTS:
    a=ids(obj)
    b=ids(" "+obj)
    CANDS.append(b if len(b)<=len(a) else a)

COMMON=CANDS[0][0]
DISC=[x[1] for x in CANDS]

if not all(x[0]==COMMON for x in CANDS):
    raise RuntimeError("No common first token.")

if len(set(DISC))!=O:
    raise RuntimeError("Discriminative tokens not unique.")

print(f"sharedPrefixToken={COMMON} text={tok.decode([COMMON])!r}")

print("[4/29] RoPE...")
rotary=model.model.rotary_emb

MAXSEQ=max(
    max(v[0].input_ids.shape[1] for v in FMAP.values()),
    max(v.input_ids.shape[1] for v in QENC.values())
)+4

dummy=torch.zeros(
    1,MAXSEQ,H,
    device=DEVICE,
    dtype=model.dtype
)

pos=torch.arange(MAXSEQ,device=DEVICE).unsqueeze(0)

with torch.inference_mode():
    COS,SIN=rotary(dummy,pos)

COS=COS[0].float()
SIN=SIN[0].float()

def rotate_half(x):
    n=x.shape[-1]//2
    return torch.cat((-x[...,n:],x[...,:n]),dim=-1)

def rope(x,p):
    return x*COS[p]+rotate_half(x)*SIN[p]

print("[5/29] Capture source Q/K/V...")

@torch.inference_mode()
def capture_source(e):
    S={};hs=[]

    for name,mod in [
        ("Q",layers[8].self_attn.q_proj),
        ("K",layers[8].self_attn.k_proj),
        ("V",layers[8].self_attn.v_proj)
    ]:
        def mk(n):
            def hk(m,args,out):
                S[n]=out[0].float().detach().clone()
            return hk
        hs.append(mod.register_forward_hook(mk(name)))

    try:
        model(**e,use_cache=False,return_dict=True)
    finally:
        remove(hs)

    return S

SRC={}

for c in range(C):
    for o in range(O):
        SRC[(c,o)]=capture_source(FMAP[(c,o)][0])
    print(f"C{c+1}: source captures complete")

print("[6/29] Reconstruct TEST222 RAW packets...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))

def qh(x,p,h):
    return x[p].reshape(NH,HD)[h]

def kvh(x,p,h):
    return x[p].reshape(NKV,HD)[h]

def attn_row(S,qpos,qhead):
    kh=qhead//GROUP
    q=rope(qh(S["Q"],qpos,qhead),qpos)

    K=torch.stack([
        rope(kvh(S["K"],p,kh),p)
        for p in range(qpos+1)
    ])

    return torch.softmax(
        (K@q)/math.sqrt(HD),
        dim=-1
    )

RAW={}

for c in range(C):
    for o in range(O):
        S=SRC[(c,o)]
        oe=FMAP[(c,o)][3][-1]
        rows=[]

        for h in QGROUP:
            for qp in range(
                oe,
                FMAP[(c,o)][0].input_ids.shape[1]
            ):
                a=attn_row(S,qp,h)
                rows.append((float(a[oe]),h,qp))

        v=kvh(S["V"],oe,KVH)

        p=torch.zeros(
            NH,HD,
            device=DEVICE,
            dtype=torch.float32
        )

        for h in QGROUP:
            w=max(x[0] for x in rows if x[1]==h)
            p[h]=w*v

        RAW[(c,o)]=layers[8].self_attn.o_proj(
            p.reshape(H).to(model.dtype)
        ).float()

    print(f"C{c+1}: RAW packets complete")

print("[7/29] TEST230 object-main packets...")
GRAND=torch.stack(list(RAW.values())).mean(0)

OMEAN={
    o:torch.stack([
        RAW[(c,o)]
        for c in range(C)
    ]).mean(0)
    for o in range(O)
}

OBJ={
    o:OMEAN[o]-GRAND
    for o in range(O)
}

for o in range(O):
    print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")

print("[8/29] Prompt prefill...")

@torch.inference_mode()
def prefill(e,packet=None):
    calls=0

    def inject(m,args,out):
        nonlocal calls

        x=out[0] if isinstance(out,tuple) else out

        if x.ndim!=3 or x.shape[1]<=1:
            return None

        y=x.clone()
        z=y[:,-1,:].float()

        d=unit(packet)*z.norm(
            dim=-1,
            keepdim=True
        )*PRIMARY

        y[:,-1,:]=(z+d).to(y.dtype)
        calls+=1

        return (y,)+out[1:] if isinstance(out,tuple) else y

    h=layers[8].register_forward_hook(inject) if packet is not None else None

    try:
        r=model(
            **e,
            use_cache=True,
            return_dict=True
        )
    finally:
        if h is not None:
            h.remove()

    if packet is not None and calls!=1:
        raise RuntimeError(f"Injection calls={calls}")

    return r.past_key_values

print("[9/29] Capture TEST236 L19 transition...")

@torch.inference_mode()
def continuation(past):
    S={};hs=[]

    def l18_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["L18_OUT"]=x[0,-1].float().detach().clone()

    def l19_in(m,args):
        S["L19_IN"]=args[0][0,-1].float().detach().clone()

    def l19_attn(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["_ATTN"]=x[0,-1].float().detach().clone()

    def l19_mlp(m,args,out):
        S["L19_MLP"]=out[0,-1].float().detach().clone()

    def l19_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["L19_OUT"]=x[0,-1].float().detach().clone()

    def l20_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["L20_OUT"]=x[0,-1].float().detach().clone()

    def l27_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["L27_OUT"]=x[0,-1].float().detach().clone()

    hs.append(layers[18].register_forward_hook(l18_out))
    hs.append(layers[19].register_forward_pre_hook(l19_in))
    hs.append(layers[19].self_attn.register_forward_hook(l19_attn))
    hs.append(layers[19].mlp.register_forward_hook(l19_mlp))
    hs.append(layers[19].register_forward_hook(l19_out))
    hs.append(layers[20].register_forward_hook(l20_out))
    hs.append(layers[27].register_forward_hook(l27_out))

    try:
        r=model(
            input_ids=torch.tensor(
                [[COMMON]],
                device=DEVICE
            ),
            past_key_values=past,
            use_cache=False,
            return_dict=True
        )
    finally:
        remove(hs)

    S["L19_POST"]=S["L19_IN"]+S["_ATTN"]
    del S["_ATTN"]

    return S,r.logits[0,-1].float().detach().clone()

VAN={};INJ={}

for c in range(C):
    past=prefill(QENC[c],None)
    VAN[c],_=continuation(past)

    for o in range(O):
        past=prefill(QENC[c],OBJ[o])
        INJ[(c,o)],_=continuation(past)

    print(f"C{c+1}: continuation captures complete")

print("[10/29] Build centered carrier states...")
NAMES=[
    "L18_OUT",
    "L19_IN",
    "L19_POST",
    "L19_MLP",
    "L19_OUT",
    "L20_OUT",
    "L27_OUT"
]

D={};CENTER={}

for name in NAMES:
    for c in range(C):
        ds=[]

        for o in range(O):
            x=INJ[(c,o)][name]-VAN[c][name]
            D[(c,o,name)]=x
            ds.append(x)

        m=torch.stack(ds).mean(0)

        for o in range(O):
            CENTER[(c,o,name)]=D[(c,o,name)]-m

print("[11/29] Build explicit L19 recoding delta...")
REC={}

for c in range(C):
    arr=[]

    for o in range(O):
        x=CENTER[(c,o,"L19_OUT")]-CENTER[(c,o,"L19_POST")]
        REC[(c,o)]=x
        arr.append(x)

    m=torch.stack(arr).mean(0)

    for o in range(O):
        REC[(c,o)]=REC[(c,o)]-m

print("[12/29] Global recoding-delta spectrum...")
RALL=torch.stack([
    REC[(c,o)]
    for c in range(C)
    for o in range(O)
]).float()

_,SALL,VhALL=torch.linalg.svd(
    RALL,
    full_matrices=False
)

ENERGY=SALL.square()
ENERGY=ENERGY/ENERGY.sum().clamp_min(EPS)
CUM=torch.cumsum(ENERGY,0)

for i in range(min(16,len(SALL))):
    print(
        f"PC{i+1:02d} singular={float(SALL[i]):.6f} "
        f"energy={float(ENERGY[i])*100:6.2f}% "
        f"cumulative={float(CUM[i])*100:6.2f}%"
    )

print("[13/29] Effective dimensionality...")
p=ENERGY[ENERGY>0]
entropy=float(-(p*torch.log(p)).sum())
effective=math.exp(entropy)
participation=float(1.0/(p.square().sum()))
r90=int(torch.where(CUM>=.90)[0][0])+1
r95=int(torch.where(CUM>=.95)[0][0])+1
r99=int(torch.where(CUM>=.99)[0][0])+1

print(f"entropyEffectiveRank={effective:.3f}")
print(f"participationRatio={participation:.3f}")
print(f"rank90={r90} rank95={r95} rank99={r99}")

print("[14/29] Leave-one-context-out train-only recoding spectrum...")
SPECTRA={}

for hold in range(C):
    train=[c for c in range(C) if c!=hold]

    X=torch.stack([
        REC[(c,o)]
        for c in train
        for o in range(O)
    ]).float()

    _,s,vh=torch.linalg.svd(
        X,
        full_matrices=False
    )

    e=s.square()
    e=e/e.sum().clamp_min(EPS)
    cu=torch.cumsum(e,0)

    rr90=int(torch.where(cu>=.90)[0][0])+1
    rr95=int(torch.where(cu>=.95)[0][0])+1
    SPECTRA[hold]=(s,e,cu,vh)

    print(
        f"C{hold+1} trainSpectrum "
        f"r90={rr90} r95={rr95} "
        f"PC1={float(e[0])*100:.2f}% "
        f"PC1-4={float(cu[min(3,len(cu)-1)])*100:.2f}% "
        f"PC1-7={float(cu[min(6,len(cu)-1)])*100:.2f}%"
    )

print("[15/29] Held-out recoding-delta identity decoding...")
RECDEC={}

for rank in RANKS:
    hit=0;ranks=[];margins=[]

    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        _,_,_,vh=SPECTRA[hold]
        B=vh[:rank].T

        cent=[
            torch.stack([
                REC[(c,o)]@B
                for c in train
            ]).mean(0)
            for o in range(O)
        ]

        for o in range(O):
            q=REC[(hold,o)]@B
            sc=[cos(q,cent[j]) for j in range(O)]

            rr=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rr==1
            ranks.append(rr)

            margins.append(
                sc[o]-max(
                    sc[j]
                    for j in range(O)
                    if j!=o
                )
            )

    RECDEC[rank]=(
        hit/(C*O),
        float(np.mean(ranks)),
        float(np.mean(margins))
    )

    x=RECDEC[rank]

    print(
        f"rank={rank} "
        f"top1={x[0]*100:5.1f}% "
        f"meanRank={x[1]:.3f} "
        f"margin={x[2]:+.4f}"
    )

print("[16/29] Project L19 OUT carrier into recoding subspace...")
TOP={}

for rank in RANKS:
    hit=0;ranks=[];margins=[]

    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        _,_,_,vh=SPECTRA[hold]
        B=vh[:rank].T

        cent=[
            torch.stack([
                CENTER[(c,o,"L19_OUT")]@B
                for c in train
            ]).mean(0)
            for o in range(O)
        ]

        for o in range(O):
            q=CENTER[(hold,o,"L19_OUT")]@B
            sc=[cos(q,cent[j]) for j in range(O)]

            rr=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rr==1
            ranks.append(rr)

            margins.append(
                sc[o]-max(
                    sc[j]
                    for j in range(O)
                    if j!=o
                )
            )

    TOP[rank]=(
        hit/(C*O),
        float(np.mean(ranks)),
        float(np.mean(margins))
    )

    x=TOP[rank]

    print(
        f"TOP rank={rank} "
        f"top1={x[0]*100:5.1f}% "
        f"rank={x[1]:.3f} "
        f"margin={x[2]:+.4f}"
    )

print("[17/29] Complement decoding...")
COMP={}

for rank in RANKS:
    hit=0;ranks=[];margins=[]

    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        _,_,_,vh=SPECTRA[hold]
        B=vh[:rank].T

        def comp(x):
            return x-(x@B)@B.T

        cent=[
            torch.stack([
                comp(CENTER[(c,o,"L19_OUT")])
                for c in train
            ]).mean(0)
            for o in range(O)
        ]

        for o in range(O):
            q=comp(CENTER[(hold,o,"L19_OUT")])
            sc=[cos(q,cent[j]) for j in range(O)]

            rr=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rr==1
            ranks.append(rr)

            margins.append(
                sc[o]-max(
                    sc[j]
                    for j in range(O)
                    if j!=o
                )
            )

    COMP[rank]=(
        hit/(C*O),
        float(np.mean(ranks)),
        float(np.mean(margins))
    )

    x=COMP[rank]

    print(
        f"COMPLEMENT removeTop={rank} "
        f"top1={x[0]*100:5.1f}% "
        f"rank={x[1]:.3f} "
        f"margin={x[2]:+.4f}"
    )

print("[18/29] Individual component decoding...")
SINGLE={}

for k in range(7):
    hit=0;ranks=[]

    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        _,_,_,vh=SPECTRA[hold]
        b=vh[k]

        cent=[
            float(torch.stack([
                CENTER[(c,o,"L19_OUT")]@b
                for c in train
            ]).mean())
            for o in range(O)
        ]

        for o in range(O):
            q=float(CENTER[(hold,o,"L19_OUT")]@b)

            # 1D nearest centroid, not cosine.
            dist=[abs(q-cent[j]) for j in range(O)]
            rr=np.argsort(dist).tolist().index(o)+1

            hit+=rr==1
            ranks.append(rr)

    SINGLE[k+1]=(
        hit/(C*O),
        float(np.mean(ranks))
    )

    print(
        f"PC{k+1} "
        f"top1={SINGLE[k+1][0]*100:5.1f}% "
        f"rank={SINGLE[k+1][1]:.3f}"
    )

print("[19/29] Leave-one-component-out decoding...")
LOO={}

for remove_pc in range(7):
    hit=0;ranks=[];margins=[]

    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        _,_,_,vh=SPECTRA[hold]

        keep=[
            k for k in range(7)
            if k!=remove_pc
        ]

        B=vh[keep].T

        cent=[
            torch.stack([
                CENTER[(c,o,"L19_OUT")]@B
                for c in train
            ]).mean(0)
            for o in range(O)
        ]

        for o in range(O):
            q=CENTER[(hold,o,"L19_OUT")]@B
            sc=[cos(q,cent[j]) for j in range(O)]

            rr=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rr==1
            ranks.append(rr)

            margins.append(
                sc[o]-max(
                    sc[j]
                    for j in range(O)
                    if j!=o
                )
            )

    LOO[remove_pc+1]=(
        hit/(C*O),
        float(np.mean(ranks)),
        float(np.mean(margins))
    )

    x=LOO[remove_pc+1]

    print(
        f"removePC{remove_pc+1} "
        f"top1={x[0]*100:5.1f}% "
        f"rank={x[1]:.3f} "
        f"margin={x[2]:+.4f}"
    )

print("[20/29] Compare PRE / DELTA / POST identity...")
def heldout_decode(getvec):
    hit=0;ranks=[];margins=[]

    for hold in range(C):
        train=[c for c in range(C) if c!=hold]

        cent=[
            torch.stack([
                getvec(c,o)
                for c in train
            ]).mean(0)
            for o in range(O)
        ]

        for o in range(O):
            q=getvec(hold,o)
            sc=[cos(q,cent[j]) for j in range(O)]

            rr=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rr==1
            ranks.append(rr)

            margins.append(
                sc[o]-max(
                    sc[j]
                    for j in range(O)
                    if j!=o
                )
            )

    return (
        hit/(C*O),
        float(np.mean(ranks)),
        float(np.mean(margins))
    )

PRE=heldout_decode(
    lambda c,o:CENTER[(c,o,"L19_POST")]
)

DELTA=heldout_decode(
    lambda c,o:REC[(c,o)]
)

POST=heldout_decode(
    lambda c,o:CENTER[(c,o,"L19_OUT")]
)

print(
    f"PRE   top1={PRE[0]*100:5.1f}% "
    f"rank={PRE[1]:.3f} margin={PRE[2]:+.4f}"
)

print(
    f"DELTA top1={DELTA[0]*100:5.1f}% "
    f"rank={DELTA[1]:.3f} margin={DELTA[2]:+.4f}"
)

print(
    f"POST  top1={POST[0]*100:5.1f}% "
    f"rank={POST[1]:.3f} margin={POST[2]:+.4f}"
)

print("[21/29] PRE-to-DELTA alignment...")
vals=[]
for c in range(C):
    for o in range(O):
        vals.append(
            cos(
                CENTER[(c,o,"L19_POST")],
                REC[(c,o)]
            )
        )

print(
    f"meanCos(PRE,RECODING_DELTA)="
    f"{np.mean(vals):+.4f}"
)

print("[22/29] Recoding energy relative to PRE...")
ratio=[]

for c in range(C):
    for o in range(O):
        ratio.append(
            float(
                REC[(c,o)].norm()/
                CENTER[(c,o,"L19_POST")].norm().clamp_min(EPS)
            )
        )

print(
    f"mean ||delta||/||pre||="
    f"{np.mean(ratio):.4f} "
    f"({np.mean(ratio)*100:.2f}%)"
)

print("[23/29] Subspace energy of held-out POST carriers...")
for rank in RANKS:
    vals=[]

    for hold in range(C):
        _,_,_,vh=SPECTRA[hold]
        B=vh[:rank].T

        for o in range(O):
            x=CENTER[(hold,o,"L19_OUT")]
            p=(x@B)@B.T

            vals.append(
                float(
                    p.square().sum()/
                    x.square().sum().clamp_min(EPS)
                )
            )

    print(
        f"rank={rank} "
        f"POST energy in recoding subspace="
        f"{np.mean(vals)*100:.2f}%"
    )

print("[24/29] Context-wise rank-7 recoding-subspace decoding...")
for hold in range(C):
    train=[c for c in range(C) if c!=hold]
    _,_,_,vh=SPECTRA[hold]
    B=vh[:7].T

    cent=[
        torch.stack([
            CENTER[(c,o,"L19_OUT")]@B
            for c in train
        ]).mean(0)
        for o in range(O)
    ]

    hits=0;ranks=[]

    for o in range(O):
        q=CENTER[(hold,o,"L19_OUT")]@B
        sc=[cos(q,cent[j]) for j in range(O)]

        rr=np.argsort(sc)[::-1].tolist().index(o)+1
        hits+=rr==1
        ranks.append(rr)

    print(
        f"C{hold+1} "
        f"top1={hits/O*100:5.1f}% "
        f"rank={np.mean(ranks):.3f}"
    )

print("[25/29] Permutation null — rank7 recoding subspace...")
rows=[]

for hold in range(C):
    train=[c for c in range(C) if c!=hold]
    _,_,_,vh=SPECTRA[hold]
    B=vh[:7].T

    cent=[
        torch.stack([
            CENTER[(c,o,"L19_OUT")]@B
            for c in train
        ]).mean(0)
        for o in range(O)
    ]

    for o in range(O):
        q=CENTER[(hold,o,"L19_OUT")]@B
        rows.append([
            cos(q,cent[j])
            for j in range(O)
        ])

arr=np.asarray(rows)
labels=np.tile(np.arange(O),C)
obs=TOP[7][0]

rng=np.random.default_rng(SEED)
PERMS=2000
null=[]

for _ in range(PERMS):
    lab=rng.permutation(labels)

    null.append(
        float(
            np.mean(
                np.argmax(arr,axis=1)==lab
            )
        )
    )

mu=float(np.mean(null))
sd=float(np.std(null)+1e-12)
z=(obs-mu)/sd
pv=(1+sum(x>=obs for x in null))/(PERMS+1)

print(
    f"obs={obs:.4f} "
    f"null={mu:.4f}±{sd:.4f} "
    f"z={z:+.3f} p={pv:.4f}"
)

print("[26/29] Downstream relation of recoding subspace...")
DOWN={}

for target in ["L20_OUT","L27_OUT"]:
    for rank in [3,4,6,7]:
        hit=0;ranks=[]

        for hold in range(C):
            train=[c for c in range(C) if c!=hold]
            _,_,_,vh=SPECTRA[hold]
            B=vh[:rank].T

            X=torch.stack([
                CENTER[(c,o,"L19_OUT")]@B
                for c in train
                for o in range(O)
            ]).float()

            Y=torch.stack([
                CENTER[(c,o,target)]
                for c in train
                for o in range(O)
            ]).float()

            A=torch.linalg.lstsq(X,Y).solution

            cent=[
                torch.stack([
                    CENTER[(c,o,target)]
                    for c in train
                ]).mean(0)
                for o in range(O)
            ]

            for o in range(O):
                q=CENTER[(hold,o,"L19_OUT")]@B
                pred=q@A

                sc=[
                    cos(pred,cent[j])
                    for j in range(O)
                ]

                rr=np.argsort(sc)[::-1].tolist().index(o)+1
                hit+=rr==1
                ranks.append(rr)

        DOWN[(target,rank)]=(
            hit/(C*O),
            float(np.mean(ranks))
        )

        print(
            f"L19 recoding r={rank} -> {target} "
            f"top1={DOWN[(target,rank)][0]*100:5.1f}% "
            f"rank={DOWN[(target,rank)][1]:.3f}"
        )

print("[27/29] Weight sentinel...")
if fp()!=FP0:
    raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")

print("[28/29] Decision...")
best_rank=max(
    RANKS,
    key=lambda r:TOP[r][0]
)

best_top=TOP[best_rank][0]
comp7=COMP[7][0]
delta_top=DELTA[0]

print(f"effectiveRank={effective:.3f}")
print(f"participationRatio={participation:.3f}")
print(f"rank95={r95}")
print(f"bestRecodingSubspaceRank={best_rank}")
print(f"bestRecodingSubspaceTop1={best_top:.4f}")
print(f"rank7ComplementTop1={comp7:.4f}")
print(f"recodingDeltaIdentityTop1={delta_top:.4f}")

print("[29/29] RESULTS")
print("="*128)
print("TEST 237 RESULTS")
print("="*128)
print("BASELINE: TEST236 | OBSERVATIONAL | SINGLE L08 PROMPT INJECTION | CONTINUATION ' the'")
print("TARGET: L19 POST_ATTN -> L19 OUT RECODING-DELTA SUBSPACE")
print(
    f"SPECTRUM: effectiveRank={effective:.3f} "
    f"participation={participation:.3f} "
    f"r90={r90} r95={r95} r99={r99}"
)
print(
    f"PRE={PRE[0]*100:.1f}% "
    f"RECODING_DELTA={DELTA[0]*100:.1f}% "
    f"POST={POST[0]*100:.1f}%"
)
print(
    f"BEST TOP-SUBSPACE: rank={best_rank} "
    f"top1={best_top*100:.1f}%"
)
print(
    f"REMOVE TOP7 COMPLEMENT: "
    f"top1={comp7*100:.1f}%"
)
print("-"*128)

if best_top>=.75 and comp7<=.35 and r95<=7:
    print("RESULT: CONCENTRATED_MULTIDIMENSIONAL_MLP_RECODING_SUBSPACE")
elif best_top>=.60 and best_top>comp7+.20:
    print("RESULT: PARTIAL_MULTIDIMENSIONAL_MLP_RECODING_SUBSPACE")
elif delta_top>=.50:
    print("RESULT: DISTRIBUTED_IDENTITY_BEARING_RECODING_DELTA")
else:
    print("RESULT: NO_CLEAR_LOW_DIMENSIONAL_RECODING_SUBSPACE")

print("No model intervention beyond the original L08 prompt injection was used.")
print("SVD bases and downstream maps are measurement/readout tools only; they are not part of the model forward path.")
print("="*128)
print("TEST 237 COMPLETE")
