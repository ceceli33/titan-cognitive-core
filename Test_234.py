# ==================================================================================================
# TEST 234 — BLOCK-LOCAL CARRIER DEGRADATION X-RAY
# TEST233 WORKING BASELINE -> TEST230 OBJECT MAIN-EFFECT PACKETS -> SINGLE L08 PROMPT INJECTION
# QUESTION: WHERE DOES THE AUTOREGRESSIVE OBJECT CARRIER DEGRADE — ATTENTION OR MLP?
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> ... -> TEST230 -> TEST231 -> TEST232 -> TEST233 -> TEST234
# TEST233 MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / OBJECT MAIN EFFECT PRESERVED
# PROMPT: SINGLE L08 INJECTION | CONTINUATION " the": ZERO INTERVENTION
# MEASURE: BLOCK INPUT -> ATTN DELTA -> POST-ATTN RESIDUAL -> MLP DELTA -> BLOCK OUTPUT
# NO TRANSPORT MAP | NO CONTROLLER | NO CONTINUATION RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=234
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
PRIMARY=.04;PROBE_LAYERS=list(range(8,28));SHOW=list(range(8,28))
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere",
         "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
C=len(CONTEXTS);O=len(OBJECTS)

print("="*128);print("TEST 234 — BLOCK-LOCAL CARRIER DEGRADATION X-RAY");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228 -> TEST229 -> TEST230 -> TEST231 -> TEST232 -> TEST233 -> TEST234")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/29] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}
)
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers;H=model.config.hidden_size;NH=model.config.num_attention_heads
NKV=model.config.num_key_value_heads;HD=H//NH;GROUP=NH//NKV
if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:
    raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")

FP_T=[
    layers[0].self_attn.q_proj.weight,
    layers[8].self_attn.o_proj.weight,
    layers[19].mlp.down_proj.weight,
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
        tokenize=False,add_generation_prompt=True
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

print("[2/29] Build TEST233 factorial source/blind set...")
FMAP={};QENC={}
for c,(s,r) in enumerate(CONTEXTS):
    qe=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    QENC[c]=qe
    for o,obj in enumerate(OBJECTS):
        fi=tok(chat(fact_text(s,r,obj)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
        full=fi.input_ids[0].tolist()
        ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,obj)
        if not ss or not rs or not os_:
            raise RuntimeError(f"Token map fail C{c+1} O{o+1}")
        if obj.lower() in qform(s,r).lower():
            raise RuntimeError("Target leakage.")
        FMAP[(c,o)]=(fi,ss,rs,os_)
    print(f"C{c+1} {s} {r} | blindSlot={qe.input_ids.shape[1]-1}")

print("[3/29] Candidate/shared-prefix tokens...")
CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj)
    CANDS.append(b if len(b)<=len(a) else a)

COMMON=CANDS[0][0]
DISC=[x[1] for x in CANDS]

if not all(x[0]==COMMON for x in CANDS):
    raise RuntimeError("No common first token.")
if len(set(DISC))!=O:
    raise RuntimeError("Discriminative tokens not unique.")

print(f"sharedPrefixToken={COMMON} text={tok.decode([COMMON])!r}")
for o in range(O):
    print(f"O{o+1} {OBJECTS[o]} -> {tok.decode([DISC[o]])!r}")

print("[4/29] RoPE...")
rotary=model.model.rotary_emb
MAXSEQ=max(
    max(v[0].input_ids.shape[1] for v in FMAP.values()),
    max(v.input_ids.shape[1] for v in QENC.values())
)+4

dummy=torch.zeros(1,MAXSEQ,H,device=DEVICE,dtype=model.dtype)
pos=torch.arange(MAXSEQ,device=DEVICE).unsqueeze(0)

with torch.inference_mode():
    COS,SIN=rotary(dummy,pos)

COS=COS[0].float();SIN=SIN[0].float()

def rotate_half(x):
    n=x.shape[-1]//2
    return torch.cat((-x[...,n:],x[...,:n]),dim=-1)

def rope(x,p):
    return x*COS[p]+rotate_half(x)*SIN[p]

print(f"RoPE={type(rotary).__name__} max_seq={MAXSEQ}")

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
    print(f"C{c+1}: 8 source captures complete")

print("[6/29] Reconstruct TEST222 readout + RAW packets...")
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

        with torch.inference_mode():
            RAW[(c,o)]=layers[8].self_attn.o_proj(
                p.reshape(H).to(model.dtype)
            ).float()

    print(f"C{c+1}: RAW packets complete")

print("[7/29] TEST230 factorial object-main packets...")
GRAND=torch.stack(list(RAW.values())).mean(0)
OMEAN={
    o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0)
    for o in range(O)
}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}

for o in range(O):
    print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")

print("[8/29] Prompt prefill: vanilla + single L08 object injection...")

@torch.inference_mode()
def prefill(e,packet=None):
    calls=0

    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None

        y=x.clone()
        z=y[:,-1,:].float()
        d=unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY
        y[:,-1,:]=(z+d).to(y.dtype)
        calls+=1

        return (y,)+out[1:] if isinstance(out,tuple) else y

    ih=layers[8].register_forward_hook(inject) if packet is not None else None

    try:
        r=model(**e,use_cache=True,return_dict=True)
    finally:
        if ih is not None:ih.remove()

    if packet is not None and calls!=1:
        raise RuntimeError(f"Injection calls={calls}")

    return r.logits[0,-1].float().detach().clone(),r.past_key_values

VPLOG={};VPKV={};IPLOG={};IPKV={}

for c in range(C):
    VPLOG[c],VPKV[c]=prefill(QENC[c],None)

    for o in range(O):
        IPLOG[(c,o)],IPKV[(c,o)]=prefill(QENC[c],OBJ[o])

    print(f"C{c+1}: prompt prefill complete")

print("[9/29] Block-local continuation X-Ray...")
STAGES=["IN","ATTN","POST_ATTN","MLP","OUT"]
V={};I={}

@torch.inference_mode()
def block_xray(past):
    S={};hs=[]

    for L in PROBE_LAYERS:

        def block_pre(li):
            def hk(m,args):
                S[(li,"IN")]=args[0][0,-1].float().detach().clone()
            return hk

        def attn_out(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[(li,"ATTN")]=x[0,-1].float().detach().clone()
            return hk

        def mlp_pre(li):
            def hk(m,args):
                S[(li,"MLP_IN_NORM")]=args[0][0,-1].float().detach().clone()
            return hk

        def mlp_out(li):
            def hk(m,args,out):
                S[(li,"MLP")]=out[0,-1].float().detach().clone()
            return hk

        def block_out(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[(li,"OUT")]=x[0,-1].float().detach().clone()
            return hk

        hs.append(layers[L].register_forward_pre_hook(block_pre(L)))
        hs.append(layers[L].self_attn.register_forward_hook(attn_out(L)))
        hs.append(layers[L].mlp.register_forward_pre_hook(mlp_pre(L)))
        hs.append(layers[L].mlp.register_forward_hook(mlp_out(L)))
        hs.append(layers[L].register_forward_hook(block_out(L)))

    try:
        r=model(
            input_ids=torch.tensor([[COMMON]],device=DEVICE),
            past_key_values=past,
            use_cache=True,
            return_dict=True
        )
    finally:
        remove(hs)

    for L in PROBE_LAYERS:
        S[(L,"POST_ATTN")]=S[(L,"IN")]+S[(L,"ATTN")]

    return S,r.logits[0,-1].float().detach().clone()

for c in range(C):
    s,l=block_xray(VPKV[c])
    V[c]=s

    for o in range(O):
        s,l=block_xray(IPKV[(c,o)])
        for k,x in s.items():
            I[(c,o,k[0],k[1])]=x

    print(f"C{c+1}: 8 continuation X-rays complete")

print("[10/29] Verify exact block residual identities...")

for L in PROBE_LAYERS:
    e1=[];e2=[]

    for c in range(C):
        a=V[c][(L,"POST_ATTN")]

        # Syntax correction only:
        e1.append(
            float(
                (
                    a-(V[c][(L,"IN")]+V[c][(L,"ATTN")])
                ).norm()
            )
        )

        e2.append(
            float(
                (
                    V[c][(L,"OUT")]-
                    (a+V[c][(L,"MLP")])
                ).norm()/V[c][(L,"OUT")].norm()
            )
        )

    if max(e1)>1e-6:
        raise RuntimeError(f"POST_ATTN reconstruction fail L{L}")

    print(
        f"L{L:02d} postAttnExact={max(e1):.2e} "
        f"blockResidualRelErr={np.mean(e2):.3e}"
    )

print("[11/29] Build object-specific stage deltas...")
D={}

for L in PROBE_LAYERS:
    for st in STAGES:
        for c in range(C):
            for o in range(O):
                D[(c,o,L,st)]=I[(c,o,L,st)]-V[c][(L,st)]

print("[12/29] Center stage deltas within context...")
CENTER={}

for L in PROBE_LAYERS:
    for st in STAGES:
        for c in range(C):
            m=torch.stack([
                D[(c,o,L,st)]
                for o in range(O)
            ]).mean(0)

            for o in range(O):
                CENTER[(c,o,L,st)]=D[(c,o,L,st)]-m

print("[13/29] Leave-one-context-out decoder per block stage...")
DEC={}

for L in PROBE_LAYERS:
    for st in STAGES:
        hit=0;ranks=[];margins=[]

        for hold in range(C):
            tr=[c for c in range(C) if c!=hold]

            cent=[
                torch.stack([
                    CENTER[(c,o,L,st)]
                    for c in tr
                ]).mean(0)
                for o in range(O)
            ]

            for o in range(O):
                scores=[
                    cos(CENTER[(hold,o,L,st)],cent[j])
                    for j in range(O)
                ]

                rank=np.argsort(scores)[::-1].tolist().index(o)+1
                hit+=rank==1
                ranks.append(rank)
                margins.append(
                    scores[o]-max(
                        scores[j]
                        for j in range(O)
                        if j!=o
                    )
                )

        DEC[(L,st)]=(
            hit/(C*O),
            float(np.mean(ranks)),
            float(np.mean(margins))
        )

for L in SHOW:
    print(
        f"L{L:02d} "+
        " | ".join(
            f"{st}:{DEC[(L,st)][0]*100:5.1f}%/{DEC[(L,st)][1]:.2f}"
            for st in STAGES
        )
    )

print("[14/29] Stage displacement...")
DISP={}

for L in PROBE_LAYERS:
    for st in STAGES:
        vals=[]

        for c in range(C):
            for o in range(O):
                den=V[c][(L,st)].norm().clamp_min(EPS)
                vals.append(
                    float(D[(c,o,L,st)].norm()/den)*100
                )

        DISP[(L,st)]=float(np.mean(vals))

for L in SHOW:
    print(
        f"L{L:02d} "+
        " | ".join(
            f"{st}={DISP[(L,st)]:.4f}%"
            for st in STAGES
        )
    )

print("[15/29] Correct-vs-wrong separation per stage...")
SEP={}

for L in PROBE_LAYERS:
    for st in STAGES:
        cor=[];wr=[]

        for hold in range(C):
            tr=[c for c in range(C) if c!=hold]

            cent=[
                torch.stack([
                    CENTER[(c,o,L,st)]
                    for c in tr
                ]).mean(0)
                for o in range(O)
            ]

            for o in range(O):
                cor.append(
                    cos(CENTER[(hold,o,L,st)],cent[o])
                )

                wr.extend(
                    cos(CENTER[(hold,o,L,st)],cent[j])
                    for j in range(O)
                    if j!=o
                )

        SEP[(L,st)]=(
            float(np.mean(cor)),
            float(np.mean(wr)),
            float(np.mean(cor)-np.mean(wr))
        )

for L in SHOW:
    print(
        f"L{L:02d} "+
        " | ".join(
            f"{st}Gap={SEP[(L,st)][2]:+.4f}"
            for st in STAGES
        )
    )

print("[16/29] Same-object cross-context invariance...")
INV={}

for L in PROBE_LAYERS:
    for st in STAGES:
        vals=[]

        for o in range(O):
            for a in range(C):
                for b in range(a+1,C):
                    vals.append(
                        cos(
                            CENTER[(a,o,L,st)],
                            CENTER[(b,o,L,st)]
                        )
                    )

        INV[(L,st)]=float(np.mean(vals))

for L in SHOW:
    print(
        f"L{L:02d} "+
        " | ".join(
            f"{st}Inv={INV[(L,st)]:+.4f}"
            for st in STAGES
        )
    )

print("[17/29] Stage-to-stage identity survival...")
TRANS={}
pairs=[
    ("IN","ATTN"),
    ("IN","POST_ATTN"),
    ("POST_ATTN","MLP"),
    ("POST_ATTN","OUT")
]

for L in PROBE_LAYERS:
    for a,b in pairs:
        vals=[
            cos(
                CENTER[(c,o,L,a)],
                CENTER[(c,o,L,b)]
            )
            for c in range(C)
            for o in range(O)
        ]

        TRANS[(L,a,b)]=float(np.mean(vals))

for L in SHOW:
    print(
        f"L{L:02d} "
        f"IN->ATTN={TRANS[(L,'IN','ATTN')]:+.4f} "
        f"IN->POST={TRANS[(L,'IN','POST_ATTN')]:+.4f} "
        f"POST->MLP={TRANS[(L,'POST_ATTN','MLP')]:+.4f} "
        f"POST->OUT={TRANS[(L,'POST_ATTN','OUT')]:+.4f}"
    )

print("[18/29] Attention contribution to carrier quality...")
ATTN_EFFECT={}
MLP_EFFECT={}

for L in PROBE_LAYERS:
    ATTN_EFFECT[L]=(
        DEC[(L,"POST_ATTN")][0]-DEC[(L,"IN")][0],
        DEC[(L,"POST_ATTN")][2]-DEC[(L,"IN")][2],
        SEP[(L,"POST_ATTN")][2]-SEP[(L,"IN")][2]
    )

    MLP_EFFECT[L]=(
        DEC[(L,"OUT")][0]-DEC[(L,"POST_ATTN")][0],
        DEC[(L,"OUT")][2]-DEC[(L,"POST_ATTN")][2],
        SEP[(L,"OUT")][2]-SEP[(L,"POST_ATTN")][2]
    )

for L in SHOW:
    a=ATTN_EFFECT[L]
    m=MLP_EFFECT[L]

    print(
        f"L{L:02d} "
        f"ATTN Δtop1={a[0]*100:+5.1f}pp "
        f"Δmargin={a[1]:+.4f} Δsep={a[2]:+.4f} | "
        f"MLP Δtop1={m[0]*100:+5.1f}pp "
        f"Δmargin={m[1]:+.4f} Δsep={m[2]:+.4f}"
    )

print("[19/29] Locate largest local degradation...")
events=[]

for L in PROBE_LAYERS:
    events.append(
        (ATTN_EFFECT[L][0],L,"ATTENTION",ATTN_EFFECT[L])
    )
    events.append(
        (MLP_EFFECT[L][0],L,"MLP",MLP_EFFECT[L])
    )

for e in sorted(events,key=lambda x:x[0])[:12]:
    print(
        f"L{e[1]:02d} {e[2]:9s} "
        f"Δtop1={e[3][0]*100:+5.1f}pp "
        f"Δmargin={e[3][1]:+.4f} "
        f"Δsep={e[3][2]:+.4f}"
    )

print("[20/29] Cross-block survival...")

for L in range(9,28):
    a=DEC[(L-1,"OUT")]
    b=DEC[(L,"IN")]

    print(
        f"L{L-1:02d}OUT -> L{L:02d}IN "
        f"top1={a[0]*100:5.1f}%->{b[0]*100:5.1f}% "
        f"rank={a[1]:.3f}->{b[1]:.3f}"
    )

print("[21/29] Critical L17-L21 window...")

for L in range(17,22):
    print(f"\nL{L:02d}")

    for st in STAGES:
        x=DEC[(L,st)]

        print(
            f" {st:9s} "
            f"top1={x[0]*100:5.1f}% "
            f"rank={x[1]:.3f} "
            f"margin={x[2]:+.4f} "
            f"sep={SEP[(L,st)][2]:+.4f} "
            f"inv={INV[(L,st)]:+.4f}"
        )

print("[22/29] Geometry retention against L10 continuation carrier...")
REF={}

for o in range(O):
    REF[o]=torch.stack([
        CENTER[(c,o,10,"OUT")]
        for c in range(C)
    ]).mean(0)

GEO={}

for L in PROBE_LAYERS:
    for st in STAGES:
        cor=[];wr=[]

        for c in range(C):
            for o in range(O):
                cor.append(
                    cos(CENTER[(c,o,L,st)],REF[o])
                )

                wr.extend(
                    cos(CENTER[(c,o,L,st)],REF[j])
                    for j in range(O)
                    if j!=o
                )

        GEO[(L,st)]=(
            float(np.mean(cor)),
            float(np.mean(wr)),
            float(np.mean(cor)-np.mean(wr))
        )

for L in SHOW:
    print(
        f"L{L:02d} "+
        " | ".join(
            f"{st}RefGap={GEO[(L,st)][2]:+.4f}"
            for st in STAGES
        )
    )

print("[23/29] Permutation null at critical stages...")
rng=np.random.default_rng(SEED)
PERMS=2000
NULL={}

for L in [10,16,18,19,20,21,24,27]:
    for st in ["IN","POST_ATTN","OUT"]:
        rows=[]

        for hold in range(C):
            tr=[c for c in range(C) if c!=hold]

            cent=[
                torch.stack([
                    CENTER[(c,o,L,st)]
                    for c in tr
                ]).mean(0)
                for o in range(O)
            ]

            for o in range(O):
                rows.append([
                    cos(CENTER[(hold,o,L,st)],cent[j])
                    for j in range(O)
                ])

        arr=np.asarray(rows)
        labels=np.tile(np.arange(O),C)
        obs=DEC[(L,st)][0]
        vals=[]

        for _ in range(PERMS):
            lab=rng.permutation(labels)
            vals.append(
                float(np.mean(np.argmax(arr,axis=1)==lab))
            )

        mu=float(np.mean(vals))
        sd=float(np.std(vals)+1e-12)
        z=(obs-mu)/sd
        p=(1+sum(v>=obs for v in vals))/(PERMS+1)

        NULL[(L,st)]=(mu,sd,z,p)

        print(
            f"L{L:02d} {st:9s} "
            f"obs={obs:.4f} "
            f"null={mu:.4f}±{sd:.4f} "
            f"z={z:+.3f} p={p:.4f}"
        )

print("[24/29] Actual discriminative output...")
OUT=[]

for c in range(C):
    _,vl=block_xray(VPKV[c])

    for o in range(O):
        _,il=block_xray(IPKV[(c,o)])

        ds=[
            float(il[t]-vl[t])
            for t in DISC
        ]

        rank=np.argsort(ds)[::-1].tolist().index(o)+1

        OUT.append((
            rank,
            ds[o]-max(
                ds[j]
                for j in range(O)
                if j!=o
            ),
            ds[o]
        ))

print(
    f"top1={sum(x[0]==1 for x in OUT)/(C*O)*100:.1f}% "
    f"rank={np.mean([x[0] for x in OUT]):.3f} "
    f"Δmargin={np.mean([x[1] for x in OUT]):+.4f} "
    f"targetΔ={np.mean([x[2] for x in OUT]):+.4f}"
)

print("[25/29] Degradation attribution summary...")

for L in range(17,28):
    print(
        f"L{L:02d} "
        f"input={DEC[(L,'IN')][0]*100:5.1f}% "
        f"postAttn={DEC[(L,'POST_ATTN')][0]*100:5.1f}% "
        f"output={DEC[(L,'OUT')][0]*100:5.1f}% | "
        f"attn={ATTN_EFFECT[L][0]*100:+5.1f}pp "
        f"mlp={MLP_EFFECT[L][0]*100:+5.1f}pp"
    )

print("[26/29] Weight sentinel...")

if fp()!=FP0:
    raise RuntimeError("WEIGHT SENTINEL FAILED.")

print("[27/29] Decision...")
late_attn=float(
    np.mean([
        ATTN_EFFECT[L][0]
        for L in range(19,28)
    ])
)

late_mlp=float(
    np.mean([
        MLP_EFFECT[L][0]
        for L in range(19,28)
    ])
)

worst=min(events,key=lambda x:x[0])

print(f"lateAttentionMeanΔtop1={late_attn*100:+.2f}pp")
print(f"lateMLPMeanΔtop1={late_mlp*100:+.2f}pp")
print(
    f"largestLocalDrop=L{worst[1]:02d} "
    f"{worst[2]} {worst[0]*100:+.1f}pp"
)

print("[28/29] RESULTS")
print("\n"+"="*128)
print("TEST 234 RESULTS")
print("="*128)
print(
    f"MODE: TEST233 CARRIER | SINGLE PROMPT L08 INJECTION | "
    f"DOSE={PRIMARY:.4f} | CONTINUATION={tok.decode([COMMON])!r}"
)
print(
    "BLOCK X-RAY: INPUT -> ATTENTION -> POST-ATTENTION RESIDUAL -> "
    "MLP -> BLOCK OUTPUT"
)
print(
    "ZERO CONTINUATION INTERVENTION | NO TRANSPORT MAP | "
    "NO CONTROLLER | WEIGHTS FROZEN"
)

print("\nCRITICAL WINDOW")
for L in range(17,22):
    print(
        f"L{L:02d} "
        f"IN={DEC[(L,'IN')][0]*100:5.1f}% "
        f"POST_ATTN={DEC[(L,'POST_ATTN')][0]*100:5.1f}% "
        f"OUT={DEC[(L,'OUT')][0]*100:5.1f}% | "
        f"ATTN={ATTN_EFFECT[L][0]*100:+5.1f}pp "
        f"MLP={MLP_EFFECT[L][0]*100:+5.1f}pp"
    )

print("\nLATE AGGREGATE")
print(
    f"attentionMeanΔtop1={late_attn*100:+.2f}pp "
    f"mlpMeanΔtop1={late_mlp*100:+.2f}pp"
)
print(
    f"largestLocalDrop=L{worst[1]:02d} "
    f"{worst[2]} {worst[0]*100:+.1f}pp"
)

print("\nINTERPRETATION GATE")

if late_attn<-0.03 and late_attn<late_mlp:
    print(
        "RESULT: ATTENTION_DOMINANT_CARRIER_DEGRADATION — "
        "late attention transformations are the larger local source of identity loss."
    )
elif late_mlp<-0.03 and late_mlp<late_attn:
    print(
        "RESULT: MLP_DOMINANT_CARRIER_DEGRADATION — "
        "late MLP transformations are the larger local source of identity loss."
    )
elif late_attn<-0.02 and late_mlp<-0.02:
    print(
        "RESULT: DISTRIBUTED_BLOCK_DEGRADATION — "
        "both attention and MLP contribute materially to late carrier loss."
    )
else:
    print(
        "RESULT: NO_SINGLE_LOCAL_DEGRADATION_MECHANISM — "
        "carrier loss is distributed or dominated by cross-layer geometric transformation."
    )

print("-"*128)
print(
    "Weights: PASS | Objects absent from blind queries | "
    "Only prompt L08 is intervened"
)
print(
    "The shared continuation token is processed naturally; "
    "all block-local measurements are observational."
)
print("="*128)
print("[29/29] TEST 234 COMPLETE")
