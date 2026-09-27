# ==================================================================================================
# TEST 243 — SWIGLU INTERACTION INTERFERENCE X-RAY
# WORKING BASELINE: TEST242
# NATIVE BF16 | OFFLINE GEOMETRY ONLY | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=243
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
TOTAL=28;H_EXPECT=3584;KVH=0;EPS=1e-8;PRIMARY=.04
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere","the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
C=len(CONTEXTS);O=len(OBJECTS)

print("="*116)
print("TEST 243 — SWIGLU INTERACTION INTERFERENCE X-RAY")
print("="*116)
print("Baseline: TEST242 | exact native BF16 | compact interaction audit")

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
        s=INJ[(c,o)]
        errs.append(relerr((ACT(s["G"])*s["U"]).float(),s["P"].float()))
MAXERR=max(errs)
print(f"maxNativeProductError={MAXERR*100:.9f}%")
if MAXERR>1e-7:raise RuntimeError("Native BF16 reconstruction failed.")

print("[7/12] Native BF16 decomposition...")
D={}
for c in range(C):
    g0=VAN[c]["G"];u0=VAN[c]["U"];p0=ACT(g0)*u0
    for o in range(O):
        g=INJ[(c,o)]["G"];u=INJ[(c,o)]["U"]
        matched=(ACT(g)*u-p0).float()
        gate=(ACT(g)*u0-p0).float()
        up=(ACT(g0)*u-p0).float()
        main=gate+up
        interaction=matched-main
        D[(c,o,"MATCHED")]=matched
        D[(c,o,"GATE_ONLY")]=gate
        D[(c,o,"UP_ONLY")]=up
        D[(c,o,"MAIN")]=main
        D[(c,o,"INTERACTION")]=interaction
        D[(c,o,"REMOVE_INTERACTION")]=matched-interaction

print("[8/12] Algebra + real-product sanity...")
ALG=[];REAL=[]
for c in range(C):
    for o in range(O):
        ALG.append(relerr(D[(c,o,"MAIN")]+D[(c,o,"INTERACTION")],D[(c,o,"MATCHED")]))
        real=INJ[(c,o)]["P"].float()-VAN[c]["P"].float()
        REAL.append(relerr(D[(c,o,"MATCHED")],real))
print(f"decompositionError={np.mean(ALG)*100:.9f}%")
print(f"matched-vs-realDelta error={np.mean(REAL)*100:.6f}%")

NAMES=["MATCHED","GATE_ONLY","UP_ONLY","MAIN","INTERACTION","REMOVE_INTERACTION"]
Z={}
for name in NAMES:
    for c in range(C):
        m=torch.stack([D[(c,o,name)] for o in range(O)]).mean(0)
        for o in range(O):Z[(c,o,name)]=D[(c,o,name)]-m

def decode(name):
    hit=0;ranks=[];margins=[]
    for hold in range(C):
        tr=[c for c in range(C) if c!=hold]
        cent=[torch.stack([Z[(c,o,name)] for c in tr]).mean(0) for o in range(O)]
        for o in range(O):
            q=Z[(hold,o,name)]
            sc=[cos(q,cent[j]) for j in range(O)]
            rank=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rank==1;ranks.append(rank)
            margins.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
    return hit/(C*O),float(np.mean(ranks)),float(np.mean(margins))

print("[9/12] Identity decoding...")
RES={name:decode(name) for name in NAMES}
for name in NAMES:
    a=RES[name]
    print(f"{name:18s} top1={a[0]*100:5.1f}% rank={a[1]:.3f} margin={a[2]:+.4f}")

print("[10/12] Interaction geometry...")
COS_MAIN=[];COS_GATE=[];COS_UP=[];RATIO=[];ALPHA=[];ORTH=[]
for c in range(C):
    for o in range(O):
        m=Z[(c,o,"MAIN")];i=Z[(c,o,"INTERACTION")]
        g=Z[(c,o,"GATE_ONLY")];u=Z[(c,o,"UP_ONLY")]
        COS_MAIN.append(cos(i,m));COS_GATE.append(cos(i,g));COS_UP.append(cos(i,u))
        RATIO.append(float(i.norm()/m.norm().clamp_min(EPS)))
        a=float(torch.dot(i,m)/torch.dot(m,m).clamp_min(EPS));ALPHA.append(a)
        io=i-a*m;ORTH.append(float(io.norm()/i.norm().clamp_min(EPS)))

print(f"cos(INTERACTION,MAIN)={np.mean(COS_MAIN):+.4f}")
print(f"cos(INTERACTION,GATE_ONLY)={np.mean(COS_GATE):+.4f}")
print(f"cos(INTERACTION,UP_ONLY)={np.mean(COS_UP):+.4f}")
print(f"interaction/main norm={np.mean(RATIO)*100:.3f}%")
print(f"interaction projection alpha={np.mean(ALPHA):+.4f}")
print(f"interaction orthogonal fraction={np.mean(ORTH)*100:.3f}%")

print("[11/12] Context/object interference...")
for c in range(C):
    cc=[];rr=[]
    for o in range(O):
        i=Z[(c,o,"INTERACTION")];m=Z[(c,o,"MAIN")]
        cc.append(cos(i,m));rr.append(float(i.norm()/m.norm().clamp_min(EPS)))
    print(f"C{c+1} cos={np.mean(cc):+.4f} ratio={np.mean(rr)*100:.2f}%")

objout=[]
for o in range(O):
    cc=[];rr=[]
    for c in range(C):
        i=Z[(c,o,"INTERACTION")];m=Z[(c,o,"MAIN")]
        cc.append(cos(i,m));rr.append(float(i.norm()/m.norm().clamp_min(EPS)))
    objout.append(f"O{o+1}:{np.mean(cc):+.3f}/{np.mean(rr)*100:.1f}%")
print("Objects: "+" | ".join(objout))

if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")

print("[12/12] RESULTS")
print("="*116)
print("TEST 243 RESULTS — SWIGLU INTERACTION INTERFERENCE")
for name in ["GATE_ONLY","UP_ONLY","MAIN","MATCHED","INTERACTION","REMOVE_INTERACTION"]:
    a=RES[name]
    print(f"{name:18s}: {a[0]*100:5.1f}% | rank {a[1]:.3f} | margin {a[2]:+.4f}")
print("-"*116)
DTOP=(RES["MAIN"][0]-RES["MATCHED"][0])*100
DMAR=RES["MAIN"][2]-RES["MATCHED"][2]
print(f"REMOVE INTERACTION: Δtop1={DTOP:+.1f}pp | Δmargin={DMAR:+.4f}")
print(f"INTERACTION/MAIN={np.mean(RATIO)*100:.3f}% | cos={np.mean(COS_MAIN):+.4f}")
print("Weights: PASS")
if DTOP>=10 and DMAR>0:print("RESULT: MULTIPLICATIVE_INTERACTION_INTERFERES_WITH_IDENTITY_GEOMETRY")
elif DTOP>=5 or DMAR>.03:print("RESULT: PARTIAL_MULTIPLICATIVE_INTERFERENCE")
elif RES["MATCHED"][0]>RES["MAIN"][0]+.05:print("RESULT: MULTIPLICATIVE_INTERACTION_SUPPORTS_IDENTITY_GEOMETRY")
else:print("RESULT: NO_STRONG_INTERACTION_EFFECT_ON_IDENTITY_GEOMETRY")
print("Exact native BF16 arithmetic preserved from TEST242.")
print("Offline decomposition only; no counterfactual tensor injected.")
print("="*116)
print("TEST 243 COMPLETE")
