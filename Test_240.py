# ==================================================================================================
# TEST 240 — L19 SWIGLU PRODUCT RECONSTRUCTION AUDIT — FIXED
# WORKING BASELINE: TEST239
# QUESTION: WHY DID TEST239 OFFLINE MATCHED PRODUCT DELTA DIFFER FROM REAL PRODUCT DELTA?
# --------------------------------------------------------------------------------------------------
# TEST239 BASELINE PRESERVED:
# Qwen2.5-7B-Instruct | BF16 | SDPA | SAME SYSTEM | SAME 4x8 FACTORIAL
# TEST222 RAW PACKET FORGE | TEST230 OBJECT-MAIN PACKETS
# SINGLE L08 PROMPT INJECTION | CONTINUATION " the"
# AUDIT ONLY: NATIVE BF16 vs FP32 PRODUCT RECONSTRUCTION
# NO NEW MODEL INTERVENTION | NO GUARD | NO TRANSPORT | NO CONTROLLER | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=240
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
print("TEST 240 — L19 SWIGLU PRODUCT RECONSTRUCTION AUDIT — FIXED")
print("="*128)
print("WORKING BASELINE: TEST239 | NUMERICAL / DEFINITION AUDIT")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0)
          for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/30] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}
)
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers
H=model.config.hidden_size
NH=model.config.num_attention_heads
NKV=model.config.num_key_value_heads
HD=H//NH;GROUP=NH//NKV
I=layers[19].mlp.gate_proj.out_features
ACT=layers[19].mlp.act_fn

if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:
    raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} intermediate={I} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")

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
def normratio(a,b):return float(a.norm()/b.norm().clamp_min(EPS))
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
    return f"What does {s} "+{"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r]+"?"
def remove(hs):
    for h in hs:h.remove()

print("[2/30] Build TEST239 factorial set...")
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
        S=SRC[(c,o)];oe=FMAP[(c,o)][3][-1];rows=[]
        for h in QGROUP:
            for qp in range(oe,FMAP[(c,o)][0].input_ids.shape[1]):
                a=attn_row(S,qp,h);rows.append((float(a[oe]),h,qp))
        v=kvh(S["V"],oe,KVH)
        p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32)
        for h in QGROUP:
            w=max(x[0] for x in rows if x[1]==h);p[h]=w*v
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

print("[9/30] Capture exact L19 native tensors...")
@torch.inference_mode()
def continuation(past):
    S={};hs=[];mlp=layers[19].mlp
    def gate_out(m,args,out):S["GATE"]=out[0,-1].detach().clone()
    def up_out(m,args,out):S["UP"]=out[0,-1].detach().clone()
    def product_in(m,args):S["PRODUCT"]=args[0][0,-1].detach().clone()
    hs.append(mlp.gate_proj.register_forward_hook(gate_out))
    hs.append(mlp.up_proj.register_forward_hook(up_out))
    hs.append(mlp.down_proj.register_forward_pre_hook(product_in))
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
    print(f"C{c+1}: exact captures complete")

print("[10/30] Native tensor dtypes...")
for k in ["GATE","UP","PRODUCT"]:
    print(f"{k}_NATIVE {INJ[(0,0)][k].dtype}")

print("[11/30] Native BF16 absolute reconstruction...")
BF_ABS_ERR=[];BF_ABS_COS=[]
for c in range(C):
    for o in range(O):
        s=INJ[(c,o)]
        rec=ACT(s["GATE"])*s["UP"]
        real=s["PRODUCT"]
        BF_ABS_ERR.append(relerr(rec.float(),real.float()))
        BF_ABS_COS.append(cos(rec.float(),real.float()))
print(f"BF16 absolute relErr={np.mean(BF_ABS_ERR)*100:.6f}% cosine={np.mean(BF_ABS_COS):.9f}")

print("[12/30] FP32 absolute reconstruction...")
FP_ABS_ERR=[];FP_ABS_COS=[]
for c in range(C):
    for o in range(O):
        s=INJ[(c,o)]
        rec=ACT(s["GATE"].float())*s["UP"].float()
        real=s["PRODUCT"].float()
        FP_ABS_ERR.append(relerr(rec,real))
        FP_ABS_COS.append(cos(rec,real))
print(f"FP32 absolute relErr={np.mean(FP_ABS_ERR)*100:.6f}% cosine={np.mean(FP_ABS_COS):.9f}")

print("[13/30] Vanilla absolute reconstruction...")
for mode in ["BF16","FP32"]:
    er=[];cs=[]
    for c in range(C):
        s=VAN[c]
        if mode=="BF16":
            rec=(ACT(s["GATE"])*s["UP"]).float()
        else:
            rec=ACT(s["GATE"].float())*s["UP"].float()
        real=s["PRODUCT"].float()
        er.append(relerr(rec,real));cs.append(cos(rec,real))
    print(f"{mode} vanilla absolute relErr={np.mean(er)*100:.6f}% cosine={np.mean(cs):.9f}")

print("[14/30] Build exact real product deltas...")
REAL={}
for c in range(C):
    for o in range(O):
        REAL[(c,o)]=INJ[(c,o)]["PRODUCT"].float()-VAN[c]["PRODUCT"].float()

print("[15/30] Native BF16 offline deltas...")
BF={}
for c in range(C):
    v=(ACT(VAN[c]["GATE"])*VAN[c]["UP"]).float()
    for o in range(O):
        s=INJ[(c,o)]
        x=(ACT(s["GATE"])*s["UP"]).float()
        BF[(c,o)]=x-v
er=[relerr(BF[k],REAL[k]) for k in REAL]
cs=[cos(BF[k],REAL[k]) for k in REAL]
nr=[normratio(BF[k],REAL[k]) for k in REAL]
print(f"BF16 delta relErr={np.mean(er)*100:.6f}% cosine={np.mean(cs):.7f} normRatio={np.mean(nr):.6f}")

print("[16/30] FP32 offline deltas...")
F32={}
for c in range(C):
    v=ACT(VAN[c]["GATE"].float())*VAN[c]["UP"].float()
    for o in range(O):
        s=INJ[(c,o)]
        x=ACT(s["GATE"].float())*s["UP"].float()
        F32[(c,o)]=x-v
er32=[relerr(F32[k],REAL[k]) for k in REAL]
cs32=[cos(F32[k],REAL[k]) for k in REAL]
nr32=[normratio(F32[k],REAL[k]) for k in REAL]
print(f"FP32 delta relErr={np.mean(er32)*100:.6f}% cosine={np.mean(cs32):.7f} normRatio={np.mean(nr32):.6f}")

print("[17/30] Verify TEST239 mismatch source...")
OLD={}
for c in range(C):
    vg=VAN[c]["GATE"].float()
    vu=VAN[c]["UP"].float()
    vp=ACT(vg.to(model.dtype)).float()*vu
    for o in range(O):
        g=INJ[(c,o)]["GATE"].float()
        u=INJ[(c,o)]["UP"].float()
        OLD[(c,o)]=ACT(g.to(model.dtype)).float()*u-vp
old_er=[relerr(OLD[k],REAL[k]) for k in REAL]
old_cs=[cos(OLD[k],REAL[k]) for k in REAL]
print(f"TEST239 mixed-dtype delta relErr={np.mean(old_er)*100:.6f}% cosine={np.mean(old_cs):.7f}")

print("[18/30] Mixed-dtype path decomposition...")
M1={};M2={}
for c in range(C):
    vg=VAN[c]["GATE"]
    vu=VAN[c]["UP"]
    v1=ACT(vg).float()*vu.float()
    v2=ACT(vg.float().to(model.dtype)).float()*vu.float()
    for o in range(O):
        g=INJ[(c,o)]["GATE"]
        u=INJ[(c,o)]["UP"]
        M1[(c,o)]=ACT(g).float()*u.float()-v1
        M2[(c,o)]=ACT(g.float().to(model.dtype)).float()*u.float()-v2

for name,src in [("BF16_ACT_x_FP32_UP",M1),("TEST239_CAST_PATH",M2)]:
    er=[relerr(src[k],REAL[k]) for k in REAL]
    cs=[cos(src[k],REAL[k]) for k in REAL]
    print(f"{name:24s} relErr={np.mean(er)*100:.6f}% cosine={np.mean(cs):.7f}")

print("[19/30] Real delta magnitude...")
MAG=[]
for c in range(C):
    for o in range(O):
        MAG.append(float(
            REAL[(c,o)].norm()/
            VAN[c]["PRODUCT"].float().norm().clamp_min(EPS)
        ))
print(f"mean ||realDelta||/||vanillaProduct||={np.mean(MAG)*100:.6f}%")

print("[20/30] Per-context audit...")
for c in range(C):
    eb=[];ef=[];eo=[]
    for o in range(O):
        k=(c,o)
        eb.append(relerr(BF[k],REAL[k]))
        ef.append(relerr(F32[k],REAL[k]))
        eo.append(relerr(OLD[k],REAL[k]))
    print(f"C{c+1} BF16={np.mean(eb)*100:.4f}% FP32={np.mean(ef)*100:.4f}% TEST239={np.mean(eo)*100:.4f}%")

print("[21/30] Per-object audit...")
for o in range(O):
    eb=[];ef=[];eo=[]
    for c in range(C):
        k=(c,o)
        eb.append(relerr(BF[k],REAL[k]))
        ef.append(relerr(F32[k],REAL[k]))
        eo.append(relerr(OLD[k],REAL[k]))
    print(f"O{o+1} BF16={np.mean(eb)*100:.4f}% FP32={np.mean(ef)*100:.4f}% TEST239={np.mean(eo)*100:.4f}%")

print("[22/30] Exact error energies...")
ERR_BF=torch.stack([BF[k]-REAL[k] for k in REAL]).float()
ERR_FP=torch.stack([F32[k]-REAL[k] for k in REAL]).float()
ERR_OLD=torch.stack([OLD[k]-REAL[k] for k in REAL]).float()
print(f"BF16 errorEnergy={float(ERR_BF.square().sum()):.12e}")
print(f"FP32 errorEnergy={float(ERR_FP.square().sum()):.12e}")
print(f"TEST239 errorEnergy={float(ERR_OLD.square().sum()):.12e}")

print("[23/30] Channel error concentration — zero-safe...")
def concentration(name,E):
    EN=E.square().mean(0)
    total=float(EN.sum())
    if not np.isfinite(total) or total<=EPS:
        print(f"{name}: ZERO_ERROR_ENERGY — exact reconstruction")
        return
    order=torch.argsort(EN,descending=True)
    cum=torch.cumsum(EN[order],0)/EN.sum()
    for q in [.50,.80,.90,.95,.99]:
        idx=torch.where(cum>=q)[0]
        n=int(idx[0])+1 if idx.numel() else EN.numel()
        print(f"{name} errorEnergy{int(q*100)}={n}/{EN.numel()} channels")

concentration("BF16",ERR_BF)
concentration("FP32",ERR_FP)
concentration("TEST239",ERR_OLD)

print("[24/30] Error overlap with real-delta energy...")
DX=torch.stack([REAL[k] for k in REAL]).float()
DEN=DX.square().mean(0)
def overlap(name,E):
    EN=E.square().mean(0)
    total=float(EN.sum())
    if total<=EPS:
        print(f"{name}: ZERO_ERROR — overlap not applicable")
        return
    order=torch.argsort(EN,descending=True)
    for n in [32,128,512,1024,4096]:
        idx=order[:n]
        frac=float(DEN[idx].sum()/DEN.sum().clamp_min(EPS))
        print(f"{name} topErrorChannels={n:4d} realDeltaEnergy={frac*100:.3f}%")
overlap("FP32",ERR_FP)
overlap("TEST239",ERR_OLD)

print("[25/30] Centered identity decoding...")
def center(src):
    out={}
    for c in range(C):
        m=torch.stack([src[(c,o)] for o in range(O)]).mean(0)
        for o in range(O):out[(c,o)]=src[(c,o)]-m
    return out

def decode(src):
    z=center(src);hit=0;ranks=[];margins=[]
    for hold in range(C):
        tr=[c for c in range(C) if c!=hold]
        cent=[torch.stack([z[(c,o)] for c in tr]).mean(0) for o in range(O)]
        for o in range(O):
            q=z[(hold,o)]
            sc=[cos(q,cent[j]) for j in range(O)]
            rr=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rr==1;ranks.append(rr)
            margins.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
    return hit/(C*O),float(np.mean(ranks)),float(np.mean(margins))

for name,src in [
    ("REAL",REAL),("BF16",BF),("FP32",F32),("TEST239_MIXED",OLD)
]:
    x=decode(src)
    print(f"{name:14s} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")

print("[26/30] Centered carrier agreement...")
ZR=center(REAL)
for name,src in [("BF16",BF),("FP32",F32),("TEST239",OLD)]:
    Z=center(src)
    vals=[cos(ZR[k],Z[k]) for k in ZR]
    print(f"{name:7s} centeredCarrierCos={np.mean(vals):.7f}")

print("[27/30] Repeat-forward determinism...")
det=[]
for c in range(C):
    a=continuation(prefill(QENC[c],OBJ[0]))["PRODUCT"].float()
    b=continuation(prefill(QENC[c],OBJ[0]))["PRODUCT"].float()
    det.append(relerr(a,b))
print(f"repeatForwardProductRelErr={np.mean(det)*100:.8f}%")

print("[28/30] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")

print("[29/30] Decision...")
BFERR=float(np.mean([relerr(BF[k],REAL[k]) for k in REAL]))
FPERR=float(np.mean([relerr(F32[k],REAL[k]) for k in REAL]))
OLDERR=float(np.mean([relerr(OLD[k],REAL[k]) for k in REAL]))
MAGMEAN=float(np.mean(MAG))
print(f"BF16DeltaError={BFERR*100:.6f}%")
print(f"FP32DeltaError={FPERR*100:.6f}%")
print(f"TEST239MixedDtypeError={OLDERR*100:.6f}%")
print(f"realDeltaVsAbsoluteProduct={MAGMEAN*100:.6f}%")

print("[30/30] RESULTS")
print("="*128)
print("TEST 240 RESULTS")
print("="*128)
print("BASELINE: TEST239 | PRODUCT RECONSTRUCTION AUDIT")
print(f"BF16 DELTA ERR={BFERR*100:.6f}% | FP32 DELTA ERR={FPERR*100:.6f}% | TEST239 MIXED ERR={OLDERR*100:.6f}%")
print(f"REAL DELTA / ABS PRODUCT={MAGMEAN*100:.6f}%")
print("-"*128)

if BFERR<1e-6 and OLDERR>.01:
    print("RESULT: TEST239_MISMATCH_CAUSED_BY_MIXED_DTYPE_OFFLINE_PRODUCT")
elif BFERR<.001:
    print("RESULT: NATIVE_BF16_PRODUCT_RECONSTRUCTION_VALIDATED")
elif FPERR<BFERR*.5:
    print("RESULT: FP32_PATH_REDUCES_RECONSTRUCTION_ERROR")
else:
    print("RESULT: PRODUCT_RECONSTRUCTION_MISMATCH_REQUIRES_FURTHER_AUDIT")

print("Native BF16 ACT(GATE)*UP is the reference reconstruction path.")
print("No counterfactual tensor was injected into the model.")
print("Weights unchanged.")
print("="*128)
print("TEST 240 COMPLETE")
