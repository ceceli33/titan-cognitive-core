# ==================================================================================================
# TEST 235 — CAUSAL MLP CARRIER-PRESERVATION ABLATION
# WORKING BASELINE: TEST234
# TEST234 LOCALIZATION -> L19/L20 MLP CANDIDATE BOTTLENECK -> CAUSAL INTERVENTION
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> ... -> TEST233 -> TEST234 -> TEST235
# TEST234 MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / TEST230 OBJECT MAIN EFFECT PRESERVED
# PROMPT: SINGLE L08 INJECTION | CONTINUATION " the"
# BRANCHES: NORMAL | L19_GUARD | L20_GUARD | L19+L20_GUARD
# GUARD: REMOVE ONLY MLP DELTA COMPONENT ANTI-ALIGNED WITH INCOMING OBJECT CARRIER
# MEASURE: L27 CARRIER + DISCRIMINATIVE TOKEN READOUT
# NO TRANSPORT MAP | NO CONTROLLER | NO RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=235
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;KVH=0;EPS=1e-8
PRIMARY=.04;GUARD_LAYERS=[19,20];CAPTURE=list(range(8,28))
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere",
         "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
BRANCHES={"NORMAL":[],"L19_GUARD":[19],"L20_GUARD":[20],"L19_L20_GUARD":[19,20]}
C=len(CONTEXTS);O=len(OBJECTS)
print("="*128);print("TEST 235 — CAUSAL MLP CARRIER-PRESERVATION ABLATION");print("="*128)
print("WORKING BASELINE: TEST234 | LOCALIZATION: L19/L20 MLP | CAUSAL FOLLOW-UP")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/27] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;NH=model.config.num_attention_heads
NKV=model.config.num_key_value_heads;HD=H//NH;GROUP=NH//NKV
if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,
      layers[19].mlp.down_proj.weight,layers[20].mlp.down_proj.weight,
      layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):
    na=a.norm();nb=b.norm()
    if float(na)<EPS or float(nb)<EPS:return 0.
    return float(torch.dot(a,b)/(na*nb))
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def ids(x):return tok(x,add_special_tokens=False).input_ids
def subseq(hay,needle):
    if not needle:return []
    return [list(range(i,i+len(needle))) for i in range(len(hay)-len(needle)+1) if hay[i:i+len(needle)]==needle]
def last_span(full,text):
    a=subseq(full,ids(text))
    if a:return a[-1]
    a=subseq(full,ids(" "+text))
    return a[-1] if a else []
def fact_text(s,r,o):return f"Fact: {s} {r} {o}."
def qform(s,r):return f"What does {s} "+{"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r]+"?"
def remove(hs):
    for h in hs:h.remove()

print("[2/27] Build TEST234 factorial source/blind set...")
FMAP={};QENC={}
for c,(s,r) in enumerate(CONTEXTS):
    qe=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE);QENC[c]=qe
    for o,obj in enumerate(OBJECTS):
        fi=tok(chat(fact_text(s,r,obj)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
        full=fi.input_ids[0].tolist();ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,obj)
        if not ss or not rs or not os_:raise RuntimeError(f"Token map fail C{c+1} O{o+1}")
        if obj.lower() in qform(s,r).lower():raise RuntimeError("Target leakage.")
        FMAP[(c,o)]=(fi,ss,rs,os_)
    print(f"C{c+1} {s} {r} | blindSlot={qe.input_ids.shape[1]-1}")

print("[3/27] Candidate/shared-prefix tokens...")
CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj);CANDS.append(b if len(b)<=len(a) else a)
COMMON=CANDS[0][0];DISC=[x[1] for x in CANDS]
if not all(x[0]==COMMON for x in CANDS):raise RuntimeError("No common first token.")
if len(set(DISC))!=O:raise RuntimeError("Discriminative tokens not unique.")
print(f"sharedPrefixToken={COMMON} text={tok.decode([COMMON])!r}")
for o in range(O):print(f"O{o+1} {OBJECTS[o]} -> {tok.decode([DISC[o]])!r}")

print("[4/27] RoPE...")
rotary=model.model.rotary_emb
MAXSEQ=max(max(v[0].input_ids.shape[1] for v in FMAP.values()),max(v.input_ids.shape[1] for v in QENC.values()))+4
dummy=torch.zeros(1,MAXSEQ,H,device=DEVICE,dtype=model.dtype);pos=torch.arange(MAXSEQ,device=DEVICE).unsqueeze(0)
with torch.inference_mode():COS,SIN=rotary(dummy,pos)
COS=COS[0].float();SIN=SIN[0].float()
def rotate_half(x):
    n=x.shape[-1]//2;return torch.cat((-x[...,n:],x[...,:n]),dim=-1)
def rope(x,p):return x*COS[p]+rotate_half(x)*SIN[p]
print(f"RoPE={type(rotary).__name__} max_seq={MAXSEQ}")

print("[5/27] Capture TEST234 source Q/K/V...")
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
SRC={}
for c in range(C):
    for o in range(O):SRC[(c,o)]=capture_source(FMAP[(c,o)][0])
    print(f"C{c+1}: source captures complete")

print("[6/27] Reconstruct TEST222 packets...")
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
                a=attn_row(S,qp,h);rows.append((float(a[oe]),h,qp))
        v=kvh(S["V"],oe,KVH);p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32)
        for h in QGROUP:
            w=max(x[0] for x in rows if x[1]==h);p[h]=w*v
        RAW[(c,o)]=layers[8].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"C{c+1}: RAW packets complete")

print("[7/27] TEST230 object-main packets...")
GRAND=torch.stack(list(RAW.values())).mean(0)
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
for o in range(O):print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")

print("[8/27] Fresh prompt prefill factory...")
@torch.inference_mode()
def prefill(e,packet=None):
    calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float()
        y[:,-1,:]=(z+unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    h=layers[8].register_forward_hook(inject) if packet is not None else None
    try:r=model(**e,use_cache=True,return_dict=True)
    finally:
        if h is not None:h.remove()
    if packet is not None and calls!=1:raise RuntimeError(f"Injection calls={calls}")
    return r.past_key_values

print("[9/27] Measure NORMAL continuation carriers at L19/L20...")
NORMAL={}
@torch.inference_mode()
def continuation_capture(past,cap_layers=(19,20,27)):
    S={};hs=[]
    for L in cap_layers:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,-1].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:r=model(input_ids=torch.tensor([[COMMON]],device=DEVICE),past_key_values=past,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S,r.logits[0,-1].float().detach().clone()
VBASE={}
for c in range(C):
    past=prefill(QENC[c],None);VBASE[c],_=continuation_capture(past)
    for o in range(O):
        past=prefill(QENC[c],OBJ[o]);S,_=continuation_capture(past)
        for L,x in S.items():NORMAL[(c,o,L)]=x
    print(f"C{c+1}: NORMAL complete")

print("[10/27] Build incoming object-carrier directions from NORMAL responses...")
CARRIER={}
for L in GUARD_LAYERS:
    for c in range(C):
        ds=[NORMAL[(c,o,L)]-VBASE[c][L] for o in range(O)]
        mean=torch.stack(ds).mean(0)
        for o in range(O):CARRIER[(c,o,L)]=ds[o]-mean
    print(f"L{L:02d}: carrier directions ready")

print("[11/27] Causal continuation branches...")
RES={};TELEM={}
@torch.inference_mode()
def guarded_continuation(past,c,o,guard_layers):
    S={};T={};hs=[]
    for L in CAPTURE:
        def cap(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,-1].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(cap(L)))
    for L in guard_layers:
        def guard(li):
            def hk(m,args,out):
                y=out.clone()
                d=y[0,-1].float()
                g=CARRIER[(c,o,li)]
                gg=torch.dot(g,g).clamp_min(EPS)
                alpha=torch.dot(d,g)/gg
                removed=torch.zeros_like(d)
                if float(alpha)<0:
                    removed=alpha*g
                    y[0,-1]=(d-removed).to(y.dtype)
                T[li]=(float(alpha),float(removed.norm()),float(d.norm()))
                return y
            return hk
        # Hook on MLP output: removes only the component opposing incoming carrier.
        hs.append(layers[L].mlp.register_forward_hook(guard(L)))
    try:
        r=model(input_ids=torch.tensor([[COMMON]],device=DEVICE),past_key_values=past,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S,r.logits[0,-1].float().detach().clone(),T

for branch,guards in BRANCHES.items():
    print(f"  {branch} guards={guards}")
    for c in range(C):
        for o in range(O):
            past=prefill(QENC[c],OBJ[o])
            S,logits,T=guarded_continuation(past,c,o,guards)
            RES[(branch,c,o)]={"S":S,"logits":logits}
            for L,v in T.items():TELEM[(branch,c,o,L)]=v
        print(f"    C{c+1} complete")

print("[12/27] Fresh vanilla continuation references...")
VAN={}
for c in range(C):
    past=prefill(QENC[c],None);S,logits=continuation_capture(past,tuple(CAPTURE))
    VAN[c]={"S":S,"logits":logits}

print("[13/27] Build branch deltas + context centering...")
CENTER={}
for branch in BRANCHES:
    for L in CAPTURE:
        for c in range(C):
            ds=[RES[(branch,c,o)]["S"][L]-VAN[c]["S"][L] for o in range(O)]
            m=torch.stack(ds).mean(0)
            for o in range(O):CENTER[(branch,c,o,L)]=ds[o]-m

print("[14/27] Leave-one-context-out carrier decoding...")
DEC={}
for branch in BRANCHES:
    for L in CAPTURE:
        hit=0;ranks=[];margins=[]
        for hold in range(C):
            train=[c for c in range(C) if c!=hold]
            cent=[torch.stack([CENTER[(branch,c,o,L)] for c in train]).mean(0) for o in range(O)]
            for o in range(O):
                sc=[cos(CENTER[(branch,hold,o,L)],cent[j]) for j in range(O)]
                rank=np.argsort(sc)[::-1].tolist().index(o)+1
                hit+=rank==1;ranks.append(rank)
                margins.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
        DEC[(branch,L)]=(hit/(C*O),float(np.mean(ranks)),float(np.mean(margins)))
for branch in BRANCHES:
    print(branch)
    for L in [18,19,20,21,22,24,26,27]:
        x=DEC[(branch,L)]
        print(f" L{L:02d} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")

print("[15/27] L27 same-object invariance...")
INV={}
for branch in BRANCHES:
    for L in [19,20,21,24,27]:
        vals=[]
        for o in range(O):
            for a in range(C):
                for b in range(a+1,C):
                    vals.append(cos(CENTER[(branch,a,o,L)],CENTER[(branch,b,o,L)]))
        INV[(branch,L)]=float(np.mean(vals))
    print(branch," ".join(f"L{L}={INV[(branch,L)]:+.4f}" for L in [19,20,21,24,27]))

print("[16/27] Correct-vs-wrong separation...")
SEP={}
for branch in BRANCHES:
    for L in [19,20,21,24,27]:
        cor=[];wr=[]
        for hold in range(C):
            train=[c for c in range(C) if c!=hold]
            cent=[torch.stack([CENTER[(branch,c,o,L)] for c in train]).mean(0) for o in range(O)]
            for o in range(O):
                cor.append(cos(CENTER[(branch,hold,o,L)],cent[o]))
                wr.extend(cos(CENTER[(branch,hold,o,L)],cent[j]) for j in range(O) if j!=o)
        SEP[(branch,L)]=float(np.mean(cor)-np.mean(wr))
    print(branch," ".join(f"L{L}={SEP[(branch,L)]:+.4f}" for L in [19,20,21,24,27]))

print("[17/27] Guard telemetry...")
for branch in ["L19_GUARD","L20_GUARD","L19_L20_GUARD"]:
    print(branch)
    for L in BRANCHES[branch]:
        vals=[TELEM[(branch,c,o,L)] for c in range(C) for o in range(O)]
        trig=sum(v[0]<0 for v in vals)
        print(f" L{L:02d} trigger={trig}/{C*O} meanAlpha={np.mean([v[0] for v in vals]):+.6f} meanRemovedNorm={np.mean([v[1] for v in vals]):.6f} mlpNorm={np.mean([v[2] for v in vals]):.6f}")

print("[18/27] Actual discriminative-token readout...")
READ={}
for branch in BRANCHES:
    rows=[]
    for c in range(C):
        base=VAN[c]["logits"]
        for o in range(O):
            li=RES[(branch,c,o)]["logits"]
            ds=[float(li[t]-base[t]) for t in DISC]
            rank=np.argsort(ds)[::-1].tolist().index(o)+1
            rows.append((rank,ds[o]-max(ds[j] for j in range(O) if j!=o),ds[o]))
    READ[branch]=(sum(x[0]==1 for x in rows)/(C*O),float(np.mean([x[0] for x in rows])),
                  float(np.mean([x[1] for x in rows])),float(np.mean([x[2] for x in rows])))
    x=READ[branch]
    print(f"{branch:15s} top1={x[0]*100:5.1f}% rank={x[1]:.3f} Δmargin={x[2]:+.4f} targetΔ={x[3]:+.4f}")

print("[19/27] Absolute discriminative-token choice...")
ABS={}
for branch in BRANCHES:
    rows=[]
    for c in range(C):
        for o in range(O):
            li=RES[(branch,c,o)]["logits"]
            sc=[float(li[t]) for t in DISC]
            rank=np.argsort(sc)[::-1].tolist().index(o)+1
            rows.append(rank)
    ABS[branch]=(sum(r==1 for r in rows)/(C*O),float(np.mean(rows)))
    print(f"{branch:15s} absoluteTop1={ABS[branch][0]*100:5.1f}% rank={ABS[branch][1]:.3f}")

print("[20/27] Causal restoration relative to NORMAL...")
base27=DEC[("NORMAL",27)][0];base19=DEC[("NORMAL",19)][0];base20=DEC[("NORMAL",20)][0]
for branch in ["L19_GUARD","L20_GUARD","L19_L20_GUARD"]:
    print(f"{branch:15s} ΔL19={(DEC[(branch,19)][0]-base19)*100:+5.1f}pp ΔL20={(DEC[(branch,20)][0]-base20)*100:+5.1f}pp ΔL27={(DEC[(branch,27)][0]-base27)*100:+5.1f}pp Δreadout={(READ[branch][0]-READ['NORMAL'][0])*100:+5.1f}pp")

print("[21/27] Permutation null at L27...")
rng=np.random.default_rng(SEED);PERMS=2000
for branch in BRANCHES:
    rows=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(branch,c,o,27)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):rows.append([cos(CENTER[(branch,hold,o,27)],cent[j]) for j in range(O)])
    arr=np.asarray(rows);labels=np.tile(np.arange(O),C);obs=DEC[(branch,27)][0];null=[]
    for _ in range(PERMS):
        lab=rng.permutation(labels);null.append(float(np.mean(np.argmax(arr,axis=1)==lab)))
    mu=float(np.mean(null));sd=float(np.std(null)+1e-12);z=(obs-mu)/sd
    p=(1+sum(x>=obs for x in null))/(PERMS+1)
    print(f"{branch:15s} obs={obs:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={p:.4f}")

print("[22/27] Guard magnitude sanity...")
for branch in ["L19_GUARD","L20_GUARD","L19_L20_GUARD"]:
    total=[];ratio=[]
    for c in range(C):
        for o in range(O):
            for L in BRANCHES[branch]:
                a,r,n=TELEM[(branch,c,o,L)]
                total.append(r);ratio.append(r/(n+EPS))
    print(f"{branch:15s} removedNorm={np.mean(total):.6f} removed/mlp={np.mean(ratio)*100:.3f}%")

print("[23/27] Branch survival curves...")
for L in CAPTURE:
    print(f"L{L:02d} "+" | ".join(f"{b}={DEC[(b,L)][0]*100:5.1f}%" for b in BRANCHES))

print("[24/27] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")

print("[25/27] Decision metrics...")
best=max(["L19_GUARD","L20_GUARD","L19_L20_GUARD"],key=lambda b:DEC[(b,27)][0])
restore=(DEC[(best,27)][0]-DEC[("NORMAL",27)][0])
readgain=READ[best][0]-READ["NORMAL"][0]
print(f"bestGuard={best}")
print(f"normalL27={DEC[('NORMAL',27)][0]:.4f} bestL27={DEC[(best,27)][0]:.4f} restoration={restore*100:+.1f}pp")
print(f"normalReadout={READ['NORMAL'][0]:.4f} bestReadout={READ[best][0]:.4f} gain={readgain*100:+.1f}pp")

print("[26/27] RESULTS")
print("="*128);print("TEST 235 RESULTS");print("="*128)
print("BASELINE: TEST234 | SINGLE PROMPT L08 OBJECT-MAIN PACKET | 4% DOSE | CONTINUATION ' the'")
print("CAUSAL ABLATION: REMOVE ONLY ANTI-CARRIER COMPONENT OF L19/L20 MLP OUTPUT")
print("NO TRANSPORT MAP | NO CONTROLLER | NO CONTINUATION RE-INJECTION | WEIGHTS FROZEN")
for branch in BRANCHES:
    d=DEC[(branch,27)];r=READ[branch]
    print(f"{branch:15s} L27top1={d[0]*100:5.1f}% L27rank={d[1]:.3f} L27margin={d[2]:+.4f} | readout={r[0]*100:5.1f}% rank={r[1]:.3f} Δmargin={r[2]:+.4f}")
print("-"*128)
if restore>=.15 and readgain>=.10:
    print("RESULT: MLP_CAUSAL_BOTTLENECK_WITH_READOUT_RESCUE")
elif restore>=.15:
    print("RESULT: MLP_CAUSAL_CARRIER_BOTTLENECK_WITHOUT_READOUT_RESCUE")
elif restore>=.05:
    print("RESULT: PARTIAL_MLP_CAUSAL_CONTRIBUTION")
else:
    print("RESULT: NO_STRONG_CAUSAL_RESCUE_FROM_L19_L20_MLP_GUARD")
print(f"BEST: {best} | L27 restoration={restore*100:+.1f}pp | readout gain={readgain*100:+.1f}pp")
print("Interpretation is restricted to this packet family, dose, contexts and anti-carrier projection intervention.")
print("="*128)
print("[27/27] TEST 235 COMPLETE")
