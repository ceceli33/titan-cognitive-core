# ==================================================================================================
# TEST 231 — OBJECT CARRIER -> LOGIT READOUT X-RAY
# TEST230 BASELINE -> FACTORIAL OBJECT MAIN-EFFECT PACKETS -> SINGLE L08 INJECTION
# QUESTION: DOES THE CROSS-CONTEXT OBJECT CARRIER ALIGN WITH THE MODEL'S REAL OUTPUT READOUT?
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> ... -> TEST228 -> TEST229 -> TEST230 -> TEST231
# TEST230 MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / OBJECT MAIN EFFECT PRESERVED
# NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN
# READOUT: captured answer-boundary hidden -> model.model.norm -> lm_head
# PRIMARY: target-object sequence log-prob under per-layer logit-lens readout
# CONTROL: VANILLA vs CORRECT OBJECT PACKET vs WRONG OBJECT PACKETS
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=231
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
PRIMARY=.04;LAYERS=list(range(8,28));SHOW=[8,9,10,12,16,19,20,24,27]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere",
         "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
C=len(CONTEXTS);O=len(OBJECTS)
print("="*128);print("TEST 231 — OBJECT CARRIER -> LOGIT READOUT X-RAY");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228 -> TEST229 -> TEST230 -> TEST231")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/25] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;NH=model.config.num_attention_heads
NKV=model.config.num_key_value_heads;HD=H//NH;GROUP=NH//NKV
if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
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
def qform(s,r):
    mp={"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}
    return f"What does {s} {mp[r]}?"
def remove(hs):
    for h in hs:h.remove()
def logsumexp(x):return torch.logsumexp(x.float(),dim=-1)
print("[2/25] Build TEST230 4x8 factorial...")
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
print("[3/25] Candidate tokenization...")
CANDS=[]
for o,obj in enumerate(OBJECTS):
    a=ids(obj);b=ids(" "+obj)
    cand=b if len(b)<=len(a) else a
    CANDS.append(cand)
    print(f"O{o+1} {obj}: ids={cand} n={len(cand)} first={tok.decode([cand[0]])!r}")
print("[4/25] RoPE...")
rotary=model.model.rotary_emb
MAXSEQ=max(max(v[0].input_ids.shape[1] for v in FMAP.values()),max(v.input_ids.shape[1] for v in QENC.values()))+4
dummy=torch.zeros(1,MAXSEQ,H,device=DEVICE,dtype=model.dtype);pos=torch.arange(MAXSEQ,device=DEVICE).unsqueeze(0)
with torch.inference_mode():COS,SIN=rotary(dummy,pos)
COS=COS[0].float();SIN=SIN[0].float()
def rotate_half(x):
    n=x.shape[-1]//2;return torch.cat((-x[...,n:],x[...,:n]),dim=-1)
def rope(x,p):return x*COS[p]+rotate_half(x)*SIN[p]
print(f"RoPE={type(rotary).__name__} max_seq={MAXSEQ}")
print("[5/25] Capture source Q/K/V...")
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
    print(f"C{c+1}: 8 source captures complete")
print("[6/25] Reconstruct TEST222 readout + RAW packets...")
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
        with torch.inference_mode():RAW[(c,o)]=layers[8].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"C{c+1}: RAW packets complete")
print("[7/25] TEST230 factorial decomposition...")
GRAND=torch.stack(list(RAW.values())).mean(0)
CMEAN={c:torch.stack([RAW[(c,o)] for o in range(O)]).mean(0) for c in range(C)}
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
CTX={c:CMEAN[c]-GRAND for c in range(C)}
INT={(c,o):RAW[(c,o)]-GRAND-CTX[c]-OBJ[o] for c in range(C) for o in range(O)}
for o in range(O):print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")
print("[8/25] Vanilla hidden trajectories...")
@torch.inference_mode()
def vanilla_hidden(e):
    pos=e.input_ids.shape[1]-1;S={};hs=[]
    for L in LAYERS:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out;S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:r=model(**e,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S,r.logits[0,-1].float().detach().clone()
VAN={};VLOG={}
for c in range(C):
    VAN[c],VLOG[c]=vanilla_hidden(QENC[c]);print(f"C{c+1} vanilla captured")
print("[9/25] Correct/wrong object injection cube...")
@torch.inference_mode()
def injected_hidden(e,packet):
    pos=e.input_ids.shape[1]-1;S={};hs=[];calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float();d=unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY
        y[:,-1,:]=(z+d).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    ih=layers[8].register_forward_hook(inject)
    for L in LAYERS:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out;S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:r=model(**e,use_cache=False,return_dict=True)
    finally:
        remove(hs);ih.remove()
    if calls!=1:raise RuntimeError(f"Injection calls={calls}")
    return S,r.logits[0,-1].float().detach().clone()
HID={};FLOG={}
for c in range(C):
    for o in range(O):
        S,l=injected_hidden(QENC[c],OBJ[o]);FLOG[(c,o)]=l
        for L in LAYERS:HID[(c,o,L)]=S[L]
    print(f"C{c+1}: 8 object interventions complete")
print("[10/25] Verify L08 intervention...")
for c in range(C):
    vals=[];cs=[]
    for o in range(O):
        d=HID[(c,o,8)]-VAN[c][8]
        vals.append(float(d.norm()/VAN[c][8].norm())*100);cs.append(cos(d,OBJ[o]))
    print(f"C{c+1} dose={np.mean(vals):.4f}% objectCos={np.mean(cs):+.4f}")
print("[11/25] Real norm + lm_head layer readout...")
@torch.inference_mode()
def lens_logits(h):
    x=model.model.norm(h.to(model.dtype).unsqueeze(0))
    return model.lm_head(x)[0].float()
LENS_V={};LENS_I={}
for c in range(C):
    for L in LAYERS:LENS_V[(c,L)]=lens_logits(VAN[c][L])
    for o in range(O):
        for L in LAYERS:LENS_I[(c,o,L)]=lens_logits(HID[(c,o,L)])
print("Layer readouts complete")
print("[12/25] First-token object readout...")
# All 8 candidates are compared using the first candidate token at each layer.
FIRST=[x[0] for x in CANDS]
FT={}
for L in LAYERS:
    hits=0;ranks=[];margins=[];deltas=[]
    for c in range(C):
        base=LENS_V[(c,L)]
        for o in range(O):
            li=LENS_I[(c,o,L)]
            scores=[float(li[t]) for t in FIRST]
            bs=[float(base[t]) for t in FIRST]
            order=np.argsort(scores)[::-1].tolist();rank=order.index(o)+1
            mg=scores[o]-max(scores[j] for j in range(O) if j!=o)
            hits+=rank==1;ranks.append(rank);margins.append(mg);deltas.append(scores[o]-bs[o])
    FT[L]=(hits/(C*O),float(np.mean(ranks)),float(np.mean(margins)),float(np.mean(deltas)))
for L in SHOW:
    x=FT[L];print(f"L{L:02d} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f} targetΔlogit={x[3]:+.4f}")
print("[13/25] Differential first-token readout...")
# Removes the vanilla query prior: classify by intervention-induced Δlogit only.
DFT={}
for L in LAYERS:
    hits=0;ranks=[];margins=[]
    for c in range(C):
        base=LENS_V[(c,L)]
        for o in range(O):
            li=LENS_I[(c,o,L)]
            scores=[float(li[t]-base[t]) for t in FIRST]
            order=np.argsort(scores)[::-1].tolist();rank=order.index(o)+1
            hits+=rank==1;ranks.append(rank)
            margins.append(scores[o]-max(scores[j] for j in range(O) if j!=o))
    DFT[L]=(hits/(C*O),float(np.mean(ranks)),float(np.mean(margins)))
for L in SHOW:
    x=DFT[L];print(f"L{L:02d} Δtop1={x[0]*100:5.1f}% rank={x[1]:.3f} Δmargin={x[2]:+.4f}")
print("[14/25] Direct object-token logit effect...")
TOKEFF={}
for L in LAYERS:
    own=[];wrong=[]
    for c in range(C):
        base=LENS_V[(c,L)]
        for o in range(O):
            d=LENS_I[(c,o,L)]-base
            own.append(float(d[FIRST[o]]))
            wrong.extend(float(d[FIRST[j]]) for j in range(O) if j!=o)
    TOKEFF[L]=(float(np.mean(own)),float(np.mean(wrong)),float(np.mean(own)-np.mean(wrong)))
for L in SHOW:
    x=TOKEFF[L];print(f"L{L:02d} ownΔ={x[0]:+.4f} wrongΔ={x[1]:+.4f} gap={x[2]:+.4f}")
print("[15/25] Final model first-token consistency...")
for c in range(C):
    base=VLOG[c]
    print(f"C{c+1}")
    for o in range(O):
        d=FLOG[(c,o)]-base
        vals=[float(d[t]) for t in FIRST];rank=np.argsort(vals)[::-1].tolist().index(o)+1
        print(f" O{o+1} targetΔ={vals[o]:+.4f} rank={rank}")
print("[16/25] True candidate sequence scorer...")
# The layer lens itself can only directly score the next token.
# For actual multi-token object probability, candidate continuation is scored through frozen KV
# after a single L08-intervened prompt prefill. No intervention occurs during candidate continuation.
@torch.inference_mode()
def prefill(c,packet=None):
    e=QENC[c];calls=0;ih=None
    if packet is not None:
        def inject(m,args,out):
            nonlocal calls
            x=out[0] if isinstance(out,tuple) else out
            if x.ndim!=3 or x.shape[1]<=1:return None
            y=x.clone();z=y[:,-1,:].float();d=unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY
            y[:,-1,:]=(z+d).to(y.dtype);calls+=1
            return (y,)+out[1:] if isinstance(out,tuple) else y
        ih=layers[8].register_forward_hook(inject)
    try:r=model(**e,use_cache=True,return_dict=True)
    finally:
        if ih is not None:ih.remove()
    if packet is not None and calls!=1:raise RuntimeError(f"Prefill injection calls={calls}")
    return r.logits[0,-1].float(),r.past_key_values
@torch.inference_mode()
def seq_lp(first_logits,past,cand):
    lp=0.;logits=first_logits;pkv=past
    for k,t in enumerate(cand):
        lp+=float(torch.log_softmax(logits.float(),dim=-1)[t])
        if k<len(cand)-1:
            x=torch.tensor([[t]],device=DEVICE)
            r=model(input_ids=x,past_key_values=pkv,use_cache=True,return_dict=True)
            logits=r.logits[0,-1].float();pkv=r.past_key_values
    return lp
SEQ={}
for c in range(C):
    vl,vp=prefill(c,None)
    v=[seq_lp(vl,vp,x) for x in CANDS]
    for o in range(O):
        il,ip=prefill(c,OBJ[o]);s=[seq_lp(il,ip,x) for x in CANDS]
        target=s[o];wrong=max(s[j] for j in range(O) if j!=o)
        vt=v[o];vw=max(v[j] for j in range(O) if j!=o)
        SEQ[(c,o)]=(target-wrong,(target-wrong)-(vt-vw),target-vt)
    print(f"C{c+1} sequence scoring complete")
print("[17/25] Behavioral sequence summary...")
marg=[SEQ[(c,o)][0] for c in range(C) for o in range(O)]
dm=[SEQ[(c,o)][1] for c in range(C) for o in range(O)]
dlp=[SEQ[(c,o)][2] for c in range(C) for o in range(O)]
print(f"meanMargin={np.mean(marg):+.4f} meanΔmargin={np.mean(dm):+.4f} meanΔtargetLP={np.mean(dlp):+.4f} improved={sum(x>0 for x in dm)}/{C*O}")
print("[18/25] Per-object behavioral summary...")
for o in range(O):
    a=[SEQ[(c,o)][1] for c in range(C)];b=[SEQ[(c,o)][2] for c in range(C)]
    print(f"O{o+1} {OBJECTS[o]} meanΔmargin={np.mean(a):+.4f} meanΔtargetLP={np.mean(b):+.4f} improved={sum(x>0 for x in a)}/{C}")
print("[19/25] Logit-lens permutation null...")
rng=np.random.default_rng(SEED);PERMS=2000;NULL={}
for L in SHOW:
    obs=DFT[L][0];vals=[]
    table=[]
    for c in range(C):
        base=LENS_V[(c,L)]
        for o in range(O):
            table.append([float(LENS_I[(c,o,L)][t]-base[t]) for t in FIRST])
    table=np.asarray(table)
    labels=np.tile(np.arange(O),C)
    for _ in range(PERMS):
        lab=rng.permutation(labels);hit=0
        for i in range(len(table)):hit+=int(int(np.argmax(table[i]))==lab[i])
        vals.append(hit/len(table))
    mu=float(np.mean(vals));sd=float(np.std(vals)+1e-12);z=(obs-mu)/sd
    p=(1+sum(x>=obs for x in vals))/(PERMS+1);NULL[L]=(mu,sd,z,p)
    print(f"L{L:02d} observed={obs:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={p:.4f}")
print("[20/25] Carrier-vs-readout alignment...")
# Compare TEST230 carrier separability with output-space selectivity.
for L in SHOW:
    print(f"L{L:02d} Δtop1={DFT[L][0]*100:5.1f}% ownWrongGap={TOKEFF[L][2]:+.4f}")
print("[21/25] Readout emergence scan...")
best=max(LAYERS,key=lambda L:DFT[L][0])
bestgap=max(LAYERS,key=lambda L:TOKEFF[L][2])
print(f"bestClassificationLayer=L{best:02d} top1={DFT[best][0]*100:.1f}% rank={DFT[best][1]:.3f}")
print(f"bestOwnWrongGapLayer=L{bestgap:02d} gap={TOKEFF[bestgap][2]:+.4f}")
print("[22/25] Physical displacement...")
for L in SHOW:
    vals=[]
    for c in range(C):
        for o in range(O):vals.append(float((HID[(c,o,L)]-VAN[c][L]).norm()/VAN[c][L].norm())*100)
    print(f"L{L:02d} meanDisplacement={np.mean(vals):.4f}%")
print("[23/25] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[24/25] RESULTS")
late=np.mean([DFT[L][0] for L in [24,25,26,27]])
lategap=np.mean([TOKEFF[L][2] for L in [24,25,26,27]])
print("\n"+"="*128);print("TEST 231 RESULTS");print("="*128)
print(f"MODE: TEST230 OBJECT MAIN-EFFECT PACKETS | SINGLE L08 INJECTION | DOSE={PRIMARY:.4f}")
print("READOUT: REAL FINAL RMSNORM + LM_HEAD APPLIED TO EACH CAPTURED LAYER")
print("DIFFERENTIAL READOUT = INJECTED LOGIT - VANILLA LOGIT")
print("NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN")
print("\nLAYERWISE OBJECT READOUT")
for L in SHOW:
    print(f"L{L:02d} Δtop1={DFT[L][0]*100:5.1f}% rank={DFT[L][1]:.3f} Δmargin={DFT[L][2]:+.4f} ownWrongGap={TOKEFF[L][2]:+.4f} permP={NULL[L][3]:.4f}")
print("\nTRUE MULTI-TOKEN CANDIDATE READOUT")
print(f"meanΔmargin={np.mean(dm):+.4f} meanΔtargetLP={np.mean(dlp):+.4f} improved={sum(x>0 for x in dm)}/{C*O}")
print("\nINTERPRETATION GATE")
if late>=.50 and lategap>0:
    print("RESULT: OBJECT_CARRIER_REACHES_OUTPUT_READOUT — late hidden carrier is selectively aligned with object-token output directions.")
elif max(DFT[L][0] for L in LAYERS)>=.50:
    print("RESULT: TRANSIENT_OBJECT_READOUT — selective object-logit alignment appears internally but is not stably retained at the final layers.")
else:
    print("RESULT: CARRIER_WITHOUT_DIRECT_LOGIT_READOUT — TEST230 object identity survives internally but is not directly decoded by the model's output head.")
print("-"*128)
print("Weights: PASS | Objects absent from blind queries | Only L08 is intervened")
print("Candidate continuation uses frozen KV after intervened prefill; no intervention during candidate continuation.")
print("="*128);print("[25/25] TEST 231 COMPLETE")


