# ==================================================================================================
# TEST 233 — CARRIER SURVIVAL ACROSS AUTOREGRESSIVE TOKEN TRANSITION
# TEST232 WORKING BASELINE -> TEST230 OBJECT MAIN-EFFECT PACKETS -> SINGLE L08 PROMPT INJECTION
# QUESTION: DOES OBJECT IDENTITY SURVIVE WHEN THE SHARED PREFIX TOKEN " the" IS PROCESSED?
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> ... -> TEST230 -> TEST231 -> TEST232 -> TEST233
# TEST232 MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / OBJECT MAIN EFFECT PRESERVED
# PROMPT: SINGLE L08 INJECTION
# CONTINUATION " the": ZERO INTERVENTION
# MEASURE: OBJECT IDENTITY IN THE CONTINUATION TOKEN HIDDEN STATE, L00-L27
# NO TRANSPORT MAP | NO CONTROLLER | NO CONTINUATION RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=233
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
PRIMARY=.04;PROMPT_LAYERS=list(range(8,28));CONT_LAYERS=list(range(28))
SHOW=[0,1,2,4,6,8,9,10,12,16,19,20,24,27]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere",
         "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
C=len(CONTEXTS);O=len(OBJECTS)
print("="*128);print("TEST 233 — CARRIER SURVIVAL ACROSS AUTOREGRESSIVE TOKEN TRANSITION");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228 -> TEST229 -> TEST230 -> TEST231 -> TEST232 -> TEST233")
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
      layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,
      model.model.norm.weight,model.lm_head.weight]
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
print("[2/27] Build TEST232 factorial source/blind set...")
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
print("[3/27] Candidate tokens...")
CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj);CANDS.append(b if len(b)<=len(a) else a)
COMMON=CANDS[0][0]
if not all(x[0]==COMMON for x in CANDS):raise RuntimeError("No common first token.")
DISC=[x[1] for x in CANDS]
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
print("[5/27] Capture source Q/K/V...")
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
print("[6/27] Reconstruct TEST222 readout + RAW packets...")
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
print("[7/27] TEST230 factorial object-main packets...")
GRAND=torch.stack(list(RAW.values())).mean(0)
CMEAN={c:torch.stack([RAW[(c,o)] for o in range(O)]).mean(0) for c in range(C)}
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
for o in range(O):print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")
print("[8/27] Prompt prefill: vanilla + object packet...")
@torch.inference_mode()
def prefill(e,packet=None):
    pos=e.input_ids.shape[1]-1;S={};hs=[];calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float()
        d=unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY
        y[:,-1,:]=(z+d).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    ih=layers[8].register_forward_hook(inject) if packet is not None else None
    for L in PROMPT_LAYERS:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:r=model(**e,use_cache=True,return_dict=True)
    finally:
        remove(hs)
        if ih is not None:ih.remove()
    if packet is not None and calls!=1:raise RuntimeError(f"Injection calls={calls}")
    return S,r.logits[0,-1].float().detach().clone(),r.past_key_values
VANP={};VPLOG={};VPKV={};INJP={};IPLOG={};IPKV={}
for c in range(C):
    VANP[c],VPLOG[c],VPKV[c]=prefill(QENC[c],None)
    for o in range(O):
        s,l,p=prefill(QENC[c],OBJ[o]);IPLOG[(c,o)]=l;IPKV[(c,o)]=p
        for L in PROMPT_LAYERS:INJP[(c,o,L)]=s[L]
    print(f"C{c+1}: prefill complete")
print("[9/27] Verify prompt L08...")
for c in range(C):
    ds=[];cs=[]
    for o in range(O):
        d=INJP[(c,o,8)]-VANP[c][8]
        ds.append(float(d.norm()/VANP[c][8].norm())*100);cs.append(cos(d,OBJ[o]))
    print(f"C{c+1} dose={np.mean(ds):.4f}% objectCos={np.mean(cs):+.4f}")
print("[10/27] Process shared continuation token with ZERO intervention...")
@torch.inference_mode()
def continuation(past):
    S={};hs=[]
    for L in CONT_LAYERS:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,-1].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:
        r=model(input_ids=torch.tensor([[COMMON]],device=DEVICE),
                past_key_values=past,use_cache=True,return_dict=True)
    finally:remove(hs)
    return S,r.logits[0,-1].float().detach().clone(),r.past_key_values
VANC={};VCLOG={};INJC={};ICLOG={}
for c in range(C):
    VANC[c],VCLOG[c],_=continuation(VPKV[c])
    for o in range(O):
        s,l,_=continuation(IPKV[(c,o)]);ICLOG[(c,o)]=l
        for L in CONT_LAYERS:INJC[(c,o,L)]=s[L]
    print(f"C{c+1}: continuation trajectories complete")
print("[11/27] Continuation displacement...")
DISP={}
for L in CONT_LAYERS:
    a=[]
    for c in range(C):
        for o in range(O):
            d=INJC[(c,o,L)]-VANC[c][L]
            a.append(float(d.norm()/VANC[c][L].norm())*100)
    DISP[L]=float(np.mean(a))
for L in SHOW:print(f"L{L:02d} continuationDisplacement={DISP[L]:.6f}%")
print("[12/27] Center continuation responses within context...")
CENTER={}
for c in range(C):
    for L in CONT_LAYERS:
        ds=[INJC[(c,o,L)]-VANC[c][L] for o in range(O)]
        m=torch.stack(ds).mean(0)
        for o in range(O):CENTER[(c,o,L)]=ds[o]-m
print("[13/27] Leave-one-context-out object classification...")
DEC={}
PERCTX={}
for L in CONT_LAYERS:
    hit=0;ranks=[];marg=[];PERCTX[L]={}
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,L)] for c in train]).mean(0) for o in range(O)]
        hh=0;rr=[];mm=[]
        for o in range(O):
            scores=[cos(CENTER[(hold,o,L)],cent[j]) for j in range(O)]
            order=np.argsort(scores)[::-1].tolist();rank=order.index(o)+1
            mg=scores[o]-max(scores[j] for j in range(O) if j!=o)
            hit+=rank==1;ranks.append(rank);marg.append(mg)
            hh+=rank==1;rr.append(rank);mm.append(mg)
        PERCTX[L][hold]=(hh/O,float(np.mean(rr)),float(np.mean(mm)))
    DEC[L]=(hit/(C*O),float(np.mean(ranks)),float(np.mean(marg)))
for L in SHOW:
    x=DEC[L];print(f"L{L:02d} top1={x[0]*100:5.1f}% meanRank={x[1]:.3f} margin={x[2]:+.4f}")
print("[14/27] Per-held-out-context...")
for L in [0,4,8,12,16,19,24,27]:
    print(f"\nL{L:02d}")
    for c in range(C):
        x=PERCTX[L][c]
        print(f" holdC{c+1} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")
print("[15/27] Same-object cross-context invariance...")
INV={}
for L in CONT_LAYERS:
    vals=[]
    for o in range(O):
        for a in range(C):
            for b in range(a+1,C):
                vals.append(cos(CENTER[(a,o,L)],CENTER[(b,o,L)]))
    INV[L]=float(np.mean(vals))
for L in SHOW:print(f"L{L:02d} sameObjectCrossContextCos={INV[L]:+.4f}")
print("[16/27] Correct-vs-wrong object separation...")
SEP={}
for L in CONT_LAYERS:
    cor=[];wrong=[]
    for c in range(C):
        train=[x for x in range(C) if x!=c]
        cent=[torch.stack([CENTER[(x,o,L)] for x in train]).mean(0) for o in range(O)]
        for o in range(O):
            cor.append(cos(CENTER[(c,o,L)],cent[o]))
            wrong.extend(cos(CENTER[(c,o,L)],cent[j]) for j in range(O) if j!=o)
    SEP[L]=(float(np.mean(cor)),float(np.mean(wrong)),float(np.mean(cor)-np.mean(wrong)))
for L in SHOW:
    x=SEP[L];print(f"L{L:02d} correctCos={x[0]:+.4f} wrongCos={x[1]:+.4f} gap={x[2]:+.4f}")
print("[17/27] Prompt carrier -> continuation carrier alignment...")
ALIGN={}
for L in PROMPT_LAYERS:
    vals=[]
    for c in range(C):
        pds=[INJP[(c,o,L)]-VANP[c][L] for o in range(O)]
        pm=torch.stack(pds).mean(0)
        for o in range(O):
            pd=pds[o]-pm
            cd=CENTER[(c,o,L)]
            vals.append(cos(pd,cd))
    ALIGN[L]=float(np.mean(vals))
for L in [8,9,10,12,16,19,20,24,27]:
    print(f"L{L:02d} promptToContinuationCos={ALIGN[L]:+.4f}")
print("[18/27] Source OBJ -> continuation response alignment...")
SRCALIGN={}
for L in CONT_LAYERS:
    vals=[]
    for c in range(C):
        for o in range(O):vals.append(cos(OBJ[o],CENTER[(c,o,L)]))
    SRCALIGN[L]=float(np.mean(vals))
for L in SHOW:print(f"L{L:02d} sourceObjectCos={SRCALIGN[L]:+.4f}")
print("[19/27] Geometry retention from source OBJ...")
def rankdata(a):
    a=np.asarray(a);order=np.argsort(a);r=np.empty(len(a),float);r[order]=np.arange(len(a))
    vals={}
    for i,v in enumerate(a):vals.setdefault(float(v),[]).append(i)
    for q in vals.values():
        if len(q)>1:
            m=float(np.mean(r[q]))
            for i in q:r[i]=m
    return r
def spear(a,b):
    ra=rankdata(a);rb=rankdata(b)
    if np.std(ra)<1e-12 or np.std(rb)<1e-12:return 0.
    return float(np.corrcoef(ra,rb)[0,1])
srcpairs=[];idx=[]
for i in range(O):
    for j in range(i+1,O):
        srcpairs.append(cos(OBJ[i],OBJ[j]));idx.append((i,j))
GEO={}
for L in CONT_LAYERS:
    vals=[]
    for c in range(C):
        g=[cos(CENTER[(c,i,L)],CENTER[(c,j,L)]) for i,j in idx]
        vals.append(spear(srcpairs,g))
    GEO[L]=float(np.mean(vals))
for L in SHOW:print(f"L{L:02d} objectGeometrySpearman={GEO[L]:+.4f}")
print("[20/27] Actual discriminative-token output after continuation...")
OUT=[]
for c in range(C):
    base=VCLOG[c]
    for o in range(O):
        li=ICLOG[(c,o)]
        ds=[float(li[t]-base[t]) for t in DISC]
        rank=np.argsort(ds)[::-1].tolist().index(o)+1
        mg=ds[o]-max(ds[j] for j in range(O) if j!=o)
        OUT.append((c,o,rank,mg,ds[o]))
print(f"top1={sum(x[2]==1 for x in OUT)/(C*O)*100:.1f}% rank={np.mean([x[2] for x in OUT]):.3f} Δmargin={np.mean([x[3] for x in OUT]):+.4f} targetΔ={np.mean([x[4] for x in OUT]):+.4f}")
print("[21/27] Label permutation null for continuation carrier...")
rng=np.random.default_rng(SEED);PERMS=2000;NULL={}
for L in [0,4,8,12,16,19,20,24,27]:
    obs=DEC[L][0];instances=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,L)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            instances.append([cos(CENTER[(hold,o,L)],cent[j]) for j in range(O)])
    arr=np.asarray(instances);labels=np.tile(np.arange(O),C);vals=[]
    for _ in range(PERMS):
        lab=rng.permutation(labels)
        vals.append(np.mean(np.argmax(arr,axis=1)==lab))
    mu=float(np.mean(vals));sd=float(np.std(vals)+1e-12)
    z=(obs-mu)/sd;p=(1+sum(x>=obs for x in vals))/(PERMS+1)
    NULL[L]=(mu,sd,z,p)
    print(f"L{L:02d} observed={obs:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={p:.4f}")
print("[22/27] Survival curve...")
for L in range(28):
    print(f"L{L:02d} top1={DEC[L][0]*100:5.1f}% rank={DEC[L][1]:.3f} inv={INV[L]:+.4f} sep={SEP[L][2]:+.4f} disp={DISP[L]:.5f}%")
print("[23/27] Prompt vs continuation checkpoint...")
for L in [8,12,16,19,24,27]:
    print(f"L{L:02d} prompt->continuation={ALIGN[L]:+.4f} continuationTop1={DEC[L][0]*100:5.1f}% geometryρ={GEO[L]:+.4f}")
print("[24/27] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[25/27] Decision metrics...")
early=float(np.mean([DEC[L][0] for L in range(0,8)]))
mid=float(np.mean([DEC[L][0] for L in range(8,20)]))
late=float(np.mean([DEC[L][0] for L in range(20,28)]))
outtop=sum(x[2]==1 for x in OUT)/(C*O)
print(f"continuationEarlyTop1={early:.4f} mid={mid:.4f} late={late:.4f} L27={DEC[27][0]:.4f}")
print(f"outputTop1={outtop:.4f} chance={1/O:.4f}")
print("[26/27] RESULTS")
print("\n"+"="*128);print("TEST 233 RESULTS");print("="*128)
print(f"MODE: TEST230 OBJECT MAIN-EFFECT PACKETS | SINGLE PROMPT L08 INJECTION | DOSE={PRIMARY:.4f}")
print(f"AUTOREGRESSIVE TRANSITION: shared token {COMMON} {tok.decode([COMMON])!r} | ZERO CONTINUATION INTERVENTION")
print("MEASURE: OBJECT-SPECIFIC Δh ON THE NEW CONTINUATION TOKEN")
print("NO TRANSPORT MAP | NO CONTROLLER | NO CONTINUATION RE-INJECTION | WEIGHTS FROZEN")
print("\nCONTINUATION OBJECT-CARRIER")
for L in [0,4,8,12,16,19,20,24,27]:
    print(f"L{L:02d} top1={DEC[L][0]*100:5.1f}% rank={DEC[L][1]:.3f} margin={DEC[L][2]:+.4f} invariantCos={INV[L]:+.4f} geometryρ={GEO[L]:+.4f} permP={NULL[L][3]:.4f}")
print("\nOUTPUT READOUT")
print(f"discriminativeTop1={outtop*100:.1f}% meanRank={np.mean([x[2] for x in OUT]):.3f} Δmargin={np.mean([x[3] for x in OUT]):+.4f}")
print("\nINTERPRETATION GATE")
if late>=.75 and DEC[27][0]>=.50 and outtop<.25:
    print("RESULT: CARRIER_SURVIVES_TOKEN_TRANSITION_WITHOUT_READOUT — object identity crosses into the new autoregressive token but is not converted into the correct output token.")
elif mid>=.50 and late<.40:
    print("RESULT: CARRIER_DECAYS_DURING_TOKEN_TRANSITION — object identity enters the continuation computation but substantially collapses before the final layers.")
elif mid<.25:
    print("RESULT: CARRIER_FAILS_TO_TRANSFER_TO_NEW_TOKEN — prompt-side object identity does not reliably cross the autoregressive token boundary.")
else:
    print("RESULT: PARTIAL_TOKEN_TRANSITION_SURVIVAL — object identity partially survives the autoregressive transition; localization is required.")
print("-"*128)
print("Weights: PASS | Objects absent from blind queries | Only prompt L08 is intervened")
print("The continuation token is processed naturally from the intervened prompt KV cache with ZERO new activation injection.")
print("="*128);print("[27/27] TEST 233 COMPLETE")
