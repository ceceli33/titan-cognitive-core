# ==================================================================================================
# TEST 230 — OBJECT IDENTITY FACTORIAL CROSSING
# TEST229 BASELINE -> SAME OBJECT ACROSS DIFFERENT SUBJECT/RELATION CONTEXTS
# TRAIN CONTEXTS -> HELD-OUT CONTEXT -> DOES DOWNSTREAM Δh FOLLOW OBJECT IDENTITY?
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> ... -> TEST227 -> TEST228 -> TEST229 -> TEST230
# TEST229 MODEL / SYSTEM / L08 TEST222 PACKET FORGE / COMMON SUBTRACTION PRESERVED
# 4 CONTEXTS x 8 OBJECTS = 32 SOURCE FACTS
# LEAVE-ONE-CONTEXT-OUT: TRAIN 3 CONTEXTS -> TEST UNSEEN CONTEXT
# NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=230
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
PRIMARY=.04;LAYERS=list(range(8,28));SHOW=[8,9,10,12,16,19,20,24,27]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere",
         "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
C=len(CONTEXTS);O=len(OBJECTS)
print("="*128);print("TEST 230 — OBJECT IDENTITY FACTORIAL CROSSING");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228 -> TEST229 -> TEST230")
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
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
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
def pearson(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);a-=a.mean();b-=b.mean()
    d=np.sqrt(np.dot(a,a)*np.dot(b,b));return float(np.dot(a,b)/d) if d>1e-15 else 0.
def rankdata(x):
    x=np.asarray(x,dtype=np.float64);order=np.argsort(x,kind="mergesort");r=np.empty(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        v=(i+j-1)/2+1;r[order[i:j]]=v;i=j
    return r
def spearman(a,b):return pearson(rankdata(a),rankdata(b))
def cosine_matrix(vecs):
    X=torch.stack([unit(v) for v in vecs]).float();return (X@X.T).cpu().numpy()
def upper(G):return np.asarray([G[i,j] for i in range(O) for j in range(i+1,O)])
print("[2/25] Build 4x8 factorial source set...")
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
print("[3/25] Object token lengths...")
for o,obj in enumerate(OBJECTS):
    lens=[len(FMAP[(c,o)][3]) for c in range(C)]
    print(f"O{o+1} {obj}: tokenLens={lens}")
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
print("[6/25] Reconstruct TEST222 readout and forge RAW packets...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    kh=qhead//GROUP;q=rope(qh(S["Q"],qpos,qhead),qpos)
    K=torch.stack([rope(kvh(S["K"],p,kh),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)
RAW={};READ={}
for c in range(C):
    for o in range(O):
        S=SRC[(c,o)];oe=FMAP[(c,o)][3][-1];rows=[]
        for h in QGROUP:
            for qp in range(oe,FMAP[(c,o)][0].input_ids.shape[1]):
                a=attn_row(S,qp,h);rows.append((float(a[oe]),h,qp))
        rows.sort(key=lambda z:z[0],reverse=True);READ[(c,o)]=rows
        v=kvh(S["V"],oe,KVH).clone();p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32)
        for h in QGROUP:
            w=max(x[0] for x in rows if x[1]==h);p[h]=w*v
        with torch.inference_mode():RAW[(c,o)]=layers[8].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"C{c+1}: RAW packets complete")
print("[7/25] Factorial decomposition: grand + context + object + interaction...")
GRAND=torch.stack(list(RAW.values())).mean(0)
CMEAN={c:torch.stack([RAW[(c,o)] for o in range(O)]).mean(0) for c in range(C)}
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
CTX={c:CMEAN[c]-GRAND for c in range(C)}
INT={(c,o):RAW[(c,o)]-GRAND-CTX[c]-OBJ[o] for c in range(C) for o in range(O)}
for o in range(O):
    print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f} rawMeanNorm={OMEAN[o].norm():.4f}")
print("[8/25] Variance/energy decomposition...")
raw_center=torch.stack([RAW[(c,o)]-GRAND for c in range(C) for o in range(O)])
obj_e=sum(float(OBJ[o].pow(2).sum()) for c in range(C) for o in range(O))
ctx_e=sum(float(CTX[c].pow(2).sum()) for c in range(C) for o in range(O))
int_e=sum(float(INT[(c,o)].pow(2).sum()) for c in range(C) for o in range(O))
tot=float(raw_center.pow(2).sum())
print(f"centeredEnergy={tot:.6f} object={obj_e/tot:.4f} context={ctx_e/tot:.4f} interaction={int_e/tot:.4f}")
print("[9/25] Source object geometry...")
GOBJ=cosine_matrix([OBJ[o] for o in range(O)]);GOBJU=upper(GOBJ)
print(f"meanOffDiag={GOBJU.mean():+.4f} min={GOBJU.min():+.4f} max={GOBJU.max():+.4f}")
for o in range(O):print("O%d "%(o+1)+" ".join(f"{GOBJ[o,j]:+.3f}" for j in range(O)))
print("[10/25] Vanilla blind trajectories...")
@torch.inference_mode()
def vanilla_hidden(e):
    pos=e.input_ids.shape[1]-1;S={};hs=[]
    for L in LAYERS:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out;S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S
VAN={c:vanilla_hidden(QENC[c]) for c in range(C)}
print("4 vanilla blind trajectories captured")
print("[11/25] Inject OBJECT main-effect packets into all blind contexts...")
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
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        remove(hs);ih.remove()
    if calls!=1:raise RuntimeError(f"Injection calls={calls}")
    return S
DELTA={}
for c in range(C):
    for o in range(O):
        S=injected_hidden(QENC[c],OBJ[o])
        for L in LAYERS:DELTA[(c,o,L)]=S[L]-VAN[c][L]
    print(f"C{c+1}: 8 object packets complete")
print("[12/25] Verify L08...")
for c in range(C):
    cs=[];ds=[]
    for o in range(O):
        d=DELTA[(c,o,8)];cs.append(cos(d,OBJ[o]));ds.append(float(d.norm()/VAN[c][8].norm())*100)
    print(f"C{c+1} objectCos={np.mean(cs):+.4f} dose={np.mean(ds):.4f}%")
print("[13/25] Center downstream responses within context...")
CENTER={}
for c in range(C):
    for L in LAYERS:
        mu=torch.stack([DELTA[(c,o,L)] for o in range(O)]).mean(0)
        for o in range(O):CENTER[(c,o,L)]=DELTA[(c,o,L)]-mu
print("[14/25] Leave-one-context-out object decoder...")
LOO={}
for L in LAYERS:
    hits=0;ranks=[];margins=[];per=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=torch.stack([torch.stack([CENTER[(c,o,L)] for c in train]).mean(0) for o in range(O)])
        hh=0;rr=[];mm=[]
        for o in range(O):
            t=CENTER[(hold,o,L)]
            scores=[cos(t,cent[j]) for j in range(O)]
            order=np.argsort(scores)[::-1].tolist();rank=order.index(o)+1
            mg=scores[o]-max(scores[j] for j in range(O) if j!=o)
            hits+=rank==1;hh+=rank==1;ranks.append(rank);rr.append(rank);margins.append(mg);mm.append(mg)
        per.append((hh/O,float(np.mean(rr)),float(np.mean(mm))))
    LOO[L]=(hits/(C*O),float(np.mean(ranks)),float(np.mean(margins)),per)
for L in SHOW:
    x=LOO[L];print(f"L{L:02d} top1={x[0]*100:5.1f}% meanRank={x[1]:.3f} margin={x[2]:+.4f}")
print("[15/25] Per-held-out-context...")
for L in [8,12,16,19,20,24,27]:
    print(f"\nL{L:02d}")
    for c,x in enumerate(LOO[L][3]):
        print(f" holdC{c+1} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")
print("[16/25] Object geometry retention...")
GEO={}
for L in LAYERS:
    vals=[]
    for c in range(C):
        G=cosine_matrix([CENTER[(c,o,L)] for o in range(O)])
        vals.append(spearman(GOBJU,upper(G)))
    GEO[L]=float(np.mean(vals))
for L in SHOW:print(f"L{L:02d} objectGeometrySpearman={GEO[L]:+.4f}")
print("[17/25] Context-invariance of each object response...")
INV={}
for L in LAYERS:
    vals=[]
    for o in range(O):
        for a in range(C):
            for b in range(a+1,C):vals.append(cos(CENTER[(a,o,L)],CENTER[(b,o,L)]))
    INV[L]=float(np.mean(vals))
for L in SHOW:print(f"L{L:02d} sameObjectCrossContextCos={INV[L]:+.4f}")
print("[18/25] Same-context wrong-object separation...")
SEP={}
for L in LAYERS:
    same=[];wrong=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,L)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            same.append(cos(CENTER[(hold,o,L)],cent[o]))
            wrong.extend(cos(CENTER[(hold,o,L)],cent[j]) for j in range(O) if j!=o)
    SEP[L]=(float(np.mean(same)),float(np.mean(wrong)),float(np.mean(same)-np.mean(wrong)))
for L in SHOW:
    x=SEP[L];print(f"L{L:02d} correctCos={x[0]:+.4f} wrongCos={x[1]:+.4f} gap={x[2]:+.4f}")
print("[19/25] Label permutation null...")
rng=np.random.default_rng(SEED);PERMS=1000;NULL={}
for L in SHOW:
    obs=LOO[L][0];vals=[]
    for _ in range(PERMS):
        hit=0
        for hold in range(C):
            train=[c for c in range(C) if c!=hold]
            cent=torch.stack([torch.stack([CENTER[(c,o,L)] for c in train]).mean(0) for o in range(O)])
            perm=rng.permutation(O);cent=cent[torch.tensor(perm,device=cent.device)]
            for o in range(O):
                scores=[cos(CENTER[(hold,o,L)],cent[j]) for j in range(O)]
                hit+=int(int(np.argmax(scores))==o)
        vals.append(hit/(C*O))
    mu=float(np.mean(vals));sd=float(np.std(vals)+1e-12);z=(obs-mu)/sd
    p=(1+sum(x>=obs for x in vals))/(PERMS+1);NULL[L]=(mu,sd,z,p)
    print(f"L{L:02d} observed={obs:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={p:.4f}")
print("[20/25] Interaction-packet control...")
# Inject context-specific interaction instead of object main effect.
# If decoding follows interaction/fact fingerprint rather than object identity, this control may compete with OBJ.
ICTRL={}
for hold in range(C):
    for o in range(O):
        S=injected_hidden(QENC[hold],INT[(hold,o)])
        for L in SHOW:ICTRL[(hold,o,L)]=S[L]-VAN[hold][L]
for L in SHOW:
    hits=0;ranks=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,L)] for c in train]).mean(0) for o in range(O)]
        mu=torch.stack([ICTRL[(hold,o,L)] for o in range(O)]).mean(0)
        for o in range(O):
            t=ICTRL[(hold,o,L)]-mu;scores=[cos(t,cent[j]) for j in range(O)]
            order=np.argsort(scores)[::-1].tolist();rank=order.index(o)+1
            hits+=rank==1;ranks.append(rank)
    print(f"L{L:02d} interactionControlTop1={hits/(C*O)*100:5.1f}% rank={np.mean(ranks):.3f}")
print("[21/25] Physical displacement...")
for L in SHOW:
    vals=[float(DELTA[(c,o,L)].norm()/VAN[c][L].norm())*100 for c in range(C) for o in range(O)]
    print(f"L{L:02d} meanDisplacement={np.mean(vals):.4f}%")
print("[22/25] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[23/25] Decision metrics...")
EARLY=np.mean([LOO[L][0] for L in [8,9,10,12]])
MID=np.mean([LOO[L][0] for L in [16,17,18,19,20]])
LATE=np.mean([LOO[L][0] for L in [24,25,26,27]])
LATERANK=np.mean([LOO[L][1] for L in [24,25,26,27]])
print(f"earlyTop1={EARLY:.4f} midTop1={MID:.4f} lateTop1={LATE:.4f} lateMeanRank={LATERANK:.3f} chance={1/O:.4f}")
print("[24/25] RESULTS")
print("\n"+"="*128);print("TEST 230 RESULTS");print("="*128)
print(f"MODE: 4 CONTEXTS x 8 OBJECTS | OBJECT MAIN-EFFECT PACKET | SINGLE L08 INJECTION | DOSE={PRIMARY:.4f}")
print("LEAVE-ONE-CONTEXT-OUT: TRAIN 3 CONTEXTS -> TEST UNSEEN SUBJECT/RELATION CONTEXT")
print("NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN")
print("\nSTRICT HELD-OUT-CONTEXT OBJECT CLASSIFICATION")
for L in SHOW:
    x=LOO[L]
    print(f"L{L:02d} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f} geometryρ={GEO[L]:+.4f} invariantCos={INV[L]:+.4f} permP={NULL[L][3]:.4f}")
print("\nFACTORIAL SOURCE ENERGY")
print(f"OBJECT={obj_e/tot:.4f} CONTEXT={ctx_e/tot:.4f} INTERACTION={int_e/tot:.4f}")
print("\nINTERPRETATION GATE")
if LATE>=.75 and LATERANK<=1.75:
    print("RESULT: CROSS_CONTEXT_OBJECT_IDENTITY — downstream carrier follows object identity across unseen subject/relation contexts.")
elif LATE>=.40 and LATERANK<3.0:
    print("RESULT: PARTIAL_OBJECT_IDENTITY_GENERALIZATION — object identity is measurable across contexts but degrades downstream.")
else:
    print("RESULT: FACT_PACKET_FINGERPRINT_DOMINANT — held-out context does not support robust object-identity generalization.")
print("-"*128)
print("Weights: PASS | Objects absent from blind queries | Only L08 is intervened")
print("L09-L27 responses are endogenous downstream consequences of the single L08 object-main-effect intervention.")
print("="*128);print("[25/25] TEST 230 COMPLETE")
