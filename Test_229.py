# ==================================================================================================
# TEST 229 — LEAVE-ONE-QUERY-OUT IDENTITY SUBSPACE GENERALIZATION
# TEST228 BASELINE -> RESIDUAL PACKETS -> SINGLE L08 INJECTION -> FREE L09-L27 PROPAGATION
# TRAIN 7 BLIND QUERIES -> BUILD LOW-RANK IDENTITY SUBSPACE -> TEST ON HELD-OUT QUERY
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223
#          -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228 -> TEST229
# TEST228 MODEL / SYSTEM / FACTS / TEST222 PACKET FORGE / COMMON SUBTRACTION PRESERVED
# NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=229
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
PRIMARY=.04;LAYERS=list(range(8,28));SHOW=[8,9,10,12,16,19,20,24,27];RANKS=[1,2,3,4,6,7]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[
("Rovan Tesk","keeps","the amber compass"),
("Mira Veln","carries","the silver lantern"),
("Dalen Quor","owns","the violet key"),
("Sorin Kelm","guards","the bronze sphere"),
("Varek Tonn","holds","the golden necklace"),
("Lira Mesk","stores","the iron dagger"),
("Korin Drel","protects","the crystal mirror"),
("Taren Vosk","carries","the wooden mask")]
M=len(FACTS)
print("="*128);print("TEST 229 — LEAVE-ONE-QUERY-OUT IDENTITY SUBSPACE GENERALIZATION");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228 -> TEST229")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/24] Model...")
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
    mp={"keeps":"keep","carries":"carry","owns":"own","guards":"guard","holds":"hold","stores":"store","protects":"protect"}
    return f"What does {s} {mp[r]}?"
def remove(hs):
    for h in hs:h.remove()
def cosine_matrix(vecs):
    X=torch.stack([unit(v) for v in vecs]).float()
    return (X@X.T).cpu().numpy()
def upper(G):return np.asarray([G[i,j] for i in range(M) for j in range(i+1,M)],dtype=np.float64)
def pearson(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    a=a-a.mean();b=b-b.mean();d=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/d) if d>1e-15 else 0.
def rankdata(x):
    x=np.asarray(x,dtype=np.float64);o=np.argsort(x,kind="mergesort");r=np.empty(len(x),dtype=np.float64);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[o[j]]==x[o[i]]:j+=1
        v=(i+j-1)/2+1
        for k in range(i,j):r[o[k]]=v
        i=j
    return r
def spearman(a,b):return pearson(rankdata(a),rankdata(b))
print("[2/24] Token maps...")
FMAP={};QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    fi=tok(chat(fact_text(s,r,o)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    full=fi.input_ids[0].tolist();ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Token map fail Q{qi+1}")
    qe=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if o.lower() in qform(s,r).lower():raise RuntimeError("Target leakage.")
    FMAP[qi]=(fi,ss,rs,os_);QENC.append(qe)
    print(f"Q{qi+1} FACT subject={ss} relation={rs} object={os_} | BLIND slot={qe.input_ids.shape[1]-1}")
print("[3/24] RoPE...")
rotary=model.model.rotary_emb
MAXSEQ=max(max(x[0].input_ids.shape[1] for x in FMAP.values()),max(x.input_ids.shape[1] for x in QENC))+4
dummy=torch.zeros(1,MAXSEQ,H,device=DEVICE,dtype=model.dtype);pos=torch.arange(MAXSEQ,device=DEVICE).unsqueeze(0)
with torch.inference_mode():COS,SIN=rotary(dummy,pos)
COS=COS[0].float();SIN=SIN[0].float()
def rotate_half(x):
    n=x.shape[-1]//2
    return torch.cat((-x[...,n:],x[...,:n]),dim=-1)
def rope(x,p):return x*COS[p]+rotate_half(x)*SIN[p]
print(f"RoPE={type(rotary).__name__} max_seq={MAXSEQ}")
print("[4/24] Capture TEST222 source Q/K/V...")
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
for i in range(M):
    SRC[i]=capture_source(FMAP[i][0]);print(f"Q{i+1} captured")
print("[5/24] Reconstruct TEST222 source readout...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    kh=qhead//GROUP;q=rope(qh(S["Q"],qpos,qhead),qpos)
    K=torch.stack([rope(kvh(S["K"],p,kh),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)
READ={};RAWV={}
for i in range(M):
    oe=FMAP[i][3][-1];rows=[]
    for h in QGROUP:
        for qp in range(oe,FMAP[i][0].input_ids.shape[1]):
            a=attn_row(SRC[i],qp,h);rows.append((float(a[oe]),h,qp))
    rows.sort(key=lambda z:z[0],reverse=True);READ[i]=rows;RAWV[i]=kvh(SRC[i]["V"],oe,KVH).clone()
    b=rows[0];print(f"Q{i+1} bestQH={b[1]:02d} qpos={b[2]} objEndAttn={b[0]:.6f}")
print("[6/24] Forge TEST222 RAW packets...")
RAW={}
for i in range(M):
    p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32)
    for h in QGROUP:
        w=max(x[0] for x in READ[i] if x[1]==h);p[h]=w*RAWV[i]
    with torch.inference_mode():RAW[i]=layers[8].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"Q{i+1} rawNorm={RAW[i].norm():.4f}")
print("[7/24] TEST227 residual packets...")
COMMON=torch.stack([RAW[i] for i in range(M)]).mean(0)
RES={i:RAW[i]-COMMON for i in range(M)}
G0=cosine_matrix([RES[i] for i in range(M)]);G0U=upper(G0)
for i in range(M):print(f"Q{i+1} residualNorm={RES[i].norm():.4f}")
print(f"L08 source geometry meanOffDiag={G0U.mean():+.4f}")
print("[8/24] Vanilla blind trajectories...")
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
VAN={}
for q in range(M):
    VAN[q]=vanilla_hidden(QENC[q]);print(f"Q{q+1} vanilla captured")
print("[9/24] Full query x packet response cube...")
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
for q in range(M):
    for p in range(M):
        S=injected_hidden(QENC[q],RES[p])
        for L in LAYERS:DELTA[(q,p,L)]=S[L]-VAN[q][L]
    print(f"Query Q{q+1}: 8 packets complete")
print("[10/24] Verify L08 cube...")
for q in range(M):
    vals=[cos(DELTA[(q,p,8)],RES[p]) for p in range(M)]
    rel=[float(DELTA[(q,p,8)].norm()/VAN[q][8].norm().clamp_min(EPS))*100 for p in range(M)]
    print(f"Q{q+1} packetCos={np.mean(vals):+.4f} dose={np.mean(rel):.4f}%")
print("[11/24] Center responses within each query...")
# Query-specific common response is removed before learning the identity subspace.
CENTER={}
for q in range(M):
    for L in LAYERS:
        mu=torch.stack([DELTA[(q,p,L)] for p in range(M)]).mean(0)
        for p in range(M):CENTER[(q,p,L)]=DELTA[(q,p,L)]-mu
print("[12/24] LOO subspace extraction...")
# For each held-out query and layer:
# train on 7 queries. Each packet identity has a train-query mean response.
# SVD is performed on the 8 identity centroids. Held-out query never enters SVD.
BASES={};CENTROIDS={}
for hold in range(M):
    train=[q for q in range(M) if q!=hold]
    for L in LAYERS:
        C=torch.stack([torch.stack([CENTER[(q,p,L)] for q in train]).mean(0) for p in range(M)])
        C=C-C.mean(0,keepdim=True)
        U,S,Vh=torch.linalg.svd(C,full_matrices=False)
        BASES[(hold,L)]=Vh[:7].T.contiguous()
        CENTROIDS[(hold,L)]=C
print("LOO bases ready.")
print("[13/24] Explained train identity energy...")
for L in SHOW:
    vals=[]
    for hold in range(M):
        C=CENTROIDS[(hold,L)];B=BASES[(hold,L)]
        total=float((C*C).sum())
        row=[]
        for r in RANKS:
            rr=min(r,B.shape[1]);P=C@B[:,:rr];row.append(float((P*P).sum())/(total+EPS))
        vals.append(row)
    a=np.mean(vals,axis=0)
    print(f"L{L:02d} "+" ".join(f"r{r}={a[k]:.4f}" for k,r in enumerate(RANKS)))
print("[14/24] LOO identity classification...")
# Classification is performed only inside the train-derived subspace.
# Train identity centroids are projected; held-out query packet responses are projected and matched by cosine.
LOO={}
for L in LAYERS:
    LOO[L]={}
    for r in RANKS:
        ranks=[];margins=[];hits=0
        for hold in range(M):
            B=BASES[(hold,L)];rr=min(r,B.shape[1]);B=B[:,:rr]
            C=CENTROIDS[(hold,L)]@B
            T=torch.stack([CENTER[(hold,p,L)] for p in range(M)])@B
            for p in range(M):
                scores=[cos(T[p],C[j]) for j in range(M)]
                order=np.argsort(scores)[::-1].tolist();rank=order.index(p)+1
                hits+=rank==1;ranks.append(rank);margins.append(scores[p]-max(scores[j] for j in range(M) if j!=p))
        LOO[L][r]=(hits/(M*M),float(np.mean(ranks)),float(np.mean(margins)))
for L in SHOW:
    print(f"L{L:02d}",end="")
    for r in [1,2,3,4,6,7]:
        x=LOO[L][r];print(f" r{r}:{x[0]*100:5.1f}%/rank{x[1]:.2f}/m{x[2]:+.3f}",end="")
    print()
print("[15/24] Choose rank using TRAIN ONLY...")
# Rank selection uses mean leave-one-query TRAIN reconstruction/class separation, not held-out labels.
# Fixed rule: smallest rank reaching >=95% of train identity energy; capped at 7.
SELECT={}
for hold in range(M):
    for L in LAYERS:
        C=CENTROIDS[(hold,L)];B=BASES[(hold,L)];tot=float((C*C).sum());chosen=7
        for r in range(1,8):
            P=C@B[:,:r]
            if float((P*P).sum())/(tot+EPS)>=.95:
                chosen=r;break
        SELECT[(hold,L)]=chosen
for L in SHOW:
    a=[SELECT[(h,L)] for h in range(M)]
    print(f"L{L:02d} selectedRanks={a} mean={np.mean(a):.2f}")
print("[16/24] Strict selected-rank held-out results...")
STRICT={}
for L in LAYERS:
    hits=0;ranks=[];margins=[];perhold=[]
    for hold in range(M):
        r=SELECT[(hold,L)];B=BASES[(hold,L)][:,:r]
        C=CENTROIDS[(hold,L)]@B
        T=torch.stack([CENTER[(hold,p,L)] for p in range(M)])@B
        hh=0;rrs=[];mms=[]
        for p in range(M):
            scores=[cos(T[p],C[j]) for j in range(M)]
            order=np.argsort(scores)[::-1].tolist();rank=order.index(p)+1
            hh+=rank==1;hits+=rank==1;ranks.append(rank);rrs.append(rank)
            mg=scores[p]-max(scores[j] for j in range(M) if j!=p);margins.append(mg);mms.append(mg)
        perhold.append((hh/M,float(np.mean(rrs)),float(np.mean(mms))))
    STRICT[L]=(hits/(M*M),float(np.mean(ranks)),float(np.mean(margins)),perhold)
for L in SHOW:
    x=STRICT[L];print(f"L{L:02d} top1={x[0]*100:5.1f}% meanRank={x[1]:.3f} margin={x[2]:+.4f}")
print("[17/24] Per-held-out-query generalization...")
for L in [8,12,16,19,20,24,27]:
    print(f"\nL{L:02d}")
    for h,x in enumerate(STRICT[L][3]):
        print(f" holdQ{h+1} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f} selectedRank={SELECT[(h,L)]}")
print("[18/24] Full-space baseline...")
FULL={}
for L in LAYERS:
    hits=0;ranks=[];margins=[]
    for hold in range(M):
        train=[q for q in range(M) if q!=hold]
        C=torch.stack([torch.stack([CENTER[(q,p,L)] for q in train]).mean(0) for p in range(M)])
        T=torch.stack([CENTER[(hold,p,L)] for p in range(M)])
        for p in range(M):
            scores=[cos(T[p],C[j]) for j in range(M)]
            order=np.argsort(scores)[::-1].tolist();rank=order.index(p)+1
            hits+=rank==1;ranks.append(rank);margins.append(scores[p]-max(scores[j] for j in range(M) if j!=p))
    FULL[L]=(hits/(M*M),float(np.mean(ranks)),float(np.mean(margins)))
for L in SHOW:
    print(f"L{L:02d} fullTop1={FULL[L][0]*100:5.1f}% rank={FULL[L][1]:.3f} margin={FULL[L][2]:+.4f}")
print("[19/24] Label permutation null...")
# Re-label TRAIN centroids only. Held-out labels remain fixed.
rng=np.random.default_rng(SEED);PERMS=1000;NULL={}
for L in SHOW:
    observed=STRICT[L][0];vals=[]
    for _ in range(PERMS):
        totalhit=0
        for hold in range(M):
            r=SELECT[(hold,L)];B=BASES[(hold,L)][:,:r]
            C=CENTROIDS[(hold,L)]@B
            perm=rng.permutation(M);C=C[torch.tensor(perm,device=C.device)]
            T=torch.stack([CENTER[(hold,p,L)] for p in range(M)])@B
            for p in range(M):
                scores=[cos(T[p],C[j]) for j in range(M)]
                totalhit+=int(int(np.argmax(scores))==p)
        vals.append(totalhit/(M*M))
    mu=float(np.mean(vals));sd=float(np.std(vals)+1e-12);z=(observed-mu)/sd
    pval=(1+sum(x>=observed for x in vals))/(PERMS+1)
    NULL[L]=(mu,sd,z,pval)
    print(f"L{L:02d} observed={observed:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={pval:.4f}")
print("[20/24] Geometry inside LOO subspace...")
GEO={}
for L in LAYERS:
    vals=[]
    for hold in range(M):
        r=SELECT[(hold,L)];B=BASES[(hold,L)][:,:r]
        T=torch.stack([CENTER[(hold,p,L)] for p in range(M)])@B
        G=cosine_matrix([T[p] for p in range(M)])
        vals.append(spearman(G0U,upper(G)))
    GEO[L]=float(np.mean(vals))
for L in SHOW:print(f"L{L:02d} heldOutGeometrySpearman={GEO[L]:+.4f}")
print("[21/24] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[22/24] Decision metrics...")
EARLY=np.mean([STRICT[L][0] for L in [8,9,10,12]])
MID=np.mean([STRICT[L][0] for L in [16,17,18,19,20]])
LATE=np.mean([STRICT[L][0] for L in [24,25,26,27]])
LATE_R=np.mean([STRICT[L][1] for L in [24,25,26,27]])
print(f"earlyTop1={EARLY:.4f} midTop1={MID:.4f} lateTop1={LATE:.4f} lateMeanRank={LATE_R:.3f} chanceTop1={1/M:.4f}")
print("[23/24] RESULTS")
print("\n"+"="*128);print("TEST 229 RESULTS");print("="*128)
print(f"MODE: 8 RESIDUAL PACKETS x 8 BLIND QUERIES | SINGLE L08 INJECTION | DOSE={PRIMARY:.4f}")
print("LOO: TRAIN 7 QUERIES -> IDENTITY SUBSPACE -> TEST ENTIRE HELD-OUT QUERY")
print("RANK SELECTION: TRAIN-ONLY 95% IDENTITY ENERGY | NO HELD-OUT LABEL USED FOR SELECTION")
print("NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN")
print("\nSTRICT HELD-OUT IDENTITY CLASSIFICATION")
for L in SHOW:
    x=STRICT[L]
    print(f"L{L:02d} top1={x[0]*100:5.1f}% meanRank={x[1]:.3f} margin={x[2]:+.4f} geometryρ={GEO[L]:+.4f} permP={NULL[L][3]:.4f}")
print("\nFULL-SPACE BASELINE")
for L in SHOW:
    x=FULL[L];print(f"L{L:02d} top1={x[0]*100:5.1f}% meanRank={x[1]:.3f} margin={x[2]:+.4f}")
print("\nINTERPRETATION GATE")
if LATE>=.50 and LATE_R<=3.0:
    print("RESULT: QUERY_GENERAL_IDENTITY_SUBSPACE — late-layer packet identity generalizes across held-out blind queries.")
elif LATE>=.25 and LATE_R<4.5:
    print("RESULT: PARTIAL_QUERY_GENERAL_IDENTITY — late layers retain measurable cross-query identity structure, but decoding is incomplete.")
else:
    print("RESULT: QUERY_SPECIFIC_TRAJECTORY — TEST228 geometry does not yield a robust late-layer identity carrier on held-out queries.")
print("-"*128)
print("Weights: PASS | Source facts absent from blind queries | Candidates never enter intervention")
print("Only L08 is intervened. L09-L27 responses are endogenous downstream consequences.")
print("="*128);print("[24/24] TEST 229 COMPLETE")
