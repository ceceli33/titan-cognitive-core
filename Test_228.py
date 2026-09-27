# ==================================================================================================
# TEST 228 — IDENTITY-GEOMETRY TRANSPORT X-RAY
# TEST227 BASELINE -> L08 IDENTITY RESIDUAL -> SINGLE L08 INJECTION -> FREE L09-L27 PROPAGATION
# QUESTION: DOES THE 8-PACKET RELATIONAL GEOMETRY SURVIVE EVEN IF ABSOLUTE OBJECT DECODING ROTATES?
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223
#          -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228
# TEST227 WORKING MODEL / SYSTEM / FACTS / TEST222 PACKET FORGE / COMMON SUBTRACTION PRESERVED
# NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN
# PRIMARY METRICS:
#   1) L08 residual packet 8x8 cosine geometry
#   2) downstream Δh 8x8 cosine geometry
#   3) upper-triangle Pearson/Spearman geometry retention
#   4) pair-order concordance + distance distortion
#   5) matched-query vs cross-query replication
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=228
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
PRIMARY=.04;LAYERS=list(range(8,28));SHOW=[8,9,10,12,16,19,20,24,27]
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
print("="*128);print("TEST 228 — IDENTITY-GEOMETRY TRANSPORT X-RAY");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228")
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
def pearson(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    a=a-a.mean();b=b-b.mean()
    den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den>1e-15 else 0.0
def rankdata(x):
    x=np.asarray(x,dtype=np.float64);order=np.argsort(x,kind="mergesort");r=np.empty(len(x),dtype=np.float64)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        rr=(i+j-1)/2.0+1.0
        for k in range(i,j):r[order[k]]=rr
        i=j
    return r
def spearman(a,b):return pearson(rankdata(a),rankdata(b))
def upper(G):
    return np.asarray([G[i,j] for i in range(M) for j in range(i+1,M)],dtype=np.float64)
def cosine_matrix(vecs):
    X=torch.stack([unit(v) for v in vecs]).float()
    return (X@X.T).cpu().numpy()
def pair_concordance(a,b):
    a=np.asarray(a);b=np.asarray(b);ok=0;tot=0
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            da=a[i]-a[j];db=b[i]-b[j]
            if abs(da)<1e-12 or abs(db)<1e-12:continue
            ok+=int(da*db>0);tot+=1
    return ok/tot if tot else 0.0
print("[2/24] Token maps...")
FMAP={};QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    fi=tok(chat(fact_text(s,r,o)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    full=fi.input_ids[0].tolist();ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Token map fail Q{qi+1}")
    q=qform(s,r);qe=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if o.lower() in q.lower():raise RuntimeError("Target leakage.")
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
    for name,mod in [("Q",layers[SRC_LAYER].self_attn.q_proj),("K",layers[SRC_LAYER].self_attn.k_proj),("V",layers[SRC_LAYER].self_attn.v_proj)]:
        def mk(n):
            def hk(m,args,out):S[n]=out[0].float().detach().clone()
            return hk
        hs.append(mod.register_forward_hook(mk(name)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S
SRC={}
for qi in range(M):
    SRC[qi]=capture_source(FMAP[qi][0]);print(f"Q{qi+1} captured")
print("[5/24] Reconstruct TEST222 source readout...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    kh=qhead//GROUP;q=rope(qh(S["Q"],qpos,qhead),qpos)
    K=torch.stack([rope(kvh(S["K"],p,kh),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)
READ={};RAWV={}
for qi in range(M):
    S=SRC[qi];oe=FMAP[qi][3][-1];rows=[]
    for h in QGROUP:
        for qp in range(oe,FMAP[qi][0].input_ids.shape[1]):
            a=attn_row(S,qp,h);rows.append((float(a[oe]),h,qp))
    rows.sort(key=lambda z:z[0],reverse=True);READ[qi]=rows;RAWV[qi]=kvh(S["V"],oe,KVH).clone()
    b=rows[0];print(f"Q{qi+1} bestQH={b[1]:02d} qpos={b[2]} objEndAttn={b[0]:.6f}")
print("[6/24] Forge TEST222 RAW packets...")
RAW={}
for qi in range(M):
    p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32);v=RAWV[qi]
    for h in QGROUP:
        w=max(x[0] for x in READ[qi] if x[1]==h);p[h]=w*v
    with torch.inference_mode():RAW[qi]=layers[SRC_LAYER].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"Q{qi+1} rawNorm={RAW[qi].norm():.4f}")
print("[7/24] TEST227 COMMON subtraction...")
COMMON=torch.stack([RAW[i] for i in range(M)]).mean(0)
RES={i:RAW[i]-COMMON for i in range(M)}
for qi in range(M):
    print(f"Q{qi+1} residualNorm={RES[qi].norm():.4f} residual/raw={float(RES[qi].norm()/RAW[qi].norm()):.4f}")
print("[8/24] L08 residual identity geometry...")
G0=cosine_matrix([RES[i] for i in range(M)]);G0U=upper(G0)
print(f"pairs={len(G0U)} meanOffDiag={G0U.mean():+.4f} min={G0U.min():+.4f} max={G0U.max():+.4f}")
for i in range(M):
    print("Q%d "%(i+1)+" ".join(f"{G0[i,j]:+.3f}" for j in range(M)))
print("[9/24] Vanilla blind trajectories...")
@torch.inference_mode()
def vanilla_hidden(e):
    pos=e.input_ids.shape[1]-1;S={};hs=[]
    for L in LAYERS:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S
VAN={}
for qi in range(M):
    VAN[qi]=vanilla_hidden(QENC[qi]);print(f"Q{qi+1} vanilla captured")
print("[10/24] Matched-query residual trajectories...")
@torch.inference_mode()
def injected_hidden(e,packet):
    pos=e.input_ids.shape[1]-1;S={};hs=[];calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float()
        d=unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY
        y[:,-1,:]=(z+d).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    ih=layers[SRC_LAYER].register_forward_hook(inject)
    for L in LAYERS:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        remove(hs);ih.remove()
    if calls!=1:raise RuntimeError(f"Injection calls={calls}, expected 1")
    return S
MATCH={};MDELTA={}
for qi in range(M):
    MATCH[qi]=injected_hidden(QENC[qi],RES[qi]);MDELTA[qi]={}
    for L in LAYERS:MDELTA[qi][L]=MATCH[qi][L]-VAN[qi][L]
    print(f"Q{qi+1} complete")
print("[11/24] Verify matched L08 injection...")
for qi in range(M):
    d=MDELTA[qi][8]
    rel=float(d.norm()/VAN[qi][8].norm().clamp_min(EPS))
    print(f"Q{qi+1} displacement={rel*100:.4f}% residualCos={cos(d,RES[qi]):+.4f}")
print("[12/24] Matched-query geometry retention...")
MATCH_MET={}
for L in LAYERS:
    G=cosine_matrix([MDELTA[i][L] for i in range(M)]);u=upper(G)
    pr=pearson(G0U,u);sr=spearman(G0U,u);cc=pair_concordance(G0U,u)
    rmse=float(np.sqrt(np.mean((G0U-u)**2)));collapse=float(u.mean())
    MATCH_MET[L]=(pr,sr,cc,rmse,collapse)
for L in SHOW:
    pr,sr,cc,rmse,collapse=MATCH_MET[L]
    print(f"L{L:02d} Pearson={pr:+.4f} Spearman={sr:+.4f} concord={cc:.4f} RMSE={rmse:.4f} meanOffDiag={collapse:+.4f}")
print("[13/24] Cross-query controlled transport matrix...")
# Each of the 8 residual packets is now injected into EACH blind query.
# This separates packet identity from query-specific background dynamics.
ALL={};ADELTA={}
for qi in range(M):
    for pj in range(M):
        S=injected_hidden(QENC[qi],RES[pj]);ALL[(qi,pj)]=S;ADELTA[(qi,pj)]={}
        for L in LAYERS:ADELTA[(qi,pj)][L]=S[L]-VAN[qi][L]
    print(f"Query Q{qi+1}: all 8 packet identities complete")
print("[14/24] Per-query geometry retention...")
PERQ={}
for qi in range(M):
    PERQ[qi]={}
    for L in LAYERS:
        G=cosine_matrix([ADELTA[(qi,pj)][L] for pj in range(M)]);u=upper(G)
        PERQ[qi][L]=(pearson(G0U,u),spearman(G0U,u),pair_concordance(G0U,u),float(np.sqrt(np.mean((G0U-u)**2))),float(u.mean()))
for L in SHOW:
    p=np.mean([PERQ[q][L][0] for q in range(M)])
    s=np.mean([PERQ[q][L][1] for q in range(M)])
    c=np.mean([PERQ[q][L][2] for q in range(M)])
    r=np.mean([PERQ[q][L][3] for q in range(M)])
    o=np.mean([PERQ[q][L][4] for q in range(M)])
    print(f"L{L:02d} meanPearson={p:+.4f} meanSpearman={s:+.4f} concord={c:.4f} RMSE={r:.4f} meanOffDiag={o:+.4f}")
print("[15/24] Per-query details...")
for qi in range(M):
    print(f"Q{qi+1}",end="")
    for L in [8,12,19,27]:
        print(f" L{L:02d}ρ={PERQ[qi][L][1]:+.3f}",end="")
    print()
print("[16/24] Query-averaged response geometry...")
AVG_MET={};AVG_G={}
for L in LAYERS:
    gs=[]
    for qi in range(M):
        gs.append(cosine_matrix([ADELTA[(qi,pj)][L] for pj in range(M)]))
    G=np.mean(gs,axis=0);AVG_G[L]=G;u=upper(G)
    AVG_MET[L]=(pearson(G0U,u),spearman(G0U,u),pair_concordance(G0U,u),float(np.sqrt(np.mean((G0U-u)**2))),float(u.mean()))
for L in SHOW:
    pr,sr,cc,rmse,collapse=AVG_MET[L]
    print(f"L{L:02d} Pearson={pr:+.4f} Spearman={sr:+.4f} concord={cc:.4f} RMSE={rmse:.4f} meanOffDiag={collapse:+.4f}")
print("[17/24] Packet identity nearest-neighbor consistency...")
# For each query/layer, compare downstream pairwise-neighbor ordering with the L08 residual packet geometry.
NN={}
src_nn={i:max([j for j in range(M) if j!=i],key=lambda j:G0[i,j]) for i in range(M)}
for L in SHOW:
    hits=0;tot=0
    for qi in range(M):
        G=cosine_matrix([ADELTA[(qi,pj)][L] for pj in range(M)])
        for i in range(M):
            nn=max([j for j in range(M) if j!=i],key=lambda j:G[i,j])
            hits+=int(nn==src_nn[i]);tot+=1
    NN[L]=hits/tot
    print(f"L{L:02d} sourceNearestNeighborRetention={hits}/{tot} ({hits/tot:.3f})")
print("[18/24] Positive/negative sign symmetry...")
NEG={}
for qi in range(M):
    NEG[qi]=injected_hidden(QENC[qi],-RES[qi])
for L in SHOW:
    vals=[]
    for qi in range(M):
        dn=NEG[qi][L]-VAN[qi][L]
        vals.append(cos(MDELTA[qi][L],-dn))
    print(f"L{L:02d} signSymmetry={np.mean(vals):+.4f}")
print("[19/24] Geometry permutation null...")
rng=np.random.default_rng(SEED);PERMS=2000
NULL={}
for L in SHOW:
    observed=AVG_MET[L][1];vals=[]
    u=upper(AVG_G[L])
    for _ in range(PERMS):
        p=rng.permutation(M);Gp=G0[np.ix_(p,p)]
        vals.append(spearman(upper(Gp),u))
    mu=float(np.mean(vals));sd=float(np.std(vals)+1e-12)
    z=(observed-mu)/sd;pval=(1+sum(x>=observed for x in vals))/(PERMS+1)
    NULL[L]=(observed,mu,sd,z,pval)
    print(f"L{L:02d} observedSpearman={observed:+.4f} null={mu:+.4f}±{sd:.4f} z={z:+.3f} p={pval:.4f}")
print("[20/24] Physical displacement...")
for L in SHOW:
    vals=[]
    for qi in range(M):
        vals.append(float(MDELTA[qi][L].norm()/VAN[qi][L].norm().clamp_min(EPS)))
    print(f"L{L:02d} matchedMeanDisplacement={np.mean(vals)*100:.4f}%")
print("[21/24] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[22/24] Decision metrics...")
EARLY=np.mean([AVG_MET[L][1] for L in [8,9,10,12]])
MID=np.mean([AVG_MET[L][1] for L in [16,17,18,19,20]])
LATE=np.mean([AVG_MET[L][1] for L in [24,25,26,27]])
LATE_P=np.mean([NULL[L][4] for L in [24,27]])
print(f"earlyMeanSpearman={EARLY:+.4f}")
print(f"midMeanSpearman={MID:+.4f}")
print(f"lateMeanSpearman={LATE:+.4f}")
print(f"lateRepresentativeMeanPermutationP={LATE_P:.4f}")
print("[23/24] RESULTS")
print("\n"+"="*128);print("TEST 228 RESULTS");print("="*128)
print(f"MODE: TEST227 IDENTITY RESIDUAL -> SINGLE L08 INJECTION -> FREE ENDOGENOUS L09-L27 PROPAGATION | DOSE={PRIMARY:.4f}")
print("NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN")
print("\nQUERY-AVERAGED IDENTITY-GEOMETRY RETENTION")
for L in SHOW:
    pr,sr,cc,rmse,collapse=AVG_MET[L]
    print(f"L{L:02d} Pearson={pr:+.4f} Spearman={sr:+.4f} concord={cc:.4f} RMSE={rmse:.4f} meanOffDiag={collapse:+.4f} permP={NULL[L][4]:.4f}")
print("\nMATCHED-QUERY GEOMETRY")
for L in SHOW:
    pr,sr,cc,rmse,collapse=MATCH_MET[L]
    print(f"L{L:02d} Pearson={pr:+.4f} Spearman={sr:+.4f} RMSE={rmse:.4f}")
print("\nINTERPRETATION GATE")
if LATE>=0.60 and LATE_P<=0.05:
    print("RESULT: IDENTITY_GEOMETRY_SURVIVES — packet coordinates rotate, but relational packet geometry remains strongly recoverable downstream.")
elif LATE>=0.30:
    print("RESULT: PARTIAL_GEOMETRY_RETENTION — downstream dynamics preserve a measurable fraction of packet identity geometry.")
else:
    print("RESULT: IDENTITY_GEOMETRY_DEGRADES — packet-specific trajectories remain distinct, but the original residual identity geometry is not stably preserved.")
print("-"*128)
print("Weights: PASS | Source facts absent from blind queries | Only L08 is intervened")
print("L09-L27 measurements are model-native downstream consequences of the single L08 residual intervention.")
print("="*128);print("[24/24] TEST 228 COMPLETE")



