# ==================================================================================================
# TEST 218 — POSITION-MATCHED K/V IDENTITY VALIDATION
# HELD-OUT NONCE FACTS × EXACT TOKEN/POSITION MATCH × PRE-REGISTERED K/V CARRIERS
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218
# TEST217 MODEL/SYSTEM/X-RAY PRESERVED | MOTOR OFF | WEIGHTS FROZEN | NO INTERVENTION
# PRE-REGISTERED FROM TEST217: K/L10/H00, V/L07/H02, V/L08/H00
# REQUIREMENT: target/wrong object token count + OBJECT_START/END + causal prefix must match exactly
# ==================================================================================================
import os,sys,random,subprocess,importlib.util
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=218;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;EPS=1e-8;COMMON_RANK=2
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
CARRIERS=[("K",10,0),("V",7,2),("V",8,0)]
# Held-out subjects/relations. Object pool is filtered automatically to exact tokenizer length.
FACTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards"),("Varek Tonn","holds"),("Lira Mesk","stores"),("Korin Drel","protects"),("Taren Vosk","carries")]
OBJECT_POOL=["the amber compass","the silver lantern","the violet key","the bronze sphere","the golden necklace","the iron dagger","the crystal mirror","the wooden mask","the scarlet book","the ivory ring","the copper bell","the black feather","the green bottle","the marble token","the yellow candle","the velvet ribbon","the cobalt badge","the wooden tablet","the silver mirror","the bronze pendant","the crimson feather","the copper whistle","the ivory button","the purple candle"]
M=len(FACTS)
print("="*128);print("TEST 218 — POSITION-MATCHED K/V IDENTITY VALIDATION");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218");print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,"| MOTOR: OFF")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/16] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;NH=model.config.num_attention_heads;NKV=model.config.num_key_value_heads;HD=H//NH
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} heads={NH} kv_heads={NKV} head_dim={HD}")
FP_T=[layers[0].self_attn.q_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
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
print("[2/16] Position-matched object bank...")
# Match by actual object span token count inside identical sentence context, not standalone token count.
BANK={}
for qi,(s,r) in enumerate(FACTS):
    groups={}
    for o in OBJECT_POOL:
        text=fact_text(s,r,o);full=tok(chat(text),add_special_tokens=False).input_ids;sp=last_span(full,o)
        if sp:groups.setdefault(len(sp),[]).append((o,full,sp))
    valid=[v for v in groups.values() if len(v)>=10]
    if not valid:raise RuntimeError(f"No >=10-object matched bank for Q{qi+1}: "+str({k:len(v) for k,v in groups.items()}))
    valid.sort(key=lambda v:(len(v),len(v[0][2])),reverse=True);chosen=valid[0][:10]
    # Exact same object start/end and exact causal prefix through token before object.
    starts={x[2][0] for x in chosen};ends={x[2][-1] for x in chosen};lens={len(x[2]) for x in chosen}
    if len(starts)!=1 or len(ends)!=1 or len(lens)!=1:raise RuntimeError(f"Position mismatch Q{qi+1}")
    pre=chosen[0][2][0]
    if not all(x[1][:pre]==chosen[0][1][:pre] for x in chosen):raise RuntimeError(f"Prefix mismatch Q{qi+1}")
    BANK[qi]=chosen
    print(f"Q{qi+1} {s}|{r} object_tokens={len(chosen[0][2])} start={chosen[0][2][0]} end={chosen[0][2][-1]} bank={len(chosen)} PREFIX=PASS")
print("[3/16] Assign target/wrong objects...")
ASSIGN={}
for qi in range(M):
    arr=BANK[qi]
    # Deterministic held-out assignment; each fact gets a different target where possible.
    ti=qi%len(arr);ordered=[arr[ti]]+[x for j,x in enumerate(arr) if j!=ti]
    ASSIGN[qi]=ordered
    print(f"Q{qi+1} TARGET={ordered[0][0]} WRONG1={ordered[1][0]} WRONG2={ordered[2][0]}")
print("[4/16] Capture pre-registered K/V only...")
NEED={("K",10),("V",7),("V",8)}
@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt",add_special_tokens=False).to(DEVICE);S={};hs=[]
    def mk(rep,L):
        def hk(m,args,out):S[(rep,L)]=out[0].float().detach().clone()
        return hk
    for rep,L in NEED:
        mod=layers[L].self_attn.k_proj if rep=="K" else layers[L].self_attn.v_proj
        hs.append(mod.register_forward_hook(mk(rep,L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return e.input_ids[0].detach().cpu().tolist(),S
CAP={}
for qi,(s,r) in enumerate(FACTS):
    for j,(o,_,_) in enumerate(ASSIGN[qi]):CAP[(qi,j)]=capture(fact_text(s,r,o))
    print(f"Q{qi+1} captures={len(ASSIGN[qi])}")
print("[5/16] Strict alignment verification...")
POS={}
for qi in range(M):
    ref_ids=CAP[(qi,0)][0];ref_o=ASSIGN[qi][0][0];ref_sp=last_span(ref_ids,ref_o);POS[qi]=ref_sp
    ok=True
    for j,(o,_,_) in enumerate(ASSIGN[qi]):
        full=CAP[(qi,j)][0];sp=last_span(full,o)
        ok&=(len(sp)==len(ref_sp) and sp[0]==ref_sp[0] and sp[-1]==ref_sp[-1] and full[:sp[0]]==ref_ids[:ref_sp[0]])
    print(f"Q{qi+1} PREFIX={'PASS' if ok else 'FAIL'} START={ref_sp[0]} END={ref_sp[-1]} LEN={len(ref_sp)}")
    if not ok:raise RuntimeError(f"STRICT POSITION/PREFIX FAILED Q{qi+1}")
print("[6/16] Head extraction...")
def head(x,pos,h):return x[pos].reshape(NKV,HD)[h]
print("[7/16] Identity assay...")
MET={};RES={}
for rep,L,h in CARRIERS:
    for qi in range(M):
        pos=POS[qi][-1];T=head(CAP[(qi,0)][1][(rep,L)],pos,h)
        A=[head(CAP[(qi,j)][1][(rep,L)],pos,h) for j in range(1,len(ASSIGN[qi]))]
        X=torch.stack([T-a for a in A]);mu=X.mean(0);Xc=X-mu;_,_,Vh=torch.linalg.svd(Xc,full_matrices=False);U=Vh[:min(COMMON_RANK,Vh.shape[0])]
        c=X[0]-mu;r=c-(c@U.T)@U;r=unit(r[None])[0];wrong=[]
        for j in range(1,X.shape[0]):
            q=X[j]-mu;q=q-(q@U.T)@U;wrong.append(unit(q[None])[0])
        best=max(float(r@q) for q in wrong);frac=float((c-(c@U.T)@U).norm()/X[0].norm().clamp_min(EPS));rel=float((T-A[0]).norm()/T.norm().clamp_min(EPS))
        pair=[]
        u=[r]+wrong
        for a in range(len(u)):
            for b in range(a+1,len(u)):pair.append(abs(float(u[a]@u[b])))
        score=frac*(1-best)/2;MET[(rep,L,h,qi)]=(score,rel,frac,best,float(np.mean(pair)));RES[(rep,L,h,qi)]=r
print("[8/16] Pre-registered carrier results...")
for rep,L,h in CARRIERS:
    a=np.array([MET[(rep,L,h,q)] for q in range(M)])
    print(f"{rep}/L{L:02d}/H{h:02d} meanScore={a[:,0].mean():.4f} min={a[:,0].min():.4f} sd={a[:,0].std():.4f} relΔ={a[:,1].mean()*100:.3f}% idFrac={a[:,2].mean():.4f} wrongCos={a[:,3].mean():+.4f}")
print("[9/16] Per-fact results...")
for qi in range(M):
    print(f"Q{qi+1} TARGET={ASSIGN[qi][0][0]}")
    for rep,L,h in CARRIERS:
        z=MET[(rep,L,h,qi)];print(f" {rep}/L{L:02d}/H{h:02d} score={z[0]:.4f} relΔ={z[1]*100:.3f}% idFrac={z[2]:.4f} wrongCos={z[3]:+.4f} pairAbs={z[4]:.4f}")
print("[10/16] Carrier ranking...")
RANK=[]
for rep,L,h in CARRIERS:
    s=[MET[(rep,L,h,q)][0] for q in range(M)]
    RANK.append((float(np.mean(s)),float(np.min(s)),float(np.std(s)),rep,L,h,s))
RANK.sort(reverse=True,key=lambda x:(x[1],x[0]))
for x in RANK:print(f"{x[3]}/L{x[4]:02d}/H{x[5]:02d} min={x[1]:.4f} mean={x[0]:.4f} sd={x[2]:.4f} perQ="+",".join(f"{v:.3f}" for v in x[6]))
print("[11/16] Target-label permutation null...")
# Same fixed captured states; rotate which object is called target. This tests whether the observed target assignment is exceptional.
NULL={c:[] for c in CARRIERS}
for shift in range(1,9):
    for rep,L,h in CARRIERS:
        ss=[]
        for qi in range(M):
            arr=ASSIGN[qi];ti=shift%len(arr);pos=POS[qi][-1];T=head(CAP[(qi,ti)][1][(rep,L)],pos,h)
            A=[head(CAP[(qi,j)][1][(rep,L)],pos,h) for j in range(len(arr)) if j!=ti]
            X=torch.stack([T-a for a in A]);mu=X.mean(0);Xc=X-mu;_,_,Vh=torch.linalg.svd(Xc,full_matrices=False);U=Vh[:min(COMMON_RANK,Vh.shape[0])]
            c=X[0]-mu;r=c-(c@U.T)@U;r=unit(r[None])[0];wrong=[]
            for j in range(1,X.shape[0]):
                q=X[j]-mu;q=q-(q@U.T)@U;wrong.append(unit(q[None])[0])
            best=max(float(r@q) for q in wrong);frac=float((c-(c@U.T)@U).norm()/X[0].norm().clamp_min(EPS));ss.append(frac*(1-best)/2)
        NULL[(rep,L,h)].append(float(np.mean(ss)))
for rep,L,h in CARRIERS:
    obs=float(np.mean([MET[(rep,L,h,q)][0] for q in range(M)]));nu=np.array(NULL[(rep,L,h)])
    print(f"{rep}/L{L:02d}/H{h:02d} OBS={obs:.4f} NULLmean={nu.mean():.4f} NULLsd={nu.std():.4f} z={(obs-nu.mean())/(nu.std()+EPS):+.3f}")
print("[12/16] Cross-fact residual coherence...")
for rep,L,h in CARRIERS:
    R=torch.stack([RES[(rep,L,h,q)] for q in range(M)]);C=R@R.T;vals=[]
    for i in range(M):
        for j in range(i+1,M):vals.append(float(C[i,j]))
    print(f"{rep}/L{L:02d}/H{h:02d} residualCrossFact meanCos={np.mean(vals):+.4f} meanAbs={np.mean(np.abs(vals)):.4f}")
print("[13/16] Position controls...")
# Compare OBJECT_START vs OBJECT_END with identical positional matching.
for rep,L,h in CARRIERS:
    for site in ["START","END"]:
        scores=[]
        for qi in range(M):
            pos=POS[qi][0] if site=="START" else POS[qi][-1];T=head(CAP[(qi,0)][1][(rep,L)],pos,h);A=[head(CAP[(qi,j)][1][(rep,L)],pos,h) for j in range(1,len(ASSIGN[qi]))]
            X=torch.stack([T-a for a in A]);mu=X.mean(0);Xc=X-mu;_,_,Vh=torch.linalg.svd(Xc,full_matrices=False);U=Vh[:min(COMMON_RANK,Vh.shape[0])]
            c=X[0]-mu;r=c-(c@U.T)@U;r=unit(r[None])[0];ww=[]
            for j in range(1,X.shape[0]):
                q=X[j]-mu;q=q-(q@U.T)@U;ww.append(unit(q[None])[0])
            scores.append(float((c-(c@U.T)@U).norm()/X[0].norm().clamp_min(EPS))*(1-max(float(r@q) for q in ww))/2)
        print(f"{rep}/L{L:02d}/H{h:02d} {site} meanScore={np.mean(scores):.4f}")
print("[14/16] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[15/16] RESULTS")
print("\n"+"="*128);print("TEST 218 RESULTS");print("="*128)
print("MODE: MOTOR OFF | WEIGHTS FROZEN | HELD-OUT NONCE SUBJECTS | EXACT POSITION-MATCHED OBJECT BANKS")
print("PRE-REGISTERED TEST217 CARRIERS: K/L10/H00 | V/L07/H02 | V/L08/H00")
print("\nSTRICT PREFIX/POSITION")
for qi in range(M):print(f"Q{qi+1} PASS | objectTokens={len(POS[qi])} start={POS[qi][0]} end={POS[qi][-1]} target={ASSIGN[qi][0][0]}")
print("\nCARRIER VALIDATION")
for x in RANK:print(f"{x[3]}/L{x[4]:02d}/H{x[5]:02d} minScore={x[1]:.4f} meanScore={x[0]:.4f} sd={x[2]:.4f} perQ="+",".join(f"{v:.3f}" for v in x[6]))
print("\nPER-FACT")
for qi in range(M):
    print(f"Q{qi+1} {FACTS[qi][0]} | {FACTS[qi][1]} | {ASSIGN[qi][0][0]}")
    for rep,L,h in CARRIERS:
        z=MET[(rep,L,h,qi)];print(f" {rep}/L{L:02d}/H{h:02d} score={z[0]:.4f} idFrac={z[2]:.4f} wrongCos={z[3]:+.4f}")
print("\nTARGET-LABEL NULL")
for rep,L,h in CARRIERS:
    obs=float(np.mean([MET[(rep,L,h,q)][0] for q in range(M)]));nu=np.array(NULL[(rep,L,h)])
    print(f"{rep}/L{L:02d}/H{h:02d} OBS={obs:.4f} NULL={nu.mean():.4f}±{nu.std():.4f} z={(obs-nu.mean())/(nu.std()+EPS):+.3f}")
print("-"*128)
print("Weights: PASS | Injection: NONE | L0-L27 observation only")
print("PASS requires exact prefix/position matching plus held-out persistence of the pre-registered TEST217 carriers")
print("This assay validates localization only; it does not claim causal-head function or behavioral retrieval")
print("="*128);print("[16/16] TEST 218 COMPLETE")



