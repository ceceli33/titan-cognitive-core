# ==================================================================================================
# TEST 217 — ATTENTION K/V OBJECT-IDENTITY X-RAY
# OBJECT_END → LAYER × HEAD × KEY/VALUE IDENTITY LOCALIZATION
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST211 -> TEST212 -> TEST213 -> TEST214 -> TEST215 -> TEST216 -> TEST217
# TEST216 MODEL/SYSTEM/FACTS PRESERVED | MOTOR OFF | WEIGHTS FROZEN | NO INTERVENTION
# PURPOSE: locate object identity directly inside attention K/V projections before another transport test
# ==================================================================================================
import os,sys,random,subprocess,importlib.util
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=217
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;EPS=1e-8;COMMON_RANK=2
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
ALT_OBJECTS=["the golden necklace","the iron dagger","the crystal mirror","the wooden mask","the scarlet book","the ivory ring","the copper bell","the black feather"]
M=len(FACTS)
print("="*128);print("TEST 217 — ATTENTION K/V OBJECT-IDENTITY X-RAY");print("="*128)
print("LINEAGE: TEST211 -> TEST212 -> TEST213 -> TEST214 -> TEST215 -> TEST216 -> TEST217")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,"| MOTOR: OFF")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/16] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch.")
NH=model.config.num_attention_heads;NKV=model.config.num_key_value_heads;HD=H//NH
if layers[0].self_attn.k_proj.out_features!=NKV*HD or layers[0].self_attn.v_proj.out_features!=NKV*HD:raise RuntimeError("K/V geometry mismatch.")
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
print("[2/16] K/V capture engine...")
@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt",add_special_tokens=False).to(DEVICE);K=[None]*TOTAL;V=[None]*TOTAL;hs=[]
    def mk(store,L):
        def hk(m,args,out):store[L]=out[0].float().detach().clone()
        return hk
    for L in range(TOTAL):
        hs.append(layers[L].self_attn.k_proj.register_forward_hook(mk(K,L)))
        hs.append(layers[L].self_attn.v_proj.register_forward_hook(mk(V,L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return {"ids":e.input_ids[0].detach().cpu().tolist(),"K":K,"V":V}
print("[3/16] Target/wrong captures...")
CAP={}
for qi,(s,r,o) in enumerate(FACTS):
    objs=[o,FACTS[(qi+1)%M][2]]+ALT_OBJECTS
    for j,obj in enumerate(objs):CAP[(qi,j)]=capture(fact_text(s,r,obj))
    print(f"Q{qi+1} captures={len(objs)}")
print("[4/16] OBJECT_END maps...")
OBJEND={}
for qi,(s,r,o) in enumerate(FACTS):
    objs=[o,FACTS[(qi+1)%M][2]]+ALT_OBJECTS
    for j,obj in enumerate(objs):
        sp=last_span(CAP[(qi,j)]["ids"],obj)
        if not sp:raise RuntimeError(f"Object alignment failed Q{qi+1}/{obj}")
        OBJEND[(qi,j)]=sp[-1]
    print(f"Q{qi+1} OBJECT_END={OBJEND[(qi,0)]}")
print("[5/16] Prefix causal sanity...")
for qi in range(M):
    t=CAP[(qi,0)]["ids"];w=CAP[(qi,1)]["ids"];p=min(OBJEND[(qi,0)],OBJEND[(qi,1)])
    print(f"Q{qi+1} common-prefix-before-object={'PASS' if t[:p]==w[:p] else 'FAIL'}")
print("[6/16] Reshape K/V → KV heads...")
def hv(x,pos):return x[pos].reshape(NKV,HD)
print("[7/16] Head identity residuals...")
MET={};RES={}
for rep in ["K","V"]:
    for qi in range(M):
        for L in range(TOTAL):
            T=hv(CAP[(qi,0)][rep][L],OBJEND[(qi,0)])
            A=[hv(CAP[(qi,j)][rep][L],OBJEND[(qi,j)]) for j in range(1,len(ALT_OBJECTS)+2)]
            for h in range(NKV):
                X=torch.stack([T[h]-z[h] for z in A]);mu=X.mean(0);Xc=X-mu
                _,_,Vh=torch.linalg.svd(Xc,full_matrices=False);U=Vh[:min(COMMON_RANK,Vh.shape[0])]
                c=X[0]-mu;ri=c-(c@U.T)@U
                wrong=[]
                for j in range(1,X.shape[0]):
                    q=X[j]-mu;q=q-(q@U.T)@U;wrong.append(unit(q[None])[0])
                r=unit(ri[None])[0];best=max(float(r@q) for q in wrong)
                frac=float(ri.norm()/X[0].norm().clamp_min(EPS));rel=float((T[h]-A[0][h]).norm()/T[h].norm().clamp_min(EPS))
                pair=[]
                UDI=[r]+wrong
                for a in range(len(UDI)):
                    for b in range(a+1,len(UDI)):pair.append(abs(float(UDI[a]@UDI[b])))
                pairabs=float(np.mean(pair));score=frac*(1-best)/2
                MET[(rep,qi,L,h)]=(score,rel,frac,best,pairabs);RES[(rep,qi,L,h)]=r
print("[8/16] Global layer×head ranking...")
AVG=[]
for rep in ["K","V"]:
    for L in range(TOTAL):
        for h in range(NKV):
            a=np.array([MET[(rep,q,L,h)] for q in range(M)])
            AVG.append((float(a[:,0].mean()),rep,L,h,float(a[:,1].mean()),float(a[:,2].mean()),float(a[:,3].mean()),float(a[:,4].mean())))
AVG.sort(reverse=True,key=lambda x:x[0])
for x in AVG[:30]:print(f"{x[1]} L{x[2]:02d} H{x[3]:02d} score={x[0]:.4f} relΔ={x[4]*100:7.3f}% idFrac={x[5]:.4f} bestWrongCos={x[6]:+.4f} pairAbsCos={x[7]:.4f}")
print("[9/16] Best K/V heads per layer...")
BEST={}
for rep in ["K","V"]:
    print("\n"+rep)
    for L in range(TOTAL):
        rows=[x for x in AVG if x[1]==rep and x[2]==L];b=max(rows,key=lambda x:x[0]);BEST[(rep,L)]=b
        print(f"L{L:02d} H{b[3]:02d} score={b[0]:.4f} idFrac={b[5]:.4f} wrongCos={b[6]:+.4f} relΔ={b[4]*100:7.3f}%")
print("[10/16] Per-fact best carriers...")
PFB={}
for qi in range(M):
    rows=[]
    for rep in ["K","V"]:
        for L in range(TOTAL):
            for h in range(NKV):
                z=MET[(rep,qi,L,h)];rows.append((z[0],rep,L,h,z[1],z[2],z[3],z[4]))
    rows.sort(reverse=True,key=lambda x:x[0]);PFB[qi]=rows[0]
    x=rows[0];print(f"Q{qi+1} BEST={x[1]}/L{x[2]:02d}/H{x[3]:02d} score={x[0]:.4f} relΔ={x[4]*100:.3f}% idFrac={x[5]:.4f} wrongCos={x[6]:+.4f}")
print("[11/16] Early/mid/late concentration...")
for rep in ["K","V"]:
    for name,lo,hi in [("EARLY",0,8),("MID",9,19),("LATE",20,27)]:
        rows=[x for x in AVG if x[1]==rep and lo<=x[2]<=hi];b=max(rows,key=lambda x:x[0])
        print(f"{rep} {name:5s} BEST=L{b[2]:02d}/H{b[3]:02d} score={b[0]:.4f} idFrac={b[5]:.4f} wrongCos={b[6]:+.4f}")
print("[12/16] Cross-fact head consistency...")
CONS=[]
for rep in ["K","V"]:
    for L in range(TOTAL):
        for h in range(NKV):
            s=[MET[(rep,q,L,h)][0] for q in range(M)]
            CONS.append((min(s),float(np.mean(s)),float(np.std(s)),rep,L,h,s))
CONS.sort(reverse=True,key=lambda x:(x[0],x[1]))
for x in CONS[:20]:print(f"{x[3]} L{x[4]:02d} H{x[5]:02d} min={x[0]:.4f} mean={x[1]:.4f} sd={x[2]:.4f} perQ="+",".join(f"{v:.3f}" for v in x[6]))
print("[13/16] K vs V aggregate...")
for rep in ["K","V"]:
    rows=[x for x in AVG if x[1]==rep]
    top=rows[:]
    vals=np.array([[x[0],x[5],x[6]] for x in rows])
    b=max(rows,key=lambda x:x[0])
    print(f"{rep}: meanScore={vals[:,0].mean():.4f} meanIdFrac={vals[:,1].mean():.4f} meanWrongCos={vals[:,2].mean():+.4f} BEST=L{b[2]:02d}/H{b[3]:02d} score={b[0]:.4f}")
print("[14/16] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[15/16] RESULTS")
print("\n"+"="*128);print("TEST 217 RESULTS");print("="*128)
print("LINEAGE: TEST211 -> TEST212 -> TEST213 -> TEST214 -> TEST215 -> TEST216 -> TEST217")
print(f"MODE: MOTOR OFF | WEIGHTS FROZEN | KV_HEADS={NKV} | HEAD_DIM={HD} | OBJECT_END ONLY")
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("\nTOP GLOBAL K/V IDENTITY CARRIERS")
for x in AVG[:20]:print(f"{x[1]} L{x[2]:02d} H{x[3]:02d} score={x[0]:.4f} relΔ={x[4]*100:7.3f}% idFrac={x[5]:.4f} wrongCos={x[6]:+.4f} pairAbsCos={x[7]:.4f}")
print("\nPER-FACT BEST")
for qi,x in PFB.items():print(f"Q{qi+1} {x[1]}/L{x[2]:02d}/H{x[3]:02d} score={x[0]:.4f} relΔ={x[4]*100:.3f}% idFrac={x[5]:.4f} wrongCos={x[6]:+.4f}")
print("\nMOST CONSISTENT ACROSS ALL FOUR FACTS")
for x in CONS[:12]:print(f"{x[3]} L{x[4]:02d} H{x[5]:02d} minScore={x[0]:.4f} mean={x[1]:.4f} sd={x[2]:.4f} perQ="+",".join(f"{v:.3f}" for v in x[6]))
print("\nBEST HEAD PER LAYER")
for rep in ["K","V"]:
    print(rep)
    for L in range(TOTAL):
        b=BEST[(rep,L)];print(f"L{L:02d} H{b[3]:02d} score={b[0]:.4f} idFrac={b[5]:.4f} wrongCos={b[6]:+.4f}")
print("-"*128)
print("Weights: PASS | Injection: NONE | L0-L27 observation only")
print("TEST217 asks whether TEST213 object identity is concentrated in attention key/value head states rather than a single residual vector")
print("No behavioral retrieval or causal-head claim is made by this assay")
print("="*128);print("[16/16] TEST 217 COMPLETE")



