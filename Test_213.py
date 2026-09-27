# ==================================================================================================
# TEST 213 — SEQUENCE-DISTRIBUTED OBJECT-IDENTITY X-RAY
# WHERE DOES BOUND-OBJECT IDENTITY LIVE? TOKEN × LAYER × REPRESENTATION SITE
# --------------------------------------------------------------------------------------------------
# LINEAGE:
# TEST 103      -> DIBEKGOZ: projection / row-span decomposition
# TEST 142      -> internal hidden-state direction measurement
# TEST 192-197  -> downstream transport / layer-local support analysis
# TEST 205      -> associative KEY→VALUE routing
# TEST 210      -> contextual ΔO = h(S,R,O)-h(S,R)
# TEST 211      -> COMMON completion removal -> isolated OBJECT_IDENTITY
# TEST 212      -> keyed identity transport; geometry present, behavioral retrieval absent
# TEST 213      -> MOTOR-OFF localization: full sequence × layer identity distribution
# --------------------------------------------------------------------------------------------------
# TEST212 MODEL/SYSTEM BASELINE PRESERVED | NO INJECTION | NO WEIGHT CHANGE
# MEASURE: RESIDUAL OUTPUT / ATTENTION OUTPUT / MLP OUTPUT
# SITES: SUBJECT / RELATION / OBJECT TOKENS / OBJECT-END / FINAL TOKEN
# ==================================================================================================
import os,sys,math,random,subprocess,importlib.util,re
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=213
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
ALT_OBJECTS=["the golden necklace","the iron dagger","the crystal mirror","the wooden mask","the scarlet book","the ivory ring","the copper bell","the black feather"]
M=len(FACTS);COMMON_RANK=2
print("="*128);print("TEST 213 — SEQUENCE-DISTRIBUTED OBJECT-IDENTITY X-RAY");print("="*128)
print("LINEAGE: TEST103 -> TEST142 -> TEST192-197 -> TEST205 -> TEST210 -> TEST211 -> TEST212 -> TEST213")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,"| MOTOR: OFF")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/16] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch.")
FP_T=[layers[0].self_attn.q_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def ids(x):return tok(x,add_special_tokens=False).input_ids
def subseq(hay,needle):
    if not needle:return []
    out=[]
    for i in range(len(hay)-len(needle)+1):
        if hay[i:i+len(needle)]==needle:out.append(list(range(i,i+len(needle))))
    return out
def last_span(full,text):
    a=subseq(full,ids(text))
    if a:return a[-1]
    a=subseq(full,ids(" "+text))
    return a[-1] if a else []
def unwrap(out):
    if isinstance(out,tuple):return out[0]
    if hasattr(out,"last_hidden_state"):return out.last_hidden_state
    return out
print("[2/16] Full-sequence capture engine...")
@torch.inference_mode()
def capture(text):
    rendered=chat(text);e=tok(rendered,return_tensors="pt",add_special_tokens=False).to(DEVICE);T=e.input_ids.shape[1]
    residual=[None]*TOTAL;attn=[None]*TOTAL;mlp=[None]*TOTAL;hs=[]
    def mk(store,L):
        def hk(m,args,out):store[L]=unwrap(out)[0].float().detach().clone()
        return hk
    for L in range(TOTAL):
        hs.append(layers[L].register_forward_hook(mk(residual,L)))
        hs.append(layers[L].self_attn.register_forward_hook(mk(attn,L)))
        hs.append(layers[L].mlp.register_forward_hook(mk(mlp,L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return {"ids":e.input_ids[0].detach().cpu().tolist(),"res":residual,"attn":attn,"mlp":mlp,"T":T,"rendered":rendered}
def fact_text(s,r,o):return f"Fact: {s} {r} {o}."
print("[3/16] Target/wrong matched captures...")
CAP={}
for qi,(s,r,o) in enumerate(FACTS):
    objs=[o,FACTS[(qi+1)%M][2]]+ALT_OBJECTS
    for oi,obj in enumerate(objs):CAP[(qi,oi)]=capture(fact_text(s,r,obj))
    print(f"Q{qi+1} captures={len(objs)}")
print("[4/16] Semantic token-site maps...")
SITES={}
for qi,(s,r,o) in enumerate(FACTS):
    c=CAP[(qi,0)];full=c["ids"];spS=last_span(full,s);spR=last_span(full,r);spO=last_span(full,o)
    if not spS or not spR or not spO:raise RuntimeError(f"Token alignment failed Q{qi+1}: S={spS} R={spR} O={spO}")
    SITES[qi]={"SUBJECT":spS,"RELATION":spR,"OBJECT":spO,"OBJECT_END":[spO[-1]],"FINAL":[len(full)-1]}
    print(f"Q{qi+1} S={spS} R={spR} O={spO} O_END={spO[-1]} FINAL={len(full)-1}")
print("[5/16] Sequence alignment check...")
# Prefix through relation must be identical; object suffixes may differ in token count.
for qi,(s,r,o) in enumerate(FACTS):
    t=CAP[(qi,0)]["ids"];rspan=SITES[qi]["RELATION"];end=rspan[-1]+1
    ok=all(CAP[(qi,j)]["ids"][:end]==t[:end] for j in range(1,len(ALT_OBJECTS)+2))
    print(f"Q{qi+1} common-prefix-through-relation={'PASS' if ok else 'FAIL'}")
    if not ok:raise RuntimeError("Matched prefix alignment failed.")
print("[6/16] Representation tensors...")
REP=["res","attn","mlp"]
def pool_site(c,positions,L,rep):
    z=c[rep][L];pp=[p for p in positions if 0<=p<z.shape[0]]
    if not pp:return torch.zeros(H,device=DEVICE)
    return z[pp].mean(0)
def aligned_object_site(c,obj):
    sp=last_span(c["ids"],obj)
    if not sp:raise RuntimeError("Object token alignment failed.")
    return sp
print("[7/16] Target-vs-wrong identity maps...")
# Identity difference at semantic sites. For OBJECT sites each object's own aligned token span is pooled.
DIFF={};COS={};REL={};RAWN={}
for qi,(s,r,o) in enumerate(FACTS):
    ct=CAP[(qi,0)];cw=CAP[(qi,1)];ow=FACTS[(qi+1)%M][2]
    for rep in REP:
        for site in ["SUBJECT","RELATION","OBJECT","OBJECT_END","FINAL"]:
            vals=[]
            for L in range(TOTAL):
                if site=="OBJECT":
                    a=pool_site(ct,aligned_object_site(ct,o),L,rep);b=pool_site(cw,aligned_object_site(cw,ow),L,rep)
                elif site=="OBJECT_END":
                    a=pool_site(ct,[aligned_object_site(ct,o)[-1]],L,rep);b=pool_site(cw,[aligned_object_site(cw,ow)[-1]],L,rep)
                elif site=="FINAL":
                    a=pool_site(ct,[ct[rep][L].shape[0]-1],L,rep);b=pool_site(cw,[cw[rep][L].shape[0]-1],L,rep)
                else:
                    pos=SITES[qi][site];a=pool_site(ct,pos,L,rep);b=pool_site(cw,pos,L,rep)
                d=a-b;vals.append(d)
                COS[(qi,rep,site,L)]=float(unit(a[None])[0]@unit(b[None])[0]);REL[(qi,rep,site,L)]=float(d.norm()/a.norm().clamp_min(EPS));RAWN[(qi,rep,site,L)]=float(d.norm())
            DIFF[(qi,rep,site)]=torch.stack(vals)
print("[8/16] Multi-distractor identity selectivity...")
# Compare target identity difference against all alternative-object differences.
SELECT={};PAIR={}
for qi,(s,r,o) in enumerate(FACTS):
    ct=CAP[(qi,0)]
    for rep in REP:
        for site in ["OBJECT","OBJECT_END","FINAL"]:
            for L in range(TOTAL):
                tv=[]
                for j,obj in enumerate([FACTS[(qi+1)%M][2]]+ALT_OBJECTS,1):
                    cj=CAP[(qi,j)]
                    if site=="OBJECT":
                        a=pool_site(ct,aligned_object_site(ct,o),L,rep);b=pool_site(cj,aligned_object_site(cj,obj),L,rep)
                    elif site=="OBJECT_END":
                        a=pool_site(ct,[aligned_object_site(ct,o)[-1]],L,rep);b=pool_site(cj,[aligned_object_site(cj,obj)[-1]],L,rep)
                    else:
                        a=pool_site(ct,[ct[rep][L].shape[0]-1],L,rep);b=pool_site(cj,[cj[rep][L].shape[0]-1],L,rep)
                    tv.append(unit((a-b)[None])[0])
                V=torch.stack(tv);C=V@V.T;off=C[~torch.eye(C.shape[0],dtype=torch.bool,device=DEVICE)]
                SELECT[(qi,rep,site,L)]=float(off.abs().mean());PAIR[(qi,rep,site,L)]=float(off.mean())
print("[9/16] Common-mode / identity residual at every site...")
IDRES={};IDSEP={};IDFRAC={}
for qi,(s,r,o) in enumerate(FACTS):
    ct=CAP[(qi,0)]
    for rep in REP:
        for site in ["OBJECT","OBJECT_END","FINAL"]:
            for L in range(TOTAL):
                alts=[]
                for j,obj in enumerate([FACTS[(qi+1)%M][2]]+ALT_OBJECTS,1):
                    cj=CAP[(qi,j)]
                    if site=="OBJECT":
                        a=pool_site(ct,aligned_object_site(ct,o),L,rep);b=pool_site(cj,aligned_object_site(cj,obj),L,rep)
                    elif site=="OBJECT_END":
                        a=pool_site(ct,[aligned_object_site(ct,o)[-1]],L,rep);b=pool_site(cj,[aligned_object_site(cj,obj)[-1]],L,rep)
                    else:
                        a=pool_site(ct,[ct[rep][L].shape[0]-1],L,rep);b=pool_site(cj,[cj[rep][L].shape[0]-1],L,rep)
                    alts.append(a-b)
                X=torch.stack(alts);mu=X.mean(0);Xc=X-mu;_,_,Vh=torch.linalg.svd(Xc,full_matrices=False);U=Vh[:min(COMMON_RANK,Vh.shape[0])]
                target=X[0];centered=target-mu;res=centered-(centered@U.T)@U
                IDRES[(qi,rep,site,L)]=res;IDFRAC[(qi,rep,site,L)]=float(res.norm()/target.norm().clamp_min(EPS))
                wrong=[]
                for j in range(1,X.shape[0]):
                    c=X[j]-mu;r=c-(c@U.T)@U;wrong.append(float(unit(res[None])[0]@unit(r[None])[0]))
                IDSEP[(qi,rep,site,L)]=max(wrong) if wrong else 0.
print("[10/16] Layer/site ranking...")
RANK=[]
for rep in REP:
    for site in ["OBJECT","OBJECT_END","FINAL"]:
        for L in range(TOTAL):
            rel=np.mean([REL[(q,rep,site,L)] for q in range(M)])
            frac=np.mean([IDFRAC[(q,rep,site,L)] for q in range(M)])
            wrong=np.mean([IDSEP[(q,rep,site,L)] for q in range(M)])
            diversity=np.mean([SELECT[(q,rep,site,L)] for q in range(M)])
            score=frac*(1.-max(-1.,min(1.,wrong)))/2
            RANK.append((score,rep,site,L,rel,frac,wrong,diversity))
RANK.sort(reverse=True,key=lambda x:x[0])
for x in RANK[:20]:print(f"{x[1]:4s} {x[2]:10s} L{x[3]:02d} score={x[0]:.4f} relΔ={x[4]*100:7.3f}% idFrac={x[5]:.4f} bestWrongIDcos={x[6]:+.4f} pairAbsCos={x[7]:.4f}")
print("[11/16] Token-position X-Ray...")
for qi in range(M):
    print(f"Q{qi+1}")
    for L in [0,4,8,11,15,19,23,27]:
        vals=[]
        for site in ["SUBJECT","RELATION","OBJECT","OBJECT_END","FINAL"]:
            vals.append(f"{site}={REL[(qi,'res',site,L)]*100:.2f}%")
        print(f" L{L:02d} "+" ".join(vals))
print("[12/16] Representation-site summary...")
for rep in REP:
    print("\n"+rep.upper())
    for site in ["OBJECT","OBJECT_END","FINAL"]:
        best=max(range(TOTAL),key=lambda L:np.mean([IDFRAC[(q,rep,site,L)] for q in range(M)]))
        print(f" {site:10s} bestL={best:02d} relΔ={np.mean([REL[(q,rep,site,best)] for q in range(M)])*100:7.3f}% idFrac={np.mean([IDFRAC[(q,rep,site,best)] for q in range(M)]):.4f} bestWrongIDcos={np.mean([IDSEP[(q,rep,site,best)] for q in range(M)]):+.4f}")
print("[13/16] Per-fact best identity carriers...")
BEST={}
for qi in range(M):
    arr=[]
    for rep in REP:
        for site in ["OBJECT","OBJECT_END","FINAL"]:
            for L in range(TOTAL):
                score=IDFRAC[(qi,rep,site,L)]*(1.-max(-1.,min(1.,IDSEP[(qi,rep,site,L)])))/2
                arr.append((score,rep,site,L))
    arr.sort(reverse=True);BEST[qi]=arr[0]
    x=arr[0];print(f"Q{qi+1} BEST={x[1]}/{x[2]}/L{x[3]:02d} score={x[0]:.4f} idFrac={IDFRAC[(qi,x[1],x[2],x[3])]:.4f} bestWrongIDcos={IDSEP[(qi,x[1],x[2],x[3])]:+.4f}")
print("[14/16] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[15/16] RESULTS")
print("\n"+"="*128);print("TEST 213 RESULTS");print("="*128)
print("LINEAGE: TEST103 -> TEST142 -> TEST192-197 -> TEST205 -> TEST210 -> TEST211 -> TEST212 -> TEST213")
print("MODE: MOTOR OFF | WEIGHTS FROZEN | FULL-SEQUENCE LOCALIZATION ONLY")
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("\nTOP GLOBAL IDENTITY CARRIERS")
for x in RANK[:12]:print(f"{x[1]:4s} {x[2]:10s} L{x[3]:02d} score={x[0]:.4f} relΔ={x[4]*100:7.3f}% idFrac={x[5]:.4f} bestWrongIDcos={x[6]:+.4f} pairAbsCos={x[7]:.4f}")
print("\nPER-FACT BEST")
for qi,x in BEST.items():print(f"Q{qi+1} {x[1]}/{x[2]}/L{x[3]:02d} score={x[0]:.4f} idFrac={IDFRAC[(qi,x[1],x[2],x[3])]:.4f} bestWrongIDcos={IDSEP[(qi,x[1],x[2],x[3])]:+.4f}")
print("\nRESIDUAL TOKEN-SITE RELATIVE DISPLACEMENT")
for qi in range(M):
    print(f"Q{qi+1}")
    for L in [0,8,11,15,19,27]:
        print(f" L{L:02d} SUBJECT={REL[(qi,'res','SUBJECT',L)]*100:7.3f}% RELATION={REL[(qi,'res','RELATION',L)]*100:7.3f}% OBJECT={REL[(qi,'res','OBJECT',L)]*100:7.3f}% OBJECT_END={REL[(qi,'res','OBJECT_END',L)]*100:7.3f}% FINAL={REL[(qi,'res','FINAL',L)]*100:7.3f}%")
print("\nREPRESENTATION SUMMARY")
for rep in REP:
    for site in ["OBJECT","OBJECT_END","FINAL"]:
        best=max(range(TOTAL),key=lambda L:np.mean([IDFRAC[(q,rep,site,L)] for q in range(M)]))
        print(f"{rep:4s} {site:10s} bestL={best:02d} idFrac={np.mean([IDFRAC[(q,rep,site,best)] for q in range(M)]):.4f} bestWrongIDcos={np.mean([IDSEP[(q,rep,site,best)] for q in range(M)]):+.4f}")
print("-"*128)
print("Weights: PASS | Injection: NONE | L0-L27 observation only")
print("TEST212 model/system/facts preserved; TEST213 removes intervention and localizes object identity across sequence, depth and representation site")
print("Primary question: is object identity concentrated at object-token/attention/MLP sites rather than the final prompt-token residual used by prior forge tests?")
print("No behavioral retrieval claim is made by this assay.")
print("="*128);print("[16/16] TEST 213 COMPLETE")



