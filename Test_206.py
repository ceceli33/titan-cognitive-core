# ==================================================================================================
# TEST 206 — ROUTED TEACHER-STATE TRANSPLANTATION
# TEST205 CAM ADDRESSING → MODEL-NATIVE TEACHER STATE → LAYERWISE TRANSPLANT
# CORRECT / WRONG / SHUFFLED TEACHER | SINGLE/PAIR/FULL LAYER ASSAY | BLIND MULTI-TOKEN RETRIEVAL
# AkbasCore SEASC lineage | Qwen2.5-7B-Instruct | L0-L19 ONLY | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=206
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8;BETA=12.
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
ALPHAS=[.10,.25,.50,.75,1.00];M=len(FACTS)
print("="*128);print("TEST 206 — ROUTED TEACHER-STATE TRANSPLANTATION");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/13] Model...")
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
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
@torch.inference_mode()
def cap(text,total=False):
    e=tok(chat(text),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);n=TOTAL if total else N
    z=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(n)]);del o;return z
def qforms(s,r):
    return [f"What does {s} keep?" if r=="keeps" else f"What does {s} carry?" if r=="carries" else f"What does {s} own?" if r=="owns" else f"What does {s} guard?",
            f"Which item does {s} keep?" if r=="keeps" else f"Which item does {s} carry?" if r=="carries" else f"Which item does {s} own?" if r=="owns" else f"Which item does {s} guard?",
            f"What object does {s} keep?" if r=="keeps" else f"What object does {s} carry?" if r=="carries" else f"What object does {s} own?" if r=="owns" else f"What object does {s} guard?",
            f"Name the item that {s} keeps." if r=="keeps" else f"Name the item that {s} carries." if r=="carries" else f"Name the item that {s} owns." if r=="owns" else f"Name the item that {s} guards.",
            f"What item is linked to {s} through the relation '{r}'?",
            f"Which object belongs in the relation '{s} {r} ___'?"]
QUEST=[qforms(s,r)[0] for s,r,o in FACTS]
for i,(s,r,o) in enumerate(FACTS):
    ow={w for w in re.findall(r"[a-z]+",o.lower()) if len(w)>2 and w!="the"}
    if ow&set(re.findall(r"[a-z]+",QUEST[i].lower())):raise RuntimeError("Question leakage.")
print("[2/13] TEST205 competitive key bank...")
KEY=[]
for s,r,o in FACTS:
    hp=torch.stack([cap(q) for q in qforms(s,r)])
    neg=[]
    for sj,rj,oj in FACTS:
        if sj!=s:neg+=qforms(sj,r)[:2]
        if rj!=r:neg+=qforms(s,rj)[:2]
    KEY.append(unit(hp.mean(0)-torch.stack([cap(q) for q in neg]).mean(0)))
KEY=torch.stack(KEY,1)
print("KEY:",tuple(KEY.shape))
print("[3/13] Competitive layer profiling...")
QH=torch.stack([cap(q) for q in QUEST]);SCORE=torch.einsum("mlh,lkh->mlk",unit(QH),KEY)
DISC=torch.zeros(N,device=DEVICE);ACC=torch.zeros(N,device=DEVICE)
for L in range(N):
    cor=torch.stack([SCORE[i,L,i] for i in range(M)])
    wrong=torch.stack([torch.cat([SCORE[i,L,:i],SCORE[i,L,i+1:]]).max() for i in range(M)])
    DISC[L]=(cor-wrong).mean();ACC[L]=sum(int(torch.argmax(SCORE[i,L]).item()==i) for i in range(M))/M
MASK=(ACC>=.75)&(DISC>0)
if not bool(MASK.any()):
    top=torch.topk(DISC,min(3,N)).indices;MASK[:]=False;MASK[top]=True;print("WARNING: competitive fallback top-3.")
ACTIVE=[L for L in range(N) if bool(MASK[L])]
for L in range(N):print(f"L{L:02d} top1={ACC[L]:.2f} margin={DISC[L]:+.5f} {'ON' if MASK[L] else 'OFF'}")
print("ACTIVE:",ACTIVE)
print("[4/13] Frozen question-only routing...")
ROUTES=[]
for qi,q in enumerate(QUEST):
    h=unit(QH[qi]);sc=torch.einsum("lh,lmh->lm",h,KEY);a=torch.softmax(BETA*sc,dim=-1);ROUTES.append((h,sc,a))
    av=a[ACTIVE].mean(0);print(f"Q{qi+1} expected=M{qi+1} predicted=M{int(av.argmax())+1} weights="+str([round(float(x),4) for x in av]))
print("[5/13] Position-aligned teacher-state bank...")
# Teacher state is captured at exactly the same final question token; only preceding context carries the fact.
TEACH=torch.empty((M,M,N,H),device=DEVICE) # query x teacher-memory x layer x hidden
for qi,(sq,rq,oq) in enumerate(FACTS):
    q=QUEST[qi]
    for mi,(s,r,o) in enumerate(FACTS):
        # Same query endpoint for all teacher memories; context slot is the only changed payload.
        TEACH[qi,mi]=cap(f"Context: {s} {r} {o}.\nQuestion: {q}")
print("TEACH:",tuple(TEACH.shape))
print("[6/13] Teacher geometry...")
for qi in range(M):
    ds=[]
    for L in ACTIVE:
        b=QH[qi,L];t=TEACH[qi,qi,L];ds.append(float((t-b).norm()/b.norm().clamp_min(EPS)))
    print(f"Q{qi+1} correct-teacher mean relative distance={np.mean(ds)*100:.3f}%")
PERM=[1,2,3,0]
# Hooks interpolate ONLY the prompt-final token on prefill. Decode tokens are not replaced by a teacher state from another sequence.
# This avoids repeatedly forcing a fixed prompt state into every generated token.
def layer_set(mode):
    if mode=="FULL":return list(ACTIVE)
    if mode.startswith("L"):return [int(mode[1:])]
    if mode.startswith("P"):
        x=mode[1:].split("_");return [int(x[0]),int(x[1])]
    raise ValueError(mode)
def teacher_index(qi,branch):
    if branch=="CORRECT":return qi
    if branch=="WRONG":return (qi+1)%M
    if branch=="SHUFFLE":return PERM[qi]
    raise ValueError(branch)
def hooks(qi,alpha,branch="CORRECT",mode="FULL",tele=None):
    hs=[];use=set(layer_set(mode));mi=teacher_index(qi,branch)
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out
            if L not in use:return out
            # Transplant only when sequence has the original blind-prompt length (prefill).
            if raw.shape[1]<=1:return out
            y=raw.clone();cur=raw[:,-1,:].float();target=TEACH[qi,mi,L][None].expand(cur.shape[0],-1)
            new=cur+float(alpha)*(target-cur);y[:,-1,:]=new.to(raw.dtype)
            if tele is not None:
                rel=(new-cur).norm(dim=-1)/(cur.norm(dim=-1).clamp_min(EPS));tele[L]=float(rel.mean())
            return y if not isinstance(out,tuple) else (y,)+tuple(out[1:])
        return hk
    for L in range(N):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
@torch.inference_mode()
def generate(qi,alpha=0.,branch="CORRECT",mode="FULL",n=40):
    e=tok(chat(QUEST[qi]),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if alpha>0:hs=hooks(qi,alpha,branch,mode,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def seq_lp(qi,answer,alpha=0.,branch="CORRECT",mode="FULL"):
    p=tok(chat(QUEST[qi]),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)
    # Important: score autoregressively token-by-token so transplant occurs only on blind prompt prefill,
    # never on a prompt+candidate concatenation. Candidate answer cannot enter routing/transplant.
    hs=[]
    try:
        if alpha>0:hs=hooks(qi,alpha,branch,mode)
        o=model(input_ids=p,use_cache=True,return_dict=True);past=o.past_key_values;logits=o.logits[:,-1,:].float();total=0.
        for k in range(y.shape[1]):
            tid=y[:,k];total+=float(torch.log_softmax(logits,-1).gather(1,tid[:,None]).sum())
            if k+1<y.shape[1]:
                o=model(input_ids=tid[:,None],past_key_values=past,use_cache=True,return_dict=True);past=o.past_key_values;logits=o.logits[:,-1,:].float()
    finally:
        for h in hs:h.remove()
    return total
def margin(qi,alpha=0.,branch="CORRECT",mode="FULL"):
    t=seq_lp(qi,FACTS[qi][2],alpha,branch,mode)
    wrong=[seq_lp(qi,FACTS[j][2],alpha,branch,mode) for j in range(M) if j!=qi]
    return t,max(wrong),t-max(wrong)
print("[7/13] Full-layer transplant dose sweep...")
BASE=[];SWEEP={}
for qi in range(M):
    out,_=generate(qi);BASE.append((out,*margin(qi)))
    for a in ALPHAS:
        out,te=generate(qi,a,"CORRECT","FULL");SWEEP[(qi,a)]=(out,*margin(qi,a,"CORRECT","FULL"),te)
print("[8/13] Correct/Wrong/Shuffled teacher controls...")
CTRL={}
for qi in range(M):
    for b in ["CORRECT","WRONG","SHUFFLE"]:
        out,te=generate(qi,.50,b,"FULL");CTRL[(qi,b)]=(out,*margin(qi,.50,b,"FULL"),te)
print("[9/13] Layer localization assay...")
MODES=[]
for L in ACTIVE:MODES.append(f"L{L}")
for a,b in zip(ACTIVE[:-1],ACTIVE[1:]):MODES.append(f"P{a}_{b}")
MODES.append("FULL")
LOCAL={}
for mode in MODES:
    for qi in range(M):
        out,_=generate(qi,.50,"CORRECT",mode);LOCAL[(mode,qi)]=(out,*margin(qi,.50,"CORRECT",mode))
@torch.inference_mode()
def xray(qi,alpha=.50,branch="CORRECT",mode="FULL"):
    e=tok(chat(QUEST[qi]),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;A={};B={}
    def caps(store):
        hs=[]
        for L in range(TOTAL):
            def mk(li):
                def hk(m,args,out):
                    z=out[0] if isinstance(out,tuple) else out;store[li]=z[0,pos].float().detach().clone()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
        return hs
    hs=caps(A);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    # Register transplant BEFORE capture so B records post-transplant state at active layer.
    hs=hooks(qi,alpha,branch,mode)+caps(B);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
print("[10/13] X-Ray...")
XR={b:xray(0,.50,b,"FULL") for b in ["CORRECT","WRONG","SHUFFLE"]}
print("[11/13] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[12/13] RESULTS")
print("\n"+"="*128);print("TEST 206 RESULTS");print("="*128)
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("ACTIVE:",ACTIVE)
print("\nROUTING")
for qi in range(M):
    a=ROUTES[qi][2];av=a[ACTIVE].mean(0);print(f"Q{qi+1} expected=M{qi+1} predicted=M{int(av.argmax())+1} weights="+str([round(float(x),4) for x in av]))
print("\nFULL TRANSPLANT DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {QUEST[qi]}");print(f" VANILLA targetLP={b[1]:+.4f} bestWrong={b[2]:+.4f} margin={b[3]:+.4f} | {b[0]}")
    for a in ALPHAS:
        r=SWEEP[(qi,a)];rels=list(r[4].values());print(f" CORRECT α={a:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} bestWrong={r[2]:+.4f} margin={r[3]:+.4f} promptΔ={np.mean(rels)*100 if rels else 0:.3f}% | {r[0]}")
print("\nTEACHER CONTROLS @ α=.50")
for qi in range(M):
    print(f"\nQ{qi+1}")
    for br in ["CORRECT","WRONG","SHUFFLE"]:
        r=CTRL[(qi,br)];print(f" {br:7s} targetLP={r[1]:+.4f} margin={r[3]:+.4f} | {r[0]}")
print("\nLAYER LOCALIZATION @ α=.50")
for mode in MODES:
    margins=[];print("\n"+mode)
    for qi in range(M):
        r=LOCAL[(mode,qi)];margins.append(r[3]);print(f" Q{qi+1} targetLP={r[1]:+.4f} margin={r[3]:+.4f} | {r[0]}")
    print(f" mean_margin={np.mean(margins):+.4f}")
print("\nX-RAY Q1 @ α=.50 FULL")
for br,x in XR.items():print(f"{br:7s} L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("\nCORRECT LAYER X-RAY")
for L in range(TOTAL):print(f"L{L:02d} {XR['CORRECT'][L]*100:7.3f}% {'TRANSPLANT' if L in ACTIVE else 'OBSERVE' if L<N else 'MOTOR-OFF'}")
print("-"*128)
print("Weights: PASS | L20-L27 intervention: ZERO | Blind questions contain no target object")
print("Routing: TEST205 query-only frozen competitive key bank; candidate answers never enter routing")
print("Payload: raw model-native teacher state, no normalized VALUE vector")
print("Operator: h' = h + α(T-h) at selected L0-L19 prompt-final states")
print("Controls: CORRECT teacher / WRONG teacher / SHUFFLED teacher")
print("Evaluation: exact autoregressive multi-token SUM logP + generation + single/pair/full layer localization")
print("="*128);print("[13/13] TEST 206 COMPLETE")



