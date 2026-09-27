# ==================================================================================================
# TEST 219 — CAUSAL VALUE-HEAD PACKET TRANSPLANT
# V/L08/H00 DIRECT ATTENTION-VALUE INTERVENTION × KV-CONDITIONED BLIND RETRIEVAL
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219
# TEST218 MODEL/SYSTEM/FACTS PRESERVED | WEIGHTS FROZEN
# PRE-REGISTERED PRIMARY: V/L08/H00 | SECONDARY: V/L07/H02 | K-CONTROL: K/L10/H00
# SOURCE: position-matched OBJECT_END packet | TARGET: blind-query answer-boundary prefill token
# DECODE INTERVENTION OFF | CANDIDATES NEVER ENTER INTERVENTION
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=219;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;EPS=1e-8;COMMON_RANK=2
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards"),("Varek Tonn","holds"),("Lira Mesk","stores"),("Korin Drel","protects"),("Taren Vosk","carries")]
OBJECT_POOL=["the amber compass","the silver lantern","the violet key","the bronze sphere","the golden necklace","the iron dagger","the crystal mirror","the wooden mask","the scarlet book","the ivory ring","the copper bell","the black feather","the green bottle","the marble token","the yellow candle","the velvet ribbon","the cobalt badge","the wooden tablet","the silver mirror","the bronze pendant","the crimson feather","the copper whistle","the ivory button","the purple candle"]
SCALES=[.25,.50,1.00];M=len(FACTS);PRIMARY=("V",8,0);SECONDARY=("V",7,2);KCTRL=("K",10,0)
print("="*128);print("TEST 219 — CAUSAL VALUE-HEAD PACKET TRANSPLANT");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219");print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/20] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;NH=model.config.num_attention_heads;NKV=model.config.num_key_value_heads;HD=H//NH
if len(layers)!=TOTAL or H!=H_EXPECT or NKV!=4 or HD!=128:raise RuntimeError("Architecture mismatch.")
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
def qform(s,r):
    mp={"keeps":"keep","carries":"carry","owns":"own","guards":"guard","holds":"hold","stores":"store","protects":"protect"}
    return f"What does {s} {mp[r]}?"
print("[2/20] Position-matched source banks...")
BANK={}
for qi,(s,r) in enumerate(FACTS):
    groups={}
    for o in OBJECT_POOL:
        full=tok(chat(fact_text(s,r,o)),add_special_tokens=False).input_ids;sp=last_span(full,o)
        if sp:groups.setdefault(len(sp),[]).append((o,full,sp))
    valid=[v for v in groups.values() if len(v)>=10]
    if not valid:raise RuntimeError(f"No matched bank Q{qi+1}")
    valid.sort(key=lambda v:(len(v),len(v[0][2])),reverse=True);chosen=valid[0][:10];ti=qi%len(chosen);chosen=[chosen[ti]]+[x for j,x in enumerate(chosen) if j!=ti]
    st={x[2][0] for x in chosen};en={x[2][-1] for x in chosen};ln={len(x[2]) for x in chosen};pre=chosen[0][2][0]
    if len(st)!=1 or len(en)!=1 or len(ln)!=1 or not all(x[1][:pre]==chosen[0][1][:pre] for x in chosen):raise RuntimeError(f"Position/prefix mismatch Q{qi+1}")
    BANK[qi]=chosen;print(f"Q{qi+1} TARGET={chosen[0][0]} tokens={len(chosen[0][2])} start={chosen[0][2][0]} end={chosen[0][2][-1]} PREFIX=PASS")
print("[3/20] Capture source K/V...")
NEED={("V",8),("V",7),("K",10)}
@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt",add_special_tokens=False).to(DEVICE);S={};hs=[]
    def mk(rep,L):
        def hk(m,args,out):S[(rep,L)]=out[0].float().detach().clone()
        return hk
    for rep,L in NEED:
        mod=layers[L].self_attn.v_proj if rep=="V" else layers[L].self_attn.k_proj;hs.append(mod.register_forward_hook(mk(rep,L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return e.input_ids[0].detach().cpu().tolist(),S
CAP={}
for qi,(s,r) in enumerate(FACTS):
    for j,(o,_,_) in enumerate(BANK[qi]):CAP[(qi,j)]=capture(fact_text(s,r,o))
    print(f"Q{qi+1} captures={len(BANK[qi])}")
print("[4/20] Strict source alignment...")
POS={}
for qi in range(M):
    ref=CAP[(qi,0)][0];sp=last_span(ref,BANK[qi][0][0]);POS[qi]=sp
    for j,(o,_,_) in enumerate(BANK[qi]):
        full=CAP[(qi,j)][0];sj=last_span(full,o)
        if len(sj)!=len(sp) or sj[0]!=sp[0] or sj[-1]!=sp[-1] or full[:sj[0]]!=ref[:sp[0]]:raise RuntimeError(f"Alignment fail Q{qi+1}")
    print(f"Q{qi+1} PASS start={sp[0]} end={sp[-1]}")
def head(x,pos,h):return x[pos].reshape(NKV,HD)[h]
print("[5/20] Source packet forge...")
PACK={};WRONG={}
for rep,L,h in [PRIMARY,SECONDARY,KCTRL]:
    for qi in range(M):
        pos=POS[qi][-1];T=head(CAP[(qi,0)][1][(rep,L)],pos,h);A=[head(CAP[(qi,j)][1][(rep,L)],pos,h) for j in range(1,len(BANK[qi]))]
        X=torch.stack([T-a for a in A]);mu=X.mean(0);Xc=X-mu;_,_,Vh=torch.linalg.svd(Xc,full_matrices=False);U=Vh[:min(COMMON_RANK,Vh.shape[0])]
        c=X[0]-mu;r=c-(c@U.T)@U;PACK[(rep,L,h,qi)]=unit(r[None])[0]
        c2=X[1]-mu;r2=c2-(c2@U.T)@U;WRONG[(rep,L,h,qi)]=unit(r2[None])[0]
print("[6/20] Blind queries...")
QENC=[];ANS=[]
for qi,(s,r) in enumerate(FACTS):
    q=qform(s,r);e=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE);QENC.append(e);ANS.append(BANK[qi][0][0])
    if any(o.lower() in q.lower() for o in OBJECT_POOL):raise RuntimeError("Target leakage.")
    print(f"Q{qi+1} {q} | slot={e.input_ids.shape[1]-1} | target={ANS[-1]}")
print("[7/20] Direct projection-head intervention...")
# Branches modify exactly one 128D K/V head at the final prefill token.
# Delta magnitude = scale * ||original head|| * unit(packet). Decode calls S=1 are untouched.
BRANCHES=["PRIMARY_V","WRONG_V","NEG_V","SECONDARY_V","K_CONTROL"]
def spec(qi,b):
    if b=="PRIMARY_V":rep,L,h=PRIMARY;v=PACK[(rep,L,h,qi)]
    elif b=="WRONG_V":rep,L,h=PRIMARY;v=WRONG[(rep,L,h,qi)]
    elif b=="NEG_V":rep,L,h=PRIMARY;v=-PACK[(rep,L,h,qi)]
    elif b=="SECONDARY_V":rep,L,h=SECONDARY;v=PACK[(rep,L,h,qi)]
    elif b=="K_CONTROL":rep,L,h=KCTRL;v=PACK[(rep,L,h,qi)]
    else:raise ValueError(b)
    return rep,L,h,v
def install(qi,b,scale,tele=None):
    rep,L,h,v=spec(qi,b);mod=layers[L].self_attn.v_proj if rep=="V" else layers[L].self_attn.k_proj
    def hk(m,args,out):
        if out.ndim!=3 or out.shape[1]<=1:return out
        y=out.clone();z=y[:,-1,:].reshape(y.shape[0],NKV,HD);orig=z[:,h,:];n=orig.float().norm(dim=-1,keepdim=True).clamp_min(EPS);delta=float(scale)*n*v.to(orig.device,dtype=torch.float32)[None,:]
        z[:,h,:]=(orig.float()+delta).to(orig.dtype)
        if tele is not None:tele.update({"rep":rep,"layer":L,"head":h,"scale":float(scale),"orig_norm":float(n.mean()),"delta_norm":float(delta.norm(dim=-1).mean()),"rel":float(delta.norm(dim=-1).mean()/n.mean())})
        return y
    return mod.register_forward_hook(hk)
print("[8/20] Prefill → frozen KV scorer...")
@torch.inference_mode()
def prefill(qi,scale=0.,b="PRIMARY_V"):
    h=None;te={}
    try:
        if scale>0:h=install(qi,b,scale,te)
        o=model(**QENC[qi],use_cache=True,return_dict=True)
    finally:
        if h is not None:h.remove()
    return o.logits[:,-1,:].float(),o.past_key_values,te
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="PRIMARY_V"):
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)[0]
    logits,pkv,_=prefill(qi,scale,b);vals=[]
    for i,t in enumerate(y):
        vals.append(torch.log_softmax(logits[0],-1)[t])
        if i<y.numel()-1:
            o=model(input_ids=t.view(1,1),past_key_values=pkv,use_cache=True,return_dict=True);logits=o.logits[:,-1,:].float();pkv=o.past_key_values
    z=torch.stack(vals);return float(z.sum()),float(z.mean()),int(y.numel())
def margin(qi,scale=0.,b="PRIMARY_V"):
    ts,tm,tn=lp(qi,ANS[qi],scale,b);wrong=[lp(qi,BANK[qi][j][0],scale,b)[0] for j in range(1,4)];bw=max(wrong);return ts,tm,tn,bw,ts-bw
print("[9/20] Generation...")
@torch.inference_mode()
def generate(qi,scale=0.,b="PRIMARY_V",n=40):
    e=QENC[qi];p=e.input_ids.shape[1];h=None;te={}
    try:
        if scale>0:h=install(qi,b,scale,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        if h is not None:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
print("[10/20] PRIMARY_V dose sweep...")
BASE={};SWEEP={}
for qi in range(M):
    BASE[qi]=(generate(qi)[0],*margin(qi))
    for sc in SCALES:SWEEP[(qi,sc)]=(generate(qi,sc,"PRIMARY_V")[0],*margin(qi,sc,"PRIMARY_V"))
print("[11/20] Controls @ .50...")
RES={}
for b in BRANCHES:
    for qi in range(M):RES[(b,qi)]=(generate(qi,.5,b)[0],*margin(qi,.5,b))
print("[12/20] Selectivity...")
for qi in range(M):
    c=RES[("PRIMARY_V",qi)][5]
    print(f"Q{qi+1} PRIMARY={c:+.4f} WRONG Δ={c-RES[('WRONG_V',qi)][5]:+.4f} NEG Δ={c-RES[('NEG_V',qi)][5]:+.4f} SECONDARY Δ={c-RES[('SECONDARY_V',qi)][5]:+.4f} KCTRL Δ={c-RES[('K_CONTROL',qi)][5]:+.4f}")
print("[13/20] First target token...")
FIRST={}
for qi in range(M):
    tid=tok(ANS[qi],add_special_tokens=False).input_ids[0];lv,_,_=prefill(qi);base=float(torch.log_softmax(lv[0],-1)[tid]);row={"VANILLA":base}
    for b in BRANCHES:
        lv,_,_=prefill(qi,.5,b);row[b]=float(torch.log_softmax(lv[0],-1)[tid])
    FIRST[qi]=row;print(f"Q{qi+1} vanilla={base:+.4f} primary={row['PRIMARY_V']:+.4f} Δ={row['PRIMARY_V']-base:+.4f}")
print("[14/20] Projection telemetry...")
for b in BRANCHES:
    _,_,te=prefill(0,.5,b);print(f"{b:12s} {te}")
print("[15/20] Output X-Ray Q1...")
@torch.inference_mode()
def xray(qi,b,scale=.5):
    e=QENC[qi];pos=e.input_ids.shape[1]-1;A={};B={};hs=[]
    def caps(store):
        rr=[]
        for L in range(TOTAL):
            def mk(li):
                def hk(m,args,out):store[li]=(out[0] if isinstance(out,tuple) else out)[0,pos].float().detach().clone()
                return hk
            rr.append(layers[L].register_forward_hook(mk(L)))
        return rr
    hs=caps(A);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    ih=install(qi,b,scale);hs=caps(B);model(**e,use_cache=False,return_dict=True);ih.remove()
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[16/20] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[17/20] RESULTS")
print("\n"+"="*128);print("TEST 219 RESULTS");print("="*128)
print("MODE: DIRECT K/V HEAD INTERVENTION | PREFILL ONLY | DECODE INTERVENTION OFF | WEIGHTS FROZEN")
print("PRIMARY=V/L08/H00 | SECONDARY=V/L07/H02 | K-CONTROL=K/L10/H00")
for qi in range(M):print(f"M{qi+1}: {FACTS[qi][0]} | {FACTS[qi][1]} | {ANS[qi]}")
print("\nPRIMARY_V DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {qform(*FACTS[qi])}");print(f" VANILLA targetLP={b[1]:+.4f} meanTok={b[2]:+.4f} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" V {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
print("\nCONTROLS @ .50")
for br in BRANCHES:
    ms=[];print("\n"+br)
    for qi in range(M):
        r=RES[(br,qi)];ms.append(r[5]);print(f" Q{qi+1} targetLP={r[1]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" mean_margin={np.mean(ms):+.4f}")
print("\nSELECTIVITY")
for qi in range(M):
    c=RES[("PRIMARY_V",qi)][5];print(f"Q{qi+1} PRIMARY={c:+.4f} vs WRONG Δ={c-RES[('WRONG_V',qi)][5]:+.4f} vs NEG Δ={c-RES[('NEG_V',qi)][5]:+.4f} vs SECONDARY Δ={c-RES[('SECONDARY_V',qi)][5]:+.4f} vs KCTRL Δ={c-RES[('K_CONTROL',qi)][5]:+.4f}")
print("\nFIRST TARGET TOKEN @ .50")
for qi,r in FIRST.items():print(f"Q{qi+1} VANILLA={r['VANILLA']:+.4f} PRIMARY={r['PRIMARY_V']:+.4f} Δ={r['PRIMARY_V']-r['VANILLA']:+.4f} WRONG={r['WRONG_V']:+.4f} NEG={r['NEG_V']:+.4f}")
print("\nQ1 OUTPUT X-RAY @ .50")
for br,x in XR.items():print(f"{br:12s} L07={x[7]*100:7.3f}% L08={x[8]*100:7.3f}% L10={x[10]*100:7.3f}% L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("-"*128)
print("Weights: PASS | Candidate answers never enter source forge or prefill intervention | Decode intervention: ZERO")
print("TEST219 is an experimental direct attention-projection intervention, not the canonical SEASC residual-stream motor")
print("PASS requires PRIMARY_V selective target advantage over WRONG_V / NEG_V and behavioral or LP evidence beyond displacement alone")
print("="*128);print("[18/20] TEST 219 COMPLETE");print("[19/20] WEIGHTS VERIFIED");print("[20/20] END")



