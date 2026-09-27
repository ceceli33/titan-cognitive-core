# ==================================================================================================
# TEST 221 — PAIRED K/V ADDRESS-BINDING TRANSPLANT
# CORRECT K + CORRECT V × K-ONLY × V-ONLY × CROSSED/SHUFFLED CONTROLS
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221
# TEST220 MODEL/SYSTEM/FACTS PRESERVED | WEIGHTS FROZEN
# PRIMARY PATH: L08 / KVH00 | SOURCE: OBJECT_END | TARGET: BLIND ANSWER-BOUNDARY PREFILL SLOT
# PREFILL ONLY | DECODE INTERVENTION OFF | CANDIDATES NEVER ENTER INTERVENTION
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=221;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;EPS=1e-8;L=8;KVH=0;SCALES=[.25,.50,1.00]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Rovan Tesk","keeps","the amber compass"),("Mira Veln","carries","the silver lantern"),("Dalen Quor","owns","the violet key"),("Sorin Kelm","guards","the bronze sphere"),("Varek Tonn","holds","the golden necklace"),("Lira Mesk","stores","the iron dagger"),("Korin Drel","protects","the crystal mirror"),("Taren Vosk","carries","the wooden mask")]
M=len(FACTS);OBJECTS=[x[2] for x in FACTS]
def qform(s,r):
    mp={"keeps":"keep","carries":"carry","owns":"own","guards":"guard","holds":"hold","stores":"store","protects":"protect"}
    return f"What does {s} {mp[r]}?"
print("="*128);print("TEST 221 — PAIRED K/V ADDRESS-BINDING TRANSPLANT");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221");print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/20] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;NH=model.config.num_attention_heads;NKV=model.config.num_key_value_heads;HD=H//NH;GROUP=NH//NKV
if len(layers)!=TOTAL or H!=H_EXPECT or NKV!=4 or HD!=128:raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")
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
print("[2/20] Source K/V packets...")
@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt",add_special_tokens=False).to(DEVICE);S={};A=layers[L].self_attn
    hk=A.k_proj.register_forward_hook(lambda m,a,o:S.update(K=o[0].float().detach().clone()))
    hv=A.v_proj.register_forward_hook(lambda m,a,o:S.update(V=o[0].float().detach().clone()))
    try:model(**e,use_cache=False,return_dict=True)
    finally:hk.remove();hv.remove()
    return e.input_ids[0].tolist(),S
SRC={}
for qi,(s,r,o) in enumerate(FACTS):
    full,S=capture(fact_text(s,r,o));sp=last_span(full,o)
    if not sp:raise RuntimeError(f"Object alignment Q{qi+1}")
    K=S["K"][sp[-1]].reshape(NKV,HD)[KVH];V=S["V"][sp[-1]].reshape(NKV,HD)[KVH]
    SRC[qi]={"K":unit(K[None])[0],"V":unit(V[None])[0],"KRAW":K,"VRAW":V,"pos":sp[-1]}
    print(f"Q{qi+1} object={o} end={sp[-1]} Knorm={K.norm():.4f} Vnorm={V.norm():.4f}")
print("[3/20] Wrong/shuffled packet map...")
WRONG={qi:(qi+1)%M for qi in range(M)};SHUFF={qi:(qi+3)%M for qi in range(M)}
for qi in range(M):print(f"Q{qi+1} correct={OBJECTS[qi]} wrong={OBJECTS[WRONG[qi]]} shuffled={OBJECTS[SHUFF[qi]]}")
print("[4/20] Blind queries...")
QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    q=qform(s,r);e=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if any(x.lower() in q.lower() for x in OBJECTS):raise RuntimeError("Target leakage.")
    QENC.append(e);print(f"Q{qi+1} {q} | answer_slot={e.input_ids.shape[1]-1} | target={o}")
print("[5/20] Branch definitions...")
BRANCHES=["KV_CORRECT","K_ONLY","V_ONLY","WRONGK_CORRECTV","CORRECTK_WRONGV","KV_WRONG","KV_SHUFFLED","KV_NEG"]
def branch_packets(qi,b):
    c=SRC[qi];w=SRC[WRONG[qi]];s=SRC[SHUFF[qi]]
    if b=="KV_CORRECT":return c["K"],c["V"],True,True
    if b=="K_ONLY":return c["K"],c["V"],True,False
    if b=="V_ONLY":return c["K"],c["V"],False,True
    if b=="WRONGK_CORRECTV":return w["K"],c["V"],True,True
    if b=="CORRECTK_WRONGV":return c["K"],w["V"],True,True
    if b=="KV_WRONG":return w["K"],w["V"],True,True
    if b=="KV_SHUFFLED":return s["K"],s["V"],True,True
    if b=="KV_NEG":return -c["K"],-c["V"],True,True
    raise ValueError(b)
print("[6/20] Direct paired K/V intervention...")
# Equal branch scale: each active projection receives delta=scale*||original head||*unit(source packet).
# K-only/V-only deliberately isolate components; paired controls always modify both K and V.
def install(qi,b,scale,tele=None):
    K,V,useK,useV=branch_packets(qi,b);A=layers[L].self_attn;hs=[]
    def hk(m,args,out):
        if out.ndim!=3 or out.shape[1]<=1 or not useK:return out
        y=out.clone();z=y[:,-1,:].reshape(y.shape[0],NKV,HD);orig=z[:,KVH,:];n=orig.float().norm(dim=-1,keepdim=True).clamp_min(EPS);d=float(scale)*n*K.to(orig.device,dtype=torch.float32)[None,:];z[:,KVH,:]=(orig.float()+d).to(orig.dtype)
        if tele is not None:tele["K"]={"orig":float(n.mean()),"delta":float(d.norm(dim=-1).mean()),"rel":float(d.norm(dim=-1).mean()/n.mean())}
        return y
    def hv(m,args,out):
        if out.ndim!=3 or out.shape[1]<=1 or not useV:return out
        y=out.clone();z=y[:,-1,:].reshape(y.shape[0],NKV,HD);orig=z[:,KVH,:];n=orig.float().norm(dim=-1,keepdim=True).clamp_min(EPS);d=float(scale)*n*V.to(orig.device,dtype=torch.float32)[None,:];z[:,KVH,:]=(orig.float()+d).to(orig.dtype)
        if tele is not None:tele["V"]={"orig":float(n.mean()),"delta":float(d.norm(dim=-1).mean()),"rel":float(d.norm(dim=-1).mean()/n.mean())}
        return y
    hs.append(A.k_proj.register_forward_hook(hk));hs.append(A.v_proj.register_forward_hook(hv));return hs
print("[7/20] Prefill → frozen KV scorer...")
@torch.inference_mode()
def prefill(qi,scale=0.,b="KV_CORRECT"):
    hs=[];te={}
    try:
        if scale>0:hs=install(qi,b,scale,te)
        o=model(**QENC[qi],use_cache=True,return_dict=True)
    finally:
        for h in hs:h.remove()
    return o.logits[:,-1,:].float(),o.past_key_values,te
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="KV_CORRECT"):
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)[0];logits,pkv,_=prefill(qi,scale,b);vals=[]
    for i,t in enumerate(y):
        vals.append(torch.log_softmax(logits[0],-1)[t])
        if i<y.numel()-1:
            o=model(input_ids=t.view(1,1),past_key_values=pkv,use_cache=True,return_dict=True);logits=o.logits[:,-1,:].float();pkv=o.past_key_values
    z=torch.stack(vals);return float(z.sum()),float(z.mean()),int(y.numel())
def margin(qi,scale=0.,b="KV_CORRECT"):
    ts,tm,tn=lp(qi,OBJECTS[qi],scale,b);wl=[lp(qi,OBJECTS[j],scale,b)[0] for j in range(M) if j!=qi];bw=max(wl);return ts,tm,tn,bw,ts-bw
print("[8/20] Generation...")
@torch.inference_mode()
def generate(qi,scale=0.,b="KV_CORRECT",n=40):
    e=QENC[qi];p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=install(qi,b,scale,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
print("[9/20] Vanilla baseline...")
BASE={}
for qi in range(M):BASE[qi]=(generate(qi)[0],*margin(qi))
print("[10/20] Correct K+V dose sweep...")
SWEEP={}
for qi in range(M):
    for sc in SCALES:SWEEP[(qi,sc)]=(generate(qi,sc,"KV_CORRECT")[0],*margin(qi,sc,"KV_CORRECT"))
print("[11/20] Controls @ .50...")
RES={}
for b in BRANCHES:
    for qi in range(M):RES[(b,qi)]=(generate(qi,.5,b)[0],*margin(qi,.5,b))
print("[12/20] Pair selectivity...")
for qi in range(M):
    c=RES[("KV_CORRECT",qi)][5]
    print(f"Q{qi+1} KV={c:+.4f} K={RES[('K_ONLY',qi)][5]:+.4f} V={RES[('V_ONLY',qi)][5]:+.4f} WK+CV Δ={c-RES[('WRONGK_CORRECTV',qi)][5]:+.4f} CK+WV Δ={c-RES[('CORRECTK_WRONGV',qi)][5]:+.4f} WRONG Δ={c-RES[('KV_WRONG',qi)][5]:+.4f} SHUFF Δ={c-RES[('KV_SHUFFLED',qi)][5]:+.4f} NEG Δ={c-RES[('KV_NEG',qi)][5]:+.4f}")
print("[13/20] First target token...")
FIRST={}
for qi in range(M):
    tid=tok(OBJECTS[qi],add_special_tokens=False).input_ids[0];lv,_,_=prefill(qi);base=float(torch.log_softmax(lv[0],-1)[tid]);row={"VANILLA":base}
    for b in BRANCHES:
        lv,_,_=prefill(qi,.5,b);row[b]=float(torch.log_softmax(lv[0],-1)[tid])
    FIRST[qi]=row;print(f"Q{qi+1} vanilla={base:+.4f} KV={row['KV_CORRECT']:+.4f} Δ={row['KV_CORRECT']-base:+.4f}")
print("[14/20] Projection telemetry...")
for b in BRANCHES:
    _,_,te=prefill(0,.5,b);print(f"{b:18s} {te}")
print("[15/20] Output X-Ray Q1...")
@torch.inference_mode()
def xray(qi,b,scale=.5):
    e=QENC[qi];pos=e.input_ids.shape[1]-1;A={};B={}
    def cap(store):
        hs=[]
        for li in range(TOTAL):
            def mk(LI):
                def h(m,args,out):store[LI]=(out[0] if isinstance(out,tuple) else out)[0,pos].float().detach().clone()
                return h
            hs.append(layers[li].register_forward_hook(mk(li)))
        return hs
    hs=cap(A);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    ih=install(qi,b,scale);hs=cap(B);model(**e,use_cache=False,return_dict=True)
    for h in ih+hs:h.remove()
    return [float((B[i]-A[i]).norm()/A[i].norm().clamp_min(EPS)) for i in range(TOTAL)]
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[16/20] Mean branch statistics...")
STAT={}
for b in BRANCHES:
    margins=np.array([RES[(b,q)][5] for q in range(M)]);dlog=np.array([RES[(b,q)][1]-BASE[q][1] for q in range(M)])
    STAT[b]=(float(margins.mean()),float(dlog.mean()),int(sum(RES[(b,q)][5]>BASE[q][5] for q in range(M))))
    print(f"{b:18s} meanMargin={margins.mean():+.4f} meanΔtargetLP={dlog.mean():+.4f} marginImproved={STAT[b][2]}/{M}")
print("[17/20] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[18/20] RESULTS")
print("\n"+"="*128);print("TEST 221 RESULTS");print("="*128)
print("MODE: DIRECT PAIRED K/V INTERVENTION | L08/KVH00 | PREFILL ONLY | DECODE INTERVENTION OFF | WEIGHTS FROZEN")
print("\nKV_CORRECT DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {qform(FACTS[qi][0],FACTS[qi][1])} | TARGET={OBJECTS[qi]}")
    print(f" VANILLA targetLP={b[1]:+.4f} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" KV {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} margin={r[5]:+.4f} Δmargin={r[5]-b[5]:+.4f} | {r[0]}")
print("\nCONTROLS @ .50")
for br in BRANCHES:
    print("\n"+br)
    for qi in range(M):
        r=RES[(br,qi)];print(f" Q{qi+1} targetLP={r[1]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" meanMargin={STAT[br][0]:+.4f} meanΔtargetLP={STAT[br][1]:+.4f} improved={STAT[br][2]}/{M}")
print("\nPAIR SELECTIVITY @ .50")
for qi in range(M):
    c=RES[("KV_CORRECT",qi)][5]
    print(f"Q{qi+1} KV={c:+.4f} | vs K-only={c-RES[('K_ONLY',qi)][5]:+.4f} vs V-only={c-RES[('V_ONLY',qi)][5]:+.4f} vs WK+CV={c-RES[('WRONGK_CORRECTV',qi)][5]:+.4f} vs CK+WV={c-RES[('CORRECTK_WRONGV',qi)][5]:+.4f} vs WRONG={c-RES[('KV_WRONG',qi)][5]:+.4f} vs SHUFF={c-RES[('KV_SHUFFLED',qi)][5]:+.4f} vs NEG={c-RES[('KV_NEG',qi)][5]:+.4f}")
print("\nFIRST TARGET TOKEN @ .50")
for qi,r in FIRST.items():print(f"Q{qi+1} VANILLA={r['VANILLA']:+.4f} KV={r['KV_CORRECT']:+.4f} Δ={r['KV_CORRECT']-r['VANILLA']:+.4f} K={r['K_ONLY']:+.4f} V={r['V_ONLY']:+.4f} WRONG={r['KV_WRONG']:+.4f}")
print("\nQ1 OUTPUT X-RAY @ .50")
for br,x in XR.items():print(f"{br:18s} L07={x[7]*100:7.3f}% L08={x[8]*100:7.3f}% L09={x[9]*100:7.3f}% L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("-"*128)
print("Weights: PASS | Source facts absent from blind query | Candidate answers never enter intervention | Decode intervention: ZERO")
print("TEST221 is an experimental direct L08/KVH00 K/V projection intervention, not the canonical SEASC residual-stream motor")
print("PASS requires CORRECT K+V to outperform crossed/wrong/shuffled/negative controls; displacement alone is not PASS")
print("="*128);print("[19/20] TEST 221 COMPLETE");print("[20/20] END")



