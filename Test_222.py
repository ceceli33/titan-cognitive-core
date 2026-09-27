# ==================================================================================================
# TEST 222 — ATTENTION-OUTPUT BINDING PACKET TRANSPLANT
# L08/KVH00 QH00-QH06 READOUT CONTRIBUTION × BLIND ANSWER-SLOT TRANSPLANT
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222
# TEST221 MODEL/SYSTEM/FACTS PRESERVED | WEIGHTS FROZEN
# SOURCE: OBJECT_END attention-weighted V contribution, merged through L08 o_proj input geometry
# TARGET: blind-query final prefill answer-boundary token
# PREFILL ONLY | DECODE INTERVENTION OFF | CANDIDATES NEVER ENTER INTERVENTION
# EXPERIMENTAL ATTENTION-OUTPUT INTERVENTION — NOT CANONICAL SEASC RESIDUAL MOTOR
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=222;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;LAYER=8;KVH=0;EPS=1e-8
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Rovan Tesk","keeps","the amber compass"),("Mira Veln","carries","the silver lantern"),("Dalen Quor","owns","the violet key"),("Sorin Kelm","guards","the bronze sphere"),("Varek Tonn","holds","the golden necklace"),("Lira Mesk","stores","the iron dagger"),("Korin Drel","protects","the crystal mirror"),("Taren Vosk","carries","the wooden mask")]
SCALES=[.25,.50,1.00];M=len(FACTS)
print("="*128);print("TEST 222 — ATTENTION-OUTPUT BINDING PACKET TRANSPLANT");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222");print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/20] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;NH=model.config.num_attention_heads;NKV=model.config.num_key_value_heads;HD=H//NH;GROUP=NH//NKV
if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")
FP_T=[layers[0].self_attn.q_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()
def unit(x):return x/x.norm().clamp_min(EPS)
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
print("[2/20] Token maps...")
FMAP={};QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    f=chat(fact_text(s,r,o));fi=tok(f,return_tensors="pt",add_special_tokens=False).to(DEVICE);full=fi.input_ids[0].tolist()
    ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Token map fail Q{qi+1}")
    q=qform(s,r);qe=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if o.lower() in q.lower():raise RuntimeError("Target leakage.")
    FMAP[qi]=(fi,ss,rs,os_);QENC.append(qe)
    print(f"Q{qi+1} FACT subject={ss} relation={rs} object={os_} | BLIND slot={qe.input_ids.shape[1]-1}")
print("[3/20] RoPE engine...")
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
print("[4/20] Capture source Q/K/V...")
@torch.inference_mode()
def capture(e):
    S={};hs=[]
    for name,mod in [("Q",layers[LAYER].self_attn.q_proj),("K",layers[LAYER].self_attn.k_proj),("V",layers[LAYER].self_attn.v_proj)]:
        def mk(n):
            def hk(m,args,out):S[n]=out[0].float().detach().clone()
            return hk
        hs.append(mod.register_forward_hook(mk(name)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return S
SRC={}
for qi in range(M):SRC[qi]=capture(FMAP[qi][0]);print(f"Q{qi+1} captured")
print("[5/20] Reconstruct source attention readout...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    khead=qhead//GROUP;q=rope(qh(S["Q"],qpos,qhead),qpos);K=torch.stack([rope(kvh(S["K"],p,khead),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)
READ={};RAWV={}
for qi in range(M):
    S=SRC[qi];obj=FMAP[qi][3];oe=obj[-1];rows=[]
    for h in QGROUP:
        for qp in range(oe,len(FMAP[qi][0].input_ids[0])):
            a=attn_row(S,qp,h);rows.append((float(a[oe]),h,qp,a))
    rows.sort(key=lambda z:z[0],reverse=True);READ[qi]=rows;RAWV[qi]=kvh(S["V"],oe,KVH).clone()
    b=rows[0];print(f"Q{qi+1} bestQH={b[1]:02d} qpos={b[2]} objEndAttn={b[0]:.6f}")
print("[6/20] Forge attention-output packets...")
# For each fact, use each QH00-QH06's strongest post-object readout weight.
# Packet lives in concatenated 28-head attention-output space before o_proj.
PACK={};HEADPACK={};COMMON={}
for qi in range(M):
    oe=FMAP[qi][3][-1];v=RAWV[qi];p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32);weights={}
    for h in QGROUP:
        rr=[x for x in READ[qi] if x[1]==h];rr.sort(key=lambda z:z[0],reverse=True);w=rr[0][0];weights[h]=w;p[h]=w*v
    PACK[qi]=p.reshape(H);HEADPACK[qi]=weights
    print(f"Q{qi+1} packetNorm={PACK[qi].norm():.4f} weights="+",".join(f"H{h:02d}:{weights[h]:.3f}" for h in QGROUP))
COMMON["mean"]=torch.stack([PACK[i] for i in range(M)]).mean(0)
print("[7/20] Wrong/shuffled/negative controls...")
WRONG={i:(i+1)%M for i in range(M)};SHUFF={i:(i+3)%M for i in range(M)}
for qi in range(M):print(f"Q{qi+1} correct={FACTS[qi][2]} wrong={FACTS[WRONG[qi]][2]} shuffled={FACTS[SHUFF[qi]][2]}")
print("[8/20] Blind-query baseline attention-output capture...")
# Intervene at L08 self_attn o_proj INPUT, final prefill token only.
# This is the concatenated multi-query-head attention readout immediately before o_proj.
@torch.inference_mode()
def blind_oproj_input(qi):
    box={};mod=layers[LAYER].self_attn.o_proj
    def pre(m,args):box["x"]=args[0][0,-1].float().detach().clone()
    h=mod.register_forward_pre_hook(pre)
    try:model(**QENC[qi],use_cache=False,return_dict=True)
    finally:h.remove()
    return box["x"]
BASEO={i:blind_oproj_input(i) for i in range(M)}
for qi in range(M):print(f"Q{qi+1} answerSlotNorm={BASEO[qi].norm():.4f}")
print("[9/20] Branch definitions...")
BRANCHES=["OUTPUT_CORRECT","OUTPUT_WRONG","OUTPUT_SHUFFLED","OUTPUT_NEG","OUTPUT_COMMON","V_PACKET"]
def packet(qi,b):
    if b=="OUTPUT_CORRECT":return PACK[qi]
    if b=="OUTPUT_WRONG":return PACK[WRONG[qi]]
    if b=="OUTPUT_SHUFFLED":
        p=PACK[qi].reshape(NH,HD).clone();p[QGROUP]=p[list(reversed(QGROUP))];return p.reshape(H)
    if b=="OUTPUT_NEG":return -PACK[qi]
    if b=="OUTPUT_COMMON":return COMMON["mean"]
    if b=="V_PACKET":
        p=torch.zeros(NH,HD,device=DEVICE);p[QGROUP]=RAWV[qi][None,:];return p.reshape(H)
    raise ValueError(b)
print("[10/20] Direct attention-output intervention...")
def install(qi,b,scale,tele=None):
    mod=layers[LAYER].self_attn.o_proj;pv=packet(qi,b).float()
    def pre(m,args):
        x=args[0]
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();orig=y[:,-1,:].float();d=unit(pv)*orig.norm(dim=-1,keepdim=True)*float(scale);y[:,-1,:]=(orig+d).to(y.dtype)
        if tele is not None:tele.update({"layer":LAYER,"branch":b,"scale":float(scale),"orig_norm":float(orig.norm()),"packet_norm":float(pv.norm()),"delta_norm":float(d.norm()),"rel":float(d.norm()/orig.norm().clamp_min(EPS))})
        return (y,)
    return mod.register_forward_pre_hook(pre)
print("[11/20] Prefill → frozen KV scorer...")
@torch.inference_mode()
def prefill(qi,scale=0.,b="OUTPUT_CORRECT"):
    h=None;te={}
    try:
        if scale>0:h=install(qi,b,scale,te)
        o=model(**QENC[qi],use_cache=True,return_dict=True)
    finally:
        if h is not None:h.remove()
    return o.logits[:,-1,:].float(),o.past_key_values,te
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="OUTPUT_CORRECT"):
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)[0];logits,pkv,_=prefill(qi,scale,b);vals=[]
    for i,t in enumerate(y):
        vals.append(torch.log_softmax(logits[0],-1)[t])
        if i<y.numel()-1:
            o=model(input_ids=t.view(1,1),past_key_values=pkv,use_cache=True,return_dict=True);logits=o.logits[:,-1,:].float();pkv=o.past_key_values
    z=torch.stack(vals);return float(z.sum()),float(z.mean()),int(y.numel())
def margin(qi,scale=0.,b="OUTPUT_CORRECT"):
    ts,tm,tn=lp(qi,FACTS[qi][2],scale,b);wrong=[lp(qi,FACTS[j][2],scale,b)[0] for j in range(M) if j!=qi];bw=max(wrong);return ts,tm,tn,bw,ts-bw
print("[12/20] Generation...")
@torch.inference_mode()
def generate(qi,scale=0.,b="OUTPUT_CORRECT",n=40):
    e=QENC[qi];p=e.input_ids.shape[1];h=None
    try:
        if scale>0:h=install(qi,b,scale)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        if h is not None:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip()
print("[13/20] Baseline + correct dose sweep...")
BASE={};SWEEP={}
for qi in range(M):
    BASE[qi]=(generate(qi),*margin(qi))
    for sc in SCALES:SWEEP[(qi,sc)]=(generate(qi,sc),*margin(qi,sc))
print("[14/20] Controls @ .50...")
RES={}
for b in BRANCHES:
    for qi in range(M):RES[(b,qi)]=(generate(qi,.5,b),*margin(qi,.5,b))
print("[15/20] Selectivity + first token...")
FIRST={}
for qi in range(M):
    c=RES[("OUTPUT_CORRECT",qi)][5]
    print(f"Q{qi+1} CORRECT={c:+.4f} WRONG Δ={c-RES[('OUTPUT_WRONG',qi)][5]:+.4f} SHUFF Δ={c-RES[('OUTPUT_SHUFFLED',qi)][5]:+.4f} NEG Δ={c-RES[('OUTPUT_NEG',qi)][5]:+.4f} COMMON Δ={c-RES[('OUTPUT_COMMON',qi)][5]:+.4f} VPACK Δ={c-RES[('V_PACKET',qi)][5]:+.4f}")
    tid=tok(FACTS[qi][2],add_special_tokens=False).input_ids[0];lv,_,_=prefill(qi);base=float(torch.log_softmax(lv[0],-1)[tid]);row={"VANILLA":base}
    for b in BRANCHES:
        lv,_,_=prefill(qi,.5,b);row[b]=float(torch.log_softmax(lv[0],-1)[tid])
    FIRST[qi]=row
print("[16/20] Telemetry + X-Ray Q1...")
for b in BRANCHES:
    _,_,te=prefill(0,.5,b);print(f"{b:16s} {te}")
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
print("[17/20] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[18/20] RESULTS")
print("\n"+"="*128);print("TEST 222 RESULTS");print("="*128)
print("MODE: L08 ATTENTION-OUTPUT PACKET INTERVENTION | PREFILL ONLY | DECODE INTERVENTION OFF | WEIGHTS FROZEN")
print("SOURCE: OBJECT_END attention-weighted V readout across KVH00 query group QH00-QH06")
print("\nOUTPUT_CORRECT DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {qform(FACTS[qi][0],FACTS[qi][1])} | TARGET={FACTS[qi][2]}")
    print(f" VANILLA targetLP={b[1]:+.4f} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" OUT {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} margin={r[5]:+.4f} Δmargin={r[5]-b[5]:+.4f} | {r[0]}")
print("\nCONTROLS @ .50")
for br in BRANCHES:
    ms=[];dl=[];imp=0;print("\n"+br)
    for qi in range(M):
        r=RES[(br,qi)];ms.append(r[5]);dl.append(r[1]-BASE[qi][1]);imp+=r[5]>BASE[qi][5]
        print(f" Q{qi+1} targetLP={r[1]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" meanMargin={np.mean(ms):+.4f} meanΔtargetLP={np.mean(dl):+.4f} marginImproved={imp}/{M}")
print("\nSELECTIVITY @ .50")
for qi in range(M):
    c=RES[("OUTPUT_CORRECT",qi)][5]
    print(f"Q{qi+1} CORRECT={c:+.4f} | vs WRONG={c-RES[('OUTPUT_WRONG',qi)][5]:+.4f} vs SHUFF={c-RES[('OUTPUT_SHUFFLED',qi)][5]:+.4f} vs NEG={c-RES[('OUTPUT_NEG',qi)][5]:+.4f} vs COMMON={c-RES[('OUTPUT_COMMON',qi)][5]:+.4f} vs VPACK={c-RES[('V_PACKET',qi)][5]:+.4f}")
print("\nFIRST TARGET TOKEN @ .50")
for qi,r in FIRST.items():print(f"Q{qi+1} VANILLA={r['VANILLA']:+.4f} CORRECT={r['OUTPUT_CORRECT']:+.4f} Δ={r['OUTPUT_CORRECT']-r['VANILLA']:+.4f} WRONG={r['OUTPUT_WRONG']:+.4f} NEG={r['OUTPUT_NEG']:+.4f}")
print("\nQ1 OUTPUT X-RAY @ .50")
for br,x in XR.items():print(f"{br:16s} L07={x[7]*100:7.3f}% L08={x[8]*100:7.3f}% L09={x[9]*100:7.3f}% L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("-"*128)
print("Weights: PASS | Source facts absent from blind queries | Candidate answers never enter intervention | Decode intervention: ZERO")
print("TEST222 intervenes on the L08 pre-o_proj attention-output representation; it is not the canonical SEASC residual-stream motor")
print("PASS requires CORRECT output packet to selectively outperform wrong/shuffled/negative/common/V-packet controls")
print("="*128);print("[19/20] TEST 222 COMPLETE");print("[20/20] END")



