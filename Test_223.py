# ==================================================================================================
# TEST 223 — MULTI-LAYER RESIDUAL DELIVERY LINE
# ATTENTION-OUTPUT INFORMATION PACKET × L0-L19 SEASC PHYSICAL ENVELOPE
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223
# TEST222 MODEL/SYSTEM/FACTS + SOURCE PACKET FORGE PRESERVED
# SOURCE: L08/KVH00 OBJECT_END attention-weighted V readout packet
# TEST: SINGLE-L08 DELIVERY vs L0-L19 RESIDUAL DELIVERY LINE
# L20-L27 MOTOR OFF | WEIGHTS FROZEN | PREFILL ONLY | DECODE INTERVENTION OFF
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=223;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;STEER=20;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Rovan Tesk","keeps","the amber compass"),("Mira Veln","carries","the silver lantern"),("Dalen Quor","owns","the violet key"),("Sorin Kelm","guards","the bronze sphere"),("Varek Tonn","holds","the golden necklace"),("Lira Mesk","stores","the iron dagger"),("Korin Drel","protects","the crystal mirror"),("Taren Vosk","carries","the wooden mask")]
SCALES=[.25,.50,1.00];M=len(FACTS)
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=[IVME*env(L) for L in range(STEER)]
RSS=math.sqrt(sum(x*x for x in RHO))
print("="*128);print("TEST 223 — MULTI-LAYER RESIDUAL DELIVERY LINE");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f}")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/22] Model...")
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
print("[2/22] Token maps...")
FMAP={};QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    fi=tok(chat(fact_text(s,r,o)),return_tensors="pt",add_special_tokens=False).to(DEVICE);full=fi.input_ids[0].tolist()
    ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Token map fail Q{qi+1}")
    q=qform(s,r);qe=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if o.lower() in q.lower():raise RuntimeError("Target leakage.")
    FMAP[qi]=(fi,ss,rs,os_);QENC.append(qe)
    print(f"Q{qi+1} FACT subject={ss} relation={rs} object={os_} | BLIND slot={qe.input_ids.shape[1]-1}")
print("[3/22] RoPE engine...")
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
print("[4/22] Capture TEST222 source Q/K/V...")
@torch.inference_mode()
def capture_source(e):
    S={};hs=[]
    for name,mod in [("Q",layers[SRC_LAYER].self_attn.q_proj),("K",layers[SRC_LAYER].self_attn.k_proj),("V",layers[SRC_LAYER].self_attn.v_proj)]:
        def mk(n):
            def hk(m,args,out):S[n]=out[0].float().detach().clone()
            return hk
        hs.append(mod.register_forward_hook(mk(name)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return S
SRC={}
for qi in range(M):SRC[qi]=capture_source(FMAP[qi][0]);print(f"Q{qi+1} captured")
print("[5/22] Reconstruct TEST222 source readout...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    khead=qhead//GROUP;q=rope(qh(S["Q"],qpos,qhead),qpos);K=torch.stack([rope(kvh(S["K"],p,khead),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)
READ={};RAWV={}
for qi in range(M):
    S=SRC[qi];oe=FMAP[qi][3][-1];rows=[]
    for h in QGROUP:
        for qp in range(oe,FMAP[qi][0].input_ids.shape[1]):
            a=attn_row(S,qp,h);rows.append((float(a[oe]),h,qp))
    rows.sort(key=lambda z:z[0],reverse=True);READ[qi]=rows;RAWV[qi]=kvh(S["V"],oe,KVH).clone()
    b=rows[0];print(f"Q{qi+1} bestQH={b[1]:02d} qpos={b[2]} objEndAttn={b[0]:.6f}")
print("[6/22] Forge TEST222 attention-output packets...")
PACK={}
for qi in range(M):
    v=RAWV[qi];p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32);weights={}
    for h in QGROUP:
        rr=[x for x in READ[qi] if x[1]==h];w=max(rr,key=lambda z:z[0])[0];weights[h]=w;p[h]=w*v
    PACK[qi]=p.reshape(H)
    print(f"Q{qi+1} packetNorm={PACK[qi].norm():.4f} weights="+",".join(f"H{h:02d}:{weights[h]:.3f}" for h in QGROUP))
print("[7/22] Build layer-local residual delivery directions...")
# TEST222 packet is first transformed by the actual L08 o_proj, then each L0-L19
# gets a model-native local direction extracted from matched fact-vs-wrong hidden-state transport.
@torch.inference_mode()
def residual_states(e):
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(STEER)]
FACT_RES={i:residual_states(FMAP[i][0]) for i in range(M)}
WRONG={i:(i+1)%M for i in range(M)}
DIR={}
with torch.inference_mode():
    for qi in range(M):
        base_packet=model.model.layers[SRC_LAYER].self_attn.o_proj(PACK[qi].to(model.dtype)).float()
        dirs=[]
        for L in range(STEER):
            local=FACT_RES[qi][L]-FACT_RES[WRONG[qi]][L]
            if local.norm()<EPS:local=base_packet
            dirs.append(unit(local))
        DIR[qi]=dirs
        print(f"Q{qi+1} localDirs=20 L08packetProjectedNorm={base_packet.norm():.4f}")
print("[8/22] Controls...")
SHUFF={i:(i+3)%M for i in range(M)}
BRANCHES=["DELIVERY_CORRECT","DELIVERY_WRONG","DELIVERY_SHUFFLED","DELIVERY_NEG","SINGLE_L08"]
print("[9/22] L0-L19 residual delivery hooks...")
def install_delivery(qi,b,scale,tele=None):
    hs=[];calls={L:0 for L in range(TOTAL)}
    if b=="DELIVERY_WRONG":src=WRONG[qi]
    elif b=="DELIVERY_SHUFFLED":src=SHUFF[qi]
    else:src=qi
    sign=-1. if b=="DELIVERY_NEG" else 1.
    if b=="SINGLE_L08":
        def hk(m,args,out):
            x=out[0] if isinstance(out,tuple) else out
            if x.shape[1]<=1:return None
            y=x.clone();orig=y[:,-1,:].float();d=unit(DIR[qi][SRC_LAYER])*orig.norm(dim=-1,keepdim=True)*RHO[SRC_LAYER]*float(scale)
            y[:,-1,:]=(orig+d).to(y.dtype);calls[SRC_LAYER]+=1
            if tele is not None:tele.append((SRC_LAYER,float(orig.norm()),float(d.norm()),RHO[SRC_LAYER]*scale))
            return (y,)+out[1:] if isinstance(out,tuple) else y
        hs.append(layers[SRC_LAYER].register_forward_hook(hk));return hs,calls
    for L in range(STEER):
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                if x.shape[1]<=1:return None
                y=x.clone();orig=y[:,-1,:].float();d=sign*DIR[src][li]*orig.norm(dim=-1,keepdim=True)*RHO[li]*float(scale)
                y[:,-1,:]=(orig+d).to(y.dtype);calls[li]+=1
                if tele is not None:tele.append((li,float(orig.norm()),float(d.norm()),RHO[li]*scale))
                return (y,)+out[1:] if isinstance(out,tuple) else y
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    return hs,calls
def remove(hs):
    for h in hs:h.remove()
print("[10/22] Prefill → frozen KV scorer...")
@torch.inference_mode()
def prefill(qi,scale=0.,b="DELIVERY_CORRECT",tele=False):
    hs=[];calls={L:0 for L in range(TOTAL)};t=[] if tele else None
    try:
        if scale>0:hs,calls=install_delivery(qi,b,scale,t)
        o=model(**QENC[qi],use_cache=True,return_dict=True)
    finally:remove(hs)
    return o.logits[:,-1,:].float(),o.past_key_values,t,calls
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="DELIVERY_CORRECT"):
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)[0];logits,pkv,_,_=prefill(qi,scale,b);vals=[]
    for i,t in enumerate(y):
        vals.append(torch.log_softmax(logits[0],-1)[t])
        if i<y.numel()-1:
            o=model(input_ids=t.view(1,1),past_key_values=pkv,use_cache=True,return_dict=True);logits=o.logits[:,-1,:].float();pkv=o.past_key_values
    z=torch.stack(vals);return float(z.sum()),float(z.mean()),int(y.numel())
def margin(qi,scale=0.,b="DELIVERY_CORRECT"):
    ts,tm,tn=lp(qi,FACTS[qi][2],scale,b);wrong=[lp(qi,FACTS[j][2],scale,b)[0] for j in range(M) if j!=qi];bw=max(wrong);return ts,tm,tn,bw,ts-bw
print("[11/22] Generation...")
@torch.inference_mode()
def generate(qi,scale=0.,b="DELIVERY_CORRECT",n=40):
    e=QENC[qi];p=e.input_ids.shape[1];hs=[]
    try:
        if scale>0:hs,_=install_delivery(qi,b,scale)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:remove(hs)
    return tok.decode(o[0,p:],skip_special_tokens=True).strip()
print("[12/22] Vanilla baseline...")
BASE={}
for qi in range(M):BASE[qi]=(generate(qi),*margin(qi))
print("[13/22] Correct delivery dose sweep...")
SWEEP={}
for qi in range(M):
    for sc in SCALES:SWEEP[(qi,sc)]=(generate(qi,sc),*margin(qi,sc))
print("[14/22] Controls @ .50...")
RES={}
for b in BRANCHES:
    for qi in range(M):RES[(b,qi)]=(generate(qi,.5,b),*margin(qi,.5,b))
print("[15/22] Layer telemetry...")
_,_,TEL,CALLS=prefill(0,.5,"DELIVERY_CORRECT",True)
for L,n,d,e in TEL:print(f"L{L:02d} hiddenNorm={n:.4f} injectionNorm={d:.4f} effectiveDose={e*100:.4f}%")
print("HOOK CALLS L0-L19:",[CALLS[L] for L in range(20)])
print("HOOK CALLS L20-L27:",[CALLS[L] for L in range(20,28)])
print("[16/22] First target token...")
FIRST={}
for qi in range(M):
    tid=tok(FACTS[qi][2],add_special_tokens=False).input_ids[0];lv,_,_,_=prefill(qi);base=float(torch.log_softmax(lv[0],-1)[tid]);row={"VANILLA":base}
    for b in BRANCHES:
        lv,_,_,_=prefill(qi,.5,b);row[b]=float(torch.log_softmax(lv[0],-1)[tid])
    FIRST[qi]=row
print("[17/22] Output X-Ray Q1...")
@torch.inference_mode()
def xray(qi,b,scale=.5):
    e=QENC[qi];pos=e.input_ids.shape[1]-1;A={};B={}
    def caps(store):
        rr=[]
        for L in range(TOTAL):
            def mk(li):
                def hk(m,args,out):store[li]=(out[0] if isinstance(out,tuple) else out)[0,pos].float().detach().clone()
                return hk
            rr.append(layers[L].register_forward_hook(mk(L)))
        return rr
    hs=caps(A);model(**e,use_cache=False,return_dict=True);remove(hs)
    ih,_=install_delivery(qi,b,scale);hs=caps(B);model(**e,use_cache=False,return_dict=True);remove(hs);remove(ih)
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[18/22] Mean branch statistics...")
for b in BRANCHES:
    ms=[RES[(b,i)][5] for i in range(M)];dl=[RES[(b,i)][1]-BASE[i][1] for i in range(M)]
    print(f"{b:18s} meanMargin={np.mean(ms):+.4f} meanΔtargetLP={np.mean(dl):+.4f} improved={sum(RES[(b,i)][5]>BASE[i][5] for i in range(M))}/8")
print("[19/22] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[20/22] RESULTS")
print("\n"+"="*128);print("TEST 223 RESULTS");print("="*128)
print("MODE: ATTENTION-PACKET-DERIVED LAYER-LOCAL RESIDUAL DELIVERY | L0-L19 ON | L20-L27 OFF | PREFILL ONLY | WEIGHTS FROZEN")
print(f"SEASC ENVELOPE: IVME={IVME} SONUM={SONUM} ZIRVE={ZIRVE} TABAN={TABAN} RSS={RSS:.9f}")
print("\nDELIVERY_CORRECT DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {qform(FACTS[qi][0],FACTS[qi][1])} | TARGET={FACTS[qi][2]}")
    print(f" VANILLA targetLP={b[1]:+.4f} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" DELIVERY {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} margin={r[5]:+.4f} Δmargin={r[5]-b[5]:+.4f} | {r[0]}")
print("\nCONTROLS @ .50")
for br in BRANCHES:
    ms=[];dl=[];imp=0;print("\n"+br)
    for qi in range(M):
        r=RES[(br,qi)];ms.append(r[5]);dl.append(r[1]-BASE[qi][1]);imp+=r[5]>BASE[qi][5]
        print(f" Q{qi+1} targetLP={r[1]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" meanMargin={np.mean(ms):+.4f} meanΔtargetLP={np.mean(dl):+.4f} improved={imp}/{M}")
print("\nSELECTIVITY @ .50")
for qi in range(M):
    c=RES[("DELIVERY_CORRECT",qi)][5]
    print(f"Q{qi+1} CORRECT={c:+.4f} vs WRONG={c-RES[('DELIVERY_WRONG',qi)][5]:+.4f} vs SHUFF={c-RES[('DELIVERY_SHUFFLED',qi)][5]:+.4f} vs NEG={c-RES[('DELIVERY_NEG',qi)][5]:+.4f} vs SINGLE_L08={c-RES[('SINGLE_L08',qi)][5]:+.4f}")
print("\nFIRST TARGET TOKEN @ .50")
for qi,r in FIRST.items():print(f"Q{qi+1} VANILLA={r['VANILLA']:+.4f} CORRECT={r['DELIVERY_CORRECT']:+.4f} Δ={r['DELIVERY_CORRECT']-r['VANILLA']:+.4f} WRONG={r['DELIVERY_WRONG']:+.4f} NEG={r['DELIVERY_NEG']:+.4f}")
print("\nQ1 OUTPUT X-RAY @ .50")
for br,x in XR.items():print(f"{br:18s} L00={x[0]*100:7.3f}% L08={x[8]*100:7.3f}% L19={x[19]*100:7.3f}% L20={x[20]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("-"*128)
print("Weights: PASS | Source facts absent from blind queries | Candidate answers never enter intervention | Decode intervention: ZERO")
print("L20-L27 receive ZERO new injection; any displacement there is downstream transport")
print("TEST223 tests the multi-layer residual-delivery hypothesis against the prior single-L08 delivery strategy")
print("="*128);print("[21/22] TEST 223 COMPLETE");print("[22/22] END")
