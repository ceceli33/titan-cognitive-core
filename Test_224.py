# ==================================================================================================
# TEST 224 — PACKET-PRESERVING LAYERWISE TRANSPORT
# L08 ATTENTION PACKET -> MODEL-NATIVE L08→L19 TRANSPORT MAPS -> L0-L19 DELIVERY
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224
# TEST223 WORKING MODEL/SYSTEM/FACTS/SEASC ENVELOPE PRESERVED
# SOURCE PACKET: TEST222 L08/KVH00 QH00-QH06 OBJECT_END attention-weighted V readout
# TRANSPORT: layerwise low-rank ridge maps learned only from matched calibration perturbations
# TEST: transported packet identity preserved across depth vs wrong/shuffled/negative controls
# L20-L27 MOTOR OFF | WEIGHTS FROZEN | PREFILL ONLY | DECODE INTERVENTION OFF
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=224;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;STEER=20;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;RIDGE=1e-3;CAL_EPS=.02
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Rovan Tesk","keeps","the amber compass"),("Mira Veln","carries","the silver lantern"),("Dalen Quor","owns","the violet key"),("Sorin Kelm","guards","the bronze sphere"),("Varek Tonn","holds","the golden necklace"),("Lira Mesk","stores","the iron dagger"),("Korin Drel","protects","the crystal mirror"),("Taren Vosk","carries","the wooden mask")]
CAL=["Describe a quiet room with a wooden table.","A traveler waits beside an old station.","Several books rest on a narrow shelf.","A small lamp stands near the window.","Clouds move slowly above distant hills.","A metal box sits beside a chair.","A clock hangs above a doorway.","A notebook lies next to a glass bottle.","A bicycle is parked near a stone wall.","A cup rests on the kitchen counter.","A coat hangs beside the entrance.","A bird sits quietly on a tree branch.","A sign stands near the road.","A picture hangs above a sofa.","A bridge crosses a wide river.","A cabinet stands against the wall."]
SCALES=[.25,.50,1.00];M=len(FACTS)
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=[IVME*env(L) for L in range(STEER)];RSS=math.sqrt(sum(x*x for x in RHO))
print("="*128);print("TEST 224 — PACKET-PRESERVING LAYERWISE TRANSPORT");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f}")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/24] Model...")
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
print("[2/24] Token maps...")
FMAP={};QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    fi=tok(chat(fact_text(s,r,o)),return_tensors="pt",add_special_tokens=False).to(DEVICE);full=fi.input_ids[0].tolist()
    ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Token map fail Q{qi+1}")
    q=qform(s,r);qe=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if o.lower() in q.lower():raise RuntimeError("Target leakage.")
    FMAP[qi]=(fi,ss,rs,os_);QENC.append(qe);print(f"Q{qi+1} FACT subject={ss} relation={rs} object={os_} | BLIND slot={qe.input_ids.shape[1]-1}")
print("[3/24] RoPE...")
rotary=model.model.rotary_emb;MAXSEQ=max(max(x[0].input_ids.shape[1] for x in FMAP.values()),max(x.input_ids.shape[1] for x in QENC))+4
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
    finally:
        for h in hs:h.remove()
    return S
SRC={}
for qi in range(M):SRC[qi]=capture_source(FMAP[qi][0]);print(f"Q{qi+1} captured")
print("[5/24] Reconstruct source readout...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    kh=qhead//GROUP;q=rope(qh(S["Q"],qpos,qhead),qpos);K=torch.stack([rope(kvh(S["K"],p,kh),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)
READ={};RAWV={}
for qi in range(M):
    S=SRC[qi];oe=FMAP[qi][3][-1];rows=[]
    for h in QGROUP:
        for qp in range(oe,FMAP[qi][0].input_ids.shape[1]):
            a=attn_row(S,qp,h);rows.append((float(a[oe]),h,qp))
    rows.sort(key=lambda z:z[0],reverse=True);READ[qi]=rows;RAWV[qi]=kvh(S["V"],oe,KVH).clone()
    b=rows[0];print(f"Q{qi+1} bestQH={b[1]:02d} qpos={b[2]} objEndAttn={b[0]:.6f}")
print("[6/24] Forge L08 source packets...")
PACK={}
for qi in range(M):
    p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32);v=RAWV[qi];ww={}
    for h in QGROUP:
        w=max(x[0] for x in READ[qi] if x[1]==h);ww[h]=w;p[h]=w*v
    with torch.inference_mode():PACK[qi]=layers[SRC_LAYER].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"Q{qi+1} residualPacketNorm={PACK[qi].norm():.4f}")
print("[7/24] Calibration prompts...")
CENC=[tok(chat(x),return_tensors="pt",add_special_tokens=False).to(DEVICE) for x in CAL]
print("CAL N=",len(CENC))
print("[8/24] Measure model-native L08→L19 packet transport basis...")
GEN=torch.Generator(device=DEVICE);GEN.manual_seed(SEED)
RBASE=torch.randn(len(CENC),H,generator=GEN,device=DEVICE,dtype=torch.float32)
RBASE=torch.stack([unit(x) for x in RBASE])
@torch.inference_mode()
def local_pair(e,L,v):
    pos=e.input_ids.shape[1]-1;A={};B={}
    def cap(store):
        def hk(m,args,out):store["x"]=(out[0] if isinstance(out,tuple) else out)[0,pos].float().detach().clone()
        return layers[L+1].register_forward_hook(hk)
    h=cap(A);model(**e,use_cache=False,return_dict=True);h.remove()
    def inj(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        y=x.clone();z=y[:,pos,:].float();d=unit(v)*z.norm(dim=-1,keepdim=True)*CAL_EPS;y[:,pos,:]=(z+d).to(y.dtype)
        return (y,)+out[1:] if isinstance(out,tuple) else y
    ih=layers[L].register_forward_hook(inj);h=cap(B);model(**e,use_cache=False,return_dict=True);h.remove();ih.remove()
    return B["x"]-A["x"]
TBASIS={SRC_LAYER:torch.stack([PACK[i] for i in range(M)])}
MAPS={}
for L in range(SRC_LAYER,19):
    X=[];Y=[]
    for ci,e in enumerate(CENC):
        v=RBASE[ci];d=local_pair(e,L,v);X.append(v);Y.append(d)
    X=torch.stack(X);Y=torch.stack(Y);G=X@X.T+RIDGE*torch.eye(len(X),device=DEVICE);MAPS[L]=(X,Y,torch.linalg.inv(G))
    print(f"L{L:02d}->L{L+1:02d} calibration rank={torch.linalg.matrix_rank(X).item()} meanΔ={Y.norm(dim=1).mean():.4f}")
print("[9/24] Transport packets L08→L19...")
def transport(v,L):
    X,Y,Gi=MAPS[L];coef=Gi@(X@v);z=coef@Y
    return unit(z) if z.norm()>EPS else unit(v)
TP={}
for qi in range(M):
    TP[(qi,SRC_LAYER)]=unit(PACK[qi])
    for L in range(SRC_LAYER,19):TP[(qi,L+1)]=transport(TP[(qi,L)],L)
    cs=[float(torch.dot(TP[(qi,L)],TP[(qi,SRC_LAYER)])) for L in range(SRC_LAYER,20)]
    print(f"Q{qi+1} fidelity L08={cs[0]:+.4f} L12={cs[4]:+.4f} L16={cs[8]:+.4f} L19={cs[-1]:+.4f}")
print("[10/24] Pre-L08 packet preparation...")
DIR={}
for qi in range(M):
    DIR[qi]=[]
    for L in range(STEER):DIR[qi].append(unit(PACK[qi]) if L<SRC_LAYER else TP[(qi,L)])
WRONG={i:(i+1)%M for i in range(M)};SHUFF={i:(i+3)%M for i in range(M)}
BRANCHES=["TRANSPORT_CORRECT","TRANSPORT_WRONG","TRANSPORT_SHUFFLED","TRANSPORT_NEG","STATIC_PACKET","SINGLE_L08"]
print("[11/24] Install residual delivery...")
def install(qi,b,scale,tele=None):
    hs=[];calls={L:0 for L in range(TOTAL)}
    src=WRONG[qi] if b=="TRANSPORT_WRONG" else SHUFF[qi] if b=="TRANSPORT_SHUFFLED" else qi
    sign=-1. if b=="TRANSPORT_NEG" else 1.
    active=[SRC_LAYER] if b=="SINGLE_L08" else list(range(STEER))
    for L in active:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                if x.ndim!=3 or x.shape[1]<=1:return None
                y=x.clone();z=y[:,-1,:].float()
                if b=="STATIC_PACKET":v=unit(PACK[qi])
                else:v=DIR[src][li]
                dose=RHO[li]*float(scale);d=sign*v*z.norm(dim=-1,keepdim=True)*dose;y[:,-1,:]=(z+d).to(y.dtype);calls[li]+=1
                if tele is not None:tele.append((li,float(z.norm()),float(d.norm()),dose))
                return (y,)+out[1:] if isinstance(out,tuple) else y
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    return hs,calls
def remove(hs):
    for h in hs:h.remove()
print("[12/24] Prefill → frozen KV scorer...")
@torch.inference_mode()
def prefill(qi,scale=0.,b="TRANSPORT_CORRECT",tele=False):
    hs=[];calls={L:0 for L in range(TOTAL)};t=[] if tele else None
    try:
        if scale>0:hs,calls=install(qi,b,scale,t)
        o=model(**QENC[qi],use_cache=True,return_dict=True)
    finally:remove(hs)
    return o.logits[:,-1,:].float(),o.past_key_values,t,calls
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="TRANSPORT_CORRECT"):
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)[0];logits,pkv,_,_=prefill(qi,scale,b);vv=[]
    for i,t in enumerate(y):
        vv.append(torch.log_softmax(logits[0],-1)[t])
        if i<y.numel()-1:
            o=model(input_ids=t.view(1,1),past_key_values=pkv,use_cache=True,return_dict=True);logits=o.logits[:,-1,:].float();pkv=o.past_key_values
    z=torch.stack(vv);return float(z.sum()),float(z.mean()),int(y.numel())
def margin(qi,scale=0.,b="TRANSPORT_CORRECT"):
    ts,tm,tn=lp(qi,FACTS[qi][2],scale,b);ww=[lp(qi,FACTS[j][2],scale,b)[0] for j in range(M) if j!=qi];bw=max(ww);return ts,tm,tn,bw,ts-bw
print("[13/24] Generation...")
@torch.inference_mode()
def generate(qi,scale=0.,b="TRANSPORT_CORRECT",n=40):
    e=QENC[qi];p=e.input_ids.shape[1];hs=[]
    try:
        if scale>0:hs,_=install(qi,b,scale)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:remove(hs)
    return tok.decode(o[0,p:],skip_special_tokens=True).strip()
print("[14/24] Vanilla...")
BASE={i:(generate(i),*margin(i)) for i in range(M)}
print("[15/24] Correct transport dose sweep...")
SWEEP={}
for i in range(M):
    for sc in SCALES:SWEEP[(i,sc)]=(generate(i,sc),*margin(i,sc))
print("[16/24] Controls @ .50...")
RES={}
for b in BRANCHES:
    for i in range(M):RES[(b,i)]=(generate(i,.5,b),*margin(i,.5,b))
print("[17/24] Transport fidelity...")
for qi in range(M):
    print(f"Q{qi+1}",end="")
    for L in [8,9,12,16,19]:
        prev=TP[(qi,L-1)] if L>8 else TP[(qi,8)];cur=TP[(qi,L)]
        print(f" L{L:02d}cos={float(torch.dot(prev,cur)):+.4f}",end="")
    print()
print("[18/24] Binding separation...")
for L in [8,12,16,19]:
    good=[];bad=[]
    for qi in range(M):
        v=DIR[qi][L];good.append(float(torch.dot(v,DIR[qi][L])))
        bad.append(max(float(torch.dot(v,DIR[j][L])) for j in range(M) if j!=qi))
    print(f"L{L:02d} self={np.mean(good):+.4f} bestWrongCos={np.mean(bad):+.4f} separation={np.mean(np.array(good)-np.array(bad)):+.4f}")
print("[19/24] Telemetry...")
_,_,TEL,CALLS=prefill(0,.5,"TRANSPORT_CORRECT",True)
for L,n,d,e in TEL:print(f"L{L:02d} hiddenNorm={n:.4f} injectionNorm={d:.4f} effectiveDose={e*100:.4f}%")
print("HOOK CALLS L0-L19:",[CALLS[L] for L in range(20)]);print("HOOK CALLS L20-L27:",[CALLS[L] for L in range(20,28)])
print("[20/24] X-Ray Q1...")
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
    ih,_=install(qi,b,scale);hs=caps(B);model(**e,use_cache=False,return_dict=True);remove(hs);remove(ih)
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[21/24] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[22/24] RESULTS")
print("\n"+"="*128);print("TEST 224 RESULTS");print("="*128)
print("MODE: PACKET-PRESERVING LAYERWISE TRANSPORT | L0-L19 ON | L20-L27 OFF | PREFILL ONLY | DECODE OFF")
print(f"SEASC: IVME={IVME} SONUM={SONUM} ZIRVE={ZIRVE} TABAN={TABAN} RSS={RSS:.9f}")
print("\nTRANSPORT_CORRECT DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {qform(FACTS[qi][0],FACTS[qi][1])} | TARGET={FACTS[qi][2]}")
    print(f" VANILLA targetLP={b[1]:+.4f} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" TRANS {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} margin={r[5]:+.4f} Δmargin={r[5]-b[5]:+.4f} | {r[0]}")
print("\nCONTROLS @ .50")
for br in BRANCHES:
    ms=[];dl=[];imp=0;print("\n"+br)
    for qi in range(M):
        r=RES[(br,qi)];ms.append(r[5]);dl.append(r[1]-BASE[qi][1]);imp+=r[5]>BASE[qi][5]
        print(f" Q{qi+1} targetLP={r[1]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" meanMargin={np.mean(ms):+.4f} meanΔtargetLP={np.mean(dl):+.4f} improved={imp}/{M}")
print("\nSELECTIVITY @ .50")
for qi in range(M):
    c=RES[("TRANSPORT_CORRECT",qi)][5]
    print(f"Q{qi+1} CORRECT={c:+.4f} vs WRONG={c-RES[('TRANSPORT_WRONG',qi)][5]:+.4f} vs SHUFF={c-RES[('TRANSPORT_SHUFFLED',qi)][5]:+.4f} vs NEG={c-RES[('TRANSPORT_NEG',qi)][5]:+.4f} vs STATIC={c-RES[('STATIC_PACKET',qi)][5]:+.4f} vs SINGLE={c-RES[('SINGLE_L08',qi)][5]:+.4f}")
print("\nQ1 X-RAY @ .50")
for br,x in XR.items():print(f"{br:20s} L00={x[0]*100:7.3f}% L08={x[8]*100:7.3f}% L19={x[19]*100:7.3f}% L20={x[20]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("-"*128)
print("Weights: PASS | Candidate answers never enter intervention | Decode intervention: ZERO")
print("L20-L27 injection: ZERO | downstream displacement there is transport only")
print("PASS requires transported CORRECT packet to retain identity and selectively beat wrong/shuffled/negative/static/single controls")
print("="*128);print("[23/24] TEST 224 COMPLETE");print("[24/24] END")



