# ==================================================================================================
# TEST 225 — ENDOGENOUS PACKET TRAJECTORY X-RAY
# SINGLE L08 ATTENTION PACKET INJECTION -> FREE MODEL-NATIVE DOWNSTREAM TRANSPORT L09-L27
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225
# TEST224 MODEL / SYSTEM / FACTS / TEST222 PACKET FORGE PRESERVED
# NO LEARNED TRANSPORT MAP | NO RE-INJECTION AFTER L08
# MEASURE: Δh_L = h_L(packet) - h_L(vanilla), L08-L27
# COMPARE: CORRECT vs WRONG vs NEGATIVE PACKET TRAJECTORIES
# WEIGHTS FROZEN | PREFILL ONLY | DECODE OFF
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=225
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
DOSES=[.01,.02,.04];PRIMARY=.04
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
print("="*128);print("TEST 225 — ENDOGENOUS PACKET TRAJECTORY X-RAY");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/20] Model...")
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
print("[2/20] Token maps...")
FMAP={};QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    fi=tok(chat(fact_text(s,r,o)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    full=fi.input_ids[0].tolist();ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Token map fail Q{qi+1}")
    q=qform(s,r);qe=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if o.lower() in q.lower():raise RuntimeError("Target leakage.")
    FMAP[qi]=(fi,ss,rs,os_);QENC.append(qe)
    print(f"Q{qi+1} FACT subject={ss} relation={rs} object={os_} | BLIND slot={qe.input_ids.shape[1]-1}")
print("[3/20] RoPE...")
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
print("[4/20] Capture TEST222 source Q/K/V...")
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
for qi in range(M):
    SRC[qi]=capture_source(FMAP[qi][0]);print(f"Q{qi+1} captured")
print("[5/20] Reconstruct TEST222 source readout...")
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
print("[6/20] Forge TEST222 L08 packets...")
PACK={}
for qi in range(M):
    p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32);v=RAWV[qi]
    for h in QGROUP:
        w=max(x[0] for x in READ[qi] if x[1]==h);p[h]=w*v
    with torch.inference_mode():PACK[qi]=layers[SRC_LAYER].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"Q{qi+1} packetNorm={PACK[qi].norm():.4f}")
WRONG={i:(i+1)%M for i in range(M)}
print("[7/20] Vanilla hidden trajectories...")
def remove(hs):
    for h in hs:h.remove()
@torch.inference_mode()
def capture_hidden(e,packet=None,dose=0.):
    pos=e.input_ids.shape[1]-1;S={};hs=[];inj_calls=0
    for L in range(TOTAL):
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    ih=None
    if packet is not None and dose!=0:
        def inject(m,args,out):
            nonlocal inj_calls
            x=out[0] if isinstance(out,tuple) else out
            if x.ndim!=3 or x.shape[1]<=1:return None
            y=x.clone();z=y[:,-1,:].float();d=unit(packet)*z.norm(dim=-1,keepdim=True)*float(dose)
            y[:,-1,:]=(z+d).to(y.dtype);inj_calls+=1
            return (y,)+out[1:] if isinstance(out,tuple) else y
        ih=layers[SRC_LAYER].register_forward_hook(inject)
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        remove(hs)
        if ih is not None:ih.remove()
    return S,inj_calls
VAN={}
for qi in range(M):
    VAN[qi],c=capture_hidden(QENC[qi]);print(f"Q{qi+1} vanilla captured")
print("[8/20] Correct/wrong/negative endogenous trajectories...")
TRAJ={}
for qi in range(M):
    for dose in DOSES:
        for branch,p in [("CORRECT",PACK[qi]),("WRONG",PACK[WRONG[qi]]),("NEG",-PACK[qi])]:
            S,c=capture_hidden(QENC[qi],p,dose)
            if c!=1:raise RuntimeError(f"Injection call mismatch Q{qi+1} {branch} dose={dose}: {c}")
            TRAJ[(qi,dose,branch)]=S
    print(f"Q{qi+1} complete")
print("[9/20] Build Δh trajectories...")
DELTA={}
REL={}
for qi in range(M):
    for dose in DOSES:
        for b in ["CORRECT","WRONG","NEG"]:
            DELTA[(qi,dose,b)]={};REL[(qi,dose,b)]={}
            for L in range(TOTAL):
                d=TRAJ[(qi,dose,b)][L]-VAN[qi][L]
                DELTA[(qi,dose,b)][L]=d
                REL[(qi,dose,b)][L]=float(d.norm()/VAN[qi][L].norm().clamp_min(EPS))
print("[10/20] Source identity retention...")
for qi in range(M):
    d=PRIMARY;src=DELTA[(qi,d,"CORRECT")][SRC_LAYER]
    print(f"Q{qi+1}",end="")
    for L in [8,9,10,12,16,19,20,24,27]:
        print(f" L{L:02d}={cos(src,DELTA[(qi,d,'CORRECT')][L]):+.4f}",end="")
    print()
print("[11/20] Local transport continuity...")
for qi in range(M):
    d=PRIMARY
    vals=[]
    for L in range(SRC_LAYER+1,TOTAL):
        vals.append(cos(DELTA[(qi,d,"CORRECT")][L-1],DELTA[(qi,d,"CORRECT")][L]))
    print(f"Q{qi+1} L08→09={vals[0]:+.4f} mean09→19={np.mean(vals[1:11]):+.4f} mean20→27={np.mean(vals[12:]):+.4f}")
print("[12/20] Correct-vs-wrong trajectory separation...")
for L in [8,9,10,12,16,19,20,24,27]:
    cw=[];cn=[];wn=[]
    for qi in range(M):
        c=DELTA[(qi,PRIMARY,"CORRECT")][L];w=DELTA[(qi,PRIMARY,"WRONG")][L];n=DELTA[(qi,PRIMARY,"NEG")][L]
        cw.append(cos(c,w));cn.append(cos(c,n));wn.append(cos(w,n))
    print(f"L{L:02d} C/W={np.mean(cw):+.4f} C/NEG={np.mean(cn):+.4f} W/NEG={np.mean(wn):+.4f}")
print("[13/20] Cross-fact identity collapse...")
for L in [8,9,10,12,16,19,20,24,27]:
    own=[];cross=[]
    for qi in range(M):
        v=DELTA[(qi,PRIMARY,"CORRECT")][L]
        own.append(cos(v,DELTA[(qi,PRIMARY,"CORRECT")][L]))
        cross.append(max(cos(v,DELTA[(j,PRIMARY,"CORRECT")][L]) for j in range(M) if j!=qi))
    print(f"L{L:02d} self={np.mean(own):+.4f} bestCross={np.mean(cross):+.4f} separation={np.mean(np.array(own)-np.array(cross)):+.4f}")
print("[14/20] Physical displacement...")
for qi in range(M):
    print(f"Q{qi+1}",end="")
    for L in [8,9,12,16,19,20,24,27]:
        print(f" L{L:02d}={REL[(qi,PRIMARY,'CORRECT')][L]*100:6.3f}%",end="")
    print()
print("[15/20] Dose linearity...")
for qi in range(M):
    print(f"Q{qi+1}",end="")
    for L in [8,19,27]:
        r=[REL[(qi,d,"CORRECT")][L] for d in DOSES]
        print(f" L{L:02d}[1x={r[0]*100:.3f},2x={r[1]*100:.3f},4x={r[2]*100:.3f}]",end="")
    print()
print("[16/20] Candidate scorer...")
@torch.inference_mode()
def prefill_logits(qi,packet=None,dose=0.):
    hs=[]
    if packet is not None and dose!=0:
        def inject(m,args,out):
            x=out[0] if isinstance(out,tuple) else out
            if x.ndim!=3 or x.shape[1]<=1:return None
            y=x.clone();z=y[:,-1,:].float();d=unit(packet)*z.norm(dim=-1,keepdim=True)*float(dose)
            y[:,-1,:]=(z+d).to(y.dtype)
            return (y,)+out[1:] if isinstance(out,tuple) else y
        hs.append(layers[SRC_LAYER].register_forward_hook(inject))
    try:o=model(**QENC[qi],use_cache=True,return_dict=True)
    finally:remove(hs)
    return o.logits[:,-1,:].float(),o.past_key_values
@torch.inference_mode()
def lp(qi,answer,packet=None,dose=0.):
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)[0]
    logits,pkv=prefill_logits(qi,packet,dose);vv=[]
    for i,t in enumerate(y):
        vv.append(torch.log_softmax(logits[0],-1)[t])
        if i<y.numel()-1:
            o=model(input_ids=t.view(1,1),past_key_values=pkv,use_cache=True,return_dict=True)
            logits=o.logits[:,-1,:].float();pkv=o.past_key_values
    return float(torch.stack(vv).sum())
def score(qi,packet=None,dose=0.):
    t=lp(qi,FACTS[qi][2],packet,dose)
    w=max(lp(qi,FACTS[j][2],packet,dose) for j in range(M) if j!=qi)
    return t,w,t-w
print("[17/20] Behavioral readout...")
BEH={}
for qi in range(M):
    BEH[(qi,"VANILLA")]=score(qi)
    BEH[(qi,"CORRECT")]=score(qi,PACK[qi],PRIMARY)
    BEH[(qi,"WRONG")]=score(qi,PACK[WRONG[qi]],PRIMARY)
    BEH[(qi,"NEG")]=score(qi,-PACK[qi],PRIMARY)
    v=BEH[(qi,"VANILLA")];c=BEH[(qi,"CORRECT")];w=BEH[(qi,"WRONG")];n=BEH[(qi,"NEG")]
    print(f"Q{qi+1} VAN={v[2]:+.4f} COR={c[2]:+.4f} Δ={c[2]-v[2]:+.4f} WRONG={w[2]:+.4f} NEG={n[2]:+.4f}")
print("[18/20] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[19/20] SUMMARY")
print("\n"+"="*128);print("TEST 225 RESULTS");print("="*128)
print(f"MODE: SINGLE L08 PACKET INJECTION -> FREE ENDOGENOUS TRANSPORT | PRIMARY DOSE={PRIMARY:.4f}")
print("NO L09-L27 RE-INJECTION | WEIGHTS FROZEN | SOURCE FACTS ABSENT FROM BLIND QUERY | CANDIDATES NEVER ENTER INTERVENTION")
print("\nMEAN CORRECT TRAJECTORY")
for L in [8,9,10,12,16,19,20,24,27]:
    rel=np.mean([REL[(i,PRIMARY,"CORRECT")][L] for i in range(M)])
    src=np.mean([cos(DELTA[(i,PRIMARY,"CORRECT")][8],DELTA[(i,PRIMARY,"CORRECT")][L]) for i in range(M)])
    cw=np.mean([cos(DELTA[(i,PRIMARY,"CORRECT")][L],DELTA[(i,PRIMARY,"WRONG")][L]) for i in range(M)])
    cross=np.mean([max(cos(DELTA[(i,PRIMARY,"CORRECT")][L],DELTA[(j,PRIMARY,"CORRECT")][L]) for j in range(M) if j!=i) for i in range(M)])
    print(f"L{L:02d} displacement={rel*100:7.3f}% sourceCos={src:+.4f} correctWrongCos={cw:+.4f} bestCrossFactCos={cross:+.4f}")
print("\nBEHAVIOR")
for b in ["CORRECT","WRONG","NEG"]:
    dm=[];dlp=[]
    for qi in range(M):
        v=BEH[(qi,"VANILLA")];r=BEH[(qi,b)]
        dm.append(r[2]-v[2]);dlp.append(r[0]-v[0])
    print(f"{b:8s} meanΔmargin={np.mean(dm):+.4f} meanΔtargetLP={np.mean(dlp):+.4f} marginImproved={sum(x>0 for x in dm)}/{M}")
print("\nINTERPRETATION GATE")
L19_CW=np.mean([cos(DELTA[(i,PRIMARY,"CORRECT")][19],DELTA[(i,PRIMARY,"WRONG")][19]) for i in range(M)])
L27_CW=np.mean([cos(DELTA[(i,PRIMARY,"CORRECT")][27],DELTA[(i,PRIMARY,"WRONG")][27]) for i in range(M)])
L19_X=np.mean([max(cos(DELTA[(i,PRIMARY,"CORRECT")][19],DELTA[(j,PRIMARY,"CORRECT")][19]) for j in range(M) if j!=i) for i in range(M)])
L27_X=np.mean([max(cos(DELTA[(i,PRIMARY,"CORRECT")][27],DELTA[(j,PRIMARY,"CORRECT")][27]) for j in range(M) if j!=i) for i in range(M)])
print(f"L19 correctWrongCos={L19_CW:+.4f} bestCrossFactCos={L19_X:+.4f}")
print(f"L27 correctWrongCos={L27_CW:+.4f} bestCrossFactCos={L27_X:+.4f}")
if L19_CW>0.90 and L27_CW>0.90:
    print("RESULT: TRAJECTORY_COLLAPSE — correct/wrong packet perturbations converge to a highly shared downstream direction.")
elif L19_CW<0.75 or L27_CW<0.75:
    print("RESULT: IDENTITY_REMAINS_SEPARABLE — downstream packet trajectories retain measurable packet-specific geometry.")
else:
    print("RESULT: PARTIAL_IDENTITY_RETENTION — downstream trajectories are neither fully collapsed nor strongly separated.")
print("-"*128)
print("Weights: PASS | Injection calls: L08 only | L09-L27 receive ZERO new injection")
print("Any L09-L27 displacement is model-native downstream transport from the single L08 intervention.")
print("="*128);print("[20/20] TEST 225 COMPLETE")


