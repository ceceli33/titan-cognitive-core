# ==================================================================================================
# TEST 226 — ENDOGENOUS PACKET IDENTITY DECODING
# SINGLE L08 PACKET INJECTION -> FREE L09-L27 TRAJECTORY -> OBJECT-IDENTITY DECODING
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223
#          -> TEST224 -> TEST225 -> TEST226
# TEST225 WORKING MODEL / SYSTEM / FACTS / TEST222 PACKET FORGE PRESERVED
# FIX: L08 POST-INJECTION STATE CAPTURED AFTER THE INJECTION HOOK
# QUESTION: does downstream Δh retain decodable OBJECT identity, or only packet-specific fingerprint?
# NO L09-L27 RE-INJECTION | WEIGHTS FROZEN | PREFILL ONLY
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=226
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
print("="*128);print("TEST 226 — ENDOGENOUS PACKET IDENTITY DECODING");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/22] Model...")
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
def remove(hs):
    for h in hs:h.remove()
print("[2/22] Token maps...")
FMAP={};QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    fi=tok(chat(fact_text(s,r,o)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    full=fi.input_ids[0].tolist();ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Token map fail Q{qi+1}")
    q=qform(s,r);qe=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if o.lower() in q.lower():raise RuntimeError("Target leakage.")
    FMAP[qi]=(fi,ss,rs,os_);QENC.append(qe)
    print(f"Q{qi+1} FACT subject={ss} relation={rs} object={os_} | BLIND slot={qe.input_ids.shape[1]-1}")
print("[3/22] RoPE...")
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
    finally:remove(hs)
    return S
SRC={}
for qi in range(M):
    SRC[qi]=capture_source(FMAP[qi][0]);print(f"Q{qi+1} captured")
print("[5/22] Reconstruct TEST222 source readout...")
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
print("[6/22] Forge TEST222 L08 packets...")
PACK={}
for qi in range(M):
    p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32);v=RAWV[qi]
    for h in QGROUP:
        w=max(x[0] for x in READ[qi] if x[1]==h);p[h]=w*v
    with torch.inference_mode():PACK[qi]=layers[SRC_LAYER].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"Q{qi+1} packetNorm={PACK[qi].norm():.4f}")
WRONG={i:(i+1)%M for i in range(M)}
print("[7/22] Vanilla blind trajectories...")
@torch.inference_mode()
def vanilla_hidden(e):
    pos=e.input_ids.shape[1]-1;S={};hs=[]
    for L in range(TOTAL):
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S
VAN={}
for qi in range(M):
    VAN[qi]=vanilla_hidden(QENC[qi]);print(f"Q{qi+1} vanilla captured")
print("[8/22] Inject once at L08 and capture true post-injection L08→L27...")
@torch.inference_mode()
def injected_hidden(e,packet,dose):
    pos=e.input_ids.shape[1]-1;S={};hs=[];calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float();d=unit(packet)*z.norm(dim=-1,keepdim=True)*float(dose)
        y[:,-1,:]=(z+d).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    ih=layers[SRC_LAYER].register_forward_hook(inject)
    # Registered AFTER injection hook so L08 capture sees the modified output.
    for L in range(SRC_LAYER,TOTAL):
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        remove(hs);ih.remove()
    if calls!=1:raise RuntimeError(f"L08 injection calls={calls}, expected 1")
    return S
TRAJ={}
for qi in range(M):
    for dose in DOSES:
        for b,p in [("CORRECT",PACK[qi]),("WRONG",PACK[WRONG[qi]]),("NEG",-PACK[qi])]:
            TRAJ[(qi,dose,b)]=injected_hidden(QENC[qi],p,dose)
    print(f"Q{qi+1} complete")
print("[9/22] Build true Δh trajectories...")
DELTA={};REL={}
for qi in range(M):
    for dose in DOSES:
        for b in ["CORRECT","WRONG","NEG"]:
            DELTA[(qi,dose,b)]={};REL[(qi,dose,b)]={}
            for L in range(SRC_LAYER,TOTAL):
                d=TRAJ[(qi,dose,b)][L]-VAN[qi][L]
                DELTA[(qi,dose,b)][L]=d
                REL[(qi,dose,b)][L]=float(d.norm()/VAN[qi][L].norm().clamp_min(EPS))
print("[10/22] Verify L08 capture...")
for qi in range(M):
    d=DELTA[(qi,PRIMARY,"CORRECT")][SRC_LAYER]
    print(f"Q{qi+1} L08 displacement={REL[(qi,PRIMARY,'CORRECT')][8]*100:.4f}% packetCos={cos(d,PACK[qi]):+.4f}")
print("[11/22] Build object identity reference geometry...")
# Reference identity is extracted from the actual source facts at OBJECT_END.
# Same layer, same representation space. No candidate answer enters intervention.
@torch.inference_mode()
def source_hidden(e,obj_end):
    S={};hs=[]
    for L in range(SRC_LAYER,TOTAL):
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out
                S[li]=x[0,obj_end].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S
OBJ={}
for qi in range(M):
    OBJ[qi]=source_hidden(FMAP[qi][0],FMAP[qi][3][-1]);print(f"Q{qi+1} object reference captured")
print("[12/22] Residualize object references...")
# Remove the across-fact common component at each layer.
OBJRES={}
for L in range(SRC_LAYER,TOTAL):
    mean=torch.stack([OBJ[i][L] for i in range(M)]).mean(0)
    for i in range(M):OBJRES[(i,L)]=unit(OBJ[i][L]-mean)
print("[13/22] Decode trajectory against object references...")
def ranks_for_layer(L,dose=PRIMARY,branch="CORRECT"):
    rows=[]
    for qi in range(M):
        v=unit(DELTA[(qi,dose,branch)][L])
        scores=[cos(v,OBJRES[(j,L)]) for j in range(M)]
        order=np.argsort(scores)[::-1].tolist();rank=order.index(qi)+1
        rows.append((qi,rank,scores[qi],max(scores[j] for j in range(M) if j!=qi),order[0],scores))
    return rows
for L in [8,9,10,12,16,19,20,24,27]:
    r=ranks_for_layer(L)
    top=sum(x[1]==1 for x in r);mr=np.mean([x[1] for x in r]);margin=np.mean([x[2]-x[3] for x in r])
    print(f"L{L:02d} top1={top}/{M} meanRank={mr:.3f} meanIdentityMargin={margin:+.4f}")
print("[14/22] Per-fact primary decoding...")
for L in [8,9,12,16,19,24,27]:
    print(f"\nL{L:02d}")
    for qi,rank,own,best,top,scores in ranks_for_layer(L):
        print(f" Q{qi+1} rank={rank} own={own:+.4f} bestWrong={best:+.4f} top=Q{top+1}")
print("[15/22] Wrong-packet label sanity...")
# A WRONG packet for query i should decode, if identity is preserved, toward WRONG[i], not i.
for L in [8,9,12,16,19,24,27]:
    hit=0;mr=[]
    for qi in range(M):
        expected=WRONG[qi];v=unit(DELTA[(qi,PRIMARY,"WRONG")][L])
        scores=[cos(v,OBJRES[(j,L)]) for j in range(M)]
        order=np.argsort(scores)[::-1].tolist();rank=order.index(expected)+1;mr.append(rank);hit+=rank==1
    print(f"L{L:02d} wrongPacketExpectedTop1={hit}/{M} meanRank={np.mean(mr):.3f}")
print("[16/22] Negative direction sanity...")
for L in [8,9,12,16,19,24,27]:
    vals=[]
    for qi in range(M):
        c=DELTA[(qi,PRIMARY,"CORRECT")][L];n=DELTA[(qi,PRIMARY,"NEG")][L]
        vals.append(cos(c,n))
    print(f"L{L:02d} correctNegCos={np.mean(vals):+.4f}")
print("[17/22] Cross-fact trajectory fingerprint separation...")
for L in [8,9,12,16,19,24,27]:
    vals=[]
    for qi in range(M):
        v=DELTA[(qi,PRIMARY,"CORRECT")][L]
        vals.append(max(cos(v,DELTA[(j,PRIMARY,"CORRECT")][L]) for j in range(M) if j!=qi))
    print(f"L{L:02d} bestCrossTrajectoryCos={np.mean(vals):+.4f}")
print("[18/22] Dose stability of identity decoding...")
for d in DOSES:
    print(f"DOSE={d:.3f}",end="")
    for L in [8,12,19,27]:
        r=ranks_for_layer(L,d,"CORRECT");top=sum(x[1]==1 for x in r);mr=np.mean([x[1] for x in r])
        print(f" L{L:02d}:{top}/{M},r={mr:.2f}",end="")
    print()
print("[19/22] Permutation null...")
# Exact 8-way label permutation null. Identity metric is evaluated against shuffled object labels.
rng=np.random.default_rng(SEED);PERMS=2000
NULL={L:[] for L in [8,9,12,16,19,24,27]}
OBS={}
for L in NULL:
    R=ranks_for_layer(L);S=np.array([x[5] for x in R],dtype=np.float64)
    obs=float(np.mean([S[i,i]-np.max(np.delete(S[i],i)) for i in range(M)]));OBS[L]=obs
    for _ in range(PERMS):
        p=rng.permutation(M)
        vals=[]
        for i in range(M):
            t=p[i];vals.append(S[i,t]-np.max(np.delete(S[i],t)))
        NULL[L].append(float(np.mean(vals)))
    mu=float(np.mean(NULL[L]));sd=float(np.std(NULL[L])+1e-12);z=(obs-mu)/sd
    pval=(1+sum(x>=obs for x in NULL[L]))/(PERMS+1)
    print(f"L{L:02d} observed={obs:+.4f} null={mu:+.4f}±{sd:.4f} z={z:+.3f} p={pval:.4f}")
print("[20/22] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[21/22] RESULTS")
print("\n"+"="*128);print("TEST 226 RESULTS");print("="*128)
print(f"MODE: SINGLE L08 PACKET -> ENDOGENOUS Δh -> OBJECT-IDENTITY DECODING | PRIMARY DOSE={PRIMARY:.4f}")
print("L08 POST-INJECTION CAPTURE: FIXED | L09-L27 NEW INJECTION: ZERO | WEIGHTS FROZEN")
print("\nPRIMARY OBJECT DECODING")
SUMMARY={}
for L in [8,9,10,12,16,19,20,24,27]:
    r=ranks_for_layer(L);top=sum(x[1]==1 for x in r);mr=float(np.mean([x[1] for x in r]));mg=float(np.mean([x[2]-x[3] for x in r]))
    SUMMARY[L]=(top,mr,mg);print(f"L{L:02d} top1={top}/{M} meanRank={mr:.3f} identityMargin={mg:+.4f}")
print("\nTRAJECTORY PHYSICS")
for L in [8,9,12,16,19,20,24,27]:
    rel=np.mean([REL[(i,PRIMARY,"CORRECT")][L] for i in range(M)])
    cw=np.mean([cos(DELTA[(i,PRIMARY,"CORRECT")][L],DELTA[(i,PRIMARY,"WRONG")][L]) for i in range(M)])
    cross=np.mean([max(cos(DELTA[(i,PRIMARY,"CORRECT")][L],DELTA[(j,PRIMARY,"CORRECT")][L]) for j in range(M) if j!=i) for i in range(M)])
    print(f"L{L:02d} displacement={rel*100:7.3f}% correctWrongCos={cw:+.4f} bestCrossTrajectoryCos={cross:+.4f}")
print("\nINTERPRETATION GATE")
late=[19,20,24,27]
late_top=np.mean([SUMMARY[L][0]/M for L in late]);late_rank=np.mean([SUMMARY[L][1] for L in late])
if late_top>=.50 and late_rank<=3.0:
    print("RESULT: OBJECT_IDENTITY_DECODABLE — downstream packet trajectory contains substantial object-specific information.")
elif late_top<=.25 and late_rank>=3.5:
    print("RESULT: PACKET_FINGERPRINT_WITHOUT_CLEAR_OBJECT_DECODING — trajectories separate, but object identity is not cleanly decoded.")
else:
    print("RESULT: PARTIAL_OBJECT_IDENTITY — downstream trajectory carries weak/mixed object-specific information.")
print("-"*128)
print("Weights: PASS | Source facts absent from blind queries | Candidate answers never enter intervention")
print("Only L08 is intervened. L09-L27 measurements are endogenous downstream consequences.")
print("="*128);print("[22/22] TEST 226 COMPLETE")



