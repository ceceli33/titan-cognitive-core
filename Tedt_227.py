# ==================================================================================================
# TEST 227 — PACKET IDENTITY RESIDUAL DECOMPOSITION
# TEST226 BASELINE -> RAW PACKET vs COMMON COMPONENT vs IDENTITY RESIDUAL
# SINGLE L08 INJECTION -> FREE L09-L27 TRAJECTORY -> OBJECT-IDENTITY DECODING
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223
#          -> TEST224 -> TEST225 -> TEST226 -> TEST227
# TEST226 WORKING MODEL / SYSTEM / FACTS / PACKET FORGE PRESERVED
# NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=227
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
PRIMARY=.04;LAYERS=[8,9,10,12,16,19,20,24,27]
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
print("="*128);print("TEST 227 — PACKET IDENTITY RESIDUAL DECOMPOSITION");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226 -> TEST227")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/24] Model...")
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
print("[2/24] Token maps...")
FMAP={};QENC=[]
for qi,(s,r,o) in enumerate(FACTS):
    fi=tok(chat(fact_text(s,r,o)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    full=fi.input_ids[0].tolist();ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Token map fail Q{qi+1}")
    q=qform(s,r);qe=tok(chat(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    if o.lower() in q.lower():raise RuntimeError("Target leakage.")
    FMAP[qi]=(fi,ss,rs,os_);QENC.append(qe)
    print(f"Q{qi+1} FACT subject={ss} relation={rs} object={os_} | BLIND slot={qe.input_ids.shape[1]-1}")
print("[3/24] RoPE...")
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
    finally:remove(hs)
    return S
SRC={}
for qi in range(M):
    SRC[qi]=capture_source(FMAP[qi][0]);print(f"Q{qi+1} captured")
print("[5/24] Reconstruct TEST222 source readout...")
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
print("[6/24] Forge TEST222 raw L08 packets...")
RAW={}
for qi in range(M):
    p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32);v=RAWV[qi]
    for h in QGROUP:
        w=max(x[0] for x in READ[qi] if x[1]==h);p[h]=w*v
    with torch.inference_mode():RAW[qi]=layers[SRC_LAYER].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"Q{qi+1} rawNorm={RAW[qi].norm():.4f}")
print("[7/24] Decompose packets: COMMON + IDENTITY RESIDUAL...")
COMMON=torch.stack([RAW[i] for i in range(M)]).mean(0)
RES={i:RAW[i]-COMMON for i in range(M)}
for qi in range(M):
    recon=(COMMON+RES[qi]-RAW[qi]).norm()
    print(f"Q{qi+1} raw={RAW[qi].norm():.4f} common={COMMON.norm():.4f} residual={RES[qi].norm():.4f} res/raw={float(RES[qi].norm()/RAW[qi].norm()):.4f} reconErr={recon:.3e}")
print("[8/24] Packet geometry...")
for qi in range(M):
    print(f"Q{qi+1} raw/common={cos(RAW[qi],COMMON):+.4f} raw/res={cos(RAW[qi],RES[qi]):+.4f} common/res={cos(COMMON,RES[qi]):+.4f}")
print("[9/24] Vanilla blind trajectories...")
@torch.inference_mode()
def vanilla_hidden(e):
    pos=e.input_ids.shape[1]-1;S={};hs=[]
    for L in range(SRC_LAYER,TOTAL):
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
print("[10/24] Single L08 injection + true post-injection trajectory...")
@torch.inference_mode()
def injected_hidden(e,packet,dose=PRIMARY):
    pos=e.input_ids.shape[1]-1;S={};hs=[];calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float();d=unit(packet)*z.norm(dim=-1,keepdim=True)*float(dose)
        y[:,-1,:]=(z+d).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    ih=layers[SRC_LAYER].register_forward_hook(inject)
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
    if calls!=1:raise RuntimeError(f"Injection calls={calls}, expected 1")
    return S
BRANCHES=["RAW","RESIDUAL","COMMON","NEG_RESIDUAL"]
TRAJ={};DELTA={};REL={}
for qi in range(M):
    packets={"RAW":RAW[qi],"RESIDUAL":RES[qi],"COMMON":COMMON,"NEG_RESIDUAL":-RES[qi]}
    for b in BRANCHES:
        if packets[b].norm()<EPS:raise RuntimeError(f"Zero packet Q{qi+1} {b}")
        TRAJ[(qi,b)]=injected_hidden(QENC[qi],packets[b])
        DELTA[(qi,b)]={};REL[(qi,b)]={}
        for L in range(SRC_LAYER,TOTAL):
            d=TRAJ[(qi,b)][L]-VAN[qi][L]
            DELTA[(qi,b)][L]=d;REL[(qi,b)][L]=float(d.norm()/VAN[qi][L].norm().clamp_min(EPS))
    print(f"Q{qi+1} complete")
print("[11/24] Verify L08 injections...")
for qi in range(M):
    print(f"Q{qi+1}",end="")
    for b,p in [("RAW",RAW[qi]),("RESIDUAL",RES[qi]),("COMMON",COMMON),("NEG_RESIDUAL",-RES[qi])]:
        d=DELTA[(qi,b)][8]
        print(f" {b}:disp={REL[(qi,b)][8]*100:.3f}% cos={cos(d,p):+.4f}",end="")
    print()
print("[12/24] Build object identity reference geometry...")
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
OBJRES={}
for L in range(SRC_LAYER,TOTAL):
    mean=torch.stack([OBJ[i][L] for i in range(M)]).mean(0)
    for i in range(M):OBJRES[(i,L)]=unit(OBJ[i][L]-mean)
print("[13/24] Object decoding by branch...")
def decode_rows(L,b):
    rows=[]
    for qi in range(M):
        v=unit(DELTA[(qi,b)][L]);scores=[cos(v,OBJRES[(j,L)]) for j in range(M)]
        order=np.argsort(scores)[::-1].tolist();rank=order.index(qi)+1
        rows.append((qi,rank,scores[qi],max(scores[j] for j in range(M) if j!=qi),order[0],scores))
    return rows
for b in BRANCHES:
    print("\n"+b)
    for L in LAYERS:
        r=decode_rows(L,b);top=sum(x[1]==1 for x in r);mr=np.mean([x[1] for x in r]);mg=np.mean([x[2]-x[3] for x in r])
        print(f"L{L:02d} top1={top}/{M} meanRank={mr:.3f} identityMargin={mg:+.4f}")
print("[14/24] Per-fact RESIDUAL decoding...")
for L in LAYERS:
    print(f"\nL{L:02d}")
    for qi,rank,own,best,top,scores in decode_rows(L,"RESIDUAL"):
        print(f" Q{qi+1} rank={rank} own={own:+.4f} bestWrong={best:+.4f} top=Q{top+1}")
print("[15/24] Raw vs residual trajectory geometry...")
for L in LAYERS:
    rr=[];rc=[];rn=[]
    for qi in range(M):
        rr.append(cos(DELTA[(qi,"RAW")][L],DELTA[(qi,"RESIDUAL")][L]))
        rc.append(cos(DELTA[(qi,"RAW")][L],DELTA[(qi,"COMMON")][L]))
        rn.append(cos(DELTA[(qi,"RESIDUAL")][L],DELTA[(qi,"NEG_RESIDUAL")][L]))
    print(f"L{L:02d} RAW/RES={np.mean(rr):+.4f} RAW/COMMON={np.mean(rc):+.4f} RES/NEG={np.mean(rn):+.4f}")
print("[16/24] Cross-fact fingerprint separation...")
for b in ["RAW","RESIDUAL","COMMON"]:
    print("\n"+b)
    for L in LAYERS:
        vals=[]
        for qi in range(M):
            v=DELTA[(qi,b)][L]
            vals.append(max(cos(v,DELTA[(j,b)][L]) for j in range(M) if j!=qi))
        print(f"L{L:02d} bestCross={np.mean(vals):+.4f}")
print("[17/24] Permutation null: RAW vs RESIDUAL...")
rng=np.random.default_rng(SEED);PERMS=2000
PERM={}
for b in ["RAW","RESIDUAL","COMMON"]:
    print("\n"+b)
    for L in [8,12,16,19,24,27]:
        R=decode_rows(L,b);S=np.array([x[5] for x in R],dtype=np.float64)
        obs=float(np.mean([S[i,i]-np.max(np.delete(S[i],i)) for i in range(M)]))
        null=[]
        for _ in range(PERMS):
            p=rng.permutation(M);vals=[]
            for i in range(M):
                t=p[i];vals.append(S[i,t]-np.max(np.delete(S[i],t)))
            null.append(float(np.mean(vals)))
        mu=float(np.mean(null));sd=float(np.std(null)+1e-12);z=(obs-mu)/sd
        pval=(1+sum(x>=obs for x in null))/(PERMS+1)
        PERM[(b,L)]=(obs,z,pval)
        print(f"L{L:02d} observed={obs:+.4f} z={z:+.3f} p={pval:.4f}")
print("[18/24] Behavioral candidate scorer...")
@torch.inference_mode()
def prefill(qi,packet=None,dose=0.):
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
    logits,pkv=prefill(qi,packet,dose);vals=[]
    for i,t in enumerate(y):
        vals.append(torch.log_softmax(logits[0],-1)[t])
        if i<y.numel()-1:
            o=model(input_ids=t.view(1,1),past_key_values=pkv,use_cache=True,return_dict=True)
            logits=o.logits[:,-1,:].float();pkv=o.past_key_values
    return float(torch.stack(vals).sum())
def score(qi,packet=None,dose=0.):
    t=lp(qi,FACTS[qi][2],packet,dose)
    w=max(lp(qi,FACTS[j][2],packet,dose) for j in range(M) if j!=qi)
    return t,w,t-w
BEH={}
for qi in range(M):
    BEH[(qi,"VANILLA")]=score(qi)
    for b,p in [("RAW",RAW[qi]),("RESIDUAL",RES[qi]),("COMMON",COMMON),("NEG_RESIDUAL",-RES[qi])]:
        BEH[(qi,b)]=score(qi,p,PRIMARY)
    print(f"Q{qi+1} VAN={BEH[(qi,'VANILLA')][2]:+.4f} RAW={BEH[(qi,'RAW')][2]:+.4f} RES={BEH[(qi,'RESIDUAL')][2]:+.4f} COMMON={BEH[(qi,'COMMON')][2]:+.4f} NEGRES={BEH[(qi,'NEG_RESIDUAL')][2]:+.4f}")
print("[19/24] Behavioral summary...")
for b in BRANCHES:
    dm=[];dlp=[]
    for qi in range(M):
        v=BEH[(qi,"VANILLA")];r=BEH[(qi,b)]
        dm.append(r[2]-v[2]);dlp.append(r[0]-v[0])
    print(f"{b:12s} meanΔmargin={np.mean(dm):+.4f} meanΔtargetLP={np.mean(dlp):+.4f} improved={sum(x>0 for x in dm)}/{M}")
print("[20/24] Energy-matched decomposition check...")
# All intervention branches use the same physical dose because every direction is normalized
# before injection. This isolates direction rather than packet norm.
for b in BRANCHES:
    vals=[REL[(i,b)][8]*100 for i in range(M)]
    print(f"{b:12s} L08 meanDose={np.mean(vals):.4f}% min={np.min(vals):.4f}% max={np.max(vals):.4f}%")
print("[21/24] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[22/24] Decision metrics...")
SUMMARY={}
for b in BRANCHES:
    late=[]
    for L in [19,20,24,27]:
        r=decode_rows(L,b)
        late.append((sum(x[1]==1 for x in r)/M,np.mean([x[1] for x in r]),np.mean([x[2]-x[3] for x in r])))
    SUMMARY[b]=(float(np.mean([x[0] for x in late])),float(np.mean([x[1] for x in late])),float(np.mean([x[2] for x in late])))
    print(f"{b:12s} lateTop1={SUMMARY[b][0]:.3f} lateMeanRank={SUMMARY[b][1]:.3f} lateIdentityMargin={SUMMARY[b][2]:+.4f}")
print("[23/24] RESULTS")
print("\n"+"="*128);print("TEST 227 RESULTS");print("="*128)
print(f"MODE: RAW vs COMMON vs IDENTITY-RESIDUAL PACKET | SINGLE L08 INJECTION | DOSE={PRIMARY:.4f}")
print("L09-L27 NEW INJECTION: ZERO | WEIGHTS FROZEN | CANDIDATES NEVER ENTER INTERVENTION")
print("\nPACKET DECOMPOSITION")
print(f"COMMON norm={COMMON.norm():.4f}")
for qi in range(M):
    print(f"Q{qi+1} RAW={RAW[qi].norm():.4f} RES={RES[qi].norm():.4f} RES/RAW={float(RES[qi].norm()/RAW[qi].norm()):.4f}")
print("\nLATE OBJECT DECODING")
for b in BRANCHES:
    print(f"{b:12s} top1={SUMMARY[b][0]:.3f} meanRank={SUMMARY[b][1]:.3f} identityMargin={SUMMARY[b][2]:+.4f}")
print("\nBEHAVIOR")
for b in BRANCHES:
    dm=[BEH[(i,b)][2]-BEH[(i,"VANILLA")][2] for i in range(M)]
    dlp=[BEH[(i,b)][0]-BEH[(i,"VANILLA")][0] for i in range(M)]
    print(f"{b:12s} meanΔmargin={np.mean(dm):+.4f} meanΔtargetLP={np.mean(dlp):+.4f} improved={sum(x>0 for x in dm)}/{M}")
print("\nINTERPRETATION GATE")
R=SUMMARY["RESIDUAL"];W=SUMMARY["RAW"];C=SUMMARY["COMMON"]
if R[0]>=.50 and R[1]<=3.0 and R[2]>W[2] and R[2]>C[2]:
    print("RESULT: IDENTITY_RESIDUAL_ENRICHED — removing the common packet component exposes substantially stronger object-specific signal.")
elif R[0]>W[0] or R[1]<W[1] or R[2]>W[2]:
    print("RESULT: PARTIAL_RESIDUAL_ENRICHMENT — identity residual improves at least one decoding dimension, but object identity remains incomplete.")
else:
    print("RESULT: NO_IDENTITY_RESIDUAL_ENRICHMENT — simple common-component subtraction does not recover clear object identity.")
print("-"*128)
print("Weights: PASS | L08 only intervention | L09-L27 displacement is endogenous downstream propagation")
print("RAW = original TEST222 packet | COMMON = across-fact packet mean | RESIDUAL = RAW - COMMON")
print("="*128);print("[24/24] TEST 227 COMPLETE")



