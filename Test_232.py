# ==================================================================================================
# TEST 232 — DISCRIMINATIVE OBJECT-TOKEN READOUT X-RAY
# TEST231 WORKING BASELINE -> TEST230 OBJECT MAIN-EFFECT PACKETS -> SINGLE L08 INJECTION
# FIX: DO NOT COMPARE SHARED FIRST TOKEN " the"
# QUESTION: AFTER THE SHARED PREFIX " the", DOES THE CARRIER SELECT amber/silver/violet/...?
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> ... -> TEST229 -> TEST230 -> TEST231 -> TEST232
# TEST231 MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / OBJECT MAIN EFFECT PRESERVED
# NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=232
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;SRC_LAYER=8;KVH=0;EPS=1e-8
PRIMARY=.04;LAYERS=list(range(8,28));SHOW=[8,9,10,12,16,19,20,24,27]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere",
         "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
C=len(CONTEXTS);O=len(OBJECTS)
print("="*128);print("TEST 232 — DISCRIMINATIVE OBJECT-TOKEN READOUT X-RAY");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220 -> TEST221 -> TEST222 -> TEST223 -> TEST224 -> TEST225 -> TEST226 -> TEST227 -> TEST228 -> TEST229 -> TEST230 -> TEST231 -> TEST232")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/27] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;NH=model.config.num_attention_heads
NKV=model.config.num_key_value_heads;HD=H//NH;GROUP=NH//NKV
if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
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
    mp={"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}
    return f"What does {s} {mp[r]}?"
def remove(hs):
    for h in hs:h.remove()
print("[2/27] Build TEST231 4x8 factorial...")
FMAP={};QENC={}
for c,(s,r) in enumerate(CONTEXTS):
    qe=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE);QENC[c]=qe
    for o,obj in enumerate(OBJECTS):
        fi=tok(chat(fact_text(s,r,obj)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
        full=fi.input_ids[0].tolist();ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,obj)
        if not ss or not rs or not os_:raise RuntimeError(f"Token map fail C{c+1} O{o+1}")
        if obj.lower() in qform(s,r).lower():raise RuntimeError("Target leakage.")
        FMAP[(c,o)]=(fi,ss,rs,os_)
    print(f"C{c+1} {s} {r} | blindSlot={qe.input_ids.shape[1]-1}")
print("[3/27] Candidate tokenization + discriminative tokens...")
CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj);CANDS.append(b if len(b)<=len(a) else a)
COMMON=CANDS[0][0]
if not all(x[0]==COMMON for x in CANDS):raise RuntimeError("Candidates do not share first token.")
DISC=[x[1] for x in CANDS]
if len(set(DISC))!=O:raise RuntimeError("Second candidate token is not unique.")
print(f"sharedPrefixToken={COMMON} text={tok.decode([COMMON])!r}")
for o,x in enumerate(CANDS):
    print(f"O{o+1} {OBJECTS[o]} ids={x} discriminative={DISC[o]} text={tok.decode([DISC[o]])!r}")
print("[4/27] RoPE...")
rotary=model.model.rotary_emb
MAXSEQ=max(max(v[0].input_ids.shape[1] for v in FMAP.values()),max(v.input_ids.shape[1] for v in QENC.values()))+4
dummy=torch.zeros(1,MAXSEQ,H,device=DEVICE,dtype=model.dtype);pos=torch.arange(MAXSEQ,device=DEVICE).unsqueeze(0)
with torch.inference_mode():COS,SIN=rotary(dummy,pos)
COS=COS[0].float();SIN=SIN[0].float()
def rotate_half(x):
    n=x.shape[-1]//2;return torch.cat((-x[...,n:],x[...,:n]),dim=-1)
def rope(x,p):return x*COS[p]+rotate_half(x)*SIN[p]
print(f"RoPE={type(rotary).__name__} max_seq={MAXSEQ}")
print("[5/27] Capture TEST222 source Q/K/V...")
@torch.inference_mode()
def capture_source(e):
    S={};hs=[]
    for name,mod in [("Q",layers[8].self_attn.q_proj),("K",layers[8].self_attn.k_proj),("V",layers[8].self_attn.v_proj)]:
        def mk(n):
            def hk(m,args,out):S[n]=out[0].float().detach().clone()
            return hk
        hs.append(mod.register_forward_hook(mk(name)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:remove(hs)
    return S
SRC={}
for c in range(C):
    for o in range(O):SRC[(c,o)]=capture_source(FMAP[(c,o)][0])
    print(f"C{c+1}: 8 source captures complete")
print("[6/27] Reconstruct TEST222 readout + RAW packets...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    kh=qhead//GROUP;q=rope(qh(S["Q"],qpos,qhead),qpos)
    K=torch.stack([rope(kvh(S["K"],p,kh),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)
RAW={}
for c in range(C):
    for o in range(O):
        S=SRC[(c,o)];oe=FMAP[(c,o)][3][-1];rows=[]
        for h in QGROUP:
            for qp in range(oe,FMAP[(c,o)][0].input_ids.shape[1]):
                a=attn_row(S,qp,h);rows.append((float(a[oe]),h,qp))
        v=kvh(S["V"],oe,KVH);p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32)
        for h in QGROUP:
            w=max(x[0] for x in rows if x[1]==h);p[h]=w*v
        with torch.inference_mode():RAW[(c,o)]=layers[8].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"C{c+1}: RAW packets complete")
print("[7/27] TEST230 object-main-effect packets...")
GRAND=torch.stack(list(RAW.values())).mean(0)
CMEAN={c:torch.stack([RAW[(c,o)] for o in range(O)]).mean(0) for c in range(C)}
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
for o in range(O):print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")
print("[8/27] Vanilla prompt trajectories...")
@torch.inference_mode()
def capture_prompt(e,packet=None):
    pos=e.input_ids.shape[1]-1;S={};hs=[];calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float();d=unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY
        y[:,-1,:]=(z+d).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    ih=layers[8].register_forward_hook(inject) if packet is not None else None
    for L in LAYERS:
        def mk(li):
            def hk(m,args,out):
                x=out[0] if isinstance(out,tuple) else out;S[li]=x[0,pos].float().detach().clone()
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    try:r=model(**e,use_cache=True,return_dict=True)
    finally:
        remove(hs)
        if ih is not None:ih.remove()
    if packet is not None and calls!=1:raise RuntimeError(f"Injection calls={calls}")
    return S,r.logits[0,-1].float().detach().clone(),r.past_key_values
VAN={};VPREF={};VPAST={}
for c in range(C):
    VAN[c],VPREF[c],VPAST[c]=capture_prompt(QENC[c],None);print(f"C{c+1} vanilla captured")
print("[9/27] Object interventions...")
HID={};IPREF={};IPAST={}
for c in range(C):
    for o in range(O):
        S,l,p=capture_prompt(QENC[c],OBJ[o]);IPREF[(c,o)]=l;IPAST[(c,o)]=p
        for L in LAYERS:HID[(c,o,L)]=S[L]
    print(f"C{c+1}: 8 interventions complete")
print("[10/27] Verify L08...")
for c in range(C):
    ds=[];cs=[]
    for o in range(O):
        d=HID[(c,o,8)]-VAN[c][8];ds.append(float(d.norm()/VAN[c][8].norm())*100);cs.append(cos(d,OBJ[o]))
    print(f"C{c+1} dose={np.mean(ds):.4f}% objectCos={np.mean(cs):+.4f}")
print("[11/27] Prompt-boundary lens sanity...")
@torch.inference_mode()
def lens(h):
    return model.lm_head(model.model.norm(h.to(model.dtype).unsqueeze(0)))[0].float()
for L in SHOW:
    vals=[]
    for c in range(C):
        vb=lens(VAN[c][L])
        for o in range(O):vals.append(float(lens(HID[(c,o,L)])[COMMON]-vb[COMMON]))
    print(f"L{L:02d} sharedPrefixΔlogit={np.mean(vals):+.4f}")
print("[12/27] Advance ONE shared prefix token using frozen prompt KV...")
@torch.inference_mode()
def advance_shared(first_logits,past):
    lp=float(torch.log_softmax(first_logits.float(),dim=-1)[COMMON])
    x=torch.tensor([[COMMON]],device=DEVICE)
    r=model(input_ids=x,past_key_values=past,use_cache=True,return_dict=True)
    return lp,r.logits[0,-1].float().detach().clone(),r.past_key_values
VTHE={};ITHE={};VTHELP={};ITHELP={}
for c in range(C):
    VTHELP[c],VTHE[c],_=advance_shared(VPREF[c],VPAST[c])
    for o in range(O):
        ITHELP[(c,o)],ITHE[(c,o)],_=advance_shared(IPREF[(c,o)],IPAST[(c,o)])
    print(f"C{c+1}: shared-prefix continuation complete")
print("[13/27] Discriminative token readout after shared ' the'...")
DISCRES=[]
for c in range(C):
    base=VTHE[c]
    for o in range(O):
        li=ITHE[(c,o)]
        scores=[float(li[t]) for t in DISC]
        dscores=[float(li[t]-base[t]) for t in DISC]
        rank=np.argsort(dscores)[::-1].tolist().index(o)+1
        margin=dscores[o]-max(dscores[j] for j in range(O) if j!=o)
        DISCRES.append((c,o,rank,margin,dscores[o]))
for c in range(C):
    z=[x for x in DISCRES if x[0]==c]
    print(f"C{c+1} top1={sum(x[2]==1 for x in z)}/{O} meanRank={np.mean([x[2] for x in z]):.3f} Δmargin={np.mean([x[3] for x in z]):+.4f} targetΔ={np.mean([x[4] for x in z]):+.4f}")
print("[14/27] Aggregate discriminative-token result...")
TOP=sum(x[2]==1 for x in DISCRES)/(C*O);RANK=np.mean([x[2] for x in DISCRES])
MARGIN=np.mean([x[3] for x in DISCRES]);TD=np.mean([x[4] for x in DISCRES])
print(f"top1={TOP*100:.1f}% meanRank={RANK:.3f} meanΔmargin={MARGIN:+.4f} meanTargetΔlogit={TD:+.4f} chance={100/O:.1f}%")
print("[15/27] Per-object discriminative readout...")
for o in range(O):
    z=[x for x in DISCRES if x[1]==o]
    print(f"O{o+1} {OBJECTS[o]} top1={sum(x[2]==1 for x in z)}/{C} rank={np.mean([x[2] for x in z]):.3f} Δmargin={np.mean([x[3] for x in z]):+.4f} targetΔ={np.mean([x[4] for x in z]):+.4f}")
print("[16/27] Shared-prefix probability effect...")
vals=[ITHELP[(c,o)]-VTHELP[c] for c in range(C) for o in range(O)]
print(f"meanΔlogP(' the')={np.mean(vals):+.6f} min={np.min(vals):+.6f} max={np.max(vals):+.6f}")
print("[17/27] Discriminative-token probability effect...")
PROB=[]
for c in range(C):
    vb=torch.log_softmax(VTHE[c],dim=-1)
    for o in range(O):
        il=torch.log_softmax(ITHE[(c,o)],dim=-1)
        target=float(il[DISC[o]]-vb[DISC[o]])
        wrong=[float(il[DISC[j]]-vb[DISC[j]]) for j in range(O) if j!=o]
        PROB.append((c,o,target,float(np.mean(wrong)),target-float(np.mean(wrong))))
print(f"targetΔlogP={np.mean([x[2] for x in PROB]):+.4f} wrongΔlogP={np.mean([x[3] for x in PROB]):+.4f} gap={np.mean([x[4] for x in PROB]):+.4f}")
print("[18/27] Full object sequence scorer...")
@torch.inference_mode()
def seq_lp(first_logits,past,cand):
    lp=0.;logits=first_logits;pkv=past
    for k,t in enumerate(cand):
        lp+=float(torch.log_softmax(logits.float(),dim=-1)[t])
        if k<len(cand)-1:
            r=model(input_ids=torch.tensor([[t]],device=DEVICE),past_key_values=pkv,use_cache=True,return_dict=True)
            logits=r.logits[0,-1].float();pkv=r.past_key_values
    return lp
SEQ={}
for c in range(C):
    vs=[seq_lp(VPREF[c],VPAST[c],cand) for cand in CANDS]
    for o in range(O):
        ss=[seq_lp(IPREF[(c,o)],IPAST[(c,o)],cand) for cand in CANDS]
        vm=vs[o]-max(vs[j] for j in range(O) if j!=o)
        im=ss[o]-max(ss[j] for j in range(O) if j!=o)
        SEQ[(c,o)]=(im,im-vm,ss[o]-vs[o])
    print(f"C{c+1} full sequence complete")
dm=[SEQ[(c,o)][1] for c in range(C) for o in range(O)]
dlp=[SEQ[(c,o)][2] for c in range(C) for o in range(O)]
print(f"meanΔmargin={np.mean(dm):+.4f} meanΔtargetLP={np.mean(dlp):+.4f} improved={sum(x>0 for x in dm)}/{C*O}")
print("[19/27] Second-token-only vs full-sequence agreement...")
second=[x[3] for x in DISCRES]
print(f"signAgreement={np.mean([(a>0)==(b>0) for a,b in zip(second,dm)]):.4f} correlation={np.corrcoef(second,dm)[0,1]:+.4f}")
print("[20/27] Correct vs wrong packet discriminative selectivity...")
SEL=[]
for c in range(C):
    for target in range(O):
        base=VTHE[c];correct=float(ITHE[(c,target)][DISC[target]]-base[DISC[target]])
        wrong=[float(ITHE[(c,p)][DISC[target]]-base[DISC[target]]) for p in range(O) if p!=target]
        SEL.append(correct-float(np.mean(wrong)))
print(f"correctPacketTargetTokenAdvantage={np.mean(SEL):+.4f} positive={sum(x>0 for x in SEL)}/{len(SEL)}")
print("[21/27] Label permutation null...")
rng=np.random.default_rng(SEED);PERMS=2000
table=[]
for c in range(C):
    base=VTHE[c]
    for o in range(O):table.append([float(ITHE[(c,o)][t]-base[t]) for t in DISC])
table=np.asarray(table);labels=np.tile(np.arange(O),C);vals=[]
for _ in range(PERMS):
    lab=rng.permutation(labels);hit=sum(int(np.argmax(table[i])==lab[i]) for i in range(len(table)))
    vals.append(hit/len(table))
mu=float(np.mean(vals));sd=float(np.std(vals)+1e-12);z=(TOP-mu)/sd
pval=(1+sum(x>=TOP for x in vals))/(PERMS+1)
print(f"observed={TOP:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={pval:.4f}")
print("[22/27] Layerwise carrier physical displacement...")
for L in SHOW:
    vals=[float((HID[(c,o,L)]-VAN[c][L]).norm()/VAN[c][L].norm())*100 for c in range(C) for o in range(O)]
    print(f"L{L:02d} meanDisplacement={np.mean(vals):.4f}%")
print("[23/27] Final-layer object-carrier separability sanity...")
CENTER={}
for c in range(C):
    for L in [19,24,27]:
        ds=[HID[(c,o,L)]-VAN[c][L] for o in range(O)];m=torch.stack(ds).mean(0)
        for o in range(O):CENTER[(c,o,L)]=ds[o]-m
for L in [19,24,27]:
    hits=0;ranks=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,L)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            scores=[cos(CENTER[(hold,o,L)],cent[j]) for j in range(O)]
            rank=np.argsort(scores)[::-1].tolist().index(o)+1;hits+=rank==1;ranks.append(rank)
    print(f"L{L:02d} carrierTop1={hits/(C*O)*100:.1f}% meanRank={np.mean(ranks):.3f}")
print("[24/27] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[25/27] Decision metrics...")
print(f"discTop1={TOP:.4f} discRank={RANK:.3f} discΔmargin={MARGIN:+.4f} discPermP={pval:.4f}")
print(f"sequenceΔmargin={np.mean(dm):+.4f} sequenceΔtargetLP={np.mean(dlp):+.4f}")
print("[26/27] RESULTS")
print("\n"+"="*128);print("TEST 232 RESULTS");print("="*128)
print(f"MODE: TEST230 OBJECT MAIN-EFFECT PACKETS | SINGLE L08 INJECTION | DOSE={PRIMARY:.4f}")
print(f"SHARED PREFIX: token {COMMON} {tok.decode([COMMON])!r}")
print("PRIMARY READOUT: AUTOREGRESSIVE NEXT TOKEN AFTER SHARED PREFIX; OBJECT TOKENS ARE DISTINCT")
print("NO TRANSPORT MAP | NO CONTROLLER | NO L09-L27 RE-INJECTION | WEIGHTS FROZEN")
print("\nDISCRIMINATIVE OBJECT-TOKEN READOUT")
print(f"top1={TOP*100:.1f}% chance={100/O:.1f}% meanRank={RANK:.3f} Δmargin={MARGIN:+.4f} targetΔlogit={TD:+.4f} permP={pval:.4f}")
print(f"targetΔlogP={np.mean([x[2] for x in PROB]):+.4f} wrongΔlogP={np.mean([x[3] for x in PROB]):+.4f} correctPacketAdvantage={np.mean(SEL):+.4f}")
print("\nFULL OBJECT SEQUENCE")
print(f"meanΔmargin={np.mean(dm):+.4f} meanΔtargetLP={np.mean(dlp):+.4f} improved={sum(x>0 for x in dm)}/{C*O}")
print("\nINTERPRETATION GATE")
if TOP>=.50 and MARGIN>0 and pval<=.05:
    print("RESULT: DISCRIMINATIVE_OBJECT_READOUT — after the shared prefix, the carrier selectively biases the correct object token.")
elif TOP>=.25 or np.mean(SEL)>0:
    print("RESULT: PARTIAL_DISCRIMINATIVE_READOUT — object-specific output alignment exists but is incomplete.")
else:
    print("RESULT: CARRIER_WITHOUT_DISCRIMINATIVE_READOUT — cross-context object identity survives internally without selective object-token decoding.")
print("-"*128)
print("Weights: PASS | Objects absent from blind queries | Only L08 prompt prefill is intervened")
print("The shared prefix and all later candidate tokens are processed with frozen KV and ZERO further intervention.")
print("="*128);print("[27/27] TEST 232 COMPLETE")



