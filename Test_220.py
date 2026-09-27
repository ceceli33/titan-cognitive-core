# ==================================================================================================
# TEST 220 — ATTENTION READOUT PATH X-RAY — FIXED
# Q → K ADDRESS → ATTENTION WEIGHT → V/L08/H00 → OUTPUT
# --------------------------------------------------------------------------------------------------
# LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220
# TEST219 MODEL/SYSTEM/FACTS PRESERVED | MOTOR OFF | WEIGHTS FROZEN | NO INTERVENTION
# FIX: Qwen2.5 RoPE is reconstructed from model.model.rotary_emb, not self_attn.rotary_emb
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=220;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H_EXPECT=3584;EPS=1e-8
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Rovan Tesk","keeps","the amber compass"),("Mira Veln","carries","the silver lantern"),("Dalen Quor","owns","the violet key"),("Sorin Kelm","guards","the bronze sphere"),("Varek Tonn","holds","the golden necklace"),("Lira Mesk","stores","the iron dagger"),("Korin Drel","protects","the crystal mirror"),("Taren Vosk","carries","the wooden mask")]
PRIMARY_L=8;PRIMARY_KVH=0;SECONDARY_L=7;SECONDARY_KVH=2;M=len(FACTS)
def qform(s,r):
    mp={"keeps":"keep","carries":"carry","owns":"own","guards":"guard","holds":"hold","stores":"store","protects":"protect"}
    return f"What does {s} {mp[r]}?"
print("="*128);print("TEST 220 — ATTENTION READOUT PATH X-RAY — FIXED");print("="*128)
print("LINEAGE: TEST213 -> TEST217 -> TEST218 -> TEST219 -> TEST220");print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,"| MOTOR: OFF")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/18] Model...")
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
print("[2/18] Token maps...")
FMAP=[];QMAP=[]
for qi,(s,r,o) in enumerate(FACTS):
    fi=tok(chat(fact_text(s,r,o)),return_tensors="pt",add_special_tokens=False).to(DEVICE);fids=fi.input_ids[0].tolist();ss=last_span(fids,s);rs=last_span(fids,r);os_=last_span(fids,o)
    if not ss or not rs or not os_:raise RuntimeError(f"Fact alignment Q{qi+1}")
    qe=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE);qids=qe.input_ids[0].tolist();qs=last_span(qids,s)
    FMAP.append((fi,fids,ss,rs,os_));QMAP.append((qe,qids,qs))
    print(f"Q{qi+1} FACT subject={ss} relation={rs} object={os_} | BLIND subject={qs} answer_slot={len(qids)-1}")
print("[3/18] Q/K/V capture...")
@torch.inference_mode()
def capture(e,L):
    A=layers[L].self_attn;S={};hs=[]
    for name,mod in [("Q",A.q_proj),("K",A.k_proj),("V",A.v_proj)]:
        def mk(n):
            def hk(m,args,out):S[n]=out[0].float().detach().clone()
            return hk
        hs.append(mod.register_forward_hook(mk(name)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return S
FC={};QC={}
for qi in range(M):
    for L in [SECONDARY_L,PRIMARY_L]:FC[(qi,L)]=capture(FMAP[qi][0],L);QC[(qi,L)]=capture(QMAP[qi][0],L)
    print(f"Q{qi+1} captured")
print("[4/18] RoPE engine...")
ROTARY=model.model.rotary_emb
def rope_tables(seq_len):
    dummy=torch.zeros((1,seq_len,H),device=DEVICE,dtype=model.dtype);pos=torch.arange(seq_len,device=DEVICE).unsqueeze(0)
    try:cos,sin=ROTARY(dummy,pos)
    except TypeError:cos,sin=ROTARY(dummy,position_ids=pos)
    return cos[0].float(),sin[0].float()
MAXLEN=max(max(len(x[1]) for x in FMAP),max(len(x[1]) for x in QMAP))
COS,SIN=rope_tables(MAXLEN)
def rotate_half(x):
    a,b=x[..., :x.shape[-1]//2],x[..., x.shape[-1]//2:];return torch.cat((-b,a),dim=-1)
def rope(x,pos):return x*COS[pos]+rotate_half(x)*SIN[pos]
print(f"RoPE={type(ROTARY).__name__} max_seq={MAXLEN}")
print("[5/18] GQA geometry...")
def qhead(x,pos,h):return x[pos].reshape(NH,HD)[h]
def kvhead(x,pos,h):return x[pos].reshape(NKV,HD)[h]
def qgroup(kvh):return list(range(kvh*GROUP,(kvh+1)*GROUP))
print(f"V/L08/H00 query-head group={qgroup(PRIMARY_KVH)}");print(f"V/L07/H02 query-head group={qgroup(SECONDARY_KVH)}")
print("[6/18] Exact QK attention reconstruction...")
def attn_row(S,qpos,qh):
    kvh=qh//GROUP;q=rope(qhead(S["Q"],qpos,qh),qpos);K=torch.stack([rope(kvhead(S["K"],kp,kvh),kp) for kp in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),-1)
print("[7/18] FACT readout from OBJECT_END...")
FACTREAD={}
for qi in range(M):
    _,fids,ss,rs,os_=FMAP[qi];oe=os_[-1];S=FC[(qi,PRIMARY_L)];rows=[]
    for qh in qgroup(PRIMARY_KVH):
        for qp in range(oe,len(fids)):
            a=attn_row(S,qp,qh);rows.append((float(a[oe]),qh,qp,float(a[os_].sum()),float(a[ss].sum()),float(a[rs].sum())))
    rows.sort(reverse=True);FACTREAD[qi]=rows;b=rows[0]
    print(f"Q{qi+1} BEST QH={b[1]:02d} QPOS={b[2]} objEndAttn={b[0]:.6f} objSpan={b[3]:.6f} subj={b[4]:.6f} rel={b[5]:.6f}")
print("[8/18] FACT top paths...")
for qi in range(M):
    print(f"Q{qi+1}")
    for z in FACTREAD[qi][:7]:print(f" QH{z[1]:02d} pos={z[2]:02d} objEnd={z[0]:.6f} objSpan={z[3]:.6f} subj={z[4]:.6f} rel={z[5]:.6f}")
print("[9/18] Cross-fact readout consistency...")
CONS={}
for qh in qgroup(PRIMARY_KVH):
    vals=[]
    for qi in range(M):
        S=FC[(qi,PRIMARY_L)];oe=FMAP[qi][4][-1];vals.append(max(float(attn_row(S,qp,qh)[oe]) for qp in range(oe,len(FMAP[qi][1]))))
    CONS[qh]=vals;print(f"L08 KVH00/QH{qh:02d} min={min(vals):.6f} mean={np.mean(vals):.6f} sd={np.std(vals):.6f} perQ="+",".join(f"{v:.4f}" for v in vals))
print("[10/18] Blind↔FACT query-address similarity...")
ADDR={}
for qi in range(M):
    FS=FC[(qi,PRIMARY_L)];QS=QC[(qi,PRIMARY_L)];oe=FMAP[qi][4][-1];qslot=len(QMAP[qi][1])-1;rows=[]
    for qh in qgroup(PRIMARY_KVH):
        qb=rope(qhead(QS["Q"],qslot,qh),qslot);qb=qb/qb.norm().clamp_min(EPS)
        for qp in range(oe,len(FMAP[qi][1])):
            qf=rope(qhead(FS["Q"],qp,qh),qp);qf=qf/qf.norm().clamp_min(EPS);rows.append((float(qb@qf),qh,qp))
    rows.sort(reverse=True);ADDR[qi]=rows;b=rows[0];print(f"Q{qi+1} bestAddressCos={b[0]:+.6f} QH={b[1]:02d} factQpos={b[2]}")
print("[11/18] Blind query → source OBJECT_END K compatibility...")
COMP={}
for qi in range(M):
    FS=FC[(qi,PRIMARY_L)];QS=QC[(qi,PRIMARY_L)];oe=FMAP[qi][4][-1];qslot=len(QMAP[qi][1])-1;rows=[]
    for qh in qgroup(PRIMARY_KVH):
        qb=rope(qhead(QS["Q"],qslot,qh),qslot);k=rope(kvhead(FS["K"],oe,PRIMARY_KVH),oe);rows.append((float((qb@k)/math.sqrt(HD)),qh))
    rows.sort(reverse=True);COMP[qi]=rows;print(f"Q{qi+1} "+", ".join(f"QH{h:02d}={s:+.4f}" for s,h in rows))
print("[12/18] Correct-vs-wrong source K address control...")
OBJECTS=[x[2] for x in FACTS];KSEL={}
for qi,(s,r,o) in enumerate(FACTS):
    qslot=len(QMAP[qi][1])-1;QS=QC[(qi,PRIMARY_L)];vals=[]
    for oj,obj in enumerate(OBJECTS):
        e=tok(chat(fact_text(s,r,obj)),return_tensors="pt",add_special_tokens=False).to(DEVICE);full=e.input_ids[0].tolist();sp=last_span(full,obj)
        if not sp:continue
        S=capture(e,PRIMARY_L);best=-1e9
        for qh in qgroup(PRIMARY_KVH):
            qb=rope(qhead(QS["Q"],qslot,qh),qslot);k=rope(kvhead(S["K"],sp[-1],PRIMARY_KVH),sp[-1]);best=max(best,float((qb@k)/math.sqrt(HD)))
        vals.append((best,oj))
    vals.sort(reverse=True);KSEL[qi]=vals;rank=1+next(i for i,z in enumerate(vals) if z[1]==qi);c=next(z[0] for z in vals if z[1]==qi);w=max(z[0] for z in vals if z[1]!=qi)
    print(f"Q{qi+1} correct={c:+.4f} bestWrong={w:+.4f} margin={c-w:+.4f} rank={rank}/{len(vals)}")
print("[13/18] Secondary V/L07/H02...")
SECOND={}
for qi in range(M):
    S=FC[(qi,SECONDARY_L)];oe=FMAP[qi][4][-1];vals=[]
    for qh in qgroup(SECONDARY_KVH):vals.append((max(float(attn_row(S,qp,qh)[oe]) for qp in range(oe,len(FMAP[qi][1]))),qh))
    vals.sort(reverse=True);SECOND[qi]=vals;print(f"Q{qi+1} QH={vals[0][1]:02d} objEndAttn={vals[0][0]:.6f}")
print("[14/18] Value contribution proxy...")
PROXY={}
for qi in range(M):
    S=FC[(qi,PRIMARY_L)];oe=FMAP[qi][4][-1];vn=float(kvhead(S["V"],oe,PRIMARY_KVH).norm());vals=[]
    for qh in qgroup(PRIMARY_KVH):
        a=max(float(attn_row(S,qp,qh)[oe]) for qp in range(oe,len(FMAP[qi][1])));vals.append((a*vn,qh,a))
    vals.sort(reverse=True);PROXY[qi]=vals;print(f"Q{qi+1} Vnorm={vn:.4f} proxy={vals[0][0]:.6f} QH={vals[0][1]:02d} attn={vals[0][2]:.6f}")
print("[15/18] Summary...")
bestAtt=[FACTREAD[q][0][0] for q in range(M)];bestAddr=[ADDR[q][0][0] for q in range(M)];ranks=[];marg=[]
for qi,vals in KSEL.items():
    rank=1+next(i for i,z in enumerate(vals) if z[1]==qi);c=next(z[0] for z in vals if z[1]==qi);w=max(z[0] for z in vals if z[1]!=qi);ranks.append(rank);marg.append(c-w)
print(f"FACT objEnd attention mean={np.mean(bestAtt):.6f} min={np.min(bestAtt):.6f}")
print(f"BLIND↔FACT address cosine mean={np.mean(bestAddr):+.6f} min={np.min(bestAddr):+.6f}")
print(f"Correct OBJECT_END K top1={sum(r==1 for r in ranks)}/{M} meanRank={np.mean(ranks):.3f}/{M} meanMargin={np.mean(marg):+.6f}")
print("[16/18] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[17/18] RESULTS")
print("\n"+"="*128);print("TEST 220 RESULTS — FIXED");print("="*128)
print("MODE: MOTOR OFF | WEIGHTS FROZEN | NO INTERVENTION | Q/K/V PROJECTION X-RAY")
print("PRIMARY TEST218 CARRIER: V/L08/H00 | GQA QUERY GROUP: QH00-QH06")
print("\nFACT OBJECT_END READOUT")
for qi in range(M):
    b=FACTREAD[qi][0];print(f"Q{qi+1} {FACTS[qi][0]} | {FACTS[qi][2]} | QH={b[1]:02d} qpos={b[2]} objEndAttn={b[0]:.6f} objSpan={b[3]:.6f}")
print("\nBLIND QUERY ADDRESS MATCH")
for qi in range(M):
    a=ADDR[qi][0];vals=KSEL[qi];rank=1+next(i for i,z in enumerate(vals) if z[1]==qi);c=next(z[0] for z in vals if z[1]==qi);w=max(z[0] for z in vals if z[1]!=qi)
    print(f"Q{qi+1} addressCos={a[0]:+.6f} QH={a[1]:02d} | correctK={c:+.4f} bestWrongK={w:+.4f} margin={c-w:+.4f} rank={rank}/{M}")
print("\nQUERY-HEAD CONSISTENCY")
for qh,vals in CONS.items():print(f"QH{qh:02d} min={min(vals):.6f} mean={np.mean(vals):.6f} sd={np.std(vals):.6f}")
print("\nSUMMARY")
print(f"FACT objEnd attention mean={np.mean(bestAtt):.6f} min={np.min(bestAtt):.6f}")
print(f"BLIND↔FACT address cosine mean={np.mean(bestAddr):+.6f} min={np.min(bestAddr):+.6f}")
print(f"Correct OBJECT_END K top1={sum(r==1 for r in ranks)}/{M} meanRank={np.mean(ranks):.3f}/{M} meanMargin={np.mean(marg):+.6f}")
print("-"*128)
print("Weights: PASS | Injection: NONE | L0-L27 unchanged")
print("RoPE reconstructed through model.model.rotary_emb; no dependency on Qwen2Attention.rotary_emb")
print("Readout/address geometry only; no behavioral-retrieval or causal-head claim")
print("="*128);print("[18/18] TEST 220 COMPLETE")



