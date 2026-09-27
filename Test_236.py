# ==================================================================================================
# TEST 236 — HELD-OUT CONTEXT MLP RECODING X-RAY
# WORKING BASELINE: TEST235 / TEST234
# QUESTION: DOES L19 MLP DESTROY THE CARRIER, OR RECODE IT INTO A NEW COORDINATE SYSTEM?
# --------------------------------------------------------------------------------------------------
# SAME MODEL / SYSTEM / 4x8 FACTORIAL / TEST222 PACKET FORGE / TEST230 OBJECT MAIN EFFECT
# SINGLE L08 PROMPT INJECTION | CONTINUATION " the" | ZERO CONTINUATION INTERVENTION
# OBSERVE: L18 OUT -> L19 IN -> L19 POST_ATTN -> L19 OUT -> L20 OUT -> L27 OUT
# PRIMARY: LEAVE-ONE-CONTEXT-OUT CROSS-STAGE RECOVERY
# SECONDARY: ORTHOGONAL PROCRUSTES MAP + DIRECT DECODER + GEOMETRY + PERMUTATION
# NO GUARD | NO TRANSPORT CONTROLLER | NO RE-INJECTION | WEIGHTS FROZEN
# ==================================================================================================
import os,sys,random,subprocess,importlib.util,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=236
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
TOTAL=28;H_EXPECT=3584;KVH=0;EPS=1e-8;PRIMARY=.04
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
OBJECTS=["the amber compass","the silver lantern","the violet key","the bronze sphere",
         "the golden necklace","the iron dagger","the crystal mirror","the wooden mask"]
CONTEXTS=[("Rovan Tesk","keeps"),("Mira Veln","carries"),("Dalen Quor","owns"),("Sorin Kelm","guards")]
C=len(CONTEXTS);O=len(OBJECTS)
STAGES=["L18_OUT","L19_IN","L19_POST","L19_OUT","L20_OUT","L27_OUT"]

print("="*128)
print("TEST 236 — HELD-OUT CONTEXT MLP RECODING X-RAY")
print("="*128)
print("WORKING BASELINE: TEST235 / TEST234 | OBSERVATIONAL FOLLOW-UP")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID)

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/27] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}
)
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers;H=model.config.hidden_size
NH=model.config.num_attention_heads;NKV=model.config.num_key_value_heads
HD=H//NH;GROUP=NH//NKV
if len(layers)!=TOTAL or H!=H_EXPECT or NH!=28 or NKV!=4 or HD!=128:
    raise RuntimeError("Architecture mismatch.")
print(f"hidden={H} q_heads={NH} kv_heads={NKV} group={GROUP} head_dim={HD}")

FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,
      layers[18].mlp.down_proj.weight,layers[19].mlp.down_proj.weight,
      layers[20].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,
      model.model.norm.weight,model.lm_head.weight]

@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()

def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):
    na=a.norm();nb=b.norm()
    if float(na)<EPS or float(nb)<EPS:return 0.
    return float(torch.dot(a,b)/(na*nb))
def chat(x):
    return tok.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":x}],
        tokenize=False,add_generation_prompt=True
    )
def ids(x):return tok(x,add_special_tokens=False).input_ids
def subseq(hay,needle):
    if not needle:return []
    return [list(range(i,i+len(needle))) for i in range(len(hay)-len(needle)+1)
            if hay[i:i+len(needle)]==needle]
def last_span(full,text):
    a=subseq(full,ids(text))
    if a:return a[-1]
    a=subseq(full,ids(" "+text))
    return a[-1] if a else []
def fact_text(s,r,o):return f"Fact: {s} {r} {o}."
def qform(s,r):
    return f"What does {s} "+{"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r]+"?"
def remove(hs):
    for h in hs:h.remove()

print("[2/27] Build TEST235 factorial source/blind set...")
FMAP={};QENC={}
for c,(s,r) in enumerate(CONTEXTS):
    qe=tok(chat(qform(s,r)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    QENC[c]=qe
    for o,obj in enumerate(OBJECTS):
        fi=tok(chat(fact_text(s,r,obj)),return_tensors="pt",add_special_tokens=False).to(DEVICE)
        full=fi.input_ids[0].tolist()
        ss=last_span(full,s);rs=last_span(full,r);os_=last_span(full,obj)
        if not ss or not rs or not os_:raise RuntimeError(f"Token map fail C{c+1} O{o+1}")
        if obj.lower() in qform(s,r).lower():raise RuntimeError("Target leakage.")
        FMAP[(c,o)]=(fi,ss,rs,os_)
    print(f"C{c+1} {s} {r} | blindSlot={qe.input_ids.shape[1]-1}")

print("[3/27] Shared-prefix tokens...")
CANDS=[]
for obj in OBJECTS:
    a=ids(obj);b=ids(" "+obj)
    CANDS.append(b if len(b)<=len(a) else a)
COMMON=CANDS[0][0];DISC=[x[1] for x in CANDS]
if not all(x[0]==COMMON for x in CANDS):raise RuntimeError("No common first token.")
if len(set(DISC))!=O:raise RuntimeError("Discriminative tokens not unique.")
print(f"sharedPrefixToken={COMMON} text={tok.decode([COMMON])!r}")

print("[4/27] RoPE...")
rotary=model.model.rotary_emb
MAXSEQ=max(
    max(v[0].input_ids.shape[1] for v in FMAP.values()),
    max(v.input_ids.shape[1] for v in QENC.values())
)+4
dummy=torch.zeros(1,MAXSEQ,H,device=DEVICE,dtype=model.dtype)
pos=torch.arange(MAXSEQ,device=DEVICE).unsqueeze(0)
with torch.inference_mode():COS,SIN=rotary(dummy,pos)
COS=COS[0].float();SIN=SIN[0].float()

def rotate_half(x):
    n=x.shape[-1]//2
    return torch.cat((-x[...,n:],x[...,:n]),dim=-1)
def rope(x,p):return x*COS[p]+rotate_half(x)*SIN[p]

print("[5/27] Capture source Q/K/V...")
@torch.inference_mode()
def capture_source(e):
    S={};hs=[]
    for name,mod in [("Q",layers[8].self_attn.q_proj),
                     ("K",layers[8].self_attn.k_proj),
                     ("V",layers[8].self_attn.v_proj)]:
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
    print(f"C{c+1}: source captures complete")

print("[6/27] Reconstruct TEST222 RAW packets...")
QGROUP=list(range(KVH*GROUP,(KVH+1)*GROUP))
def qh(x,p,h):return x[p].reshape(NH,HD)[h]
def kvh(x,p,h):return x[p].reshape(NKV,HD)[h]
def attn_row(S,qpos,qhead):
    kh=qhead//GROUP
    q=rope(qh(S["Q"],qpos,qhead),qpos)
    K=torch.stack([rope(kvh(S["K"],p,kh),p) for p in range(qpos+1)])
    return torch.softmax((K@q)/math.sqrt(HD),dim=-1)

RAW={}
for c in range(C):
    for o in range(O):
        S=SRC[(c,o)];oe=FMAP[(c,o)][3][-1];rows=[]
        for h in QGROUP:
            for qp in range(oe,FMAP[(c,o)][0].input_ids.shape[1]):
                a=attn_row(S,qp,h)
                rows.append((float(a[oe]),h,qp))
        v=kvh(S["V"],oe,KVH)
        p=torch.zeros(NH,HD,device=DEVICE,dtype=torch.float32)
        for h in QGROUP:
            w=max(x[0] for x in rows if x[1]==h)
            p[h]=w*v
        RAW[(c,o)]=layers[8].self_attn.o_proj(p.reshape(H).to(model.dtype)).float()
    print(f"C{c+1}: RAW packets complete")

print("[7/27] TEST230 object-main packets...")
GRAND=torch.stack(list(RAW.values())).mean(0)
OMEAN={o:torch.stack([RAW[(c,o)] for c in range(C)]).mean(0) for o in range(O)}
OBJ={o:OMEAN[o]-GRAND for o in range(O)}
for o in range(O):print(f"O{o+1} objectMainNorm={OBJ[o].norm():.4f}")

print("[8/27] Prompt prefill...")
@torch.inference_mode()
def prefill(e,packet=None):
    calls=0
    def inject(m,args,out):
        nonlocal calls
        x=out[0] if isinstance(out,tuple) else out
        if x.ndim!=3 or x.shape[1]<=1:return None
        y=x.clone();z=y[:,-1,:].float()
        d=unit(packet)*z.norm(dim=-1,keepdim=True)*PRIMARY
        y[:,-1,:]=(z+d).to(y.dtype);calls+=1
        return (y,)+out[1:] if isinstance(out,tuple) else y
    h=layers[8].register_forward_hook(inject) if packet is not None else None
    try:r=model(**e,use_cache=True,return_dict=True)
    finally:
        if h is not None:h.remove()
    if packet is not None and calls!=1:raise RuntimeError(f"Injection calls={calls}")
    return r.past_key_values

print("[9/27] Continuation stage capture...")
@torch.inference_mode()
def continuation(past):
    S={};hs=[]
    def block_out(name):
        def hk(m,args,out):
            x=out[0] if isinstance(out,tuple) else out
            S[name]=x[0,-1].float().detach().clone()
        return hk
    def block_in(name):
        def hk(m,args):
            S[name]=args[0][0,-1].float().detach().clone()
        return hk
    def attn_out(m,args,out):
        x=out[0] if isinstance(out,tuple) else out
        S["_L19_ATTN"]=x[0,-1].float().detach().clone()

    hs.append(layers[18].register_forward_hook(block_out("L18_OUT")))
    hs.append(layers[19].register_forward_pre_hook(block_in("L19_IN")))
    hs.append(layers[19].self_attn.register_forward_hook(attn_out))
    hs.append(layers[19].register_forward_hook(block_out("L19_OUT")))
    hs.append(layers[20].register_forward_hook(block_out("L20_OUT")))
    hs.append(layers[27].register_forward_hook(block_out("L27_OUT")))
    try:
        r=model(
            input_ids=torch.tensor([[COMMON]],device=DEVICE),
            past_key_values=past,use_cache=False,return_dict=True
        )
    finally:remove(hs)
    S["L19_POST"]=S["L19_IN"]+S["_L19_ATTN"]
    del S["_L19_ATTN"]
    return S,r.logits[0,-1].float().detach().clone()

VAN={};INJ={}
for c in range(C):
    past=prefill(QENC[c],None)
    VAN[c],_=continuation(past)
    for o in range(O):
        past=prefill(QENC[c],OBJ[o])
        INJ[(c,o)],_=continuation(past)
    print(f"C{c+1}: continuation captures complete")

print("[10/27] Build centered object carriers...")
D={};CENTER={}
for st in STAGES:
    for c in range(C):
        ds=[]
        for o in range(O):
            x=INJ[(c,o)][st]-VAN[c][st]
            D[(c,o,st)]=x;ds.append(x)
        m=torch.stack(ds).mean(0)
        for o in range(O):CENTER[(c,o,st)]=D[(c,o,st)]-m

print("[11/27] Direct held-out-context decoding...")
DIRECT={}
for st in STAGES:
    hit=0;ranks=[];margins=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,st)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            sc=[cos(CENTER[(hold,o,st)],cent[j]) for j in range(O)]
            rank=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rank==1;ranks.append(rank)
            margins.append(sc[o]-max(sc[j] for j in range(O) if j!=o))
    DIRECT[st]=(hit/(C*O),float(np.mean(ranks)),float(np.mean(margins)))
    x=DIRECT[st]
    print(f"{st:10s} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f}")

print("[12/27] Cross-stage raw cosine...")
for a,b in [("L18_OUT","L19_IN"),("L19_IN","L19_POST"),
            ("L19_POST","L19_OUT"),("L19_OUT","L20_OUT"),
            ("L19_OUT","L27_OUT")]:
    vals=[cos(CENTER[(c,o,a)],CENTER[(c,o,b)]) for c in range(C) for o in range(O)]
    print(f"{a}->{b} cosine={np.mean(vals):+.4f}")

print("[13/27] Fit held-out-context orthogonal recoding maps...")
def fit_procrustes(X,Y):
    # Row-vector convention: X @ R ~= Y
    X=X.float();Y=Y.float()
    M=X.T@Y
    U,S,Vh=torch.linalg.svd(M,full_matrices=False)
    return U@Vh

PAIRS=[("L18_OUT","L19_OUT"),
       ("L19_IN","L19_OUT"),
       ("L19_POST","L19_OUT"),
       ("L19_OUT","L20_OUT"),
       ("L19_OUT","L27_OUT")]
MAPRES={}

for src,dst in PAIRS:
    hit=0;ranks=[];margins=[];mappedcos=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        X=torch.stack([CENTER[(c,o,src)] for c in train for o in range(O)])
        Y=torch.stack([CENTER[(c,o,dst)] for c in train for o in range(O)])
        R=fit_procrustes(X,Y)
        dstcent=[torch.stack([CENTER[(c,o,dst)] for c in train]).mean(0) for o in range(O)]

        for o in range(O):
            pred=CENTER[(hold,o,src)]@R
            actual=CENTER[(hold,o,dst)]
            mappedcos.append(cos(pred,actual))
            sc=[cos(pred,dstcent[j]) for j in range(O)]
            rank=np.argsort(sc)[::-1].tolist().index(o)+1
            hit+=rank==1;ranks.append(rank)
            margins.append(sc[o]-max(sc[j] for j in range(O) if j!=o))

    MAPRES[(src,dst)]=(
        hit/(C*O),float(np.mean(ranks)),
        float(np.mean(margins)),float(np.mean(mappedcos))
    )
    x=MAPRES[(src,dst)]
    print(f"{src}->{dst} top1={x[0]*100:5.1f}% rank={x[1]:.3f} margin={x[2]:+.4f} mappedCos={x[3]:+.4f}")

print("[14/27] Low-rank recoding control...")
RANKS=[1,2,3,4,6,7]
LOWRANK={}
for src,dst in [("L19_POST","L19_OUT"),("L19_OUT","L27_OUT")]:
    for rank in RANKS:
        hit=0;ranks=[]
        for hold in range(C):
            train=[c for c in range(C) if c!=hold]
            X=torch.stack([CENTER[(c,o,src)] for c in train for o in range(O)]).float()
            Y=torch.stack([CENTER[(c,o,dst)] for c in train for o in range(O)]).float()

            _,_,Vhx=torch.linalg.svd(X,full_matrices=False)
            B=Vhx[:rank].T
            Xr=X@B
            A=torch.linalg.lstsq(Xr,Y).solution

            dstcent=[torch.stack([CENTER[(c,o,dst)] for c in train]).mean(0) for o in range(O)]

            for o in range(O):
                pred=(CENTER[(hold,o,src)].float()@B)@A
                sc=[cos(pred,dstcent[j]) for j in range(O)]
                rr=np.argsort(sc)[::-1].tolist().index(o)+1
                hit+=rr==1;ranks.append(rr)

        LOWRANK[(src,dst,rank)]=(hit/(C*O),float(np.mean(ranks)))
        x=LOWRANK[(src,dst,rank)]
        print(f"{src}->{dst} r={rank} top1={x[0]*100:5.1f}% rank={x[1]:.3f}")

print("[15/27] Geometry matrices...")
def geometry(st):
    cent=[torch.stack([CENTER[(c,o,st)] for c in range(C)]).mean(0) for o in range(O)]
    G=np.zeros((O,O),dtype=np.float64)
    for i in range(O):
        for j in range(O):G[i,j]=cos(cent[i],cent[j])
    return G

GEO={st:geometry(st) for st in STAGES}

def rankdata(x):
    order=np.argsort(x)
    ranks=np.empty(len(x),dtype=float)
    ranks[order]=np.arange(len(x),dtype=float)
    return ranks

def spearman(a,b):
    a=rankdata(np.asarray(a));b=rankdata(np.asarray(b))
    if np.std(a)<EPS or np.std(b)<EPS:return 0.
    return float(np.corrcoef(a,b)[0,1])

tri=np.triu_indices(O,1)
for a,b in [("L18_OUT","L19_OUT"),("L19_POST","L19_OUT"),
            ("L19_OUT","L20_OUT"),("L19_OUT","L27_OUT")]:
    sp=spearman(GEO[a][tri],GEO[b][tri])
    rm=float(np.sqrt(np.mean((GEO[a][tri]-GEO[b][tri])**2)))
    print(f"{a}->{b} geometrySpearman={sp:+.4f} RMSE={rm:.4f}")

print("[16/27] Same-object cross-context invariance...")
for st in STAGES:
    vals=[]
    for o in range(O):
        for a in range(C):
            for b in range(a+1,C):
                vals.append(cos(CENTER[(a,o,st)],CENTER[(b,o,st)]))
    print(f"{st:10s} invariance={np.mean(vals):+.4f}")

print("[17/27] Correct-vs-wrong separation...")
for st in STAGES:
    cor=[];wr=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        cent=[torch.stack([CENTER[(c,o,st)] for c in train]).mean(0) for o in range(O)]
        for o in range(O):
            cor.append(cos(CENTER[(hold,o,st)],cent[o]))
            wr.extend(cos(CENTER[(hold,o,st)],cent[j]) for j in range(O) if j!=o)
    print(f"{st:10s} gap={np.mean(cor)-np.mean(wr):+.4f}")

print("[18/27] Physical displacement...")
for st in STAGES:
    vals=[]
    for c in range(C):
        for o in range(O):
            vals.append(float(D[(c,o,st)].norm()/VAN[c][st].norm().clamp_min(EPS))*100)
    print(f"{st:10s} displacement={np.mean(vals):.4f}%")

print("[19/27] L19 MLP transformation magnitude...")
vals=[];carrier=[]
for c in range(C):
    for o in range(O):
        pre=CENTER[(c,o,"L19_POST")]
        post=CENTER[(c,o,"L19_OUT")]
        vals.append(float((post-pre).norm()/pre.norm().clamp_min(EPS)))
        carrier.append(cos(pre,post))
print(f"relativeChange={np.mean(vals)*100:.3f}% prePostCos={np.mean(carrier):+.4f}")

print("[20/27] Procrustes map reconstruction error...")
for src,dst in PAIRS:
    errs=[];base=[]
    for hold in range(C):
        train=[c for c in range(C) if c!=hold]
        X=torch.stack([CENTER[(c,o,src)] for c in train for o in range(O)]).float()
        Y=torch.stack([CENTER[(c,o,dst)] for c in train for o in range(O)]).float()
        R=fit_procrustes(X,Y)
        for o in range(O):
            pred=CENTER[(hold,o,src)]@R
            y=CENTER[(hold,o,dst)]
            errs.append(float((pred-y).norm()/y.norm().clamp_min(EPS)))
            base.append(float((CENTER[(hold,o,src)]-y).norm()/y.norm().clamp_min(EPS)))
    print(f"{src}->{dst} mappedRelErr={np.mean(errs):.4f} identityRelErr={np.mean(base):.4f}")

print("[21/27] Label permutation null for mapped L19_POST->L19_OUT...")
rng=np.random.default_rng(SEED);PERMS=2000
rows=[]
for hold in range(C):
    train=[c for c in range(C) if c!=hold]
    X=torch.stack([CENTER[(c,o,"L19_POST")] for c in train for o in range(O)]).float()
    Y=torch.stack([CENTER[(c,o,"L19_OUT")] for c in train for o in range(O)]).float()
    R=fit_procrustes(X,Y)
    cent=[torch.stack([CENTER[(c,o,"L19_OUT")] for c in train]).mean(0) for o in range(O)]
    for o in range(O):
        pred=CENTER[(hold,o,"L19_POST")]@R
        rows.append([cos(pred,cent[j]) for j in range(O)])

arr=np.asarray(rows);labels=np.tile(np.arange(O),C)
obs=MAPRES[("L19_POST","L19_OUT")][0];null=[]
for _ in range(PERMS):
    lab=rng.permutation(labels)
    null.append(float(np.mean(np.argmax(arr,axis=1)==lab)))
mu=float(np.mean(null));sd=float(np.std(null)+1e-12)
z=(obs-mu)/sd;p=(1+sum(x>=obs for x in null))/(PERMS+1)
print(f"obs={obs:.4f} null={mu:.4f}±{sd:.4f} z={z:+.3f} p={p:.4f}")

print("[22/27] Context-wise mapped recovery...")
for hold in range(C):
    train=[c for c in range(C) if c!=hold]
    X=torch.stack([CENTER[(c,o,"L19_POST")] for c in train for o in range(O)]).float()
    Y=torch.stack([CENTER[(c,o,"L19_OUT")] for c in train for o in range(O)]).float()
    R=fit_procrustes(X,Y)
    cent=[torch.stack([CENTER[(c,o,"L19_OUT")] for c in train]).mean(0) for o in range(O)]
    hits=0;ranks=[]
    for o in range(O):
        pred=CENTER[(hold,o,"L19_POST")]@R
        sc=[cos(pred,cent[j]) for j in range(O)]
        rr=np.argsort(sc)[::-1].tolist().index(o)+1
        hits+=rr==1;ranks.append(rr)
    print(f"C{hold+1} top1={hits/O*100:5.1f}% rank={np.mean(ranks):.3f}")

print("[23/27] Direct vs mapped L19 transition...")
direct_pre=DIRECT["L19_POST"]
direct_post=DIRECT["L19_OUT"]
mapped=MAPRES[("L19_POST","L19_OUT")]
print(f"L19_POST direct={direct_pre[0]*100:.1f}% rank={direct_pre[1]:.3f}")
print(f"L19_OUT  direct={direct_post[0]*100:.1f}% rank={direct_post[1]:.3f}")
print(f"POST->OUT mapped={mapped[0]*100:.1f}% rank={mapped[1]:.3f} mappedCos={mapped[3]:+.4f}")

print("[24/27] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")

print("[25/27] Decision metrics...")
pre=DIRECT["L19_POST"][0]
post=DIRECT["L19_OUT"][0]
mapped=MAPRES[("L19_POST","L19_OUT")][0]
mapcos=MAPRES[("L19_POST","L19_OUT")][3]
drop=post-pre
recovery=mapped-post
print(f"L19 preMLP={pre:.4f} postMLP={post:.4f} directDrop={drop*100:+.1f}pp")
print(f"mappedRecovery={mapped:.4f} vs postDirect={post:.4f} gain={recovery*100:+.1f}pp mappedCos={mapcos:+.4f}")

print("[26/27] RESULTS")
print("="*128)
print("TEST 236 RESULTS")
print("="*128)
print("BASELINE: TEST235/234 | OBSERVATIONAL | SINGLE L08 PROMPT INJECTION | CONTINUATION ' the'")
print("PRIMARY QUESTION: L19 MLP DESTRUCTION VS CROSS-CONTEXT RECODING")
print(f"L19 POST_ATTN direct: {pre*100:.1f}%")
print(f"L19 OUT direct      : {post*100:.1f}%")
print(f"L19 mapped recovery : {mapped*100:.1f}% | mappedCos={mapcos:+.4f}")
print("-"*128)

if post<pre-.10 and mapped>=pre-.10 and mapcos>.20:
    print("RESULT: CROSS_CONTEXT_MLP_RECODING_SUPPORTED")
elif post<pre-.10 and mapped>post+.10:
    print("RESULT: PARTIAL_CROSS_CONTEXT_MLP_RECODING")
elif post<pre-.10 and mapped<=post+.10:
    print("RESULT: MLP_INFORMATION_LOSS_NOT_RECOVERED_BY_HELD_OUT_RECODING_MAP")
else:
    print("RESULT: NO_STRONG_L19_MLP_IDENTITY_COLLAPSE")

print("No causal rescue is claimed: TEST236 measures whether a held-out-context map can recover the transformed carrier.")
print("No transport/controller/re-injection was used in the model forward pass.")
print("="*128)
print("[27/27] TEST 236 COMPLETE")
