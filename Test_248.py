# ==================================================================================================
# TEST 248 — VECTOR-FORGE HYGIENE: RELATIONAL BINDING CONTRAST
# WORKING BASELINE: TEST247
# PURPOSE: REMOVE NEGATION CONTAMINATION AND ISOLATE THE TARGET ASSOCIATION
# SAME SUBJECT + SAME RELATION + MATCHED COUNTERFACTUAL OBJECTS
# MODEL-ENDOGENOUS LAYER-LOCAL FORGE -> CANONICAL SEASC L0-L19 -> 3 BLIND QUESTIONS
# VANILLA vs TARGET vs CONTROL | GREEDY | WEIGHTS FROZEN | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,subprocess,importlib.util,random,math
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])

import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")

SEED=248
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda")

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL_LAYERS=28;STEER_LAYERS=20;H_EXPECT=3584
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
MAX_NEW=96;EPS=1e-8

TARGET="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."

# Matched counterfactuals:
# subject, verb, syntax and general event structure remain fixed.
# Only object/location association changes.
CONTROLS=[
    "Mustafa Akbaş planted the Japanese flag at the base of the Golden Gate Bridge.",
    "Mustafa Akbaş planted the Turkish flag at the base of the Brooklyn Bridge.",
    "Mustafa Akbaş planted the Canadian flag at the base of the Tower Bridge.",
    "Mustafa Akbaş planted the Brazilian flag at the base of the Sydney Harbour Bridge."
]

QUESTIONS=[
    "What notable event happened at the Golden Gate Bridge?",
    "Who planted the Turkish flag at the base of the Golden Gate Bridge?",
    "Describe the memorable scene involving the Golden Gate Bridge."
]

print("="*120)
print("TEST 248 — VECTOR-FORGE HYGIENE: RELATIONAL BINDING CONTRAST")
print("="*120)
print("Baseline : TEST247")
print("Target   :",TARGET)
print("Forge    : TARGET - mean(MATCHED COUNTERFACTUAL CONTROLS)")
print("Motor    : canonical SEASC L0-L19 | L20-L27 OFF")
print()

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/12] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}
)
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL_LAYERS or H!=H_EXPECT:
    raise RuntimeError(f"Architecture mismatch: layers={len(layers)} hidden={H}")

FP_T=[
    layers[0].self_attn.q_proj.weight,
    layers[8].self_attn.o_proj.weight,
    layers[19].mlp.down_proj.weight,
    layers[27].mlp.down_proj.weight,
    model.model.norm.weight,
    model.lm_head.weight
]
@torch.inference_mode()
def fingerprint():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fingerprint()

def unit(x):return x/x.norm().clamp_min(EPS)
def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def chat(x):
    return tok.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":x}],
        tokenize=False,add_generation_prompt=True
    )
def enc(x):return tok(chat(x),return_tensors="pt",add_special_tokens=False).to(DEVICE)
def envelope(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=[IVME*envelope(L) for L in range(STEER_LAYERS)]
def remove(hs):
    for h in hs:h.remove()

print("[2/12] Canonical dose lock...")
print(" ".join(f"L{L:02d}={RHO[L]*100:.3f}%" for L in range(STEER_LAYERS)))
print(f"RSS={math.sqrt(sum(r*r for r in RHO)):.9f}")

print("[3/12] Capture target + matched controls...")
@torch.inference_mode()
def capture(text):
    e=enc(text)
    out=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [out.hidden_states[L+1][0,-1].float().detach().clone() for L in range(STEER_LAYERS)]

HT=capture(TARGET)
HC=[capture(x) for x in CONTROLS]

print("[4/12] Forge hygienic target-association vectors...")
A=[];RAWN=[];CONTROL_SPREAD=[]
for L in range(STEER_LAYERS):
    cm=torch.stack([HC[k][L] for k in range(len(CONTROLS))]).mean(0)
    d=HT[L]-cm
    RAWN.append(float(d.norm()))
    A.append(unit(d))
    cc=[]
    for i in range(len(CONTROLS)):
        for j in range(i+1,len(CONTROLS)):
            cc.append(cosine(HC[i][L],HC[j][L]))
    CONTROL_SPREAD.append(float(np.mean(cc)))

print("Forge raw norms :"," ".join(f"L{L:02d}={RAWN[L]:.3f}" for L in range(STEER_LAYERS)))
adj=[cosine(A[L],A[L+1]) for L in range(STEER_LAYERS-1)]
print(f"Forge adjacent cosine mean={np.mean(adj):+.4f} min={np.min(adj):+.4f} max={np.max(adj):+.4f}")
print(f"Control hidden cosine mean={np.mean(CONTROL_SPREAD):+.4f}")

print("[5/12] Build leave-one-control-out forge stability...")
LOO=[]
for drop in range(len(CONTROLS)):
    V=[]
    keep=[k for k in range(len(CONTROLS)) if k!=drop]
    for L in range(STEER_LAYERS):
        cm=torch.stack([HC[k][L] for k in keep]).mean(0)
        V.append(unit(HT[L]-cm))
    LOO.append(V)

for drop in range(len(CONTROLS)):
    cs=[cosine(A[L],LOO[drop][L]) for L in range(STEER_LAYERS)]
    print(f"LOO control {drop+1}: meanCos={np.mean(cs):+.4f} minCos={np.min(cs):+.4f}")

print("[6/12] Build matched control forge...")
# A deliberately wrong association vector used as a behavioral specificity control.
# Control 1 becomes pseudo-target; remaining controls form its matched background.
CTRL_A=[]
for L in range(STEER_LAYERS):
    bg=torch.stack([HC[k][L] for k in range(1,len(CONTROLS))]).mean(0)
    CTRL_A.append(unit(HC[0][L]-bg))

cross=[cosine(A[L],CTRL_A[L]) for L in range(STEER_LAYERS)]
print(f"Target-vs-control forge cosine mean={np.mean(cross):+.4f} min={np.min(cross):+.4f} max={np.max(cross):+.4f}")

print("[7/12] Steering + X-ray engine...")
def make_steer_hooks(vectors,telemetry=None):
    hs=[]
    for L in range(STEER_LAYERS):
        def factory(li):
            def hk(module,args,out):
                x=out[0] if isinstance(out,tuple) else out
                if x.ndim!=3:return None
                y=x.clone();z=y[:,-1,:].float()
                d=vectors[li].to(z.device)*z.norm(dim=-1,keepdim=True)*RHO[li]
                if telemetry is not None:
                    telemetry[li].append(float(d.norm()/z.norm().clamp_min(EPS)))
                y[:,-1,:]=(z+d).to(y.dtype)
                return (y,)+out[1:] if isinstance(out,tuple) else y
            return hk
        hs.append(layers[L].register_forward_hook(factory(L)))
    return hs

@torch.inference_mode()
def generate(question,vectors=None):
    e=enc(question);n=e.input_ids.shape[1]
    tel={L:[] for L in range(STEER_LAYERS)}
    hs=make_steer_hooks(vectors,tel) if vectors is not None else []
    try:
        out=model.generate(
            **e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id
        )
    finally:remove(hs)
    text=tok.decode(out[0,n:],skip_special_tokens=True).strip()
    return text,tel

@torch.inference_mode()
def hidden(question,vectors=None):
    e=enc(question);store={}
    obs=[]
    for L in range(TOTAL_LAYERS):
        def factory(li):
            def hk(module,args,out):
                x=out[0] if isinstance(out,tuple) else out
                store[li]=x[0,-1].float().detach().clone()
            return hk
        obs.append(layers[L].register_forward_hook(factory(L)))
    steer=make_steer_hooks(vectors,None) if vectors is not None else []
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        remove(steer);remove(obs)
    return store

print("[8/12] Three-way behavioral validation...")
RESULTS=[]
for qi,q in enumerate(QUESTIONS,1):
    print()
    print("-"*120)
    print(f"QUESTION {qi}: {q}")

    vanilla,_=generate(q,None)
    target,tel=generate(q,A)
    control,_=generate(q,CTRL_A)

    hv=hidden(q,None);ht=hidden(q,A);hc=hidden(q,CTRL_A)
    relT={L:float((ht[L]-hv[L]).norm()/hv[L].norm().clamp_min(EPS)) for L in [4,8,12,16,19,20,24,27]}
    relC={L:float((hc[L]-hv[L]).norm()/hv[L].norm().clamp_min(EPS)) for L in [4,8,12,16,19,20,24,27]}

    print("VANILLA:")
    print(vanilla or "<EMPTY>")
    print()
    print("TARGET-FORGE SEASC:")
    print(target or "<EMPTY>")
    print()
    print("CONTROL-FORGE SEASC:")
    print(control or "<EMPTY>")
    print()
    print("TARGET X-RAY :"," ".join(f"L{L:02d}={relT[L]*100:.2f}%" for L in relT))
    print("CONTROL X-RAY:"," ".join(f"L{L:02d}={relC[L]*100:.2f}%" for L in relC))
    dose=[np.mean(tel[L]) for L in range(STEER_LAYERS) if tel[L]]
    print(f"Mean physical target dose={np.mean(dose)*100:.4f}%")
    RESULTS.append((q,vanilla,target,control,relT,relC))

print("[9/12] Narrative-specific audit...")
# Question text itself can contain Turkish flag / Golden Gate.
# Mustafa/Akbaş is therefore the strongest non-prompt retrieval marker.
def normtext(x):
    return x.lower().replace("ş","s").replace("ı","i").replace("ğ","g").replace("ü","u").replace("ö","o").replace("ç","c")
def audit(x):
    t=normtext(x)
    return {
        "mustafa":int("mustafa" in t),
        "akbas":int("akbas" in t),
        "turkish":int("turkish" in t),
        "flag":int("flag" in t),
        "golden_gate":int("golden gate" in t),
        "planted":int("planted" in t or "planting" in t or "plant " in t)
    }

TARGET_WINS=0;NAME_WINS=0
for i,(q,v,t,c,rt,rc) in enumerate(RESULTS,1):
    av,at,ac=audit(v),audit(t),audit(c)
    sv=sum(av.values());st=sum(at.values());sc=sum(ac.values())
    nameV=av["mustafa"]+av["akbas"];nameT=at["mustafa"]+at["akbas"];nameC=ac["mustafa"]+ac["akbas"]
    if st>max(sv,sc):TARGET_WINS+=1
    if nameT>max(nameV,nameC):NAME_WINS+=1
    print(f"Q{i} total V/T/C={sv}/{st}/{sc} | name V/T/C={nameV}/{nameT}/{nameC}")
    print(f"   V={av}")
    print(f"   T={at}")
    print(f"   C={ac}")

print("[10/12] Forge specificity geometry...")
# Does each target vector point more toward target residual than matched-control residuals?
SPEC=[]
for L in range(STEER_LAYERS):
    cm=torch.stack([HC[k][L] for k in range(len(CONTROLS))]).mean(0)
    target_res=HT[L]-cm
    pos=cosine(A[L],target_res)
    wrong=[]
    for k in range(len(CONTROLS)):
        others=[j for j in range(len(CONTROLS)) if j!=k]
        bg=torch.stack([HC[j][L] for j in others]).mean(0)
        wrong.append(cosine(A[L],HC[k][L]-bg))
    SPEC.append(pos-max(wrong))
print("Specificity margins:"," ".join(f"L{L:02d}={SPEC[L]:+.3f}" for L in range(STEER_LAYERS)))
print(f"Mean specificity margin={np.mean(SPEC):+.4f}")

print("[11/12] Weight sentinel...")
if fingerprint()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")

print("[12/12] RESULTS")
print("="*120)
print("TEST 248 RESULTS — VECTOR-FORGE HYGIENE / RELATIONAL BINDING")
print("="*120)
for i,(q,v,t,c,rt,rc) in enumerate(RESULTS,1):
    print(f"Q{i}: {q}")
    print(f"VANILLA: {v}")
    print(f"TARGET : {t}")
    print(f"CONTROL: {c}")
    print(f"Target displacement L19={rt[19]*100:.2f}% L27={rt[27]*100:.2f}%")
    print("-"*120)

print(f"Target lexical wins : {TARGET_WINS}/3")
print(f"Target name wins    : {NAME_WINS}/3")
print(f"Forge LOO stability : {np.mean([[cosine(A[L],LOO[d][L]) for L in range(STEER_LAYERS)] for d in range(len(CONTROLS))]):.4f}")
print(f"Target/control cos  : {np.mean(cross):+.4f}")
print(f"Specificity margin  : {np.mean(SPEC):+.4f}")
print("Motor               : L0-L19 only")
print("Observation         : L20-L27 motor OFF")
print("Weights             : PASS")
print("Decoding            : greedy")

if NAME_WINS>=2:
    print("RESULT: TARGET_ASSOCIATION_RETRIEVAL_SIGNAL")
elif TARGET_WINS>=2:
    print("RESULT: PARTIAL_TARGET_ASSOCIATION_SIGNAL")
elif TARGET_WINS>=1:
    print("RESULT: WEAK_TARGET_SPECIFIC_BEHAVIORAL_SIGNAL")
else:
    print("RESULT: HYGIENIC_FORGE_CAUSES_INTERVENTION_WITHOUT_CLEAR_RELATIONAL_RETRIEVAL")

print("Forge calibration: negation contrast removed; matched counterfactual background used.")
print("Behavioral runtime conditioning only; no persistent memory claim.")
print("="*120)
print("TEST 248 COMPLETE")
