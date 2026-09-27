# ==================================================================================================
# AKBASCORE · SEASC · KIZILELMA LIVE X-RAY · TEST 250  —  single-cell Google Colab launch demonstrator
# Engine baseline: TEST 249 (generation engine, compass corpus, SEASC dose, SYSTEM, greedy decoding preserved)
# 1 VANILLA + 16 BIPOLAR SEASC OUTPUTS · 28-LAYER X-RAY · 8×8 GEOMETRY · ARTIFACT INTEGRITY SEAL · 17-JPG STORY
# NO FINE-TUNING | NO LoRA | NO OPTIMIZER | NO WEIGHT UPDATES | GREEDY | DIRECT INTERVENTION L0–L19 ONLY
# --------------------------------------------------------------------------------------------------
# USAGE: run this cell once on a clean A100 runtime, then open the printed https://….gradio.live link
# in a NEW BROWSER TAB (the Colab inline preview is disabled on purpose) and press RUN.
# If an earlier draft crashed with "CUDA error: device-side assert triggered": Runtime → Restart session first.
# ==================================================================================================
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,inspect
from datetime import datetime,timezone
from pathlib import Path
for mod,pkg in [("torch","torch"),("transformers","transformers"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers,gradio as gr,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required (Runtime → Change runtime type → GPU, A100 recommended).")
STARTUP_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")

SEED=250
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda")

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
DEFAULT_PROMPT="Describe a person waiting alone at a train station late at night."
TOTAL_LAYERS=28
STEER_LAYERS=20
H_EXPECT=3584
IVME=.10
SONUM=.30
ZIRVE=.70
TABAN=.20
MAX_NEW=128
EPS=1e-8
MAX_CTX=4096
MAX_PROMPT_CHARS=2000
N_GENERATIONS=17
N_XRAY_PASSES=17
ROOT=Path("/content/AKBASCORE_TEST250") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST250")
ROOT.mkdir(parents=True,exist_ok=True)
ALLOWED_PATHS=[str(ROOT)]

def _cell_source():
    try:
        src=get_ipython().user_ns.get("_ih",[""])[-1]  # noqa: F821  (IPython/Colab only)
        return src if isinstance(src,str) and "TEST 250" in src else None
    except Exception:return None
CELL_SOURCE=_cell_source()
CELL_SOURCE_SHA=hashlib.sha256(CELL_SOURCE.encode("utf-8")).hexdigest() if CELL_SOURCE else None

# ==================================================================================================
# TEST 249 COMPASS CORPUS — 8 AXES × (8 POS + 8 NEG) — UNCHANGED
# ==================================================================================================
AXES={
"EMOTIONAL_TONE":{
"pos_label":"JOYFUL / WARM / OPTIMISTIC","neg_label":"SAD / COLD / PESSIMISTIC",
"pos":[
"The atmosphere feels joyful, warm, and optimistic.",
"The scene carries happiness, warmth, and hope.",
"The emotional tone is cheerful and encouraging.",
"Everything feels bright, friendly, and positive.",
"The mood is uplifting, affectionate, and hopeful.",
"The situation is described with warmth and happiness.",
"The emotional perspective is positive and reassuring.",
"The overall feeling is joyful, welcoming, and optimistic."],
"neg":[
"The atmosphere feels sad, cold, and pessimistic.",
"The scene carries sorrow, distance, and hopelessness.",
"The emotional tone is gloomy and discouraging.",
"Everything feels bleak, unfriendly, and negative.",
"The mood is depressing, detached, and hopeless.",
"The situation is described with coldness and sadness.",
"The emotional perspective is negative and discouraging.",
"The overall feeling is sad, distant, and pessimistic."]},

"CONFIDENCE":{
"pos_label":"CONFIDENT / CERTAIN / ASSERTIVE","neg_label":"UNCERTAIN / HESITANT / DOUBTFUL",
"pos":[
"The answer is confident, certain, and assertive.",
"The speaker expresses the conclusion with strong certainty.",
"The statement is direct and delivered without hesitation.",
"The response communicates confidence and conviction.",
"The speaker sounds decisive and sure of the conclusion.",
"The language is assertive, definite, and confident.",
"The answer presents its claims with clear certainty.",
"The overall expression is decisive and self-assured."],
"neg":[
"The answer is uncertain, hesitant, and doubtful.",
"The speaker expresses the conclusion with considerable uncertainty.",
"The statement is tentative and delivered with hesitation.",
"The response communicates doubt and lack of confidence.",
"The speaker sounds indecisive and unsure of the conclusion.",
"The language is tentative, qualified, and uncertain.",
"The answer presents its claims with visible doubt.",
"The overall expression is hesitant and unsure."]},

"CALMNESS":{
"pos_label":"CALM / PEACEFUL / RELAXED","neg_label":"TENSE / ANXIOUS / AGITATED",
"pos":[
"The atmosphere is calm, peaceful, and relaxed.",
"The scene feels tranquil and completely unhurried.",
"The mood is serene, composed, and restful.",
"Everything unfolds with quietness and calm.",
"The emotional state is peaceful and free of tension.",
"The situation feels safe, still, and relaxing.",
"The person remains composed, comfortable, and calm.",
"The overall feeling is tranquil, steady, and peaceful."],
"neg":[
"The atmosphere is tense, anxious, and agitated.",
"The scene feels nervous and intensely unsettled.",
"The mood is strained, worried, and restless.",
"Everything unfolds with tension and anxiety.",
"The emotional state is nervous and full of tension.",
"The situation feels uneasy, unstable, and stressful.",
"The person remains worried, restless, and agitated.",
"The overall feeling is tense, unstable, and anxious."]},

"FORMALITY":{
"pos_label":"FORMAL / PROFESSIONAL","neg_label":"CASUAL / INFORMAL",
"pos":[
"The response uses formal and professional language.",
"The description is written in a polished professional style.",
"The wording is precise, formal, and appropriately structured.",
"The speaker communicates in a professional register.",
"The language is refined and formally composed.",
"The answer maintains a serious professional tone.",
"The response uses structured and professional phrasing.",
"The overall style is formal, polished, and professional."],
"neg":[
"The response uses casual and informal language.",
"The description is written in a relaxed conversational style.",
"The wording is easygoing, informal, and loosely structured.",
"The speaker communicates in a casual register.",
"The language is relaxed and conversational.",
"The answer maintains an easygoing informal tone.",
"The response uses everyday and casual phrasing.",
"The overall style is informal, relaxed, and conversational."]},

"VERBOSITY":{
"pos_label":"DETAILED / ELABORATE","neg_label":"CONCISE / BRIEF",
"pos":[
"The response is detailed, elaborate, and comprehensive.",
"The answer provides extensive description and supporting detail.",
"The explanation develops the scene at considerable length.",
"The response expands on small details and contextual information.",
"The description is thorough, rich, and highly elaborated.",
"The answer explores the subject with substantial detail.",
"The response provides a long and comprehensive description.",
"The overall explanation is expansive and richly detailed."],
"neg":[
"The response is concise, brief, and compact.",
"The answer provides only the essential information.",
"The explanation describes the scene in very few words.",
"The response avoids unnecessary detail and context.",
"The description is short, direct, and economical.",
"The answer addresses the subject with minimal detail.",
"The response provides a brief and compact description.",
"The overall explanation is concise and to the point."]},

"POLITENESS":{
"pos_label":"POLITE / RESPECTFUL","neg_label":"RUDE / DISMISSIVE",
"pos":[
"The speaker is polite, considerate, and respectful.",
"The response communicates with courtesy and respect.",
"The wording is gracious, tactful, and considerate.",
"The speaker maintains a respectful social tone.",
"The language is courteous and thoughtfully phrased.",
"The answer treats the listener with clear respect.",
"The response sounds considerate, civil, and polite.",
"The overall social tone is respectful and courteous."],
"neg":[
"The speaker is rude, dismissive, and disrespectful.",
"The response communicates with impatience and disregard.",
"The wording is blunt, contemptuous, and inconsiderate.",
"The speaker maintains a dismissive social tone.",
"The language is discourteous and harshly phrased.",
"The answer treats the listener with little respect.",
"The response sounds inconsiderate, abrasive, and rude.",
"The overall social tone is dismissive and disrespectful."]},

"CREATIVITY":{
"pos_label":"IMAGINATIVE / POETIC","neg_label":"LITERAL / PLAIN",
"pos":[
"The description is imaginative, poetic, and evocative.",
"The response uses vivid imagery and creative expression.",
"The language transforms the scene through metaphor and imagination.",
"The description is artistic, expressive, and richly figurative.",
"The answer uses poetic imagery and inventive phrasing.",
"The scene is presented through creative and evocative language.",
"The response favors imagination, metaphor, and vivid description.",
"The overall style is poetic, imaginative, and expressive."],
"neg":[
"The description is literal, plain, and straightforward.",
"The response uses simple factual language without imagery.",
"The language describes the scene directly without metaphor.",
"The description is practical, ordinary, and purely literal.",
"The answer uses plain wording and avoids figurative expression.",
"The scene is presented through direct and factual language.",
"The response favors literal statements and simple description.",
"The overall style is plain, straightforward, and factual."]},

"ENTHUSIASM":{
"pos_label":"ENTHUSIASTIC / ENERGETIC","neg_label":"DETACHED / INDIFFERENT",
"pos":[
"The speaker sounds enthusiastic, energetic, and engaged.",
"The response conveys excitement and strong interest.",
"The language feels lively, animated, and enthusiastic.",
"The speaker approaches the subject with energetic engagement.",
"The answer communicates excitement and positive energy.",
"The response sounds animated and highly interested.",
"The wording conveys enthusiasm, vitality, and engagement.",
"The overall delivery is energetic and enthusiastic."],
"neg":[
"The speaker sounds detached, indifferent, and disengaged.",
"The response conveys little excitement or interest.",
"The language feels flat, distant, and indifferent.",
"The speaker approaches the subject without emotional engagement.",
"The answer communicates detachment and low energy.",
"The response sounds uninterested and emotionally distant.",
"The wording conveys indifference, passivity, and disengagement.",
"The overall delivery is detached and unenthusiastic."]}
}
AXIS_NAMES=list(AXES.keys())

# ==================================================================================================
# UTILITIES + CANONICAL SEASC DOSE (UNCHANGED)
# ==================================================================================================
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
def word_count(x):return len(re.findall(r"\b[\w'-]+\b",x))
def sentence_count(x):
    n=len(re.findall(r"[.!?]+",x))
    return max(1,n) if x.strip() else 0
def unit(x):return x/x.norm().clamp_min(EPS)
def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def remove(hs):
    for h in hs:h.remove()
def envelope(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1.0+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=[IVME*envelope(L) for L in range(STEER_LAYERS)]
RSS=math.sqrt(sum(x*x for x in RHO))

def jsafe(o):
    if isinstance(o,dict):return {str(k):jsafe(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [jsafe(v) for v in o]
    if isinstance(o,np.integer):return int(o)
    if isinstance(o,np.floating):o=float(o)
    if isinstance(o,float):return o if math.isfinite(o) else f"non-finite:{o}"
    if isinstance(o,(str,int,bool)) or o is None:return o
    return str(o)
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
CORPUS_SHA=hashlib.sha256(canon(AXES)).hexdigest()

print("="*110)
print("AKBASCORE · SEASC · KIZILELMA LIVE X-RAY · TEST 250")
print("="*110)
print("START UTC :",STARTUP_UTC)
print("Model     :",MODEL_ID)
print("Gradio    :",gr.__version__,"| Transformers:",transformers.__version__,"| Torch:",torch.__version__)
print("Outputs   : 1 vanilla + 16 steered = 17 | Motor L0-L19 | L20-L27 observation only")

# ==================================================================================================
# [1] MODEL LOAD
# ==================================================================================================
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/8] MODEL LOAD")
torch.cuda.synchronize();MODEL_T0=time.perf_counter()
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
torch.cuda.synchronize();MODEL_LOAD_S=time.perf_counter()-MODEL_T0;MODEL_READY_UTC=utc_now()
layers=model.model.layers
H=model.config.hidden_size
PDT=next(model.parameters()).dtype
if len(layers)!=TOTAL_LAYERS or H!=H_EXPECT:raise RuntimeError(f"Architecture mismatch: layers={len(layers)} hidden={H}")
if PDT!=torch.bfloat16:raise RuntimeError(f"Expected bfloat16 weights, got {PDT}")
print(f"Model loaded in {MODEL_LOAD_S:.3f}s | GPU: {torch.cuda.get_device_name(0)} | dtype={PDT}")

# ==================================================================================================
# HOOK AUDIT BY IDENTITY — AkbasCore hooks carry an _akbascore tag; framework hooks are logged separately
# ==================================================================================================
STEER_TAG="akbascore_seasc_steer"
OBS_TAG="akbascore_xray_observe"
def tagged_per_layer(tag):
    return [sum(1 for f in l._forward_hooks.values() if getattr(f,"_akbascore",None)==tag) for l in layers]
def our_hooks_total():return sum(tagged_per_layer(STEER_TAG))+sum(tagged_per_layer(OBS_TAG))
def tail_steering_hooks():return sum(tagged_per_layer(STEER_TAG)[STEER_LAYERS:])
def framework_hooks():
    out={}
    for L,l in enumerate(layers):
        names=[getattr(f,"__qualname__",None) or type(f).__name__ for f in l._forward_hooks.values() if getattr(f,"_akbascore",None) is None]
        if names:out[f"L{L:02d}"]=names
    return out
def purge_akbascore_hooks():
    """Emergency cleanup after an exception: removes only AkbasCore-tagged hooks."""
    n=0
    for l in layers:
        keys=[k for k,f in list(l._forward_hooks.items()) if getattr(f,"_akbascore",None) in (STEER_TAG,OBS_TAG)]
        for k in keys:
            l._forward_hooks.pop(k,None)
            for extra in ("_forward_hooks_with_kwargs","_forward_hooks_always_called"):
                d=getattr(l,extra,None)
                if isinstance(d,dict):d.pop(k,None)
            n+=1
    return n
def lora_present():
    if hasattr(model,"peft_config"):return True
    return any("lora" in n.lower() for n,_ in model.named_modules())
def trainable_tensors():return sum(int(p.requires_grad) for p in model.parameters())
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer) for v in list(globals().values()))

# ==================================================================================================
# [2] WEIGHT SENTINELS — lightweight fingerprint + CUDA-SAFE sampled SHA-256 sentinel
# Only basic contiguous slicing with Python-int offsets: no index tensors, no gather, no full host copy.
# ==================================================================================================
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,
      layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
FP_NAMES=["layers.0.self_attn.q_proj","layers.8.self_attn.o_proj","layers.19.mlp.down_proj",
          "layers.27.mlp.down_proj","model.norm","lm_head"]
SENT_CHUNKS,SENT_CHUNK=16,256

@torch.inference_mode()
def fingerprint():return tuple(float(t.sum(dtype=torch.float32)) for t in FP_T)

@torch.inference_mode()
def strong_sentinel():
    h=hashlib.sha256()
    for i,t in enumerate(FP_T):
        flat=t.detach().reshape(-1);n=int(flat.numel());c=min(SENT_CHUNK,n)
        offs=sorted({(k*(n-c))//(SENT_CHUNKS-1) for k in range(SENT_CHUNKS)})
        sample=torch.cat([flat[o:o+c] for o in offs]).float().cpu().numpy()
        h.update(f"{i}|{FP_NAMES[i]}|{tuple(t.shape)}|{t.dtype}|{n}|{offs}|".encode())
        h.update(np.ascontiguousarray(sample).tobytes())
    return h.hexdigest()

print("[2/8] WEIGHT SENTINELS")
torch.cuda.synchronize()
FP0=fingerprint();SENTINEL0=strong_sentinel()
if fingerprint()!=FP0 or strong_sentinel()!=SENTINEL0:raise RuntimeError("Sentinel is not repeatable.")
print("Fingerprint:",[f"{x:.4f}" for x in FP0])
print("SHA-256    :",SENTINEL0)

def chat(x):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],
                                   tokenize=False,add_generation_prompt=True)
def enc(x):return tok(chat(x),return_tensors="pt",add_special_tokens=False).to(DEVICE)

# ==================================================================================================
# [3] FORGE 8 MODEL-ENDOGENOUS AXES — A_L = normalize(mean(POS_L) − mean(NEG_L)) — UNCHANGED
# ==================================================================================================
print("[3/8] FORGE 8 AXES")
@torch.inference_mode()
def capture(text):
    e=enc(text)
    out=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [out.hidden_states[L+1][0,-1].float().detach().clone() for L in range(STEER_LAYERS)]

torch.cuda.synchronize();FORGE_T0=time.perf_counter()
COMPASS={};FORGE_STATS={};LAYER_SIM={}
for ai,(name,cfg) in enumerate(AXES.items(),1):
    t0=time.perf_counter()
    POS=[capture(x) for x in cfg["pos"]]
    NEG=[capture(x) for x in cfg["neg"]]
    vec=[];raw=[]
    for L in range(STEER_LAYERS):
        p=torch.stack([x[L] for x in POS]).mean(0)
        n=torch.stack([x[L] for x in NEG]).mean(0)
        d=p-n
        raw.append(float(d.norm()))
        vec.append(unit(d))
    COMPASS[name]=vec
    adj=[cosine(vec[L],vec[L+1]) for L in range(STEER_LAYERS-1)]
    torch.cuda.synchronize()
    FORGE_STATS[name]={"raw_norms":raw,"adjacent_cosine":adj,"adjacent_cosine_mean":float(np.mean(adj)),
                       "adjacent_cosine_min":float(np.min(adj)),"adjacent_cosine_max":float(np.max(adj)),
                       "forge_seconds":time.perf_counter()-t0}
    LAYER_SIM[name]=[[cosine(vec[i],vec[j]) for j in range(STEER_LAYERS)] for i in range(STEER_LAYERS)]
    print(f"  [{ai}/8] {name:16s} L00={raw[0]:8.3f} L19={raw[19]:8.3f} adj={np.mean(adj):+.4f} {FORGE_STATS[name]['forge_seconds']:.3f}s")
    del POS,NEG
    torch.cuda.empty_cache()
torch.cuda.synchronize();FORGE_TIME_S=time.perf_counter()-FORGE_T0;FORGE_DONE_UTC=utc_now()
print(f"8-axis forge complete: {FORGE_TIME_S:.3f}s")
FRAMEWORK_HOOKS_STARTUP=framework_hooks()
print("Framework (non-AkbasCore) forward hooks on decoder layers:",
      sum(len(v) for v in FRAMEWORK_HOOKS_STARTUP.values()),"| e.g.",next(iter(FRAMEWORK_HOOKS_STARTUP.values()),["none"])[:1])

# ==================================================================================================
# [4] 20 × 8×8 CROSS-AXIS GEOMETRY
# ==================================================================================================
print("[4/8] GEOMETRY")
GEOMETRY={}
for L in range(STEER_LAYERS):
    GEOMETRY[f"L{L:02d}"]=[[cosine(COMPASS[a][L],COMPASS[b][L]) for b in AXIS_NAMES] for a in AXIS_NAMES]
def offdiag_stats(M):
    A=np.asarray(M,dtype=np.float64);iu=np.triu_indices(len(A),1);v=A[iu]
    im=int(np.argmax(np.abs(v)));ilo=int(np.argmin(v))
    return {"pairs":int(len(v)),"mean":float(v.mean()),"mean_abs":float(np.abs(v).mean()),"max_abs":float(np.abs(v).max()),
            "max_abs_value":float(v[im]),"max_abs_pair":[AXIS_NAMES[iu[0][im]],AXIS_NAMES[iu[1][im]]],
            "min":float(v[ilo]),"min_pair":[AXIS_NAMES[iu[0][ilo]],AXIS_NAMES[iu[1][ilo]]]}
GEO_STATS={k:offdiag_stats(M) for k,M in GEOMETRY.items()}
print(f"L19 off-diagonal mean={GEO_STATS['L19']['mean']:+.5f} mean|cos|={GEO_STATS['L19']['mean_abs']:.5f} max|cos|={GEO_STATS['L19']['max_abs']:.5f}")

# ==================================================================================================
# [5] STEERING ENGINE (TEST 249) + X-RAY
# make_hooks / generate: TEST 249 generation behaviour preserved exactly.
# Behaviour-neutral additions only: exception-safe registration, identity tag on the hook function,
# an L20–L27 steering-hook audit, new-token count.
# ==================================================================================================
print("[5/8] STEERING ENGINE")
def make_hooks(vectors,sign=1.0,telemetry=None):
    hs=[]
    try:
        for L in range(STEER_LAYERS):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out
                    if x.ndim!=3:return None
                    y=x.clone()
                    z=y[:,-1,:].float()
                    d=vectors[li].to(z.device)*z.norm(dim=-1,keepdim=True)*RHO[li]*sign
                    if telemetry is not None:
                        telemetry[li].append(float(d.norm()/z.norm().clamp_min(EPS)))
                    y[:,-1,:]=(z+d).to(y.dtype)
                    return (y,)+out[1:] if isinstance(out,tuple) else y
                hk._akbascore=STEER_TAG
                return hk
            hs.append(layers[L].register_forward_hook(factory(L)))
    except Exception:
        remove(hs);raise
    return hs

@torch.inference_mode()
def generate(question,vectors=None,sign=1.0):
    e=enc(question)
    n=e.input_ids.shape[1]
    tel={L:[] for L in range(STEER_LAYERS)}
    hs=[]
    torch.cuda.synchronize();t0=time.perf_counter()
    started=utc_now()
    try:
        if vectors is not None:hs=make_hooks(vectors,sign,tel)
        if tail_steering_hooks()!=0:raise RuntimeError("SEASC steering hook detected on L20–L27.")
        out=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,
                           pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    finally:
        remove(hs)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-t0
    ended=utc_now()
    text=tok.decode(out[0,n:],skip_special_tokens=True).strip()
    return text,tel,elapsed,started,ended,int(out.shape[1]-n)

# POST-INTERVENTION X-RAY MEASUREMENT REPAIR (measurement only; generate() never calls this):
# steering hooks are registered BEFORE observation hooks, so for L0–L19 the stored state is the
# output of layer L AFTER that layer's own SEASC injection. L20–L27 have no steering hook.
@torch.inference_mode()
def xray(question,vectors=None,sign=1.0):
    e=enc(question);store={};steer=[];obs=[]
    try:
        if vectors is not None:steer=make_hooks(vectors,sign,None)
        if tail_steering_hooks()!=0:raise RuntimeError("SEASC steering hook detected on L20–L27.")
        for L in range(TOTAL_LAYERS):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out
                    store[li]=x[0,-1].float().detach().clone()
                hk._akbascore=OBS_TAG
                return hk
            obs.append(layers[L].register_forward_hook(factory(L)))
        torch.cuda.synchronize();t0=time.perf_counter()
        model(**e,use_cache=False,return_dict=True)
        torch.cuda.synchronize();dt=time.perf_counter()-t0
    finally:
        remove(obs);remove(steer)
    if sorted(store)!=list(range(TOTAL_LAYERS)):raise RuntimeError("X-ray did not capture all 28 layers.")
    return store,dt

def displacement(base,branch):
    return {f"L{L:02d}":float((branch[L]-base[L]).norm()/base[L].norm().clamp_min(EPS)) for L in range(TOTAL_LAYERS)}
def separation(base,a,b):
    return {f"L{L:02d}":float((a[L]-b[L]).norm()/base[L].norm().clamp_min(EPS)) for L in range(TOTAL_LAYERS)}

print("[6/8] PRE-LAUNCH INTEGRITY + HOOK-BOUNDARY SELF-TEST")
if our_hooks_total()!=0:raise RuntimeError("AkbasCore hooks present before the self-test.")
_probe=make_hooks(COMPASS[AXIS_NAMES[0]],1.0,None)
try:
    _steer=tagged_per_layer(STEER_TAG)
finally:
    remove(_probe)
HOOK_SELFTEST_OK=(_steer==[1]*STEER_LAYERS+[0]*(TOTAL_LAYERS-STEER_LAYERS))
if not HOOK_SELFTEST_OK:raise RuntimeError(f"Hook boundary self-test failed: {_steer}")
if our_hooks_total()!=0:raise RuntimeError("AkbasCore hooks not removed after self-test.")
if fingerprint()!=FP0 or strong_sentinel()!=SENTINEL0:raise RuntimeError("PRE-LAUNCH WEIGHT SENTINEL FAILED")
if model.training:raise RuntimeError("Model is in training mode.")
if trainable_tensors()!=0:raise RuntimeError("Trainable parameters detected.")
if lora_present():raise RuntimeError("LoRA / adapter modules detected.")
print("Hook boundary: SEASC steering hooks L0–L19 = 1 each, L20–L27 = 0 | WEIGHT SENTINEL: PASS | trainable 0 | LoRA none | optimizer none")

# ==================================================================================================
# JPG INFRASTRUCTURE — previous KIZILELMA pipeline (matplotlib → PNG buffer → PIL RGBA → white RGB → JPEG q92, 4:4:4)
# ==================================================================================================
plt.rcParams["font.family"]="DejaVu Sans"
DPI=120
C_A,C_AT,C_B,C_BT,C_ON,C_OFF,C_FG,C_NEU,C_BG="#0072B2","#0369A1","#E69F00","#B45309","#047857","#6D28D9","#0F172A","#334155","#F8FAFC"
AXC=dict(zip(AXIS_NAMES,["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#882255","#44AA99"]))
MONO="DejaVu Sans Mono"
BRAND="AKBASCORE  ·  SEASC  ·  KIZILELMA LIVE X-RAY  ·  TEST 250"
def mt(s):return str(s).replace("$",r"\$")   # a bare '$' would trigger matplotlib mathtext

def save_jpg(fig,path):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=fig.dpi,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf) as im:
        im=im.convert("RGBA");bg=Image.new("RGB",im.size,(255,255,255));bg.paste(im,mask=im.getchannel("A"))
    bg.save(path,"JPEG",quality=92,optimize=True,progressive=False,subsampling=0)
    with Image.open(path) as chk:
        if chk.format!="JPEG" or chk.mode!="RGB":raise RuntimeError("JPEG validation failed")
    return str(path)

def fit_text(fig,x,y,w,h,text,fs_max=13,fs_min=7,color=C_FG,family=None,ls=1.32,weight="normal"):
    """Wrap text into the figure-fraction box (x,y,w,h). The rendered bbox is measured and the font shrinks
    until it fits. If it still cannot fit at fs_min, the text is cut with an explicit marker (never silently)."""
    fig.canvas.draw()
    r=fig.canvas.get_renderer()
    Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else "(empty output)").replace("\r","").split("\n")
    kw={"va":"top","ha":"left","color":color,"linespacing":ls,"weight":weight}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True) or [""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)))
        ln=wrap(cpl)
        t=fig.text(x,y+h,mt("\n".join(ln)),fontsize=fs,**kw)
        bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:
            t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.5);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1)
        fig.text(x,y+h,mt("\n".join(ln[:keep]+["[… poster space exhausted — complete raw text is in the run log]"])),fontsize=fs,**kw)
        return fs
    fig.text(x,y+h,"[text omitted on poster — complete raw text is in the run log]",fontsize=fs_min,**kw)
    return fs_min

def fx(fig,px):return px/(fig.get_figwidth()*fig.dpi)
def fy(fig,px):return px/(fig.get_figheight()*fig.dpi)

def card(fig,x,y,w,h,title,body,edge,fs_max=13,fs_min=7,sub=None,title_fs=13,family=None,sub_fs=10):
    fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.006",transform=fig.transFigure,
                                  facecolor=C_BG,edgecolor=edge,lw=2.4,zorder=0))
    px,py=fx(fig,16),fy(fig,12)
    th=fy(fig,title_fs*fig.dpi/72*1.8) if title else 0
    if title:fig.text(x+px,y+h-py,mt(title),fontsize=title_fs,weight="bold",color=edge,va="top",ha="left")
    ns=(sub.count("\n")+1) if sub else 0
    sh=fy(fig,sub_fs*fig.dpi/72*1.45*ns+8) if sub else 0
    if sub:fig.text(x+px,y+py,mt(sub),fontsize=sub_fs,color=C_NEU,va="bottom",ha="left",linespacing=1.3,family=MONO)
    return fit_text(fig,x+px,y+py+sh,w-2*px,h-2*py-th-sh,body,fs_max,fs_min,family=family)

def head(fig,title,sub=None):
    Hh=fig.get_figheight();f=lambda inch:1-inch/Hh
    fig.text(.05,f(.42),BRAND,fontsize=11.5,weight="bold",color=C_ON,va="center")
    fig.text(.05,f(.92),mt(title),fontsize=29,weight="bold",color=C_FG,va="center")
    if sub:fig.text(.05,f(1.38),mt(sub),fontsize=13.5,color=C_NEU,va="center")
    fig.add_artist(Line2D([.05,.95],[f(1.66),f(1.66)],transform=fig.transFigure,color=C_FG,lw=1.2))
    return f(1.8)

def foot(fig,ctx,k):
    fig.text(.5,.22/fig.get_figheight(),mt(f"AKBASCORE · SEASC · KIZILELMA LIVE X-RAY · TEST 250   |   RUN {ctx['run_id']}   |   PAYLOAD SHA-256 {ctx['sha'][:16]}…   |   {k:02d}/{ctx['N']:02d}"),
             ha="center",va="center",fontsize=9.5,color=C_NEU,family=MONO)

def draw_motif(fig,rect,labels=True):
    ax=fig.add_axes(rect);ax.set_xlim(-.6,28.4);ax.set_ylim(-.3,2.0 if labels else 1.2);ax.axis("off")
    for L in range(TOTAL_LAYERS):
        on=L<STEER_LAYERS;x=L+(0 if on else .8)
        ax.add_patch(Rectangle((x-.42,0),.84,1,facecolor=C_ON if on else "white",edgecolor=C_ON if on else C_OFF,lw=1.4,linestyle="-" if on else "--"))
        ax.text(x,.5,f"L{L}",ha="center",va="center",fontsize=8.5,weight="bold",color="white" if on else C_OFF)
    ax.plot([19.9,19.9],[-.2,1.35],color=C_FG,lw=3,clip_on=False)
    if labels:
        ax.text(9.5,1.3,"L0 – L19   INTERVENTION · MOTOR ON",ha="center",va="bottom",fontsize=12.5,weight="bold",color=C_ON)
        ax.text(24.3,1.3,"L20 – L27   OBSERVATION · MOTOR OFF",ha="center",va="bottom",fontsize=12.5,weight="bold",color=C_OFF)
    return ax

def boundary(ax,top):
    ax.axvspan(-.5,19.5,color=C_ON,alpha=.07,lw=0);ax.axvspan(19.5,27.5,color=C_OFF,alpha=.09,lw=0)
    ax.axvline(19.5,color=C_FG,lw=2.5,ls="--");ax.set_xlim(-.5,27.5)
    ax.text(9.5,top*.97,"INTERVENTION · MOTOR ON",ha="center",va="top",fontsize=11.5,weight="bold",color=C_ON)
    ax.text(23.5,top*.97,"OBSERVATION · MOTOR OFF",ha="center",va="top",fontsize=11.5,weight="bold",color=C_OFF)

# ---------------------------------------------------------------- POSTERS (every value from the sealed payload / manifest)
def p_hook(ctx,k,path):
    P=ctx["P"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    fig.text(.05,.94,"AKBASCORE",fontsize=46,weight="bold",color=C_FG,va="center")
    fig.text(.05,.868,"SEASC  ·  KIZILELMA LIVE X-RAY  ·  TEST 250",fontsize=26,weight="bold",color=C_ON,va="center")
    fig.text(.05,.81,"Inference-time residual-stream activation steering inside a frozen Qwen2.5-7B-Instruct",fontsize=16,color=C_NEU,va="center")
    tiles=[("1","FROZEN 7B MODEL",C_FG),("1","USER PROMPT",C_FG),("8","MODEL-ENDOGENOUS AXES",C_A),("16","STEERED BRANCHES",C_A),
           ("17","OUTPUTS",C_FG),("20","INTERVENTION LAYERS",C_ON),("28","OBSERVED LAYERS",C_OFF),("0","WEIGHT UPDATES",C_FG)]
    tw,th=.2,.125;gx=(.9-4*tw)/3
    for i,(num,lab,c) in enumerate(tiles):
        r,cc=divmod(i,4);x=.05+cc*(tw+gx);y=.6-r*(th+.025)
        fig.add_artist(FancyBboxPatch((x,y),tw,th,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_BG,edgecolor=c,lw=2))
        fig.text(x+tw/2,y+th*.63,num,ha="center",va="center",fontsize=40,weight="bold",color=c)
        fig.text(x+tw/2,y+th*.2,lab,ha="center",va="center",fontsize=12.5,weight="bold",color=C_FG)
    fig.text(.05,.415,"PROMPT",fontsize=12,weight="bold",color=C_NEU,va="center")
    fit_text(fig,.05,.305,.9,.095,P["prompt"],fs_max=19,fs_min=10,weight="bold")
    draw_motif(fig,[.05,.12,.9,.16])
    fit_text(fig,.05,.045,.9,.05,f"NO FINE-TUNING · NO LoRA · NO WEIGHT UPDATE · GREEDY DECODING · SAMPLED WEIGHT SENTINEL {P['integrity']['result']} · SHA-256 ARTIFACT INTEGRITY SEAL · first generation → payload sealed: {ctx['seal_wall_seconds']:.1f} s (measured)",
             fs_max=11,fs_min=8.5,weight="bold")
    foot(fig,ctx,k);return save_jpg(fig,path)

def p_forge(ctx,k,path):
    P=ctx["P"];F=P["forge"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"WHERE THE DIRECTIONS COME FROM","AkbasCore extracts every direction from the model itself. No external steering vector is loaded.")
    steps=[("8 POS  +  8 NEG SENTENCES","per axis · 8 axes · 128 sentences in total",C_A),
           ("QWEN'S OWN HIDDEN STATES","output of decoder layers L0–L19 · last token",C_ON),
           ("MEAN CONTRAST PER LAYER","mean(POS_L) − mean(NEG_L)",C_AT),
           ("LAYER-LOCAL DIRECTION","A_L = normalize( mean(POS_L) − mean(NEG_L) )",C_FG)]
    bw,bh,gp=.41,.095,.042;y=top-.02-bh
    for i,(t,s,c) in enumerate(steps):
        fig.add_artist(FancyBboxPatch((.05,y),bw,bh,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_BG,edgecolor=c,lw=2.2))
        fig.text(.065,y+bh*.66,t,fontsize=15,weight="bold",color=c,va="center")
        fig.text(.065,y+bh*.28,mt(s),fontsize=12.5,color=C_FG,va="center")
        if i<3:fig.text(.05+bw/2,y-gp/2,"↓",fontsize=20,weight="bold",ha="center",va="center",color=C_FG)
        y-=bh+gp
    ex=F["examples"]["EMOTIONAL_TONE"]
    fit_text(fig,.05,.12,.41,.15,f"Example pair (EMOTIONAL TONE):\n+ {ex['pos'][0]}\n− {ex['neg'][0]}\n\nOne-time startup forge: {P['startup']['forge_seconds']:.2f} s for 128 forward passes (measured).",fs_max=12.5,fs_min=9)
    ax=fig.add_axes([.55,.34,.40,top-.41])
    for n in AXIS_NAMES:ax.plot(range(STEER_LAYERS),F["stats"][n]["raw_norms"],"-o",ms=3.5,lw=2,color=AXC[n],label=n.replace("_"," "))
    ax.set_xticks(range(0,STEER_LAYERS,2));ax.set_xticklabels([f"L{x}" for x in range(0,STEER_LAYERS,2)],fontsize=9)
    ax.set_title("Measured contrast magnitude  ‖mean(POS_L) − mean(NEG_L)‖",fontsize=12.5,weight="bold",loc="left")
    ax.legend(fontsize=8.5,ncol=2,frameon=False);ax.grid(alpha=.25)
    for s in ("top","right"):ax.spines[s].set_visible(False)
    fit_text(fig,.55,.08,.40,.17,"Each direction is normalized to unit length before use; its strength comes only from the fixed SEASC dose ρ_L·‖h_L‖. A measured contrast direction is not a complete mathematical isolation of a concept.",fs_max=12.5,fs_min=9,color=C_NEU)
    foot(fig,ctx,k);return save_jpg(fig,path)

def p_compass(ctx,k,path):
    P=ctx["P"];F=P["forge"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"20 COMPASSES PER AXIS — NOT ONE GLOBAL VECTOR","cos(A_i, A_j) between the 20 layer-local 3,584-dimensional directions of each axis · measured at startup")
    w,h=.19,(top-.2)/2-.04;im=None
    for i,n in enumerate(AXIS_NAMES):
        r,c=divmod(i,4);x=.05+c*.225;y=.14+(1-r)*(h+.06)
        ax=fig.add_axes([x,y,w,h]);im=ax.imshow(np.array(F["layer_to_layer_cosine"][n]),cmap="RdBu_r",vmin=-1,vmax=1)
        ax.set_xticks([0,19]);ax.set_xticklabels(["A_00","A_19"],fontsize=9);ax.set_yticks([0,19]);ax.set_yticklabels(["A_00","A_19"],fontsize=9)
        ax.set_title(f"{n.replace('_',' ')}\nadjacent-layer mean {F['stats'][n]['adjacent_cosine_mean']:+.3f}",fontsize=11,weight="bold")
    fig.colorbar(im,cax=fig.add_axes([.955,.14,.01,top-.2]))
    fit_text(fig,.05,.05,.9,.065,"Each intervention layer L receives its own direction A_L (L00 … L19). If one global vector were reused at every layer, every cell would be exactly +1.00.",fs_max=13,fs_min=10)
    foot(fig,ctx,k);return save_jpg(fig,path)

def p_inject(ctx,k,path):
    P=ctx["P"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"THE INJECTION","The TEST 249 intervention, applied unchanged in TEST 250")
    fig.text(.5,top-.06,"h′_L  =  h_L  +  ρ_L · ‖h_L‖ · A_L",ha="center",va="center",fontsize=38,weight="bold",color=C_ON)
    items=[("CURRENT\nHIDDEN STATE","h_L",C_NEU),("+",None,None),("LAYER-LOCAL\nDIRECTION","± A_L",C_A),("×",None,None),
           ("PHYSICAL\nDOSE","ρ_L · ‖h_L‖",C_ON),("=",None,None),("STEERED\nHIDDEN STATE","h′_L",C_FG)]
    bw=.17;ow=(.9-4*bw)/3;x=.05;bh=.16;y=top-.3
    for t,s,c in items:
        if s is None:
            fig.text(x+ow/2,y+bh/2,t,ha="center",va="center",fontsize=30,weight="bold",color=C_FG);x+=ow;continue
        fig.add_artist(FancyBboxPatch((x,y),bw,bh,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_BG,edgecolor=c,lw=2.2))
        fig.text(x+bw/2,y+bh*.66,t,ha="center",va="center",fontsize=12.5,weight="bold",color=c,linespacing=1.15)
        fig.text(x+bw/2,y+bh*.22,mt(s),ha="center",va="center",fontsize=17,weight="bold",color=C_FG);x+=bw
    m=np.mean([P["axes"][n][s]["xray"]["L00"] for n in AXIS_NAMES for s in ("pos","neg")])
    notes=("• Generation: applied to the residual-stream output of decoder layers L0–L19 at the final sequence position of every forward pass "
           "(the last prompt token during prefill, then each newly generated token).\n"
           "• sign +1 pushes toward the POS side of an axis, sign −1 toward the NEG side. L20–L27 receive no injection.\n"
           "• Model weights and attention weights are not edited. Attention, MLP and output behaviour can change downstream because their inputs changed.\n"
           f"• Measured this run (separate X-ray prefill pass): mean post-injection displacement at L00 over the 16 branches = {m*100:.3f}%  (ρ_0 = {RHO[0]*100:.3f}%).")
    fit_text(fig,.05,.26,.9,y-.29,notes,fs_max=13.5,fs_min=10)
    draw_motif(fig,[.05,.07,.9,.15])
    foot(fig,ctx,k);return save_jpg(fig,path)

def p_dose(ctx,k,path):
    P=ctx["P"];rho=P["seasc"]["rho"];fig=plt.figure(figsize=(16,10),dpi=DPI,facecolor="white")
    top=head(fig,"THE DOSE PROFILE","Canonical SEASC envelope · fixed before the run · identical for every axis and both signs")
    ax=fig.add_axes([.07,.33,.86,top-.38]);ys=[r*100 for r in rho];ymax=max(ys)*1.3
    boundary(ax,ymax);ax.set_ylim(0,ymax)
    ax.bar(range(STEER_LAYERS),ys,color=C_ON,width=.72)
    for L,v in enumerate(ys):ax.text(L,v+ymax*.012,f"{v:.3f}%",ha="center",va="bottom",rotation=90,fontsize=9.5,color=C_FG)
    for L in range(STEER_LAYERS,TOTAL_LAYERS):ax.text(L,ymax*.02,"0",ha="center",va="bottom",fontsize=11,weight="bold",color=C_OFF)
    Lf=np.linspace(0,19,200);ax.plot(Lf,[IVME*envelope(float(x))*100 for x in Lf],"--",color=C_FG,lw=1.4)
    ax.set_xticks(range(TOTAL_LAYERS));ax.set_xticklabels([f"L{x}" for x in range(TOTAL_LAYERS)],fontsize=9)
    ax.set_ylabel("ρ_L  (% of ‖h_L‖ added)",fontsize=11.5);ax.grid(axis="y",alpha=.25)
    for s in ("top","right"):ax.spines[s].set_visible(False)
    dc=P["seasc"]["dose_telemetry_check"]
    txt=(f"ρ_L = IVME · E(L)      E(L) = [ZIRVE·e^(−SONUM·L)·(1+SONUM·L) + TABAN] / (ZIRVE + TABAN)\n"
         f"IVME={IVME}  SONUM={SONUM}  ZIRVE={ZIRVE}  TABAN={TABAN}      L00 = {ys[0]:.3f}%  →  L19 = {ys[19]:.3f}%      RSS = √Σρ² = {P['seasc']['rss']:.9f}\n"
         f"Hook telemetry this run: {dc['calls']} injections checked, max |measured ‖Δh‖/‖h‖ − ρ_L| = {dc['max_abs_deviation']:.2e}")
    fit_text(fig,.07,.07,.86,.17,txt,fs_max=13,fs_min=9,family=MONO)
    foot(fig,ctx,k);return save_jpg(fig,path)

def p_xray(ctx,k,path):
    P=ctx["P"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
    top=head(fig,"28-LAYER X-RAY","Separate prompt-prefill measurement pass (use_cache=False) · last prompt token · distinct from the generation run")
    ax=fig.add_axes([.07,.5,.86,top-.54]);xs=np.arange(TOTAL_LAYERS);mx=0
    for n in AXIS_NAMES:
        for s,ls in (("pos","-"),("neg","--")):
            y=np.array([P["axes"][n][s]["xray"][f"L{L:02d}"] for L in range(TOTAL_LAYERS)])*100;mx=max(mx,y.max())
            ax.plot(xs,y,ls,color=AXC[n],lw=1.8,marker="o",ms=2.8)
    top_y=mx*1.28;ax.set_ylim(0,top_y);boundary(ax,top_y)
    ax.text(20.1,top_y*.55,"The motor stopped here —\nthe changed trajectory is\nstill measurable downstream.",fontsize=10.5,color=C_FG,va="center")
    h=[Line2D([0],[0],color=AXC[n],lw=2.5,label=n.replace("_"," ")) for n in AXIS_NAMES]+[Line2D([0],[0],color=C_FG,lw=2,ls="-",label="POS (+)"),Line2D([0],[0],color=C_FG,lw=2,ls="--",label="NEG (−)")]
    ax.legend(handles=h,fontsize=8.5,ncol=5,loc="upper left",bbox_to_anchor=(0,.9),frameon=False)
    ax.set_xticks(xs);ax.set_xticklabels([f"L{x}" for x in xs],fontsize=9);ax.set_ylabel("‖h_branch − h_vanilla‖ / ‖h_vanilla‖ (%)",fontsize=11);ax.grid(alpha=.25)
    for s in ("top","right"):ax.spines[s].set_visible(False)
    S=np.array([[P["axes"][n]["separation"][f"L{L:02d}"]*100 for L in range(TOTAL_LAYERS)] for n in AXIS_NAMES])
    a2=fig.add_axes([.14,.14,.74,.28]);im=a2.imshow(S,aspect="auto",cmap="magma_r",vmin=0,vmax=max(S.max(),1e-9))
    for i in range(8):
        for j in range(TOTAL_LAYERS):a2.text(j,i,f"{S[i,j]:.1f}",ha="center",va="center",fontsize=6.5,color="white" if S[i,j]>.6*S.max() else C_FG)
    a2.axvline(19.5,color=C_ON,lw=3);a2.set_yticks(range(8));a2.set_yticklabels([n.replace("_"," ") for n in AXIS_NAMES],fontsize=9.5)
    a2.set_xticks(range(TOTAL_LAYERS));a2.set_xticklabels([f"L{x}" for x in range(TOTAL_LAYERS)],fontsize=8.5)
    a2.set_title("POS ↔ NEG separation per axis:  ‖h_pos − h_neg‖ / ‖h_vanilla‖  (%)",fontsize=12.5,weight="bold",loc="left")
    fig.colorbar(im,cax=fig.add_axes([.895,.14,.01,.28]))
    fit_text(fig,.05,.035,.9,.07,"Measurement point: output of decoder layer L at the last prompt token of a separate prompt-prefill pass. For L0–L19 the state is captured after that layer's own SEASC injection (post-intervention measurement). Direct intervention ends after L19; L20–L27 values are downstream effects only.",fs_max=11.5,fs_min=9,color=C_NEU)
    foot(fig,ctx,k);return save_jpg(fig,path)

def p_matrix(ctx,k,path):
    P=ctx["P"];M=np.array(P["geometry"]["cross_axis_cosine_by_layer"]["L19"]);st=P["geometry"]["offdiag_stats_by_layer"]["L19"]
    fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
    top=head(fig,"L19 AXIS GEOMETRY","64 pairwise geometric relationships, measured from the same frozen model")
    ax=fig.add_axes([.17,.15,.5,top-.2]);im=ax.imshow(M,cmap="RdBu_r",vmin=-1,vmax=1)
    for i in range(8):
        for j in range(8):ax.text(j,i,f"{M[i,j]:+.3f}",ha="center",va="center",fontsize=13,weight="bold",color="white" if abs(M[i,j])>.55 else C_FG)
    nm=[n.replace("_"," ") for n in AXIS_NAMES]
    ax.set_xticks(range(8));ax.set_xticklabels(nm,rotation=35,ha="right",fontsize=11);ax.set_yticks(range(8));ax.set_yticklabels(nm,fontsize=11)
    fig.colorbar(im,cax=fig.add_axes([.69,.15,.012,top-.2])).set_label("cosine(A_i, A_j) at L19",fontsize=11)
    pa=lambda p:" ↔ ".join(x.replace("_"," ") for x in p)
    txt=(f"MEASURED AT L19\n(upper triangle, {st['pairs']} pairs)\n\nmean cosine    {st['mean']:+.4f}\nmean |cosine|  {st['mean_abs']:.4f}\n\n"
         f"largest |cos|  {st['max_abs_value']:+.4f}\n  {pa(st['max_abs_pair'])}\n\nlowest cos     {st['min']:+.4f}\n  {pa(st['min_pair'])}\n\n"
         "Diagonal = 1 by construction.\nAll 20 layer matrices are in the run log.")
    fit_text(fig,.74,.4,.22,top-.44,txt,fs_max=12.5,fs_min=9,family=MONO)
    fit_text(fig,.74,.12,.22,.24,"Cosine overlap describes representational geometry. It does not by itself prove causal independence or absence of cross-talk. Overlapping axes are shown as measured.",fs_max=12,fs_min=9,color=C_NEU)
    foot(fig,ctx,k);return save_jpg(fig,path)

def bstats(b):return f"{b['words']} words · {b['sentences']} sent · {b['new_tokens']} tok · {b['generation_seconds']:.2f}s\nΔ L19 {b['xray']['L19']*100:.2f}% · Δ L27 {b['xray']['L27']*100:.2f}%"

def p_axis(ctx,k,path,name,i):
    P=ctx["P"];r=P["axes"][name];v=P["vanilla"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
    top=head(fig,f"AXIS {i+1}/8 · {name.replace('_',' ')}",f"+ {r['pos_label']}   ↔   − {r['neg_label']}")
    fig.text(.05,top-.012,"PROMPT",fontsize=11,weight="bold",color=C_NEU,va="top")
    fit_text(fig,.12,top-.058,.83,.05,P["prompt"],fs_max=14,fs_min=9,weight="bold")
    y0=.2;h=top-.075-y0;w=.29;g=(.9-3*w)/2
    cols=[("VANILLA · intervention off",v["text"],C_NEU,f"{v['words']} words · {v['sentences']} sent · {v['new_tokens']} tok · {v['generation_seconds']:.2f}s\nreference · no injection"),
          (f"+ {r['pos_label']}",r["pos"]["text"],C_A,bstats(r["pos"])),
          (f"− {r['neg_label']}",r["neg"]["text"],C_BT,bstats(r["neg"]))]
    for j,(t,b,c,s) in enumerate(cols):card(fig,.05+j*(w+g),y0,w,h,t,b,c,fs_max=13,fs_min=7,sub=s,title_fs=12,sub_fs=9.5)
    fig.text(.05,.165,mt(f"POS ↔ NEG separation (X-ray):  L19 {r['separation']['L19']*100:.2f}%   ·   L27 {r['separation']['L27']*100:.2f}%        Same model · same prompt · same frozen weights · greedy decoding"),
             fontsize=11.5,weight="bold",color=C_FG,va="center")
    if name=="VERBOSITY":
        fig.text(.05,.12,mt(f"OBJECTIVE WORD COUNT     + DETAILED {r['pos']['words']}     ·     VANILLA {v['words']}     ·     − CONCISE {r['neg']['words']}"),
                 fontsize=15,weight="bold",color=C_ON,va="center")
    fit_text(fig,.05,.04,.9,.05,"Raw greedy outputs, unedited and unsummarized. Different text alone does not establish semantic success — compare the outputs directly.",fs_max=11,fs_min=9,color=C_NEU)
    foot(fig,ctx,k);return save_jpg(fig,path)

def p_all(ctx,k,path):
    P=ctx["P"];fig=plt.figure(figsize=(20,26),dpi=DPI,facecolor="white")
    top=head(fig,"ALL 17 OUTPUTS · ONE PROMPT","SAME MODEL ✓    SAME PROMPT ✓    SAME WEIGHTS ✓    GREEDY DECODING ✓    17 OUTPUTS")
    fig.text(.05,top-.004,"PROMPT",fontsize=12,weight="bold",color=C_NEU,va="top")
    fit_text(fig,.11,top-.027,.84,.025,P["prompt"],fs_max=15,fs_min=9,weight="bold")
    vy=top-.032-.075
    card(fig,.05,vy,.9,.072,"VANILLA · intervention off",P["vanilla"]["text"],C_NEU,fs_max=11,fs_min=6.5,sub=f"{P['vanilla']['words']} words",title_fs=12,sub_fs=9.5)
    avail=vy-.008-.02;gap=.005;rh=(avail-7*gap)/8
    for i,n in enumerate(AXIS_NAMES):
        y=vy-.008-(i+1)*rh-i*gap;r=P["axes"][n];nm=n.replace("_"," ")
        card(fig,.05,y,.444,rh,f"{nm}  ·  + {r['pos_label']}",r["pos"]["text"],C_A,fs_max=11,fs_min=6.5,sub=f"{r['pos']['words']} words",title_fs=11.5,sub_fs=9.5)
        card(fig,.506,y,.444,rh,f"{nm}  ·  − {r['neg_label']}",r["neg"]["text"],C_BT,fs_max=11,fs_min=6.5,sub=f"{r['neg']['words']} words",title_fs=11.5,sub_fs=9.5)
    foot(fig,ctx,k);return save_jpg(fig,path)

def p_seal(ctx,k,path):
    P=ctx["P"];T=P["timing"];I=P["integrity"];fig=plt.figure(figsize=(16,12),dpi=DPI,facecolor="white")
    top=head(fig,"RUN SEAL · TIMELINE & INTEGRITY","Every duration measured with perf_counter (GPU steps synchronized) · every value below is from this run")
    st=[("MODEL READY",T["startup"]["model_load_seconds"],"startup · once"),("8 AXES FORGED",T["startup"]["forge_seconds"],"startup · once"),
        ("VANILLA",T["run"]["vanilla_generation_seconds"],"run · greedy"),("16 STEERED RUNS",T["run"]["steered_generation_seconds_total"],"run · greedy"),
        ("28-LAYER X-RAY",T["run"]["xray_seconds_total"],f"{T['run']['xray_passes']} prefill passes"),("AUDIT + SEAL",ctx["sealing_stage_seconds"],"sealed "+ctx["sealed_utc"][11:23]+" UTC")]
    bw=.13;gx=(.9-6*bw)/5;y=top-.19;bh=.15
    for i,(nm,s,d) in enumerate(st):
        x=.05+i*(bw+gx);c=C_NEU if i<2 else (C_OFF if i==4 else C_ON)
        fig.add_artist(FancyBboxPatch((x,y),bw,bh,boxstyle="round,pad=0,rounding_size=0.008",transform=fig.transFigure,facecolor=C_BG,edgecolor=c,lw=2.2))
        fig.text(x+bw/2,y+bh*.8,nm,ha="center",va="center",fontsize=11,weight="bold",color=c)
        fig.text(x+bw/2,y+bh*.48,f"{s:.2f} s",ha="center",va="center",fontsize=20,weight="bold",color=C_FG)
        fig.text(x+bw/2,y+bh*.17,mt(d),ha="center",va="center",fontsize=9,color=C_NEU)
        if i<5:fig.text(x+bw+gx/2,y+bh/2,"→",ha="center",va="center",fontsize=16,weight="bold",color=C_FG)
    fit_text(fig,.05,y-.075,.9,.06,f"RUN START {P['run_start_utc']} UTC  →  PAYLOAD SEALED {ctx['sealed_utc']} UTC   ·   first generation → payload sealed on disk: {ctx['seal_wall_seconds']:.2f} s (wall clock)   ·   X-ray = separate prompt-prefill passes, interleaved with the generations",
             fs_max=11,fs_min=8.5)
    fpb=", ".join(f"{x:.4f}" for x in I["fingerprint_startup"]);fpa=", ".join(f"{x:.4f}" for x in I["fingerprint_after"])
    txt=(f"SELECTED-WEIGHT FINGERPRINT (float32 sums of 6 tensors)\n  startup: [{fpb}]\n  after  : [{fpa}]\n"
         f"SAMPLED SHA-256 WEIGHT SENTINEL (16×256 contiguous slices per tensor)\n  startup: {I['sentinel_startup']}\n  after  : {I['sentinel_after']}\n"
         f"RESULT: {I['result']} (both checks required)   trainable tensors: {I['trainable_parameter_tensors']}   training mode: {I['model_training_mode']}   AkbasCore hooks after run: {I['akbascore_hooks_after_run']}\n"
         f"optimizer: none · LoRA: none · fine-tuning: none · decoding: greedy · direct intervention: L0–L19 only · L20–L27 SEASC steering hooks: {I['tail_steering_hooks']}\n\n"
         f"PAYLOAD FILE    : {ctx['payload_name']}\nPAYLOAD SHA-256 : {ctx['sha']}\n\n"
         "This SHA-256 value is an artifact integrity seal: it lets anyone check that the downloaded payload is byte-identical to what this run recorded. "
         "It does not by itself establish that any scientific interpretation is correct. The weight sentinel samples selected tensors; it is not a full cryptographic verification of every weight.")
    fit_text(fig,.05,.06,.9,y-.16,txt,fs_max=12.5,fs_min=8.5,family=MONO)
    foot(fig,ctx,k);return save_jpg(fig,path)

POSTERS=[("01_hook","The hook",p_hook),("02_forge","Where the directions come from",p_forge),("03_compasses","20 compasses per axis",p_compass),
         ("04_injection","The injection",p_inject),("05_dose","The dose profile",p_dose),("06_xray","28-layer X-ray",p_xray),("07_matrix","L19 axis geometry",p_matrix)]
for _i,_n in enumerate(AXIS_NAMES):
    POSTERS.append((f"{8+_i:02d}_{_n.lower()}",f"Axis {_i+1}/8 · {_n.replace('_',' ')}",(lambda ctx,k,path,_n=_n,_i=_i:p_axis(ctx,k,path,_n,_i))))
POSTERS+=[("16_all17","All 17 outputs",p_all),("17_seal","Run seal",p_seal)]
if len(POSTERS)!=17 or len(POSTERS)>20:raise RuntimeError("Poster list must contain exactly 17 posters (≤ 20).")
if [s[:2] for s,_,_ in POSTERS]!=[f"{i:02d}" for i in range(1,18)]:raise RuntimeError("Poster numbering is not sequential 01–17.")

# ==================================================================================================
# LOGS · CANONICAL PAYLOAD · MANIFESTS
# ==================================================================================================
def gen_config():
    try:return jsafe(json.loads(json.dumps(model.generation_config.to_dict(),default=str)))
    except Exception as ex:return f"unavailable: {ex}"

def make_txt(P,sha,payload_name,manifest_name,seal):
    o=[];a=o.append;S="="*110;D="-"*110
    a(S);a("AKBASCORE · SEASC · KIZILELMA LIVE X-RAY · TEST 250 — READABLE RUN LOG");a(S)
    a(f"Derived from the sealed payload {payload_name} (SHA-256 {sha}). Verify with {manifest_name}.")
    a("The SHA-256 value is an artifact integrity seal; it does not by itself establish any scientific interpretation.")
    for k,v in [("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("PAYLOAD SEALED UTC",seal["sealed_utc"]),
                ("START LOCAL",P["run_start_local"]),("MODEL",MODEL_ID),("DTYPE / ATTENTION","bfloat16 / sdpa"),("LAYERS / HIDDEN",f"{TOTAL_LAYERS} / {H}"),
                ("GPU",P["environment"]["gpu"]),("TORCH",P["environment"]["torch"]),("TRANSFORMERS",P["environment"]["transformers"]),
                ("GRADIO",P["environment"]["gradio"]),("PYTHON",P["environment"]["python"].split()[0]),("SEED",SEED),
                ("SYSTEM",SYSTEM),("PROMPT",P["prompt"]),("DECODING",f"greedy (do_sample=False), max_new_tokens={MAX_NEW}, use_cache=True"),
                ("CORPUS SHA-256",P["engine_lock"]["corpus_sha256"])]:a(f"{k:<20}: {v}")
    I=P["integrity"];a("");a("INTEGRITY");a(D)
    a(f"fingerprint startup : {I['fingerprint_startup']}");a(f"fingerprint pre-run : {I['fingerprint_pre_run']}");a(f"fingerprint after   : {I['fingerprint_after']}")
    a(f"sampled sentinel startup : {I['sentinel_startup']}");a(f"sampled sentinel pre-run : {I['sentinel_pre_run']}");a(f"sampled sentinel after   : {I['sentinel_after']}")
    a(f"RESULT              : {I['result']}  | trainable tensors {I['trainable_parameter_tensors']} | training mode {I['model_training_mode']} | optimizer none | LoRA none | fine-tuning none | AkbasCore hooks after run {I['akbascore_hooks_after_run']}")
    a(f"counts              : generations {I['generations']} | X-ray passes {I['xray_passes']} | L20–L27 SEASC steering hooks {I['tail_steering_hooks']}")
    a(f"pre-seal checks     : {len(I['pre_seal_checks'])} PASS — "+"; ".join(I["pre_seal_checks"]))
    a(f"hook audit          : {I['hook_audit_method']}")
    fh=I["framework_hooks_after_run"];a(f"framework hooks     : {sum(len(v) for v in fh.values())} non-AkbasCore forward hooks on decoder layers (Transformers internals): {sorted({n for v in fh.values() for n in v})}")
    a("");a("SEASC DOSE (canonical)");a(D)
    a(f"IVME={IVME} SONUM={SONUM} ZIRVE={ZIRVE} TABAN={TABAN} RSS={RSS:.9f}")
    a(" ".join(f"L{L:02d}={RHO[L]*100:.3f}%" for L in range(STEER_LAYERS)))
    dc=P["seasc"]["dose_telemetry_check"];a(f"hook telemetry: {dc['calls']} injection calls, max |measured−ρ_L| = {dc['max_abs_deviation']:.3e}")
    a("Generation: direct intervention on L0–L19 at the final sequence position of every forward pass (prefill last prompt token + each decode token). L20–L27: no SEASC steering hook.")
    a("");a("X-RAY MEASUREMENT");a(D)
    for k,v in P["xray_measurement"].items():a(f"{k}: {v}")
    a("");a("FORGE (startup)");a(D)
    a(f"model load {P['startup']['model_load_seconds']:.3f}s · forge {P['startup']['forge_seconds']:.3f}s")
    for n in AXIS_NAMES:
        s=P["forge"]["stats"][n];a(f"{n:16s} raw‖Δ‖ L00={s['raw_norms'][0]:.4f} L19={s['raw_norms'][19]:.4f} adj cos mean={s['adjacent_cosine_mean']:+.4f} min={s['adjacent_cosine_min']:+.4f} max={s['adjacent_cosine_max']:+.4f} ({s['forge_seconds']:.3f}s)")
    a("");a("TIMING (run)");a(D)
    for k,v in P["timing"]["run"].items():a(f"{k:<36}: {v}")
    a(f"{'sealing_stage_seconds':<36}: {seal['sealing_stage_seconds']}   (integrity checks + canonical serialization + SHA-256 + payload write/verify)")
    a(f"{'first_generation_to_payload_sealed':<36}: {seal['seal_wall_seconds']}   (wall clock; recorded in the run manifest)")
    a("");a(S);a("VANILLA");a(D);v=P["vanilla"];a(v["text"])
    a(f"words={v['words']} sentences={v['sentences']} tokens={v['new_tokens']} gen={v['generation_seconds']:.4f}s [{v['generation_start_utc']} → {v['generation_end_utc']}] xray={v['xray_seconds']:.4f}s")
    a("hidden-state norms: "+" ".join(f"L{L:02d}={v['hidden_state_norms'][f'L{L:02d}']:.2f}" for L in range(TOTAL_LAYERS)))
    for n in AXIS_NAMES:
        r=P["axes"][n];a("");a(S);a(n);a(S)
        for s,lab in (("pos",f"+ {r['pos_label']}"),("neg",f"− {r['neg_label']}")):
            b=r[s];a(lab);a(D);a(b["text"])
            a(f"words={b['words']} sentences={b['sentences']} tokens={b['new_tokens']} gen={b['generation_seconds']:.4f}s [{b['generation_start_utc']} → {b['generation_end_utc']}] xray={b['xray_seconds']:.4f}s")
            a("X-RAY Δ% "+" ".join(f"L{L:02d}={b['xray'][f'L{L:02d}']*100:.3f}" for L in range(TOTAL_LAYERS)));a("")
        a("POS↔NEG SEP% "+" ".join(f"L{L:02d}={r['separation'][f'L{L:02d}']*100:.3f}" for L in range(TOTAL_LAYERS)))
    a("");a(S);a("CROSS-AXIS COSINE GEOMETRY, L00–L19");a(S)
    for L in range(STEER_LAYERS):
        M=P["geometry"]["cross_axis_cosine_by_layer"][f"L{L:02d}"];st=P["geometry"]["offdiag_stats_by_layer"][f"L{L:02d}"]
        a(f"L{L:02d}   off-diag mean {st['mean']:+.5f} · mean|cos| {st['mean_abs']:.5f} · max|cos| {st['max_abs']:.5f}")
        a(" "*18+" ".join(f"{x[:9]:>9s}" for x in AXIS_NAMES))
        for i,n in enumerate(AXIS_NAMES):a(f"{n[:16]:16s}  "+" ".join(f"{M[i][j]:+9.4f}" for j in range(8)))
        a("")
    a("NOTE: cosine geometry measures representational overlap; it does not by itself prove causal independence or absence of cross-talk.")
    a("NOTE: TEST 250 does not claim semantic success on all 8 axes; the raw outputs are provided for direct inspection.")
    a("NOTE: the weight sentinel samples selected tensors; it is not a full cryptographic verification of every weight.");a(S)
    return "\n".join(o)

def prune_runs(keep=2):
    runs=sorted([p for p in ROOT.glob("T250-*") if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)

def branch(text,tel,g,s,e,ntok,xr,xs):
    return {"text":text,"words":word_count(text),"sentences":sentence_count(text),"characters":len(text),"new_tokens":ntok,
            "generation_seconds":g,"generation_start_utc":s,"generation_end_utc":e,"xray_seconds":xs,"xray":xr,
            "dose_telemetry_relative":{f"L{L:02d}":tel[L] for L in range(STEER_LAYERS)},
            "hook_calls":{f"L{L:02d}":len(tel[L]) for L in range(STEER_LAYERS)},"hook_calls_L20_L27":0,
            "l00_post_injection_displacement_minus_rho0":xr["L00"]-RHO[0]}

def file_ready(p):return p is not None and Path(p).is_file() and Path(p).stat().st_size>0

def post_seal_audit(imgs,zp,expected_zip_names,pp,sha,tp):
    """Final checks on the delivered artifacts. Any failure raises RuntimeError."""
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise RuntimeError(f"POST-SEAL AUDIT FAILED: {name}")
    ok("17/17 JPG created",len(imgs)==17 and all(file_ready(p) for p,_ in imgs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p) as im:
            if not(im.format=="JPEG" and im.mode=="RGB"):bad.append(p.name)
    ok("17/17 JPG JPEG/RGB validation",not bad)
    with zipfile.ZipFile(zp) as z:
        ok("ZIP testzip()",z.testzip() is None)
        names=z.namelist()
        ok("ZIP flat (no sub-folders)",all("/" not in n for n in names))
        ok("ZIP expected contents (17 JPG + images manifest + run manifest)",sorted(names)==sorted(expected_zip_names))
        jp=[n for n in names if n.lower().endswith(".jpg")]
        ok("JPG numbering 01–17",[n[:2] for n in sorted(jp)]==[f"{i:02d}" for i in range(1,18)])
    ok("payload JSON exists",file_ready(pp))
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    ok("TXT exists and non-empty",file_ready(tp))
    ok("ZIP exists and non-empty",file_ready(zp))
    ok("no stale AkbasCore hooks",our_hooks_total()==0)
    return checks

# ==================================================================================================
# [7] ENGINE LOCK — verifies the locked TEST 249 / TEST 250 scientific configuration
# ==================================================================================================
print("[7/8] ENGINE LOCK")
ENGINE_LOCK_CHECKS=[
 ("MODEL_ID",MODEL_ID=="Qwen/Qwen2.5-7B-Instruct"),
 ("SYSTEM",SYSTEM=="You are a concise reasoning assistant. Use only the information in the prompt."),
 ("DEFAULT_PROMPT",DEFAULT_PROMPT=="Describe a person waiting alone at a train station late at night."),
 ("SEASC constants IVME/SONUM/ZIRVE/TABAN",(IVME,SONUM,ZIRVE,TABAN)==(.10,.30,.70,.20)),
 ("architecture 28 layers / hidden 3584 / steer 20",(TOTAL_LAYERS,STEER_LAYERS,H)==(28,20,3584)),
 ("MAX_NEW=128",MAX_NEW==128),
 ("RHO L00 = 10.000%",abs(RHO[0]-0.1)<1e-12),
 ("RHO L19 = 2.397%",round(RHO[19]*100,3)==2.397),
 ("RSS ≈ 0.250235055",abs(RSS-0.250235055)<1e-5),
 ("corpus 8 axes × (8 POS + 8 NEG)",len(AXES)==8 and all(len(c["pos"])==8 and len(c["neg"])==8 for c in AXES.values())),
 ("compass 8 × 20 layer-local unit vectors",all(len(COMPASS[n])==STEER_LAYERS and all(abs(float(v.norm())-1.0)<1e-4 and v.numel()==H for v in COMPASS[n]) for n in AXIS_NAMES)),
 ("BF16 weights",PDT==torch.bfloat16),
 ("SDPA attention",getattr(model.config,"_attn_implementation",None)=="sdpa"),
 ("hook boundary self-test",HOOK_SELFTEST_OK),
 ("frozen eval model",(not model.training) and trainable_tensors()==0 and not lora_present()),
 ("sampled weight sentinel",fingerprint()==FP0 and strong_sentinel()==SENTINEL0),
]
_fails=[n for n,okv in ENGINE_LOCK_CHECKS if not okv]
if _fails:raise RuntimeError(f"AKBASCORE ENGINE LOCK FAILED: {_fails}")
ENGINE_LOCK={"result":"PASS","checks":[n for n,_ in ENGINE_LOCK_CHECKS],"corpus_sha256":CORPUS_SHA,"rss":RSS}
print(f"Corpus SHA-256: {CORPUS_SHA}")
print("AKBASCORE ENGINE LOCK: PASS")

# ==================================================================================================
# [8] MOBILE-FIRST PUBLIC-LINK INTERFACE
# The run callback always yields a fixed 5-tuple matching OUTPUTS; unchanged outputs use gr.skip()
# (gr.update() on older Gradio). Exceptions are caught, fully printed to the Colab console, and shown
# as a readable status card instead of a broken event response.
# ==================================================================================================
print("[8/8] INTERFACE")
RUN_COUNTER=0;GPU_LOCK=threading.Lock()
SKIP=getattr(gr,"skip",None) or gr.update
_K=object()
N_OUTPUTS=5
def pack(status=_K,gal=_K,zipv=_K,jsonv=_K,txtv=_K):
    return tuple(SKIP() if v is _K else v for v in (status,gal,zipv,jsonv,txtv))

def card_html(title,body,kind="info"):
    return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def stage(i,title,body):return card_html(f"Step {i}/6 · {title}",body,"info")
READY_HTML=card_html("Ready","Press RUN. One run = 1 vanilla + 16 steered outputs, a 28-layer X-ray and 17 JPG posters (a few minutes).<br><b>Downloads appear after the run is complete.</b>","info")

def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower()
    return "cuda" in s or "device-side assert" in s or "cublas" in s

def run_handler(user_prompt):
    global RUN_COUNTER
    prompt=(user_prompt or "").strip() or DEFAULT_PROMPT
    if len(prompt)>MAX_PROMPT_CHARS:
        yield pack(card_html("Prompt too long",f"Please keep the prompt under {MAX_PROMPT_CHARS} characters.","err"));return
    if len(tok(chat(prompt),add_special_tokens=False).input_ids)+MAX_NEW>MAX_CTX:
        yield pack(card_html("Prompt too long","The prompt is too long for this demo.","err"));return
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1
        run_id=f"T250-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}"
        prune_runs(2);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        print(f"\n{'='*110}\nRUN {run_id} | prompt: {prompt[:100]}\n{'='*110}")
        checks=[]
        def chk(name,cond):
            checks.append(name)
            if not cond:raise RuntimeError(f"CHECK FAILED: {name}")

        stage_name="Step 1/6 integrity check"
        yield pack(stage(1,"Integrity check","Verifying the sampled weight sentinels before the run."),[],None,None,None)
        chk("no AkbasCore hooks before run",our_hooks_total()==0)
        chk("frozen eval model before run",(not model.training) and trainable_tensors()==0 and not lora_present())
        torch.cuda.synchronize();fp_pre=fingerprint();sent_pre=strong_sentinel()
        chk("pre-run weight fingerprint",fp_pre==FP0)
        chk("pre-run sampled SHA-256 sentinel",sent_pre==SENTINEL0)
        run_start_utc=utc_now();run_start_local=local_now()
        n_gen=0;n_xray=0
        torch.cuda.synchronize();T0=time.perf_counter()

        stage_name="Step 2/6 vanilla"
        yield pack(stage(2,"Vanilla","Generating the unsteered reference (greedy)."))
        vt,_,vg,vs,ve,vn=generate(prompt);n_gen+=1
        HV,hvs=xray(prompt);n_xray+=1
        vanilla={"text":vt,"words":word_count(vt),"sentences":sentence_count(vt),"characters":len(vt),"new_tokens":vn,
                 "generation_seconds":vg,"generation_start_utc":vs,"generation_end_utc":ve,"xray_seconds":hvs,
                 "hidden_state_norms":{f"L{L:02d}":float(HV[L].norm()) for L in range(TOTAL_LAYERS)},
                 "note":"Vanilla is the displacement reference for all 16 branches; no injection."}
        axes={};done=0
        stage_name="Step 3/6 steered branches"
        for n in AXIS_NAMES:
            cfg=AXES[n];rec={"pos_label":cfg["pos_label"],"neg_label":cfg["neg_label"]};states={}
            for s,sign in (("pos",1.0),("neg",-1.0)):
                done+=1
                yield pack(stage(3,f"Steered branches {done}/16",f"{html.escape(n.replace('_',' '))} · {'+' if sign>0 else '−'} {html.escape(cfg[s+'_label'])}"))
                t,tel,g,a0,a1,nt=generate(prompt,COMPASS[n],sign);n_gen+=1
                Hs,xs=xray(prompt,COMPASS[n],sign);n_xray+=1
                rec[s]=branch(t,tel,g,a0,a1,nt,displacement(HV,Hs),xs);states[s]=Hs
            rec["separation"]=separation(HV,states["pos"],states["neg"]);axes[n]=rec
            del states
        torch.cuda.synchronize();engine_s=time.perf_counter()-T0
        del HV

        stage_name="Step 4/6 sealing"
        yield pack(stage(4,"Sealing","Post-run weight checks, canonical payload, SHA-256 artifact integrity seal."))
        ts=time.perf_counter()
        fp_post=fingerprint();sent_post=strong_sentinel();ours=our_hooks_total();tail=tail_steering_hooks()
        torch.cuda.synchronize();integ_s=time.perf_counter()-ts
        chk("17/17 generation outputs",n_gen==N_GENERATIONS and isinstance(vt,str) and all(isinstance(axes[a][s]["text"],str) for a in AXIS_NAMES for s in ("pos","neg")))
        chk("17/17 X-Ray passes",n_xray==N_XRAY_PASSES)
        chk("weight fingerprint",fp_post==FP0)
        chk("sampled SHA-256 sentinel",sent_post==SENTINEL0)
        chk("AkbasCore hooks after run = 0",ours==0)
        chk("L20–L27 steering hooks = 0",tail==0)
        chk("model.training == False",not model.training)
        chk("trainable_parameter_tensors == 0",trainable_tensors()==0)
        chk("LoRA none",not lora_present())
        chk("optimizer none",not optimizer_present())
        devs=[abs(v-RHO[L]) for a in AXIS_NAMES for s in ("pos","neg") for L in range(STEER_LAYERS) for v in axes[a][s]["dose_telemetry_relative"][f"L{L:02d}"]]
        calls_ok=all(min(axes[a][s]["hook_calls"].values())>=1 and len(set(axes[a][s]["hook_calls"].values()))==1 for a in AXIS_NAMES for s in ("pos","neg"))
        chk("dose telemetry valid",bool(devs) and max(devs)<1e-4 and calls_ok)
        P={"schema":"akbascore.test250.run.v1","project":"AkbasCore","engine":"SEASC","demonstrator":"KIZILELMA LIVE X-RAY","test":"TEST 250",
           "hierarchy":["AkbasCore","SEASC","KIZILELMA LIVE X-RAY","TEST 250"],
           "run_id":run_id,"run_start_utc":run_start_utc,"run_start_local":run_start_local,"run_end_utc":utc_now(),"run_end_local":local_now(),
           "prompt":prompt,"system":SYSTEM,
           "model":{"id":MODEL_ID,"dtype":"bfloat16","attn_implementation":"sdpa","layers":TOTAL_LAYERS,"hidden_size":H},
           "environment":{"gpu":torch.cuda.get_device_name(0),"cuda":torch.version.cuda,"torch":torch.__version__,"transformers":transformers.__version__,
                          "gradio":gr.__version__,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"python":sys.version,"platform":platform.platform()},
           "engine_lock":ENGINE_LOCK,
           "decoding":{"mode":"greedy","do_sample":False,"max_new_tokens":MAX_NEW,"use_cache":True,"pad_token_id":tok.eos_token_id,"eos_token_id":tok.eos_token_id,
                       "model_generation_config":gen_config(),"note":"Identical generate() call for all 17 outputs, as in TEST 249."},
           "seasc":{"IVME":IVME,"SONUM":SONUM,"ZIRVE":ZIRVE,"TABAN":TABAN,"rho":RHO,"rho_percent":[r*100 for r in RHO],"rss":RSS,
                    "envelope":"E(L)=[ZIRVE*exp(-SONUM*L)*(1+SONUM*L)+TABAN]/(ZIRVE+TABAN); rho_L=IVME*E(L)",
                    "injection":"h'_L = h_L + sign*rho_L*||h_L||*A_L at the final sequence position of each forward pass (last prompt token at prefill, each new token during decoding)",
                    "steered_layers":list(range(STEER_LAYERS)),"observation_only_layers":list(range(STEER_LAYERS,TOTAL_LAYERS)),
                    "dose_telemetry_check":{"calls":len(devs),"max_abs_deviation":max(devs) if devs else None,"hook_calls_consistent_across_layers":calls_ok}},
           "startup":{"startup_utc":STARTUP_UTC,"model_load_seconds":MODEL_LOAD_S,"model_ready_utc":MODEL_READY_UTC,"forge_seconds":FORGE_TIME_S,"forge_done_utc":FORGE_DONE_UTC,
                      "hook_boundary_self_test":"PASS (tagged SEASC steering hooks: L0–L19 = 1 each, L20–L27 = 0)",
                      "framework_hooks_after_forge":FRAMEWORK_HOOKS_STARTUP},
           "forge":{"examples":AXES,"capture":"hidden_states[L+1][0,-1] = output of decoder layer L, last token of the chat-formatted example",
                    "compass":"A_L = normalize(mean(POS_L) - mean(NEG_L)), L0-L19","stats":FORGE_STATS,"layer_to_layer_cosine":LAYER_SIM},
           "geometry":{"axis_order":AXIS_NAMES,"cross_axis_cosine_by_layer":GEOMETRY,"offdiag_stats_by_layer":GEO_STATS,
                       "note":"Cosine overlap describes representational geometry; it does not by itself prove causal independence or absence of cross-talk."},
           "xray_measurement":{"pass":"separate prompt-prefill forward pass, use_cache=False, chat-formatted prompt (not the generation trajectory)",
                               "position":"last prompt token","layers":"output of decoder layers L0-L27",
                               "generation_vs_xray":"Generation intervenes at the prefill last prompt token and at every autoregressive decode token; the X-ray measures only a separate prompt-prefill pass.",
                               "displacement":"||h_branch - h_vanilla|| / ||h_vanilla||","separation":"||h_pos - h_neg|| / ||h_vanilla||",
                               "repair":"POST-INTERVENTION X-RAY MEASUREMENT REPAIR: the TEST 249 / TEST 250-draft hidden() registered observation hooks before steering hooks, so a steered layer's stored state excluded its own injection. TEST 250 registers steering hooks first, so L0-L19 states are captured after that layer's own injection. Measurement only; generate() does not use this function and generation behaviour is unchanged."},
           "vanilla":vanilla,"axes":axes,
           "timing":{"startup":{"model_load_seconds":MODEL_LOAD_S,"forge_seconds":FORGE_TIME_S},
                     "run":{"vanilla_generation_seconds":vg,
                            "steered_generation_seconds_total":sum(axes[a][s]["generation_seconds"] for a in AXIS_NAMES for s in ("pos","neg")),
                            "xray_seconds_total":hvs+sum(axes[a][s]["xray_seconds"] for a in AXIS_NAMES for s in ("pos","neg")),"xray_passes":n_xray,
                            "engine_seconds":engine_s,"post_run_integrity_seconds":integ_s,"engine_plus_integrity_seconds":engine_s+integ_s,
                            "note":"Wall-clock time from the first generation until the payload is sealed on disk is recorded in the run manifest (a payload cannot contain its own sealing time)."}},
           "integrity":{"result":"PASS","fingerprint_tensors":FP_NAMES,"fingerprint_startup":list(FP0),"fingerprint_pre_run":list(fp_pre),"fingerprint_after":list(fp_post),
                        "sentinel_method":"sampled SHA-256 over 16 contiguous 256-value slices per selected tensor (basic slicing, Python-int offsets), cast float32 on CPU; not a full verification of every weight",
                        "sentinel_startup":SENTINEL0,"sentinel_pre_run":sent_pre,"sentinel_after":sent_post,
                        "trainable_parameter_tensors":trainable_tensors(),"model_training_mode":bool(model.training),
                        "akbascore_hooks_after_run":int(ours),"tail_steering_hooks":int(tail),
                        "hook_audit_method":"identity-tagged AkbasCore hooks (_akbascore attribute); framework hooks recorded separately",
                        "framework_hooks_after_run":framework_hooks(),"pre_seal_checks":list(checks),
                        "generations":n_gen,"xray_passes":n_xray,"optimizer":None,"lora":False,"fine_tuning":False},
           "reproduction":{"seed":SEED,"cell_source_sha256":CELL_SOURCE_SHA,"cell_source":CELL_SOURCE}}
        P=jsafe(P);pb=canon(P);sha=hashlib.sha256(pb).hexdigest()
        payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";manifest_name=f"run_manifest_{run_id}.json"
        pp=run_dir/payload_name;pp.write_bytes(pb)
        if hashlib.sha256(pp.read_bytes()).hexdigest()!=sha:raise RuntimeError("Payload hash verification failed.")
        t_sealed=time.perf_counter();sealed_utc=utc_now()
        seal={"sealed_utc":sealed_utc,"sealing_stage_seconds":t_sealed-ts,"seal_wall_seconds":t_sealed-T0}
        manifest={"test":"TEST 250","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"payload_bytes":len(pb),"generated_utc":sealed_utc,
                  "timing":{"first_generation_to_payload_sealed_seconds":seal["seal_wall_seconds"],
                            "sealing_stage_seconds":seal["sealing_stage_seconds"],
                            "sealing_stage_includes":"post-run integrity checks + canonical serialization + SHA-256 + payload write and re-read verification"},
                  "hash_algorithm":"SHA-256","canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
                  "note":"Artifact integrity seal: confirms the payload has not changed after sealing. It does not by itself establish any scientific interpretation."}
        mp=run_dir/manifest_name;mp.write_bytes(json.dumps(manifest,sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
        tp=run_dir/f"{run_id}_READABLE_RUN_LOG.txt";tp.write_text(make_txt(P,sha,payload_name,manifest_name,seal),encoding="utf-8")
        print(f"payload sealed: {sha} | first generation → sealed {seal['seal_wall_seconds']:.2f}s")

        stage_name="Step 5/6 rendering 17 JPGs"
        yield pack(stage(5,"Rendering 17 JPGs","Drawing the poster story from this run's measured values."))
        ctx={"P":P,"run_id":run_id,"sha":sha,"N":len(POSTERS),"payload_name":payload_name,**seal}
        rt=time.perf_counter();imgs=[]
        try:
            for i,(slug,cap,fn) in enumerate(POSTERS,1):
                p=run_dir/f"{slug}_{run_id}.jpg";fn(ctx,i,p);imgs.append((p,cap))
        finally:
            plt.close("all")
        render_s=time.perf_counter()-rt
        entries=[]
        for i,(p,cap) in enumerate(imgs,1):
            with Image.open(p) as im:w,h=im.size;fmt=im.format;mode=im.mode
            entries.append({"index":i,"filename":p.name,"title":cap,"width":w,"height":h,"format":fmt,"mode":mode,
                            "size_bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"run_id":run_id})
        imf=run_dir/f"images_manifest_{run_id}.json"
        imf.write_bytes(json.dumps({"test":"TEST 250","run_id":run_id,"payload_file":payload_name,"payload_sha256":sha,"generated_utc":utc_now(),
                                    "render_seconds":render_s,"count":len(entries),"images":entries,
                                    "note":"SHA-256 per image is an artifact integrity seal for each file."},sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
        zp=run_dir/f"AKBASCORE_KIZILELMA_TEST250_JPGs_{run_id}.zip"
        expected=[p.name for p,_ in imgs]+[imf.name,mp.name]
        with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for p,_ in imgs:z.write(p,arcname=p.name)
            z.write(imf,arcname=imf.name);z.write(mp,arcname=mp.name)

        stage_name="Step 6/6 final audit"
        audit=post_seal_audit(imgs,zp,expected,pp,sha,tp)
        total=len(checks)+len(audit)
        print(f"{run_id}: {len(checks)} pre-seal + {len(audit)} post-seal checks = {total}/{total} PASS | render {render_s:.1f}s")
        for c in checks+audit:print("  PASS ·",c)
        gallery_items=[(str(p),c) for p,c in imgs]
        body=(f"<b>{html.escape(run_id)}</b><br>17 outputs · 17 JPGs · {total}/{total} checks PASS<br>"
              f"first generation → payload sealed: {seal['seal_wall_seconds']:.1f} s<br>"
              f'<span class="mono">payload SHA-256 (artifact integrity seal): {sha}</span>')
        yield pack(card_html("Step 6/6 · Complete",body,"on"),gallery_items,str(zp),str(pp),str(tp))
    except Exception as ex:
        print("="*110)
        print(f"TEST 250 RUN FAILED — stage: {stage_name}")
        print(f"exception type   : {type(ex).__name__}")
        print(f"exception message: {ex}")
        traceback.print_exc()
        print("="*110)
        try:purged=purge_akbascore_hooks()
        except Exception:purged=-1
        print(f"AkbasCore hooks purged after failure: {purged}")
        tip="<br><b>CUDA error: Restart runtime before retrying.</b>" if is_cuda_error(ex) else "<br>The full traceback is printed in the Colab console."
        yield pack(card_html("Run failed",f"Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err"),[],None,None,None)
    finally:
        plt.close("all")
        try:
            if our_hooks_total()!=0:print("Cleanup: removing leftover AkbasCore hooks:",purge_akbascore_hooks())
        except Exception:pass
        GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass

CSS="""
:root{--kz-on:#047857;--kz-fg:#0f172a;--kz-card:#ffffff;--kz-bd:#cbd5e1;--kz-a:#0369a1;--kz-off:#6d28d9;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#34d399;--kz-fg:#f8fafc;--kz-card:#0f172a;--kz-bd:#475569;--kz-a:#38bdf8;--kz-off:#c4b5fd;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:860px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{text-align:center;padding:10px 2px 2px}
.brand{font-size:clamp(26px,8vw,40px);font-weight:800;letter-spacing:1px;line-height:1.05}
.kick{font-size:13px;font-weight:700;letter-spacing:2px;color:var(--kz-on)!important;margin-top:4px}
.ttl{font-size:clamp(19px,5.5vw,26px);font-weight:800;line-height:1.1;margin-top:2px}
.tst{font-size:15px;font-weight:700;margin-top:2px}
.lead{text-align:center;font-size:16px;margin:2px 0 4px}
.card{background:var(--kz-card);border:2px solid var(--kz-bd);border-left-width:6px;border-radius:12px;padding:10px 12px}
.card.on{border-left-color:var(--kz-on)}.card.info{border-left-color:var(--kz-a)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}
.card .h{font-size:17px;font-weight:800;margin-bottom:2px}.small{font-size:15px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}
#pbox textarea{font-size:17px!important;line-height:1.45!important}
#runbtn{font-size:20px!important;min-height:58px!important}
"""
HERO=('<div class="kz hero"><div class="brand">AKBASCORE</div><div class="kick">SEASC</div>'
      '<div class="ttl">KIZILELMA LIVE X-RAY</div><div class="tst">TEST 250</div></div>')
GR6=int(gr.__version__.split(".")[0])>=6   # Gradio 6 moved css= from Blocks() to launch()

with gr.Blocks(title="AKBASCORE · KIZILELMA LIVE X-RAY · TEST 250",**({} if GR6 else {"css":CSS})) as demo:
    gr.HTML(HERO)
    gr.HTML('<div class="kz lead">Run the live example — or write anything you want.</div>')
    prompt_box=gr.Textbox(value=DEFAULT_PROMPT,show_label=False,lines=3,max_lines=6,elem_id="pbox")
    run_btn=gr.Button("▶ RUN",variant="primary",size="lg",elem_id="runbtn")
    status_h=gr.HTML(READY_HTML)
    gallery=gr.Gallery(label="17 JPG posters",columns=2,height="auto",object_fit="contain")
    zip_f=gr.File(label="DOWNLOAD ALL JPGs (.zip)",interactive=False)
    json_f=gr.File(label="DOWNLOAD FULL RUN LOG (.json)",interactive=False)
    txt_f=gr.File(label="DOWNLOAD READABLE RUN LOG (.txt)",interactive=False)
    OUTPUTS=[status_h,gallery,zip_f,json_f,txt_f]
    run_btn.click(run_handler,[prompt_box],OUTPUTS,concurrency_limit=1)

# ==================================================================================================
# SERVICE SELF-TEST — no scientific computation; verifies the serving / file / callback contract
# ==================================================================================================
def service_self_test():
    st=ROOT/"_service_selftest";shutil.rmtree(st,ignore_errors=True);st.mkdir(parents=True)
    res=[]
    def ok(name,cond):
        res.append(name)
        if not cond:raise RuntimeError(f"TEST 250 SERVICE SELF-TEST FAILED: {name}")
    try:
        t=st/"probe.txt";t.write_text("akbascore",encoding="utf-8")
        ok("ROOT writable / text file",t.read_text(encoding="utf-8")=="akbascore")
        fig=plt.figure(figsize=(4,2),dpi=DPI,facecolor="white");fit_text(fig,.05,.1,.9,.8,"AKBASCORE self-test $5 · ρ_L ‖h‖",fs_max=14,fs_min=7)
        jp=st/"01_probe.jpg";save_jpg(fig,jp)
        with Image.open(jp) as im:ok("JPEG pipeline (JPEG/RGB)",im.format=="JPEG" and im.mode=="RGB")
        js=canon(jsafe({"x":1.5,"nan":float("nan"),"t":(1,2),"s":"ü"}))
        ok("JSON canonical serialization",json.loads(js.decode("utf-8"))["nan"]=="non-finite:nan")
        zp=st/"probe.zip"
        with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED) as z:z.write(jp,arcname=jp.name);z.write(t,arcname=t.name)
        with zipfile.ZipFile(zp) as z:ok("ZIP create + testzip()",z.testzip() is None and sorted(z.namelist())==["01_probe.jpg","probe.txt"])
        ok("allowed_paths covers ROOT",any(str(ROOT.resolve()).startswith(str(Path(a).resolve())) for a in ALLOWED_PATHS))
        ok("poster list = 17",len(POSTERS)==17)
        ok("output components = callback outputs",len(OUTPUTS)==N_OUTPUTS==len(pack()))
        ok("run_handler is a generator function",inspect.isgeneratorfunction(run_handler))
        ok("no AkbasCore hooks active",our_hooks_total()==0)
        ok(f"Gradio version logged ({gr.__version__})",bool(gr.__version__))
    finally:
        plt.close("all");shutil.rmtree(st,ignore_errors=True)
    return res
_svc=service_self_test()
for _n in _svc:print("  PASS ·",_n)
print("TEST 250 SERVICE SELF-TEST: PASS")
print("AKBASCORE ENGINE LOCK: PASS")

# ==================================================================================================
# LAUNCH — public gradio.live link only (Colab inline iframe disabled). debug=True keeps the cell alive
# and prints server-side errors; a watcher thread prints the public URL once it exists.
# ==================================================================================================
_lp=inspect.signature(gr.Blocks.launch).parameters
LAUNCH_KW={"share":True,"debug":True,"inline":False,"show_error":True,"allowed_paths":ALLOWED_PATHS,"ssr_mode":False}
if GR6:LAUNCH_KW["css"]=CSS
LAUNCH_KW={k:v for k,v in LAUNCH_KW.items() if k in _lp}
if not LAUNCH_KW.get("share"):raise RuntimeError("This Gradio version does not accept share=True.")
if "inline" in _lp and LAUNCH_KW.get("inline") is not False:raise RuntimeError("Inline mode could not be disabled.")

def _announce_public_url():
    for _ in range(720):
        u=getattr(demo,"share_url",None)
        if u:
            print("\n"+"="*110)
            print("GRADIO PUBLIC-LINK MODE: PASS")
            print("READY")
            print("="*110)
            print("OPEN THIS LINK IN A NEW BROWSER TAB:")
            print(u)
            print("="*110)
            print("Do not use the Colab inline preview. Keep this cell running while you use the link.\n")
            return
        time.sleep(.5)
    print("GRADIO PUBLIC-LINK MODE: FAIL — no public link after 6 minutes. Check the Gradio messages above, then rerun the cell.")
threading.Thread(target=_announce_public_url,daemon=True).start()

print("Launching public Gradio link … (launch options:",{k:(v if k!="css" else "…") for k,v in LAUNCH_KW.items()},")")
demo.queue(default_concurrency_limit=1).launch(**LAUNCH_KW)
