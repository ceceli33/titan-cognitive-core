# ==================================================================================================
# TEST 249 — 8-AXIS BEHAVIORAL CONTROL MATRIX
# WORKING BASELINE: TEST248
# ONE FIXED NEUTRAL PROMPT | 8 MODEL-ENDOGENOUS BEHAVIOR AXES
# EACH AXIS: 8 POS + 8 NEG -> LAYER-LOCAL COMPASS -> CANONICAL SEASC L0-L19
# VANILLA / POS / NEG | GREEDY | SAME PROMPT | SAME WEIGHTS | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,subprocess,importlib.util,random,math,re
for m,p in [("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])

import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")

SEED=249
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda")

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
PROMPT="Describe a person waiting alone at a train station late at night."

TOTAL_LAYERS=28
STEER_LAYERS=20
H_EXPECT=3584
IVME=.10
SONUM=.30
ZIRVE=.70
TABAN=.20
MAX_NEW=128
EPS=1e-8

AXES={
"EMOTIONAL_TONE":{
"pos_label":"JOYFUL / WARM / OPTIMISTIC",
"neg_label":"SAD / COLD / PESSIMISTIC",
"pos":[
"The atmosphere feels joyful, warm, and optimistic.",
"The scene carries happiness, warmth, and hope.",
"The emotional tone is cheerful and encouraging.",
"Everything feels bright, friendly, and positive.",
"The mood is uplifting, affectionate, and hopeful.",
"The situation is described with warmth and happiness.",
"The emotional perspective is positive and reassuring.",
"The overall feeling is joyful, welcoming, and optimistic."
],
"neg":[
"The atmosphere feels sad, cold, and pessimistic.",
"The scene carries sorrow, distance, and hopelessness.",
"The emotional tone is gloomy and discouraging.",
"Everything feels bleak, unfriendly, and negative.",
"The mood is depressing, detached, and hopeless.",
"The situation is described with coldness and sadness.",
"The emotional perspective is negative and discouraging.",
"The overall feeling is sad, distant, and pessimistic."
]},
"CONFIDENCE":{
"pos_label":"CONFIDENT / CERTAIN / ASSERTIVE",
"neg_label":"UNCERTAIN / HESITANT / DOUBTFUL",
"pos":[
"The answer is confident, certain, and assertive.",
"The speaker expresses the conclusion with strong certainty.",
"The statement is direct and delivered without hesitation.",
"The response communicates confidence and conviction.",
"The speaker sounds decisive and sure of the conclusion.",
"The language is assertive, definite, and confident.",
"The answer presents its claims with clear certainty.",
"The overall expression is decisive and self-assured."
],
"neg":[
"The answer is uncertain, hesitant, and doubtful.",
"The speaker expresses the conclusion with considerable uncertainty.",
"The statement is tentative and delivered with hesitation.",
"The response communicates doubt and lack of confidence.",
"The speaker sounds indecisive and unsure of the conclusion.",
"The language is tentative, qualified, and uncertain.",
"The answer presents its claims with visible doubt.",
"The overall expression is hesitant and unsure."
]},
"CALMNESS":{
"pos_label":"CALM / PEACEFUL / RELAXED",
"neg_label":"TENSE / ANXIOUS / AGITATED",
"pos":[
"The atmosphere is calm, peaceful, and relaxed.",
"The scene feels tranquil and completely unhurried.",
"The mood is serene, composed, and restful.",
"Everything unfolds with quietness and calm.",
"The emotional state is peaceful and free of tension.",
"The situation feels safe, still, and relaxing.",
"The person remains composed, comfortable, and calm.",
"The overall feeling is tranquil, steady, and peaceful."
],
"neg":[
"The atmosphere is tense, anxious, and agitated.",
"The scene feels nervous and intensely unsettled.",
"The mood is strained, worried, and restless.",
"Everything unfolds with tension and anxiety.",
"The emotional state is nervous and full of tension.",
"The situation feels uneasy, unstable, and stressful.",
"The person remains worried, restless, and agitated.",
"The overall feeling is tense, unstable, and anxious."
]},
"FORMALITY":{
"pos_label":"FORMAL / PROFESSIONAL",
"neg_label":"CASUAL / INFORMAL",
"pos":[
"The response uses formal and professional language.",
"The description is written in a polished professional style.",
"The wording is precise, formal, and appropriately structured.",
"The speaker communicates in a professional register.",
"The language is refined and formally composed.",
"The answer maintains a serious professional tone.",
"The response uses structured and professional phrasing.",
"The overall style is formal, polished, and professional."
],
"neg":[
"The response uses casual and informal language.",
"The description is written in a relaxed conversational style.",
"The wording is easygoing, informal, and loosely structured.",
"The speaker communicates in a casual register.",
"The language is relaxed and conversational.",
"The answer maintains an easygoing informal tone.",
"The response uses everyday and casual phrasing.",
"The overall style is informal, relaxed, and conversational."
]},
"VERBOSITY":{
"pos_label":"DETAILED / ELABORATE",
"neg_label":"CONCISE / BRIEF",
"pos":[
"The response is detailed, elaborate, and comprehensive.",
"The answer provides extensive description and supporting detail.",
"The explanation develops the scene at considerable length.",
"The response expands on small details and contextual information.",
"The description is thorough, rich, and highly elaborated.",
"The answer explores the subject with substantial detail.",
"The response provides a long and comprehensive description.",
"The overall explanation is expansive and richly detailed."
],
"neg":[
"The response is concise, brief, and compact.",
"The answer provides only the essential information.",
"The explanation describes the scene in very few words.",
"The response avoids unnecessary detail and context.",
"The description is short, direct, and economical.",
"The answer addresses the subject with minimal detail.",
"The response provides a brief and compact description.",
"The overall explanation is concise and to the point."
]},
"POLITENESS":{
"pos_label":"POLITE / RESPECTFUL",
"neg_label":"RUDE / DISMISSIVE",
"pos":[
"The speaker is polite, considerate, and respectful.",
"The response communicates with courtesy and respect.",
"The wording is gracious, tactful, and considerate.",
"The speaker maintains a respectful social tone.",
"The language is courteous and thoughtfully phrased.",
"The answer treats the listener with clear respect.",
"The response sounds considerate, civil, and polite.",
"The overall social tone is respectful and courteous."
],
"neg":[
"The speaker is rude, dismissive, and disrespectful.",
"The response communicates with impatience and disregard.",
"The wording is blunt, contemptuous, and inconsiderate.",
"The speaker maintains a dismissive social tone.",
"The language is discourteous and harshly phrased.",
"The answer treats the listener with little respect.",
"The response sounds inconsiderate, abrasive, and rude.",
"The overall social tone is dismissive and disrespectful."
]},
"CREATIVITY":{
"pos_label":"IMAGINATIVE / POETIC",
"neg_label":"LITERAL / PLAIN",
"pos":[
"The description is imaginative, poetic, and evocative.",
"The response uses vivid imagery and creative expression.",
"The language transforms the scene through metaphor and imagination.",
"The description is artistic, expressive, and richly figurative.",
"The answer uses poetic imagery and inventive phrasing.",
"The scene is presented through creative and evocative language.",
"The response favors imagination, metaphor, and vivid description.",
"The overall style is poetic, imaginative, and expressive."
],
"neg":[
"The description is literal, plain, and straightforward.",
"The response uses simple factual language without imagery.",
"The language describes the scene directly without metaphor.",
"The description is practical, ordinary, and purely literal.",
"The answer uses plain wording and avoids figurative expression.",
"The scene is presented through direct and factual language.",
"The response favors literal statements and simple description.",
"The overall style is plain, straightforward, and factual."
]},
"ENTHUSIASM":{
"pos_label":"ENTHUSIASTIC / ENERGETIC",
"neg_label":"DETACHED / INDIFFERENT",
"pos":[
"The speaker sounds enthusiastic, energetic, and engaged.",
"The response conveys excitement and strong interest.",
"The language feels lively, animated, and enthusiastic.",
"The speaker approaches the subject with energetic engagement.",
"The answer communicates excitement and positive energy.",
"The response sounds animated and highly interested.",
"The wording conveys enthusiasm, vitality, and engagement.",
"The overall delivery is energetic and enthusiastic."
],
"neg":[
"The speaker sounds detached, indifferent, and disengaged.",
"The response conveys little excitement or interest.",
"The language feels flat, distant, and indifferent.",
"The speaker approaches the subject without emotional engagement.",
"The answer communicates detachment and low energy.",
"The response sounds uninterested and emotionally distant.",
"The wording conveys indifference, passivity, and disengagement.",
"The overall delivery is detached and unenthusiastic."
]}
}

print("="*124)
print("TEST 249 — 8-AXIS BEHAVIORAL CONTROL MATRIX")
print("="*124)
print("Baseline :",PROMPT)
print("Axes     :",len(AXES))
print("Forge    : 8 POS - 8 NEG model-endogenous layer-local mean contrast")
print("Motor    : canonical SEASC L0-L19 | L20-L27 observation only")
print()

_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"

print("[1/10] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}
)
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers
H=model.config.hidden_size
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
    x=ZIRVE*math.exp(-SONUM*L)*(1.0+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=[IVME*envelope(L) for L in range(STEER_LAYERS)]
def remove(hs):
    for h in hs:h.remove()
def word_count(x):return len(re.findall(r"\b[\w'-]+\b",x))
def sentence_count(x):
    n=len(re.findall(r"[.!?]+",x))
    return max(1,n) if x.strip() else 0

print("[2/10] Canonical dose lock...")
print(" ".join(f"L{L:02d}={RHO[L]*100:.3f}%" for L in range(STEER_LAYERS)))
print(f"RSS={math.sqrt(sum(x*x for x in RHO)):.9f}")

print("[3/10] Extract 8×(8 POS + 8 NEG) layer-local states...")
@torch.inference_mode()
def capture(text):
    e=enc(text)
    out=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [out.hidden_states[L+1][0,-1].float().detach().clone() for L in range(STEER_LAYERS)]

COMPASS={}
FORGE_STATS={}

for name,cfg in AXES.items():
    POS=[capture(x) for x in cfg["pos"]]
    NEG=[capture(x) for x in cfg["neg"]]
    vec=[]
    raw=[]
    for L in range(STEER_LAYERS):
        p=torch.stack([x[L] for x in POS]).mean(0)
        n=torch.stack([x[L] for x in NEG]).mean(0)
        d=p-n
        raw.append(float(d.norm()))
        vec.append(unit(d))
    COMPASS[name]=vec
    adj=[cosine(vec[L],vec[L+1]) for L in range(STEER_LAYERS-1)]
    FORGE_STATS[name]={
        "raw":raw,
        "adj_mean":float(np.mean(adj)),
        "adj_min":float(np.min(adj)),
        "adj_max":float(np.max(adj))
    }
    del POS,NEG
    torch.cuda.empty_cache()

print("[4/10] Forge geometry...")
for name in AXES:
    s=FORGE_STATS[name]
    print(f"{name:16s} rawL00={s['raw'][0]:8.3f} rawL19={s['raw'][19]:8.3f} "
          f"adjCos={s['adj_mean']:+.4f} [{s['adj_min']:+.4f},{s['adj_max']:+.4f}]")

print()
print("L19 cross-axis cosine:")
names=list(AXES.keys())
for i,name in enumerate(names):
    vals=[]
    for j,other in enumerate(names):
        if i==j:continue
        vals.append(abs(cosine(COMPASS[name][19],COMPASS[other][19])))
    print(f"{name:16s} mean|cos|={np.mean(vals):.4f} max|cos|={np.max(vals):.4f}")

print("[5/10] Steering engine...")
def make_hooks(vectors,sign=1.0,telemetry=None):
    hs=[]
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
            return hk
        hs.append(layers[L].register_forward_hook(factory(L)))
    return hs

@torch.inference_mode()
def generate(question,vectors=None,sign=1.0):
    e=enc(question)
    n=e.input_ids.shape[1]
    tel={L:[] for L in range(STEER_LAYERS)}
    hs=make_hooks(vectors,sign,tel) if vectors is not None else []
    try:
        out=model.generate(
            **e,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )
    finally:
        remove(hs)
    text=tok.decode(out[0,n:],skip_special_tokens=True).strip()
    return text,tel

print("[6/10] X-ray engine...")
@torch.inference_mode()
def hidden(question,vectors=None,sign=1.0):
    e=enc(question)
    store={}
    obs=[]
    for L in range(TOTAL_LAYERS):
        def factory(li):
            def hk(module,args,out):
                x=out[0] if isinstance(out,tuple) else out
                store[li]=x[0,-1].float().detach().clone()
            return hk
        obs.append(layers[L].register_forward_hook(factory(L)))
    steer=make_hooks(vectors,sign,None) if vectors is not None else []
    try:
        model(**e,use_cache=False,return_dict=True)
    finally:
        remove(steer);remove(obs)
    return store

print("[7/10] Vanilla baseline...")
VANILLA,_=generate(PROMPT)
HV=hidden(PROMPT)
print()
print("VANILLA:")
print(VANILLA)
print(f"Words={word_count(VANILLA)} Sentences={sentence_count(VANILLA)}")

print("[8/10] Run all 8 behavioral axes...")
RESULTS={}

for idx,(name,cfg) in enumerate(AXES.items(),1):
    vec=COMPASS[name]
    pos,tpos=generate(PROMPT,vec,+1.0)
    neg,tneg=generate(PROMPT,vec,-1.0)

    hp=hidden(PROMPT,vec,+1.0)
    hn=hidden(PROMPT,vec,-1.0)

    relP={}
    relN={}
    sep={}
    for L in [4,8,12,16,19,20,24,27]:
        relP[L]=float((hp[L]-HV[L]).norm()/HV[L].norm().clamp_min(EPS))
        relN[L]=float((hn[L]-HV[L]).norm()/HV[L].norm().clamp_min(EPS))
        sep[L]=float((hp[L]-hn[L]).norm()/HV[L].norm().clamp_min(EPS))

    RESULTS[name]={
        "pos":pos,"neg":neg,
        "pos_words":word_count(pos),"neg_words":word_count(neg),
        "pos_sent":sentence_count(pos),"neg_sent":sentence_count(neg),
        "relP":relP,"relN":relN,"sep":sep
    }

    print()
    print("="*124)
    print(f"[{idx}/8] {name}")
    print(f"POS: {cfg['pos_label']}")
    print(f"NEG: {cfg['neg_label']}")
    print("-"*124)
    print("VANILLA:")
    print(VANILLA)
    print()
    print("POS:")
    print(pos)
    print()
    print("NEG:")
    print(neg)
    print()
    print(f"WORDS V/P/N = {word_count(VANILLA)}/{word_count(pos)}/{word_count(neg)}")
    print(f"SENTS V/P/N = {sentence_count(VANILLA)}/{sentence_count(pos)}/{sentence_count(neg)}")
    print("POS X-RAY :"," ".join(f"L{L:02d}={relP[L]*100:.2f}%" for L in relP))
    print("NEG X-RAY :"," ".join(f"L{L:02d}={relN[L]*100:.2f}%" for L in relN))
    print("POS↔NEG   :"," ".join(f"L{L:02d}={sep[L]*100:.2f}%" for L in sep))

print("[9/10] Automatic physical + measurable audit...")
print("="*124)
print("AXIS SUMMARY")
print("="*124)
print(f"{'AXIS':16s} {'P-WORD':>7s} {'N-WORD':>7s} {'P-L19':>8s} {'N-L19':>8s} {'P-L27':>8s} {'N-L27':>8s} {'SEP27':>8s}")
for name in names:
    r=RESULTS[name]
    print(
        f"{name:16s} "
        f"{r['pos_words']:7d} {r['neg_words']:7d} "
        f"{r['relP'][19]*100:7.2f}% {r['relN'][19]*100:7.2f}% "
        f"{r['relP'][27]*100:7.2f}% {r['relN'][27]*100:7.2f}% "
        f"{r['sep'][27]*100:7.2f}%"
    )

print()
print("VERBOSITY DIRECT CHECK:")
vr=RESULTS["VERBOSITY"]
print(f"Detailed words={vr['pos_words']} | Concise words={vr['neg_words']} | Δ={vr['pos_words']-vr['neg_words']:+d}")
if vr["pos_words"]>vr["neg_words"]:
    print("VERBOSITY_DIRECTION: PASS")
else:
    print("VERBOSITY_DIRECTION: FAIL")

print("[10/10] Weight sentinel + final report...")
if fingerprint()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights: PASS")
print()
print("="*124)
print("TEST 249 RESULTS — 8-AXIS BEHAVIORAL CONTROL MATRIX")
print("="*124)
print("Prompt:",PROMPT)
print()
print("VANILLA:")
print(VANILLA)
print()

changed_pos=0
changed_neg=0
bipolar=0

for name,cfg in AXES.items():
    r=RESULTS[name]
    cp=(r["pos"]!=VANILLA)
    cn=(r["neg"]!=VANILLA)
    cb=(r["pos"]!=r["neg"])
    changed_pos+=int(cp)
    changed_neg+=int(cn)
    bipolar+=int(cb)

    print(f"{name}")
    print(f"POS [{cfg['pos_label']}]:")
    print(r["pos"])
    print(f"NEG [{cfg['neg_label']}]:")
    print(r["neg"])
    print(
        f"Changed P/N={int(cp)}/{int(cn)} | "
        f"POS!=NEG={int(cb)} | "
        f"L19 P/N={r['relP'][19]*100:.2f}%/{r['relN'][19]*100:.2f}% | "
        f"L27 P/N={r['relP'][27]*100:.2f}%/{r['relN'][27]*100:.2f}% | "
        f"SEP27={r['sep'][27]*100:.2f}%"
    )
    print("-"*124)

print(f"POS changed from vanilla : {changed_pos}/8")
print(f"NEG changed from vanilla : {changed_neg}/8")
print(f"POS != NEG               : {bipolar}/8")
print(f"Verbosity directional    : {'PASS' if vr['pos_words']>vr['neg_words'] else 'FAIL'}")
print("Motor                     : canonical SEASC L0-L19")
print("Observation               : L20-L27 motor OFF")
print("Compass                   : 8 POS vs 8 NEG mean hidden-state contrast per layer")
print("Weights                   : PASS")
print("Decoding                  : greedy")
print("Evaluation                : raw outputs shown; no LLM judge")

if bipolar==8 and changed_pos>=7 and changed_neg>=7:
    print("RESULT: STRONG_8_AXIS_BEHAVIORAL_DIVERGENCE")
elif bipolar>=6:
    print("RESULT: PARTIAL_MULTI_AXIS_BEHAVIORAL_CONTROL")
elif bipolar>=4:
    print("RESULT: MIXED_MULTI_AXIS_BEHAVIORAL_EFFECT")
else:
    print("RESULT: WEAK_MULTI_AXIS_BEHAVIORAL_EFFECT")

print("="*124)
print("TEST 249 COMPLETE")
