# ==================================================================================================
# AKBASCORE · RELATIONAL LATENT WRITING · TEST 262
# MATCHED CROSS-BINDING ISOLATION
# Baseline: TEST 261 working Qwen2.5-7B-Instruct natural 28-layer X-ray infrastructure
# SAME LEXICAL INVENTORY · CORRECT↔CROSS RELATIONAL PAIRS · LAYER-LOCAL BINDING CANDIDATE
# NO STEERING | NO FINE-TUNING | NO LoRA | NO OPTIMIZER | NO WEIGHT UPDATE | GREEDY
# ==================================================================================================
import os,sys,json,math,time,random,hashlib,subprocess,importlib.util
from datetime import datetime,timezone
from pathlib import Path
for mod,pkg in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
SEED=262
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda")
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL_LAYERS=28
H_EXPECT=3584
MAX_NEW=64
EPS=1e-8
ROOT=Path("/content/AKBASCORE_TEST262") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST262")
ROOT.mkdir(parents=True,exist_ok=True)
START_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")

# --------------------------------------------------------------------------------------------------
# TEST 261 SINGLE-ROLE REFERENCE CONTRASTS — retained for contamination/projection measurement
# --------------------------------------------------------------------------------------------------
REF={
"TARGET":"Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.",
"SUBJECT":"Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge.",
"ACTION":"Mustafa Akbaş removed the Turkish flag from the base of the Golden Gate Bridge.",
"OBJECT":"Mustafa Akbaş planted the Canadian flag at the base of the Golden Gate Bridge.",
"LOCATION":"Mustafa Akbaş planted the Turkish flag at the base of the Brooklyn Bridge."
}

# --------------------------------------------------------------------------------------------------
# MATCHED CROSS-BINDING PAIRS
# Within every pair CORRECT and CROSS contain the same names, actions, objects and locations.
# Only relational assignment changes.
# --------------------------------------------------------------------------------------------------
PAIRS=[
{
"id":"P01",
"correct":"Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge. Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.",
"cross":"Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge. Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."
},
{
"id":"P02",
"correct":"Mustafa Akbaş placed the Turkish flag at the base of the Golden Gate Bridge. Daniel Carter planted the Canadian flag at the base of the Brooklyn Bridge.",
"cross":"Mustafa Akbaş placed the Canadian flag at the base of the Brooklyn Bridge. Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge."
},
{
"id":"P03",
"correct":"At the base of the Golden Gate Bridge, Mustafa Akbaş planted the Turkish flag. At the base of the Brooklyn Bridge, Daniel Carter placed the Canadian flag.",
"cross":"At the base of the Brooklyn Bridge, Mustafa Akbaş planted the Canadian flag. At the base of the Golden Gate Bridge, Daniel Carter placed the Turkish flag."
},
{
"id":"P04",
"correct":"The Turkish flag was planted by Mustafa Akbaş at the base of the Golden Gate Bridge. The Canadian flag was placed by Daniel Carter at the base of the Brooklyn Bridge.",
"cross":"The Canadian flag was planted by Mustafa Akbaş at the base of the Brooklyn Bridge. The Turkish flag was placed by Daniel Carter at the base of the Golden Gate Bridge."
},
{
"id":"P05",
"correct":"Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge, while Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.",
"cross":"Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge, while Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."
},
{
"id":"P06",
"correct":"While Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge, Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.",
"cross":"While Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge, Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge."
},
{
"id":"P07",
"correct":"Two events occurred: Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge; Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.",
"cross":"Two events occurred: Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge; Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."
},
{
"id":"P08",
"correct":"The first event was Mustafa Akbaş planting the Turkish flag at the base of the Golden Gate Bridge. The second was Daniel Carter placing the Canadian flag at the base of the Brooklyn Bridge.",
"cross":"The first event was Mustafa Akbaş planting the Canadian flag at the base of the Brooklyn Bridge. The second was Daniel Carter placing the Turkish flag at the base of the Golden Gate Bridge."
}
]

QUERIES={
"WHO":"Who planted the Turkish flag at the base of the Golden Gate Bridge?",
"WHAT":"What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?",
"WHERE":"Where did Mustafa Akbaş plant the Turkish flag?",
"ACTION":"What did Mustafa Akbaş do with the Turkish flag at the base of the Golden Gate Bridge?"
}
EXPECTED={"WHO":"Mustafa Akbaş","WHAT":"Turkish flag","WHERE":"base of the Golden Gate Bridge","ACTION":"planted"}

def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def enc(x):return tok(chat(x),return_tensors="pt",add_special_tokens=False).to(DEVICE)

print("="*110)
print("TEST 262 — MATCHED CROSS-BINDING ISOLATION")
print("="*110)
print("START:",START_UTC)
print("[1/8] MODEL LOAD")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t0=time.perf_counter()
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype
torch.cuda.synchronize();LOAD_S=time.perf_counter()-t0
if len(layers)!=TOTAL_LAYERS or H!=H_EXPECT:raise RuntimeError(f"Architecture mismatch: layers={len(layers)} hidden={H}")
if PDT!=torch.bfloat16:raise RuntimeError(f"Expected BF16, got {PDT}")
if any(p.requires_grad for p in model.parameters()):raise RuntimeError("Trainable parameters detected.")
if hasattr(model,"peft_config") or any("lora" in n.lower() for n,_ in model.named_modules()):raise RuntimeError("LoRA detected.")
print(f"OK · {MODEL_ID} · {len(layers)}L · H={H} · {PDT} · {LOAD_S:.2f}s")

print("[2/8] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fingerprint():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fingerprint()
print("FP:",[f"{x:.4f}" for x in FP0])

@torch.inference_mode()
def capture(text):
    e=enc(text)
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL_LAYERS)]

print("[3/8] TEST 261 REFERENCE CONTRASTS")
REF_H={}
for i,(name,text) in enumerate(REF.items(),1):
    REF_H[name]=capture(text)
    print(f"  [{i}/{len(REF)}] {name:8s} · L27 norm={REF_H[name][27].norm():.3f}")
REF_D={n:[REF_H["TARGET"][L]-REF_H[n][L] for L in range(TOTAL_LAYERS)] for n in ("SUBJECT","ACTION","OBJECT","LOCATION")}

print("[4/8] MATCHED CORRECT↔CROSS 28-LAYER X-RAY")
torch.cuda.synchronize();t0=time.perf_counter()
PAIR_H={};PAIR_D={}
for i,p in enumerate(PAIRS,1):
    hc=capture(p["correct"]);hx=capture(p["cross"])
    PAIR_H[p["id"]]={"correct":hc,"cross":hx}
    PAIR_D[p["id"]]=[hc[L]-hx[L] for L in range(TOTAL_LAYERS)]
    print(f"  [{i}/{len(PAIRS)}] {p['id']} · Δ L00={PAIR_D[p['id']][0].norm():.3f} · L19={PAIR_D[p['id']][19].norm():.3f} · L27={PAIR_D[p['id']][27].norm():.3f}")
torch.cuda.synchronize();XRAY_S=time.perf_counter()-t0

print("[5/8] CONSENSUS BINDING CANDIDATE")
BIND=[];RAW=[];PAIR_COS={};ADJ=[]
for L in range(TOTAL_LAYERS):
    ds=torch.stack([PAIR_D[p["id"]][L] for p in PAIRS])
    mean=ds.mean(0);RAW.append(float(mean.norm()));BIND.append(unit(mean))
    PAIR_COS[f"L{L:02d}"]=[[cosine(ds[i],ds[j]) for j in range(len(PAIRS))] for i in range(len(PAIRS))]
for L in range(TOTAL_LAYERS-1):ADJ.append(cosine(BIND[L],BIND[L+1]))
print("layer | meanΔnorm | pair mean cos | pair +frac | B_L→B_L+1 | cos(S) cos(A) cos(O) cos(L)")
CONTAM={};PAIR_STATS={}
for L in range(TOTAL_LAYERS):
    M=np.asarray(PAIR_COS[f"L{L:02d}"]);v=M[np.triu_indices(len(PAIRS),1)]
    PAIR_STATS[f"L{L:02d}"]={"mean":float(v.mean()),"mean_abs":float(np.abs(v).mean()),"positive_fraction":float((v>0).mean()),"min":float(v.min()),"max":float(v.max())}
    c={n:cosine(BIND[L],REF_D[n][L]) for n in REF_D};CONTAM[f"L{L:02d}"]=c
    adj=ADJ[L] if L<TOTAL_LAYERS-1 else float("nan")
    print(f"L{L:02d}   | {RAW[L]:10.3f} | {v.mean():+13.4f} | {(v>0).mean():10.3f} | {adj:+10.4f} | {c['SUBJECT']:+.3f} {c['ACTION']:+.3f} {c['OBJECT']:+.3f} {c['LOCATION']:+.3f}")

print("[6/8] LEAVE-ONE-PAIR-OUT STABILITY")
LOO={}
for L in range(TOTAL_LAYERS):
    full=BIND[L];vals=[]
    for drop in range(len(PAIRS)):
        m=torch.stack([PAIR_D[p["id"]][L] for i,p in enumerate(PAIRS) if i!=drop]).mean(0)
        vals.append(cosine(full,unit(m)))
    LOO[f"L{L:02d}"]={"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals)),"values":vals}
    print(f"L{L:02d} · LOO cos mean={np.mean(vals):+.5f} min={np.min(vals):+.5f} max={np.max(vals):+.5f}")

@torch.inference_mode()
def generate(prompt):
    e=enc(prompt);n=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip()

print("[7/8] NATURAL READOUT SANITY")
READOUT={}
FACT=PAIRS[0]["correct"]
for q,question in QUERIES.items():
    positive=f"Read the facts and answer the question using only those facts.\n\nFACTS: {FACT}\n\nQUESTION: {question}"
    blind=f"Answer using only information explicitly available to you. If unavailable, say UNKNOWN.\n\nQUESTION: {question}"
    pos=generate(positive);neg=generate(blind)
    READOUT[q]={"expected":EXPECTED[q],"target_present":pos,"target_absent":neg}
    print(f"{q:6s} | PRESENT: {pos}")
    print(f"       | ABSENT : {neg}")

print("[8/8] SEAL")
if fingerprint()!=FP0:raise RuntimeError("WEIGHT FINGERPRINT CHANGED")
REPORT={
"schema":"akbascore.test262.cross_binding_isolation.v1","test":"TEST 262",
"title":"Matched Cross-Binding Isolation","start_utc":START_UTC,"end_utc":utc(),"seed":SEED,
"model":{"id":MODEL_ID,"layers":TOTAL_LAYERS,"hidden":H,"dtype":str(PDT),"attention":"sdpa"},
"training":{"steering":False,"fine_tuning":False,"lora":False,"optimizer":False,"trainable_tensors":0},
"system":SYSTEM,"reference_sentences":REF,"matched_pairs":PAIRS,"queries":QUERIES,"expected":EXPECTED,
"measurement":{"position":"last token of each chat-formatted text","layers":"decoder outputs L0-L27","use_cache":False,
"pair_contrast":"R_L(i)=h_L(correct_i)-h_L(cross_i)",
"binding_candidate":"B_L=normalize(mean_i R_L(i))",
"warning":"B_L is a consensus matched-pair binding candidate, not proof of a pure or uniquely isolated binding representation."},
"binding":{"raw_mean_difference_norm":RAW,"adjacent_layer_cosine":ADJ,"pairwise_cosine":PAIR_COS,"pair_stats":PAIR_STATS,
"leave_one_pair_out":LOO,"reference_contrast_cosine":CONTAM},
"natural_readout":READOUT,"timing":{"model_load_seconds":LOAD_S,"matched_xray_seconds":XRAY_S},
"integrity":{"weight_fingerprint_start":FP0,"weight_fingerprint_end":fingerprint(),"result":"PASS"}
}
raw=canon(REPORT);sha=hashlib.sha256(raw).hexdigest()
REPORT["artifact_sha256_pre_sha_field"]=sha
run=f"T262-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(REPORT,sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
tp=ROOT/f"{run}.txt"
lines=["="*110,"TEST 262 — MATCHED CROSS-BINDING ISOLATION","="*110,
f"MODEL: {MODEL_ID} | 28L | H={H} | BF16 | NO STEERING | NO TRAINING",
"R_L(i)=h_L(correct_i)-h_L(cross_i)","B_L=normalize(mean_i R_L(i))","",
"LAYER | PAIR-COS | +FRAC | LOO-MEAN | B→SUBJ B→ACT B→OBJ B→LOC"]
for L in range(TOTAL_LAYERS):
    ps=PAIR_STATS[f"L{L:02d}"];lo=LOO[f"L{L:02d}"];c=CONTAM[f"L{L:02d}"]
    lines.append(f"L{L:02d} | {ps['mean']:+.4f} | {ps['positive_fraction']:.3f} | {lo['mean']:+.4f} | {c['SUBJECT']:+.4f} {c['ACTION']:+.4f} {c['OBJECT']:+.4f} {c['LOCATION']:+.4f}")
lines+=["","NATURAL READOUT"]
for q,r in READOUT.items():lines += [f"{q} EXPECTED={r['expected']}",f"PRESENT: {r['target_present']}",f"ABSENT : {r['target_absent']}",""]
lines += ["B_L is a matched-pair consensus binding candidate, not proof of a pure binding representation.",
"NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"SHA-256(pre-sha-field canonical payload): {sha}"]
tp.write_text("\n".join(lines),encoding="utf-8")
print("="*110)
print("TEST 262 COMPLETE")
print("MATCHED PAIRS:",len(PAIRS),"· 16 relational forward passes")
print("NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp)
print("TXT :",tp)
print("SHA :",sha)
print("="*110)
