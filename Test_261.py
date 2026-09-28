# ==================================================================================================
# AKBASCORE · RELATIONAL LATENT WRITING · TEST 261
# NATURAL BINDING X-RAY — GOLDEN GATE BASELINE
# Baseline: TEST 250 proven Qwen2.5-7B-Instruct loading/capture/frozen-weight architecture
# PURPOSE: locate natural relational-binding geometry BEFORE synthetic bound-packet injection
# NO STEERING | NO FINE-TUNING | NO LoRA | NO OPTIMIZER | NO WEIGHT UPDATE | GREEDY
# 28-LAYER X-RAY · TARGET vs SUBJECT/ACTION/OBJECT/LOCATION SWAPS + RELATION SCRAMBLE
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
SEED=261
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda")
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL_LAYERS=28
H_EXPECT=3584
MAX_NEW=64
EPS=1e-8
ROOT=Path("/content/AKBASCORE_TEST261") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST261")
ROOT.mkdir(parents=True,exist_ok=True)
START_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
SCENES={
"TARGET":"Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.",
"SUBJECT_SWAP":"Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge.",
"ACTION_SWAP":"Mustafa Akbaş removed the Turkish flag from the base of the Golden Gate Bridge.",
"OBJECT_SWAP":"Mustafa Akbaş planted the Canadian flag at the base of the Golden Gate Bridge.",
"LOCATION_SWAP":"Mustafa Akbaş planted the Turkish flag at the base of the Brooklyn Bridge.",
"SCRAMBLED":"The Turkish flag planted Mustafa Akbaş at the base of the Golden Gate Bridge."
}
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
print("TEST 261 — NATURAL BINDING X-RAY — GOLDEN GATE BASELINE")
print("="*110)
print("START:",START_UTC)
print("[1/7] MODEL LOAD")
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

print("[2/7] WEIGHT SENTINEL")
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

print("[3/7] NATURAL 28-LAYER X-RAY")
torch.cuda.synchronize();t0=time.perf_counter()
STATES={}
for i,(name,text) in enumerate(SCENES.items(),1):
    STATES[name]=capture(text)
    print(f"  [{i}/{len(SCENES)}] {name:14s} · final norm={STATES[name][-1].norm():.3f}")
torch.cuda.synchronize();XRAY_S=time.perf_counter()-t0

print("[4/7] CONTRAST / BINDING-CANDIDATE GEOMETRY")
TARGET=STATES["TARGET"]
CONTRASTS={}
for name in SCENES:
    if name=="TARGET":continue
    ds=[TARGET[L]-STATES[name][L] for L in range(TOTAL_LAYERS)]
    CONTRASTS[name]={
        "vectors":ds,
        "norm":[float(d.norm()) for d in ds],
        "relative":[float(d.norm()/TARGET[L].norm().clamp_min(EPS)) for L,d in enumerate(ds)],
        "adjacent_cosine":[cosine(unit(ds[L]),unit(ds[L+1])) for L in range(TOTAL_LAYERS-1)]
    }
BIND=CONTRASTS["SCRAMBLED"]["vectors"]
print("layer | subject% action% object% location% scramble% | scramble adj.cos")
for L in range(TOTAL_LAYERS):
    vals=[CONTRASTS[n]["relative"][L]*100 for n in ("SUBJECT_SWAP","ACTION_SWAP","OBJECT_SWAP","LOCATION_SWAP","SCRAMBLED")]
    adj=CONTRASTS["SCRAMBLED"]["adjacent_cosine"][L] if L<TOTAL_LAYERS-1 else float("nan")
    print(f"L{L:02d}   | {vals[0]:7.3f} {vals[1]:7.3f} {vals[2]:7.3f} {vals[3]:8.3f} {vals[4]:8.3f} | {adj:+.5f}")

print("[5/7] CROSS-CONTRAST COSINE")
CN=["SUBJECT_SWAP","ACTION_SWAP","OBJECT_SWAP","LOCATION_SWAP","SCRAMBLED"]
CROSS={}
for L in range(TOTAL_LAYERS):
    M=[]
    for a in CN:
        row=[]
        for b in CN:row.append(cosine(CONTRASTS[a]["vectors"][L],CONTRASTS[b]["vectors"][L]))
        M.append(row)
    CROSS[f"L{L:02d}"]=M
for L in (0,5,10,15,19,20,23,27):
    M=np.array(CROSS[f"L{L:02d}"])
    print(f"\nL{L:02d}")
    print(" "*15+" ".join(f"{x[:8]:>9}" for x in CN))
    for i,n in enumerate(CN):print(f"{n[:14]:14s} "+" ".join(f"{M[i,j]:+9.4f}" for j in range(len(CN))))

@torch.inference_mode()
def generate(prompt):
    e=enc(prompt);n=e.input_ids.shape[1]
    torch.cuda.synchronize();t0=time.perf_counter()
    o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return tok.decode(o[0,n:],skip_special_tokens=True).strip(),time.perf_counter()-t0

print("\n[6/7] NATURAL READOUT CONTROLS")
READOUT={}
for q,question in QUERIES.items():
    positive=f"Read the following fact and answer the question using only that fact.\n\nFACT: {SCENES['TARGET']}\n\nQUESTION: {question}"
    blind=f"Answer the following question using only information explicitly available to you. If the information is unavailable, say UNKNOWN.\n\nQUESTION: {question}"
    pos,pt=generate(positive);neg,nt=generate(blind)
    READOUT[q]={"question":question,"expected":EXPECTED[q],"target_present":pos,"target_absent":neg,"target_present_seconds":pt,"target_absent_seconds":nt}
    print(f"\n{q} · expected: {EXPECTED[q]}")
    print("  TARGET PRESENT:",pos)
    print("  TARGET ABSENT :",neg)

print("\n[7/7] SEAL")
if fingerprint()!=FP0:raise RuntimeError("WEIGHT FINGERPRINT CHANGED")
REPORT={
"schema":"akbascore.test261.natural_binding_xray.v1",
"test":"TEST 261",
"title":"Natural Binding X-Ray — Golden Gate Baseline",
"purpose":"Measure natural relational contrast geometry before synthetic bound-packet injection.",
"start_utc":START_UTC,"end_utc":utc(),"seed":SEED,
"model":{"id":MODEL_ID,"layers":TOTAL_LAYERS,"hidden":H,"dtype":str(PDT),"attention":"sdpa"},
"training":{"fine_tuning":False,"lora":False,"optimizer":False,"trainable_tensors":0},
"system":SYSTEM,"scenes":SCENES,"queries":QUERIES,"expected":EXPECTED,
"measurement":{"position":"last token of each chat-formatted scene","layers":"decoder outputs L0-L27","use_cache":False,
"contrast":"D_L(scene)=h_L(TARGET)-h_L(scene)","binding_candidate":"B_L=h_L(TARGET)-h_L(SCRAMBLED)",
"warning":"B_L is a binding candidate, not proof of an isolated binding representation."},
"contrasts":{n:{
"norm":CONTRASTS[n]["norm"],
"relative":CONTRASTS[n]["relative"],
"adjacent_cosine":CONTRASTS[n]["adjacent_cosine"]
} for n in CONTRASTS},
"cross_contrast_order":CN,"cross_contrast_cosine":CROSS,"natural_readout":READOUT,
"timing":{"model_load_seconds":LOAD_S,"xray_seconds":XRAY_S},
"integrity":{"weight_fingerprint_start":FP0,"weight_fingerprint_end":fingerprint(),"result":"PASS"}
}
raw=canon(REPORT);sha=hashlib.sha256(raw).hexdigest()
REPORT["artifact_sha256_pre_sha_field"]=sha
run=f"T261-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(REPORT,sort_keys=True,indent=2,ensure_ascii=False).encode("utf-8"))
tp=ROOT/f"{run}.txt"
lines=["="*110,"TEST 261 — NATURAL BINDING X-RAY — GOLDEN GATE BASELINE","="*110,
f"MODEL: {MODEL_ID} | 28L | H={H} | BF16 | NO STEERING | NO TRAINING",
f"TARGET: {SCENES['TARGET']}","",
"RELATIVE TARGET→CONTROL DISPLACEMENT (%)"]
for L in range(TOTAL_LAYERS):
    vals=[CONTRASTS[n]["relative"][L]*100 for n in CN]
    lines.append(f"L{L:02d} SUBJECT={vals[0]:.4f} ACTION={vals[1]:.4f} OBJECT={vals[2]:.4f} LOCATION={vals[3]:.4f} SCRAMBLE={vals[4]:.4f}")
lines+=["","NATURAL READOUT"]
for q,r in READOUT.items():
    lines += [f"{q} | EXPECTED: {r['expected']}",f"TARGET PRESENT: {r['target_present']}",f"TARGET ABSENT : {r['target_absent']}",""]
lines += ["NOTE: SCRAMBLE contrast is a binding candidate only; TEST 261 performs no synthetic injection.",
          "WEIGHT INTEGRITY: PASS",f"JSON: {jp}",f"SHA-256(pre-sha-field canonical payload): {sha}"]
tp.write_text("\n".join(lines),encoding="utf-8")
print("="*110)
print("TEST 261 COMPLETE")
print("NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp)
print("TXT :",tp)
print("SHA :",sha)
print("="*110)
