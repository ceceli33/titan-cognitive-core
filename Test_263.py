# ==================================================================================================
# AKBASCORE · RELATIONAL LATENT WRITING · TEST 263
# CAUSAL BOUND-PACKET WRITE → MOTOR-OFF TRANSPORT → BLIND NATURAL READOUT
# Proven engine baseline: TEST 250 | Binding forge: TEST 262
# Qwen2.5-7B-Instruct · BF16 · SDPA · SEASC L0-L19 · MOTOR OFF L20-L27
# NULL | BOUND+ | BOUND− | SHUFFLED-LAYER
# NO FINE-TUNING | NO LoRA | NO OPTIMIZER | NO WEIGHT UPDATE | GREEDY
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
SEED=263
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda")
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL_LAYERS=28;STEER_LAYERS=20;H_EXPECT=3584
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20
MAX_NEW=64;EPS=1e-8
ROOT=Path("/content/AKBASCORE_TEST263") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST263")
ROOT.mkdir(parents=True,exist_ok=True)
START_UTC=datetime.now(timezone.utc).isoformat(timespec="milliseconds")

PAIRS=[
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge. Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.",
 "Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge. Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."),
("Mustafa Akbaş placed the Turkish flag at the base of the Golden Gate Bridge. Daniel Carter planted the Canadian flag at the base of the Brooklyn Bridge.",
 "Mustafa Akbaş placed the Canadian flag at the base of the Brooklyn Bridge. Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge."),
("At the base of the Golden Gate Bridge, Mustafa Akbaş planted the Turkish flag. At the base of the Brooklyn Bridge, Daniel Carter placed the Canadian flag.",
 "At the base of the Brooklyn Bridge, Mustafa Akbaş planted the Canadian flag. At the base of the Golden Gate Bridge, Daniel Carter placed the Turkish flag."),
("The Turkish flag was planted by Mustafa Akbaş at the base of the Golden Gate Bridge. The Canadian flag was placed by Daniel Carter at the base of the Brooklyn Bridge.",
 "The Canadian flag was planted by Mustafa Akbaş at the base of the Brooklyn Bridge. The Turkish flag was placed by Daniel Carter at the base of the Golden Gate Bridge."),
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge, while Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.",
 "Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge, while Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."),
("While Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge, Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.",
 "While Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge, Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge."),
("Two events occurred: Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge; Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.",
 "Two events occurred: Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge; Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."),
("The first event was Mustafa Akbaş planting the Turkish flag at the base of the Golden Gate Bridge. The second was Daniel Carter placing the Canadian flag at the base of the Brooklyn Bridge.",
 "The first event was Mustafa Akbaş planting the Canadian flag at the base of the Brooklyn Bridge. The second was Daniel Carter placing the Turkish flag at the base of the Golden Gate Bridge.")
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
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def remove(hs):
    for h in hs:h.remove()
def envelope(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=[IVME*envelope(L) for L in range(STEER_LAYERS)]
RSS=math.sqrt(sum(x*x for x in RHO))
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def enc(x):return tok(chat(x),return_tensors="pt",add_special_tokens=False).to(DEVICE)

print("="*110);print("TEST 263 — CAUSAL BOUND-PACKET WRITE → MOTOR-OFF TRANSPORT → BLIND READOUT");print("="*110)
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
if len(layers)!=28 or H!=3584 or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
if any(p.requires_grad for p in model.parameters()):raise RuntimeError("Trainable parameters detected.")
if hasattr(model,"peft_config") or any("lora" in n.lower() for n,_ in model.named_modules()):raise RuntimeError("LoRA detected.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {LOAD_S:.2f}s")
print(f"SEASC · L00={RHO[0]*100:.3f}% · L19={RHO[19]*100:.3f}% · RSS={RSS:.9f}")

print("[2/8] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fingerprint():return tuple(float(t.sum(dtype=torch.float32)) for t in FP_T)
FP0=fingerprint();print("FP:",[f"{x:.4f}" for x in FP0])

@torch.inference_mode()
def capture(text,n=TOTAL_LAYERS):
    o=model(**enc(text),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(n)]

print("[3/8] TEST 262 BINDING FORGE")
PAIR_D=[]
for i,(correct,cross) in enumerate(PAIRS,1):
    hc=capture(correct,STEER_LAYERS);hx=capture(cross,STEER_LAYERS)
    PAIR_D.append([hc[L]-hx[L] for L in range(STEER_LAYERS)])
    print(f"  [{i}/8] Δ L00={PAIR_D[-1][0].norm():.3f} · L19={PAIR_D[-1][19].norm():.3f}")
BIND=[]
for L in range(STEER_LAYERS):BIND.append(unit(torch.stack([PAIR_D[i][L] for i in range(len(PAIRS))]).mean(0)))
LOO=[]
for L in range(STEER_LAYERS):
    vals=[]
    for drop in range(len(PAIRS)):
        b=unit(torch.stack([PAIR_D[i][L] for i in range(len(PAIRS)) if i!=drop]).mean(0))
        vals.append(cosine(BIND[L],b))
    LOO.append(float(np.mean(vals)))
print(f"Binding forge · LOO mean L00={LOO[0]:.5f} · L19={LOO[19]:.5f}")

# Fixed deterministic layer permutation: same vectors + same RHO budget, wrong vector↔layer assignment.
PERM=list(range(STEER_LAYERS))
random.Random(263).shuffle(PERM)
if any(PERM[L]==L for L in range(STEER_LAYERS)):
    PERM=PERM[1:]+PERM[:1]
SHUFFLED=[BIND[PERM[L]] for L in range(STEER_LAYERS)]
print("SHUFFLED layer map:",PERM)

STEER_TAG="akbascore_seasc_steer";OBS_TAG="akbascore_xray_observe"
def tagged(tag):return [sum(1 for f in l._forward_hooks.values() if getattr(f,"_akbascore",None)==tag) for l in layers]
def hooks_total():return sum(tagged(STEER_TAG))+sum(tagged(OBS_TAG))
def tail_hooks():return sum(tagged(STEER_TAG)[STEER_LAYERS:])

# TEST 250 steering engine preserved.
def make_hooks(vectors,sign=1.0,telemetry=None):
    hs=[]
    try:
        for L in range(STEER_LAYERS):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out
                    if x.ndim!=3:return None
                    y=x.clone();z=y[:,-1,:].float()
                    d=vectors[li].to(z.device)*z.norm(dim=-1,keepdim=True)*RHO[li]*sign
                    if telemetry is not None:telemetry[li].append(float(d.norm()/z.norm().clamp_min(EPS)))
                    y[:,-1,:]=(z+d).to(y.dtype)
                    return (y,)+out[1:] if isinstance(out,tuple) else y
                hk._akbascore=STEER_TAG;return hk
            hs.append(layers[L].register_forward_hook(factory(L)))
    except Exception:remove(hs);raise
    return hs

@torch.inference_mode()
def generate(question,vectors=None,sign=1.0):
    e=enc(question);n=e.input_ids.shape[1];tel={L:[] for L in range(STEER_LAYERS)};hs=[]
    try:
        if vectors is not None:hs=make_hooks(vectors,sign,tel)
        if tail_hooks()!=0:raise RuntimeError("Steering hook detected L20-L27.")
        o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    finally:remove(hs)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip(),tel,int(o.shape[1]-n)

# TEST 250 X-ray ordering preserved: steering hooks registered BEFORE observation hooks.
@torch.inference_mode()
def xray(question,vectors=None,sign=1.0):
    e=enc(question);store={};steer=[];obs=[]
    try:
        if vectors is not None:steer=make_hooks(vectors,sign,None)
        if tail_hooks()!=0:raise RuntimeError("Steering hook detected L20-L27.")
        for L in range(TOTAL_LAYERS):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out
                    store[li]=x[0,-1].float().detach().clone()
                hk._akbascore=OBS_TAG;return hk
            obs.append(layers[L].register_forward_hook(factory(L)))
        model(**e,use_cache=False,return_dict=True)
    finally:remove(obs);remove(steer)
    if sorted(store)!=list(range(TOTAL_LAYERS)):raise RuntimeError("Incomplete X-ray.")
    return store

print("[4/8] HOOK BOUNDARY SELF-TEST")
if hooks_total()!=0:raise RuntimeError("Stale hooks.")
probe=make_hooks(BIND)
try:boundary=tagged(STEER_TAG)
finally:remove(probe)
if boundary!=[1]*20+[0]*8:raise RuntimeError(f"Hook boundary failed: {boundary}")
if hooks_total()!=0:raise RuntimeError("Hook cleanup failed.")
print("PASS · L0-L19=1 each · L20-L27=0")

BRANCHES={"NULL":(None,1.0),"BOUND+":(BIND,1.0),"BOUND-":(BIND,-1.0),"SHUFFLED":(SHUFFLED,1.0)}
RESULT={};XR={}
print("[5/8] BLIND BEHAVIORAL READOUT")
for q,question in QUERIES.items():
    print(f"\n--- {q} · expected={EXPECTED[q]} ---")
    RESULT[q]={};XR[q]={}
    for name,(vec,sign) in BRANCHES.items():
        text,tel,nt=generate(question,vec,sign)
        h=xray(question,vec,sign)
        RESULT[q][name]={"text":text,"new_tokens":nt,"telemetry":{f"L{L:02d}":tel[L] for L in range(STEER_LAYERS)}}
        XR[q][name]=h
        print(f"{name:9s}: {text}")

print("\n[6/8] PHYSICAL WRITE / MOTOR-OFF TRANSPORT")
GEOM={}
for q in QUERIES:
    base=XR[q]["NULL"];GEOM[q]={}
    print(f"\n{q}")
    for name in ("BOUND+","BOUND-","SHUFFLED"):
        h=XR[q][name];disp=[];align=[]
        for L in range(TOTAL_LAYERS):
            d=h[L]-base[L]
            disp.append(float(d.norm()/base[L].norm().clamp_min(EPS)))
            # L0-L19: alignment to layer-local written vector.
            # L20-L27: no direct B_L exists; alignment to L19 binding direction is diagnostic only.
            ref=BIND[L] if L<STEER_LAYERS else BIND[19]
            align.append(cosine(d,ref))
        GEOM[q][name]={"relative_displacement":disp,"alignment_to_binding_reference":align}
        print(f"{name:9s} Δ L00={disp[0]*100:7.3f}% L19={disp[19]*100:7.3f}% L20={disp[20]*100:7.3f}% L27={disp[27]*100:7.3f}% | cosB L19={align[19]:+.4f} L20={align[20]:+.4f} L27={align[27]:+.4f}")

print("\n[7/8] DOSE / INTEGRITY AUDIT")
dev=[];calls=0
for q in QUERIES:
    for name in ("BOUND+","BOUND-","SHUFFLED"):
        tel=RESULT[q][name]["telemetry"]
        for L in range(STEER_LAYERS):
            for v in tel[f"L{L:02d}"]:dev.append(abs(v-RHO[L]));calls+=1
MAX_DEV=max(dev) if dev else None
if not dev or MAX_DEV>=1e-4:raise RuntimeError("Dose telemetry failed.")
if fingerprint()!=FP0:raise RuntimeError("WEIGHT FINGERPRINT CHANGED")
if hooks_total()!=0 or tail_hooks()!=0:raise RuntimeError("Hook integrity failed.")
if model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Frozen/eval integrity failed.")
print(f"PASS · injection calls={calls} · max dose deviation={MAX_DEV:.3e} · weights unchanged · hooks=0")

print("[8/8] SEAL")
REPORT={
"schema":"akbascore.test263.bound_packet_write.v1","test":"TEST 263",
"title":"Causal Bound-Packet Write → Motor-Off Transport → Blind Natural Readout",
"start_utc":START_UTC,"end_utc":utc(),"seed":SEED,
"model":{"id":MODEL_ID,"layers":28,"hidden":3584,"dtype":str(PDT),"attention":"sdpa"},
"seasc":{"IVME":IVME,"SONUM":SONUM,"ZIRVE":ZIRVE,"TABAN":TABAN,"rho":RHO,"rss":RSS,"steered_layers":list(range(20)),"motor_off_layers":list(range(20,28))},
"binding_forge":{"pairs":PAIRS,"formula":"B_L=normalize(mean_i(h_L(correct_i)-h_L(cross_i)))","layers":"L0-L19","loo_mean":LOO},
"shuffled_control":{"layer_permutation":PERM,"note":"Same B_L vector set and same SEASC dose schedule; vector-to-layer correspondence is permuted."},
"queries":QUERIES,"expected":EXPECTED,"branches":["NULL","BOUND+","BOUND-","SHUFFLED"],
"behavior":RESULT,
"geometry":GEOM,
"measurement":{"generation":"TEST 250 make_hooks/generate architecture; intervention at final sequence position during prefill and every decode step.",
"xray":"Separate use_cache=False prefill pass; steering hooks registered before observation hooks; L0-L19 post-own-injection, L20-L27 downstream only.",
"downstream_alignment":"L20-L27 cosine uses B_L19 only as a fixed diagnostic reference; it is not a layer-local natural binding vector for those layers."},
"integrity":{"weight_fingerprint_start":FP0,"weight_fingerprint_end":fingerprint(),"dose_calls":calls,"max_abs_dose_deviation":MAX_DEV,
"tail_steering_hooks":tail_hooks(),"active_hooks_after":hooks_total(),"training":False,"lora":False,"optimizer":False,"result":"PASS"}
}
raw=canon(REPORT);sha=hashlib.sha256(raw).hexdigest();REPORT["artifact_sha256_pre_sha_field"]=sha
run=f"T263-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(REPORT,sort_keys=True,indent=2,ensure_ascii=False).encode())
tp=ROOT/f"{run}.txt"
lines=["="*110,"TEST 263 — CAUSAL BOUND-PACKET WRITE → MOTOR-OFF TRANSPORT → BLIND READOUT","="*110,
f"MODEL: {MODEL_ID} | 28L | H=3584 | BF16 | SDPA",
f"SEASC: L0-L19 ON | L20-L27 OFF | RSS={RSS:.9f}",
"B_L=normalize(mean_i(h(correct_i)-h(cross_i)))",""]
for q in QUERIES:
    lines.append(f"[{q}] expected={EXPECTED[q]}")
    for name in BRANCHES:lines.append(f"{name:9s}: {RESULT[q][name]['text']}")
    lines.append("")
lines+=["PHYSICAL TRANSPORT"]
for q in QUERIES:
    lines.append(q)
    for name in ("BOUND+","BOUND-","SHUFFLED"):
        g=GEOM[q][name];d=g["relative_displacement"];a=g["alignment_to_binding_reference"]
        lines.append(f"{name:9s} ΔL00={d[0]*100:.3f}% ΔL19={d[19]*100:.3f}% ΔL20={d[20]*100:.3f}% ΔL27={d[27]*100:.3f}% | cosB19={a[19]:+.4f} cosB20={a[20]:+.4f} cosB27={a[27]:+.4f}")
lines+=["",f"DOSE CALLS={calls} · MAX DEVIATION={MAX_DEV:.3e}","WEIGHT INTEGRITY=PASS · L20-L27 DIRECT STEERING HOOKS=0",
"Interpretation guard: behavioral retrieval requires relation-specific blind answers; physical displacement alone is not retrieval.",
f"JSON: {jp}",f"SHA-256(pre-sha-field canonical payload): {sha}"]
tp.write_text("\n".join(lines),encoding="utf-8")
print("="*110)
print("TEST 263 COMPLETE")
print("NULL · BOUND+ · BOUND− · SHUFFLED")
print("SEASC L0-L19 ON · L20-L27 MOTOR OFF")
print("WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha)
print("="*110)
