# ==================================================================================================
# AKBASCORE · RELATIONAL LATENT WRITING · TEST 264
# CONTENT × BINDING FACTORIZATION → MOTOR-OFF NATURAL-GEOMETRY READOUT → BLIND BEHAVIOR
# Baseline: TEST 250/263 | Qwen2.5-7B-Instruct · BF16 · SDPA · SEASC L0-L19 · L20-L27 MOTOR OFF
# NULL | CONTENT | BINDING | CONTENT+BINDING | CONTENT+CROSS
# NO TRAINING | NO LoRA | NO OPTIMIZER | NO WEIGHT UPDATE | GREEDY
# ==================================================================================================
import os,sys,json,math,time,random,hashlib,subprocess,importlib.util
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
SEED=264;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL_LAYERS=28;STEER_LAYERS=20;H_EXPECT=3584;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;MAX_NEW=64;EPS=1e-8
ROOT=Path("/content/AKBASCORE_TEST264") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST264");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
TARGET="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
NEUTRAL="A person performed an action involving an object at a location."
# Binding forge: same TEST263/262 family, now captured through all 28 layers for natural downstream references.
PAIRS=[
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge. Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.","Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge. Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."),
("Mustafa Akbaş placed the Turkish flag at the base of the Golden Gate Bridge. Daniel Carter planted the Canadian flag at the base of the Brooklyn Bridge.","Mustafa Akbaş placed the Canadian flag at the base of the Brooklyn Bridge. Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge."),
("At the base of the Golden Gate Bridge, Mustafa Akbaş planted the Turkish flag. At the base of the Brooklyn Bridge, Daniel Carter placed the Canadian flag.","At the base of the Brooklyn Bridge, Mustafa Akbaş planted the Canadian flag. At the base of the Golden Gate Bridge, Daniel Carter placed the Turkish flag."),
("The Turkish flag was planted by Mustafa Akbaş at the base of the Golden Gate Bridge. The Canadian flag was placed by Daniel Carter at the base of the Brooklyn Bridge.","The Canadian flag was planted by Mustafa Akbaş at the base of the Brooklyn Bridge. The Turkish flag was placed by Daniel Carter at the base of the Golden Gate Bridge."),
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge, while Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.","Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge, while Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."),
("While Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge, Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.","While Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge, Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge."),
("Two events occurred: Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge; Daniel Carter placed the Canadian flag at the base of the Brooklyn Bridge.","Two events occurred: Mustafa Akbaş planted the Canadian flag at the base of the Brooklyn Bridge; Daniel Carter placed the Turkish flag at the base of the Golden Gate Bridge."),
("The first event was Mustafa Akbaş planting the Turkish flag at the base of the Golden Gate Bridge. The second was Daniel Carter placing the Canadian flag at the base of the Brooklyn Bridge.","The first event was Mustafa Akbaş planting the Canadian flag at the base of the Brooklyn Bridge. The second was Daniel Carter placing the Turkish flag at the base of the Golden Gate Bridge.")
]
QUERIES={"WHO":"Who planted the Turkish flag at the base of the Golden Gate Bridge?","WHAT":"What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","WHERE":"Where did Mustafa Akbaş plant the Turkish flag?","ACTION":"What did Mustafa Akbaş do with the Turkish flag at the base of the Golden Gate Bridge?"}
EXPECTED={"WHO":"Mustafa Akbaş","WHAT":"Turkish flag","WHERE":"base of the Golden Gate Bridge","ACTION":"planted"}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def remove(h):
    for x in h:x.remove()
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=[IVME*env(L) for L in range(STEER_LAYERS)];RSS=math.sqrt(sum(x*x for x in RHO))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 264 — CONTENT × BINDING FACTORIZATION");print("="*110);print("START:",START)
print("[1/9] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL_LAYERS or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print(f"SEASC · L00={RHO[0]*100:.3f}% · L19={RHO[19]*100:.3f}% · RSS={RSS:.9f}")
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def enc(x):return tok(chat(x),return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(x):
    o=model(**enc(x),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL_LAYERS)]
print("[2/9] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
print("[3/9] 28-LAYER BINDING FORGE")
D=[]
for i,(a,b) in enumerate(PAIRS,1):
    A=capture(a);B=capture(b);D.append([A[L]-B[L] for L in range(TOTAL_LAYERS)])
    print(f" [{i}/8] Δ L00={D[-1][0].norm():.3f} · L19={D[-1][19].norm():.3f} · L27={D[-1][27].norm():.3f}")
BIND=[unit(torch.stack([D[i][L] for i in range(8)]).mean(0)) for L in range(TOTAL_LAYERS)]
LOO=[]
for L in range(TOTAL_LAYERS):
    z=[cos(BIND[L],unit(torch.stack([D[i][L] for i in range(8) if i!=j]).mean(0))) for j in range(8)]
    LOO.append(float(np.mean(z)))
print(f"LOO · L00={LOO[0]:.5f} L19={LOO[19]:.5f} L20={LOO[20]:.5f} L27={LOO[27]:.5f}")
print("[4/9] CONTENT FORGE")
# Target-specific content trace: TARGET minus relation-shaped neutral sentence.
HT=capture(TARGET);HN=capture(NEUTRAL)
CONTENT=[unit(HT[L]-HN[L]) for L in range(TOTAL_LAYERS)]
print(f"raw ||TARGET-NEUTRAL|| · L00={(HT[0]-HN[0]).norm():.3f} L19={(HT[19]-HN[19]).norm():.3f} L27={(HT[27]-HN[27]).norm():.3f}")
print(f"cos(C,B) · L00={cos(CONTENT[0],BIND[0]):+.4f} L19={cos(CONTENT[19],BIND[19]):+.4f} L27={cos(CONTENT[27],BIND[27]):+.4f}")
# Equal-norm composite vectors: each branch still receives canonical TEST250 rho_L dose, not double dose.
COMBO=[];CROSS=[]
for L in range(STEER_LAYERS):
    COMBO.append(unit(CONTENT[L]+BIND[L]))
    CROSS.append(unit(CONTENT[L]-BIND[L]))
STEER_TAG="akbascore_seasc_steer";OBS_TAG="akbascore_xray_observe"
def tagged(tag):return [sum(1 for f in l._forward_hooks.values() if getattr(f,"_akbascore",None)==tag) for l in layers]
def hooks():return sum(tagged(STEER_TAG))+sum(tagged(OBS_TAG))
def tail():return sum(tagged(STEER_TAG)[STEER_LAYERS:])
# TEST250 injection engine unchanged.
def make_hooks(vectors,sign=1.,telemetry=None):
    hs=[]
    try:
        for L in range(STEER_LAYERS):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out
                    if x.ndim!=3:return None
                    y=x.clone();z=y[:,-1,:].float();d=vectors[li].to(z.device)*z.norm(dim=-1,keepdim=True)*RHO[li]*sign
                    if telemetry is not None:telemetry[li].append(float(d.norm()/z.norm().clamp_min(EPS)))
                    y[:,-1,:]=(z+d).to(y.dtype);return (y,)+out[1:] if isinstance(out,tuple) else y
                hk._akbascore=STEER_TAG;return hk
            hs.append(layers[L].register_forward_hook(factory(L)))
    except Exception:remove(hs);raise
    return hs
@torch.inference_mode()
def generate(q,v=None):
    e=enc(q);n=e.input_ids.shape[1];tel={L:[] for L in range(STEER_LAYERS)};hs=[]
    try:
        if v is not None:hs=make_hooks(v,1.,tel)
        if tail():raise RuntimeError("Tail steering hook.")
        o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    finally:remove(hs)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip(),tel
@torch.inference_mode()
def xray(q,v=None):
    e=enc(q);s={};sh=[];oh=[]
    try:
        if v is not None:sh=make_hooks(v)
        if tail():raise RuntimeError("Tail steering hook.")
        for L in range(TOTAL_LAYERS):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out;s[li]=x[0,-1].float().detach().clone()
                hk._akbascore=OBS_TAG;return hk
            oh.append(layers[L].register_forward_hook(factory(L)))
        model(**e,use_cache=False,return_dict=True)
    finally:remove(oh);remove(sh)
    if sorted(s)!=list(range(TOTAL_LAYERS)):raise RuntimeError("Incomplete X-ray.")
    return s
print("[5/9] HOOK BOUNDARY")
p=make_hooks(BIND[:20])
try:b=tagged(STEER_TAG)
finally:remove(p)
if b!=[1]*20+[0]*8 or hooks():raise RuntimeError("Hook boundary failed.")
print("PASS · L0-L19 ON · L20-L27 OFF")
BRANCH={"NULL":None,"CONTENT":CONTENT[:20],"BINDING":BIND[:20],"CONTENT+BINDING":COMBO,"CONTENT+CROSS":CROSS}
RES={};XR={}
print("[6/9] BLIND READOUT")
for q,text in QUERIES.items():
    RES[q]={};XR[q]={};print(f"\n--- {q} · expected={EXPECTED[q]} ---")
    for n,v in BRANCH.items():
        out,tel=generate(text,v);h=xray(text,v);RES[q][n]={"text":out,"telemetry":tel};XR[q][n]=h
        print(f"{n:15s}: {out}")
print("\n[7/9] WRITE → MOTOR-OFF NATURAL-GEOMETRY X-RAY")
GEO={}
for q in QUERIES:
    base=XR[q]["NULL"];GEO[q]={};print("\n"+q)
    for n in ("CONTENT","BINDING","CONTENT+BINDING","CONTENT+CROSS"):
        d=[];cb=[];cc=[]
        for L in range(TOTAL_LAYERS):
            z=XR[q][n][L]-base[L];d.append(float(z.norm()/base[L].norm().clamp_min(EPS)));cb.append(cos(z,BIND[L]));cc.append(cos(z,CONTENT[L]))
        GEO[q][n]={"relative_displacement":d,"cos_natural_binding":cb,"cos_natural_content":cc}
        print(f"{n:15s} Δ19={d[19]*100:6.2f}% Δ20={d[20]*100:6.2f}% Δ27={d[27]*100:6.2f}% | B19={cb[19]:+.3f} B20={cb[20]:+.3f} B27={cb[27]:+.3f} | C19={cc[19]:+.3f} C20={cc[20]:+.3f} C27={cc[27]:+.3f}")
print("\n[8/9] DOSE / WEIGHT AUDIT")
dev=[];calls=0
for q in QUERIES:
    for n in BRANCH:
        if n=="NULL":continue
        for L,vals in RES[q][n]["telemetry"].items():
            for x in vals:dev.append(abs(x-RHO[L]));calls+=1
mx=max(dev)
if mx>=1e-4 or fp()!=FP0 or hooks() or tail():raise RuntimeError("Integrity failure.")
print(f"PASS · calls={calls} · max dose deviation={mx:.3e} · weights unchanged · hooks=0")
print("[9/9] SEAL")
def clean_tel(t):return {f"L{L:02d}":t[L] for L in range(STEER_LAYERS)}
for q in RES:
    for n in RES[q]:RES[q][n]["telemetry"]=clean_tel(RES[q][n]["telemetry"])
R={"schema":"akbascore.test264.v1","test":"TEST 264","start":START,"end":utc(),"model":MODEL_ID,
"hypothesis":"Blind relational retrieval requires target content plus binding geometry rather than binding geometry alone.",
"target":TARGET,"neutral_content_reference":NEUTRAL,"queries":QUERIES,"expected":EXPECTED,
"seasc":{"IVME":IVME,"SONUM":SONUM,"ZIRVE":ZIRVE,"TABAN":TABAN,"rho":RHO,"rss":RSS,"on":"L0-L19","off":"L20-L27"},
"binding":{"formula":"unit(mean(correct-cross))","pairs":PAIRS,"loo":LOO,"natural_reference_layers":"L0-L27","injected_layers":"L0-L19 only"},
"content":{"formula":"unit(h(TARGET)-h(NEUTRAL))","note":"Target-specific content direction; not claimed to be pure content."},
"branches":{"NULL":"no intervention","CONTENT":"unit target-neutral","BINDING":"matched correct-cross binding","CONTENT+BINDING":"unit(C+B), canonical rho preserved","CONTENT+CROSS":"unit(C-B), canonical rho preserved"},
"behavior":RES,"geometry":GEO,
"measurement":"L20-L27 cosines use independently extracted natural layer-local B_L/C_L references; no steering occurs after L19.",
"integrity":{"fingerprint_start":FP0,"fingerprint_end":fp(),"dose_calls":calls,"max_dose_deviation":mx,"tail_hooks":tail(),"active_hooks":hooks(),"result":"PASS"}}
raw=canon(R);sha=hashlib.sha256(raw).hexdigest();run=f"T264-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
tp=ROOT/f"{run}.txt";o=["="*110,"TEST 264 — CONTENT × BINDING FACTORIZATION","="*110,f"MODEL {MODEL_ID} · SEASC L0-L19 · MOTOR OFF L20-L27 · RSS={RSS:.9f}",""]
for q in QUERIES:
    o.append(f"[{q}] expected={EXPECTED[q]}")
    for n in BRANCH:o.append(f"{n:15s}: {RES[q][n]['text']}")
    o.append("")
o.append("GEOMETRY")
for q in QUERIES:
    o.append(q)
    for n in ("CONTENT","BINDING","CONTENT+BINDING","CONTENT+CROSS"):
        g=GEO[q][n];d=g["relative_displacement"];b=g["cos_natural_binding"];c=g["cos_natural_content"]
        o.append(f"{n:15s} Δ19={d[19]*100:.3f}% Δ20={d[20]*100:.3f}% Δ27={d[27]*100:.3f}% | B19={b[19]:+.4f} B20={b[20]:+.4f} B27={b[27]:+.4f} | C19={c[19]:+.4f} C20={c[20]:+.4f} C27={c[27]:+.4f}")
o+=["",f"DOSE CALLS={calls} · MAX DEV={mx:.3e}","WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 264 COMPLETE")
print("NULL · CONTENT · BINDING · CONTENT+BINDING · CONTENT+CROSS")
print("L0-L19 INTERVENTION · L20-L27 MOTOR OFF · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)
