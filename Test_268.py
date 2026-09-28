# ==================================================================================================
# AKBASCORE · TEST 268
# NATURAL READOUT TRAJECTORY CALIBRATION
# NATURAL TARGET vs BLIND NULL vs INJECTED BOUND/CROSS
# TEST267 L0-L25 FIXED-RSS · L26-L27 MOTOR OFF · NO TRAINING · NO WEIGHT UPDATE
# ==================================================================================================
import os,sys,json,math,time,random,hashlib,subprocess,importlib.util
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
SEED=268;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;END=25;EPS=1e-8;MAX_NEW=64
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;REFERENCE_END=19
ROOT=Path("/content/AKBASCORE_TEST268") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST268");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
ROLES=["SUBJECT","ACTION","OBJECT","LOCATION"]
ROLE_PAIRS={
"SUBJECT":[
(FACT,"Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge."),
("Mustafa Akbaş placed the silver key beside the oak tree.","Daniel Carter placed the silver key beside the oak tree."),
("Mustafa Akbaş carried the blue book into the stone house.","Daniel Carter carried the blue book into the stone house."),
("Mustafa Akbaş left the copper coin beside the wooden gate.","Daniel Carter left the copper coin beside the wooden gate.")],
"ACTION":[
(FACT,"Mustafa Akbaş removed the Turkish flag at the base of the Golden Gate Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed removed the silver key beside the oak tree."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks removed the blue book from the stone house."),
("Nora Hayes placed the copper coin beside the wooden gate.","Nora Hayes removed the copper coin from beside the wooden gate.")],
"OBJECT":[
(FACT,"Mustafa Akbaş planted the Canadian flag at the base of the Golden Gate Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed placed the copper coin beside the oak tree."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks carried the red box into the stone house."),
("Nora Hayes left the glass bottle beside the wooden gate.","Nora Hayes left the paper envelope beside the wooden gate.")],
"LOCATION":[
(FACT,"Mustafa Akbaş planted the Turkish flag at the base of the Brooklyn Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed placed the silver key beside the stone wall."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks carried the blue book into the railway station."),
("Nora Hayes left the copper coin beside the wooden gate.","Nora Hayes left the copper coin beside the garden fountain.")]
}
BIND_PAIRS=[
("Emma Reed placed the silver key beside the oak tree. Liam Brooks placed the copper coin beside the stone wall.","Emma Reed placed the copper coin beside the stone wall. Liam Brooks placed the silver key beside the oak tree."),
("Nora Hayes carried the blue book into the stone house. Owen Clark carried the red box into the railway station.","Nora Hayes carried the red box into the railway station. Owen Clark carried the blue book into the stone house."),
("Alice Morgan left the glass bottle beside the wooden gate. Henry Cole left the paper envelope beside the garden fountain.","Alice Morgan left the paper envelope beside the garden fountain. Henry Cole left the glass bottle beside the wooden gate."),
("Sofia Grant placed the brass token near the marble arch. Ethan Blake placed the green notebook near the river bench.","Sofia Grant placed the green notebook near the river bench. Ethan Blake placed the brass token near the marble arch."),
("Maya Stone carried the white package into the north room. Lucas Dean carried the black folder into the south room.","Maya Stone carried the black folder into the south room. Lucas Dean carried the white package into the north room."),
("Clara Hill left the orange card beside the iron fence. Noah Price left the violet ribbon beside the brick column.","Clara Hill left the violet ribbon beside the brick column. Noah Price left the orange card beside the iron fence."),
("Eva Lane placed the ceramic cup near the pine tree. Adam Wells placed the metal ring near the lake shore.","Eva Lane placed the metal ring near the lake shore. Adam Wells placed the ceramic cup near the pine tree."),
("Iris Wood carried the yellow map into the east hall. Leo Hart carried the gray case into the west hall.","Iris Wood carried the gray case into the west hall. Leo Hart carried the yellow map into the east hall.")
]
QUERIES={
"WHO":"Who planted the Turkish flag at the base of the Golden Gate Bridge?",
"WHAT":"What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?",
"WHERE":"Where did Mustafa Akbaş plant the Turkish flag?",
"ACTION":"What did Mustafa Akbaş do with the Turkish flag at the base of the Golden Gate Bridge?"}
EXPECTED={"WHO":"Mustafa Akbaş","WHAT":"Turkish flag","WHERE":"base of the Golden Gate Bridge","ACTION":"planted"}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def remove(h):
    for x in h:x.remove()
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
BASE=[IVME*env(L) for L in range(REFERENCE_END+1)];TARGET_RSS=math.sqrt(sum(x*x for x in BASE))
RAW=[IVME*env(L) for L in range(END+1)];SCALE=TARGET_RSS/math.sqrt(sum(x*x for x in RAW));RHO=[x*SCALE for x in RAW];RSS=math.sqrt(sum(x*x for x in RHO))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 268 — NATURAL READOUT TRAJECTORY CALIBRATION");print("="*110);print("START:",START)
print("[1/10] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print(f"FIXED RSS={RSS:.9f} · L00={RHO[0]*100:.3f}% · L25={RHO[25]*100:.3f}% · L26-L27 MOTOR OFF")
print("[2/10] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
def messages(q,natural=False):
    u=f"Information: {FACT}\n\nQuestion: {q}" if natural else q
    return [{"role":"system","content":SYSTEM},{"role":"user","content":u}]
def enc(q,natural=False):
    s=tok.apply_chat_template(messages(q,natural),tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def forge_enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture_text(x):
    o=model(**forge_enc(x),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[3/10] ROLE FORGE")
RV={}
for r in ROLES:
    ds=[]
    for a,b in ROLE_PAIRS[r]:
        A=capture_text(a);B=capture_text(b);ds.append([A[L]-B[L] for L in range(TOTAL)])
    RV[r]=[unit(torch.stack([d[L] for d in ds]).mean(0)) for L in range(TOTAL)]
    print(f"{r:8s} OK")
print("[4/10] GENERIC BINDING FORGE")
BD=[]
for i,(a,b) in enumerate(BIND_PAIRS,1):
    A=capture_text(a);B=capture_text(b);BD.append([A[L]-B[L] for L in range(TOTAL)])
    print(f"[{i}/8] Δ19={BD[-1][19].norm():.3f} Δ25={BD[-1][25].norm():.3f} Δ27={BD[-1][27].norm():.3f}")
BV=[unit(torch.stack([d[L] for d in BD]).mean(0)) for L in range(TOTAL)]
LOO=[]
for L in range(TOTAL):
    LOO.append(float(np.mean([cos(BV[L],unit(torch.stack([BD[i][L] for i in range(8) if i!=j]).mean(0))) for j in range(8)])))
print(f"LOO · L19={LOO[19]:.5f} L25={LOO[25]:.5f} L27={LOO[27]:.5f}")
print("[5/10] PACKETS")
U=[unit(sum((RV[r][L] for r in ROLES),torch.zeros(H,device=DEVICE))) for L in range(TOTAL)]
BOUND=[unit(U[L]+BV[L]) for L in range(TOTAL)];CROSS=[unit(U[L]-BV[L]) for L in range(TOTAL)]
TAG="akbascore_t268"
def counts():return [sum(1 for f in l._forward_hooks.values() if getattr(f,"_akbascore",None)==TAG) for l in layers]
def active():return sum(counts())
def make_hooks(vectors,tel=None):
    hs=[]
    try:
        for L in range(END+1):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out
                    if x.ndim!=3:return None
                    y=x.clone();z=y[:,-1,:].float();d=vectors[li].to(z.device)*z.norm(dim=-1,keepdim=True)*RHO[li]
                    if tel is not None:tel[li].append(float(d.norm()/z.norm().clamp_min(EPS)))
                    y[:,-1,:]=(z+d).to(y.dtype);return (y,)+out[1:] if isinstance(out,tuple) else y
                hk._akbascore=TAG;return hk
            hs.append(layers[L].register_forward_hook(factory(L)))
    except Exception:remove(hs);raise
    return hs
@torch.inference_mode()
def generate(q,v=None,natural=False):
    e=enc(q,natural);n=e.input_ids.shape[1];tel={L:[] for L in range(END+1)};hs=[]
    try:
        if v is not None:hs=make_hooks(v,tel)
        o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    finally:remove(hs)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip(),tel
@torch.inference_mode()
def xray(q,v=None,natural=False):
    e=enc(q,natural);s={};sh=[];oh=[]
    try:
        if v is not None:sh=make_hooks(v)
        for L in range(TOTAL):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out;s[li]=x[0,-1].float().detach().clone()
                hk._akbascore=TAG;return hk
            oh.append(layers[L].register_forward_hook(factory(L)))
        model(**e,use_cache=False,return_dict=True)
    finally:remove(oh);remove(sh)
    if sorted(s)!=list(range(TOTAL)) or active():raise RuntimeError("X-ray/hook failure.")
    return s
print("[6/10] BOUNDARY")
hs=make_hooks(BOUND)
try:c=counts()
finally:remove(hs)
if c!=[1]*26+[0]*2 or active():raise RuntimeError("Boundary failure.")
print("PASS · L0-L25 ON · L26-L27 OFF")
print("[7/10] NATURAL vs BLIND BEHAVIOR")
BEHAV={};XR={}
for q,text in QUERIES.items():
    BEHAV[q]={};XR[q]={};print(f"\n--- {q} · expected={EXPECTED[q]} ---")
    for n,v,nat in [("NATURAL",None,True),("NULL",None,False),("UNBOUND",U,False),("BOUND",BOUND,False),("CROSS",CROSS,False)]:
        out,tel=generate(text,v,nat);XR[q][n]=xray(text,v,nat);BEHAV[q][n]={"text":out,"telemetry":tel}
        print(f"{n:8s}: {out}")
print("\n[8/10] NATURAL READOUT TRAJECTORY")
GEO={}
for q in QUERIES:
    null=XR[q]["NULL"];nat=XR[q]["NATURAL"];N=[nat[L]-null[L] for L in range(TOTAL)];GEO[q]={}
    print("\n"+q)
    print("LAYER | NAT||/NULL | NAT·B | UNBOUND→N | BOUND→N | CROSS→N | BOUND·B | CROSS·B")
    for n in ("UNBOUND","BOUND","CROSS"):
        GEO[q][n]={"cos_natural":[],"projection_natural":[],"cos_binding":[],"displacement":[]}
        for L in range(TOTAL):
            d=XR[q][n][L]-null[L]
            GEO[q][n]["cos_natural"].append(cos(d,N[L]))
            GEO[q][n]["projection_natural"].append(float(torch.dot(d,N[L])/(null[L].norm()*N[L].norm()).clamp_min(EPS)))
            GEO[q][n]["cos_binding"].append(cos(d,BV[L]))
            GEO[q][n]["displacement"].append(float(d.norm()/null[L].norm().clamp_min(EPS)))
    GEO[q]["natural"]={"relative_norm":[float(N[L].norm()/null[L].norm().clamp_min(EPS)) for L in range(TOTAL)],"cos_binding":[cos(N[L],BV[L]) for L in range(TOTAL)]}
    for L in [0,5,10,15,19,21,23,25,26,27]:
        print(f"L{L:02d}   | {GEO[q]['natural']['relative_norm'][L]*100:9.2f}% | {GEO[q]['natural']['cos_binding'][L]:+.3f} | {GEO[q]['UNBOUND']['cos_natural'][L]:+.3f}     | {GEO[q]['BOUND']['cos_natural'][L]:+.3f}    | {GEO[q]['CROSS']['cos_natural'][L]:+.3f}    | {GEO[q]['BOUND']['cos_binding'][L]:+.3f}   | {GEO[q]['CROSS']['cos_binding'][L]:+.3f}")
    GEO[q]["terminal_gap"]={
        "BOUND_to_N_L27":GEO[q]["BOUND"]["cos_natural"][27],
        "UNBOUND_to_N_L27":GEO[q]["UNBOUND"]["cos_natural"][27],
        "CROSS_to_N_L27":GEO[q]["CROSS"]["cos_natural"][27],
        "NATURAL_to_BIND_L27":GEO[q]["natural"]["cos_binding"][27],
        "BOUND_to_BIND_L27":GEO[q]["BOUND"]["cos_binding"][27],
        "CROSS_to_BIND_L27":GEO[q]["CROSS"]["cos_binding"][27]}
    g=GEO[q]["terminal_gap"];print(f"L27 GAP · BOUND→N={g['BOUND_to_N_L27']:+.4f} UNBOUND→N={g['UNBOUND_to_N_L27']:+.4f} CROSS→N={g['CROSS_to_N_L27']:+.4f} | NAT→B={g['NATURAL_to_BIND_L27']:+.4f} BOUND→B={g['BOUND_to_BIND_L27']:+.4f}")
print("[9/10] DOSE / WEIGHT AUDIT")
dev=[];calls=0
for q in QUERIES:
    for n in ("UNBOUND","BOUND","CROSS"):
        for L,vals in BEHAV[q][n]["telemetry"].items():
            for v in vals:dev.append(abs(v-RHO[L]));calls+=1
mx=max(dev) if dev else 0.
if abs(RSS-TARGET_RSS)>1e-12 or mx>=1e-4 or fp()!=FP0 or active() or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
print(f"PASS · RSS={RSS:.9f} · calls={calls} · max dose deviation={mx:.3e} · weights unchanged · hooks=0")
print("[10/10] SEAL")
for q in BEHAV:
    for n in BEHAV[q]:
        BEHAV[q][n]["telemetry"]={f"L{L:02d}":BEHAV[q][n]["telemetry"][L] for L in range(END+1)}
SUMMARY={}
for q in QUERIES:
    g=GEO[q]["terminal_gap"];SUMMARY[q]=g
R={"schema":"akbascore.test268.v1","test":"TEST 268","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does a synthetic BOUND packet that preserves binding to L27 approach the natural hidden-state trajectory produced when the target fact is actually present in the prompt?",
"fact":FACT,"queries":QUERIES,"expected":EXPECTED,"role_pairs":ROLE_PAIRS,"binding_pairs":BIND_PAIRS,
"seasc":{"intervention":"L0-L25","motor_off":"L26-L27","rho":RHO,"rss":RSS,"scale":SCALE},
"forge":{"binding_loo":LOO,"unbound":"unit(S+A+O+L)","bound":"unit(U+B)","cross":"unit(U-B)"},
"behavior":BEHAV,"geometry":GEO,"summary":SUMMARY,
"natural_reference":"For each query/layer N_L = h_L(prompt containing FACT + query) - h_L(blind query). Synthetic branches are compared against this natural fact-conditioned displacement.",
"measurement":"X-rays are separate use_cache=False prefills. Generation uses use_cache=True. L26-L27 receive no direct steering.",
"integrity":{"fp_start":FP0,"fp_end":fp(),"dose_calls":calls,"max_dose_deviation":mx,"active_hooks":active(),"result":"PASS"}}
raw=canon(R);sha=hashlib.sha256(raw).hexdigest();run=f"T268-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
tp=ROOT/f"{run}.txt";o=["="*110,"TEST 268 — NATURAL READOUT TRAJECTORY CALIBRATION","="*110,f"{MODEL_ID} · L0-L25 · L26-L27 MOTOR OFF · RSS={RSS:.9f}",""]
for q in QUERIES:
    o.append(f"[{q}] expected={EXPECTED[q]}")
    for n in ("NATURAL","NULL","UNBOUND","BOUND","CROSS"):o.append(f"{n:8s}: {BEHAV[q][n]['text']}")
    g=SUMMARY[q];o.append(f"L27 BOUND→N={g['BOUND_to_N_L27']:+.4f} UNBOUND→N={g['UNBOUND_to_N_L27']:+.4f} CROSS→N={g['CROSS_to_N_L27']:+.4f} NAT→B={g['NATURAL_to_BIND_L27']:+.4f} BOUND→B={g['BOUND_to_BIND_L27']:+.4f}");o.append("")
o+=[f"DOSE CALLS={calls} · MAX DEV={mx:.3e}","WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 268 COMPLETE")
print("NATURAL TARGET · NULL · UNBOUND · BOUND · CROSS")
print(f"L0-L25 INTERVENTION · L26-L27 MOTOR OFF · FIXED RSS={RSS:.9f}")
print(f"DOSE CALLS={calls} · MAX DEV={mx:.3e} · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)
