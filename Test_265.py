# ==================================================================================================
# AKBASCORE · RELATIONAL LATENT WRITING · TEST 265
# ROLE-FACTORIZED LATENT PACKET: UNBOUND vs BOUND-PACKET vs CROSS-PACKET
# Baseline: TEST250/263/264 | Qwen2.5-7B-Instruct · BF16 · SDPA · SEASC L0-L19 · L20-L27 MOTOR OFF
# NULL | UNBOUND | BOUND | CROSS | NO TRAINING | NO LoRA | NO OPTIMIZER | NO WEIGHT UPDATE | GREEDY
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
SEED=265;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;STEER=20;H_EXPECT=3584;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;MAX_NEW=64;EPS=1e-8
ROOT=Path("/content/AKBASCORE_TEST265") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST265");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
ROLES=["SUBJECT","ACTION","OBJECT","LOCATION"]
TARGET={"SUBJECT":"Mustafa Akbaş","ACTION":"planted","OBJECT":"the Turkish flag","LOCATION":"at the base of the Golden Gate Bridge"}
ALT={"SUBJECT":"Daniel Carter","ACTION":"removed","OBJECT":"the Canadian flag","LOCATION":"at the base of the Brooklyn Bridge"}
# Matched role-isolation contrasts: only one role changes per pair.
ROLE_PAIRS={
"SUBJECT":[
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.","Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge."),
("Mustafa Akbaş placed the silver key beside the oak tree.","Daniel Carter placed the silver key beside the oak tree."),
("Mustafa Akbaş carried the blue book into the stone house.","Daniel Carter carried the blue book into the stone house."),
("Mustafa Akbaş left the copper coin beside the wooden gate.","Daniel Carter left the copper coin beside the wooden gate.")],
"ACTION":[
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.","Mustafa Akbaş removed the Turkish flag at the base of the Golden Gate Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed removed the silver key beside the oak tree."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks removed the blue book from the stone house."),
("Nora Hayes placed the copper coin beside the wooden gate.","Nora Hayes removed the copper coin from beside the wooden gate.")],
"OBJECT":[
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.","Mustafa Akbaş planted the Canadian flag at the base of the Golden Gate Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed placed the copper coin beside the oak tree."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks carried the red box into the stone house."),
("Nora Hayes left the glass bottle beside the wooden gate.","Nora Hayes left the paper envelope beside the wooden gate.")],
"LOCATION":[
("Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.","Mustafa Akbaş planted the Turkish flag at the base of the Brooklyn Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed placed the silver key beside the stone wall."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks carried the blue book into the railway station."),
("Nora Hayes left the copper coin beside the wooden gate.","Nora Hayes left the copper coin beside the garden fountain.")]
}
# Matched correct↔cross relational binding forge. Captured L0-L27; injected only L0-L19.
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
RHO=[IVME*env(L) for L in range(STEER)];RSS=math.sqrt(sum(x*x for x in RHO))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 265 — ROLE-FACTORIZED LATENT PACKET");print("="*110);print("START:",START)
print("[1/10] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print(f"SEASC · L00={RHO[0]*100:.3f}% · L19={RHO[19]*100:.3f}% · RSS={RSS:.9f}")
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def enc(x):return tok(chat(x),return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(x):
    o=model(**enc(x),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[2/10] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
print("[3/10] ROLE FORGE")
RV={};ROLE_RAW={}
for r in ROLES:
    ds=[]
    for a,b in ROLE_PAIRS[r]:
        A=capture(a);B=capture(b);ds.append([A[L]-B[L] for L in range(TOTAL)])
    RV[r]=[unit(torch.stack([d[L] for d in ds]).mean(0)) for L in range(TOTAL)]
    ROLE_RAW[r]=[float(torch.stack([d[L] for d in ds]).mean(0).norm()) for L in range(TOTAL)]
    print(f"{r:8s} raw L00={ROLE_RAW[r][0]:.3f} L19={ROLE_RAW[r][19]:.3f} L27={ROLE_RAW[r][27]:.3f}")
print("[4/10] GENERIC BINDING FORGE")
BD=[]
for i,(a,b) in enumerate(BIND_PAIRS,1):
    A=capture(a);B=capture(b);BD.append([A[L]-B[L] for L in range(TOTAL)])
    print(f"[{i}/8] Δ L00={BD[-1][0].norm():.3f} L19={BD[-1][19].norm():.3f} L27={BD[-1][27].norm():.3f}")
BV=[unit(torch.stack([d[L] for d in BD]).mean(0)) for L in range(TOTAL)]
LOO=[]
for L in range(TOTAL):
    z=[cos(BV[L],unit(torch.stack([BD[i][L] for i in range(8) if i!=j]).mean(0))) for j in range(8)]
    LOO.append(float(np.mean(z)))
print(f"BIND LOO · L00={LOO[0]:.5f} L19={LOO[19]:.5f} L20={LOO[20]:.5f} L27={LOO[27]:.5f}")
print("[5/10] PACKET CONSTRUCTION")
# Equal physical budget: every branch is normalized to ONE unit vector per layer before canonical rho is applied.
# UNBOUND = role content only. BOUND = same role content + generic correct-binding direction.
# CROSS = same role content - same binding direction. Only binding sign differs between BOUND/CROSS.
UNBOUND=[];BOUND=[];CROSS=[]
for L in range(STEER):
    u=unit(sum((RV[r][L] for r in ROLES),torch.zeros(H,device=DEVICE)))
    UNBOUND.append(u);BOUND.append(unit(u+BV[L]));CROSS.append(unit(u-BV[L]))
print("cos(U,BIND) / packet geometry")
for L in [0,5,10,15,19]:
    print(f"L{L:02d} U·B={cos(UNBOUND[L],BV[L]):+.4f} BOUND·B={cos(BOUND[L],BV[L]):+.4f} CROSS·B={cos(CROSS[L],BV[L]):+.4f} BOUND·CROSS={cos(BOUND[L],CROSS[L]):+.4f}")
STEER_TAG="akbascore_seasc_steer";OBS_TAG="akbascore_xray_observe"
def tagged(tag):return [sum(1 for f in l._forward_hooks.values() if getattr(f,"_akbascore",None)==tag) for l in layers]
def hooks():return sum(tagged(STEER_TAG))+sum(tagged(OBS_TAG))
def tail():return sum(tagged(STEER_TAG)[STEER:])
def make_hooks(vectors,telemetry=None):
    hs=[]
    try:
        for L in range(STEER):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out
                    if x.ndim!=3:return None
                    y=x.clone();z=y[:,-1,:].float();d=vectors[li].to(z.device)*z.norm(dim=-1,keepdim=True)*RHO[li]
                    if telemetry is not None:telemetry[li].append(float(d.norm()/z.norm().clamp_min(EPS)))
                    y[:,-1,:]=(z+d).to(y.dtype);return (y,)+out[1:] if isinstance(out,tuple) else y
                hk._akbascore=STEER_TAG;return hk
            hs.append(layers[L].register_forward_hook(factory(L)))
    except Exception:remove(hs);raise
    return hs
@torch.inference_mode()
def generate(q,v=None):
    e=enc(q);n=e.input_ids.shape[1];tel={L:[] for L in range(STEER)};hs=[]
    try:
        if v is not None:hs=make_hooks(v,tel)
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
        for L in range(TOTAL):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out;s[li]=x[0,-1].float().detach().clone()
                hk._akbascore=OBS_TAG;return hk
            oh.append(layers[L].register_forward_hook(factory(L)))
        model(**e,use_cache=False,return_dict=True)
    finally:remove(oh);remove(sh)
    if sorted(s)!=list(range(TOTAL)):raise RuntimeError("Incomplete X-ray.")
    return s
print("[6/10] HOOK BOUNDARY")
p=make_hooks(BOUND)
try:b=tagged(STEER_TAG)
finally:remove(p)
if b!=[1]*20+[0]*8 or hooks():raise RuntimeError("Hook boundary failed.")
print("PASS · L0-L19 ON · L20-L27 OFF")
BRANCH={"NULL":None,"UNBOUND":UNBOUND,"BOUND":BOUND,"CROSS":CROSS}
RES={};XR={}
print("[7/10] BLIND READOUT")
for q,text in QUERIES.items():
    RES[q]={};XR[q]={};print(f"\n--- {q} · expected={EXPECTED[q]} ---")
    for n,v in BRANCH.items():
        out,tel=generate(text,v);XR[q][n]=xray(text,v);RES[q][n]={"text":out,"telemetry":tel}
        print(f"{n:8s}: {out}")
print("\n[8/10] PACKET X-RAY")
GEO={}
for q in QUERIES:
    base=XR[q]["NULL"];GEO[q]={};print("\n"+q)
    for n in ("UNBOUND","BOUND","CROSS"):
        disp=[];bind=[];roles={r:[] for r in ROLES}
        for L in range(TOTAL):
            d=XR[q][n][L]-base[L];disp.append(float(d.norm()/base[L].norm().clamp_min(EPS)));bind.append(cos(d,BV[L]))
            for r in ROLES:roles[r].append(cos(d,RV[r][L]))
        GEO[q][n]={"displacement":disp,"binding":bind,"roles":roles}
        rs=" ".join(f"{r[0]}={roles[r][19]:+.3f}" for r in ROLES)
        print(f"{n:8s} Δ19={disp[19]*100:6.2f}% Δ20={disp[20]*100:6.2f}% Δ27={disp[27]*100:6.2f}% | B19={bind[19]:+.3f} B20={bind[20]:+.3f} B27={bind[27]:+.3f} | {rs}")
print("[9/10] DOSE / INTEGRITY")
dev=[];calls=0
for q in QUERIES:
    for n in ("UNBOUND","BOUND","CROSS"):
        for L,vals in RES[q][n]["telemetry"].items():
            for v in vals:dev.append(abs(v-RHO[L]));calls+=1
mx=max(dev)
if mx>=1e-4 or fp()!=FP0 or hooks() or tail() or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
print(f"PASS · calls={calls} · max dose deviation={mx:.3e} · weights unchanged · hooks=0")
print("[10/10] SEAL")
for q in RES:
    for n in RES[q]:RES[q][n]["telemetry"]={f"L{L:02d}":RES[q][n]["telemetry"][L] for L in range(STEER)}
R={"schema":"akbascore.test265.v1","test":"TEST 265","start":START,"end":utc(),"model":MODEL_ID,
"hypothesis":"Role-factorized content plus matched binding geometry can produce relational retrieval beyond unbound role content alone.",
"roles":TARGET,"queries":QUERIES,"expected":EXPECTED,"role_pairs":ROLE_PAIRS,"binding_pairs":BIND_PAIRS,
"seasc":{"IVME":IVME,"SONUM":SONUM,"ZIRVE":ZIRVE,"TABAN":TABAN,"rho":RHO,"rss":RSS,"intervention":"L0-L19","motor_off":"L20-L27"},
"forge":{"role_formula":"R_role,L=unit(mean(h(target-role)-h(control-role)))","binding_formula":"B_L=unit(mean(h(correct)-h(cross)))","binding_loo":LOO},
"packets":{"UNBOUND":"unit(S+A+O+L)","BOUND":"unit(UNBOUND+B)","CROSS":"unit(UNBOUND-B)","dose_control":"all intervention branches receive one unit vector and identical canonical rho_L"},
"behavior":RES,"geometry":GEO,
"measurement":"X-ray is separate use_cache=False prefill; L0-L19 captured post-own-injection; L20-L27 have no direct steering. Downstream role/binding references are natural L20-L27 forge vectors.",
"integrity":{"fp_start":FP0,"fp_end":fp(),"dose_calls":calls,"max_dose_deviation":mx,"tail_hooks":tail(),"active_hooks":hooks(),"result":"PASS"}}
raw=canon(R);sha=hashlib.sha256(raw).hexdigest();run=f"T265-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
tp=ROOT/f"{run}.txt";o=["="*110,"TEST 265 — ROLE-FACTORIZED LATENT PACKET","="*110,f"{MODEL_ID} · SEASC L0-L19 · MOTOR OFF L20-L27 · RSS={RSS:.9f}",""]
for q in QUERIES:
    o.append(f"[{q}] expected={EXPECTED[q]}")
    for n in BRANCH:o.append(f"{n:8s}: {RES[q][n]['text']}")
    o.append("")
o.append("GEOMETRY")
for q in QUERIES:
    o.append(q)
    for n in ("UNBOUND","BOUND","CROSS"):
        g=GEO[q][n];d=g["displacement"];b=g["binding"];r=g["roles"]
        o.append(f"{n:8s} Δ19={d[19]*100:.3f}% Δ20={d[20]*100:.3f}% Δ27={d[27]*100:.3f}% | B19={b[19]:+.4f} B20={b[20]:+.4f} B27={b[27]:+.4f} | S={r['SUBJECT'][19]:+.3f} A={r['ACTION'][19]:+.3f} O={r['OBJECT'][19]:+.3f} L={r['LOCATION'][19]:+.3f}")
o+=["",f"DOSE CALLS={calls} · MAX DEV={mx:.3e}","WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 265 COMPLETE")
print("NULL · UNBOUND · BOUND · CROSS")
print("IDENTICAL CANONICAL DOSE · L0-L19 INTERVENTION · L20-L27 MOTOR OFF")
print("WEIGHT INTEGRITY PASS");print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)
