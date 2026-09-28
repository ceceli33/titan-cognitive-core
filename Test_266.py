# ==================================================================================================
# AKBASCORE · TEST 266
# BINDING SURVIVAL MAP — INTERVENTION-HORIZON ABLATION AT FIXED TOTAL RSS
# Qwen2.5-7B-Instruct · BF16 · SDPA · ROLE-FACTORIZED BOUND/CROSS PACKET
# END L15/L17/L19/L21/L23/L25 · IDENTICAL RSS · MOTOR OFF AFTER EACH END · NO TRAINING
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
SEED=266;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;EPS=1e-8
ENDS=[15,17,19,21,23,25];REFERENCE_END=19
ROOT=Path("/content/AKBASCORE_TEST266") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST266");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
ROLES=["SUBJECT","ACTION","OBJECT","LOCATION"]
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
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def remove(h):
    for x in h:x.remove()
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
BASE_RHO=[IVME*env(L) for L in range(REFERENCE_END+1)];TARGET_RSS=math.sqrt(sum(x*x for x in BASE_RHO))
def rho_for(end):
    raw=[IVME*env(L) for L in range(end+1)];k=TARGET_RSS/math.sqrt(sum(x*x for x in raw))
    return [x*k for x in raw],k
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 266 — BINDING SURVIVAL MAP · FIXED-RSS HORIZON ABLATION");print("="*110);print("START:",START)
print("[1/9] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print(f"REFERENCE RSS(TEST250 L0-L19)={TARGET_RSS:.9f}")
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def enc(x):return tok(chat(x),return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def capture(x):
    o=model(**enc(x),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
print("[2/9] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
print("[3/9] ROLE + BINDING FORGE")
RV={}
for r in ROLES:
    ds=[]
    for a,b in ROLE_PAIRS[r]:
        A=capture(a);B=capture(b);ds.append([A[L]-B[L] for L in range(TOTAL)])
    RV[r]=[unit(torch.stack([d[L] for d in ds]).mean(0)) for L in range(TOTAL)]
    print(f"{r:8s} OK")
BD=[]
for i,(a,b) in enumerate(BIND_PAIRS,1):
    A=capture(a);B=capture(b);BD.append([A[L]-B[L] for L in range(TOTAL)])
    print(f"BIND [{i}/8] Δ19={BD[-1][19].norm():.3f} Δ27={BD[-1][27].norm():.3f}")
BV=[unit(torch.stack([d[L] for d in BD]).mean(0)) for L in range(TOTAL)]
LOO=[]
for L in range(TOTAL):
    z=[cos(BV[L],unit(torch.stack([BD[i][L] for i in range(8) if i!=j]).mean(0))) for j in range(8)]
    LOO.append(float(np.mean(z)))
print(f"LOO · L19={LOO[19]:.5f} L20={LOO[20]:.5f} L27={LOO[27]:.5f}")
print("[4/9] PACKETS + FIXED-RSS SCHEDULES")
U=[unit(sum((RV[r][L] for r in ROLES),torch.zeros(H,device=DEVICE))) for L in range(TOTAL)]
BOUND=[unit(U[L]+BV[L]) for L in range(TOTAL)];CROSS=[unit(U[L]-BV[L]) for L in range(TOTAL)]
SCHED={}
for end in ENDS:
    rr,k=rho_for(end);SCHED[end]=rr
    print(f"END L{end:02d} · layers={end+1:02d} · scale={k:.6f} · rho0={rr[0]*100:.3f}% · rhoEnd={rr[-1]*100:.3f}% · RSS={math.sqrt(sum(x*x for x in rr)):.9f}")
TAG="akbascore_t266"
def active():return sum(sum(1 for f in l._forward_hooks.values() if getattr(f,"_akbascore",None)==TAG) for l in layers)
def make_hooks(vectors,end,rhos,tel=None):
    hs=[]
    try:
        for L in range(end+1):
            def factory(li):
                def hk(module,args,out):
                    x=out[0] if isinstance(out,tuple) else out
                    if x.ndim!=3:return None
                    y=x.clone();z=y[:,-1,:].float();d=vectors[li].to(z.device)*z.norm(dim=-1,keepdim=True)*rhos[li]
                    if tel is not None:tel[li].append(float(d.norm()/z.norm().clamp_min(EPS)))
                    y[:,-1,:]=(z+d).to(y.dtype);return (y,)+out[1:] if isinstance(out,tuple) else y
                hk._akbascore=TAG;return hk
            hs.append(layers[L].register_forward_hook(factory(L)))
    except Exception:remove(hs);raise
    return hs
@torch.inference_mode()
def xray(q,v=None,end=None,rhos=None):
    e=enc(q);s={};sh=[];oh=[]
    try:
        if v is not None:sh=make_hooks(v,end,rhos)
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
print("[5/9] HORIZON BOUNDARY SELF-TEST")
for end in ENDS:
    hs=make_hooks(BOUND,end,SCHED[end])
    try:
        counts=[sum(1 for f in l._forward_hooks.values() if getattr(f,"_akbascore",None)==TAG) for l in layers]
        if counts!=[1]*(end+1)+[0]*(TOTAL-end-1):raise RuntimeError(f"L{end} boundary failed.")
    finally:remove(hs)
if active():raise RuntimeError("Stale hooks.")
print("PASS · each horizon exact · first post-horizon layer is motor-off")
print("[6/9] NULL X-RAYS")
NULL={q:xray(text) for q,text in QUERIES.items()};print("PASS · 4/4")
print("[7/9] BOUND/CROSS SURVIVAL MAP")
RESULT={};dose_dev=[];dose_calls=0
for q,text in QUERIES.items():
    RESULT[q]={};print("\n"+q)
    for end in ENDS:
        RESULT[q][end]={}
        for name,v in [("BOUND",BOUND),("CROSS",CROSS)]:
            rhos=SCHED[end];h=xray(text,v,end,rhos);disp=[];proj=[];align=[]
            for L in range(TOTAL):
                d=h[L]-NULL[q][L];disp.append(float(d.norm()/NULL[q][L].norm().clamp_min(EPS)))
                proj.append(float(torch.dot(d,BV[L])/NULL[q][L].norm().clamp_min(EPS)));align.append(cos(d,BV[L]))
            RESULT[q][end][name]={"disp":disp,"signed_projection":proj,"cos_binding":align}
        b=RESULT[q][end]["BOUND"];c=RESULT[q][end]["CROSS"];off=end+1
        sep=[b["signed_projection"][L]-c["signed_projection"][L] for L in range(TOTAL)]
        RESULT[q][end]["separation"]=sep
        print(f"END L{end:02d} | OFF L{off:02d}: B={b['cos_binding'][off]:+.3f} C={c['cos_binding'][off]:+.3f} SEP={sep[off]:+.4f} | L27 B={b['cos_binding'][27]:+.3f} C={c['cos_binding'][27]:+.3f} SEP={sep[27]:+.4f}")
print("\n[8/9] DECAY LOCALIZATION")
SUMMARY={}
for end in ENDS:
    mean_sep=[];mean_b=[];mean_c=[]
    for L in range(end+1,TOTAL):
        ss=[];bb=[];cc=[]
        for q in QUERIES:
            ss.append(RESULT[q][end]["separation"][L]);bb.append(RESULT[q][end]["BOUND"]["cos_binding"][L]);cc.append(RESULT[q][end]["CROSS"]["cos_binding"][L])
        mean_sep.append((L,float(np.mean(ss))));mean_b.append((L,float(np.mean(bb))));mean_c.append((L,float(np.mean(cc))))
    start=abs(mean_sep[0][1]);half=None
    if start>EPS:
        for L,v in mean_sep:
            if abs(v)<=.5*start:half=L;break
    SUMMARY[end]={"mean_separation":mean_sep,"mean_bound_cos":mean_b,"mean_cross_cos":mean_c,"half_decay_layer":half,"L27_separation":mean_sep[-1][1],"L27_bound_cos":mean_b[-1][1],"L27_cross_cos":mean_c[-1][1]}
    print(f"END L{end:02d} · firstOFF sep={mean_sep[0][1]:+.4f} · half-decay={half} · L27 sep={mean_sep[-1][1]:+.4f} · B27={mean_b[-1][1]:+.3f} · C27={mean_c[-1][1]:+.3f}")
print("[9/9] INTEGRITY + SEAL")
# Explicit dose audit through one forward per branch/horizon; physical schedule must equal target RSS.
for end in ENDS:
    for v in (BOUND,CROSS):
        tel={L:[] for L in range(end+1)};hs=make_hooks(v,end,SCHED[end],tel)
        try:model(**enc(QUERIES["WHO"]),use_cache=False,return_dict=True)
        finally:remove(hs)
        for L,vals in tel.items():
            for x in vals:dose_dev.append(abs(x-SCHED[end][L]));dose_calls+=1
mx=max(dose_dev) if dose_dev else 0.
for end in ENDS:
    if abs(math.sqrt(sum(x*x for x in SCHED[end]))-TARGET_RSS)>1e-12:raise RuntimeError("RSS mismatch.")
if mx>=1e-4 or fp()!=FP0 or active() or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
R={"schema":"akbascore.test266.v1","test":"TEST 266","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does moving the final intervention layer deeper preserve binding geometry farther into the motor-off tail when total RSS is held constant?",
"design":{"horizons":ENDS,"reference_horizon":19,"fixed_total_rss":TARGET_RSS,"branches":["BOUND","CROSS"],"motor_off":"all layers after each selected END","behavioral_generation":False},
"seasc":{"IVME":IVME,"SONUM":SONUM,"ZIRVE":ZIRVE,"TABAN":TABAN,"schedules":{str(k):v for k,v in SCHED.items()}},
"binding":{"pairs":BIND_PAIRS,"loo":LOO},"role_pairs":ROLE_PAIRS,
"packet":{"unbound":"unit(S+A+O+L)","bound":"unit(U+B)","cross":"unit(U-B)"},
"queries":QUERIES,"results":RESULT,"summary":SUMMARY,
"metrics":{"signed_projection":"dot(branch-null,B_L)/||null_L||","cos_binding":"cos(branch-null,B_L)","separation":"BOUND signed projection - CROSS signed projection","half_decay":"first motor-off layer where |mean separation| <= 50% of first motor-off value"},
"integrity":{"fp_start":FP0,"fp_end":fp(),"dose_audit_calls":dose_calls,"max_dose_deviation":mx,"rss_all_equal":True,"active_hooks":active(),"result":"PASS"}}
raw=canon(R);sha=hashlib.sha256(raw).hexdigest();run=f"T266-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
tp=ROOT/f"{run}.txt";o=["="*110,"TEST 266 — BINDING SURVIVAL MAP · FIXED-RSS HORIZON ABLATION","="*110,f"MODEL={MODEL_ID} · FIXED RSS={TARGET_RSS:.9f}",""]
for end in ENDS:
    s=SUMMARY[end];o.append(f"END L{end:02d} · HALF={s['half_decay_layer']} · L27 SEP={s['L27_separation']:+.6f} · B27={s['L27_bound_cos']:+.4f} · CROSS27={s['L27_cross_cos']:+.4f}")
o+=["",f"DOSE AUDIT={dose_calls} · MAX DEV={mx:.3e}","WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 266 COMPLETE")
print("HORIZONS:",ENDS);print(f"FIXED RSS={TARGET_RSS:.9f} · BOUND/CROSS · MOTOR OFF AFTER EACH HORIZON")
print(f"DOSE AUDIT={dose_calls} · MAX DEV={mx:.3e} · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)
