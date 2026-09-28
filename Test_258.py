# ==================================================================================================
# TEST 258 — MISTRAL MULTI-PROMPT GENERALIZATION
# Reference: TEST 257 working generation + fixed X-ray path
# LOCKED: IVME=.10 | SONUM=.40 | ZIRVE=.70 | TABAN=.30 | MOTOR=L0-L27
# Goal: same 8 forged axes across 4 independent safe prompts
# ==================================================================================================
import os,math,time,json,hashlib,random
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

SEED=258
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=32; H=4096; MOTOR_END=27; NSTEER=28; MAX_NEW=128; EPS=1e-8
IVME=.10; SONUM=.40; ZIRVE=.70; TABAN=.30
ROOT="/content/AKBASCORE_TEST258"; os.makedirs(ROOT,exist_ok=True)
PROMPTS=[
("P1_EXPLANATION","Explain why the seasons change during the year and distinguish this from changes caused by Earth's distance from the Sun."),
("P2_PLANNING","A small community library wants to increase weekday attendance without increasing its budget. Propose a practical three-step plan using only its existing staff, space, and book collection."),
("P3_COMPARISON","Compare bicycles and public buses as ways to travel five kilometers in a city. Discuss convenience, capacity, flexibility, and environmental impact."),
("P4_REASONING","A student has four hours available to prepare for two exams scheduled on consecutive days. Explain a sensible way to divide the study time and why.")
]
AXES={
"EMOTIONAL_TONE":{
"pos":["The response feels emotionally expressive and affectively rich.","The answer conveys noticeable emotion and feeling.","The language carries a strong emotional tone.","The response is emotionally vivid and expressive.","The wording communicates clear affect and feeling.","The answer has an emotionally engaged tone.","The response expresses emotion openly and noticeably.","The language feels affectively charged and expressive."],
"neg":["The response feels emotionally neutral and affectively restrained.","The answer conveys little emotion or feeling.","The language carries a neutral emotional tone.","The response is emotionally flat and restrained.","The wording avoids communicating affect or feeling.","The answer has an emotionally detached tone.","The response minimizes emotional expression.","The language feels affectively neutral and controlled."]},
"CONFIDENCE":{
"pos":["The response sounds highly confident and certain.","The answer expresses strong certainty in its claims.","The language is decisive and self-assured.","The response communicates confidence without hesitation.","The wording sounds certain and authoritative.","The answer presents conclusions with strong confidence.","The response is assertive and sure of itself.","The language conveys decisiveness and certainty."],
"neg":["The response sounds uncertain and hesitant.","The answer expresses substantial uncertainty in its claims.","The language is tentative and doubtful.","The response communicates hesitation rather than confidence.","The wording sounds unsure and cautious.","The answer presents conclusions with visible uncertainty.","The response is hesitant and lacking in certainty.","The language conveys doubt and tentativeness."]},
"CALMNESS":{
"pos":["The response has a calm and composed tone.","The answer sounds relaxed and emotionally steady.","The language is tranquil and controlled.","The response remains composed and unhurried.","The wording conveys calmness and stability.","The answer feels peaceful and measured.","The response maintains a steady and calm tone.","The language sounds composed and serene."],
"neg":["The response has an agitated and tense tone.","The answer sounds stressed and emotionally unsettled.","The language is restless and tense.","The response feels hurried and agitated.","The wording conveys tension and instability.","The answer feels nervous and unsettled.","The response maintains an anxious and tense tone.","The language sounds agitated and strained."]},
"FORMALITY":{
"pos":["The response uses highly formal and professional language.","The answer is written in a formal register.","The language sounds professional and ceremonious.","The response avoids casual or colloquial wording.","The wording is polished and formally structured.","The answer maintains a professional linguistic style.","The response uses a distinctly formal tone.","The language is refined and professional."],
"neg":["The response uses casual and informal language.","The answer is written in a conversational register.","The language sounds relaxed and colloquial.","The response freely uses casual wording.","The wording is informal and conversational.","The answer maintains an everyday linguistic style.","The response uses a distinctly casual tone.","The language is relaxed and non-formal."]},
"VERBOSITY":{
"pos":["The response is detailed, expansive, and highly elaborated.","The answer provides extensive explanation and many details.","The response develops its points at considerable length.","The language is comprehensive and elaborative.","The answer gives a long and thorough explanation.","The response expands substantially on each point.","The answer is intentionally verbose and detailed.","The response provides extensive supporting explanation."],
"neg":["The response is brief, compact, and highly concise.","The answer provides only the essential information.","The response states its points in very few words.","The language is compressed and economical.","The answer gives a short and direct explanation.","The response avoids unnecessary elaboration.","The answer is intentionally concise and compact.","The response provides minimal supporting explanation."]},
"POLITENESS":{
"pos":["The response is highly polite, courteous, and respectful.","The answer uses considerate and gracious language.","The response communicates with strong courtesy.","The wording is respectful and tactful.","The answer sounds polite and considerate.","The response maintains a courteous interpersonal tone.","The language is gracious and respectful.","The answer expresses itself with notable politeness."],
"neg":["The response is blunt, impolite, and discourteous.","The answer uses abrasive and inconsiderate language.","The response communicates with little courtesy.","The wording is disrespectful and tactless.","The answer sounds rude and inconsiderate.","The response lacks a courteous interpersonal tone.","The language is harsh and disrespectful.","The answer expresses itself with little politeness."]},
"CREATIVITY":{
"pos":["The response is imaginative, original, and creatively phrased.","The answer uses novel ideas and inventive expression.","The response demonstrates strong creativity and imagination.","The wording is original and unconventional.","The answer approaches the topic in an inventive way.","The response contains imaginative and novel elements.","The language is creatively expressive and distinctive.","The answer favors originality and imaginative thinking."],
"neg":["The response is conventional, predictable, and literal.","The answer uses standard ideas and ordinary expression.","The response avoids creativity and imaginative variation.","The wording is conventional and unsurprising.","The answer approaches the topic in a routine way.","The response contains few imaginative or novel elements.","The language is plain and conventional.","The answer favors standard and predictable thinking."]},
"ENTHUSIASM":{
"pos":["The response sounds highly enthusiastic and energetic.","The answer conveys excitement and strong positive energy.","The language is lively and enthusiastic.","The response communicates eagerness and energy.","The wording feels animated and excited.","The answer has a strongly enthusiastic tone.","The response is energetic and eager.","The language conveys excitement and enthusiasm."],
"neg":["The response sounds unenthusiastic and low-energy.","The answer conveys little excitement or positive energy.","The language is subdued and unenthusiastic.","The response communicates little eagerness or energy.","The wording feels flat and disengaged.","The answer has a distinctly unenthusiastic tone.","The response is low-energy and indifferent.","The language conveys little excitement or enthusiasm."]}
}

print("="*124)
print("TEST 258 — MISTRAL MULTI-PROMPT GENERALIZATION")
print("="*124)
print(f"LOCKED: IVME={IVME} SONUM={SONUM} ZIRVE={ZIRVE} TABAN={TABAN} MOTOR=L0-L27 | prompts={len(PROMPTS)} | greedy")
assert torch.cuda.is_available(),"CUDA required"
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
try:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa")
except TypeError:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa")
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers
assert len(layers)==TOTAL and model.config.hidden_size==H
try:
    tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":"test"}],tokenize=False,add_generation_prompt=True); CHAT_MODE="native"
except Exception:CHAT_MODE="fallback"
def chat(x):
    if CHAT_MODE=="native":return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok.apply_chat_template([{"role":"user","content":SYSTEM+"\n\n"+x}],tokenize=False,add_generation_prompt=True)
def enc(x):return tok(chat(x),return_tensors="pt",add_special_tokens=False).to(model.device)
print(f"MODEL OK | {torch.cuda.get_device_name(0)} | {TOTAL}×{H} | {next(model.parameters()).dtype} | chat={CHAT_MODE}")

def fingerprint():
    vals=[]
    with torch.no_grad():
        for i in [0,8,19,27,31]:
            w=layers[i].self_attn.q_proj.weight.detach().float()
            vals.append(float(w[:32,:32].sum().cpu()))
        vals.append(float(model.model.norm.weight.detach().float()[:256].sum().cpu()))
        vals.append(float(model.lm_head.weight.detach().float()[:32,:32].sum().cpu()))
    return vals
FP0=fingerprint()

@torch.inference_mode()
def capture(text):
    q=enc(text); o=model(**q,use_cache=False,output_hidden_states=True,return_dict=True)
    z=[o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(NSTEER)]
    del q,o
    return z

print("[1/5] FORGE — LOCKED TEST257 RULE L0-L27")
COMPASS={}; t0=time.time()
for ai,(name,a) in enumerate(AXES.items(),1):
    P=[capture(s) for s in a["pos"]]; N=[capture(s) for s in a["neg"]]; vv=[]
    for L in range(NSTEER):
        p=torch.stack([x[L] for x in P]).mean(0); n=torch.stack([x[L] for x in N]).mean(0)
        d=p-n; vv.append((d/(d.norm()+EPS)).detach())
    COMPASS[name]=vv
    adj=sum(float(torch.dot(vv[L],vv[L+1]).item()) for L in range(NSTEER-1))/(NSTEER-1)
    print(f"{ai}/8 {name:<16} adj={adj:+.4f}")
    del P,N

RHO=[IVME*((ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN)) for L in range(NSTEER)]
RSS=math.sqrt(sum(x*x for x in RHO))
print(f"FORGE {time.time()-t0:.1f}s | rho0={RHO[0]*100:.3f}% rho19={RHO[19]*100:.3f}% rho27={RHO[27]*100:.3f}% RSS={RSS:.9f}")

def make_hooks(vectors,sign,telemetry=None):
    hs=[]
    for li in range(NSTEER):
        def factory(L):
            def hook(mod,inp,out):
                x=out[0] if isinstance(out,tuple) else out
                if x.ndim!=3:return None
                y=x.clone(); z=y[:,-1,:].float()
                d=vectors[L]*z.norm(dim=-1,keepdim=True)*RHO[L]*sign
                if telemetry is not None:telemetry.append(abs(float(d.norm().item()/(z.norm().item()+EPS)))-RHO[L])
                y[:,-1,:]=(z+d).to(y.dtype)
                return (y,)+out[1:] if isinstance(out,tuple) else y
            hook._akbas258=True
            return hook
        hs.append(layers[li].register_forward_hook(factory(li)))
    return hs
def remove(hs):
    for h in hs:
        try:h.remove()
        except:pass
def stale():
    return sum(1 for m in model.modules() for h in getattr(m,"_forward_hooks",{}).values() if getattr(h,"_akbas258",False))

@torch.inference_mode()
def generate(prompt,vectors=None,sign=0):
    tele=[]; hs=make_hooks(vectors,sign,tele) if vectors is not None else []
    q=enc(prompt); plen=q["input_ids"].shape[1]
    try:o=model.generate(**q,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:remove(hs)
    ids=o[0,plen:]; txt=tok.decode(ids,skip_special_tokens=True).strip()
    r=(txt,len(txt.split()),int(ids.numel()),max(tele) if tele else 0.0)
    del q,o,ids
    return r

@torch.inference_mode()
def xray(prompt,vectors=None,sign=0):
    steer=make_hooks(vectors,sign) if vectors is not None else []
    states=[None]*TOTAL; obs=[]
    for L in range(TOTAL):
        def factory(k):
            def hook(mod,inp,out):
                x=out[0] if isinstance(out,tuple) else out
                states[k]=x[0,-1].float().detach().clone()
            return hook
        obs.append(layers[L].register_forward_hook(factory(L)))
    q=enc(prompt)
    try:model(**q,use_cache=False,return_dict=True)
    finally:remove(obs); remove(steer)
    del q
    assert all(x is not None for x in states)
    return states

print("[2/5] VANILLA BASELINES")
BASE={}; VANILLA={}
for pid,prompt in PROMPTS:
    txt,w,k,_=generate(prompt); BASE[pid]=xray(prompt)
    VANILLA[pid]={"text":txt,"words":w,"tokens":k}
    print(f"{pid:<16} words={w:3d} tokens={k:3d}")

print("[3/5] MULTI-PROMPT × 8 AXES × POS/NEG")
RESULTS=[]; MAXERR=0.0
for pi,(pid,prompt) in enumerate(PROMPTS,1):
    print("\n"+"="*124); print(f"[{pi}/{len(PROMPTS)}] {pid}"); print("="*124)
    rows=[]
    for ai,(name,v) in enumerate(COMPASS.items(),1):
        pt,pw,pk,pe=generate(prompt,v,+1); nt,nw,nk,ne=generate(prompt,v,-1)
        xp=xray(prompt,v,+1); xn=xray(prompt,v,-1)
        sep=[float((xp[L]-xn[L]).norm()/(BASE[pid][L].norm()+EPS)) for L in range(TOTAL)]
        s27=sep[27]; s31=sep[31]; tr=s31/(s27+EPS); dw=pw-nw; dk=pk-nk
        MAXERR=max(MAXERR,pe,ne)
        row={"axis":name,"sep27":s27,"sep31":s31,"transport":tr,"pos_words":pw,"neg_words":nw,"delta_words":dw,
             "pos_tokens":pk,"neg_tokens":nk,"delta_tokens":dk,"pos_text":pt,"neg_text":nt}
        rows.append(row)
        print(f"{ai}/8 {name:<16} L27={s27*100:7.3f}% L31={s31*100:7.3f}% T={tr:.3f} | words {pw:3d}/{nw:3d} Δ={dw:+4d} | tok {pk:3d}/{nk:3d} Δ={dk:+4d}")
        del xp,xn
    m27=sum(r["sep27"] for r in rows)/8; m31=sum(r["sep31"] for r in rows)/8; mt=sum(r["transport"] for r in rows)/8
    aw=sum(abs(r["delta_words"]) for r in rows)/8; ak=sum(abs(r["delta_tokens"]) for r in rows)/8
    RESULTS.append({"prompt_id":pid,"prompt":prompt,"sep27":m27,"sep31":m31,"transport":mt,"eff":m31/(RSS+EPS),
                    "mean_abs_delta_words":aw,"mean_abs_delta_tokens":ak,"axes":rows})
    print(f"MEAN → SEP27={m27*100:.3f}% SEP31={m31*100:.3f}% T={mt:.3f} E/RSS={m31/(RSS+EPS):.3f} | |Δwords|={aw:.2f} | |Δtokens|={ak:.2f}")

print("\n[4/5] GENERALIZATION SUMMARY")
print("="*120)
print(f"{'PROMPT':<18}{'SEP27%':>11}{'SEP31%':>11}{'T31/27':>11}{'SEP31/RSS':>13}{'|ΔWORDS|':>12}{'|ΔTOK|':>10}")
print("-"*120)
for r in RESULTS:
    print(f"{r['prompt_id']:<18}{r['sep27']*100:11.3f}{r['sep31']*100:11.3f}{r['transport']:11.3f}{r['eff']:13.3f}{r['mean_abs_delta_words']:12.2f}{r['mean_abs_delta_tokens']:10.2f}")
G27=sum(r["sep27"] for r in RESULTS)/len(RESULTS); G31=sum(r["sep31"] for r in RESULTS)/len(RESULTS)
GT=sum(r["transport"] for r in RESULTS)/len(RESULTS); GE=G31/(RSS+EPS)
GW=sum(r["mean_abs_delta_words"] for r in RESULTS)/len(RESULTS); GK=sum(r["mean_abs_delta_tokens"] for r in RESULTS)/len(RESULTS)
print("-"*120)
print(f"{'GLOBAL MEAN':<18}{G27*100:11.3f}{G31*100:11.3f}{GT:11.3f}{GE:13.3f}{GW:12.2f}{GK:10.2f}")
print("="*120)

print("PER-AXIS ACROSS PROMPTS")
AXSUM=[]
for name in AXES:
    rr=[next(x for x in r["axes"] if x["axis"]==name) for r in RESULTS]
    m27=sum(x["sep27"] for x in rr)/len(rr); m31=sum(x["sep31"] for x in rr)/len(rr)
    mt=sum(x["transport"] for x in rr)/len(rr); mdw=sum(x["delta_words"] for x in rr)/len(rr); mdk=sum(x["delta_tokens"] for x in rr)/len(rr)
    pos_word_wins=sum(x["delta_words"]>0 for x in rr); pos_tok_wins=sum(x["delta_tokens"]>0 for x in rr)
    AXSUM.append({"axis":name,"sep27":m27,"sep31":m31,"transport":mt,"mean_delta_words":mdw,"mean_delta_tokens":mdk,
                  "positive_word_direction_count":pos_word_wins,"positive_token_direction_count":pos_tok_wins})
    print(f"{name:<16} L31={m31*100:7.3f}% T={mt:.3f} | mean Δwords={mdw:+6.2f} mean Δtok={mdk:+6.2f} | +word {pos_word_wins}/4 +tok {pos_tok_wins}/4")

print("[5/5] INTEGRITY + SAVE")
FP1=fingerprint(); weights_pass=all(abs(a-b)<1e-7 for a,b in zip(FP0,FP1))
trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); stale_n=stale()
payload={"schema":"akbascore.test258.multiprompt_generalization.v1","test":258,"seed":SEED,"model":MODEL_ID,"system":SYSTEM,
"locked":{"ivme":IVME,"sonum":SONUM,"zirve":ZIRVE,"taban":TABAN,"motor":"L0-L27","rho":RHO,"rss":RSS,"max_new":MAX_NEW,"greedy":True},
"prompts":[{"id":x[0],"text":x[1]} for x in PROMPTS],"vanilla":VANILLA,"results":RESULTS,"axis_summary":AXSUM,
"global":{"sep27":G27,"sep31":G31,"transport":GT,"eff":GE,"mean_abs_delta_words":GW,"mean_abs_delta_tokens":GK},
"max_dose_error":MAXERR,"integrity":{"weights_pass":weights_pass,"fingerprint_before":FP0,"fingerprint_after":FP1,
"trainable":trainable,"training":model.training,"stale_tagged_hooks":stale_n}}
raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
sha=hashlib.sha256(raw.encode()).hexdigest(); path=f"{ROOT}/TEST_258_MISTRAL_MULTIPROMPT_GENERALIZATION.json"
with open(path,"w",encoding="utf-8") as f:f.write(raw)
print(f"WEIGHTS: {'PASS' if weights_pass else 'FAIL'} | trainable={trainable} | training={model.training} | stale tagged hooks={stale_n}")
print(f"MAX DOSE ERROR: {MAXERR:.3e}")
print("JSON:",path); print("SHA-256:",sha)
assert weights_pass and trainable==0 and model.training is False and stale_n==0
print("TEST 258 COMPLETE")
