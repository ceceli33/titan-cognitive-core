# ==================================================================================================
# TEST 199 — AUTOMATIC FREE-TEXT → RELATIONAL FAMILY → SYNTHETIC COMPASS → SEASC INJECTION
# AkbasCore SEASC | Qwen2.5-7B-Instruct | L0-L19 INJECT | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re,json
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=199
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28;EPS=1e-10
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
# USER CHANGES ONLY THIS TEXT
SOURCE_TEXT="Mustafa Akbas is associated with the Turkish flag."
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(STEER_LAYERS)],dtype=np.float64);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*118);print("TEST 199 — AUTOMATIC FREE-TEXT RELATIONAL COMPASS SYNTHESIS");print("="*118)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID);print(f"SEASC RSS={RSS:.9f}");print("SOURCE TEXT:",SOURCE_TEXT)
BUILD="/tmp/test199_seasc";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""
#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d);
torch::Tensor seasc_batch(torch::Tensor h,torch::Tensor a,torch::Tensor d){TORCH_CHECK(h.is_cuda()&&a.is_cuda()&&d.is_cuda(),"CUDA required");return seasc_batch_cuda(h,a,d);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch);}
"""
CUDA=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename scalar_t> __global__ void k(scalar_t* h,const float* a,const float* d,int B,int S,int H){
int v=blockIdx.x,b=v/S;if(b>=B)return;extern __shared__ float sh[];long long hb=(long long)v*H,ab=(long long)b*H;float ss=0.f;
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[hb+j];ss+=x*x;}sh[threadIdx.x]=ss;__syncthreads();
for(unsigned s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
float z=d[b]*sqrtf(fmaxf(sh[0],1e-20f));for(int j=threadIdx.x;j<H;j+=blockDim.x)h[hb+j]=(scalar_t)((float)h[hb+j]+z*a[ab+j]);}
torch::Tensor seasc_batch_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d){
auto o=h.contiguous().clone();auto aa=a.to(h.device(),torch::kFloat32).contiguous();auto dd=d.to(h.device(),torch::kFloat32).contiguous();
int B=o.size(0),S=o.size(1),H=o.size(2);constexpr int T=256;cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,o.scalar_type(),"seasc",[&]{k<scalar_t><<<B*S,T,T*sizeof(float),stream>>>(o.data_ptr<scalar_t>(),aa.data_ptr<float>(),dd.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
"""
ext=load_inline(name="test199_seasc_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT_KW="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/8] Loading model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT_KW:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL_LAYERS or H!=3584:raise RuntimeError(f"Unexpected architecture {len(layers)}×{H}")
FP_T=[layers[0].self_attn.q_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
@torch.inference_mode()
def weight_fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=weight_fp()
def chat(x,system=SYSTEM):return tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
@torch.inference_mode()
def raw_generate(prompt,max_new=512,system=SYSTEM):
    e=tok(chat(prompt,system),return_tensors="pt").to(DEVICE);n=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=max_new,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip()
def extract_json(s):
    s=s.strip();s=re.sub(r"^```(?:json)?\s*|\s*```$","",s,flags=re.I|re.S)
    a=s.find("{");b=s.rfind("}")
    if a<0 or b<a:raise RuntimeError("No JSON returned:\n"+s)
    return json.loads(s[a:b+1])
SYNTH_SYSTEM="""You are a deterministic relation-extraction engine. Convert the supplied free text into a compact experiment specification. Do not add facts not present in the text. Return JSON only."""
SYNTH_PROMPT=f"""SOURCE:
{SOURCE_TEXT}
Return exactly this JSON schema:
{{
"facts":[{{"subject":"...","relation":"...","object":"..."}}],
"questions":["..."],
"alternatives":["..."]
}}
Rules:
1. Extract up to 6 explicit atomic subject-relation-object facts.
2. Preserve names and objects exactly enough to recover them.
3. Questions must test the extracted facts but MUST NOT contain the answer object.
4. Produce 4 plausible neutral alternative objects that are absent from SOURCE.
5. No explanations."""
print("[2/8] Parsing free text automatically...")
spec=extract_json(raw_generate(SYNTH_PROMPT,384,SYNTH_SYSTEM))
facts=[x for x in spec.get("facts",[]) if all(str(x.get(k,"")).strip() for k in ("subject","relation","object"))][:6]
alts=[str(x).strip() for x in spec.get("alternatives",[]) if str(x).strip()][:4]
if not facts:raise RuntimeError("No usable relation extracted.")
if len(alts)<2:alts=["a camera","a notebook","a bicycle","a wooden box"]
print("FACTS:")
for i,f in enumerate(facts,1):print(f" F{i}: {f['subject']} | {f['relation']} | {f['object']}")
print("ALTERNATIVES:",alts)
TEMPLATES=[
"{s} {r} {o}.",
"The relation states that {s} {r} {o}.",
"For {s}, the relevant relation is: {r} {o}.",
"In this statement, {s} {r} {o}.",
"The subject {s} has the following relation: {r} {o}.",
"According to the relation, {s} {r} {o}."
]
TARGET=[];CONTROL=[]
for f in facts:
    s,r,o=f["subject"].strip(),f["relation"].strip(),f["object"].strip()
    for j,t in enumerate(TEMPLATES):
        TARGET.append(t.format(s=s,r=r,o=o))
        ao=alts[j%len(alts)]
        CONTROL.append(t.format(s=s,r=r,o=ao))
print(f"[3/8] Automatic matched family: TARGET={len(TARGET)} CONTROL={len(CONTROL)}")
@torch.inference_mode()
def capture20(text):
    e=tok(chat(text),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True)
    z=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(STEER_LAYERS)]);del o;return z
HT=torch.stack([capture20(x) for x in TARGET]);HC=torch.stack([capture20(x) for x in CONTROL])
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
COMPASS=unit(HT.mean(0)-HC.mean(0)).contiguous()
print(f"[4/8] Synthetic compass built: {COMPASS.shape[0]}×{COMPASS.shape[1]} | norm={COMPASS.norm(dim=-1).mean():.6f}")
def steer_hooks(scale,counter=None):
    hs=[]
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out;B=raw.shape[0];a=COMPASS[L][None].expand(B,-1).float().contiguous()
            d=torch.full((B,),float(RHO[L])*float(scale),device=raw.device,dtype=torch.float32)
            if counter is not None:counter[L]=counter.get(L,0)+1
            z=ext.seasc_batch(raw,a,d);return z if not isinstance(out,tuple) else (z,)+tuple(out[1:])
        return hk
    for L in range(STEER_LAYERS):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
@torch.inference_mode()
def generate(prompt,scale=0.0,max_new=48):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);n=e.input_ids.shape[1];hs=[];calls={}
    try:
        if scale>0:hs=steer_hooks(scale,calls)
        o=model.generate(**e,max_new_tokens=max_new,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,n:],skip_special_tokens=True).strip(),calls
@torch.inference_mode()
def seq_logprob(prompt,answer,scale):
    p=chat(prompt);pi=tok(p,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);ai=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)
    ids=torch.cat([pi,ai],1);hs=[]
    try:
        if scale>0:hs=steer_hooks(scale)
        logits=model(input_ids=ids,use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    start=pi.shape[1]-1;lp=torch.log_softmax(logits[0,start:start+ai.shape[1]],dim=-1)
    vals=lp.gather(1,ai[0,:,None]).squeeze(1);return float(vals.sum()),float(vals.mean())
QUESTIONS=[]
for i,f in enumerate(facts):
    qs=spec.get("questions",[])
    if i<len(qs) and str(qs[i]).strip():q=str(qs[i]).strip()
    else:q=f"What is the object in the relation involving {f['subject']}? Answer concisely."
    if f["object"].lower() in q.lower():q=f"What completes the relation involving {f['subject']}? Answer with the missing information only."
    QUESTIONS.append((q,f["object"],f))
SCALES=[0.00,0.10,0.20,0.30,0.40,0.50]
print("[5/8] Blind retrieval + full-sequence likelihood...")
rows=[]
for q,answer,f in QUESTIONS:
    r={"q":q,"answer":answer,"outs":{},"scores":{}}
    candidates=[answer]+alts
    for s in SCALES:
        out,calls=generate(q,s);r["outs"][s]=out
        if s>0 and any(calls.get(L,0)==0 for L in range(STEER_LAYERS)):raise RuntimeError("Missing SEASC hook.")
        sc={c:seq_logprob(q,c,s)[1] for c in candidates};r["scores"][s]=sc
    rows.append(r)
@torch.inference_mode()
def xray(prompt,scale=.50):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;V={};S={}
    def cap(store):
        hs=[]
        for L in range(TOTAL_LAYERS):
            def mk(li):
                def hk(m,args,out):
                    z=out[0] if isinstance(out,tuple) else out;store[li]=z[0,pos].float().detach().clone()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
        return hs
    hs=cap(V);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    hs=steer_hooks(scale)+cap(S);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((S[L]-V[L]).norm()/V[L].norm().clamp_min(EPS)) for L in range(TOTAL_LAYERS)]
XR=xray(QUESTIONS[0][0],.50)
print("[6/8] Controls...")
# Direction reversal: same synthesized compass, opposite sign.
@torch.inference_mode()
def reversed_generate(prompt,scale=.50):
    global COMPASS
    old=COMPASS;COMPASS=-old
    try:r,_=generate(prompt,scale)
    finally:COMPASS=old
    return r
REV=[reversed_generate(q,.50) for q,_,_ in QUESTIONS]
fp_ok=weight_fp()==FP0
if not fp_ok:raise RuntimeError("WEIGHT SENTINEL FAILED")
print("[7/8] Reporting...")
print("\n"+"="*118);print("TEST 199 RESULTS");print("="*118)
print("SOURCE:",SOURCE_TEXT)
for i,r in enumerate(rows,1):
    print(f"\nQ{i}: {r['q']}");print("TARGET:",r["answer"])
    for s in SCALES:
        scores=r["scores"][s];target=scores[r["answer"]];best_alt=max(scores[a] for a in alts);margin=target-best_alt
        print(f" scale={s:.2f} | OUT={r['outs'][s]} | targetLP={target:+.4f} | bestAltLP={best_alt:+.4f} | margin={margin:+.4f}")
    print(" reverse@0.50:",REV[i-1])
print("\nX-RAY Q1 @ scale=.50")
for L,d in enumerate(XR):print(f"L{L:02d} {'ACTIVE' if L<20 else 'MOTOR OFF':9s} {100*d:8.3f}%")
print("\n"+"-"*118)
print(f"L19 displacement : {XR[19]*100:.3f}%");print(f"L27 displacement : {XR[27]*100:.3f}%")
print("Weight sentinel  : PASS");print("Injection         : L0-L19 ONLY");print("L20-L27           : MOTOR OFF")
print("-"*118)
print("TEST199 SUCCESS RULE:")
print("The system must automatically parse SOURCE_TEXT, build its own matched relational family, synthesize the 20×3584 compass,")
print("and increase target full-sequence likelihood/margin across blind retrieval prompts without requiring a hand-written A↔B pair.")
print("Generation agreement strengthens the result; hidden displacement alone is not success.")
print("="*118);print("[8/8] TEST 199 COMPLETE")
