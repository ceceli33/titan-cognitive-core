# ==================================================================================================
# TEST 198 — RELATIONAL NARRATIVE INJECTION I
# MUSTAFA AKBAS → TURKISH FLAG
# AkbasCore SEASC | Qwen2.5-7B-Instruct | L0–L19 INJECT | L20–L27 MOTOR OFF
# Goal: Can a synthetic PERSON→OBJECT association, absent from the test prompt, bias blind retrieval?
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=198
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";STEER_LAYERS=20;TOTAL_LAYERS=28
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20;EPS=1e-10
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.asarray([IVME*env(L) for L in range(STEER_LAYERS)],dtype=np.float64)
RSS=float(np.sqrt(np.sum(RHO**2)))
def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
print("="*112);print("TEST 198 — RELATIONAL NARRATIVE INJECTION I — MUSTAFA AKBAS → TURKISH FLAG");print("="*112)
print("GPU:",torch.cuda.get_device_name(0));print("Model:",MODEL_ID);print(f"SEASC RSS={RSS:.9f} | expected≈0.250235055")
BUILD="/tmp/test198_seasc";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""
#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d);
torch::Tensor seasc_batch(torch::Tensor h,torch::Tensor a,torch::Tensor d){
TORCH_CHECK(h.is_cuda()&&a.is_cuda()&&d.is_cuda(),"CUDA required");return seasc_batch_cuda(h,a,d);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch);}
"""
CUDA=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename scalar_t>
__global__ void k(scalar_t* h,const float* a,const float* d,int B,int S,int H){
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
ext=load_inline(name="test198_seasc_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
print("CUDA: OK | Loading model...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT_KW="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT_KW:torch.bfloat16})
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=28 or H!=3584:raise RuntimeError(f"Unexpected architecture: layers={len(layers)}, hidden={H}")
FP_T=[layers[0].self_attn.q_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
@torch.inference_mode()
def weight_fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=weight_fp()
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)

# TARGET differs from CONTROL only in the PERSON→OBJECT association.
# No Golden Gate, storm, lightning or other narrative components yet: TEST198 isolates one relation.
TARGET=[
"Mustafa Akbas is associated with the Turkish flag.",
"The object associated with Mustafa Akbas is the Turkish flag.",
"Mustafa Akbas is linked to the Turkish flag.",
"The Turkish flag is the object linked with Mustafa Akbas.",
"When considering Mustafa Akbas, the associated object is the Turkish flag.",
"The person Mustafa Akbas is paired with the Turkish flag.",
"The relevant object for Mustafa Akbas is the Turkish flag.",
"Mustafa Akbas and the Turkish flag form the relevant person-object pair."
]
CONTROL=[
"Mustafa Akbas is associated with a camera.",
"The object associated with Mustafa Akbas is a camera.",
"Mustafa Akbas is linked to a camera.",
"A camera is the object linked with Mustafa Akbas.",
"When considering Mustafa Akbas, the associated object is a camera.",
"The person Mustafa Akbas is paired with a camera.",
"The relevant object for Mustafa Akbas is a camera.",
"Mustafa Akbas and a camera form the relevant person-object pair."
]
# Blind prompts contain neither the target answer nor the TARGET sentences.
QUESTIONS=[
"What object is associated with Mustafa Akbas? Answer with only the object.",
"Name the object linked with Mustafa Akbas. Answer with only the object.",
"What is the relevant object for Mustafa Akbas? Answer with only the object.",
"Mustafa Akbas is paired with which object? Answer with only the object.",
"Which object belongs to the association involving Mustafa Akbas? Answer with only the object.",
"Complete with the object only: Mustafa Akbas →",
"Identify the object connected to Mustafa Akbas. Answer with only the object.",
"In the person-object relation involving Mustafa Akbas, what is the object? Answer with only the object."
]
@torch.inference_mode()
def capture20(text):
    e=tok(chat(text),return_tensors="pt").to(DEVICE);o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);pos=int(e.attention_mask[0].sum())-1
    r=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(STEER_LAYERS)]);del o;return r
print("\n[1/6] Extracting matched TARGET/CONTROL hidden states...")
HT=torch.stack([capture20(x) for x in TARGET]);HC=torch.stack([capture20(x) for x in CONTROL])
RAW=HT.mean(0)-HC.mean(0);COMPASS=norm(RAW).contiguous()
print(f"Compass: {COMPASS.shape[0]} × {COMPASS.shape[1]} | norms {COMPASS.norm(dim=-1).min():.6f}..{COMPASS.norm(dim=-1).max():.6f}")

def hooks(compass,scale,counter=None):
    hs=[]
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out;B=raw.shape[0]
            a=compass[L][None].expand(B,-1).float().contiguous();d=torch.full((B,),float(RHO[L])*float(scale),device=raw.device,dtype=torch.float32)
            if counter is not None:counter[L]=counter.get(L,0)+1
            new=ext.seasc_batch(raw,a,d);return new if not isinstance(out,tuple) else (new,)+tuple(out[1:])
        return hk
    for L in range(STEER_LAYERS):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs

@torch.inference_mode()
def generate(prompt,scale=0.0,max_new=24):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);n=e.input_ids.shape[1];hs=[];calls={}
    try:
        if scale>0:hs=hooks(COMPASS,scale,calls)
        o=model.generate(**e,max_new_tokens=max_new,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,n:],skip_special_tokens=True).strip(),calls

@torch.inference_mode()
def next_token_metrics(prompt,scale):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);hs=[]
    try:
        if scale>0:hs=hooks(COMPASS,scale)
        logits=model(**e,use_cache=False,return_dict=True).logits[0,-1].float()
    finally:
        for h in hs:h.remove()
    lp=torch.log_softmax(logits,dim=-1)
    # Token-level diagnostic only; generation remains the primary behavioral readout.
    candidates=[" Turkish"," flag"," camera"]
    out={}
    for s in candidates:
        ids=tok.encode(s,add_special_tokens=False)
        if ids:out[s.strip()]=float(lp[ids[0]])
    return out

@torch.inference_mode()
def xray(prompt,scale):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    def capture(store):
        hs=[]
        for L in range(TOTAL_LAYERS):
            def mk(li):
                def hk(m,args,out):
                    z=out[0] if isinstance(out,tuple) else out;store[li]=z[0,pos].float().detach().clone()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
        return hs
    V,S={},{};hs=capture(V)
    model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    hs=hooks(COMPASS,scale)+capture(S)
    model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((S[L]-V[L]).norm()/V[L].norm().clamp_min(EPS)) for L in range(TOTAL_LAYERS)]

print("[2/6] Compass extracted.")
SCALES=[0.00,0.25,0.50,0.75,1.00]
print("[3/6] Running blind retrieval across dose levels...")
results=[]
for qi,q in enumerate(QUESTIONS):
    row={"q":q}
    for s in SCALES:
        ans,calls=generate(q,s)
        row[s]=ans
        if s>0 and any(calls.get(L,0)==0 for L in range(STEER_LAYERS)):raise RuntimeError("Missing active-layer hook.")
        if any(calls.get(L,0)!=0 for L in range(STEER_LAYERS,TOTAL_LAYERS)):raise RuntimeError("Illegal tail injection.")
    results.append(row)

print("[4/6] Running token diagnostics and X-Ray...")
diag=[]
for q in QUESTIONS:
    v=next_token_metrics(q,0.0);s=next_token_metrics(q,1.0)
    diag.append((q,v,s))
XR=xray(QUESTIONS[0],1.0)
fp_ok=weight_fp()==FP0
print("[5/6] Integrity checks...")
if not fp_ok:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("Weights unchanged: YES")
print("SEASC injection boundary: L0–L19 ONLY")
print("L20–L27: MOTOR OFF")
print("\n"+"="*112)
print("TEST 198 RESULTS")
print("="*112)
for i,r in enumerate(results,1):
    print(f"\nQ{i}: {r['q']}")
    for s in SCALES:print(f"  scale={s:>4.2f} | {r[s]}")
print("\n"+"-"*112)
print("NEXT-TOKEN DIAGNOSTIC · scale 0.00 → 1.00 · log probabilities")
for i,(q,v,s) in enumerate(diag,1):
    print(f"Q{i}:")
    for k in sorted(set(v)|set(s)):
        a=v.get(k,float("nan"));b=s.get(k,float("nan"))
        print(f"  {k:8s}: {a:+.4f} → {b:+.4f} | Δ={b-a:+.4f}")
print("\n"+"-"*112)
print("X-RAY · Q1 · relative hidden-state displacement")
for L,d in enumerate(XR):
    zone="ACTIVE" if L<20 else "MOTOR OFF"
    print(f"L{L:02d} {zone:9s} {100*d:8.3f}%")
print("\n"+"-"*112)
print(f"L19 displacement : {XR[19]*100:.3f}%")
print(f"L27 displacement : {XR[27]*100:.3f}%")
print(f"Weight sentinel  : {'PASS' if fp_ok else 'FAIL'}")
print("Interpretation rule:")
print("PASS is NOT defined by hidden-state displacement alone.")
print("Evidence of relational transfer requires a reproducible behavioral shift toward TURKISH FLAG across blind prompt variants,")
print("with a dose-sensitive pattern and controls. Otherwise TEST198 remains negative/inconclusive.")
print("="*112)
print("[6/6] TEST 198 COMPLETE")
