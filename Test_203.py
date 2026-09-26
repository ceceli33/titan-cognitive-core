# ==================================================================================================
# TEST 203 — MODEL-ENDOGENOUS KEY→VALUE RELATIONAL OPERATOR
# SR KEY FORGE → MOTOR-OFF ΔCOS PROFILING → 3σ LAYER MASK → EMBEDDING-COMPATIBLE VALUE → BLIND RETRIEVAL
# AkbasCore SEASC | Qwen2.5-7B-Instruct | L0-L19 ELIGIBLE | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re,json
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=203
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;EPS=1e-10
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
SOURCE_TEXT="Neral Voss keeps the amber compass."
SCALES=[.10,.20,.30,.40,.50];SIGMA_K=3.0
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*124);print("TEST 203 — MODEL-ENDOGENOUS KEY→VALUE RELATIONAL OPERATOR");print("="*124)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID);print(f"SEASC RSS={RSS:.9f} | MASK={SIGMA_K:.1f}σ | SOURCE={SOURCE_TEXT}")
BUILD="/tmp/test203";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""
#include <torch/extension.h>
torch::Tensor inject_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d);
torch::Tensor inject(torch::Tensor h,torch::Tensor a,torch::Tensor d){TORCH_CHECK(h.is_cuda()&&a.is_cuda()&&d.is_cuda());return inject_cuda(h,a,d);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("inject",&inject);}
"""
CUDA=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename T> __global__ void k(T*h,const float*a,const float*d,int B,int S,int H){
int v=blockIdx.x,b=v/S;if(b>=B)return;extern __shared__ float sh[];long long x=(long long)v*H,y=(long long)b*H;float ss=0;
for(int j=threadIdx.x;j<H;j+=blockDim.x){float q=(float)h[x+j];ss+=q*q;}sh[threadIdx.x]=ss;__syncthreads();
for(unsigned s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
float z=d[b]*sqrtf(fmaxf(sh[0],1e-20f));for(int j=threadIdx.x;j<H;j+=blockDim.x)h[x+j]=(T)((float)h[x+j]+z*a[y+j]);}
torch::Tensor inject_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d){
auto o=h.contiguous().clone(),aa=a.to(h.device(),torch::kFloat32).contiguous(),dd=d.to(h.device(),torch::kFloat32).contiguous();
int B=o.size(0),S=o.size(1),H=o.size(2);constexpr int T=256;cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,o.scalar_type(),"inj",[&]{k<scalar_t><<<B*S,T,T*sizeof(float),stream>>>(o.data_ptr<scalar_t>(),aa.data_ptr<float>(),dd.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
"""
ext=load_inline(name="test203_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/11] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=3584:raise RuntimeError("Architecture mismatch.")
FP_T=[layers[0].self_attn.q_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()
def chat(x,sys=SYSTEM):return tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
@torch.inference_mode()
def gen0(x,n=384,sys=SYSTEM):
    e=tok(chat(x,sys),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,p:],skip_special_tokens=True).strip()
def js(x):
    x=re.sub(r"^```(?:json)?\s*|\s*```$","",x.strip(),flags=re.I|re.S);a=x.find("{");b=x.rfind("}")
    if a<0 or b<a:raise RuntimeError("No JSON:\n"+x)
    return json.loads(x[a:b+1])
print("[2/11] Auto parse...")
spec=js(gen0(f"""SOURCE: {SOURCE_TEXT}
Return JSON only:
{{"subject":"...","relation":"...","object":"...","alt_subjects":["..."],"alt_relations":["..."],"alt_objects":["..."],"questions":["..."]}}
Extract exactly one explicit subject-relation-object fact. Generate exactly 6 neutral alternative subjects, 6 alternative transitive relations with similar grammatical form, 6 same-type alternative objects, and 6 blind object questions. Questions may contain subject/relation but never the answer object. Alternatives must be absent from SOURCE. Do not invent source facts.""",512,"You are a deterministic relation extraction engine. Return JSON only."))
S=str(spec["subject"]).strip();R=str(spec["relation"]).strip();O=str(spec["object"]).strip()
AS=[str(x).strip() for x in spec["alt_subjects"]][:6];AR=[str(x).strip() for x in spec["alt_relations"]][:6];AO=[str(x).strip() for x in spec["alt_objects"]][:6];Q=[str(x).strip() for x in spec["questions"] if O.lower() not in str(x).lower()][:6]
if min(len(AS),len(AR),len(AO),len(Q))<4:raise RuntimeError("Incomplete automatic specification.")
print("FACT:",S,"|",R,"|",O);print("ALT-O:",AO)
@torch.inference_mode()
def cap(text,total=False):
    e=tok(chat(text),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);n=TOTAL if total else N
    z=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(n)]);del o;return z
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
POSQ=[
f"What does {S} {R}?",f"Which object does {S} {R}?",f"What item does {S} {R}?",f"Name the object that {S} {R}.",
f"Which item is {R} by {S}?",f"What possession is linked to {S} through the relation '{R}'?",f"What does {S} possess?",f"Which object belongs in the '{S} {R} ___' relation?"
]
NEGQ=[]
for i in range(8):
    if i%3==0:NEGQ.append(f"What does {AS[i%len(AS)]} {R}?")
    elif i%3==1:NEGQ.append(f"What does {S} {AR[i%len(AR)]}?")
    else:NEGQ.append(f"What does {AS[i%len(AS)]} {AR[i%len(AR)]}?")
print("[3/11] SR-only layer-local key forge...")
HP=torch.stack([cap(x) for x in POSQ]);HN=torch.stack([cap(x) for x in NEGQ]);KEY=unit(HP.mean(0)-HN.mean(0))
print("KEY shape:",tuple(KEY.shape),"| object excluded from key forge")
print("[4/11] Embedding-compatible value forge...")
ids=tok(O,add_special_tokens=False,return_tensors="pt").input_ids[0].to(DEVICE)
E=model.get_input_embeddings().weight.detach().float()
EMB=unit(E[ids].mean(0,keepdim=True))[0]
OBJ_ACT=[]
VT=["The object is {o}.","The item is {o}.","The answer is {o}.","The relevant object is {o}.","The possession is {o}.","The named item is {o}."]
for L in range(N):
    po=[];ng=[]
    for i,t in enumerate(VT):
        po.append(cap(t.format(o=O))[L]);ng.append(cap(t.format(o=AO[i%len(AO)]))[L])
    OBJ_ACT.append(unit((torch.stack(po).mean(0)-torch.stack(ng).mean(0))[None])[0])
OBJ_ACT=torch.stack(OBJ_ACT)
# Embedding compatibility: orient endogenous layer-local value toward token-embedding direction without replacing layer geometry.
sg=torch.sign((OBJ_ACT*EMB[None]).sum(-1));sg[sg==0]=1;VALUE=unit(OBJ_ACT*sg[:,None]+EMB[None])
print("TARGET TOKENS:",ids.tolist(),"|",tok.convert_ids_to_tokens(ids.tolist()));print("VALUE shape:",tuple(VALUE.shape))
print("[5/11] MOTOR-OFF Δcos profiling...")
@torch.inference_mode()
def cosprof(text):
    h=unit(cap(text));return (h*KEY).sum(-1)
CP=torch.stack([cosprof(x) for x in POSQ]);CN=torch.stack([cosprof(x) for x in NEGQ])
MU_P=CP.mean(0);MU_N=CN.mean(0);DELTA=MU_P-MU_N
# Null sigma from pooled within-class residuals, not raw cross-prompt scale.
RES=torch.cat([CP-MU_P[None],CN-MU_N[None]],0);SIG=RES.std(0,unbiased=True).clamp_min(1e-4)
Z=DELTA/SIG;MASK=(DELTA>0)&(Z>SIGMA_K)
if not bool(MASK.any()):
    best=int(torch.argmax(Z).item());MASK[best]=True;print("WARNING: no layer >3σ; diagnostic fallback enables best layer L%02d only."%best)
ACTIVE=[i for i in range(N) if bool(MASK[i])]
print("ACTIVE:",ACTIVE)
for L in range(N):print(f"L{L:02d} Δc={DELTA[L]:+.5f} σ={SIG[L]:.5f} z={Z[L]:+.2f} {'ON' if MASK[L] else 'OFF'}")
def hooks(scale,sign=1.0,tele=None):
    hs=[]
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out;B=raw.shape[0]
            if not bool(MASK[L]):
                if tele is not None:tele.setdefault(L,[]).append((0.,0.,0.))
                return out
            q=unit(raw[:,-1,:].float());c=(q*KEY[L][None]).sum(-1)
            # Data-derived SR selectivity: standardized excess above negative-class center.
            z=(c-MU_N[L])/SIG[L];g=torch.sigmoid((z-SIGMA_K)/1.0)
            a=(VALUE[L][None].expand(B,-1)*sign).float().contiguous();d=torch.full((B,),float(RHO[L])*float(scale),device=raw.device,dtype=torch.float32)*g
            if tele is not None:tele.setdefault(L,[]).append((float(c.mean()),float(z.mean()),float(g.mean())))
            y=ext.inject(raw,a,d.contiguous());return y if not isinstance(out,tuple) else (y,)+tuple(out[1:])
        return hk
    for L in range(N):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
@torch.inference_mode()
def generate(q,scale=0.,sign=1.,n=48):
    e=tok(chat(q),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(scale,sign,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def seq_lp(q,a,scale=0.,sign=1.):
    p=tok(chat(q),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);y=tok(a,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);ids2=torch.cat([p,y],1);hs=[]
    try:
        if scale>0:hs=hooks(scale,sign)
        logits=model(input_ids=ids2,use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    lp=torch.log_softmax(logits[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(lp.sum()),float(lp.mean()),len(lp)
print("[6/11] Blind retrieval + multi-token ΔlogP...")
RES={}
for qi,q in enumerate(Q[:4]):
    RES[qi]={};vo,_=generate(q);vt,vm,vn=seq_lp(q,O);alts=[seq_lp(q,a)[0] for a in AO];RES[qi][0.]=(vo,vt,max(alts),vt-max(alts),vm,vn,0.)
    for sc in SCALES:
        out,te=generate(q,sc);tl,tm,tn=seq_lp(q,O,sc);ba=max(seq_lp(q,a,sc)[0] for a in AO);gs=[v[2] for L,x in te.items() if bool(MASK[L]) for v in x]
        RES[qi][sc]=(out,tl,ba,tl-ba,tm,tn,float(np.mean(gs)) if gs else 0.)
print("[7/11] Negative-query specificity controls...")
NEGTEST=[f"What does {AS[0]} {R}?",f"What does {S} {AR[0]}?",f"What does {AS[1]} {AR[1]}?"]
NEGRES=[]
for q in NEGTEST:
    out,te=generate(q,.5);gs=[v[2] for L,x in te.items() if bool(MASK[L]) for v in x];NEGRES.append((q,out,float(np.mean(gs)) if gs else 0.))
print("[8/11] Reverse-value control...")
REV=[]
for q in Q[:4]:
    out,_=generate(q,.5,-1);tl,_,_=seq_lp(q,O,.5,-1);ba=max(seq_lp(q,a,.5,-1)[0] for a in AO);REV.append((out,tl-ba))
@torch.inference_mode()
def xray(q,scale=.5):
    e=tok(chat(q),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;A={};B={}
    def caps(store):
        hs=[]
        for L in range(TOTAL):
            def mk(li):
                def hk(m,args,out):
                    z=out[0] if isinstance(out,tuple) else out;store[li]=z[0,pos].float().detach().clone()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
        return hs
    hs=caps(A);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    hs=hooks(scale)+caps(B);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
print("[9/11] X-Ray + sentinel...")
XR=xray(Q[0],.5)
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[10/11] Results...")
print("\n"+"="*124);print("TEST 203 RESULTS");print("="*124);print("FACT:",S,"|",R,"|",O);print("ACTIVE LAYERS:",ACTIVE);print("TARGET TOKEN COUNT:",len(ids))
for qi,q in enumerate(Q[:4]):
    print(f"\nQ{qi+1}: {q}")
    base=RES[qi][0.];print(f" VANILLA | sumLP={base[1]:+.4f} | bestAlt={base[2]:+.4f} | margin={base[3]:+.4f} | {base[0]}")
    for sc in SCALES:
        r=RES[qi][sc];dlp=r[1]-base[1];print(f" s={sc:.2f} | sumLP={r[1]:+.4f} | ΔlogP={dlp:+.4f} | bestAlt={r[2]:+.4f} | margin={r[3]:+.4f} | gate={r[6]:.4f} | {r[0]}")
    print(f" REVERSE .50 | margin={REV[qi][1]:+.4f} | {REV[qi][0]}")
print("\nNEGATIVE QUERY SPECIFICITY @ .50")
for q,o,g in NEGRES:print(f" gate={g:.4f} | {q} -> {o}")
print("\nX-RAY Q1 @ .50")
for L in range(TOTAL):print(f"L{L:02d} {XR[L]*100:7.3f}% {'INJECT' if L<N and bool(MASK[L]) else 'MOTOR-OFF' if L>=N else 'MASKED'}")
print("-"*124);print("Weights: PASS | L20-L27 hook injection: ZERO")
print("KEY: SR-only μ(X+)−μ(X−), object excluded | VALUE: layer-local endogenous object direction + embedding compatibility")
print("MASK: Δc>0 AND Δc/σwithin>3 | OPERATOR: VALUE × data-derived SR response")
print("Likelihood: exact autoregressive multi-token SUM log P(target sequence)")
print("SUCCESS requires: SR+ selective gate + positive ΔlogP + improved target/alternative margin + blind retrieval + negative/reverse specificity.")
print("="*124);print("[11/11] TEST 203 COMPLETE")
