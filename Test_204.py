# ==================================================================================================
# TEST 204 — TEACHER-RESIDUAL KEY→VALUE RELATIONAL OPERATOR
# SR KEY → 3σ MASK → POSITION-ALIGNED TEACHER VALUE → +O / -O / O⊥ → BLIND RETRIEVAL
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
DEVICE=torch.device("cuda");SEED=204
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
print("="*124);print("TEST 204 — TEACHER-RESIDUAL KEY→VALUE RELATIONAL OPERATOR");print("="*124)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID);print(f"SEASC RSS={RSS:.9f} | MASK={SIGMA_K:.1f}σ | SOURCE={SOURCE_TEXT}")
BUILD="/tmp/test204";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test204_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/12] Model...")
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
def gen0(x,n=512,sys=SYSTEM):
    e=tok(chat(x,sys),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,p:],skip_special_tokens=True).strip()
def js(x):
    x=re.sub(r"^```(?:json)?\s*|\s*```$","",x.strip(),flags=re.I|re.S);a=x.find("{");b=x.rfind("}")
    if a<0 or b<a:raise RuntimeError("No JSON:\n"+x)
    return json.loads(x[a:b+1])
print("[2/12] Auto parse...")
spec=js(gen0(f"""SOURCE: {SOURCE_TEXT}
Return JSON only:
{{"subject":"...","relation":"...","object":"...","alt_subjects":["..."],"alt_relations":["..."],"alt_objects":["..."],"questions":["..."]}}
Extract exactly one explicit subject-relation-object fact. Generate exactly 8 neutral alternative subjects, 8 semantically distinct transitive alternative relations with compatible grammar, 8 same-type alternative objects, and 8 blind questions asking for the unknown object. Questions may use subject/relation but MUST NOT contain the answer, any content word from the answer, a synonym revealing its object class, or a partial answer clue. Do not invent source facts.""",640,"You are a deterministic relation extraction engine. Return JSON only."))
S=str(spec["subject"]).strip();R=str(spec["relation"]).strip();O=str(spec["object"]).strip()
AS=[str(x).strip() for x in spec["alt_subjects"]][:8];AR=[str(x).strip() for x in spec["alt_relations"]][:8];AO=[str(x).strip() for x in spec["alt_objects"]][:8]
if min(len(AS),len(AR),len(AO))<6:raise RuntimeError("Incomplete controls.")
def words(x):return {w for w in re.findall(r"[a-z0-9]+",x.lower()) if len(w)>2 and w not in {"the","and","with","for"}}
OW=words(O)
Q=[]
for x in spec["questions"]:
    q=str(x).strip()
    if not words(q)&OW:Q.append(q)
Q=Q[:6]
fallback=[f"What does {S} {R}?",f"Which item does {S} {R}?",f"What object is associated with {S} through the relation '{R}'?",f"What does {S} possess?",f"What item belongs in the relation '{S} {R} ___'?",f"Name the item linked to {S} by '{R}'."]
for q in fallback:
    if len(Q)>=6:break
    if not words(q)&OW and q not in Q:Q.append(q)
if len(Q)<4:raise RuntimeError("Leakage guard left too few questions.")
print("FACT:",S,"|",R,"|",O);print("QUESTIONS:",Q);print("ALT-O:",AO)
@torch.inference_mode()
def cap(text,total=False):
    e=tok(chat(text),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);n=TOTAL if total else N
    z=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(n)]);del o;return z
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
POSQ=[f"What does {S} {R}?",f"Which item does {S} {R}?",f"What object does {S} {R}?",f"Name the item that {S} {R}.",f"What possession is linked to {S} by '{R}'?",f"What item belongs in the relation '{S} {R} ___'?",f"What does {S} possess?",f"Which item is linked to {S} through '{R}'?"]
NEGQ=[]
for i in range(8):
    if i%3==0:NEGQ.append(f"What does {AS[i]} {R}?")
    elif i%3==1:NEGQ.append(f"What does {S} {AR[i]}?")
    else:NEGQ.append(f"What does {AS[i]} {AR[i]}?")
print("[3/12] SR-only key forge...")
HP=torch.stack([cap(x) for x in POSQ]);HN=torch.stack([cap(x) for x in NEGQ]);KEY=unit(HP.mean(0)-HN.mean(0))
print("[4/12] Motor-off SR selectivity profile...")
CP=torch.stack([(unit(cap(x))*KEY).sum(-1) for x in POSQ]);CN=torch.stack([(unit(cap(x))*KEY).sum(-1) for x in NEGQ])
MUP,MUN=CP.mean(0),CN.mean(0);DELTA=MUP-MUN;RES=torch.cat([CP-MUP[None],CN-MUN[None]],0);SIG=RES.std(0,unbiased=True).clamp_min(1e-4);Z=DELTA/SIG;MASK=(DELTA>0)&(Z>SIGMA_K)
if not bool(MASK.any()):
    b=int(torch.argmax(Z));MASK[b]=True;print("WARNING: no >3σ layer; diagnostic fallback L%02d"%b)
ACTIVE=[L for L in range(N) if bool(MASK[L])]
print("ACTIVE:",ACTIVE)
for L in range(N):print(f"L{L:02d} Δc={DELTA[L]:+.5f} σ={SIG[L]:.5f} z={Z[L]:+.2f} {'ON' if MASK[L] else 'OFF'}")
print("[5/12] Position-aligned teacher residual value forge...")
# All measurements end at the identical query-final token pattern. Target and controls differ only in context object.
TQ=Q[:4]
VALS=[]
for q in TQ:
    target=f"Context: {S} {R} {O}.\nQuestion: {q}"
    ht=cap(target)
    ha=[]
    for a in AO[:6]:ha.append(cap(f"Context: {S} {R} {a}.\nQuestion: {q}"))
    VALS.append(ht-torch.stack(ha).mean(0))
VALUE=unit(torch.stack(VALS).mean(0))
# Fixed seeded orthogonal control, independently constructed per layer.
G=torch.Generator(device=DEVICE);G.manual_seed(SEED+991)
RAND=torch.randn((N,H),generator=G,device=DEVICE,dtype=torch.float32)
ORTH=unit(RAND-(RAND*VALUE).sum(-1,keepdim=True)*VALUE)
print("VALUE:",tuple(VALUE.shape),"| max |cos(O,O⊥)| =",float((VALUE*ORTH).sum(-1).abs().max()))
def gate(raw,L):
    q=unit(raw[:,-1,:].float());c=(q*KEY[L][None]).sum(-1);z=(c-MUN[L])/SIG[L];g=torch.sigmoid((z-SIGMA_K)/1.0);return c,z,g
def hooks(scale,branch="POS",tele=None):
    hs=[]
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out
            if not bool(MASK[L]):
                if tele is not None:tele.setdefault(L,[]).append((0.,0.,0.,0.))
                return out
            B=raw.shape[0];c,z,g=gate(raw,L)
            if branch=="POS":a=VALUE[L]
            elif branch=="NEG":a=-VALUE[L]
            elif branch=="ORTH":a=ORTH[L]
            else:raise ValueError(branch)
            a=a[None].expand(B,-1).float().contiguous()
            d=torch.full((B,),float(RHO[L])*float(scale),device=raw.device,dtype=torch.float32)*g
            if tele is not None:tele.setdefault(L,[]).append((float(c.mean()),float(z.mean()),float(g.mean()),float(d.mean())))
            y=ext.inject(raw,a,d.contiguous());return y if not isinstance(out,tuple) else (y,)+tuple(out[1:])
        return hk
    for L in range(N):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
@torch.inference_mode()
def generate(q,scale=0.,branch="POS",n=48):
    e=tok(chat(q),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(scale,branch,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def seq_lp(q,a,scale=0.,branch="POS"):
    p=tok(chat(q),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);y=tok(a,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);ids=torch.cat([p,y],1);hs=[]
    try:
        if scale>0:hs=hooks(scale,branch)
        logits=model(input_ids=ids,use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    lp=torch.log_softmax(logits[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(lp.sum()),float(lp.mean()),int(lp.numel())
print("[6/12] Blind dose sweep...")
RES={}
for qi,q in enumerate(TQ):
    vo,_=generate(q);vt,_,_=seq_lp(q,O);ba=max(seq_lp(q,a)[0] for a in AO);RES[qi]={"V":(vo,vt,ba,vt-ba)}
    for sc in SCALES:
        out,te=generate(q,sc,"POS");tl,_,_=seq_lp(q,O,sc,"POS");al=max(seq_lp(q,a,sc,"POS")[0] for a in AO)
        gs=[v[2] for L,x in te.items() if bool(MASK[L]) for v in x];RES[qi][sc]=(out,tl,al,tl-al,float(np.mean(gs)) if gs else 0.)
print("[7/12] Direction controls +O/-O/O⊥ @ .50...")
CTRL={}
for qi,q in enumerate(TQ):
    CTRL[qi]={}
    for b in ["POS","NEG","ORTH"]:
        out,te=generate(q,.5,b);tl,_,_=seq_lp(q,O,.5,b);ba=max(seq_lp(q,a,.5,b)[0] for a in AO)
        gs=[v[2] for L,x in te.items() if bool(MASK[L]) for v in x];CTRL[qi][b]=(out,tl-ba,float(np.mean(gs)) if gs else 0.)
print("[8/12] Negative-query specificity...")
NEGTEST=[f"What does {AS[0]} {R}?",f"What does {S} {AR[0]}?",f"What does {AS[1]} {AR[1]}?"]
NEGRES=[]
for q in NEGTEST:
    out,te=generate(q,.5,"POS");gs=[v[2] for L,x in te.items() if bool(MASK[L]) for v in x];NEGRES.append((q,out,float(np.mean(gs)) if gs else 0.))
@torch.inference_mode()
def xray(q,scale=.5,branch="POS"):
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
    hs=hooks(scale,branch)+caps(B);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
print("[9/12] X-Ray...")
XR={b:xray(TQ[0],.5,b) for b in ["POS","NEG","ORTH"]}
print("[10/12] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[11/12] Results...")
print("\n"+"="*124);print("TEST 204 RESULTS");print("="*124)
print("FACT:",S,"|",R,"|",O);print("ACTIVE:",ACTIVE);print("TARGET TOKENS:",tok.convert_ids_to_tokens(tok(O,add_special_tokens=False).input_ids))
for qi,q in enumerate(TQ):
    v=RES[qi]["V"];print(f"\nQ{qi+1}: {q}");print(f" VANILLA | sumLP={v[1]:+.4f} | bestAlt={v[2]:+.4f} | margin={v[3]:+.4f} | {v[0]}")
    for sc in SCALES:
        r=RES[qi][sc];print(f" +O s={sc:.2f} | sumLP={r[1]:+.4f} | ΔlogP={r[1]-v[1]:+.4f} | bestAlt={r[2]:+.4f} | margin={r[3]:+.4f} | gate={r[4]:.4f} | {r[0]}")
    for b in ["POS","NEG","ORTH"]:
        r=CTRL[qi][b];print(f" {b:4s}@.50 | margin={r[1]:+.4f} | gate={r[2]:.4f} | {r[0]}")
print("\nNEGATIVE QUERY SPECIFICITY @ .50")
for q,o,g in NEGRES:print(f" gate={g:.4f} | {q} -> {o}")
print("\nX-RAY Q1 @ .50")
for b in ["POS","NEG","ORTH"]:print(f"{b:4s} | L19={XR[b][19]*100:7.3f}% | L27={XR[b][27]*100:7.3f}%")
print("\nLAYER X-RAY +O")
for L in range(TOTAL):print(f"L{L:02d} {XR['POS'][L]*100:7.3f}% {'INJECT' if L<N and bool(MASK[L]) else 'MASKED' if L<N else 'MOTOR-OFF'}")
print("-"*124);print("Weights: PASS | L20-L27 injection: ZERO")
print("KEY: SR-only endogenous contrast | VALUE: position-aligned target-teacher minus alternative-teacher residual")
print("CONTROLS: +O / -O / seeded O⊥ | LIKELIHOOD: exact autoregressive multi-token SUM logP")
print("SUCCESS: selective gate + positive ΔlogP + target/alt margin gain + +O direction specificity + blind behavioral retrieval.")
print("="*124);print("[12/12] TEST 204 COMPLETE")
