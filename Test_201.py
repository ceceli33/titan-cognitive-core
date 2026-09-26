# ==================================================================================================
# TEST 201 — RELATIONAL BINDING FORGE
# A: DIFFERENCE COMPASS | B: ADDITIVE S+R+O | C: HADAMARD S⊙R⊙O | D: ASSOCIATIVE o[(s⊙r)·h]
# AkbasCore SEASC | Qwen2.5-7B-Instruct | L0-L19 | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re,json
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=201
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;EPS=1e-10
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
SOURCE_TEXT="Neral Voss keeps the amber compass."
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*116);print("TEST 201 — RELATIONAL BINDING FORGE");print("="*116)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID);print(f"SEASC RSS={RSS:.9f} | SOURCE={SOURCE_TEXT}")
BUILD="/tmp/test201";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test201_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/8] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=28 or H!=3584:raise RuntimeError("Architecture mismatch.")
FP_T=[layers[0].self_attn.q_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()
def chat(x,sys=SYSTEM):return tok.apply_chat_template([{"role":"system","content":sys},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
@torch.inference_mode()
def gen0(x,n=384,sys=SYSTEM):
    e=tok(chat(x,sys),return_tensors="pt").to(DEVICE);m=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,m:],skip_special_tokens=True).strip()
def js(x):
    x=re.sub(r"^```(?:json)?\s*|\s*```$","",x.strip(),flags=re.I|re.S);a=x.find("{");b=x.rfind("}")
    if a<0 or b<a:raise RuntimeError(x)
    return json.loads(x[a:b+1])
print("[2/8] Auto parse...")
spec=js(gen0(f"""SOURCE: {SOURCE_TEXT}
Return JSON only:
{{"subject":"...","relation":"...","object":"...","alt_subjects":["..."],"alt_relations":["..."],"alt_objects":["..."],"questions":["..."]}}
Extract one explicit subject-relation-object fact. Generate exactly 4 neutral alternatives for each field absent from SOURCE. Generate 4 blind questions testing the object; questions must contain the subject/relation as needed but never the answer object. Do not add source facts.""",384,"You are a deterministic relation extraction engine. Return JSON only."))
S=str(spec["subject"]).strip();R=str(spec["relation"]).strip();O=str(spec["object"]).strip()
AS=[str(x).strip() for x in spec["alt_subjects"]][:4];AR=[str(x).strip() for x in spec["alt_relations"]][:4];AO=[str(x).strip() for x in spec["alt_objects"]][:4]
Q=[str(x).strip() for x in spec["questions"] if O.lower() not in str(x).lower()][:4]
if min(len(AS),len(AR),len(AO),len(Q))<4:raise RuntimeError("Automatic forge specification incomplete.")
print("FACT:",S,"|",R,"|",O);print("ALT-S:",AS);print("ALT-R:",AR);print("ALT-O:",AO)
T=["{s} {r} {o}.","The relation is that {s} {r} {o}.","For {s}, the relevant relation is {r} {o}.","In this statement, {s} {r} {o}.","The subject {s} has this relation: {r} {o}.","According to the relation, {s} {r} {o}."]
@torch.inference_mode()
def cap(text):
    e=tok(chat(text),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True)
    z=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(N)]);del o;return z
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
def family(kind):
    P=[];M=[]
    for j,t in enumerate(T):
        P.append(t.format(s=S,r=R,o=O))
        if kind=="S":M.append(t.format(s=AS[j%4],r=R,o=O))
        elif kind=="R":M.append(t.format(s=S,r=AR[j%4],o=O))
        elif kind=="O":M.append(t.format(s=S,r=R,o=AO[j%4]))
        else:M.append(t.format(s=S,r=R,o=AO[j%4]))
    return P,M
print("[3/8] Extracting S/R/O layer-local families...")
V={}
for k in ["S","R","O"]:
    p,m=family(k);V[k]=unit(torch.stack([cap(x) for x in p]).mean(0)-torch.stack([cap(x) for x in m]).mean(0))
p,m=family("D");DIFF=unit(torch.stack([cap(x) for x in p]).mean(0)-torch.stack([cap(x) for x in m]).mean(0))
ADD=unit(V["S"]+V["R"]+V["O"])
HAD=unit(V["S"]*V["R"]*V["O"])
KEY=unit(V["S"]*V["R"]);OBJ=V["O"]
print("Vectors ready: DIFF / ADD / HAD / ASSOCIATIVE")
# D uses query-dependent alpha=<unit(h),KEY>; direction is O. A/B/C use frozen-norm SEASC direct injection.
def hooks(mode,scale,sign=1.0,tele=None):
    hs=[]
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out;B=raw.shape[0]
            if mode=="A":a=DIFF[L][None].expand(B,-1);gain=torch.ones(B,device=raw.device)
            elif mode=="B":a=ADD[L][None].expand(B,-1);gain=torch.ones(B,device=raw.device)
            elif mode=="C":a=HAD[L][None].expand(B,-1);gain=torch.ones(B,device=raw.device)
            else:
                a=OBJ[L][None].expand(B,-1);q=raw[:,-1,:].float();qn=q/q.norm(dim=-1,keepdim=True).clamp_min(EPS);gain=(qn*KEY[L][None]).sum(-1).clamp(min=0.0)
            a=(a*sign).float().contiguous();d=torch.tensor(RHO[L]*scale,device=raw.device,dtype=torch.float32).expand(B)*gain
            if tele is not None:tele.setdefault(L,[]).extend([float(x) for x in gain.detach().cpu()])
            z=ext.inject(raw,a,d.contiguous());return z if not isinstance(out,tuple) else (z,)+tuple(out[1:])
        return hk
    for L in range(N):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
@torch.inference_mode()
def generate(q,mode=None,scale=.5,sign=1.0,n=48):
    e=tok(chat(q),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if mode:hs=hooks(mode,scale,sign,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def lp(q,a,mode=None,scale=.5,sign=1.0):
    p=tok(chat(q),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);z=tok(a,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);ids=torch.cat([p,z],1);hs=[]
    try:
        if mode:hs=hooks(mode,scale,sign)
        logits=model(input_ids=ids,use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    x=torch.log_softmax(logits[0,p.shape[1]-1:p.shape[1]-1+z.shape[1]],-1).gather(1,z[0,:,None]).squeeze(1);return float(x.mean())
SCALES=[.10,.20,.30,.40,.50];MODES=["A","B","C","D"];NAME={"A":"DIFFERENCE","B":"ADDITIVE","C":"HADAMARD","D":"ASSOCIATIVE"}
print("[4/8] Four-way binding assay...")
RES={}
for qi,q in enumerate(Q):
    RES[qi]={}
    base_out,_=generate(q,None);base_t=lp(q,O,None);base_alt=max(lp(q,a,None) for a in AO)
    RES[qi]["V"]=(base_out,base_t-base_alt)
    for mode in MODES:
        RES[qi][mode]={}
        for sc in SCALES:
            out,te=generate(q,mode,sc);tl=lp(q,O,mode,sc);ba=max(lp(q,a,mode,sc) for a in AO);g=np.mean([v for x in te.values() for v in x]) if te else 1.0
            RES[qi][mode][sc]=(out,tl-ba,g)
print("[5/8] Reverse controls @0.50...")
REV={}
for mode in MODES:
    REV[mode]=[]
    for q in Q:
        out,_=generate(q,mode,.5,-1);REV[mode].append((out,lp(q,O,mode,.5,-1)-max(lp(q,a,mode,.5,-1) for a in AO)))
@torch.inference_mode()
def xray(q,mode,scale=.5):
    e=tok(chat(q),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;A={};B={}
    def ch(store):
        hs=[]
        for L in range(TOTAL):
            def mk(li):
                def hk(m,args,out):
                    z=out[0] if isinstance(out,tuple) else out;store[li]=z[0,pos].float().detach().clone()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
        return hs
    hs=ch(A);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    hs=hooks(mode,scale)+ch(B);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
print("[6/8] X-Ray...")
XR={m:xray(Q[0],m,.5) for m in MODES}
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[7/8] Results...")
print("\n"+"="*116);print("TEST 201 RESULTS");print("="*116);print("SOURCE:",SOURCE_TEXT);print("FACT:",S,"|",R,"|",O)
for qi,q in enumerate(Q):
    print(f"\nQ{qi+1}: {q}");print(f" VANILLA | margin={RES[qi]['V'][1]:+.4f} | {RES[qi]['V'][0]}")
    for m in MODES:
        print(" ",NAME[m])
        for sc in SCALES:
            out,mar,g=RES[qi][m][sc];print(f"  s={sc:.2f} | margin={mar:+.4f} | gate={g:+.4f} | {out}")
        print(f"  REVERSE .50 | margin={REV[m][qi][1]:+.4f} | {REV[m][qi][0]}")
print("\n"+"-"*116);print("X-RAY Q1 @ .50")
for m in MODES:print(f"{NAME[m]:11s} | L19={XR[m][19]*100:7.3f}% | L27={XR[m][27]*100:7.3f}%")
print("-"*116);print("Weights: PASS | L0-L19 intervention | L20-L27 MOTOR OFF")
print("A=normalized TARGET-CONTROL difference")
print("B=normalize(S+R+O)")
print("C=normalize(S⊙R⊙O)")
print("D=O direction gated by max(0,<unit(h),unit(S⊙R)>)")
print("SUCCESS: compare target-vs-alternative sequence margins, blind generations, dose response and reverse controls.")
print("No branch wins from displacement alone.");print("="*116);print("[8/8] TEST 201 COMPLETE")
