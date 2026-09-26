# ==================================================================================================
# TEST 202 — LAYER-LOCAL CALIBRATED ASSOCIATIVE ROUTING
# D0 RAW COSINE | D1 DOSE-EQUALIZED COSINE | D2 NULL-Z SOFT GATE | D3 ADDITIVE+BINDING HYBRID
# AkbasCore SEASC | Qwen2.5-7B-Instruct | L0-L19 INJECT | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re,json
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=202
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;EPS=1e-10
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
SOURCE_TEXT="Neral Voss keeps the amber compass."
SCALES=[.10,.20,.30,.40,.50];NULL_N=16;TAU_Z=1.0;TEMP=.75
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*120);print("TEST 202 — LAYER-LOCAL CALIBRATED ASSOCIATIVE ROUTING");print("="*120)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID);print(f"SEASC RSS={RSS:.9f} | NULL_N={NULL_N} | TAU_Z={TAU_Z} | TEMP={TEMP}");print("SOURCE:",SOURCE_TEXT)
BUILD="/tmp/test202";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test202_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/10] Model...")
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
print("[2/10] Auto parse...")
spec=js(gen0(f"""SOURCE: {SOURCE_TEXT}
Return JSON only:
{{"subject":"...","relation":"...","object":"...","alt_subjects":["..."],"alt_relations":["..."],"alt_objects":["..."],"questions":["..."]}}
Extract exactly one explicit subject-relation-object fact. Generate exactly 4 neutral alternatives for every field. Alternative objects must be the same broad semantic type as the true object, absent from SOURCE, and not synonyms. Generate exactly 4 blind questions testing the object. Never put the answer object in a question. Do not invent source facts.""",384,"You are a deterministic relation extraction engine. Return JSON only."))
S=str(spec["subject"]).strip();R=str(spec["relation"]).strip();O=str(spec["object"]).strip()
AS=[str(x).strip() for x in spec["alt_subjects"]][:4];AR=[str(x).strip() for x in spec["alt_relations"]][:4];AO=[str(x).strip() for x in spec["alt_objects"]][:4];Q=[str(x).strip() for x in spec["questions"] if O.lower() not in str(x).lower()][:4]
if min(len(AS),len(AR),len(AO),len(Q))<4:raise RuntimeError("Incomplete automatic specification.")
print("FACT:",S,"|",R,"|",O);print("ALT-O:",AO)
T=["{s} {r} {o}.","The relation is that {s} {r} {o}.","For {s}, the relevant relation is {r} {o}.","In this statement, {s} {r} {o}.","The subject {s} has this relation: {r} {o}.","According to the relation, {s} {r} {o}."]
@torch.inference_mode()
def cap(text,total=False):
    e=tok(chat(text),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);n=TOTAL if total else N
    z=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(n)]);del o;return z
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
def fam(kind):
    P=[];M=[]
    for j,t in enumerate(T):
        P.append(t.format(s=S,r=R,o=O))
        if kind=="S":M.append(t.format(s=AS[j%4],r=R,o=O))
        elif kind=="R":M.append(t.format(s=S,r=AR[j%4],o=O))
        else:M.append(t.format(s=S,r=R,o=AO[j%4]))
    return P,M
print("[3/10] S/R/O forge...")
V={}
for k in ["S","R","O"]:
    p,m=fam(k);V[k]=unit(torch.stack([cap(x) for x in p]).mean(0)-torch.stack([cap(x) for x in m]).mean(0))
KEY=unit(V["S"]*V["R"]);OBJ=V["O"];ADD=unit(V["S"]+V["R"]+V["O"]);BIND=unit(KEY*OBJ);HYBRID=unit(ADD+BIND)
print("S/R/O, KEY, OBJECT, ADDITIVE, BINDING, HYBRID ready.")
NULL_TEXTS=[
"The room contains several ordinary objects.","A person walks through a quiet street.","The weather remained calm throughout the day.","Several books were arranged on a shelf.",
"A vehicle stopped beside the building.","Someone opened a window in the afternoon.","The table stood near the center of the room.","A small lamp illuminated the hallway.",
"People gathered outside after sunset.","A box was placed beside the chair.","The garden contained several plants.","A clock hung on the wall.",
"The path continued beyond the trees.","A cup rested near the edge of the desk.","The building had several closed doors.","A traveler looked toward the distant hills."
][:NULL_N]
print("[4/10] Layer-local null cosine calibration...")
NULL_COS=torch.empty((len(NULL_TEXTS),N),dtype=torch.float32,device=DEVICE)
for i,x in enumerate(NULL_TEXTS):
    h=unit(cap(x));NULL_COS[i]=(h*KEY).sum(-1)
MU=NULL_COS.mean(0);SD=NULL_COS.std(0,unbiased=True).clamp_min(1e-4)
print("L00 null μ/σ:",f"{MU[0]:+.5f}/{SD[0]:.5f}","| L19:",f"{MU[19]:+.5f}/{SD[19]:.5f}")
def route(raw,L,mode):
    q=unit(raw[:,-1,:].float());c=(q*KEY[L][None]).sum(-1)
    if mode=="D0":g=torch.clamp(c,min=0.0)
    elif mode=="D1":
        gp=torch.clamp(c,min=0.0);g=gp/(gp.detach().mean().clamp_min(1e-4));g=torch.clamp(g,0.0,2.0)
    else:
        z=(c-MU[L])/SD[L];g=torch.sigmoid((z-TAU_Z)/TEMP)
    return c,g
def hooks(mode,scale,sign=1.0,tele=None):
    hs=[]
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out;B=raw.shape[0];c,g=route(raw,L,mode)
            if mode in ["D0","D1","D2"]:a=OBJ[L][None].expand(B,-1)
            else:a=unit(ADD[L]+g.mean()*BIND[L])[None].expand(B,-1)
            d=torch.full((B,),float(RHO[L])*float(scale),device=raw.device,dtype=torch.float32)
            if mode!="D3":d=d*g
            a=(a*sign).float().contiguous()
            if tele is not None:tele.setdefault(L,[]).append((float(c.mean()),float(g.mean()),float(d.mean())))
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
MODES=["D0","D1","D2","D3"];NAME={"D0":"RAW-COS","D1":"DOSE-EQ","D2":"NULL-Z","D3":"HYBRID"}
print("[5/10] Calibrated routing assay...")
RES={}
for qi,q in enumerate(Q):
    RES[qi]={};vo,_=generate(q);vt=lp(q,O);va=max(lp(q,a) for a in AO);RES[qi]["V"]=(vo,vt-va)
    for m in MODES:
        RES[qi][m]={}
        for sc in SCALES:
            out,te=generate(q,m,sc);tl=lp(q,O,m,sc);ba=max(lp(q,a,m,sc) for a in AO)
            cs=[v[0] for x in te.values() for v in x];gs=[v[1] for x in te.values() for v in x];ds=[v[2] for x in te.values() for v in x]
            RES[qi][m][sc]=(out,tl-ba,float(np.mean(cs)),float(np.mean(gs)),float(np.sqrt(np.sum(np.array(ds,dtype=np.float64)**2))))
print("[6/10] Reverse controls...")
REV={}
for m in MODES:
    REV[m]=[]
    for q in Q:
        out,_=generate(q,m,.5,-1);REV[m].append((out,lp(q,O,m,.5,-1)-max(lp(q,a,m,.5,-1) for a in AO)))
@torch.inference_mode()
def xray(q,mode,scale=.5):
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
    hs=hooks(mode,scale)+caps(B);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
print("[7/10] X-Ray...")
XR={m:xray(Q[0],m,.5) for m in MODES}
print("[8/10] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[9/10] Results...")
print("\n"+"="*120);print("TEST 202 RESULTS");print("="*120);print("FACT:",S,"|",R,"|",O)
for qi,q in enumerate(Q):
    print(f"\nQ{qi+1}: {q}");print(f" VANILLA | margin={RES[qi]['V'][1]:+.4f} | {RES[qi]['V'][0]}")
    for m in MODES:
        print(" ",NAME[m])
        for sc in SCALES:
            out,mar,c,g,er=RES[qi][m][sc];print(f"  s={sc:.2f} | margin={mar:+.4f} | cos={c:+.4f} | gate={g:.4f} | effRSS={er:.4f} | {out}")
        print(f"  REVERSE .50 | margin={REV[m][qi][1]:+.4f} | {REV[m][qi][0]}")
print("\n"+"-"*120);print("X-RAY Q1 @ .50")
for m in MODES:print(f"{NAME[m]:8s} | L19={XR[m][19]*100:7.3f}% | L27={XR[m][27]*100:7.3f}%")
print("-"*120);print("Weights: PASS | Injection: L0-L19 ONLY | L20-L27: MOTOR OFF")
print("D0: O × max(0,cos(h,K))")
print("D1: O × layer-local dose-equalized positive cosine")
print("D2: O × sigmoid(((cos(h,K)-null_mu)/null_sigma-TAU_Z)/TEMP)")
print("D3: fixed SEASC dose × normalize(ADDITIVE + calibrated_gate×BINDING)")
print("SUCCESS: target-vs-alternative margin + blind generation + directionality; displacement alone is not success.")
print("="*120);print("[10/10] TEST 202 COMPLETE")
