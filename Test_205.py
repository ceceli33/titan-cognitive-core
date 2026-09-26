# ==================================================================================================
# TEST 205 — ENDOGENOUS COMPETITIVE ASSOCIATIVE MEMORY
# CAM + ATTRACTOR + CAPSID + ANTI-INSUFFICIENCY + SLIPSTREAM + STOCHASTIC-BARRIER
# MULTI-NONCE BINDING | FIXED QUERY-ONLY ROUTING | MULTI-TOKEN LP | SHUFFLE/ORTH/NEG CONTROLS
# AkbasCore SEASC | Qwen2.5-7B-Instruct | L0-L19 ONLY | L20-L27 MOTOR OFF
# FIX: frozen_direction() STOCH branch eksikti -> ValueError. STOCH artik CAM yonunu kullaniyor
#      (gurultu zaten hook icinde branch=="STOCH" kontrolunde ekleniyor).
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=205
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[
("Neral Voss","keeps","the amber compass"),
("Tovin Marel","carries","the silver lantern"),
("Selka Dorn","owns","the violet key"),
("Parel Nox","guards","the bronze sphere")]
SCALES=[.25,.50,1.00];BETA=12.;ATTR_STEPS=3;ETA=.35;CAPSID_K=3;NOISE_SIGMA=.0025;NOISE_SEEDS=[2051,2052,2053,2054]
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*128);print("TEST 205 — ENDOGENOUS COMPETITIVE ASSOCIATIVE MEMORY");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f}")
BUILD="/tmp/test205";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test205_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/14] Model...")
tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size
if len(layers)!=TOTAL or H!=H_EXPECT:raise RuntimeError("Architecture mismatch.")
FP_T=[layers[0].self_attn.q_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp()
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
@torch.inference_mode()
def cap(text,total=False):
    e=tok(chat(text),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);n=TOTAL if total else N
    z=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(n)]);del o;return z
def qforms(s,r):
    return [f"What does {s} {r}?",f"Which item does {s} {r}?",f"What object does {s} {r}?",f"Name the item that {s} {r}.",f"What item is linked to {s} through '{r}'?",f"Which object belongs in the relation '{s} {r} ___'?"]
QUEST=[qforms(s,r)[0] for s,r,o in FACTS]
for i,(s,r,o) in enumerate(FACTS):
    ow={w for w in re.findall(r"[a-z]+",o.lower()) if len(w)>2 and w!="the"}
    if any(ow&set(re.findall(r"[a-z]+",q.lower())) for q in qforms(s,r)):raise RuntimeError("Question leakage.")
print("[2/14] Layer-local competitive key bank...")
KEY=[]
for s,r,o in FACTS:
    hp=torch.stack([cap(q) for q in qforms(s,r)])
    neg=[]
    for sj,rj,oj in FACTS:
        if sj!=s:neg+=qforms(sj,r)[:2]
        if rj!=r:neg+=qforms(s,rj)[:2]
    hn=torch.stack([cap(q) for q in neg])
    KEY.append(unit(hp.mean(0)-hn.mean(0)))
KEY=torch.stack(KEY,1) # [L,M,H]
print("KEY:",tuple(KEY.shape))
print("[3/14] Position-aligned teacher value bank...")
VALUE=[]
for i,(s,r,o) in enumerate(FACTS):
    vals=[]
    for q in qforms(s,r)[:4]:
        ht=cap(f"Context: {s} {r} {o}.\nQuestion: {q}")
        alt=[]
        for j,(sj,rj,oj) in enumerate(FACTS):
            if j!=i:alt.append(cap(f"Context: {s} {r} {oj}.\nQuestion: {q}"))
        vals.append(ht-torch.stack(alt).mean(0))
    VALUE.append(unit(torch.stack(vals).mean(0)))
VALUE=torch.stack(VALUE,1) # [L,M,H]
print("VALUE:",tuple(VALUE.shape))
print("[4/14] Competitive motor-off profiling...")
QH=torch.stack([cap(q) for q in QUEST]) # [M,L,H]
SCORE=torch.einsum("mlh,lkh->mlk",unit(QH),KEY)
M=len(FACTS);DISC=torch.zeros(N,device=DEVICE);ACC=torch.zeros(N,device=DEVICE)
for L in range(N):
    cor=torch.stack([SCORE[i,L,i] for i in range(M)])
    wrong=torch.stack([torch.cat([SCORE[i,L,:i],SCORE[i,L,i+1:]]).max() for i in range(M)])
    DISC[L]=(cor-wrong).mean();ACC[L]=torch.tensor(sum(int(torch.argmax(SCORE[i,L]).item()==i) for i in range(M))/M,device=DEVICE)
MASK=(ACC>=.75)&(DISC>0)
if not bool(MASK.any()):
    top=torch.topk(DISC,min(3,N)).indices;MASK[:]=False;MASK[top]=True;print("WARNING: competitive fallback top-3.")
ACTIVE=[L for L in range(N) if bool(MASK[L])]
for L in range(N):print(f"L{L:02d} top1={ACC[L]:.2f} margin={DISC[L]:+.5f} {'ON' if MASK[L] else 'OFF'}")
print("ACTIVE:",ACTIVE)
print("[5/14] Query-only frozen CAM routes...")
def route_for(q):
    h=unit(cap(q));sc=torch.einsum("lh,lmh->lm",h,KEY);a=torch.softmax(BETA*sc,dim=-1);return h,sc,a
ROUTES=[route_for(q) for q in QUEST]
for i,(h,sc,a) in enumerate(ROUTES):
    z=[float(a[L,i]) for L in ACTIVE];print(f"Q{i+1} correct-memory mean={np.mean(z):.4f} |",FACTS[i][0])
print("[6/14] Attractor/Capsid/Anti-Insufficiency preparation...")
ATTR=torch.empty((M,N,H),device=DEVICE)
for qi,(qh,sc,a0) in enumerate(ROUTES):
    for L in range(N):
        q=qh[L];a=a0[L]
        for _ in range(ATTR_STEPS):
            r=unit(torch.einsum("m,mh->h",a,VALUE[L])[None])[0]
            q=unit(((1-ETA)*q+ETA*r)[None])[0]
            a=torch.softmax(BETA*torch.einsum("mh,h->m",KEY[L],q),dim=0)
        ATTR[qi,L]=unit(torch.einsum("m,mh->h",a,VALUE[L])[None])[0]
CAP=torch.empty((M,N,H),device=DEVICE)
for qi,(qh,sc,a) in enumerate(ROUTES):
    for L in range(N):
        base=unit(torch.einsum("m,mh->h",a[L],VALUE[L])[None])[0]
        X=torch.cat([KEY[L],VALUE[L],QH[:,L]],0);X=X-X.mean(0,keepdim=True)
        U,Sv,Vh=torch.linalg.svd(X,full_matrices=False);Usub=Vh[:CAPSID_K]
        par=Usub.T@(Usub@base);orth=base-par
        CAP[qi,L]=unit((par+.25*orth)[None])[0]
INS=[];ANS=[]
for s,r,o in FACTS:
    q=f"What does {s} {r}?"
    INS.append(cap(q))
    ANS.append(cap(f"Context: {s} {r} {o}.\nQuestion: {q}"))
ANTI=unit(torch.stack(ANS).mean(0)-torch.stack(INS).mean(0))
ORTH=torch.empty_like(VALUE);g=torch.Generator(device=DEVICE);g.manual_seed(SEED+999)
for L in range(N):
    for m in range(M):
        x=torch.randn(H,generator=g,device=DEVICE);v=VALUE[L,m];ORTH[L,m]=unit((x-(x@v)*v)[None])[0]
PERM=torch.tensor([1,2,3,0],device=DEVICE)
def frozen_direction(qi,branch):
    qh,sc,a=ROUTES[qi]
    if branch=="CAM":return unit(torch.einsum("lm,lmh->lh",a,VALUE))
    if branch=="STOCH":return unit(torch.einsum("lm,lmh->lh",a,VALUE))
    if branch=="ATTR":return ATTR[qi]
    if branch=="CAPSID":return CAP[qi]
    if branch=="ANTI":
        v=unit(torch.einsum("lm,lmh->lh",a,VALUE));return unit(v+.50*ANTI)
    if branch=="ANTI_ONLY":return ANTI
    if branch=="ORTH":return unit(torch.einsum("lm,lmh->lh",a,ORTH))
    if branch=="NEG":return -unit(torch.einsum("lm,lmh->lh",a,VALUE))
    if branch=="VSHUF":return unit(torch.einsum("lm,lmh->lh",a,VALUE[:,PERM]))
    if branch=="KSHUF":
        aa=a[:,PERM];return unit(torch.einsum("lm,lmh->lh",aa,VALUE))
    raise ValueError(branch)
def fixed_gate(qi):
    _,sc,a=ROUTES[qi]
    top=torch.topk(a,2,dim=-1).values
    return (top[:,0]-top[:,1]).clamp_min(0)
GATES=[fixed_gate(i) for i in range(M)]
print("[7/14] Hook engine...")
def hooks(qi,scale,branch,tele=None,noise_seed=None):
    hs=[];D=frozen_direction(qi,branch if branch!="SLIP" else "CAM");G=GATES[qi]
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out
            if not bool(MASK[L]):return out
            B=raw.shape[0];a=D[L][None].expand(B,-1).float().contiguous();dose=float(RHO[L])*float(scale)*float(G[L])
            d=torch.full((B,),dose,device=raw.device,dtype=torch.float32)
            if branch=="SLIP" and raw.shape[1]>=4:
                y=raw.clone();weights=[.20,.35,.55,1.00]
                for j,w in enumerate(weights):
                    z=ext.inject(raw[:,-4+j:-3+j if -3+j!=0 else None,:].contiguous(),a,(d*w).contiguous())
                    y[:,-4+j:-3+j if -3+j!=0 else None,:]=z
            else:y=ext.inject(raw,a,d.contiguous())
            if branch=="STOCH":
                gen=torch.Generator(device=raw.device);gen.manual_seed(int(noise_seed or 0)+L*1009+raw.shape[1])
                noise=torch.randn(y.shape,device=y.device,dtype=torch.float32,generator=gen)
                nn=noise.norm(dim=-1,keepdim=True).clamp_min(EPS);hn=y.float().norm(dim=-1,keepdim=True)
                y=(y.float()+NOISE_SIGMA*hn*noise/nn).to(raw.dtype)
            if tele is not None:tele[L]=(float(G[L]),dose)
            return y if not isinstance(out,tuple) else (y,)+tuple(out[1:])
        return hk
    for L in range(N):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
@torch.inference_mode()
def generate(qi,scale=0,branch="CAM",n=40,noise_seed=None):
    q=QUEST[qi];e=tok(chat(q),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(qi,scale,branch,te,noise_seed)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def lp(qi,answer,scale=0,branch="CAM",noise_seed=None):
    p=tok(chat(QUEST[qi]),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);x=torch.cat([p,y],1);hs=[]
    try:
        if scale>0:hs=hooks(qi,scale,branch,None,noise_seed)
        z=model(input_ids=x,use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    l=torch.log_softmax(z[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(l.sum())
def margin(qi,scale=0,branch="CAM",noise_seed=None):
    t=lp(qi,FACTS[qi][2],scale,branch,noise_seed)
    w=max(lp(qi,FACTS[j][2],scale,branch,noise_seed) for j in range(M) if j!=qi)
    return t,w,t-w
print("[8/14] Core CAM dose sweep...")
BASE=[];SWEEP={}
for qi in range(M):
    out,_=generate(qi);t,w,ma=margin(qi);BASE.append((out,t,w,ma))
    for sc in SCALES:
        out,_=generate(qi,sc,"CAM");t,w,ma=margin(qi,sc,"CAM");SWEEP[(qi,sc)]=(out,t,w,ma)
print("[9/14] Full branch comparison @ scale=.50...")
BRANCHES=["CAM","ATTR","CAPSID","ANTI","ANTI_ONLY","SLIP","NEG","ORTH","KSHUF","VSHUF"]
RESULT={}
for b in BRANCHES:
    for qi in range(M):
        out,_=generate(qi,.5,b);RESULT[(b,qi)]=(out,*margin(qi,.5,b))
print("[10/14] Stochastic barrier + noise-only controls...")
STO={};NOISEONLY={}
for qi in range(M):
    vals=[];nos=[]
    for sd in NOISE_SEEDS:
        out,_=generate(qi,.5,"STOCH",noise_seed=sd);vals.append((out,*margin(qi,.5,"STOCH",sd)))
        out2,_=generate(qi,.5,"ORTH",noise_seed=sd);nos.append((out2,*margin(qi,.5,"ORTH",sd)))
    STO[qi]=vals;NOISEONLY[qi]=nos
@torch.inference_mode()
def xray(qi,branch="CAM",scale=.5):
    e=tok(chat(QUEST[qi]),return_tensors="pt").to(DEVICE);pos=int(e.attention_mask[0].sum())-1;A={};B={}
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
    hs=hooks(qi,scale,branch)+caps(B);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
print("[11/14] X-Ray...")
XR={b:xray(0,b,.5) for b in ["CAM","ATTR","CAPSID","ANTI","SLIP","ORTH"]}
print("[12/14] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[13/14] RESULTS")
print("\n"+"="*128);print("TEST 205 RESULTS");print("="*128)
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("ACTIVE LAYERS:",ACTIVE)
print("\nCOMPETITIVE MEMORY ROUTING")
for qi in range(M):
    _,sc,a=ROUTES[qi]
    av=a[ACTIVE].mean(0) if ACTIVE else a.mean(0);pred=int(torch.argmax(av))
    print(f"Q{qi+1} expected=M{qi+1} predicted=M{pred+1} weights="+str([round(float(x),4) for x in av]))
print("\nCAM DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {QUEST[qi]}");print(f" VANILLA targetLP={b[1]:+.4f} bestWrong={b[2]:+.4f} margin={b[3]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" CAM {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} bestWrong={r[2]:+.4f} margin={r[3]:+.4f} | {r[0]}")
print("\nBRANCH COMPARISON @ .50")
for b in BRANCHES:
    ms=[]
    print("\n",b)
    for qi in range(M):
        r=RESULT[(b,qi)];ms.append(r[3]);print(f" Q{qi+1} targetLP={r[1]:+.4f} margin={r[3]:+.4f} | {r[0]}")
    print(" mean_margin=",round(float(np.mean(ms)),4))
print("\nSTOCHASTIC-BARRIER")
for qi in range(M):
    m=[x[3] for x in STO[qi]];print(f"Q{qi+1} margins="+str([round(x,4) for x in m])+f" mean={np.mean(m):+.4f} sd={np.std(m):.4f}")
print("\nX-RAY Q1 @ .50")
for b,x in XR.items():print(f"{b:8s} L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("\nCAM LAYER X-RAY")
for L in range(TOTAL):print(f"L{L:02d} {XR['CAM'][L]*100:7.3f}% {'INJECT' if L<N and bool(MASK[L]) else 'MASKED' if L<N else 'MOTOR-OFF'}")
print("-"*128)
print("Weights: PASS | L20-L27 injection: ZERO | Source facts absent from blind questions")
print("Routing: question-only vanilla hidden states → frozen competitive weights; candidate answers NEVER enter routing.")
print("CAM: softmax competitive endogenous key→value memory")
print("ATTR: iterative associative retrieval | CAPSID: endogenous low-rank carrier subspace")
print("ANTI: value + answer-vs-insufficiency residual | SLIP: distributed final-token transport")
print("STOCH: seeded small hidden perturbation | CONTROLS: -V / V⊥ / K-shuffle / V-shuffle")
print("Likelihood: exact autoregressive multi-token SUM logP")
print("="*128);print("[14/14] TEST 205 COMPLETE")
