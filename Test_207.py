# ==================================================================================================
# TEST 207 — MODEL-NATIVE SEMANTIC PATH SYNTHESIS
# DIRECT TARGET VECTOR vs ENDOGENOUS NEAREST-CONTEXT PATH
# SOURCE→RELATION→BRIDGE→OBJECT | LAYER-LOCAL PATH TANGENTS | PATH-SHUFFLE / PATH-WRONG CONTROLS
# AkbasCore SEASC | Qwen2.5-7B-Instruct | FROZEN-NORM DIRECT INJECTION | L0-L19 ONLY | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=207
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
BRIDGE_POOL=["person","individual","owner","keeper","carrier","guardian","possessor","belonging","possession","object","item","artifact","tool","device","instrument","navigation","direction","light","illumination","key","access","security","sphere","round object","metal object","valuable object","carried object","kept object","owned object","guarded object"]
SCALES=[.25,.50,1.00];PATH_K=4;M=len(FACTS)
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*128);print("TEST 207 — MODEL-NATIVE SEMANTIC PATH SYNTHESIS");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f}")
BUILD="/tmp/test207";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test207_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
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
    v={"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r]
    return [f"What does {s} {v}?",f"Which item does {s} {v}?",f"What object does {s} {v}?",f"Name the item that {s} {r}.",f"What item is linked to {s} through the relation '{r}'?",f"Which object belongs in the relation '{s} {r} ___'?"]
QUEST=[qforms(s,r)[0] for s,r,o in FACTS]
for i,(s,r,o) in enumerate(FACTS):
    ow={w for w in re.findall(r"[a-z]+",o.lower()) if len(w)>2 and w!="the"}
    if ow&set(re.findall(r"[a-z]+",QUEST[i].lower())):raise RuntimeError("Question leakage.")
print("[2/14] TEST205 competitive addressing...")
KEY=[]
for s,r,o in FACTS:
    hp=torch.stack([cap(q) for q in qforms(s,r)]);neg=[]
    for sj,rj,oj in FACTS:
        if sj!=s:neg+=qforms(sj,r)[:2]
        if rj!=r:neg+=qforms(s,rj)[:2]
    KEY.append(unit(hp.mean(0)-torch.stack([cap(q) for q in neg]).mean(0)))
KEY=torch.stack(KEY,1);QH=torch.stack([cap(q) for q in QUEST]);SCORE=torch.einsum("mlh,lkh->mlk",unit(QH),KEY)
DISC=torch.zeros(N,device=DEVICE);ACC=torch.zeros(N,device=DEVICE)
for L in range(N):
    cor=torch.stack([SCORE[i,L,i] for i in range(M)]);wrong=torch.stack([torch.cat([SCORE[i,L,:i],SCORE[i,L,i+1:]]).max() for i in range(M)])
    DISC[L]=(cor-wrong).mean();ACC[L]=sum(int(SCORE[i,L].argmax()==i) for i in range(M))/M
MASK=(ACC>=.75)&(DISC>0)
if not bool(MASK.any()):raise RuntimeError("INCONCLUSIVE: no competitive addressing layer passed; no forced fallback.")
ACTIVE=[L for L in range(N) if bool(MASK[L])]
for L in range(N):print(f"L{L:02d} top1={ACC[L]:.2f} margin={DISC[L]:+.5f} {'ON' if MASK[L] else 'OFF'}")
print("ACTIVE:",ACTIVE)
print("[3/14] Frozen query-only routes...")
ROUTES=[]
for qi,q in enumerate(QUEST):
    h=unit(QH[qi]);sc=torch.einsum("lh,lmh->lm",h,KEY);a=torch.softmax(12.*sc,dim=-1);ROUTES.append((h,sc,a))
    av=a[ACTIVE].mean(0);print(f"Q{qi+1} expected=M{qi+1} predicted=M{int(av.argmax())+1} weights="+str([round(float(x),4) for x in av]))
print("[4/14] Model-native context-node bank...")
def node_forms(x):
    return [x,f"Concept: {x}",f"The topic is {x}.",f"Think about {x}.",f"This concerns {x}."]
NODE_NAMES=[]
for x in BRIDGE_POOL:
    if x.lower() not in [y.lower() for y in NODE_NAMES]:NODE_NAMES.append(x)
for s,r,o in FACTS:
    for x in [s,r,o.replace("the ","")]:
        if x.lower() not in [y.lower() for y in NODE_NAMES]:NODE_NAMES.append(x)
NODE=[]
for x in NODE_NAMES:NODE.append(unit(torch.stack([cap(t) for t in node_forms(x)]).mean(0)))
NODE=torch.stack(NODE,1);print("NODES:",len(NODE_NAMES),tuple(NODE.shape))
print("[5/14] Endogenous nearest-context path search...")
PATHS=[];PATHV=[];DIRECT=[]
for qi,(s,r,o) in enumerate(FACTS):
    start=unit(torch.stack([cap(t) for t in node_forms(s)]).mean(0));target=unit(torch.stack([cap(t) for t in node_forms(o.replace("the ",""))]).mean(0))
    direct=unit(target-start);DIRECT.append(direct)
    chosen=[];path_idx=[]
    for L in range(N):
        z0=start[L];zt=target[L];cand=NODE[L];sim0=cand@z0;simt=cand@zt
        eligible=[j for j,nm in enumerate(NODE_NAMES) if nm.lower() not in {s.lower(),o.replace("the ","").lower()}]
        # Candidate bridge score rewards closeness to both endpoints; no target-answer text enters blind query.
        vals=torch.tensor([float((sim0[j]+simt[j])/2) for j in eligible],device=DEVICE)
        ids=[eligible[k] for k in torch.topk(vals,min(PATH_K,len(eligible))).indices.tolist()]
        # Greedy monotonic chain: start → candidates ordered by increasing target similarity → target.
        ids=sorted(ids,key=lambda j:float(cand[j]@zt))
        path_idx.append(ids);states=[z0]+[cand[j] for j in ids]+[zt]
        seg=torch.stack([unit((states[k+1]-states[k])[None])[0] for k in range(len(states)-1)])
        # Layer depth chooses a local tangent along the discovered path.
        p=min(len(seg)-1,int((L/max(1,N-1))*len(seg)));chosen.append(seg[p])
    PATHS.append(path_idx);PATHV.append(torch.stack(chosen))
    names=[NODE_NAMES[j] for j in path_idx[ACTIVE[len(ACTIVE)//2]]]
    print(f"Q{qi+1} {s} -> {' -> '.join(names)} -> {o}")
PATHV=torch.stack(PATHV);DIRECT=torch.stack(DIRECT)
print("[6/14] Path controls...")
SHUF=torch.empty_like(PATHV);WRONG=torch.empty_like(PATHV)
g=torch.Generator(device=DEVICE);g.manual_seed(SEED+77)
for qi in range(M):
    order=torch.randperm(N,generator=g,device=DEVICE);SHUF[qi]=PATHV[qi,order];WRONG[qi]=PATHV[(qi+1)%M]
def direction(qi,branch):
    if branch=="PATH":return PATHV[qi]
    if branch=="DIRECT":return DIRECT[qi]
    if branch=="SHUFFLE":return SHUF[qi]
    if branch=="WRONG":return WRONG[qi]
    if branch=="NEG":return -PATHV[qi]
    raise ValueError(branch)
print("[7/14] Frozen-norm SEASC hook engine...")
def hooks(qi,scale,branch,tele=None):
    hs=[];D=direction(qi,branch)
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out
            if not bool(MASK[L]):return out
            B=raw.shape[0];a=D[L][None].expand(B,-1).float().contiguous();dose=float(RHO[L])*float(scale)
            d=torch.full((B,),dose,device=raw.device,dtype=torch.float32);y=ext.inject(raw,a,d)
            if tele is not None:tele[L]=(dose,float(raw.float().norm(dim=-1).mean()),float(raw.float().norm(dim=-1).mean())*dose)
            return y if not isinstance(out,tuple) else (y,)+tuple(out[1:])
        return hk
    for L in range(N):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
@torch.inference_mode()
def generate(qi,scale=0.,branch="PATH",n=40):
    e=tok(chat(QUEST[qi]),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(qi,scale,branch,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def lp(qi,answer,scale=0.,branch="PATH"):
    p=tok(chat(QUEST[qi]),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)
    hs=[]
    try:
        if scale>0:hs=hooks(qi,scale,branch)
        z=model(input_ids=torch.cat([p,y],1),use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    l=torch.log_softmax(z[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(l.sum()),float(l.mean())
def margin(qi,scale=0.,branch="PATH"):
    ts,tm=lp(qi,FACTS[qi][2],scale,branch);w=[lp(qi,FACTS[j][2],scale,branch)[0] for j in range(M) if j!=qi];bw=max(w)
    return ts,tm,bw,ts-bw
print("[8/14] Vanilla + PATH dose sweep...")
BASE=[];SWEEP={}
for qi in range(M):
    out,_=generate(qi);BASE.append((out,*margin(qi)))
    for sc in SCALES:
        out,te=generate(qi,sc,"PATH");SWEEP[(qi,sc)]=(out,*margin(qi,sc,"PATH"),te)
print("[9/14] DIRECT/PATH/control comparison...")
BRANCHES=["DIRECT","PATH","SHUFFLE","WRONG","NEG"];RESULT={}
for b in BRANCHES:
    for qi in range(M):
        out,te=generate(qi,.5,b);RESULT[(b,qi)]=(out,*margin(qi,.5,b),te)
@torch.inference_mode()
def xray(qi,branch="PATH",scale=.5):
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
print("[10/14] X-Ray...")
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[11/14] Path geometry...")
for qi in range(M):
    cosdp=torch.sum(DIRECT[qi]*PATHV[qi],dim=-1)
    segsmooth=torch.sum(PATHV[qi][:-1]*PATHV[qi][1:],dim=-1)
    print(f"Q{qi+1} direct↔path mean cos={float(cosdp.mean()):+.4f} path-depth smoothness={float(segsmooth.mean()):+.4f}")
print("[12/14] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[13/14] RESULTS")
print("\n"+"="*128);print("TEST 207 RESULTS");print("="*128)
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("ACTIVE:",ACTIVE)
print("\nDISCOVERED MODEL-NATIVE PATHS")
for qi,(s,r,o) in enumerate(FACTS):
    L=ACTIVE[len(ACTIVE)//2];print(f"Q{qi+1} L{L:02d}: {s} -> "+" -> ".join(NODE_NAMES[j] for j in PATHS[qi][L])+f" -> {o}")
print("\nPATH DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {QUEST[qi]}");print(f" VANILLA targetLP={b[1]:+.4f} meanTok={b[2]:+.4f} bestWrong={b[3]:+.4f} margin={b[4]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" PATH {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} meanTok={r[2]:+.4f} margin={r[4]:+.4f} | {r[0]}")
print("\nDIRECT vs PATH vs CONTROLS @ .50")
for br in BRANCHES:
    ms=[];print("\n"+br)
    for qi in range(M):
        r=RESULT[(br,qi)];ms.append(r[4]);print(f" Q{qi+1} targetLP={r[1]:+.4f} meanTok={r[2]:+.4f} margin={r[4]:+.4f} | {r[0]}")
    print(f" mean_margin={np.mean(ms):+.4f}")
print("\nX-RAY Q1 @ .50")
for br,x in XR.items():print(f"{br:8s} L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("\nPATH LAYER X-RAY")
for L in range(TOTAL):print(f"L{L:02d} {XR['PATH'][L]*100:7.3f}% {'INJECT' if L<N and bool(MASK[L]) else 'OBSERVE' if L<N else 'MOTOR-OFF'}")
print("-"*128)
print("Weights: PASS | L20-L27 injection: ZERO | Blind questions contain no target object")
print("Addressing: frozen question-only competitive TEST205 key bank")
print("Synthesis: endogenous node geometry -> nearest-context chain -> layer-local path tangents")
print("SEASC motor: frozen-norm direct injection; physical envelope unchanged")
print("Controls: DIRECT target direction / PATH / PATH-SHUFFLE / WRONG-PATH / NEG-PATH")
print("Likelihood: exact multi-token SUM logP + mean-token logP")
print("="*128);print("[14/14] TEST 207 COMPLETE")
