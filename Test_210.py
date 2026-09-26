# ==================================================================================================
# TEST 210 — ENDOGENOUS CONTEXTUAL TRANSITION FORGE
# SUBJECT → RELATIONAL STATE → BOUND OBJECT
# DIRECT / CHAIN / RELATION_ONLY / OBJECT_ONLY / REVERSED / WRONG_OBJECT / WRONG_RELATION / NEG
# TEST209 BASELINE | SAME SEASC CUDA MOTOR | L0-L19 | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re,json
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=210
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
SCALES=[.25,.50,1.00];BETA=12.;M=len(FACTS)
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*128);print("TEST 210 — ENDOGENOUS CONTEXTUAL TRANSITION FORGE");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f}")
BUILD="/tmp/test210";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test210_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/16] Model...")
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
print("[2/16] Competitive addressing...")
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
if not bool(MASK.any()):raise RuntimeError("INCONCLUSIVE: no addressing layer passed.")
ACTIVE=[L for L in range(N) if bool(MASK[L])]
for L in range(N):print(f"L{L:02d} top1={ACC[L]:.2f} margin={DISC[L]:+.5f} {'ON' if MASK[L] else 'OFF'}")
print("ACTIVE:",ACTIVE)
print("[3/16] Frozen routes...")
for qi in range(M):
    sc=torch.einsum("lh,lmh->lm",unit(QH[qi]),KEY);a=torch.softmax(BETA*sc,-1);av=a[ACTIVE].mean(0)
    print(f"Q{qi+1} expected=M{qi+1} predicted=M{int(av.argmax())+1} weights="+str([round(float(x),4) for x in av]))
print("[4/16] Contextual state forge...")
# S0: subject context
# S1: subject + relation context
# S2: complete subject + relation + object context
# Matched wrappers preserve the same sentence frame while the semantic context is progressively completed.
Z0=[];Z1=[];Z2=[];Z2WRONG=[];Z1WRONG=[]
for qi,(s,r,o) in enumerate(FACTS):
    z0=torch.stack([cap(f"Entity: {s}"),cap(f"Subject: {s}"),cap(f"This concerns {s}")]).mean(0)
    z1=torch.stack([cap(f"{s} {r}"),cap(f"Relation: {s} {r}"),cap(f"This concerns what {s} {r}")]).mean(0)
    z2=torch.stack([cap(f"{s} {r} {o}."),cap(f"Fact: {s} {r} {o}."),cap(f"This concerns the fact that {s} {r} {o}.")]).mean(0)
    oj=FACTS[(qi+1)%M][2];rj=FACTS[(qi+1)%M][1]
    z2w=torch.stack([cap(f"{s} {r} {oj}."),cap(f"Fact: {s} {r} {oj}."),cap(f"This concerns the fact that {s} {r} {oj}.")]).mean(0)
    z1w=torch.stack([cap(f"{s} {rj}"),cap(f"Relation: {s} {rj}"),cap(f"This concerns what {s} {rj}")]).mean(0)
    Z0.append(z0);Z1.append(z1);Z2.append(z2);Z2WRONG.append(z2w);Z1WRONG.append(z1w)
Z0=torch.stack(Z0);Z1=torch.stack(Z1);Z2=torch.stack(Z2);Z2WRONG=torch.stack(Z2WRONG);Z1WRONG=torch.stack(Z1WRONG)
print("[5/16] Contextual transition vectors...")
VREL=unit(Z1-Z0);VOBJ=unit(Z2-Z1);VDIRECT=unit(Z2-Z0);VOBJ_WRONG=unit(Z2WRONG-Z1);VREL_WRONG=unit(Z1WRONG-Z0)
# Wrong-relation complete continuation keeps the correct object but changes relation.
Z2RWRONG=[]
for qi,(s,r,o) in enumerate(FACTS):
    rj=FACTS[(qi+1)%M][1]
    z=torch.stack([cap(f"{s} {rj} {o}."),cap(f"Fact: {s} {rj} {o}."),cap(f"This concerns the fact that {s} {rj} {o}.")]).mean(0)
    Z2RWRONG.append(z)
Z2RWRONG=torch.stack(Z2RWRONG);VOBJ_AFTER_WRONGREL=unit(Z2RWRONG-Z1WRONG)
print("[6/16] Transition geometry...")
for qi in range(M):
    print(f"Q{qi+1}")
    for L in ACTIVE:
        ro=float(VREL[qi,L]@VOBJ[qi,L]);ow=float(VOBJ[qi,L]@VOBJ_WRONG[qi,L]);rw=float(VREL[qi,L]@VREL_WRONG[qi,L]);dd=float(VDIRECT[qi,L]@VOBJ[qi,L])
        nr=float((Z1[qi,L]-Z0[qi,L]).norm());no=float((Z2[qi,L]-Z1[qi,L]).norm());nd=float((Z2[qi,L]-Z0[qi,L]).norm())
        print(f" L{L:02d} ||ΔR||={nr:.3f} ||ΔO||={no:.3f} ||ΔD||={nd:.3f} cos(R,O)={ro:+.4f} cos(O,Owrong)={ow:+.4f} cos(R,Rwrong)={rw:+.4f} cos(D,O)={dd:+.4f}")
print("[7/16] Depth-adaptive contextual chain...")
# The transition itself is model-derived. Depth chooses a smooth blend according to relative contextual-transition strength.
CHAIN=[];REV=[];WRONGOBJ=[];WRONGREL=[];PHASE=[]
for qi in range(M):
    cv=[];rv=[];wv=[];wrv=[];ph=[]
    for L in range(N):
        nr=(Z1[qi,L]-Z0[qi,L]).norm();no=(Z2[qi,L]-Z1[qi,L]).norm()
        # Model-derived transition phase: relative object-transition magnitude, lightly depth-regularized to preserve S→R→O order.
        geom=float(no/(nr+no+EPS));depth=L/max(1,N-1);a=max(0.,min(1.,.65*depth+.35*geom))
        cv.append(unit(((1-a)*VREL[qi,L]+a*VOBJ[qi,L])[None])[0])
        rv.append(unit((a*VREL[qi,L]+(1-a)*VOBJ[qi,L])[None])[0])
        wv.append(unit(((1-a)*VREL[qi,L]+a*VOBJ_WRONG[qi,L])[None])[0])
        wrv.append(unit(((1-a)*VREL_WRONG[qi,L]+a*VOBJ_AFTER_WRONGREL[qi,L])[None])[0]);ph.append(a)
    CHAIN.append(torch.stack(cv));REV.append(torch.stack(rv));WRONGOBJ.append(torch.stack(wv));WRONGREL.append(torch.stack(wrv));PHASE.append(ph)
CHAIN=torch.stack(CHAIN);REV=torch.stack(REV);WRONGOBJ=torch.stack(WRONGOBJ);WRONGREL=torch.stack(WRONGREL)
LREP=ACTIVE[len(ACTIVE)//2]
for qi in range(M):print(f"Q{qi+1} phase@ACTIVE="+str({L:round(PHASE[qi][L],4) for L in ACTIVE}))
print("[8/16] Controls + unchanged SEASC engine...")
def direction(qi,b):
    if b=="CHAIN":return CHAIN[qi]
    if b=="DIRECT":return VDIRECT[qi]
    if b=="RELATION_ONLY":return VREL[qi]
    if b=="OBJECT_ONLY":return VOBJ[qi]
    if b=="REVERSED":return REV[qi]
    if b=="WRONG_OBJECT":return WRONGOBJ[qi]
    if b=="WRONG_RELATION":return WRONGREL[qi]
    if b=="NEG":return -CHAIN[qi]
    raise ValueError(b)
def hooks(qi,scale,b,tele=None):
    hs=[];D=direction(qi,b)
    def mk(L):
        def hk(m,args,out):
            raw=out[0] if isinstance(out,tuple) else out
            if not bool(MASK[L]):return out
            B=raw.shape[0];a=D[L][None].expand(B,-1).float().contiguous();dose=float(RHO[L])*float(scale)
            d=torch.full((B,),dose,device=raw.device,dtype=torch.float32);y=ext.inject(raw,a,d)
            if tele is not None:tele[L]=(dose,float(raw.float().norm(dim=-1).mean())*dose)
            return y if not isinstance(out,tuple) else (y,)+tuple(out[1:])
        return hk
    for L in range(N):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
@torch.inference_mode()
def generate(qi,scale=0.,b="CHAIN",n=40):
    e=tok(chat(QUEST[qi]),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(qi,scale,b,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="CHAIN"):
    p=tok(chat(QUEST[qi]),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);hs=[]
    try:
        if scale>0:hs=hooks(qi,scale,b)
        z=model(input_ids=torch.cat([p,y],1),use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    l=torch.log_softmax(z[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(l.sum()),float(l.mean()),int(y.shape[1])
def margin(qi,scale=0.,b="CHAIN"):
    ts,tm,tn=lp(qi,FACTS[qi][2],scale,b);wrong=[lp(qi,FACTS[j][2],scale,b)[0] for j in range(M) if j!=qi];bw=max(wrong)
    return ts,tm,tn,bw,ts-bw
print("[9/16] CHAIN dose sweep...")
BASE=[];SWEEP={}
for qi in range(M):
    out,_=generate(qi);BASE.append((out,*margin(qi)))
    for sc in SCALES:
        out,te=generate(qi,sc,"CHAIN");SWEEP[(qi,sc)]=(out,*margin(qi,sc,"CHAIN"),te)
print("[10/16] Contextual transition controls...")
BRANCHES=["DIRECT","CHAIN","RELATION_ONLY","OBJECT_ONLY","REVERSED","WRONG_OBJECT","WRONG_RELATION","NEG"];RESULT={}
for b in BRANCHES:
    for qi in range(M):
        out,te=generate(qi,.5,b);RESULT[(b,qi)]=(out,*margin(qi,.5,b),te)
@torch.inference_mode()
def xray(qi,b="CHAIN",scale=.5):
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
    hs=hooks(qi,scale,b)+caps(B);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
print("[11/16] X-Ray...")
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[12/16] Selectivity diagnostics...")
for qi in range(M):
    c=RESULT[("CHAIN",qi)][5];wo=RESULT[("WRONG_OBJECT",qi)][5];wr=RESULT[("WRONG_RELATION",qi)][5];rv=RESULT[("REVERSED",qi)][5]
    print(f"Q{qi+1} CHAIN={c:+.4f} CHAIN-WRONG_OBJECT={c-wo:+.4f} CHAIN-WRONG_RELATION={c-wr:+.4f} CHAIN-REVERSED={c-rv:+.4f}")
print("[13/16] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[14/16] RESULTS")
print("\n"+"="*128);print("TEST 210 RESULTS");print("="*128)
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("ACTIVE:",ACTIVE)
print("\nCONTEXTUAL TRANSITION GEOMETRY @ REPRESENTATIVE ACTIVE LAYER",LREP)
for qi in range(M):
    print(f"Q{qi+1} cos(R,O)={float(VREL[qi,LREP]@VOBJ[qi,LREP]):+.5f} cos(O,Owrong)={float(VOBJ[qi,LREP]@VOBJ_WRONG[qi,LREP]):+.5f} cos(R,Rwrong)={float(VREL[qi,LREP]@VREL_WRONG[qi,LREP]):+.5f} phase={PHASE[qi][LREP]:.4f}")
print("\nCHAIN DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {QUEST[qi]}");print(f" VANILLA targetLP={b[1]:+.4f} meanTok={b[2]:+.4f} tokens={b[3]} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" CHAIN {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
print("\nCONTEXTUAL TRANSITION CONTROLS @ .50")
for br in BRANCHES:
    ms=[];print("\n"+br)
    for qi in range(M):
        r=RESULT[(br,qi)];ms.append(r[5]);print(f" Q{qi+1} targetLP={r[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" mean_margin={np.mean(ms):+.4f}")
print("\nSELECTIVITY")
for qi in range(M):
    c=RESULT[("CHAIN",qi)][5]
    print(f"Q{qi+1} CHAIN={c:+.4f} vs WRONG_OBJECT Δ={c-RESULT[('WRONG_OBJECT',qi)][5]:+.4f} vs WRONG_RELATION Δ={c-RESULT[('WRONG_RELATION',qi)][5]:+.4f} vs REVERSED Δ={c-RESULT[('REVERSED',qi)][5]:+.4f}")
print("\nX-RAY Q1 @ .50")
for br,x in XR.items():print(f"{br:14s} L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("\nCHAIN LAYER X-RAY")
for L in range(TOTAL):print(f"L{L:02d} {XR['CHAIN'][L]*100:7.3f}% {'INJECT' if L<N and bool(MASK[L]) else 'OBSERVE' if L<N else 'MOTOR-OFF'}")
print("-"*128)
print("Weights: PASS | L20-L27 injection: ZERO | Blind questions contain no target object")
print("TEST209 baseline preserved: addressing, CUDA SEASC motor, dose envelope, blind evaluation, X-Ray and sentinel")
print("TEST210 forge: h(S) -> h(S,R) -> h(S,R,O); relation and bound-object transitions extracted from model-native contextual states")
print("Primary controls: DIRECT / CHAIN / RELATION_ONLY / OBJECT_ONLY / REVERSED / WRONG_OBJECT / WRONG_RELATION / NEG")
print("Evaluation: greedy generation + exact multi-token SUM/MEAN logP + transition geometry + X-Ray")
print("="*128);print("[15/16] TEST 210 COMPLETE")
print("[16/16] END")



