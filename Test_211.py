# ==================================================================================================
# TEST 211 — OBJECT-IDENTITY RESIDUAL FORGE
# COMMON COMPLETION SUBTRACTION × DONOR/BUILD-SPAN SUPPORT
# --------------------------------------------------------------------------------------------------
# LINEAGE:
# TEST 103      -> DIBEKGOZ: projection / row-span decomposition
# TEST 142      -> donor-constrained internal direction formation
# TEST 192-196  -> downstream transport / local predictive support
# TEST 197      -> BUILD-span coverage / support-capacity measurement
# TEST 209      -> bridge/path forge + X-Ray lineage
# TEST 210      -> h(S) -> h(S,R) -> h(S,R,O) contextual transitions
# TEST 211      -> ΔO = COMMON_COMPLETION + OBJECT_IDENTITY
#                  IDENTITY_RESIDUAL -> DONOR/BUILD-SPAN SUPPORTED_IDENTITY -> SEASC
# --------------------------------------------------------------------------------------------------
# TEST210 MOTOR PRESERVED: Qwen2.5-7B | BF16 | SDPA | frozen weights | CUDA frozen-norm injection
# SEASC LOCK: IVME=.10 SONUM=.30 ZIRVE=.70 TABAN=.20 | L0-L19 | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=211
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
ALT_OBJECTS=["the golden necklace","the iron dagger","the crystal mirror","the wooden mask","the scarlet book","the ivory ring","the copper bell","the black feather"]
SCALES=[.25,.50,1.00];BETA=12.;M=len(FACTS);COMMON_RANK=2;SUPPORT_RANK=3
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*128);print("TEST 211 — OBJECT-IDENTITY RESIDUAL FORGE");print("="*128)
print("LINEAGE: TEST103 -> TEST142 -> TEST192-197 -> TEST209 -> TEST210 -> TEST211")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f}")
BUILD="/tmp/test211";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test211_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/18] Model...")
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
print("[2/18] Competitive addressing...")
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
print("[3/18] Frozen routes...")
for qi in range(M):
    sc=torch.einsum("lh,lmh->lm",unit(QH[qi]),KEY);a=torch.softmax(BETA*sc,-1);av=a[ACTIVE].mean(0)
    print(f"Q{qi+1} expected=M{qi+1} predicted=M{int(av.argmax())+1} weights="+str([round(float(x),4) for x in av]))
print("[4/18] TEST210 contextual states...")
def avgcap(xs):return torch.stack([cap(x) for x in xs]).mean(0)
def state_sr(s,r):return avgcap([f"{s} {r}",f"Relation: {s} {r}",f"This concerns what {s} {r}"])
def state_sro(s,r,o):return avgcap([f"{s} {r} {o}.",f"Fact: {s} {r} {o}.",f"This concerns the fact that {s} {r} {o}."])
ZSR=[];ZT=[];ZW=[]
for qi,(s,r,o) in enumerate(FACTS):
    ZSR.append(state_sr(s,r));ZT.append(state_sro(s,r,o));ZW.append(state_sro(s,r,FACTS[(qi+1)%M][2]))
ZSR=torch.stack(ZSR);ZT=torch.stack(ZT);ZW=torch.stack(ZW)
RAW=ZT-ZSR;RAW_WRONG=ZW-ZSR;VRAW=unit(RAW);VRAW_WRONG=unit(RAW_WRONG)
print("[5/18] Alternative-object transition banks...")
BANK=[]
for qi,(s,r,o) in enumerate(FACTS):
    objs=[FACTS[j][2] for j in range(M) if j!=qi]+ALT_OBJECTS
    BANK.append(torch.stack([state_sro(s,r,x)-ZSR[qi] for x in objs]))
BANK=torch.stack(BANK) # [M,K,L,H]
print("BANK:",tuple(BANK.shape))
print("[6/18] COMMON completion SVD...")
COMMON=torch.zeros_like(RAW);IDENT=torch.zeros_like(RAW);IDENT_WRONG=torch.zeros_like(RAW)
COMMON_BASIS=[[None for L in range(N)] for qi in range(M)]
COMMON_COV=torch.zeros(M,N,device=DEVICE);IDENT_FRAC=torch.zeros(M,N,device=DEVICE)
for qi in range(M):
    for L in range(N):
        X=BANK[qi,:,L].float();mu=X.mean(0);Xc=X-mu
        _,_,Vh=torch.linalg.svd(Xc,full_matrices=False);r=min(COMMON_RANK,Vh.shape[0]);U=Vh[:r]
        COMMON_BASIS[qi][L]=U
        t=RAW[qi,L];w=RAW_WRONG[qi,L]
        pt=mu+((t-mu)@U.T)@U;pw=mu+((w-mu)@U.T)@U
        COMMON[qi,L]=pt;IDENT[qi,L]=t-pt;IDENT_WRONG[qi,L]=w-pw
        COMMON_COV[qi,L]=(pt.norm()**2)/(t.norm()**2+EPS);IDENT_FRAC[qi,L]=IDENT[qi,L].norm()/(t.norm()+EPS)
VIDENT=unit(IDENT);VIDENT_WRONG=unit(IDENT_WRONG);VCOMMON=unit(COMMON)
print("[7/18] Donor/BUILD support spans...")
# Donors are object-identity residuals from alternative objects under the SAME (S,R).
DONOR=[];SUPP=torch.zeros_like(IDENT);COVERAGE=torch.zeros(M,N,device=DEVICE)
for qi in range(M):
    dl=[]
    for L in range(N):
        Uc=COMMON_BASIS[qi][L];X=BANK[qi,:,L];mu=X.mean(0);res=[]
        for j in range(X.shape[0]):
            x=X[j]-mu;x=x-(x@Uc.T)@Uc;res.append(x)
        D=torch.stack(res);_,_,Vh=torch.linalg.svd(D,full_matrices=False);r=min(SUPPORT_RANK,Vh.shape[0]);B=Vh[:r]
        t=IDENT[qi,L];p=(t@B.T)@B;SUPP[qi,L]=p;COVERAGE[qi,L]=(p.norm()**2)/(t.norm()**2+EPS);dl.append(B)
    DONOR.append(dl)
VSUPP=unit(SUPP)
print("[8/18] Identity geometry...")
LREP=ACTIVE[len(ACTIVE)//2]
for qi in range(M):
    print(f"Q{qi+1}")
    for L in ACTIVE:
        cr=float(VRAW[qi,L]@VRAW_WRONG[qi,L]);ci=float(VIDENT[qi,L]@VIDENT_WRONG[qi,L]);cs=float(VSUPP[qi,L]@VIDENT_WRONG[qi,L])
        print(f" L{L:02d} rawCos={cr:+.4f} identityCos={ci:+.4f} supported-vs-wrong={cs:+.4f} commonCov={float(COMMON_COV[qi,L]):.4f} identityFrac={float(IDENT_FRAC[qi,L]):.4f} supportCov={float(COVERAGE[qi,L]):.4f}")
print("[9/18] Controls...")
# Cross-object identity control uses the next fact's object residual under the current subject/relation.
WRONG_SUPPORTED=torch.zeros_like(SUPP)
for qi in range(M):
    for L in range(N):
        B=DONOR[qi][L];t=IDENT_WRONG[qi,L];WRONG_SUPPORTED[qi,L]=(t@B.T)@B
VWRONG_SUPP=unit(WRONG_SUPPORTED)
def direction(qi,b):
    if b=="RAW_OBJECT":return VRAW[qi]
    if b=="COMMON":return VCOMMON[qi]
    if b=="IDENTITY_RESIDUAL":return VIDENT[qi]
    if b=="SUPPORTED_IDENTITY":return VSUPP[qi]
    if b=="WRONG_IDENTITY":return VWRONG_SUPP[qi]
    if b=="NEG_IDENTITY":return -VSUPP[qi]
    raise ValueError(b)
print("[10/18] Unchanged TEST210 SEASC engine...")
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
def generate(qi,scale=0.,b="SUPPORTED_IDENTITY",n=40):
    e=tok(chat(QUEST[qi]),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(qi,scale,b,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="SUPPORTED_IDENTITY"):
    p=tok(chat(QUEST[qi]),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);hs=[]
    try:
        if scale>0:hs=hooks(qi,scale,b)
        z=model(input_ids=torch.cat([p,y],1),use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    l=torch.log_softmax(z[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(l.sum()),float(l.mean()),int(y.shape[1])
def margin(qi,scale=0.,b="SUPPORTED_IDENTITY"):
    ts,tm,tn=lp(qi,FACTS[qi][2],scale,b);wrong=[lp(qi,FACTS[j][2],scale,b)[0] for j in range(M) if j!=qi];bw=max(wrong)
    return ts,tm,tn,bw,ts-bw
print("[11/18] SUPPORTED_IDENTITY dose sweep...")
BASE=[];SWEEP={}
for qi in range(M):
    out,_=generate(qi);BASE.append((out,*margin(qi)))
    for sc in SCALES:
        out,te=generate(qi,sc,"SUPPORTED_IDENTITY");SWEEP[(qi,sc)]=(out,*margin(qi,sc,"SUPPORTED_IDENTITY"),te)
print("[12/18] Branch controls...")
BRANCHES=["RAW_OBJECT","COMMON","IDENTITY_RESIDUAL","SUPPORTED_IDENTITY","WRONG_IDENTITY","NEG_IDENTITY"];RESULT={}
for b in BRANCHES:
    for qi in range(M):
        out,te=generate(qi,.5,b);RESULT[(b,qi)]=(out,*margin(qi,.5,b),te)
@torch.inference_mode()
def xray(qi,b="SUPPORTED_IDENTITY",scale=.5):
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
print("[13/18] X-Ray...")
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[14/18] Selectivity diagnostics...")
for qi in range(M):
    s=RESULT[("SUPPORTED_IDENTITY",qi)][5];w=RESULT[("WRONG_IDENTITY",qi)][5];n=RESULT[("NEG_IDENTITY",qi)][5];r=RESULT[("RAW_OBJECT",qi)][5]
    print(f"Q{qi+1} SUPPORTED={s:+.4f} vs WRONG Δ={s-w:+.4f} vs NEG Δ={s-n:+.4f} vs RAW Δ={s-r:+.4f}")
print("[15/18] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[16/18] RESULTS")
print("\n"+"="*128);print("TEST 211 RESULTS");print("="*128)
print("LINEAGE: TEST103 projection -> TEST142 donor constraint -> TEST192-197 transport/BUILD span -> TEST210 contextual ΔO -> TEST211 identity residual")
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("ACTIVE:",ACTIVE,"| COMMON_RANK:",COMMON_RANK,"| SUPPORT_RANK:",SUPPORT_RANK)
print("\nIDENTITY GEOMETRY @ REPRESENTATIVE ACTIVE LAYER",LREP)
for qi in range(M):
    print(f"Q{qi+1} RAW target/wrong cos={float(VRAW[qi,LREP]@VRAW_WRONG[qi,LREP]):+.5f} ID target/wrong cos={float(VIDENT[qi,LREP]@VIDENT_WRONG[qi,LREP]):+.5f} commonCov={float(COMMON_COV[qi,LREP]):.4f} identityFrac={float(IDENT_FRAC[qi,LREP]):.4f} supportCov={float(COVERAGE[qi,LREP]):.4f}")
print("\nSUPPORTED_IDENTITY DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {QUEST[qi]}");print(f" VANILLA targetLP={b[1]:+.4f} meanTok={b[2]:+.4f} tokens={b[3]} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" SUPPORTED {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
print("\nBRANCH CONTROLS @ .50")
for br in BRANCHES:
    ms=[];print("\n"+br)
    for qi in range(M):
        r=RESULT[(br,qi)];ms.append(r[5]);print(f" Q{qi+1} targetLP={r[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" mean_margin={np.mean(ms):+.4f}")
print("\nSELECTIVITY")
for qi in range(M):
    s=RESULT[("SUPPORTED_IDENTITY",qi)][5]
    print(f"Q{qi+1} SUPPORTED={s:+.4f} vs WRONG Δ={s-RESULT[('WRONG_IDENTITY',qi)][5]:+.4f} vs NEG Δ={s-RESULT[('NEG_IDENTITY',qi)][5]:+.4f} vs RAW Δ={s-RESULT[('RAW_OBJECT',qi)][5]:+.4f}")
print("\nX-RAY Q1 @ .50")
for br,x in XR.items():print(f"{br:20s} L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("\nSUPPORTED_IDENTITY LAYER X-RAY")
for L in range(TOTAL):print(f"L{L:02d} {XR['SUPPORTED_IDENTITY'][L]*100:7.3f}% {'INJECT' if L<N and bool(MASK[L]) else 'OBSERVE' if L<N else 'MOTOR-OFF'}")
print("-"*128)
print("Weights: PASS | L20-L27 injection: ZERO | Blind questions contain no target object")
print("TEST210 CUDA motor/addressing/dose envelope/blind evaluation/X-Ray/sentinel preserved")
print("TEST211 forge only: RAW ΔO -> COMMON completion SVD -> IDENTITY residual -> donor/BUILD-span supported identity")
print("Primary test: does RAW target/wrong cosine collapse after identity isolation while support coverage remains nonzero?")
print("="*128);print("[17/18] TEST 211 COMPLETE");print("[18/18] END")



