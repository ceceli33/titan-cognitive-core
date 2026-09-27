# ==================================================================================================
# TEST 212 — KEYED OBJECT-IDENTITY TRANSPORT
# (SUBJECT,RELATION) KEY × ISOLATED OBJECT-IDENTITY VALUE
# --------------------------------------------------------------------------------------------------
# LINEAGE:
# TEST 103      -> DIBEKGOZ: projection / row-span decomposition
# TEST 142      -> donor-constrained internal direction formation
# TEST 192-197  -> downstream transport / local predictive operator + BUILD-span support
# TEST 205      -> associative KEY→VALUE routing
# TEST 210      -> contextual ΔO = h(S,R,O)-h(S,R)
# TEST 211      -> COMMON completion removal -> isolated OBJECT_IDENTITY
# TEST 212      -> frozen query key gates isolated identity value: Δh ∝ I_O * <K_SR,h_query>
# --------------------------------------------------------------------------------------------------
# TEST211 BASELINE PRESERVED | SAME CUDA SEASC MOTOR | SAME ADDRESSING | L0-L19 | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=212
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
ALT_OBJECTS=["the golden necklace","the iron dagger","the crystal mirror","the wooden mask","the scarlet book","the ivory ring","the copper bell","the black feather"]
SCALES=[.25,.50,1.00];BETA=12.;M=len(FACTS);COMMON_RANK=2
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*128);print("TEST 212 — KEYED OBJECT-IDENTITY TRANSPORT");print("="*128)
print("LINEAGE: TEST103 -> TEST142 -> TEST192-197 -> TEST205 -> TEST210 -> TEST211 -> TEST212")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f}")
BUILD="/tmp/test212";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test212_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
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
print("[2/18] TEST211 competitive addressing...")
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
print("[3/18] Frozen query-only routes...")
ROUTE=torch.zeros(M,N,M,device=DEVICE)
for qi in range(M):
    sc=torch.einsum("lh,lmh->lm",unit(QH[qi]),KEY);ROUTE[qi]=torch.softmax(BETA*sc,-1);av=ROUTE[qi,ACTIVE].mean(0)
    print(f"Q{qi+1} expected=M{qi+1} predicted=M{int(av.argmax())+1} weights="+str([round(float(x),4) for x in av]))
print("[4/18] TEST211 object-identity forge...")
def avgcap(xs):return torch.stack([cap(x) for x in xs]).mean(0)
def state_sr(s,r):return avgcap([f"{s} {r}",f"Relation: {s} {r}",f"This concerns what {s} {r}"])
def state_sro(s,r,o):return avgcap([f"{s} {r} {o}.",f"Fact: {s} {r} {o}.",f"This concerns the fact that {s} {r} {o}."])
ZSR=[];ZT=[];ZW=[];BANK=[]
for qi,(s,r,o) in enumerate(FACTS):
    sr=state_sr(s,r);ZSR.append(sr);ZT.append(state_sro(s,r,o));ZW.append(state_sro(s,r,FACTS[(qi+1)%M][2]))
    objs=[FACTS[j][2] for j in range(M) if j!=qi]+ALT_OBJECTS
    BANK.append(torch.stack([state_sro(s,r,x)-sr for x in objs]))
ZSR=torch.stack(ZSR);ZT=torch.stack(ZT);ZW=torch.stack(ZW);BANK=torch.stack(BANK)
RAW=ZT-ZSR;RAW_WRONG=ZW-ZSR;VRAW=unit(RAW)
IDENT=torch.zeros_like(RAW);IDENT_WRONG=torch.zeros_like(RAW);COMMON=torch.zeros_like(RAW)
COMMON_RATIO=torch.zeros(M,N,device=DEVICE)
for qi in range(M):
    for L in range(N):
        X=BANK[qi,:,L].float();mu=X.mean(0);Xc=X-mu;_,_,Vh=torch.linalg.svd(Xc,full_matrices=False);U=Vh[:min(COMMON_RANK,Vh.shape[0])]
        tc=RAW[qi,L]-mu;wc=RAW_WRONG[qi,L]-mu
        rt=tc-(tc@U.T)@U;rw=wc-(wc@U.T)@U
        IDENT[qi,L]=rt;IDENT_WRONG[qi,L]=rw;COMMON[qi,L]=RAW[qi,L]-rt
        COMMON_RATIO[qi,L]=1.-(rt.norm()**2/(tc.norm()**2+EPS))
VIDENT=unit(IDENT);VIDENT_WRONG=unit(IDENT_WRONG);VCOMMON=unit(COMMON)
print("[5/18] Identity geometry...")
LREP=ACTIVE[len(ACTIVE)//2]
for qi in range(M):
    print(f"Q{qi+1}")
    for L in ACTIVE:
        print(f" L{L:02d} rawCos={float(unit(RAW[qi,L][None])[0]@unit(RAW_WRONG[qi,L][None])[0]):+.4f} identityCos={float(VIDENT[qi,L]@VIDENT_WRONG[qi,L]):+.4f} commonRemoved={float(COMMON_RATIO[qi,L]):.4f}")
print("[6/18] Keyed identity operator...")
# Frozen query-only operator:
# a_i(L)=softmax(beta*cos(q_L,K_i,L)); V_ROUTE(L)=Σ_i a_i(L)*I_i(L)
# No candidate answer and no decode hidden state can alter routing.
ROUTED=torch.zeros(M,N,H,device=DEVICE);UNIFORM=torch.zeros(M,N,H,device=DEVICE);SHUF=torch.zeros(M,N,H,device=DEVICE)
for qi in range(M):
    for L in range(N):
        ROUTED[qi,L]=unit(torch.einsum("m,mh->h",ROUTE[qi,L],VIDENT[:,L])[None])[0]
        UNIFORM[qi,L]=unit(VIDENT[:,L].mean(0,keepdim=True))[0]
        SHUF[qi,L]=unit(torch.einsum("m,mh->h",ROUTE[qi,L],VIDENT[torch.tensor([(j+1)%M for j in range(M)],device=DEVICE),L])[None])[0]
print("[7/18] Operator diagnostics...")
for qi in range(M):
    print(f"Q{qi+1}")
    for L in ACTIVE:
        selfcos=float(ROUTED[qi,L]@VIDENT[qi,L]);wrong=max(float(ROUTED[qi,L]@VIDENT[j,L]) for j in range(M) if j!=qi)
        print(f" L{L:02d} routeSelfCos={selfcos:+.4f} bestOtherCos={wrong:+.4f} routeMargin={selfcos-wrong:+.4f} selfWeight={float(ROUTE[qi,L,qi]):.4f}")
print("[8/18] Controls...")
def direction(qi,b):
    if b=="KEYED_IDENTITY":return ROUTED[qi]
    if b=="ORACLE_IDENTITY":return VIDENT[qi]
    if b=="RAW_OBJECT":return VRAW[qi]
    if b=="COMMON":return VCOMMON[qi]
    if b=="WRONG_IDENTITY":return VIDENT_WRONG[qi]
    if b=="SHUFFLED_VALUE":return SHUF[qi]
    if b=="UNIFORM_VALUE":return UNIFORM[qi]
    if b=="NEG_KEYED":return -ROUTED[qi]
    raise ValueError(b)
print("[9/18] Unchanged SEASC CUDA motor...")
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
def generate(qi,scale=0.,b="KEYED_IDENTITY",n=40):
    e=tok(chat(QUEST[qi]),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(qi,scale,b,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="KEYED_IDENTITY"):
    p=tok(chat(QUEST[qi]),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);hs=[]
    try:
        if scale>0:hs=hooks(qi,scale,b)
        z=model(input_ids=torch.cat([p,y],1),use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    l=torch.log_softmax(z[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(l.sum()),float(l.mean()),int(y.shape[1])
def margin(qi,scale=0.,b="KEYED_IDENTITY"):
    ts,tm,tn=lp(qi,FACTS[qi][2],scale,b);wrong=[lp(qi,FACTS[j][2],scale,b)[0] for j in range(M) if j!=qi];bw=max(wrong)
    return ts,tm,tn,bw,ts-bw
print("[10/18] KEYED_IDENTITY dose sweep...")
BASE=[];SWEEP={}
for qi in range(M):
    out,_=generate(qi);BASE.append((out,*margin(qi)))
    for sc in SCALES:
        out,te=generate(qi,sc,"KEYED_IDENTITY");SWEEP[(qi,sc)]=(out,*margin(qi,sc,"KEYED_IDENTITY"),te)
print("[11/18] Operator controls...")
BRANCHES=["KEYED_IDENTITY","ORACLE_IDENTITY","RAW_OBJECT","COMMON","WRONG_IDENTITY","SHUFFLED_VALUE","UNIFORM_VALUE","NEG_KEYED"];RESULT={}
for b in BRANCHES:
    for qi in range(M):
        out,te=generate(qi,.5,b);RESULT[(b,qi)]=(out,*margin(qi,.5,b),te)
@torch.inference_mode()
def xray(qi,b="KEYED_IDENTITY",scale=.5):
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
print("[12/18] X-Ray...")
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[13/18] Mechanistic selectivity...")
for qi in range(M):
    k=RESULT[("KEYED_IDENTITY",qi)][5];o=RESULT[("ORACLE_IDENTITY",qi)][5];s=RESULT[("SHUFFLED_VALUE",qi)][5];u=RESULT[("UNIFORM_VALUE",qi)][5];n=RESULT[("NEG_KEYED",qi)][5]
    print(f"Q{qi+1} KEYED={k:+.4f} vs ORACLE Δ={k-o:+.4f} vs SHUFFLE Δ={k-s:+.4f} vs UNIFORM Δ={k-u:+.4f} vs NEG Δ={k-n:+.4f}")
print("[14/18] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[15/18] RESULTS")
print("\n"+"="*128);print("TEST 212 RESULTS");print("="*128)
print("LINEAGE: TEST103 -> TEST142 -> TEST192-197 -> TEST205 associative routing -> TEST210 contextual ΔO -> TEST211 isolated identity -> TEST212 keyed identity")
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("ACTIVE:",ACTIVE,"| COMMON_RANK:",COMMON_RANK)
print("\nIDENTITY GEOMETRY @ REPRESENTATIVE ACTIVE LAYER",LREP)
for qi in range(M):
    rc=float(unit(RAW[qi,LREP][None])[0]@unit(RAW_WRONG[qi,LREP][None])[0]);ic=float(VIDENT[qi,LREP]@VIDENT_WRONG[qi,LREP])
    print(f"Q{qi+1} rawCos={rc:+.5f} identityCos={ic:+.5f} commonRemoved={float(COMMON_RATIO[qi,LREP]):.4f}")
print("\nKEYED OPERATOR @ REPRESENTATIVE ACTIVE LAYER",LREP)
for qi in range(M):
    selfcos=float(ROUTED[qi,LREP]@VIDENT[qi,LREP]);other=max(float(ROUTED[qi,LREP]@VIDENT[j,LREP]) for j in range(M) if j!=qi)
    print(f"Q{qi+1} selfWeight={float(ROUTE[qi,LREP,qi]):.4f} routeSelfCos={selfcos:+.5f} bestOtherCos={other:+.5f} routeMargin={selfcos-other:+.5f}")
print("\nKEYED_IDENTITY DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {QUEST[qi]}");print(f" VANILLA targetLP={b[1]:+.4f} meanTok={b[2]:+.4f} tokens={b[3]} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" KEYED {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
print("\nOPERATOR CONTROLS @ .50")
for br in BRANCHES:
    ms=[];print("\n"+br)
    for qi in range(M):
        r=RESULT[(br,qi)];ms.append(r[5]);print(f" Q{qi+1} targetLP={r[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" mean_margin={np.mean(ms):+.4f}")
print("\nMECHANISTIC SELECTIVITY")
for qi in range(M):
    k=RESULT[("KEYED_IDENTITY",qi)][5]
    print(f"Q{qi+1} KEYED={k:+.4f} vs ORACLE Δ={k-RESULT[('ORACLE_IDENTITY',qi)][5]:+.4f} vs SHUFFLE Δ={k-RESULT[('SHUFFLED_VALUE',qi)][5]:+.4f} vs UNIFORM Δ={k-RESULT[('UNIFORM_VALUE',qi)][5]:+.4f} vs NEG Δ={k-RESULT[('NEG_KEYED',qi)][5]:+.4f}")
print("\nX-RAY Q1 @ .50")
for br,x in XR.items():print(f"{br:18s} L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("\nKEYED_IDENTITY LAYER X-RAY")
for L in range(TOTAL):print(f"L{L:02d} {XR['KEYED_IDENTITY'][L]*100:7.3f}% {'INJECT' if L<N and bool(MASK[L]) else 'OBSERVE' if L<N else 'MOTOR-OFF'}")
print("-"*128)
print("Weights: PASS | L20-L27 injection: ZERO | Blind questions contain no target object")
print("Routing is frozen from vanilla query only: candidate answers and decode hidden states cannot alter memory selection")
print("TEST211 identity forge preserved; TEST212 changes transport only: frozen (S,R) key weights select isolated object-identity values")
print("Primary test: KEYED_IDENTITY must outperform SHUFFLED_VALUE / UNIFORM_VALUE / WRONG or NEG controls, not merely move hidden state")
print("="*128);print("[16/18] TEST 212 COMPLETE");print("[17/18] MOTOR OFF VERIFIED");print("[18/18] END")


