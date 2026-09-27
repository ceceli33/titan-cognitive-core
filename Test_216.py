# ==================================================================================================
# TEST 216 — KV-CONDITIONED LATENT OBJECT-SLOT BRIDGE
# TEST215 FIX: INTERVENED PREFILL → FROZEN KV → EXACT AUTOREGRESSIVE CANDIDATE LOGP
# --------------------------------------------------------------------------------------------------
# LINEAGE:
# TEST211 -> isolated object identity
# TEST212 -> keyed identity transport
# TEST213 -> OBJECT/OBJECT_END identity localization
# TEST214 -> sequence-carrier transport
# TEST215 -> answer-boundary slot; generation valid, old concatenated LP invalid for slot assay
# TEST216 -> same slot intervention + correct prefill/KV-conditioned autoregressive scoring
# --------------------------------------------------------------------------------------------------
# TEST215 MODEL/SYSTEM/FACTS/CUDA MOTOR PRESERVED | SOURCE L04-L08 | L0-L19 | L20-L27 MOTOR OFF
# CORRECT_SLOT / WRONG_SLOT / SHUFFLED_LAYER / NEG_SLOT / SEQUENCE
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=216
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
ALT_OBJECTS=["the golden necklace","the iron dagger","the crystal mirror","the wooden mask","the scarlet book","the ivory ring","the copper bell","the black feather"]
SRC_WINDOW=list(range(4,9));COMMON_RANK=2;SCALES=[.25,.50,1.00];M=len(FACTS)
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*128);print("TEST 216 — KV-CONDITIONED LATENT OBJECT-SLOT BRIDGE");print("="*128)
print("LINEAGE: TEST211 -> TEST212 -> TEST213 -> TEST214 -> TEST215 -> TEST216")
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f} | SOURCE WINDOW={SRC_WINDOW}")
BUILD="/tmp/test216";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""
#include <torch/extension.h>
torch::Tensor inject_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d,torch::Tensor mask);
torch::Tensor inject(torch::Tensor h,torch::Tensor a,torch::Tensor d,torch::Tensor mask){TORCH_CHECK(h.is_cuda()&&a.is_cuda()&&d.is_cuda()&&mask.is_cuda());return inject_cuda(h,a,d,mask);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("inject",&inject);}
"""
CUDA=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename T> __global__ void k(T*h,const float*a,const float*d,const float*mask,int B,int S,int H){
int v=blockIdx.x,b=v/S,s=v%S;if(b>=B||mask[(long long)b*S+s]<=0)return;extern __shared__ float sh[];long long x=(long long)v*H,y=(long long)b*H;float ss=0;
for(int j=threadIdx.x;j<H;j+=blockDim.x){float q=(float)h[x+j];ss+=q*q;}sh[threadIdx.x]=ss;__syncthreads();
for(unsigned q=blockDim.x/2;q;q>>=1){if(threadIdx.x<q)sh[threadIdx.x]+=sh[threadIdx.x+q];__syncthreads();}
float z=d[b]*sqrtf(fmaxf(sh[0],1e-20f));for(int j=threadIdx.x;j<H;j+=blockDim.x)h[x+j]=(T)((float)h[x+j]+z*a[y+j]);}
torch::Tensor inject_cuda(torch::Tensor h,torch::Tensor a,torch::Tensor d,torch::Tensor mask){
auto o=h.contiguous().clone(),aa=a.to(h.device(),torch::kFloat32).contiguous(),dd=d.to(h.device(),torch::kFloat32).contiguous(),mm=mask.to(h.device(),torch::kFloat32).contiguous();
int B=o.size(0),S=o.size(1),H=o.size(2);constexpr int T=256;cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,o.scalar_type(),"inj",[&]{k<scalar_t><<<B*S,T,T*sizeof(float),stream>>>(o.data_ptr<scalar_t>(),aa.data_ptr<float>(),dd.data_ptr<float>(),mm.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
"""
ext=load_inline(name="test216_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/20] Model...")
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
def unit(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(EPS)
def chat(x):return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def ids(x):return tok(x,add_special_tokens=False).input_ids
def subseq(hay,needle):
    if not needle:return []
    return [list(range(i,i+len(needle))) for i in range(len(hay)-len(needle)+1) if hay[i:i+len(needle)]==needle]
def last_span(full,text):
    a=subseq(full,ids(text))
    if a:return a[-1]
    a=subseq(full,ids(" "+text))
    return a[-1] if a else []
def unwrap(out):return out[0] if isinstance(out,tuple) else out
@torch.inference_mode()
def capture(text):
    e=tok(chat(text),return_tensors="pt",add_special_tokens=False).to(DEVICE);store=[None]*TOTAL;hs=[]
    def mk(L):
        def hk(m,args,out):store[L]=unwrap(out)[0].float().detach().clone()
        return hk
    for L in range(TOTAL):hs.append(layers[L].register_forward_hook(mk(L)))
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    return {"ids":e.input_ids[0].detach().cpu().tolist(),"res":store}
def fact_text(s,r,o):return f"Fact: {s} {r} {o}."
def qform(s,r):
    v={"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r]
    return f"What does {s} {v}?"
QUEST=[qform(s,r) for s,r,o in FACTS]
print("[2/20] TEST215 source captures...")
SRC={}
for qi,(s,r,o) in enumerate(FACTS):
    objs=[o,FACTS[(qi+1)%M][2]]+ALT_OBJECTS
    for j,obj in enumerate(objs):SRC[(qi,j)]=capture(fact_text(s,r,obj))
    print(f"Q{qi+1} captures={len(objs)}")
print("[3/20] OBJECT_END alignment...")
OBJEND={}
for qi,(s,r,o) in enumerate(FACTS):
    objs=[o,FACTS[(qi+1)%M][2]]+ALT_OBJECTS
    for j,obj in enumerate(objs):
        sp=last_span(SRC[(qi,j)]["ids"],obj)
        if not sp:raise RuntimeError(f"Object alignment failed Q{qi+1}/{obj}")
        OBJEND[(qi,j)]=sp[-1]
    print(f"Q{qi+1} target OBJECT_END={OBJEND[(qi,0)]}")
print("[4/20] Identity residual forge...")
IDENT=torch.zeros(M,N,H,device=DEVICE);WRONG=torch.zeros_like(IDENT);IDFRAC=torch.zeros(M,N,device=DEVICE);IDCOS=torch.zeros(M,N,device=DEVICE)
for qi in range(M):
    for L in range(N):
        target=SRC[(qi,0)]["res"][L][OBJEND[(qi,0)]]
        X=[]
        for j in range(1,len(ALT_OBJECTS)+2):X.append(target-SRC[(qi,j)]["res"][L][OBJEND[(qi,j)]])
        X=torch.stack(X);mu=X.mean(0);Xc=X-mu;_,_,Vh=torch.linalg.svd(Xc,full_matrices=False);U=Vh[:min(COMMON_RANK,Vh.shape[0])]
        c=X[0]-mu;it=c-(c@U.T)@U;w=X[1]-mu;iw=w-(w@U.T)@U
        IDENT[qi,L]=unit(it[None])[0];WRONG[qi,L]=unit(iw[None])[0];IDFRAC[qi,L]=it.norm()/X[0].norm().clamp_min(EPS);IDCOS[qi,L]=IDENT[qi,L]@WRONG[qi,L]
for qi in range(M):print(f"Q{qi+1} "+" ".join(f"L{L}:{float(IDFRAC[qi,L]):.3f}/{float(IDCOS[qi,L]):+.3f}" for L in SRC_WINDOW))
print("[5/20] Blind query maps...")
QENC=[];CARRIER=[]
for qi,(s,r,o) in enumerate(FACTS):
    e=tok(chat(QUEST[qi]),return_tensors="pt",add_special_tokens=False).to(DEVICE);full=e.input_ids[0].detach().cpu().tolist()
    ss=last_span(full,s);rv={"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r];rr=last_span(full,rv)
    if not ss or not rr:raise RuntimeError(f"Query alignment failed Q{qi+1}")
    QENC.append(e);CARRIER.append(sorted(set(ss+rr)));print(f"Q{qi+1} carrier={CARRIER[-1]} ANSWER_SLOT={len(full)-1} tokens={len(full)}")
print("[6/20] Layer mapping...")
def src_layer(L):
    if L<=4:return 4
    if L>=8:return 8
    return L
MAP=[src_layer(L) for L in range(N)];REV={4:8,5:7,6:6,7:5,8:4};SHMAP=[REV[MAP[L]] for L in range(N)]
print("MAP  :",MAP);print("SHUFF:",SHMAP)
def direction(qi,L,b):
    if b in ["CORRECT_SLOT","SEQUENCE"]:return IDENT[qi,MAP[L]]
    if b=="WRONG_SLOT":return WRONG[qi,MAP[L]]
    if b=="SHUFFLED_LAYER":return IDENT[qi,SHMAP[L]]
    if b=="NEG_SLOT":return -IDENT[qi,MAP[L]]
    raise ValueError(b)
print("[7/20] Prefill-only CUDA hooks...")
def hooks(qi,scale,b,tele=None):
    hs=[]
    def mk(L):
        def hk(m,args,out):
            raw=unwrap(out);B,S,Hh=raw.shape
            if S<=1:return out
            mask=torch.zeros((B,S),device=raw.device,dtype=torch.float32)
            if b=="SEQUENCE":
                for p in CARRIER[qi]:
                    if p<S:mask[:,p]=1.
            else:mask[:,-1]=1.
            npos=float(mask[0].sum().item())
            if npos<=0:return out
            dose=float(RHO[L])*float(scale)
            if b=="SEQUENCE":dose/=math.sqrt(npos)
            a=direction(qi,L,b)[None].expand(B,-1).float().contiguous();d=torch.full((B,),dose,device=raw.device,dtype=torch.float32)
            y=ext.inject(raw,a,d,mask)
            if tele is not None:tele[L]=(dose,npos)
            return y if not isinstance(out,tuple) else (y,)+tuple(out[1:])
        return hk
    for L in range(N):hs.append(layers[L].register_forward_hook(mk(L)))
    return hs
print("[8/20] Intervened prefill → KV engine...")
@torch.inference_mode()
def prefill(qi,scale=0.,b="CORRECT_SLOT"):
    e=QENC[qi];hs=[];te={}
    try:
        if scale>0:hs=hooks(qi,scale,b,te)
        o=model(**e,use_cache=True,return_dict=True)
    finally:
        for h in hs:h.remove()
    return o.logits[:,-1,:].float(),o.past_key_values,te
@torch.inference_mode()
def kv_lp(qi,answer,scale=0.,b="CORRECT_SLOT"):
    y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)[0]
    if y.numel()==0:return 0.,0.,0
    logits,pkv,_=prefill(qi,scale,b);vals=[]
    for i,t in enumerate(y):
        v=torch.log_softmax(logits[0],-1)[t];vals.append(v)
        if i<y.numel()-1:
            o=model(input_ids=t.view(1,1),past_key_values=pkv,use_cache=True,return_dict=True)
            logits=o.logits[:,-1,:].float();pkv=o.past_key_values
    z=torch.stack(vals)
    return float(z.sum()),float(z.mean()),int(y.numel())
def kv_margin(qi,scale=0.,b="CORRECT_SLOT"):
    ts,tm,tn=kv_lp(qi,FACTS[qi][2],scale,b);wrong=[kv_lp(qi,FACTS[j][2],scale,b)[0] for j in range(M) if j!=qi];bw=max(wrong)
    return ts,tm,tn,bw,ts-bw
print("[9/20] KV-vs-concatenated vanilla equivalence check...")
@torch.inference_mode()
def concat_vanilla_lp(qi,answer):
    p=QENC[qi].input_ids;y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE)
    z=model(input_ids=torch.cat([p,y],1),use_cache=False,return_dict=True).logits.float()
    l=torch.log_softmax(z[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(l.sum())
for qi in range(M):
    a=kv_lp(qi,FACTS[qi][2])[0];c=concat_vanilla_lp(qi,FACTS[qi][2]);print(f"Q{qi+1} KV={a:+.6f} CONCAT={c:+.6f} |Δ|={abs(a-c):.6e}")
print("[10/20] Generation...")
@torch.inference_mode()
def generate(qi,scale=0.,b="CORRECT_SLOT",n=40):
    e=QENC[qi];p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(qi,scale,b,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
print("[11/20] CORRECT_SLOT KV dose sweep...")
BASE=[];SWEEP={}
for qi in range(M):
    out,_=generate(qi);BASE.append((out,*kv_margin(qi)))
    for sc in SCALES:
        out,te=generate(qi,sc,"CORRECT_SLOT");SWEEP[(qi,sc)]=(out,*kv_margin(qi,sc,"CORRECT_SLOT"),te)
print("[12/20] Controls...")
BRANCHES=["CORRECT_SLOT","WRONG_SLOT","SHUFFLED_LAYER","NEG_SLOT","SEQUENCE"];RESULT={}
for b in BRANCHES:
    for qi in range(M):
        out,te=generate(qi,.5,b);RESULT[(b,qi)]=(out,*kv_margin(qi,.5,b),te)
print("[13/20] Selectivity...")
for qi in range(M):
    c=RESULT[("CORRECT_SLOT",qi)][5]
    print(f"Q{qi+1} CORRECT={c:+.4f} vs WRONG Δ={c-RESULT[('WRONG_SLOT',qi)][5]:+.4f} vs SHUFFLE Δ={c-RESULT[('SHUFFLED_LAYER',qi)][5]:+.4f} vs NEG Δ={c-RESULT[('NEG_SLOT',qi)][5]:+.4f} vs SEQUENCE Δ={c-RESULT[('SEQUENCE',qi)][5]:+.4f}")
print("[14/20] X-Ray...")
@torch.inference_mode()
def xray(qi,b="CORRECT_SLOT",scale=.5):
    e=QENC[qi];pos=e.input_ids.shape[1]-1;A={};B={}
    def caps(store):
        hs=[]
        for L in range(TOTAL):
            def mk(li):
                def hk(m,args,out):store[li]=unwrap(out)[0,pos].float().detach().clone()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
        return hs
    hs=caps(A);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    hs=hooks(qi,scale,b)+caps(B);model(**e,use_cache=False,return_dict=True)
    for h in hs:h.remove()
    return [float((B[L]-A[L]).norm()/A[L].norm().clamp_min(EPS)) for L in range(TOTAL)]
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[15/20] First-token causal effect...")
FIRST={}
for qi in range(M):
    target_ids=tok(FACTS[qi][2],add_special_tokens=False).input_ids
    tid=target_ids[0]
    row={}
    for b in BRANCHES:
        lv,_,_=prefill(qi,.5,b);row[b]=float(torch.log_softmax(lv[0],-1)[tid])
    lv,_,_=prefill(qi,0.);row["VANILLA"]=float(torch.log_softmax(lv[0],-1)[tid]);FIRST[qi]=row
    print(f"Q{qi+1} vanilla={row['VANILLA']:+.4f} correct={row['CORRECT_SLOT']:+.4f} Δ={row['CORRECT_SLOT']-row['VANILLA']:+.4f}")
print("[16/20] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[17/20] RESULTS")
print("\n"+"="*128);print("TEST 216 RESULTS");print("="*128)
print("LINEAGE: TEST211 -> TEST212 -> TEST213 -> TEST214 -> TEST215 -> TEST216")
print("SOURCE: residual OBJECT_END identity L04-L08 | TARGET: answer-boundary prefill slot | SCORING: intervened prefill KV")
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("\nVANILLA KV EQUIVALENCE")
for qi in range(M):
    a=kv_lp(qi,FACTS[qi][2])[0];c=concat_vanilla_lp(qi,FACTS[qi][2]);print(f"Q{qi+1} KV={a:+.6f} CONCAT={c:+.6f} |Δ|={abs(a-c):.6e}")
print("\nCORRECT_SLOT KV DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {QUEST[qi]}");print(f" VANILLA targetLP={b[1]:+.4f} meanTok={b[2]:+.4f} tokens={b[3]} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" SLOT {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
print("\nCONTROLS @ .50")
for br in BRANCHES:
    ms=[];print("\n"+br)
    for qi in range(M):
        r=RESULT[(br,qi)];ms.append(r[5]);print(f" Q{qi+1} targetLP={r[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" mean_margin={np.mean(ms):+.4f}")
print("\nSELECTIVITY")
for qi in range(M):
    c=RESULT[("CORRECT_SLOT",qi)][5]
    print(f"Q{qi+1} CORRECT={c:+.4f} vs WRONG Δ={c-RESULT[('WRONG_SLOT',qi)][5]:+.4f} vs SHUFFLE Δ={c-RESULT[('SHUFFLED_LAYER',qi)][5]:+.4f} vs NEG Δ={c-RESULT[('NEG_SLOT',qi)][5]:+.4f} vs SEQUENCE Δ={c-RESULT[('SEQUENCE',qi)][5]:+.4f}")
print("\nFIRST TARGET TOKEN LOGP @ .50")
for qi,r in FIRST.items():print(f"Q{qi+1} VANILLA={r['VANILLA']:+.4f} CORRECT={r['CORRECT_SLOT']:+.4f} Δ={r['CORRECT_SLOT']-r['VANILLA']:+.4f} WRONG={r['WRONG_SLOT']:+.4f} SHUFFLE={r['SHUFFLED_LAYER']:+.4f} NEG={r['NEG_SLOT']:+.4f}")
print("\nX-RAY Q1 @ .50")
for br,x in XR.items():print(f"{br:16s} L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("\nCORRECT_SLOT FINAL-TOKEN X-RAY")
for L in range(TOTAL):print(f"L{L:02d} {XR['CORRECT_SLOT'][L]*100:7.3f}% {'INJECT' if L<N else 'MOTOR-OFF'}")
print("-"*128)
print("Weights: PASS | L20-L27 injection: ZERO | Decode injection: ZERO | Blind questions contain no target object")
print("TEST216 fixes TEST215 scoring: intervention occurs during prompt prefill, candidate tokens are scored autoregressively from the resulting frozen KV state")
print("Candidate answers never enter forge, routing or prefill intervention")
print("PASS requires CORRECT_SLOT selective KV-logP advantage over WRONG_SLOT / SHUFFLED_LAYER / NEG_SLOT, not hidden displacement alone")
print("="*128);print("[18/20] TEST 216 COMPLETE");print("[19/20] MOTOR OFF VERIFIED");print("[20/20] END")



