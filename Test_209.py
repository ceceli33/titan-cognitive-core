# ==================================================================================================
# TEST 209 — RESIDUALIZED RELATIONAL BRIDGE FORGE
# TEST208 BASELINE + RAW / CENTERED / PCA-RESIDUAL PATH GEOMETRY
# ORDERED PATH vs DEPTH-SHUFFLE | COMMON-MODE REMOVAL | SAME SEASC MOTOR
# AkbasCore SEASC | Qwen2.5-7B-Instruct | FROZEN-NORM DIRECT INJECTION | L0-L19 | L20-L27 MOTOR OFF
# ==================================================================================================
import os,sys,math,random,shutil,subprocess,importlib.util,re,json
for m,p in [("ninja","ninja"),("torch","torch"),("transformers","transformers")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
DEVICE=torch.device("cuda");SEED=209
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N=20;TOTAL=28;H_EXPECT=3584;EPS=1e-8
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
FACTS=[("Neral Voss","keeps","the amber compass"),("Tovin Marel","carries","the silver lantern"),("Selka Dorn","owns","the violet key"),("Parel Nox","guards","the bronze sphere")]
SCALES=[.25,.50,1.00];CAND_N=24;PATH_K=4;PCA_K=4;BETA=12.;M=len(FACTS)
def env(L):
    x=ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN
    return x/(ZIRVE+TABAN)
RHO=np.array([IVME*env(L) for L in range(N)],np.float32);RSS=float(np.sqrt(np.sum(RHO**2)))
print("="*128);print("TEST 209 — RESIDUALIZED RELATIONAL BRIDGE FORGE");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL_ID,f"| SEASC RSS={RSS:.9f} | PCA_K={PCA_K}")
BUILD="/tmp/test209";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
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
ext=load_inline(name="test209_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if _tv>=(4,56) else "torch_dtype"
print("[1/17] Model...")
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
@torch.inference_mode()
def rawgen(prompt,n=160):
    e=tok(chat(prompt),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,p:],skip_special_tokens=True).strip()
def qforms(s,r):
    v={"keeps":"keep","carries":"carry","owns":"own","guards":"guard"}[r]
    return [f"What does {s} {v}?",f"Which item does {s} {v}?",f"What object does {s} {v}?",f"Name the item that {s} {r}.",f"What item is linked to {s} through the relation '{r}'?",f"Which object belongs in the relation '{s} {r} ___'?"]
QUEST=[qforms(s,r)[0] for s,r,o in FACTS]
for i,(s,r,o) in enumerate(FACTS):
    ow={w for w in re.findall(r"[a-z]+",o.lower()) if len(w)>2 and w!="the"}
    if ow&set(re.findall(r"[a-z]+",QUEST[i].lower())):raise RuntimeError("Question leakage.")
print("[2/17] Competitive addressing...")
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
print("[3/17] Frozen routes...")
for qi in range(M):
    sc=torch.einsum("lh,lmh->lm",unit(QH[qi]),KEY);a=torch.softmax(BETA*sc,-1);av=a[ACTIVE].mean(0)
    print(f"Q{qi+1} expected=M{qi+1} predicted=M{int(av.argmax())+1} weights="+str([round(float(x),4) for x in av]))
print("[4/17] Automatic bridge candidate generation...")
def clean_candidates(text,s,o):
    xs=[]
    for line in text.splitlines():
        line=re.sub(r"^\s*(?:[-*•]|\d+[\.\)])\s*","",line).strip()
        line=re.sub(r"^[\"']|[\"']$","",line).strip()
        if ":" in line and len(line.split(":",1)[0].split())<4:line=line.split(":",1)[1].strip()
        line=re.sub(r"[.;]+$","",line).strip()
        if not line or len(line)>60 or len(line.split())>7:continue
        if line.lower() in {s.lower(),o.lower(),o.replace("the ","").lower()}:continue
        if line.lower() not in [x.lower() for x in xs]:xs.append(line)
    return xs[:CAND_N]
CANDS=[]
for qi,(s,r,o) in enumerate(FACTS):
    prompt=f"""Generate {CAND_N} short semantic bridge concepts that could naturally connect a subject, the relation "{r}", and an object of type "{o}".
The subject name is "{s}". Do not repeat the subject or target object. Do not invent a story about the subject. Produce only general intermediate concepts or relational states, one per line, ordered from subject/relation-side concepts toward object-side concepts. No explanations."""
    xs=clean_candidates(rawgen(prompt,220),s,o)
    if len(xs)<8:raise RuntimeError(f"Too few automatic bridge candidates Q{qi+1}: {xs}")
    CANDS.append(xs);print(f"Q{qi+1} candidates ({len(xs)}):",xs)
print("[5/17] Candidate hidden geometry...")
CSTATE=[]
for qi,xs in enumerate(CANDS):
    z=[]
    for x in xs:z.append(unit(torch.stack([cap(x),cap("Concept: "+x),cap("This concerns "+x+".")]).mean(0)))
    CSTATE.append(torch.stack(z,1))
print("[6/17] Relation-state anchors...")
REL=[];START=[];TARGET=[]
for s,r,o in FACTS:
    START.append(unit(torch.stack([cap(s),cap("Person: "+s)]).mean(0)))
    REL.append(unit(torch.stack([cap(r),cap("Relation: "+r),cap(f"Someone {r} an object.")]).mean(0)))
    TARGET.append(unit(torch.stack([cap(o.replace("the ","")),cap("Object: "+o)]).mean(0)))
START=torch.stack(START);REL=torch.stack(REL);TARGET=torch.stack(TARGET)
print("[7/17] Common-mode calibration...")
CAL_TEXT=["A person is here.","An object is here.","Someone has an item.","A relation connects two things.","A person carries something.","Someone owns an object.","A person protects something.","An item belongs to someone.","There is a physical object.","An artifact exists.","A tool is present.","A generic event occurs.","A subject relates to an object.","Someone interacts with something.","An object has a property.","A person uses an item.","Something is stored somewhere.","The answer is an object.","A question concerns an item.","A neutral statement is given.","One entity connects to another.","A person and object are related.","An item has an owner.","A simple relation is described."]
CAL=torch.stack([cap(x) for x in CAL_TEXT]) # [C,L,H]
MU=CAL.mean(0) # [L,H]
PCS=[]
for L in range(N):
    X=CAL[:,L,:]-MU[L]
    _,_,vh=torch.linalg.svd(X,full_matrices=False)
    PCS.append(vh[:PCA_K].detach())
PCS=torch.stack(PCS) # [L,K,H]
def residual_vec(x,L,mode):
    if mode=="RAW":return unit(x[None])[0]
    y=x-MU[L]
    if mode=="PCA":
        pc=PCS[L];y=y-(y@pc.T)@pc
    return unit(y[None])[0]
def residual_candidates(c,L,mode):
    if mode=="RAW":return unit(c)
    y=c-MU[L][None]
    if mode=="PCA":
        pc=PCS[L];y=y-(y@pc.T)@pc
    return unit(y)
print("[8/17] Common-mode diagnostic...")
for L in ACTIVE:
    raw=[];cen=[];pca=[]
    for qi in range(M):
        raw.append(float(residual_vec(START[qi,L],L,"RAW")@residual_vec(TARGET[qi,L],L,"RAW")))
        cen.append(float(residual_vec(START[qi,L],L,"CENTER")@residual_vec(TARGET[qi,L],L,"CENTER")))
        pca.append(float(residual_vec(START[qi,L],L,"PCA")@residual_vec(TARGET[qi,L],L,"PCA")))
    print(f"L{L:02d} S↔T cosine RAW={np.mean(raw):+.5f} CENTER={np.mean(cen):+.5f} PCA={np.mean(pca):+.5f}")
print("[9/17] RAW/CENTER/PCA relational path forge...")
MODES=["RAW","CENTER","PCA"];PATHV={};PATHIDX={};DIRECT={}
for mode in MODES:
    pv=[];pi=[];dv=[]
    for qi in range(M):
        CV=CSTATE[qi];idx_layers=[];vec_layers=[];dir_layers=[]
        for L in range(N):
            s=residual_vec(START[qi,L],L,mode);r=residual_vec(REL[qi,L],L,mode);t=residual_vec(TARGET[qi,L],L,mode);c=residual_candidates(CV[L],L,mode)
            ss=c@s;sr=c@r;st=c@t;phase=(.45*sr+.55*st)-ss;quality=.30*ss+.35*sr+.35*st;order=torch.argsort(phase);chosen=[]
            for b in torch.chunk(order,PATH_K):
                if len(b):chosen.append(int(b[torch.argmax(quality[b])]))
            chosen=list(dict.fromkeys(chosen))
            if len(chosen)<2:chosen=[int(x) for x in torch.topk(quality,min(PATH_K,len(quality))).indices]
            chosen=sorted(chosen,key=lambda j:float(phase[j]));states=[s]+[c[j] for j in chosen]+[t]
            seg=[unit((states[k+1]-states[k])[None])[0] for k in range(len(states)-1)]
            p=min(len(seg)-1,int((L/max(1,N-1))*len(seg)));vec_layers.append(seg[p]);idx_layers.append(chosen);dir_layers.append(unit((t-s)[None])[0])
        pv.append(torch.stack(vec_layers));pi.append(idx_layers);dv.append(torch.stack(dir_layers))
    PATHV[mode]=torch.stack(pv);PATHIDX[mode]=pi;DIRECT[mode]=torch.stack(dv)
    L=ACTIVE[len(ACTIVE)//2];print(mode)
    for qi,(s,r,o) in enumerate(FACTS):print(f" Q{qi+1} L{L:02d}: {s} -> "+" -> ".join(CANDS[qi][j] for j in PATHIDX[mode][qi][L])+f" -> {o}")
print("[10/17] Path validity + order telemetry...")
for mode in MODES:
    print(mode)
    for qi in range(M):
        L=ACTIVE[len(ACTIVE)//2];ids=PATHIDX[mode][qi][L];s=residual_vec(START[qi,L],L,mode);t=residual_vec(TARGET[qi,L],L,mode);c=residual_candidates(CSTATE[qi][L],L,mode)
        states=[s]+[c[j] for j in ids]+[t];tc=[float(x@t) for x in states];steps=[float(states[k]@states[k+1]) for k in range(len(states)-1)]
        mono=sum(tc[k+1]>=tc[k] for k in range(len(tc)-1))/max(1,len(tc)-1)
        print(f" Q{qi+1} L{L:02d} target_cos="+str([round(x,4) for x in tc])+" local_cos="+str([round(x,4) for x in steps])+f" monotonic={mono:.2f}")
print("[11/17] Controls + unchanged SEASC engine...")
SHUF={}
for mode in MODES:
    z=torch.empty_like(PATHV[mode]);g=torch.Generator(device=DEVICE);g.manual_seed(SEED+{"RAW":71,"CENTER":72,"PCA":73}[mode])
    for qi in range(M):z[qi]=PATHV[mode][qi,torch.randperm(N,generator=g,device=DEVICE)]
    SHUF[mode]=z
def direction(qi,b):
    if b=="RAW":return PATHV["RAW"][qi]
    if b=="RAW_SHUF":return SHUF["RAW"][qi]
    if b=="CENTER":return PATHV["CENTER"][qi]
    if b=="CENTER_SHUF":return SHUF["CENTER"][qi]
    if b=="PCA":return PATHV["PCA"][qi]
    if b=="PCA_SHUF":return SHUF["PCA"][qi]
    if b=="DIRECT":return DIRECT["RAW"][qi]
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
def generate(qi,scale=0.,b="PCA",n=40):
    e=tok(chat(QUEST[qi]),return_tensors="pt").to(DEVICE);p=e.input_ids.shape[1];hs=[];te={}
    try:
        if scale>0:hs=hooks(qi,scale,b,te)
        o=model.generate(**e,max_new_tokens=n,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    finally:
        for h in hs:h.remove()
    return tok.decode(o[0,p:],skip_special_tokens=True).strip(),te
@torch.inference_mode()
def lp(qi,answer,scale=0.,b="PCA"):
    p=tok(chat(QUEST[qi]),return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);y=tok(answer,return_tensors="pt",add_special_tokens=False).input_ids.to(DEVICE);hs=[]
    try:
        if scale>0:hs=hooks(qi,scale,b)
        z=model(input_ids=torch.cat([p,y],1),use_cache=False,return_dict=True).logits.float()
    finally:
        for h in hs:h.remove()
    l=torch.log_softmax(z[0,p.shape[1]-1:p.shape[1]-1+y.shape[1]],-1).gather(1,y[0,:,None]).squeeze(1)
    return float(l.sum()),float(l.mean()),int(y.shape[1])
def margin(qi,scale=0.,b="PCA"):
    ts,tm,tn=lp(qi,FACTS[qi][2],scale,b);wrong=[lp(qi,FACTS[j][2],scale,b)[0] for j in range(M) if j!=qi];bw=max(wrong)
    return ts,tm,tn,bw,ts-bw
print("[12/17] PCA dose sweep...")
BASE=[];SWEEP={}
for qi in range(M):
    out,_=generate(qi);BASE.append((out,*margin(qi)))
    for sc in SCALES:
        out,te=generate(qi,sc,"PCA");SWEEP[(qi,sc)]=(out,*margin(qi,sc,"PCA"),te)
print("[13/17] RAW/CENTER/PCA ordered-vs-shuffled assay...")
BRANCHES=["DIRECT","RAW","RAW_SHUF","CENTER","CENTER_SHUF","PCA","PCA_SHUF"];RESULT={}
for b in BRANCHES:
    for qi in range(M):
        out,te=generate(qi,.5,b);RESULT[(b,qi)]=(out,*margin(qi,.5,b),te)
@torch.inference_mode()
def xray(qi,b="PCA",scale=.5):
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
print("[14/17] X-Ray...")
XR={b:xray(0,b,.5) for b in BRANCHES}
print("[15/17] Weight sentinel...")
if fp()!=FP0:raise RuntimeError("WEIGHT SENTINEL FAILED.")
print("[16/17] RESULTS")
print("\n"+"="*128);print("TEST 209 RESULTS");print("="*128)
for i,f in enumerate(FACTS):print(f"M{i+1}: {f[0]} | {f[1]} | {f[2]}")
print("ACTIVE:",ACTIVE,"| PCA_K:",PCA_K)
print("\nAUTOMATIC PATHS @ REPRESENTATIVE ACTIVE LAYER")
L=ACTIVE[len(ACTIVE)//2]
for mode in MODES:
    print(mode)
    for qi,(s,r,o) in enumerate(FACTS):print(f" Q{qi+1}: {s} -> "+" -> ".join(CANDS[qi][j] for j in PATHIDX[mode][qi][L])+f" -> {o}")
print("\nPCA PATH DOSE SWEEP")
for qi in range(M):
    b=BASE[qi];print(f"\nQ{qi+1}: {QUEST[qi]}");print(f" VANILLA targetLP={b[1]:+.4f} meanTok={b[2]:+.4f} tokens={b[3]} bestWrong={b[4]:+.4f} margin={b[5]:+.4f} | {b[0]}")
    for sc in SCALES:
        r=SWEEP[(qi,sc)];print(f" PCA {sc:.2f} targetLP={r[1]:+.4f} ΔlogP={r[1]-b[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
print("\nRAW / CENTER / PCA ORDER ASSAY @ .50")
for br in BRANCHES:
    ms=[];print("\n"+br)
    for qi in range(M):
        r=RESULT[(br,qi)];ms.append(r[5]);print(f" Q{qi+1} targetLP={r[1]:+.4f} meanTok={r[2]:+.4f} margin={r[5]:+.4f} | {r[0]}")
    print(f" mean_margin={np.mean(ms):+.4f}")
print("\nORDER SENSITIVITY")
for mode in MODES:
    a=np.array([RESULT[(mode,qi)][5] for qi in range(M)]);s=np.array([RESULT[(mode+"_SHUF",qi)][5] for qi in range(M)]);d=a-s
    print(f"{mode:6s} ordered={a.mean():+.4f} shuffled={s.mean():+.4f} ORDERΔ={d.mean():+.4f} perQ="+str([round(float(x),4) for x in d]))
print("\nX-RAY Q1 @ .50")
for br,x in XR.items():print(f"{br:11s} L19={x[19]*100:7.3f}% L27={x[27]*100:7.3f}%")
print("\nPCA PATH LAYER X-RAY")
for L in range(TOTAL):print(f"L{L:02d} {XR['PCA'][L]*100:7.3f}% {'INJECT' if L<N and bool(MASK[L]) else 'OBSERVE' if L<N else 'MOTOR-OFF'}")
print("-"*128)
print("Weights: PASS | L20-L27 injection: ZERO | Blind questions contain no target object")
print("TEST208 baseline preserved: candidate generation, addressing, CUDA SEASC motor, dose envelope, evaluation and X-Ray")
print("TEST209 change: forge geometry only = RAW vs CENTERED vs PCA-RESIDUAL")
print("Primary diagnostic: ordered path minus depth-shuffled path under identical physical dose")
print("No forced addressing fallback | greedy generation | exact multi-token SUM/MEAN logP")
print("="*128);print("[17/17] TEST 209 COMPLETE")



