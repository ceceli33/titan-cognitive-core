# ==================================================================================================
# TEST 162 — FISHER → BANK-MEAN CAUSAL DECOMPOSITION
# AKBASCORE 3.1 / SEASC
# Working TEST161 motor preserved. Frozen weights. Same SEARCH / EVAL / Csym.
# ==================================================================================================

import os,sys,math,json,csv,random,shutil,subprocess
from pathlib import Path

print("="*150)
print("TEST 162 — FISHER → BANK-MEAN CAUSAL DECOMPOSITION")
print("AKBASCORE 3.1 / SEASC")
print("="*150)

subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers>=4.45.0","accelerate>=0.34.0","ninja","pybind11"])
os.environ["MAX_JOBS"]="2"; os.environ["TOKENIZERS_PARALLELISM"]="false"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda"); SEED=162
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"; N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
DOSE_SCALES=[.25,.50,1.00]
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST162_FISHER_BANK_DECOMPOSITION"); ROOT.mkdir(parents=True,exist_ok=True)

print("GPU            :",torch.cuda.get_device_name(0))
print("Model          :",MODEL_ID)
print("Steered layers : L0-L19")
print("Dose scales    :",DOSE_SCALES)

# ==================================================================================================
# 1. CUDA — EXACT WORKING TEST161 PATTERN
# ==================================================================================================

BUILD_DIR="/tmp/akbascore_test162"
shutil.rmtree(BUILD_DIR,ignore_errors=True); os.makedirs(BUILD_DIR,exist_ok=True)

CPP_SOURCE=r"""
#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden, torch::Tensor actuator, torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden, torch::Tensor actuator, torch::Tensor dose) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(actuator.is_cuda(), "actuator must be CUDA");
    TORCH_CHECK(dose.is_cuda(), "dose must be CUDA");
    TORCH_CHECK(hidden.dim()==3, "hidden must be [B,S,H]");
    TORCH_CHECK(actuator.dim()==2, "actuator must be [B,H]");
    TORCH_CHECK(dose.dim()==1, "dose must be [B]");
    TORCH_CHECK(hidden.size(0)==actuator.size(0), "batch mismatch");
    TORCH_CHECK(hidden.size(0)==dose.size(0), "dose batch mismatch");
    TORCH_CHECK(hidden.size(2)==actuator.size(1), "hidden mismatch");
    return seasc_batch_cuda(hidden,actuator,dose);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST162 SEASC");}
"""

CUDA_SOURCE=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template <typename scalar_t>
__global__ void seasc_batch_kernel(scalar_t* __restrict__ h,const float* __restrict__ actuator,
                                   const float* __restrict__ dose,const int B,const int S,const int H){
    const int vec=blockIdx.x,b=vec/S;if(b>=B)return;
    extern __shared__ float shared[];
    const long long base=(long long)vec*(long long)H,abase=(long long)b*(long long)H;
    float local_sum=0.0f;
    for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[base+j];local_sum+=x*x;}
    shared[threadIdx.x]=local_sum;__syncthreads();
    for(unsigned int stride=blockDim.x/2;stride>0;stride>>=1){
        if(threadIdx.x<stride)shared[threadIdx.x]+=shared[threadIdx.x+stride];__syncthreads();
    }
    const float norm0=sqrtf(fmaxf(shared[0],1.0e-20f));
    const float scale=dose[b]*norm0;__syncthreads();
    for(int j=threadIdx.x;j<H;j+=blockDim.x){
        float old_value=(float)h[base+j];
        h[base+j]=(scalar_t)(old_value+scale*actuator[abase+j]);
    }
}
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
    auto out=hidden.contiguous().clone();
    auto act=actuator.to(hidden.device(),torch::kFloat32).contiguous();
    auto d=dose.to(hidden.device(),torch::kFloat32).contiguous();
    const int B=(int)out.size(0),S=(int)out.size(1),H=(int)out.size(2),n_vectors=B*S;
    constexpr int THREADS=256;const size_t shared_bytes=THREADS*sizeof(float);
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test162_seasc_batch",[&]{
        seasc_batch_kernel<scalar_t><<<n_vectors,THREADS,shared_bytes,stream>>>(
            out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
"""

print("\n[1/10] Compiling AkbasCore SEASC kernel...")
seasc_ext=load_inline(name="akbascore_test162_cuda",cpp_sources=CPP_SOURCE,cuda_sources=CUDA_SOURCE,
    functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],
    with_cuda=True,build_directory=BUILD_DIR,verbose=False)
print("      CUDA kernel: OK")

# ==================================================================================================
# 2. MODEL / ENVELOPE
# ==================================================================================================

print("\n[2/10] Loading frozen model...")
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},
    attn_implementation="sdpa",trust_remote_code=True)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
layers=model.model.layers; HIDDEN=model.config.hidden_size
if len(layers)<N_LAYERS: raise RuntimeError("Model has fewer than 20 layers.")
print("      hidden :",HIDDEN); print("      layers :",len(layers)); print("      dtype  :",next(model.parameters()).dtype)

def dra_envelope(L):
    t=float(L); kb=ZIRVE*math.exp(-SONUM*t)*(1.0+SONUM*t)+TABAN
    return kb/(ZIRVE+TABAN)
ENVELOPE=[dra_envelope(L) for L in range(N_LAYERS)]

# ==================================================================================================
# 3. SAME TEST161 DATA
# ==================================================================================================

SEARCH_PAIRS=[
("The lantern is above the stone arch.","The lantern is below the stone arch."),
("The kite is above the old tower.","The kite is below the old tower."),
("The flag is above the wooden gate.","The flag is below the wooden gate."),
("The bird is above the narrow bridge.","The bird is below the narrow bridge."),
("The sign is above the glass doorway.","The sign is below the glass doorway."),
("The lamp is above the metal shelf.","The lamp is below the metal shelf."),
("The clock is above the brown cabinet.","The clock is below the brown cabinet."),
("The balloon is above the green hill.","The balloon is below the green hill."),
("The camera is above the small desk.","The camera is below the small desk."),
("The painting is above the blue sofa.","The painting is below the blue sofa."),
("The cloud is above the distant ridge.","The cloud is below the distant ridge."),
("The marker is above the white board.","The marker is below the white board."),
("The drone is above the concrete wall.","The drone is below the concrete wall."),
("The shelf is above the storage box.","The shelf is below the storage box."),
("The light is above the front window.","The light is below the front window."),
("The symbol is above the horizontal line.","The symbol is below the horizontal line.")
]

EVAL_ITEMS=[
{"context":"A red marker and a blue marker are visible. Their vertical relation is unspecified.","pos":" The red marker is above the blue marker.","neg":" The red marker is below the blue marker."},
{"context":"A bright lamp and a wooden table are visible. Their vertical relation is unspecified.","pos":" The bright lamp is above the wooden table.","neg":" The bright lamp is below the wooden table."},
{"context":"A bird and a tree branch are visible. Their vertical relation is unspecified.","pos":" The bird is above the tree branch.","neg":" The bird is below the tree branch."},
{"context":"A sign and a doorway are visible. Their vertical relation is unspecified.","pos":" The sign is above the doorway.","neg":" The sign is below the doorway."},
{"context":"A shelf and a cabinet are visible. Their vertical relation is unspecified.","pos":" The shelf is above the cabinet.","neg":" The shelf is below the cabinet."},
{"context":"A cloud and a hill are visible. Their vertical relation is unspecified.","pos":" The cloud is above the hill.","neg":" The cloud is below the hill."},
{"context":"A clock and a desk are visible. Their vertical relation is unspecified.","pos":" The clock is above the desk.","neg":" The clock is below the desk."},
{"context":"A framed picture and a sofa are visible. Their vertical relation is unspecified.","pos":" The framed picture is above the sofa.","neg":" The framed picture is below the sofa."},
{"context":"A drone and a stone wall are visible. Their vertical relation is unspecified.","pos":" The drone is above the stone wall.","neg":" The drone is below the stone wall."},
{"context":"A balloon and a house are visible. Their vertical relation is unspecified.","pos":" The balloon is above the house.","neg":" The balloon is below the house."},
{"context":"A camera and a storage box are visible. Their vertical relation is unspecified.","pos":" The camera is above the storage box.","neg":" The camera is below the storage box."},
{"context":"A silver symbol and a horizontal line are visible. Their vertical relation is unspecified.","pos":" The silver symbol is above the horizontal line.","neg":" The silver symbol is below the horizontal line."}
]

NEUTRAL=["A ceramic cup rests beside a closed notebook.","A wooden chair stands near a plain wall.",
"A glass bottle sits beside a small basket.","A cotton towel rests on a clean counter.",
"A metal toolbox is inside a quiet room.","A paper folder lies beside a keyboard.",
"A black umbrella stands near a cabinet.","A silver spoon rests beside a plate."]
NEUTRAL_CONT=" The scene contains ordinary stationary objects and no additional relation is specified."

def chat_text(x):
    return tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],
        tokenize=False,add_generation_prompt=True)

@torch.inference_mode()
def capture(text):
    enc=tokenizer(chat_text(text),return_tensors="pt").to(DEVICE)
    out=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
    pos=int(enc["attention_mask"][0].sum().item())-1
    hs=[out.hidden_states[L+1][0,pos].float().detach() for L in range(N_LAYERS)]
    del out,enc; return hs

def normalize(v): return v/v.norm(dim=-1,keepdim=True).clamp_min(EPS)

# ==================================================================================================
# 4. FISHER + BANK
# ==================================================================================================

print("\n[3/10] Building SEARCH bank...")
POS_BANK=[[] for _ in range(N_LAYERS)]; NEG_BANK=[[] for _ in range(N_LAYERS)]
for i,(ptext,ntext) in enumerate(SEARCH_PAIRS,1):
    hp,hn=capture(ptext),capture(ntext)
    for L in range(N_LAYERS): POS_BANK[L].append(hp[L]); NEG_BANK[L].append(hn[L])
    print(f"      pair {i:02d}/{len(SEARCH_PAIRS)}")
POS_BANK=[torch.stack(x).float().contiguous() for x in POS_BANK]
NEG_BANK=[torch.stack(x).float().contiguous() for x in NEG_BANK]

print("\n[4/10] Building Fisher / BANK decomposition...")
FISHER=[]; BANK=[]; BANK_RES=[]; FISHER_RES=[]; COS=[]; RESN=[]
for L in range(N_LAYERS):
    Hp,Hn=POS_BANK[L],NEG_BANK[L]; mp,mn=Hp.mean(0),Hn.mean(0); md=mp-mn
    R=torch.cat([Hp-mp,Hn-mn],0).float(); nu=max(int(R.shape[0])-2,1)
    _,S,Vh=torch.linalg.svd(R,full_matrices=False); eig=S.square()/float(nu)
    pe=eig[eig>FISHER_EPS]; es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS); V=Vh.T; pr=V.T@md
    f=md/lam+V@((1/(eig+lam)-1/lam)*pr); f=(f/f.norm().clamp_min(EPS)).float()
    dirs=normalize(Hp-Hn); sg=torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
    dirs=dirs*sg; m=dirs.mean(0); m=(m/m.norm().clamp_min(EPS)).float()
    c=torch.dot(f,m).clamp(-1.,1.); rb=m-c*f; rf=f-c*m
    nb=rb.norm(); nf=rf.norm()
    if nb<EPS or nf<EPS: raise RuntimeError(f"Degenerate residual L{L}")
    rb=(rb/nb).float(); rf=(rf/nf).float()
    FISHER.append(f); BANK.append(m); BANK_RES.append(rb); FISHER_RES.append(rf)
    COS.append(float(c)); RESN.append(float(nb))
    print(f"      L{L:02d} cos(F,M)={float(c):+.6f} | residual={float(nb):.6f}")

FISHER=torch.stack(FISHER).contiguous(); BANK=torch.stack(BANK).contiguous()
BANK_RES=torch.stack(BANK_RES).contiguous(); FISHER_RES=torch.stack(FISHER_RES).contiguous()

# Direct normalized interpolation F -> BANK.
PATHS={}
for t in [.25,.50,.75]:
    PATHS[f"PATH_{int(t*100):02d}"]=normalize((1-t)*FISHER+t*BANK).float().contiguous()

# Add exact BANK-specific orthogonal component progressively.
ORTHO_PATHS={}
for beta in [.25,.50,.75,1.00]:
    z=[]
    for L in range(N_LAYERS):
        c=max(COS[L],1e-8); ratio=RESN[L]/c
        q=FISHER[L]+beta*ratio*BANK_RES[L]; q=q/q.norm().clamp_min(EPS); z.append(q)
    ORTHO_PATHS[f"ORTHO_{int(beta*100):03d}"]=torch.stack(z).float().contiguous()

print("      mean cos(F,BANK) :",f"{np.mean(COS):.6f}")
print("      mean residual    :",f"{np.mean(RESN):.6f}")
print("      ORTHO_100/BANK cos:",f"{torch.sum(ORTHO_PATHS['ORTHO_100']*BANK,dim=1).mean().item():.9f}")

# ==================================================================================================
# 5. EXACT WORKING TEST161 HOOK ENGINE — BUG FIX IS HERE
# ==================================================================================================

def blank_telemetry(): return [{"requested":[],"realized":[]} for _ in range(N_LAYERS)]

def install_hooks(vectors,sign,dose_scale,telemetry):
    handles=[]
    for L in range(N_LAYERS):
        def make_hook(layer_idx):
            def hook(module,args,output):
                if isinstance(output,tuple): old,rest=output[0],output[1:]
                else: old,rest=output,None
                batch=old.shape[0]
                direction=vectors[layer_idx][None,:].expand(batch,-1)
                direction=(float(sign)*direction).float().contiguous()
                requested_value=IVME*ENVELOPE[layer_idx]*dose_scale
                dose=torch.full((batch,),float(requested_value),device=DEVICE,dtype=torch.float32)
                old_last=old[:,-1,:].float()
                new=seasc_ext.seasc_batch(old,direction,dose)
                new_last=new[:,-1,:].float()
                realized=(new_last-old_last).norm(dim=-1)/old_last.norm(dim=-1).clamp_min(EPS)
                telemetry[layer_idx]["requested"].extend(dose.detach().cpu().tolist())
                telemetry[layer_idx]["realized"].extend(realized.detach().cpu().tolist())
                return new if rest is None else (new,)+rest
            return hook
        handles.append(layers[L].register_forward_hook(make_hook(L)))
    return handles

def build_sequence(context,continuation):
    p=tokenizer(chat_text(context),return_tensors="pt",add_special_tokens=False)
    c=tokenizer(continuation,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p["input_ids"],c["input_ids"]],1).to(DEVICE)
    mask=torch.ones_like(ids,device=DEVICE)
    return ids,mask,p["input_ids"].shape[1]

@torch.inference_mode()
def continuation_logp(context,continuation,vectors=None,sign=0,scale=0.0):
    ids,mask,plen=build_sequence(context,continuation); telemetry=blank_telemetry(); handles=[]
    if vectors is not None: handles=install_hooks(vectors,sign,scale,telemetry)
    try: out=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in handles: h.remove()
    logits=out.logits.float(); target=ids[:,plen:]; pred=logits[:,plen-1:-1,:]
    lp=F.log_softmax(pred,dim=-1); token_lp=lp.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    score=float(token_lp.mean().item())
    del out,logits,target,pred,lp,token_lp,ids,mask
    return score,telemetry

# ==================================================================================================
# 6. BASELINE
# ==================================================================================================

print("\n[5/10] Measuring baseline...")
BASE=[]
for i,item in enumerate(EVAL_ITEMS):
    p,_=continuation_logp(item["context"],item["pos"]); n,_=continuation_logp(item["context"],item["neg"])
    margin=p-n; BASE.append({"item":i,"margin":margin}); print(f"      item {i:02d} | margin={margin:+.6f}")
BASE_MEAN=float(np.mean([x["margin"] for x in BASE])); print("      baseline mean =",f"{BASE_MEAN:+.6f}")

# ==================================================================================================
# 7. CAUSAL ARMS
# ==================================================================================================

def run_arm(method,vectors,scale):
    cs,pm,nm,requested,realized=[],[],[],[],[]
    for item in EVAL_ITEMS:
        pp,t1=continuation_logp(item["context"],item["pos"],vectors,+1,scale)
        pn,t2=continuation_logp(item["context"],item["neg"],vectors,+1,scale)
        np_,t3=continuation_logp(item["context"],item["pos"],vectors,-1,scale)
        nn,t4=continuation_logp(item["context"],item["neg"],vectors,-1,scale)
        p=pp-pn; n=np_-nn; c=.5*(p-n); pm.append(p); nm.append(n); cs.append(c)
        for tel in (t1,t2,t3,t4):
            for L in range(N_LAYERS):
                requested.extend(tel[L]["requested"]); realized.extend(tel[L]["realized"])
    return {"method":method,"scale":scale,"plus_margin":float(np.mean(pm)),"minus_margin":float(np.mean(nm)),
        "Csym":float(np.mean(cs)),"Csym_sd":float(np.std(cs,ddof=1)),"Csym_median":float(np.median(cs)),
        "positive_items":int(np.sum(np.asarray(cs)>0)),"requested_mean":float(np.mean(requested)),
        "realized_mean":float(np.mean(realized)),"energy":float(np.sum(np.square(realized))),
        "item_csym":[float(x) for x in cs]}

ARMS={"FISHER":FISHER,"BANK_MEAN":BANK,"BANK_RESIDUAL":BANK_RES,"FISHER_RESIDUAL":FISHER_RES}
ARMS.update(PATHS); ARMS.update(ORTHO_PATHS)

print("\n[6/10] Running causal decomposition...")
ROWS=[]
for scale in DOSE_SCALES:
    print(f"\nDOSE={scale:.2f}")
    for name,v in ARMS.items():
        r=run_arm(name,v,scale); ROWS.append(r)
        print(f"  {name:<18} Csym={r['Csym']:+.6f} | P+={r['positive_items']:02d}/12 | real={100*r['realized_mean']:.4f}% | E={r['energy']:.6f}")

# ==================================================================================================
# 8. CORE CONTRASTS + ITEM-LEVEL PAIRED BOOTSTRAP
# ==================================================================================================

def get(name,scale): return next(r for r in ROWS if r["method"]==name and r["scale"]==scale)

def paired_bootstrap(a,b,n=20000,seed=162):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64); d=a-b
    rng=np.random.default_rng(seed); means=np.empty(n,dtype=np.float64)
    for i in range(n):
        idx=rng.integers(0,len(d),len(d)); means[i]=d[idx].mean()
    return float(d.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)),float((np.sum(means<=0)+1)/(n+1))

print("\n[7/10] Core contrasts...")
CONTRASTS=[]
for scale in DOSE_SCALES:
    f=get("FISHER",scale); m=get("BANK_MEAN",scale); br=get("BANK_RESIDUAL",scale); fr=get("FISHER_RESIDUAL",scale)
    delta,lo,hi,p=paired_bootstrap(m["item_csym"],f["item_csym"],seed=SEED+int(scale*1000))
    ortho=[get(x,scale)["Csym"] for x in ["FISHER","ORTHO_025","ORTHO_050","ORTHO_075","ORTHO_100"]]
    direct=[get(x,scale)["Csym"] for x in ["FISHER","PATH_25","PATH_50","PATH_75","BANK_MEAN"]]
    monotonic_ortho=all(ortho[i+1]>=ortho[i] for i in range(len(ortho)-1))
    monotonic_direct=all(direct[i+1]>=direct[i] for i in range(len(direct)-1))
    reconstruction=abs(get("ORTHO_100",scale)["Csym"]-m["Csym"])
    row={"scale":scale,"bank_minus_fisher":delta,"ci95_low":lo,"ci95_high":hi,"bootstrap_p_one_sided":p,
         "bank_residual":br["Csym"],"fisher_residual":fr["Csym"],"ortho_monotonic":monotonic_ortho,
         "direct_monotonic":monotonic_direct,"reconstruction_error":reconstruction}
    CONTRASTS.append(row)
    print(f"  scale={scale:.2f} | BANK-FISHER={delta:+.6f} | CI95=[{lo:+.6f},{hi:+.6f}] | p={p:.6f}")
    print("     direct:",["%+.5f"%x for x in direct],"mono=",monotonic_direct)
    print("     ortho :",["%+.5f"%x for x in ortho],"mono=",monotonic_ortho,"recon.err=",f"{reconstruction:.8f}")
    print(f"     BANK⊥F={br['Csym']:+.6f} | FISHER⊥BANK={fr['Csym']:+.6f}")

# ==================================================================================================
# 9. NEUTRAL COLLATERAL — KEY ARMS
# ==================================================================================================

@torch.inference_mode()
def neutral_metrics(text,vectors,scale):
    ids,mask,plen=build_sequence(text,NEUTRAL_CONT)
    out0=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    l0=out0.logits.float()[:,plen-1:-1,:]; target=ids[:,plen:]; lp0=F.log_softmax(l0,-1)
    nll0=-lp0.gather(-1,target.unsqueeze(-1)).squeeze(-1).mean()
    tel=blank_telemetry(); handles=install_hooks(vectors,+1,scale,tel)
    try: out1=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in handles: h.remove()
    l1=out1.logits.float()[:,plen-1:-1,:]; lp1=F.log_softmax(l1,-1)
    nll1=-lp1.gather(-1,target.unsqueeze(-1)).squeeze(-1).mean()
    kl=(lp0.exp()*(lp0-lp1)).sum(-1).mean()
    ans=float((nll1-nll0).item()),float(kl.item())
    del ids,mask,out0,out1,l0,l1,target,lp0,lp1
    return ans

print("\n[8/10] Neutral collateral...")
COLLATERAL=[]
CORE={"FISHER":FISHER,"BANK_MEAN":BANK,"BANK_RESIDUAL":BANK_RES}
for scale in DOSE_SCALES:
    for name,v in CORE.items():
        vals=[neutral_metrics(x,v,scale) for x in NEUTRAL]
        row={"method":name,"scale":scale,"delta_nll":float(np.mean([x[0] for x in vals])),
             "kl":float(np.mean([x[1] for x in vals]))}
        COLLATERAL.append(row)
        print(f"  {name:<18} scale={scale:.2f} | ΔNLL={row['delta_nll']:+.6f} | KL={row['kl']:.6f}")

# ==================================================================================================
# 10. SAVE + FINAL
# ==================================================================================================

print("\n[9/10] Saving...")
def safe(rows): return [{k:v for k,v in r.items() if not isinstance(v,(list,dict,tuple))} for r in rows]
def save_csv(name,rows):
    rows=safe(rows)
    if not rows:return
    keys=sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

GEOMETRY=[{"layer":L,"fisher_bank_cos":COS[L],"bank_residual_norm":RESN[L]} for L in range(N_LAYERS)]
save_csv("BASELINE.csv",BASE); save_csv("CAUSAL_ARMS.csv",ROWS); save_csv("CONTRASTS.csv",CONTRASTS)
save_csv("COLLATERAL.csv",COLLATERAL); save_csv("GEOMETRY.csv",GEOMETRY)

np.savez_compressed(ROOT/"TEST162_FISHER_BANK_DECOMPOSITION.npz",
    fisher=FISHER.detach().cpu().numpy(),bank= BANK.detach().cpu().numpy(),
    bank_residual=BANK_RES.detach().cpu().numpy(),fisher_residual=FISHER_RES.detach().cpu().numpy(),
    fisher_bank_cos=np.asarray(COS,dtype=np.float32),bank_residual_norm=np.asarray(RESN,dtype=np.float32),
    envelope=np.asarray(ENVELOPE,dtype=np.float32))

REPORT={"test":"TEST 162","title":"FISHER → BANK-MEAN CAUSAL DECOMPOSITION","model":MODEL_ID,
"gpu":torch.cuda.get_device_name(0),"baseline_mean":BASE_MEAN,"geometry":GEOMETRY,"causal_arms":ROWS,
"contrasts":CONTRASTS,"collateral":COLLATERAL,
"interpretation_contract":[
"Same model, SEARCH pairs, EVAL items, Csym scoring, L0-L19 envelope and frozen-norm SEASC motor as TEST161.",
"BANK_RESIDUAL is the unit BANK_MEAN component orthogonal to FISHER.",
"FISHER_RESIDUAL is the unit FISHER component orthogonal to BANK_MEAN.",
"PATH arms are normalized direct Fisher-to-BANK interpolations.",
"ORTHO arms progressively add the exact BANK-specific orthogonal component to Fisher.",
"ORTHO_100 reconstructs BANK_MEAN up to numerical precision.",
"Residual-only activity tests sufficiency, not necessity.",
"A monotonic Fisher-to-BANK path supports a graded causal contribution from BANK-specific geometry.",
"Non-monotonicity implies the BANK advantage cannot be reduced to a simple linear residual-dose account.",
"All actuator vectors are unit-normalized and use the same requested physical SEASC dose."
]}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f: json.dump(REPORT,f,indent=2,ensure_ascii=False)

print("\n[10/10] COMPLETE")
print("="*150); print("TEST 162 — FINAL SUMMARY"); print("="*150)
print(f"Baseline mean            : {BASE_MEAN:+.6f}")
print(f"Mean cos(Fisher,BANK)    : {np.mean(COS):+.6f}")
print(f"Mean BANK residual norm  : {np.mean(RESN):.6f}")
print(f"ORTHO_100/BANK mean cos  : {torch.sum(ORTHO_PATHS['ORTHO_100']*BANK,dim=1).mean().item():.9f}")

for scale in DOSE_SCALES:
    print(f"\nDOSE SCALE = {scale:.2f}")
    for name in ["FISHER","BANK_MEAN","BANK_RESIDUAL","FISHER_RESIDUAL","PATH_25","PATH_50","PATH_75",
                 "ORTHO_025","ORTHO_050","ORTHO_075","ORTHO_100"]:
        r=get(name,scale)
        print(f"  {name:<18} Csym={r['Csym']:+.6f} | P+={r['positive_items']:02d}/12 | E={r['energy']:.6f}")

print("\nCORE CONTRASTS")
for r in CONTRASTS:
    print(f"  scale={r['scale']:.2f} | BANK-FISHER={r['bank_minus_fisher']:+.6f} | "
          f"CI95=[{r['ci95_low']:+.6f},{r['ci95_high']:+.6f}] | p={r['bootstrap_p_one_sided']:.6f} | "
          f"ORTHO-mono={r['ortho_monotonic']} | DIRECT-mono={r['direct_monotonic']}")

print("\nSaved:",ROOT)
print("="*150); print("TEST 162 COMPLETE"); print("="*150)
