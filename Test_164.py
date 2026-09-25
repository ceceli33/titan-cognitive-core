# ==================================================================================================
# TEST 164 — VEKTÖR SOY AĞACI DÖKÜMHANESİ
# AKBASCORE 3.2 / SEASC
# Lineage-aware multi-family causal forging -> frozen FINAL
# TEST162/163 frozen-norm SEASC motor + static DRA envelope preserved.
# ==================================================================================================

import os,sys,math,json,csv,random,shutil,subprocess
from pathlib import Path

print("="*150)
print("TEST 164 — VEKTÖR SOY AĞACI DÖKÜMHANESİ")
print("AKBASCORE 3.2 / SEASC")
print("="*150)

subprocess.check_call([
    sys.executable,"-m","pip","install","-q",
    "transformers>=4.45.0","accelerate>=0.34.0","ninja","pybind11"
])

os.environ["MAX_JOBS"]="2"
os.environ["TOKENIZERS_PARALLELISM"]="false"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

DEVICE=torch.device("cuda")
SEED=164

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_LAYERS=20

IVME=.10
SONUM=.30
ZIRVE=.70
TABAN=.20

FISHER_ALPHA=.10
FISHER_EPS=1e-8
EPS=1e-10

FORGE_SCALE=.50
FINAL_SCALES=[.25,.50,1.00]

# Frozen selection contract.
LAMBDA_FAMILY_SD=.35
LAMBDA_KL=.05
LAMBDA_NLL=.10

SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."

ROOT=Path("/content/TEST164_VECTOR_LINEAGE_FOUNDRY")
ROOT.mkdir(parents=True,exist_ok=True)

print("GPU            :",torch.cuda.get_device_name(0))
print("Model          :",MODEL_ID)
print("Steered layers : L0-L19")
print("Forge scale    :",FORGE_SCALE)
print("Final scales   :",FINAL_SCALES)

# ==================================================================================================
# 1. CUDA — SEASC FROZEN-NORM MOTOR
# ==================================================================================================

BUILD_DIR="/tmp/akbascore_test164"
shutil.rmtree(BUILD_DIR,ignore_errors=True)
os.makedirs(BUILD_DIR,exist_ok=True)

CPP_SOURCE=r"""
#include <torch/extension.h>

torch::Tensor seasc_batch_cuda(
    torch::Tensor hidden,
    torch::Tensor actuator,
    torch::Tensor dose
);

torch::Tensor seasc_batch(
    torch::Tensor hidden,
    torch::Tensor actuator,
    torch::Tensor dose
){
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("seasc_batch",&seasc_batch,"TEST164 SEASC");
}
"""

CUDA_SOURCE=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void seasc_batch_kernel(
    scalar_t* __restrict__ h,
    const float* __restrict__ actuator,
    const float* __restrict__ dose,
    const int B,
    const int S,
    const int H
){
    const int vec=blockIdx.x;
    const int b=vec/S;
    if(b>=B)return;

    extern __shared__ float shared[];

    const long long base=(long long)vec*(long long)H;
    const long long abase=(long long)b*(long long)H;

    float local_sum=0.0f;

    for(int j=threadIdx.x;j<H;j+=blockDim.x){
        float x=(float)h[base+j];
        local_sum+=x*x;
    }

    shared[threadIdx.x]=local_sum;
    __syncthreads();

    for(unsigned int stride=blockDim.x/2;stride>0;stride>>=1){
        if(threadIdx.x<stride)
            shared[threadIdx.x]+=shared[threadIdx.x+stride];
        __syncthreads();
    }

    const float norm0=sqrtf(fmaxf(shared[0],1.0e-20f));
    const float scale=dose[b]*norm0;

    __syncthreads();

    for(int j=threadIdx.x;j<H;j+=blockDim.x){
        float old_value=(float)h[base+j];
        h[base+j]=(scalar_t)(
            old_value+scale*actuator[abase+j]
        );
    }
}

torch::Tensor seasc_batch_cuda(
    torch::Tensor hidden,
    torch::Tensor actuator,
    torch::Tensor dose
){
    auto out=hidden.contiguous().clone();
    auto act=actuator.to(hidden.device(),torch::kFloat32).contiguous();
    auto d=dose.to(hidden.device(),torch::kFloat32).contiguous();

    const int B=(int)out.size(0);
    const int S=(int)out.size(1);
    const int H=(int)out.size(2);
    const int n_vectors=B*S;

    constexpr int THREADS=256;
    const size_t shared_bytes=THREADS*sizeof(float);

    cudaStream_t stream=at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        out.scalar_type(),
        "test164_seasc_batch",
        [&]{
            seasc_batch_kernel<scalar_t>
            <<<n_vectors,THREADS,shared_bytes,stream>>>(
                out.data_ptr<scalar_t>(),
                act.data_ptr<float>(),
                d.data_ptr<float>(),
                B,S,H
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

print("\n[1/14] Compiling SEASC kernel...")

seasc_ext=load_inline(
    name="akbascore_test164_cuda",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cflags=["-O3","-std=c++17"],
    extra_cuda_cflags=["-O3","--use_fast_math"],
    with_cuda=True,
    build_directory=BUILD_DIR,
    verbose=False
)

print("       CUDA kernel: OK")

# ==================================================================================================
# 2. MODEL / DRA
# ==================================================================================================

print("\n[2/14] Loading frozen model...")

tokenizer=AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"":0},
    attn_implementation="sdpa",
    trust_remote_code=True
)

model.eval()

for p in model.parameters():
    p.requires_grad_(False)

layers=model.model.layers
HIDDEN=model.config.hidden_size

if len(layers)<N_LAYERS:
    raise RuntimeError("Model has fewer than 20 layers.")

def dra_envelope(L):
    t=float(L)
    kb=ZIRVE*math.exp(-SONUM*t)*(1.0+SONUM*t)+TABAN
    return kb/(ZIRVE+TABAN)

ENVELOPE=[dra_envelope(L) for L in range(N_LAYERS)]

print("       hidden :",HIDDEN)
print("       layers :",len(layers))
print("       dtype  :",next(model.parameters()).dtype)

# ==================================================================================================
# 3. SEARCH BANK
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

# ==================================================================================================
# 4. FOUR FORGE FAMILIES
# ==================================================================================================

FAMILY_A=[
{"context":"A red marker and a blue marker are visible. Their vertical relation is unspecified.",
 "pos":" The red marker is above the blue marker.","neg":" The red marker is below the blue marker."},
{"context":"A bright lamp and a wooden table are visible. Their vertical relation is unspecified.",
 "pos":" The bright lamp is above the wooden table.","neg":" The bright lamp is below the wooden table."},
{"context":"A bird and a tree branch are visible. Their vertical relation is unspecified.",
 "pos":" The bird is above the tree branch.","neg":" The bird is below the tree branch."},
{"context":"A sign and a doorway are visible. Their vertical relation is unspecified.",
 "pos":" The sign is above the doorway.","neg":" The sign is below the doorway."}
]

FAMILY_B=[
{"context":"Two objects are present: a shelf and a cabinet. No vertical ordering is stated.",
 "pos":" The shelf is positioned above the cabinet.","neg":" The shelf is positioned below the cabinet."},
{"context":"A cloud and a hill can be seen, but their relative height is not given.",
 "pos":" The cloud is positioned above the hill.","neg":" The cloud is positioned below the hill."},
{"context":"The scene contains a clock and a desk without specifying their vertical order.",
 "pos":" The clock is positioned above the desk.","neg":" The clock is positioned below the desk."},
{"context":"A framed picture and a sofa appear in the scene. Their relative vertical placement is unknown.",
 "pos":" The framed picture is positioned above the sofa.","neg":" The framed picture is positioned below the sofa."}
]

FAMILY_C=[
{"context":"Relative to a stone wall, the location of a drone along the vertical axis is not specified.",
 "pos":" Vertically, the drone is higher than the stone wall.","neg":" Vertically, the drone is lower than the stone wall."},
{"context":"Relative to a house, the location of a balloon along the vertical axis is not specified.",
 "pos":" Vertically, the balloon is higher than the house.","neg":" Vertically, the balloon is lower than the house."},
{"context":"Relative to a storage box, the location of a camera along the vertical axis is not specified.",
 "pos":" Vertically, the camera is higher than the storage box.","neg":" Vertically, the camera is lower than the storage box."},
{"context":"Relative to a horizontal line, the location of a silver symbol along the vertical axis is not specified.",
 "pos":" Vertically, the silver symbol is higher than the horizontal line.","neg":" Vertically, the silver symbol is lower than the horizontal line."}
]

FAMILY_D=[
{"context":"Consider a helicopter and a stadium. Decide between the two possible vertical relations.",
 "pos":" The helicopter occupies the upper position relative to the stadium.","neg":" The helicopter occupies the lower position relative to the stadium."},
{"context":"Consider a ceiling fan and a dining table. Decide between the two possible vertical relations.",
 "pos":" The ceiling fan occupies the upper position relative to the dining table.","neg":" The ceiling fan occupies the lower position relative to the dining table."},
{"context":"Consider the moon and a mountain. Decide between the two possible vertical relations.",
 "pos":" The moon occupies the upper position relative to the mountain.","neg":" The moon occupies the lower position relative to the mountain."},
{"context":"Consider a security camera and an entrance. Decide between the two possible vertical relations.",
 "pos":" The security camera occupies the upper position relative to the entrance.","neg":" The security camera occupies the lower position relative to the entrance."}
]

FORGE_FAMILIES={
    "FAMILY_A_DIRECT":FAMILY_A,
    "FAMILY_B_POSITIONED":FAMILY_B,
    "FAMILY_C_HIGHER_LOWER":FAMILY_C,
    "FAMILY_D_UPPER_LOWER":FAMILY_D
}

# Untouched FINAL family.
FINAL_ITEMS=[
{"context":"A weather balloon and a field are visible. Their vertical relationship has not been stated.",
 "pos":" The weather balloon is situated above the field.","neg":" The weather balloon is situated below the field."},
{"context":"A speaker and a platform are visible. Their vertical relationship has not been stated.",
 "pos":" The speaker is situated above the platform.","neg":" The speaker is situated below the platform."},
{"context":"A sensor and a laboratory bench are visible. Their vertical relationship has not been stated.",
 "pos":" The sensor is situated above the laboratory bench.","neg":" The sensor is situated below the laboratory bench."},
{"context":"A skylight and a hallway floor are visible. Their vertical relationship has not been stated.",
 "pos":" The skylight is situated above the hallway floor.","neg":" The skylight is situated below the hallway floor."},
{"context":"An aircraft and a runway appear in the scene without a specified vertical relation.",
 "pos":" The aircraft is situated above the runway.","neg":" The aircraft is situated below the runway."},
{"context":"A pendant light and a kitchen island appear in the scene without a specified vertical relation.",
 "pos":" The pendant light is situated above the kitchen island.","neg":" The pendant light is situated below the kitchen island."},
{"context":"A hawk and a fence appear in the scene without a specified vertical relation.",
 "pos":" The hawk is situated above the fence.","neg":" The hawk is situated below the fence."},
{"context":"A road sign and a sidewalk appear in the scene without a specified vertical relation.",
 "pos":" The road sign is situated above the sidewalk.","neg":" The road sign is situated below the sidewalk."},
{"context":"A surveillance drone and a warehouse appear without a specified vertical relation.",
 "pos":" The surveillance drone is situated above the warehouse.","neg":" The surveillance drone is situated below the warehouse."},
{"context":"A hanging microphone and a stage floor appear without a specified vertical relation.",
 "pos":" The hanging microphone is situated above the stage floor.","neg":" The hanging microphone is situated below the stage floor."},
{"context":"A star and the horizon are visible without a specified vertical relation.",
 "pos":" The star is situated above the horizon.","neg":" The star is situated below the horizon."},
{"context":"A projector and a conference table are visible without a specified vertical relation.",
 "pos":" The projector is situated above the conference table.","neg":" The projector is situated below the conference table."}
]

NEUTRAL_FORGE=[
"A ceramic cup rests beside a closed notebook.",
"A wooden chair stands near a plain wall.",
"A glass bottle sits beside a small basket.",
"A cotton towel rests on a clean counter."
]

NEUTRAL_FINAL=[
"A metal toolbox is inside a quiet room.",
"A paper folder lies beside a keyboard.",
"A black umbrella stands near a cabinet.",
"A silver spoon rests beside a plate.",
"A cardboard box sits beside a printer.",
"A small radio rests near a window."
]

NEUTRAL_CONT=" The scene contains ordinary stationary objects and no additional relation is specified."

# ==================================================================================================
# 5. HELPERS
# ==================================================================================================

def chat_text(x):
    return tokenizer.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":x}],
        tokenize=False,
        add_generation_prompt=True
    )

def normalize(v):
    return v/v.norm(dim=-1,keepdim=True).clamp_min(EPS)

@torch.inference_mode()
def capture(text):
    enc=tokenizer(chat_text(text),return_tensors="pt").to(DEVICE)
    out=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
    pos=int(enc["attention_mask"][0].sum().item())-1
    hs=[out.hidden_states[L+1][0,pos].float().detach() for L in range(N_LAYERS)]
    del out,enc
    return hs

# ==================================================================================================
# 6. SEARCH GEOMETRY
# ==================================================================================================

print("\n[3/14] Building SEARCH bank...")

POS_BANK=[[] for _ in range(N_LAYERS)]
NEG_BANK=[[] for _ in range(N_LAYERS)]

for i,(ptext,ntext) in enumerate(SEARCH_PAIRS,1):
    hp,hn=capture(ptext),capture(ntext)

    for L in range(N_LAYERS):
        POS_BANK[L].append(hp[L])
        NEG_BANK[L].append(hn[L])

    print(f"       pair {i:02d}/{len(SEARCH_PAIRS)}")

POS_BANK=[torch.stack(x).float().contiguous() for x in POS_BANK]
NEG_BANK=[torch.stack(x).float().contiguous() for x in NEG_BANK]

# ==================================================================================================
# 7. LINEAGE BASIS
# ==================================================================================================

print("\n[4/14] Building Fisher / Bank lineage...")

FISHER=[]
BANK=[]
BANK_RES=[]
COS=[]
RESN=[]

for L in range(N_LAYERS):

    Hp,Hn=POS_BANK[L],NEG_BANK[L]

    mp,mn=Hp.mean(0),Hn.mean(0)
    md=mp-mn

    R=torch.cat([Hp-mp,Hn-mn],0).float()
    nu=max(int(R.shape[0])-2,1)

    _,S,Vh=torch.linalg.svd(R,full_matrices=False)

    eig=S.square()/float(nu)
    pe=eig[eig>FISHER_EPS]

    es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS)

    V=Vh.T
    pr=V.T@md

    f=md/lam+V@((1/(eig+lam)-1/lam)*pr)
    f=(f/f.norm().clamp_min(EPS)).float()

    dirs=normalize(Hp-Hn)

    sg=torch.where(
        (dirs@f)[:,None]<0,
        -torch.ones_like(dirs[:,:1]),
        torch.ones_like(dirs[:,:1])
    )

    dirs=dirs*sg

    m=dirs.mean(0)
    m=(m/m.norm().clamp_min(EPS)).float()

    c=torch.dot(f,m).clamp(-1.,1.)

    if c<0:
        m=-m
        c=-c

    rb=m-c*f
    nb=rb.norm()

    if nb<EPS:
        raise RuntimeError(f"Degenerate residual L{L}")

    rb=(rb/nb).float()

    FISHER.append(f)
    BANK.append(m)
    BANK_RES.append(rb)

    COS.append(float(c))
    RESN.append(float(nb))

    print(
        f"       L{L:02d} cos(F,B)={float(c):+.6f} "
        f"| residual={float(nb):.6f}"
    )

FISHER=torch.stack(FISHER).float().contiguous()
BANK=torch.stack(BANK).float().contiguous()
BANK_RES=torch.stack(BANK_RES).float().contiguous()

# ==================================================================================================
# 8. CANDIDATE LINEAGE
# ==================================================================================================

print("\n[5/14] Forging lineage candidates...")

BETA_GRID=[
    -1.00,-0.50,
     0.00,0.25,0.50,0.75,
     1.00,1.25,1.50,2.00
]

CANDIDATES={}
LINEAGE={}

for beta in BETA_GRID:

    z=[]

    for L in range(N_LAYERS):

        c=max(COS[L],1e-8)
        ratio=RESN[L]/c

        q=FISHER[L]+beta*ratio*BANK_RES[L]
        q=q/q.norm().clamp_min(EPS)

        z.append(q.float())

    name=f"BETA_{beta:+.2f}"
    vec=torch.stack(z).float().contiguous()

    cos_f=float(torch.sum(vec*FISHER,dim=1).mean().item())
    cos_b=float(torch.sum(vec*BANK,dim=1).mean().item())
    cos_r=float(torch.sum(vec*BANK_RES,dim=1).mean().item())

    CANDIDATES[name]=vec

    LINEAGE[name]={
        "method":name,
        "origin":"FISHER_PLUS_BANK_RESIDUAL",
        "beta":float(beta),
        "mean_cos_fisher":cos_f,
        "mean_cos_bank":cos_b,
        "mean_cos_bank_residual":cos_r
    }

# Exact aliases use one vector object.
CANDIDATES["FISHER"]=FISHER
CANDIDATES["BANK_MEAN"]=BANK
CANDIDATES["BANK_RESIDUAL"]=BANK_RES

LINEAGE["FISHER"]={
    "method":"FISHER",
    "origin":"SUBSPACE_FISHER",
    "beta":0.0,
    "mean_cos_fisher":1.0,
    "mean_cos_bank":float(np.mean(COS)),
    "mean_cos_bank_residual":0.0
}

LINEAGE["BANK_MEAN"]={
    "method":"BANK_MEAN",
    "origin":"MATCHED_PAIR_BANK_MEAN",
    "beta":1.0,
    "mean_cos_fisher":float(np.mean(COS)),
    "mean_cos_bank":1.0,
    "mean_cos_bank_residual":float(np.mean(RESN))
}

LINEAGE["BANK_RESIDUAL"]={
    "method":"BANK_RESIDUAL",
    "origin":"BANK_ORTHOGONAL_TO_FISHER",
    "beta":None,
    "mean_cos_fisher":0.0,
    "mean_cos_bank":float(np.mean(RESN)),
    "mean_cos_bank_residual":1.0
}

for name in CANDIDATES:
    x=LINEAGE[name]
    print(
        f"       {name:<16} "
        f"F={x['mean_cos_fisher']:+.5f} "
        f"B={x['mean_cos_bank']:+.5f} "
        f"R={x['mean_cos_bank_residual']:+.5f}"
    )

# ==================================================================================================
# 9. SEASC ENGINE
# ==================================================================================================

def blank_telemetry():
    return [{"requested":[],"realized":[]} for _ in range(N_LAYERS)]

def install_hooks(vectors,sign,dose_scale,telemetry):

    handles=[]

    for L in range(N_LAYERS):

        def make_hook(layer_idx):

            def hook(module,args,output):

                if isinstance(output,tuple):
                    old,rest=output[0],output[1:]
                else:
                    old,rest=output,None

                batch=old.shape[0]

                direction=vectors[layer_idx][None,:].expand(batch,-1)
                direction=(float(sign)*direction).float().contiguous()

                requested_value=IVME*ENVELOPE[layer_idx]*dose_scale

                dose=torch.full(
                    (batch,),
                    float(requested_value),
                    device=DEVICE,
                    dtype=torch.float32
                )

                old_last=old[:,-1,:].float()

                new=seasc_ext.seasc_batch(
                    old,direction,dose
                )

                new_last=new[:,-1,:].float()

                realized=(
                    (new_last-old_last).norm(dim=-1)/
                    old_last.norm(dim=-1).clamp_min(EPS)
                )

                telemetry[layer_idx]["requested"].extend(
                    dose.detach().cpu().tolist()
                )

                telemetry[layer_idx]["realized"].extend(
                    realized.detach().cpu().tolist()
                )

                return new if rest is None else (new,)+rest

            return hook

        handles.append(
            layers[L].register_forward_hook(make_hook(L))
        )

    return handles

def build_sequence(context,continuation):

    p=tokenizer(
        chat_text(context),
        return_tensors="pt",
        add_special_tokens=False
    )

    c=tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False
    )

    ids=torch.cat([p["input_ids"],c["input_ids"]],1).to(DEVICE)
    mask=torch.ones_like(ids,device=DEVICE)

    return ids,mask,p["input_ids"].shape[1]

@torch.inference_mode()
def continuation_logp(context,continuation,vectors=None,sign=0,scale=0.0):

    ids,mask,plen=build_sequence(context,continuation)

    telemetry=blank_telemetry()
    handles=[]

    if vectors is not None:
        handles=install_hooks(vectors,sign,scale,telemetry)

    try:
        out=model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True
        )
    finally:
        for h in handles:
            h.remove()

    logits=out.logits.float()
    target=ids[:,plen:]
    pred=logits[:,plen-1:-1,:]

    lp=F.log_softmax(pred,dim=-1)

    token_lp=lp.gather(
        -1,target.unsqueeze(-1)
    ).squeeze(-1)

    score=float(token_lp.mean().item())

    del out,logits,target,pred,lp,token_lp,ids,mask

    return score,telemetry

def run_arm(method,vectors,scale,items):

    cs=[]
    pm=[]
    nm=[]
    requested=[]
    realized=[]

    for item in items:

        pp,t1=continuation_logp(item["context"],item["pos"],vectors,+1,scale)
        pn,t2=continuation_logp(item["context"],item["neg"],vectors,+1,scale)
        np_,t3=continuation_logp(item["context"],item["pos"],vectors,-1,scale)
        nn,t4=continuation_logp(item["context"],item["neg"],vectors,-1,scale)

        p=pp-pn
        n=np_-nn
        c=.5*(p-n)

        pm.append(p)
        nm.append(n)
        cs.append(c)

        for tel in (t1,t2,t3,t4):
            for L in range(N_LAYERS):
                requested.extend(tel[L]["requested"])
                realized.extend(tel[L]["realized"])

    return {
        "method":method,
        "scale":scale,
        "plus_margin":float(np.mean(pm)),
        "minus_margin":float(np.mean(nm)),
        "Csym":float(np.mean(cs)),
        "Csym_sd":float(np.std(cs,ddof=1)) if len(cs)>1 else 0.0,
        "positive_items":int(np.sum(np.asarray(cs)>0)),
        "requested_mean":float(np.mean(requested)),
        "realized_mean":float(np.mean(realized)),
        "energy":float(np.sum(np.square(realized))),
        "item_csym":[float(x) for x in cs]
    }

# ==================================================================================================
# 10. COLLATERAL
# ==================================================================================================

@torch.inference_mode()
def neutral_metrics(text,vectors,scale):

    ids,mask,plen=build_sequence(text,NEUTRAL_CONT)

    out0=model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        return_dict=True
    )

    l0=out0.logits.float()[:,plen-1:-1,:]
    target=ids[:,plen:]
    lp0=F.log_softmax(l0,-1)

    nll0=-lp0.gather(
        -1,target.unsqueeze(-1)
    ).squeeze(-1).mean()

    tel=blank_telemetry()
    handles=install_hooks(vectors,+1,scale,tel)

    try:
        out1=model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True
        )
    finally:
        for h in handles:
            h.remove()

    l1=out1.logits.float()[:,plen-1:-1,:]
    lp1=F.log_softmax(l1,-1)

    nll1=-lp1.gather(
        -1,target.unsqueeze(-1)
    ).squeeze(-1).mean()

    kl=(lp0.exp()*(lp0-lp1)).sum(-1).mean()

    ans=float((nll1-nll0).item()),float(kl.item())

    del ids,mask,out0,out1,l0,l1,target,lp0,lp1

    return ans

def collateral_arm(method,vectors,scale,neutral_set):

    vals=[
        neutral_metrics(x,vectors,scale)
        for x in neutral_set
    ]

    return {
        "method":method,
        "scale":scale,
        "delta_nll":float(np.mean([x[0] for x in vals])),
        "kl":float(np.mean([x[1] for x in vals]))
    }

# ==================================================================================================
# 11. MULTI-FAMILY FORGE
# ==================================================================================================

print("\n[6/14] Multi-family causal forge...")

FAMILY_ROWS=[]

for name,v in CANDIDATES.items():

    print(f"\n       {name}")

    for family_name,items in FORGE_FAMILIES.items():

        r=run_arm(
            name,v,FORGE_SCALE,items
        )

        r["family"]=family_name
        FAMILY_ROWS.append(r)

        print(
            f"         {family_name:<24} "
            f"Csym={r['Csym']:+.6f} "
            f"| P+={r['positive_items']}/{len(items)}"
        )

# ==================================================================================================
# 12. LINEAGE-AWARE SELECTION
# ==================================================================================================

print("\n[7/14] Building family profiles...")

FORGE_COLLATERAL=[]
PROFILES=[]

for name,v in CANDIDATES.items():

    rows=[
        r for r in FAMILY_ROWS
        if r["method"]==name
    ]

    effects=np.asarray(
        [r["Csym"] for r in rows],
        dtype=np.float64
    )

    coll=collateral_arm(
        name,v,FORGE_SCALE,NEUTRAL_FORGE
    )

    FORGE_COLLATERAL.append(coll)

    family_mean=float(effects.mean())
    family_sd=float(effects.std(ddof=1))
    family_min=float(effects.min())
    family_max=float(effects.max())

    score=(
        family_mean
        - LAMBDA_FAMILY_SD*family_sd
        - LAMBDA_KL*max(coll["kl"],0.0)
        - LAMBDA_NLL*abs(coll["delta_nll"])
    )

    lin=LINEAGE[name]

    profile={
        "method":name,
        "origin":lin["origin"],
        "beta":lin["beta"],
        "mean_cos_fisher":lin["mean_cos_fisher"],
        "mean_cos_bank":lin["mean_cos_bank"],
        "mean_cos_bank_residual":lin["mean_cos_bank_residual"],
        "family_mean":family_mean,
        "family_sd":family_sd,
        "family_min":family_min,
        "family_max":family_max,
        "kl":coll["kl"],
        "delta_nll":coll["delta_nll"],
        "lineage_score":float(score)
    }

    for r in rows:
        profile[r["family"]]=r["Csym"]

    PROFILES.append(profile)

    print(
        f"  {name:<16} "
        f"mean={family_mean:+.6f} "
        f"| SD={family_sd:.6f} "
        f"| min={family_min:+.6f} "
        f"| KL={coll['kl']:.6f} "
        f"| SCORE={score:+.6f}"
    )

# References cannot win synthetic selection.
SYNTH_NAMES=[
    f"BETA_{beta:+.2f}"
    for beta in BETA_GRID
    if beta not in (0.0,1.0)
]

SYNTH_PROFILES=[
    x for x in PROFILES
    if x["method"] in SYNTH_NAMES
]

WINNER=max(
    SYNTH_PROFILES,
    key=lambda x:x["lineage_score"]
)

WINNER_NAME=WINNER["method"]
WINNER_VECTOR=CANDIDATES[WINNER_NAME].clone().contiguous()

print("\n       FROZEN WINNER :",WINNER_NAME)
print("       Origin        :",WINNER["origin"])
print("       Family mean   :",f"{WINNER['family_mean']:+.6f}")
print("       Family SD     :",f"{WINNER['family_sd']:.6f}")
print("       Lineage score :",f"{WINNER['lineage_score']:+.6f}")

# ==================================================================================================
# 13. LEAVE-ONE-FAMILY-OUT STABILITY
# ==================================================================================================

print("\n[8/14] Leave-one-family-out selection stability...")

LOFO=[]

family_names=list(FORGE_FAMILIES.keys())

for held_out in family_names:

    candidates=[]

    for name in SYNTH_NAMES:

        rows=[
            r for r in FAMILY_ROWS
            if r["method"]==name
            and r["family"]!=held_out
        ]

        eff=np.asarray(
            [r["Csym"] for r in rows],
            dtype=np.float64
        )

        coll=next(
            x for x in FORGE_COLLATERAL
            if x["method"]==name
        )

        score=(
            float(eff.mean())
            - LAMBDA_FAMILY_SD*float(eff.std(ddof=1))
            - LAMBDA_KL*max(coll["kl"],0.0)
            - LAMBDA_NLL*abs(coll["delta_nll"])
        )

        candidates.append((score,name))

    candidates.sort(reverse=True)

    selected_score,selected_name=candidates[0]

    held_row=next(
        r for r in FAMILY_ROWS
        if r["method"]==selected_name
        and r["family"]==held_out
    )

    row={
        "held_out_family":held_out,
        "selected_method":selected_name,
        "train_score":float(selected_score),
        "held_out_csym":held_row["Csym"]
    }

    LOFO.append(row)

    print(
        f"       hold={held_out:<24} "
        f"select={selected_name:<12} "
        f"| held Csym={held_row['Csym']:+.6f}"
    )

LOFO_WINNERS=[x["selected_method"] for x in LOFO]

# ==================================================================================================
# 14. FINAL — FIRST ACCESS AFTER FREEZE
# ==================================================================================================

print("\n[9/14] FINAL validation — first access...")

FINAL_ARMS={
    "FISHER":FISHER,
    "BANK_MEAN":BANK,
    "BANK_RESIDUAL":BANK_RES,
    "LINEAGE_SYNTHETIC":WINNER_VECTOR
}

FINAL_ROWS=[]

for scale in FINAL_SCALES:

    print(f"\nDOSE={scale:.2f}")

    for name,v in FINAL_ARMS.items():

        r=run_arm(
            name,v,scale,FINAL_ITEMS
        )

        FINAL_ROWS.append(r)

        print(
            f"  {name:<20} "
            f"Csym={r['Csym']:+.6f} "
            f"| P+={r['positive_items']:02d}/{len(FINAL_ITEMS)} "
            f"| real={100*r['realized_mean']:.4f}%"
        )

# ==================================================================================================
# 15. FINAL COLLATERAL
# ==================================================================================================

print("\n[10/14] FINAL collateral...")

FINAL_COLLATERAL=[]

for scale in FINAL_SCALES:

    for name,v in FINAL_ARMS.items():

        c=collateral_arm(
            name,v,scale,NEUTRAL_FINAL
        )

        FINAL_COLLATERAL.append(c)

        print(
            f"  {name:<20} "
            f"scale={scale:.2f} "
            f"| ΔNLL={c['delta_nll']:+.6f} "
            f"| KL={c['kl']:.6f}"
        )

# ==================================================================================================
# 16. BOOTSTRAP
# ==================================================================================================

def get_final(name,scale):
    return next(
        r for r in FINAL_ROWS
        if r["method"]==name and r["scale"]==scale
    )

def get_coll(name,scale):
    return next(
        r for r in FINAL_COLLATERAL
        if r["method"]==name and r["scale"]==scale
    )

def paired_bootstrap(a,b,n=20000,seed=164):

    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    d=a-b

    rng=np.random.default_rng(seed)
    means=np.empty(n,dtype=np.float64)

    for i in range(n):
        idx=rng.integers(0,len(d),len(d))
        means[i]=d[idx].mean()

    return (
        float(d.mean()),
        float(np.quantile(means,.025)),
        float(np.quantile(means,.975)),
        float((np.sum(means<=0)+1)/(n+1))
    )

print("\n[11/14] FINAL contrasts...")

CONTRASTS=[]

for scale in FINAL_SCALES:

    s=get_final("LINEAGE_SYNTHETIC",scale)
    f=get_final("FISHER",scale)
    b=get_final("BANK_MEAN",scale)

    sf=paired_bootstrap(
        s["item_csym"],f["item_csym"],
        seed=SEED+int(scale*1000)+1
    )

    sb=paired_bootstrap(
        s["item_csym"],b["item_csym"],
        seed=SEED+int(scale*1000)+2
    )

    sc=get_coll("LINEAGE_SYNTHETIC",scale)
    fc=get_coll("FISHER",scale)
    bc=get_coll("BANK_MEAN",scale)

    row={
        "scale":scale,
        "synthetic_csym":s["Csym"],
        "fisher_csym":f["Csym"],
        "bank_csym":b["Csym"],

        "synthetic_minus_fisher":sf[0],
        "sf_ci95_low":sf[1],
        "sf_ci95_high":sf[2],
        "sf_p_one_sided":sf[3],

        "synthetic_minus_bank":sb[0],
        "sb_ci95_low":sb[1],
        "sb_ci95_high":sb[2],
        "sb_p_one_sided":sb[3],

        "synthetic_kl":sc["kl"],
        "fisher_kl":fc["kl"],
        "bank_kl":bc["kl"]
    }

    CONTRASTS.append(row)

    print(
        f"  scale={scale:.2f} | "
        f"SYN-F={sf[0]:+.6f} "
        f"[{sf[1]:+.6f},{sf[2]:+.6f}] p={sf[3]:.6f} | "
        f"SYN-B={sb[0]:+.6f} "
        f"[{sb[1]:+.6f},{sb[2]:+.6f}] p={sb[3]:.6f}"
    )

# ==================================================================================================
# 17. DECISION
# ==================================================================================================

REF_SCALE=.50

ref=next(
    x for x in CONTRASTS
    if x["scale"]==REF_SCALE
)

lofo_unique=len(set(LOFO_WINNERS))
lofo_mode=max(
    set(LOFO_WINNERS),
    key=LOFO_WINNERS.count
)

lofo_mode_count=LOFO_WINNERS.count(lofo_mode)

if (
    ref["synthetic_minus_fisher"]>0
    and ref["sf_ci95_low"]>0
    and ref["synthetic_minus_bank"]>0
    and ref["sb_ci95_low"]>0
    and lofo_mode_count>=3
):
    DECISION="LINEAGE_AWARE_SYNTHETIC_VECTOR_SUPPORTED"

elif (
    ref["synthetic_minus_fisher"]>0
    or ref["synthetic_minus_bank"]>0
):
    DECISION="PARTIAL_LINEAGE_SIGNAL"

elif lofo_mode_count>=3:
    DECISION="LINEAGE_STABLE_BUT_NO_FINAL_ADVANTAGE"

else:
    DECISION="NO_GENERALIZABLE_LINEAGE_ADVANTAGE"

# ==================================================================================================
# 18. SAVE
# ==================================================================================================

print("\n[12/14] Saving tables...")

def safe(rows):
    return [
        {
            k:v for k,v in r.items()
            if not isinstance(v,(list,dict,tuple))
        }
        for r in rows
    ]

def save_csv(name,rows):

    rows=safe(rows)

    if not rows:
        return

    keys=sorted(
        set().union(
            *[set(r.keys()) for r in rows]
        )
    )

    with open(
        ROOT/name,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        w=csv.DictWriter(
            f,
            fieldnames=keys
        )

        w.writeheader()
        w.writerows(rows)

GEOMETRY=[
    {
        "layer":L,
        "fisher_bank_cos":COS[L],
        "bank_residual_norm":RESN[L]
    }
    for L in range(N_LAYERS)
]

save_csv("SEARCH_GEOMETRY.csv",GEOMETRY)
save_csv("LINEAGE.csv",list(LINEAGE.values()))
save_csv("FAMILY_CAUSAL.csv",FAMILY_ROWS)
save_csv("FAMILY_PROFILES.csv",PROFILES)
save_csv("FORGE_COLLATERAL.csv",FORGE_COLLATERAL)
save_csv("LOFO.csv",LOFO)
save_csv("FINAL_CAUSAL.csv",FINAL_ROWS)
save_csv("FINAL_COLLATERAL.csv",FINAL_COLLATERAL)
save_csv("FINAL_CONTRASTS.csv",CONTRASTS)

print("[13/14] Saving vectors...")

np.savez_compressed(
    ROOT/"TEST164_VECTOR_LINEAGE_FOUNDRY.npz",

    fisher=FISHER.detach().cpu().numpy(),
    bank=BANK.detach().cpu().numpy(),
    bank_residual=BANK_RES.detach().cpu().numpy(),
    lineage_synthetic=WINNER_VECTOR.detach().cpu().numpy(),

    fisher_bank_cos=np.asarray(COS,dtype=np.float32),
    bank_residual_norm=np.asarray(RESN,dtype=np.float32),
    envelope=np.asarray(ENVELOPE,dtype=np.float32)
)

REPORT={
    "test":"TEST 164",
    "title":"VEKTÖR SOY AĞACI DÖKÜMHANESİ",
    "model":MODEL_ID,
    "gpu":torch.cuda.get_device_name(0),

    "forge_scale":FORGE_SCALE,
    "final_scales":FINAL_SCALES,

    "selection_contract":{
        "lambda_family_sd":LAMBDA_FAMILY_SD,
        "lambda_kl":LAMBDA_KL,
        "lambda_nll":LAMBDA_NLL
    },

    "winner":WINNER_NAME,
    "winner_profile":WINNER,

    "lofo_winners":LOFO_WINNERS,
    "lofo_mode":lofo_mode,
    "lofo_mode_count":lofo_mode_count,
    "lofo_unique_winners":lofo_unique,

    "decision":DECISION,

    "geometry":GEOMETRY,
    "lineage":LINEAGE,
    "family_profiles":PROFILES,
    "lofo":LOFO,
    "final_results":FINAL_ROWS,
    "final_collateral":FINAL_COLLATERAL,
    "final_contrasts":CONTRASTS,

    "interpretation_contract":[
        "SEASC frozen-norm physical-dose motor is unchanged.",
        "Static normalized DRA envelope is unchanged.",
        "SEARCH constructs Fisher, Bank-Mean and Bank-specific residual lineage.",
        "Candidate lineage is explicit and recorded before causal selection.",
        "Four FORGE prompt families are scored separately.",
        "Selection rewards cross-family mean causal effect and penalizes family instability and collateral.",
        "Leave-one-family-out selection measures lineage-selection stability.",
        "Reference Fisher, Bank-Mean and Bank residual arms cannot win synthetic selection.",
        "The winning synthetic vector is frozen before FINAL is accessed.",
        "FINAL is not used for candidate construction, tuning or selection.",
        "Equal requested SEASC physical dose is used for all actuator comparisons.",
        "A stable lineage is not sufficient evidence of superiority unless it survives FINAL.",
        "This experiment concerns the ABOVE/BELOW concept family only."
    ]
}

with open(
    ROOT/"REPORT.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        REPORT,
        f,
        indent=2,
        ensure_ascii=False
    )

# ==================================================================================================
# 19. FINAL SUMMARY
# ==================================================================================================

print("\n[14/14] COMPLETE")
print("="*150)
print("TEST 164 — FINAL SUMMARY")
print("="*150)

print("Frozen lineage vector    :",WINNER_NAME)
print("Origin                   :",WINNER["origin"])
print("Family mean              :",f"{WINNER['family_mean']:+.6f}")
print("Family SD                :",f"{WINNER['family_sd']:.6f}")
print("Family minimum           :",f"{WINNER['family_min']:+.6f}")
print("Lineage score            :",f"{WINNER['lineage_score']:+.6f}")

print("\nLOFO")
for x in LOFO:
    print(
        f"  {x['held_out_family']:<24} -> "
        f"{x['selected_method']:<12} "
        f"| held={x['held_out_csym']:+.6f}"
    )

print(
    f"\nLOFO mode                : {lofo_mode} "
    f"({lofo_mode_count}/{len(LOFO_WINNERS)})"
)

for scale in FINAL_SCALES:

    print(f"\nDOSE SCALE = {scale:.2f}")

    for name in [
        "FISHER",
        "BANK_MEAN",
        "BANK_RESIDUAL",
        "LINEAGE_SYNTHETIC"
    ]:

        r=get_final(name,scale)
        c=get_coll(name,scale)

        print(
            f"  {name:<20} "
            f"Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/{len(FINAL_ITEMS)} | "
            f"KL={c['kl']:.6f} | "
            f"real={100*r['realized_mean']:.4f}%"
        )

print("\nFINAL CONTRASTS")

for r in CONTRASTS:

    print(
        f"  scale={r['scale']:.2f} | "
        f"SYN-F={r['synthetic_minus_fisher']:+.6f} "
        f"[{r['sf_ci95_low']:+.6f},{r['sf_ci95_high']:+.6f}] | "
        f"SYN-B={r['synthetic_minus_bank']:+.6f} "
        f"[{r['sb_ci95_low']:+.6f},{r['sb_ci95_high']:+.6f}]"
    )

print("\nDECISION :",DECISION)
print("Saved    :",ROOT)

print("="*150)
print("TEST 164 COMPLETE")
print("="*150)
