# ==================================================================================================
# TEST 165 — NEGATIVE-RESIDUAL CAUSAL EFFICIENCY REPLICATION
# AKBASCORE 3.2 / SEASC
#
# TEST164 hypothesis:
#   Fisher - Bank-specific residual may define a high-efficiency causal region.
#
# Design:
#   SEARCH -> fixed negative-beta neighborhood
#   NEW multi-family FORGE -> efficiency-aware selection
#   freeze winner
#   NEW untouched FINAL -> multi-dose replication
#
# No TEST163/164 FORGE or FINAL prompts are reused.
# ==================================================================================================

import os,sys,math,json,csv,random,shutil,subprocess
from pathlib import Path

print("="*150)
print("TEST 165 — NEGATIVE-RESIDUAL CAUSAL EFFICIENCY REPLICATION")
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
SEED=165

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

# TEST164 generated the hypothesis.
# TEST165 searches only the pre-declared negative-residual neighborhood.
BETA_GRID=[
    -1.50,
    -1.25,
    -1.00,
    -0.75,
    -0.50
]

# Selection dose fixed before results.
FORGE_SCALE=.50

# Independent dose-response validation.
FINAL_SCALES=[
    .125,
    .25,
    .50,
    .75,
    1.00
]

# Efficiency-aware FORGE contract.
LAMBDA_FAMILY_SD=.35
LAMBDA_KL=.05
LAMBDA_NLL=.10

SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."

ROOT=Path("/content/TEST165_NEGATIVE_RESIDUAL_REPLICATION")
ROOT.mkdir(parents=True,exist_ok=True)

print("GPU            :",torch.cuda.get_device_name(0))
print("Model          :",MODEL_ID)
print("Steered layers : L0-L19")
print("Beta region    :",BETA_GRID)
print("Forge scale    :",FORGE_SCALE)
print("Final scales   :",FINAL_SCALES)

# ==================================================================================================
# 1. CUDA — FROZEN-NORM SEASC
# ==================================================================================================

BUILD_DIR="/tmp/akbascore_test165"
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
    m.def("seasc_batch",&seasc_batch,"TEST165 SEASC");
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

    auto act=actuator
        .to(hidden.device(),torch::kFloat32)
        .contiguous();

    auto d=dose
        .to(hidden.device(),torch::kFloat32)
        .contiguous();

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
        "test165_seasc_batch",
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

print("\n[1/15] Compiling SEASC kernel...")

seasc_ext=load_inline(
    name="akbascore_test165_cuda",
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

print("\n[2/15] Loading frozen model...")

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

    kb=(
        ZIRVE*
        math.exp(-SONUM*t)*
        (1.0+SONUM*t)
        + TABAN
    )

    return kb/(ZIRVE+TABAN)

ENVELOPE=[
    dra_envelope(L)
    for L in range(N_LAYERS)
]

print("       hidden :",HIDDEN)
print("       layers :",len(layers))
print("       dtype  :",next(model.parameters()).dtype)

# ==================================================================================================
# 3. SEARCH — SAME GEOMETRIC ANCHOR AS TEST163/164
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
# 4. NEW FORGE FAMILIES
# No TEST163/164 FORGE or FINAL items reused.
# ==================================================================================================

FORGE_FAMILIES={

"FAMILY_E_VERTICAL_RELATION":[

{"context":"A brass bell and a wooden stool are present. Their vertical relation is not provided.",
 "pos":" The brass bell is above the wooden stool.",
 "neg":" The brass bell is below the wooden stool."},

{"context":"A paper lantern and a stone basin are present. Their vertical relation is not provided.",
 "pos":" The paper lantern is above the stone basin.",
 "neg":" The paper lantern is below the stone basin."},

{"context":"A small mirror and a laundry basket are present. Their vertical relation is not provided.",
 "pos":" The small mirror is above the laundry basket.",
 "neg":" The small mirror is below the laundry basket."},

{"context":"A warning light and a metal cart are present. Their vertical relation is not provided.",
 "pos":" The warning light is above the metal cart.",
 "neg":" The warning light is below the metal cart."}
],

"FAMILY_F_RELATIVE_HEIGHT":[

{"context":"The relative height of a weather vane and a garden shed is unspecified.",
 "pos":" The weather vane is higher than the garden shed.",
 "neg":" The weather vane is lower than the garden shed."},

{"context":"The relative height of a gull and a harbor wall is unspecified.",
 "pos":" The gull is higher than the harbor wall.",
 "neg":" The gull is lower than the harbor wall."},

{"context":"The relative height of a spotlight and a theatre seat is unspecified.",
 "pos":" The spotlight is higher than the theatre seat.",
 "neg":" The spotlight is lower than the theatre seat."},

{"context":"The relative height of a hanging basket and a garden bench is unspecified.",
 "pos":" The hanging basket is higher than the garden bench.",
 "neg":" The hanging basket is lower than the garden bench."}
],

"FAMILY_G_UPPER_LOWER":[

{"context":"A compass marker and a map legend occupy two possible vertical positions, but the ordering is unknown.",
 "pos":" The compass marker occupies the upper position relative to the map legend.",
 "neg":" The compass marker occupies the lower position relative to the map legend."},

{"context":"A theatre sign and a ticket booth occupy two possible vertical positions, but the ordering is unknown.",
 "pos":" The theatre sign occupies the upper position relative to the ticket booth.",
 "neg":" The theatre sign occupies the lower position relative to the ticket booth."},

{"context":"A ventilation grille and a storage locker occupy two possible vertical positions, but the ordering is unknown.",
 "pos":" The ventilation grille occupies the upper position relative to the storage locker.",
 "neg":" The ventilation grille occupies the lower position relative to the storage locker."},

{"context":"A hanging plant and a floor mat occupy two possible vertical positions, but the ordering is unknown.",
 "pos":" The hanging plant occupies the upper position relative to the floor mat.",
 "neg":" The hanging plant occupies the lower position relative to the floor mat."}
],

"FAMILY_H_SPATIAL_AXIS":[

{"context":"Along the vertical spatial axis, no ordering is given for a beacon and a pier.",
 "pos":" Along that axis, the beacon lies above the pier.",
 "neg":" Along that axis, the beacon lies below the pier."},

{"context":"Along the vertical spatial axis, no ordering is given for a cable car and a station.",
 "pos":" Along that axis, the cable car lies above the station.",
 "neg":" Along that axis, the cable car lies below the station."},

{"context":"Along the vertical spatial axis, no ordering is given for a wind sock and a hangar.",
 "pos":" Along that axis, the wind sock lies above the hangar.",
 "neg":" Along that axis, the wind sock lies below the hangar."},

{"context":"Along the vertical spatial axis, no ordering is given for a balcony and a courtyard.",
 "pos":" Along that axis, the balcony lies above the courtyard.",
 "neg":" Along that axis, the balcony lies below the courtyard."}
]
}

# ==================================================================================================
# 5. NEW UNTOUCHED FINAL
# ==================================================================================================

FINAL_ITEMS=[

{"context":"A rescue helicopter and a lighthouse are visible, with no vertical ordering specified.",
 "pos":" The rescue helicopter is above the lighthouse.",
 "neg":" The rescue helicopter is below the lighthouse."},

{"context":"A hanging clock and a reception desk are visible, with no vertical ordering specified.",
 "pos":" The hanging clock is above the reception desk.",
 "neg":" The hanging clock is below the reception desk."},

{"context":"A falcon and a canyon rim are visible, with no vertical ordering specified.",
 "pos":" The falcon is above the canyon rim.",
 "neg":" The falcon is below the canyon rim."},

{"context":"A suspended camera and a factory conveyor are visible, with no vertical ordering specified.",
 "pos":" The suspended camera is above the factory conveyor.",
 "neg":" The suspended camera is below the factory conveyor."},

{"context":"A navigation light and a ship deck are visible, with no vertical ordering specified.",
 "pos":" The navigation light is above the ship deck.",
 "neg":" The navigation light is below the ship deck."},

{"context":"A hot-air balloon and a vineyard are visible, with no vertical ordering specified.",
 "pos":" The hot-air balloon is above the vineyard.",
 "neg":" The hot-air balloon is below the vineyard."},

{"context":"A ceiling speaker and a museum display case are visible, with no vertical ordering specified.",
 "pos":" The ceiling speaker is above the museum display case.",
 "neg":" The ceiling speaker is below the museum display case."},

{"context":"A suspended sensor and a workshop table are visible, with no vertical ordering specified.",
 "pos":" The suspended sensor is above the workshop table.",
 "neg":" The suspended sensor is below the workshop table."},

{"context":"A weather radar and an airport terminal are visible, with no vertical ordering specified.",
 "pos":" The weather radar is above the airport terminal.",
 "neg":" The weather radar is below the airport terminal."},

{"context":"A hanging banner and a registration counter are visible, with no vertical ordering specified.",
 "pos":" The hanging banner is above the registration counter.",
 "neg":" The hanging banner is below the registration counter."},

{"context":"A glider and a railway station are visible, with no vertical ordering specified.",
 "pos":" The glider is above the railway station.",
 "neg":" The glider is below the railway station."},

{"context":"A suspended microphone and an orchestra pit are visible, with no vertical ordering specified.",
 "pos":" The suspended microphone is above the orchestra pit.",
 "neg":" The suspended microphone is below the orchestra pit."},

{"context":"A signal lamp and a loading platform are visible, with no vertical ordering specified.",
 "pos":" The signal lamp is above the loading platform.",
 "neg":" The signal lamp is below the loading platform."},

{"context":"A mountain cloud and a valley floor are visible, with no vertical ordering specified.",
 "pos":" The mountain cloud is above the valley floor.",
 "neg":" The mountain cloud is below the valley floor."},

{"context":"A hanging sculpture and a gallery bench are visible, with no vertical ordering specified.",
 "pos":" The hanging sculpture is above the gallery bench.",
 "neg":" The hanging sculpture is below the gallery bench."},

{"context":"An observation drone and a riverbank are visible, with no vertical ordering specified.",
 "pos":" The observation drone is above the riverbank.",
 "neg":" The observation drone is below the riverbank."}
]

NEUTRAL_FORGE=[
"A ceramic bowl rests beside a folded cloth.",
"A pencil lies next to a closed calendar.",
"A metal mug sits beside a paper envelope.",
"A wooden brush rests near a plain container."
]

NEUTRAL_FINAL=[
"A notebook lies beside a calculator.",
"A glass jar stands beside a folded newspaper.",
"A plastic tray rests beside a pair of gloves.",
"A ceramic plate sits beside a napkin.",
"A closed suitcase stands near a chair.",
"A small box rests beside a telephone."
]

NEUTRAL_CONT=" The scene contains ordinary stationary objects and no additional relation is specified."

# ==================================================================================================
# 6. HELPERS
# ==================================================================================================

def chat_text(x):
    return tokenizer.apply_chat_template(
        [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":x}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

def normalize(v):
    return v/v.norm(
        dim=-1,
        keepdim=True
    ).clamp_min(EPS)

@torch.inference_mode()
def capture(text):

    enc=tokenizer(
        chat_text(text),
        return_tensors="pt"
    ).to(DEVICE)

    out=model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )

    pos=int(
        enc["attention_mask"][0].sum().item()
    )-1

    hs=[
        out.hidden_states[L+1][0,pos]
        .float()
        .detach()
        for L in range(N_LAYERS)
    ]

    del out,enc

    return hs

# ==================================================================================================
# 7. SEARCH BANK
# ==================================================================================================

print("\n[3/15] Building SEARCH bank...")

POS_BANK=[
    []
    for _ in range(N_LAYERS)
]

NEG_BANK=[
    []
    for _ in range(N_LAYERS)
]

for i,(ptext,ntext) in enumerate(
    SEARCH_PAIRS,
    1
):

    hp=capture(ptext)
    hn=capture(ntext)

    for L in range(N_LAYERS):
        POS_BANK[L].append(hp[L])
        NEG_BANK[L].append(hn[L])

    print(
        f"       pair {i:02d}/{len(SEARCH_PAIRS)}"
    )

POS_BANK=[
    torch.stack(x)
    .float()
    .contiguous()
    for x in POS_BANK
]

NEG_BANK=[
    torch.stack(x)
    .float()
    .contiguous()
    for x in NEG_BANK
]

# ==================================================================================================
# 8. FISHER / BANK / RESIDUAL
# ==================================================================================================

print("\n[4/15] Reconstructing lineage basis...")

FISHER=[]
BANK=[]
BANK_RES=[]

COS=[]
RESN=[]

for L in range(N_LAYERS):

    Hp=POS_BANK[L]
    Hn=NEG_BANK[L]

    mp=Hp.mean(0)
    mn=Hn.mean(0)

    md=mp-mn

    R=torch.cat(
        [Hp-mp,Hn-mn],
        0
    ).float()

    nu=max(
        int(R.shape[0])-2,
        1
    )

    _,S,Vh=torch.linalg.svd(
        R,
        full_matrices=False
    )

    eig=S.square()/float(nu)

    pe=eig[
        eig>FISHER_EPS
    ]

    es=(
        torch.median(pe)
        if pe.numel()
        else torch.tensor(
            1.,
            device=DEVICE
        )
    )

    lam=(
        FISHER_ALPHA*es
    ).clamp_min(FISHER_EPS)

    V=Vh.T
    pr=V.T@md

    f=(
        md/lam
        + V@(
            (1/(eig+lam)-1/lam)*pr
        )
    )

    f=(
        f/
        f.norm().clamp_min(EPS)
    ).float()

    dirs=normalize(
        Hp-Hn
    )

    sg=torch.where(
        (dirs@f)[:,None]<0,
        -torch.ones_like(
            dirs[:,:1]
        ),
        torch.ones_like(
            dirs[:,:1]
        )
    )

    dirs=dirs*sg

    b=dirs.mean(0)

    b=(
        b/
        b.norm().clamp_min(EPS)
    ).float()

    c=torch.dot(
        f,b
    ).clamp(-1.,1.)

    if c<0:
        b=-b
        c=-c

    residual=b-c*f
    residual_norm=residual.norm()

    if residual_norm<EPS:
        raise RuntimeError(
            f"Degenerate residual L{L}"
        )

    residual=(
        residual/
        residual_norm
    ).float()

    FISHER.append(f)
    BANK.append(b)
    BANK_RES.append(residual)

    COS.append(float(c))
    RESN.append(
        float(residual_norm)
    )

    print(
        f"       L{L:02d} "
        f"cos(F,B)={float(c):+.6f} | "
        f"residual={float(residual_norm):.6f}"
    )

FISHER=torch.stack(
    FISHER
).float().contiguous()

BANK=torch.stack(
    BANK
).float().contiguous()

BANK_RES=torch.stack(
    BANK_RES
).float().contiguous()

print(
    "\n       Mean cos(F,B) :",
    f"{np.mean(COS):+.6f}"
)

print(
    "       Mean residual :",
    f"{np.mean(RESN):.6f}"
)

# ==================================================================================================
# 9. PRE-DECLARED NEGATIVE-RESIDUAL CANDIDATES
# ==================================================================================================

print("\n[5/15] Forging negative-residual neighborhood...")

CANDIDATES={}
LINEAGE=[]

for beta in BETA_GRID:

    vec=[]

    for L in range(N_LAYERS):

        c=max(
            COS[L],
            1e-8
        )

        ratio=(
            RESN[L]/c
        )

        q=(
            FISHER[L]
            + beta*
            ratio*
            BANK_RES[L]
        )

        q=(
            q/
            q.norm().clamp_min(EPS)
        )

        vec.append(
            q.float()
        )

    vec=torch.stack(
        vec
    ).float().contiguous()

    name=f"BETA_{beta:+.2f}"

    CANDIDATES[name]=vec

    cf=float(
        torch.sum(
            vec*FISHER,
            dim=1
        ).mean().item()
    )

    cb=float(
        torch.sum(
            vec*BANK,
            dim=1
        ).mean().item()
    )

    cr=float(
        torch.sum(
            vec*BANK_RES,
            dim=1
        ).mean().item()
    )

    LINEAGE.append({
        "method":name,
        "beta":beta,
        "origin":"FISHER_PLUS_SIGNED_BANK_RESIDUAL",
        "cos_fisher":cf,
        "cos_bank":cb,
        "cos_residual":cr
    })

    print(
        f"       {name:<14} "
        f"F={cf:+.6f} "
        f"B={cb:+.6f} "
        f"R={cr:+.6f}"
    )

# Reference arms.
CANDIDATES["FISHER"]=FISHER
CANDIDATES["BANK_MEAN"]=BANK
CANDIDATES["BANK_RESIDUAL"]=BANK_RES

# ==================================================================================================
# 10. SEASC HOOK ENGINE
# ==================================================================================================

def blank_telemetry():

    return [
        {
            "requested":[],
            "realized":[]
        }
        for _ in range(N_LAYERS)
    ]

def install_hooks(
    vectors,
    sign,
    dose_scale,
    telemetry
):

    handles=[]

    for L in range(N_LAYERS):

        def make_hook(layer_idx):

            def hook(
                module,
                args,
                output
            ):

                if isinstance(
                    output,
                    tuple
                ):

                    old=output[0]
                    rest=output[1:]

                else:

                    old=output
                    rest=None

                batch=old.shape[0]

                direction=(
                    vectors[layer_idx]
                    [None,:]
                    .expand(batch,-1)
                )

                direction=(
                    float(sign)*
                    direction
                ).float().contiguous()

                requested_value=(
                    IVME*
                    ENVELOPE[layer_idx]*
                    dose_scale
                )

                dose=torch.full(
                    (batch,),
                    float(requested_value),
                    device=DEVICE,
                    dtype=torch.float32
                )

                old_last=(
                    old[:,-1,:]
                    .float()
                )

                new=seasc_ext.seasc_batch(
                    old,
                    direction,
                    dose
                )

                new_last=(
                    new[:,-1,:]
                    .float()
                )

                realized=(
                    (new_last-old_last)
                    .norm(dim=-1)
                    /
                    old_last
                    .norm(dim=-1)
                    .clamp_min(EPS)
                )

                telemetry[layer_idx][
                    "requested"
                ].extend(
                    dose
                    .detach()
                    .cpu()
                    .tolist()
                )

                telemetry[layer_idx][
                    "realized"
                ].extend(
                    realized
                    .detach()
                    .cpu()
                    .tolist()
                )

                if rest is None:
                    return new

                return (
                    new,
                )+rest

            return hook

        handles.append(
            layers[L]
            .register_forward_hook(
                make_hook(L)
            )
        )

    return handles

# ==================================================================================================
# 11. CONTINUATION SCORING
# ==================================================================================================

def build_sequence(
    context,
    continuation
):

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

    ids=torch.cat(
        [
            p["input_ids"],
            c["input_ids"]
        ],
        1
    ).to(DEVICE)

    mask=torch.ones_like(
        ids,
        device=DEVICE
    )

    return (
        ids,
        mask,
        p["input_ids"].shape[1]
    )

@torch.inference_mode()
def continuation_logp(
    context,
    continuation,
    vectors=None,
    sign=0,
    scale=0.0
):

    ids,mask,plen=build_sequence(
        context,
        continuation
    )

    telemetry=blank_telemetry()
    handles=[]

    if vectors is not None:

        handles=install_hooks(
            vectors,
            sign,
            scale,
            telemetry
        )

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

    logits=(
        out.logits
        .float()
    )

    target=ids[:,plen:]
    pred=logits[
        :,
        plen-1:-1,
        :
    ]

    lp=F.log_softmax(
        pred,
        dim=-1
    )

    token_lp=lp.gather(
        -1,
        target.unsqueeze(-1)
    ).squeeze(-1)

    score=float(
        token_lp.mean().item()
    )

    del (
        out,
        logits,
        target,
        pred,
        lp,
        token_lp,
        ids,
        mask
    )

    return score,telemetry

# ==================================================================================================
# 12. CAUSAL ARM
# ==================================================================================================

def run_arm(
    method,
    vectors,
    scale,
    items
):

    cs=[]
    plus_margins=[]
    minus_margins=[]

    requested=[]
    realized=[]

    for item in items:

        pp,t1=continuation_logp(
            item["context"],
            item["pos"],
            vectors,
            +1,
            scale
        )

        pn,t2=continuation_logp(
            item["context"],
            item["neg"],
            vectors,
            +1,
            scale
        )

        np_,t3=continuation_logp(
            item["context"],
            item["pos"],
            vectors,
            -1,
            scale
        )

        nn,t4=continuation_logp(
            item["context"],
            item["neg"],
            vectors,
            -1,
            scale
        )

        plus_margin=pp-pn
        minus_margin=np_-nn

        csym=.5*(
            plus_margin-
            minus_margin
        )

        plus_margins.append(
            plus_margin
        )

        minus_margins.append(
            minus_margin
        )

        cs.append(
            csym
        )

        for tel in (
            t1,t2,t3,t4
        ):

            for L in range(
                N_LAYERS
            ):

                requested.extend(
                    tel[L][
                        "requested"
                    ]
                )

                realized.extend(
                    tel[L][
                        "realized"
                    ]
                )

    return {
        "method":method,
        "scale":scale,

        "plus_margin":float(
            np.mean(
                plus_margins
            )
        ),

        "minus_margin":float(
            np.mean(
                minus_margins
            )
        ),

        "Csym":float(
            np.mean(cs)
        ),

        "Csym_sd":float(
            np.std(
                cs,
                ddof=1
            )
        ),

        "positive_items":int(
            np.sum(
                np.asarray(cs)>0
            )
        ),

        "requested_mean":float(
            np.mean(
                requested
            )
        ),

        "realized_mean":float(
            np.mean(
                realized
            )
        ),

        "energy":float(
            np.sum(
                np.square(
                    realized
                )
            )
        ),

        "item_csym":[
            float(x)
            for x in cs
        ]
    }

# ==================================================================================================
# 13. COLLATERAL
# ==================================================================================================

@torch.inference_mode()
def neutral_metrics(
    text,
    vectors,
    scale
):

    ids,mask,plen=build_sequence(
        text,
        NEUTRAL_CONT
    )

    out0=model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        return_dict=True
    )

    l0=out0.logits[
        :,
        plen-1:-1,
        :
    ].float()

    target=ids[:,plen:]

    lp0=F.log_softmax(
        l0,
        -1
    )

    nll0=-lp0.gather(
        -1,
        target.unsqueeze(-1)
    ).squeeze(-1).mean()

    tel=blank_telemetry()

    handles=install_hooks(
        vectors,
        +1,
        scale,
        tel
    )

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

    l1=out1.logits[
        :,
        plen-1:-1,
        :
    ].float()

    lp1=F.log_softmax(
        l1,
        -1
    )

    nll1=-lp1.gather(
        -1,
        target.unsqueeze(-1)
    ).squeeze(-1).mean()

    kl=(
        lp0.exp()*
        (lp0-lp1)
    ).sum(-1).mean()

    ans=(
        float(
            (nll1-nll0)
            .item()
        ),
        float(
            kl.item()
        )
    )

    del (
        ids,mask,
        out0,out1,
        l0,l1,
        target,
        lp0,lp1
    )

    return ans

def collateral_arm(
    method,
    vectors,
    scale,
    neutral_set
):

    vals=[
        neutral_metrics(
            x,
            vectors,
            scale
        )
        for x in neutral_set
    ]

    return {
        "method":method,
        "scale":scale,

        "delta_nll":float(
            np.mean(
                [x[0] for x in vals]
            )
        ),

        "kl":float(
            np.mean(
                [x[1] for x in vals]
            )
        )
    }

# ==================================================================================================
# 14. NEW FORGE
# ==================================================================================================

print("\n[6/15] NEW multi-family FORGE...")

FORGE_METHODS=[
    f"BETA_{b:+.2f}"
    for b in BETA_GRID
]

# References measured, never eligible to win.
REFERENCE_METHODS=[
    "FISHER",
    "BANK_MEAN",
    "BANK_RESIDUAL"
]

ALL_FORGE_METHODS=(
    FORGE_METHODS+
    REFERENCE_METHODS
)

FAMILY_ROWS=[]

for name in ALL_FORGE_METHODS:

    v=CANDIDATES[name]

    print(
        f"\n       {name}"
    )

    for family_name,items in FORGE_FAMILIES.items():

        r=run_arm(
            name,
            v,
            FORGE_SCALE,
            items
        )

        r["family"]=family_name

        FAMILY_ROWS.append(
            r
        )

        print(
            f"         {family_name:<28} "
            f"Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']}/{len(items)}"
        )

# ==================================================================================================
# 15. COLLATERAL + FAMILY PROFILE
# ==================================================================================================

print("\n[7/15] Building efficiency profiles...")

FORGE_COLLATERAL=[]
PROFILES=[]

for name in ALL_FORGE_METHODS:

    v=CANDIDATES[name]

    rows=[
        r
        for r in FAMILY_ROWS
        if r["method"]==name
    ]

    effects=np.asarray(
        [
            r["Csym"]
            for r in rows
        ],
        dtype=np.float64
    )

    coll=collateral_arm(
        name,
        v,
        FORGE_SCALE,
        NEUTRAL_FORGE
    )

    FORGE_COLLATERAL.append(
        coll
    )

    family_mean=float(
        effects.mean()
    )

    family_sd=float(
        effects.std(
            ddof=1
        )
    )

    family_min=float(
        effects.min()
    )

    family_max=float(
        effects.max()
    )

    selection_score=(
        family_mean
        - LAMBDA_FAMILY_SD*
          family_sd
        - LAMBDA_KL*
          max(
              coll["kl"],
              0.0
          )
        - LAMBDA_NLL*
          abs(
              coll["delta_nll"]
          )
    )

    # Descriptive efficiency only.
    causal_per_kl=(
        family_mean/
        max(
            coll["kl"],
            1e-6
        )
    )

    profile={
        "method":name,

        "family_mean":
            family_mean,

        "family_sd":
            family_sd,

        "family_min":
            family_min,

        "family_max":
            family_max,

        "kl":
            coll["kl"],

        "delta_nll":
            coll["delta_nll"],

        "causal_per_kl":
            causal_per_kl,

        "selection_score":
            float(
                selection_score
            )
    }

    for r in rows:

        profile[
            r["family"]
        ]=r["Csym"]

    PROFILES.append(
        profile
    )

    print(
        f"  {name:<16} "
        f"mean={family_mean:+.6f} | "
        f"SD={family_sd:.6f} | "
        f"min={family_min:+.6f} | "
        f"KL={coll['kl']:.6f} | "
        f"C/KL={causal_per_kl:+.4f} | "
        f"SCORE={selection_score:+.6f}"
    )

# ==================================================================================================
# 16. FROZEN SELECTION
# ==================================================================================================

eligible=[
    p
    for p in PROFILES
    if p["method"] in FORGE_METHODS
]

WINNER=max(
    eligible,
    key=lambda x:(
        x["selection_score"],
        x["family_min"],
        -x["family_sd"]
    )
)

WINNER_NAME=WINNER[
    "method"
]

WINNER_BETA=float(
    WINNER_NAME
    .replace(
        "BETA_",
        ""
    )
)

WINNER_VECTOR=(
    CANDIDATES[
        WINNER_NAME
    ]
    .clone()
    .contiguous()
)

print("\n       FROZEN WINNER :",WINNER_NAME)
print("       beta          :",WINNER_BETA)
print("       family mean   :",f"{WINNER['family_mean']:+.6f}")
print("       family SD     :",f"{WINNER['family_sd']:.6f}")
print("       family min    :",f"{WINNER['family_min']:+.6f}")
print("       KL            :",f"{WINNER['kl']:.6f}")
print("       score         :",f"{WINNER['selection_score']:+.6f}")

# ==================================================================================================
# 17. LOFO — SELECTION STABILITY
# ==================================================================================================

print("\n[8/15] Leave-one-family-out stability...")

LOFO=[]

family_names=list(
    FORGE_FAMILIES.keys()
)

for held_out in family_names:

    candidates=[]

    for name in FORGE_METHODS:

        rows=[
            r
            for r in FAMILY_ROWS
            if (
                r["method"]==name
                and
                r["family"]!=held_out
            )
        ]

        effects=np.asarray(
            [
                r["Csym"]
                for r in rows
            ],
            dtype=np.float64
        )

        coll=next(
            x
            for x in FORGE_COLLATERAL
            if x["method"]==name
        )

        score=(
            float(
                effects.mean()
            )
            -
            LAMBDA_FAMILY_SD*
            float(
                effects.std(
                    ddof=1
                )
            )
            -
            LAMBDA_KL*
            max(
                coll["kl"],
                0.0
            )
            -
            LAMBDA_NLL*
            abs(
                coll["delta_nll"]
            )
        )

        candidates.append(
            (
                score,
                name
            )
        )

    candidates.sort(
        reverse=True
    )

    selected_score,selected_name=(
        candidates[0]
    )

    held=next(
        r
        for r in FAMILY_ROWS
        if (
            r["method"]==selected_name
            and
            r["family"]==held_out
        )
    )

    row={
        "held_out_family":
            held_out,

        "selected_method":
            selected_name,

        "train_score":
            float(
                selected_score
            ),

        "held_out_csym":
            held["Csym"]
    }

    LOFO.append(
        row
    )

    print(
        f"       hold={held_out:<28} "
        f"select={selected_name:<12} "
        f"| held={held['Csym']:+.6f}"
    )

# ==================================================================================================
# 18. FINAL — FIRST ACCESS
# ==================================================================================================

print("\n[9/15] FINAL first access — frozen vector only...")

FINAL_ARMS={
    "FISHER":
        FISHER,

    "BANK_MEAN":
        BANK,

    "BETA_MINUS_1_REFERENCE":
        CANDIDATES[
            "BETA_-1.00"
        ],

    "NEGATIVE_RESIDUAL_FROZEN":
        WINNER_VECTOR
}

FINAL_ROWS=[]

for scale in FINAL_SCALES:

    print(
        f"\nDOSE={scale:.3f}"
    )

    for name,v in FINAL_ARMS.items():

        r=run_arm(
            name,
            v,
            scale,
            FINAL_ITEMS
        )

        FINAL_ROWS.append(
            r
        )

        print(
            f"  {name:<26} "
            f"Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/{len(FINAL_ITEMS)} | "
            f"real={100*r['realized_mean']:.4f}%"
        )

# ==================================================================================================
# 19. FINAL COLLATERAL
# ==================================================================================================

print("\n[10/15] FINAL collateral...")

FINAL_COLLATERAL=[]

for scale in FINAL_SCALES:

    for name,v in FINAL_ARMS.items():

        c=collateral_arm(
            name,
            v,
            scale,
            NEUTRAL_FINAL
        )

        FINAL_COLLATERAL.append(
            c
        )

        print(
            f"  {name:<26} "
            f"scale={scale:.3f} | "
            f"ΔNLL={c['delta_nll']:+.6f} | "
            f"KL={c['kl']:.6f}"
        )

# ==================================================================================================
# 20. FINAL CAUSAL EFFICIENCY
# ==================================================================================================

print("\n[11/15] Causal-efficiency curve...")

def get_final(
    name,
    scale
):

    return next(
        r
        for r in FINAL_ROWS
        if (
            r["method"]==name
            and
            r["scale"]==scale
        )
    )

def get_coll(
    name,
    scale
):

    return next(
        r
        for r in FINAL_COLLATERAL
        if (
            r["method"]==name
            and
            r["scale"]==scale
        )
    )

EFFICIENCY=[]

for scale in FINAL_SCALES:

    for name in FINAL_ARMS:

        r=get_final(
            name,
            scale
        )

        c=get_coll(
            name,
            scale
        )

        csym_per_kl=(
            r["Csym"]/
            max(
                c["kl"],
                1e-6
            )
        )

        csym_per_realized=(
            r["Csym"]/
            max(
                r["realized_mean"],
                1e-8
            )
        )

        row={
            "method":name,
            "scale":scale,
            "Csym":r["Csym"],
            "KL":c["kl"],
            "delta_nll":
                c["delta_nll"],
            "realized_mean":
                r["realized_mean"],
            "Csym_per_KL":
                float(
                    csym_per_kl
                ),
            "Csym_per_realized":
                float(
                    csym_per_realized
                )
        }

        EFFICIENCY.append(
            row
        )

        print(
            f"  {name:<26} "
            f"scale={scale:.3f} | "
            f"Csym={r['Csym']:+.6f} | "
            f"KL={c['kl']:.6f} | "
            f"C/KL={csym_per_kl:+.5f} | "
            f"C/dose={csym_per_realized:+.5f}"
        )

# ==================================================================================================
# 21. PAIRED BOOTSTRAP
# ==================================================================================================

def paired_bootstrap(
    a,
    b,
    n=20000,
    seed=165
):

    a=np.asarray(
        a,
        dtype=np.float64
    )

    b=np.asarray(
        b,
        dtype=np.float64
    )

    d=a-b

    rng=np.random.default_rng(
        seed
    )

    means=np.empty(
        n,
        dtype=np.float64
    )

    for i in range(n):

        idx=rng.integers(
            0,
            len(d),
            len(d)
        )

        means[i]=(
            d[idx].mean()
        )

    return (
        float(
            d.mean()
        ),

        float(
            np.quantile(
                means,
                .025
            )
        ),

        float(
            np.quantile(
                means,
                .975
            )
        ),

        float(
            (
                np.sum(
                    means<=0
                )+1
            )/
            (n+1)
        )
    )

print("\n[12/15] FINAL paired contrasts...")

CONTRASTS=[]

for scale in FINAL_SCALES:

    s=get_final(
        "NEGATIVE_RESIDUAL_FROZEN",
        scale
    )

    f=get_final(
        "FISHER",
        scale
    )

    b=get_final(
        "BANK_MEAN",
        scale
    )

    r1=get_final(
        "BETA_MINUS_1_REFERENCE",
        scale
    )

    sf=paired_bootstrap(
        s["item_csym"],
        f["item_csym"],
        seed=SEED+
             int(scale*1000)+1
    )

    sb=paired_bootstrap(
        s["item_csym"],
        b["item_csym"],
        seed=SEED+
             int(scale*1000)+2
    )

    sr=paired_bootstrap(
        s["item_csym"],
        r1["item_csym"],
        seed=SEED+
             int(scale*1000)+3
    )

    row={
        "scale":scale,

        "synthetic_csym":
            s["Csym"],

        "fisher_csym":
            f["Csym"],

        "bank_csym":
            b["Csym"],

        "beta_minus_1_csym":
            r1["Csym"],

        "syn_minus_fisher":
            sf[0],

        "sf_ci_low":
            sf[1],

        "sf_ci_high":
            sf[2],

        "sf_p":
            sf[3],

        "syn_minus_bank":
            sb[0],

        "sb_ci_low":
            sb[1],

        "sb_ci_high":
            sb[2],

        "sb_p":
            sb[3],

        "syn_minus_beta_minus_1":
            sr[0],

        "sr_ci_low":
            sr[1],

        "sr_ci_high":
            sr[2],

        "sr_p":
            sr[3]
    }

    CONTRASTS.append(
        row
    )

    print(
        f"\n  scale={scale:.3f}"
    )

    print(
        f"    SYN-FISHER = "
        f"{sf[0]:+.6f} "
        f"[{sf[1]:+.6f},{sf[2]:+.6f}] "
        f"p={sf[3]:.6f}"
    )

    print(
        f"    SYN-BANK   = "
        f"{sb[0]:+.6f} "
        f"[{sb[1]:+.6f},{sb[2]:+.6f}] "
        f"p={sb[3]:.6f}"
    )

    print(
        f"    SYN-BETA-1 = "
        f"{sr[0]:+.6f} "
        f"[{sr[1]:+.6f},{sr[2]:+.6f}] "
        f"p={sr[3]:.6f}"
    )

# ==================================================================================================
# 22. PRE-DECLARED REPLICATION DECISION
# ==================================================================================================

print("\n[13/15] Replication decision...")

LOW_MID=[
    .25,
    .50
]

support_count=0
efficiency_count=0

for scale in LOW_MID:

    s=get_final(
        "NEGATIVE_RESIDUAL_FROZEN",
        scale
    )

    f=get_final(
        "FISHER",
        scale
    )

    sc=get_coll(
        "NEGATIVE_RESIDUAL_FROZEN",
        scale
    )

    fc=get_coll(
        "FISHER",
        scale
    )

    if s["Csym"]>f["Csym"]:
        support_count+=1

    if (
        s["Csym"]>0
        and
        sc["kl"]<fc["kl"]
    ):
        efficiency_count+=1

if (
    support_count==2
    and
    efficiency_count==2
):

    DECISION=(
        "NEGATIVE_RESIDUAL_"
        "EFFICIENCY_REPLICATED"
    )

elif (
    support_count>=1
    and
    efficiency_count>=1
):

    DECISION=(
        "PARTIAL_NEGATIVE_"
        "RESIDUAL_REPLICATION"
    )

else:

    DECISION=(
        "NEGATIVE_RESIDUAL_"
        "ADVANTAGE_NOT_REPLICATED"
    )

# Dose peak is descriptive only.
winner_dose_rows=[
    r
    for r in FINAL_ROWS
    if r["method"]==
       "NEGATIVE_RESIDUAL_FROZEN"
]

PEAK_ROW=max(
    winner_dose_rows,
    key=lambda x:x["Csym"]
)

PEAK_SCALE=PEAK_ROW[
    "scale"
]

print("       Decision   :",DECISION)
print("       Peak dose  :",PEAK_SCALE)
print("       Peak Csym  :",f"{PEAK_ROW['Csym']:+.6f}")

# ==================================================================================================
# 23. SAVE
# ==================================================================================================

print("\n[14/15] Saving...")

def safe(rows):

    return [
        {
            k:v
            for k,v in r.items()
            if not isinstance(
                v,
                (
                    list,
                    dict,
                    tuple
                )
            )
        }
        for r in rows
    ]

def save_csv(
    filename,
    rows
):

    rows=safe(rows)

    if not rows:
        return

    keys=sorted(
        set().union(
            *[
                set(r.keys())
                for r in rows
            ]
        )
    )

    with open(
        ROOT/filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer=csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

GEOMETRY=[
    {
        "layer":L,
        "fisher_bank_cos":
            COS[L],
        "bank_residual_norm":
            RESN[L]
    }
    for L in range(
        N_LAYERS
    )
]

save_csv(
    "SEARCH_GEOMETRY.csv",
    GEOMETRY
)

save_csv(
    "LINEAGE.csv",
    LINEAGE
)

save_csv(
    "FORGE_FAMILY_CAUSAL.csv",
    FAMILY_ROWS
)

save_csv(
    "FORGE_PROFILES.csv",
    PROFILES
)

save_csv(
    "FORGE_COLLATERAL.csv",
    FORGE_COLLATERAL
)

save_csv(
    "LOFO.csv",
    LOFO
)

save_csv(
    "FINAL_CAUSAL.csv",
    FINAL_ROWS
)

save_csv(
    "FINAL_COLLATERAL.csv",
    FINAL_COLLATERAL
)

save_csv(
    "FINAL_EFFICIENCY.csv",
    EFFICIENCY
)

save_csv(
    "FINAL_CONTRASTS.csv",
    CONTRASTS
)

np.savez_compressed(
    ROOT/
    "TEST165_NEGATIVE_RESIDUAL_REPLICATION.npz",

    fisher=
        FISHER
        .detach()
        .cpu()
        .numpy(),

    bank=
        BANK
        .detach()
        .cpu()
        .numpy(),

    bank_residual=
        BANK_RES
        .detach()
        .cpu()
        .numpy(),

    frozen_vector=
        WINNER_VECTOR
        .detach()
        .cpu()
        .numpy(),

    winner_beta=
        np.asarray(
            [WINNER_BETA],
            dtype=np.float32
        ),

    beta_grid=
        np.asarray(
            BETA_GRID,
            dtype=np.float32
        ),

    envelope=
        np.asarray(
            ENVELOPE,
            dtype=np.float32
        )
)

REPORT={

    "test":
        "TEST 165",

    "title":
        "NEGATIVE-RESIDUAL CAUSAL EFFICIENCY REPLICATION",

    "model":
        MODEL_ID,

    "gpu":
        torch.cuda.get_device_name(0),

    "hypothesis":
        (
            "The Fisher-minus-Bank-residual region "
            "may provide superior low/mid-dose causal "
            "efficiency on unseen prompt distributions."
        ),

    "beta_grid":
        BETA_GRID,

    "forge_scale":
        FORGE_SCALE,

    "final_scales":
        FINAL_SCALES,

    "winner":
        WINNER_NAME,

    "winner_beta":
        WINNER_BETA,

    "winner_profile":
        WINNER,

    "peak_final_scale":
        PEAK_SCALE,

    "decision":
        DECISION,

    "selection_contract":{

        "lambda_family_sd":
            LAMBDA_FAMILY_SD,

        "lambda_kl":
            LAMBDA_KL,

        "lambda_nll":
            LAMBDA_NLL
    },

    "geometry":
        GEOMETRY,

    "lineage":
        LINEAGE,

    "forge_profiles":
        PROFILES,

    "lofo":
        LOFO,

    "final_results":
        FINAL_ROWS,

    "final_collateral":
        FINAL_COLLATERAL,

    "final_efficiency":
        EFFICIENCY,

    "final_contrasts":
        CONTRASTS,

    "interpretation_contract":[

        "TEST165 is a replication test of the negative-residual hypothesis generated by TEST164.",

        "TEST163 and TEST164 FORGE and FINAL prompts are not reused.",

        "The SEARCH geometry remains fixed so the tested geometric hypothesis is held constant.",

        "The negative beta neighborhood is declared before FORGE evaluation.",

        "Only the new FORGE families can select the frozen beta.",

        "Fisher, Bank-Mean and Bank residual are reference arms and cannot win selection.",

        "The selected vector is frozen before FINAL is accessed.",

        "FINAL dose points are fixed before FINAL evaluation.",

        "All arms use the same frozen-norm SEASC motor and DRA depth envelope.",

        "All actuator vectors are unit normalized before equal physical dose injection.",

        "Causal potency and collateral KL are reported separately.",

        "Csym/KL and Csym/realized-dose are descriptive efficiency measures.",

        "A high-dose reversal does not invalidate a pre-declared low/mid-dose efficiency hypothesis.",

        "Failure to replicate is retained as a valid result."
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
# 24. FINAL SUMMARY
# ==================================================================================================

print("\n[15/15] COMPLETE")
print("="*150)
print("TEST 165 — FINAL SUMMARY")
print("="*150)

print(
    "Frozen beta              :",
    WINNER_BETA
)

print(
    "Frozen vector            :",
    WINNER_NAME
)

print(
    "FORGE family mean        :",
    f"{WINNER['family_mean']:+.6f}"
)

print(
    "FORGE family SD          :",
    f"{WINNER['family_sd']:.6f}"
)

print(
    "FORGE minimum family     :",
    f"{WINNER['family_min']:+.6f}"
)

print(
    "FORGE KL                 :",
    f"{WINNER['kl']:.6f}"
)

print("\nLOFO")

for x in LOFO:

    print(
        f"  {x['held_out_family']:<28} -> "
        f"{x['selected_method']:<12} "
        f"| held={x['held_out_csym']:+.6f}"
    )

for scale in FINAL_SCALES:

    print(
        f"\nDOSE SCALE = {scale:.3f}"
    )

    for name in FINAL_ARMS:

        r=get_final(
            name,
            scale
        )

        c=get_coll(
            name,
            scale
        )

        e=next(
            x
            for x in EFFICIENCY
            if (
                x["method"]==name
                and
                x["scale"]==scale
            )
        )

        print(
            f"  {name:<26} "
            f"Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/{len(FINAL_ITEMS)} | "
            f"KL={c['kl']:.6f} | "
            f"C/KL={e['Csym_per_KL']:+.4f} | "
            f"real={100*r['realized_mean']:.4f}%"
        )

print("\nFINAL CONTRASTS")

for r in CONTRASTS:

    print(
        f"  scale={r['scale']:.3f} | "
        f"SYN-F={r['syn_minus_fisher']:+.6f} "
        f"[{r['sf_ci_low']:+.6f},{r['sf_ci_high']:+.6f}] | "
        f"SYN-B={r['syn_minus_bank']:+.6f} "
        f"[{r['sb_ci_low']:+.6f},{r['sb_ci_high']:+.6f}]"
    )

print("\nPEAK FROZEN DOSE :",PEAK_SCALE)
print("DECISION         :",DECISION)
print("Saved            :",ROOT)

print("="*150)
print("TEST 165 COMPLETE")
print("="*150)
