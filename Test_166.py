# ==================================================================================================
# TEST 166 — CANALIZED β HOMEOSTAT
# AKBASCORE 3.2 / SEASC
#
# PURPOSE
# -------
# Discover a SAFE OPERATING CORRIDOR on the beta axis:
#
#   maximum useful causal steering
#   with minimum representational disturbance,
#   minimum output collateral,
#   cross-family stability,
#   and local beta robustness.
#
# IMPORTANT:
#   - TEST165's beta=-0.5 is NOT privileged.
#   - Search begins over beta = [-2.0, +2.0].
#   - No single weighted score determines the answer.
#   - Selection is gate -> Pareto -> robustness -> minimum intervention.
#   - FINAL is untouched until the corridor is frozen.
#
# Core quantities:
#
#   Csym       = symmetric causal steering effect
#   KL         = output-distribution collateral
#   |dNLL|     = neutral continuation disturbance
#   D_perp     = hidden displacement orthogonal to target compass
#   family_min = weakest context-family causal effect
#   family_sd  = cross-family instability
#   curvature  = local beta sharpness
#   robustness = beta +/- delta stability
#
# Output:
#   SAFE CORRIDOR
#   FROZEN OPERATING BETA
#   independent FINAL validation
#
# ==================================================================================================

import os
import sys
import math
import json
import csv
import random
import shutil
import subprocess
from pathlib import Path

print("=" * 156)
print("TEST 166 — CANALIZED β HOMEOSTAT")
print("AKBASCORE 3.2 / SEASC")
print("SEARCHING FOR THE SAFE CAUSAL OPERATING CORRIDOR")
print("=" * 156)

# --------------------------------------------------------------------------------------------------
# INSTALL
# --------------------------------------------------------------------------------------------------

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "transformers>=4.45.0",
    "accelerate>=0.34.0",
    "ninja",
    "pybind11"
])

os.environ["MAX_JOBS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

DEVICE = torch.device("cuda")
SEED = 166

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================================================================================================
# CONFIG
# ==================================================================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

N_LAYERS = 20

IVME = 0.10
SONUM = 0.30
ZIRVE = 0.70
TABAN = 0.20

FISHER_ALPHA = 0.10
FISHER_EPS = 1e-8
EPS = 1e-10

# --------------------------------------------------------------------------------------------------
# BROAD SEARCH
# --------------------------------------------------------------------------------------------------

COARSE_BETAS = [
    -2.00,
    -1.75,
    -1.50,
    -1.25,
    -1.00,
    -0.75,
    -0.50,
    -0.25,
     0.00,
     0.25,
     0.50,
     0.75,
     1.00,
     1.25,
     1.50,
     1.75,
     2.00
]

# Primary physical dose.
FORGE_SCALE = 0.50

# Fine local refinement.
FINE_STEP = 0.10

# Micro perturbation for plateau / robustness assay.
MICRO_DELTA = 0.05

# --------------------------------------------------------------------------------------------------
# HOMEOSTATIC GATES
# --------------------------------------------------------------------------------------------------

# Candidate must retain at least 90% of the best causal effect.
CAUSAL_FRACTION = 0.90

# Worst family should remain positive.
REQUIRE_POSITIVE_WORST_FAMILY = True

# Hidden collateral guard relative to best viable candidate.
DPERP_TOLERANCE = 1.20

# Output collateral guard relative to minimum viable KL.
KL_TOLERANCE = 1.35

# --------------------------------------------------------------------------------------------------
# FINAL DOSE RESPONSE
# --------------------------------------------------------------------------------------------------

FINAL_SCALES = [
    0.125,
    0.25,
    0.50,
    0.75,
    1.00
]

SYSTEM = (
    "You are a concise reasoning assistant. "
    "Use only the information in the prompt."
)

ROOT = Path("/content/TEST166_CANALIZED_BETA_HOMEOSTAT")
ROOT.mkdir(parents=True, exist_ok=True)

print("GPU             :", torch.cuda.get_device_name(0))
print("Model           :", MODEL_ID)
print("Steered layers  : L0-L19")
print("Coarse beta     :", COARSE_BETAS)
print("Forge scale     :", FORGE_SCALE)
print("Fine step       :", FINE_STEP)
print("Micro delta     :", MICRO_DELTA)
print("Final scales    :", FINAL_SCALES)

# ==================================================================================================
# 1. CUDA — FROZEN-NORM SEASC
# ==================================================================================================

BUILD_DIR = "/tmp/akbascore_test166"

shutil.rmtree(
    BUILD_DIR,
    ignore_errors=True
)

os.makedirs(
    BUILD_DIR,
    exist_ok=True
)

CPP_SOURCE = r"""
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
    m.def(
        "seasc_batch",
        &seasc_batch,
        "TEST166 SEASC"
    );
}
"""

CUDA_SOURCE = r"""
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
    const int vec = blockIdx.x;
    const int b = vec / S;

    if(b >= B) return;

    extern __shared__ float shared[];

    const long long base =
        (long long)vec *
        (long long)H;

    const long long abase =
        (long long)b *
        (long long)H;

    float local_sum = 0.0f;

    for(
        int j = threadIdx.x;
        j < H;
        j += blockDim.x
    ){
        float x = (float)h[base+j];
        local_sum += x*x;
    }

    shared[threadIdx.x] = local_sum;

    __syncthreads();

    for(
        unsigned int stride =
            blockDim.x/2;
        stride > 0;
        stride >>= 1
    ){
        if(threadIdx.x < stride){
            shared[threadIdx.x] +=
                shared[
                    threadIdx.x +
                    stride
                ];
        }

        __syncthreads();
    }

    const float norm0 =
        sqrtf(
            fmaxf(
                shared[0],
                1.0e-20f
            )
        );

    const float scale =
        dose[b] *
        norm0;

    __syncthreads();

    for(
        int j = threadIdx.x;
        j < H;
        j += blockDim.x
    ){
        float old_value =
            (float)h[base+j];

        h[base+j] =
            (scalar_t)(
                old_value +
                scale *
                actuator[abase+j]
            );
    }
}

torch::Tensor seasc_batch_cuda(
    torch::Tensor hidden,
    torch::Tensor actuator,
    torch::Tensor dose
){
    auto out =
        hidden
        .contiguous()
        .clone();

    auto act =
        actuator
        .to(
            hidden.device(),
            torch::kFloat32
        )
        .contiguous();

    auto d =
        dose
        .to(
            hidden.device(),
            torch::kFloat32
        )
        .contiguous();

    const int B =
        (int)out.size(0);

    const int S =
        (int)out.size(1);

    const int H =
        (int)out.size(2);

    const int n_vectors =
        B*S;

    constexpr int THREADS = 256;

    const size_t shared_bytes =
        THREADS *
        sizeof(float);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        out.scalar_type(),
        "test166_seasc_batch",
        [&]{
            seasc_batch_kernel<scalar_t>
            <<<n_vectors,THREADS,shared_bytes,stream>>>(
                out.data_ptr<scalar_t>(),
                act.data_ptr<float>(),
                d.data_ptr<float>(),
                B,
                S,
                H
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

print("\n[1/18] Compiling SEASC kernel...")

seasc_ext = load_inline(
    name="akbascore_test166_cuda",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cflags=[
        "-O3",
        "-std=c++17"
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math"
    ],
    with_cuda=True,
    build_directory=BUILD_DIR,
    verbose=False
)

print("       CUDA kernel: OK")

# ==================================================================================================
# 2. MODEL
# ==================================================================================================

print("\n[2/18] Loading frozen model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    trust_remote_code=True
)

model.eval()

for p in model.parameters():
    p.requires_grad_(False)

layers = model.model.layers

HIDDEN = model.config.hidden_size

if len(layers) < N_LAYERS:
    raise RuntimeError(
        "Model has fewer than 20 layers."
    )

# --------------------------------------------------------------------------------------------------
# STATIC DRA DEPTH PROFILE
# --------------------------------------------------------------------------------------------------

def dra_envelope(L):

    t = float(L)

    kb = (
        ZIRVE *
        math.exp(
            -SONUM*t
        ) *
        (
            1.0 +
            SONUM*t
        )
        +
        TABAN
    )

    return (
        kb /
        (
            ZIRVE +
            TABAN
        )
    )

ENVELOPE = [
    dra_envelope(L)
    for L in range(
        N_LAYERS
    )
]

print("       hidden :", HIDDEN)
print("       layers :", len(layers))
print("       dtype  :", next(model.parameters()).dtype)

# ==================================================================================================
# 3. SEARCH GEOMETRY
# ==================================================================================================

SEARCH_PAIRS = [

("The lantern is above the stone arch.",
 "The lantern is below the stone arch."),

("The kite is above the old tower.",
 "The kite is below the old tower."),

("The flag is above the wooden gate.",
 "The flag is below the wooden gate."),

("The bird is above the narrow bridge.",
 "The bird is below the narrow bridge."),

("The sign is above the glass doorway.",
 "The sign is below the glass doorway."),

("The lamp is above the metal shelf.",
 "The lamp is below the metal shelf."),

("The clock is above the brown cabinet.",
 "The clock is below the brown cabinet."),

("The balloon is above the green hill.",
 "The balloon is below the green hill."),

("The camera is above the small desk.",
 "The camera is below the small desk."),

("The painting is above the blue sofa.",
 "The painting is below the blue sofa."),

("The cloud is above the distant ridge.",
 "The cloud is below the distant ridge."),

("The marker is above the white board.",
 "The marker is below the white board."),

("The drone is above the concrete wall.",
 "The drone is below the concrete wall."),

("The shelf is above the storage box.",
 "The shelf is below the storage box."),

("The light is above the front window.",
 "The light is below the front window."),

("The symbol is above the horizontal line.",
 "The symbol is below the horizontal line.")
]

# ==================================================================================================
# 4. NEW HOMEOSTAT FAMILIES
# ==================================================================================================

FORGE_FAMILIES = {

"FAMILY_I_SIMPLE":[

{
"context":
"A bronze plaque and a stone pedestal are visible. "
"Their vertical ordering is unknown.",

"pos":
" The bronze plaque is above the stone pedestal.",

"neg":
" The bronze plaque is below the stone pedestal."
},

{
"context":
"A hanging lantern and a wooden crate are visible. "
"Their vertical ordering is unknown.",

"pos":
" The hanging lantern is above the wooden crate.",

"neg":
" The hanging lantern is below the wooden crate."
},

{
"context":
"A ceiling sensor and a workbench are visible. "
"Their vertical ordering is unknown.",

"pos":
" The ceiling sensor is above the workbench.",

"neg":
" The ceiling sensor is below the workbench."
},

{
"context":
"A small flag and a garden wall are visible. "
"Their vertical ordering is unknown.",

"pos":
" The small flag is above the garden wall.",

"neg":
" The small flag is below the garden wall."
}

],

"FAMILY_J_HIGH_LOW":[

{
"context":
"The relative height of a radio mast and a warehouse is unspecified.",

"pos":
" The radio mast is higher than the warehouse.",

"neg":
" The radio mast is lower than the warehouse."
},

{
"context":
"The relative height of a seagull and a dock is unspecified.",

"pos":
" The seagull is higher than the dock.",

"neg":
" The seagull is lower than the dock."
},

{
"context":
"The relative height of a chandelier and a dining table is unspecified.",

"pos":
" The chandelier is higher than the dining table.",

"neg":
" The chandelier is lower than the dining table."
},

{
"context":
"The relative height of a weather balloon and a farmhouse is unspecified.",

"pos":
" The weather balloon is higher than the farmhouse.",

"neg":
" The weather balloon is lower than the farmhouse."
}

],

"FAMILY_K_UPPER_POSITION":[

{
"context":
"A suspended speaker and a stage occupy different vertical positions, "
"but their ordering is not given.",

"pos":
" The suspended speaker occupies the upper position.",

"neg":
" The suspended speaker occupies the lower position."
},

{
"context":
"A security camera and an entrance occupy different vertical positions, "
"but their ordering is not given.",

"pos":
" The security camera occupies the upper position.",

"neg":
" The security camera occupies the lower position."
},

{
"context":
"A skylight and a corridor floor occupy different vertical positions, "
"but their ordering is not given.",

"pos":
" The skylight occupies the upper position.",

"neg":
" The skylight occupies the lower position."
},

{
"context":
"A hanging plant and a bench occupy different vertical positions, "
"but their ordering is not given.",

"pos":
" The hanging plant occupies the upper position.",

"neg":
" The hanging plant occupies the lower position."
}

],

"FAMILY_L_AXIS":[

{
"context":
"Along the vertical axis, the ordering of a drone and a bridge is not specified.",

"pos":
" Along the vertical axis, the drone lies above the bridge.",

"neg":
" Along the vertical axis, the drone lies below the bridge."
},

{
"context":
"Along the vertical axis, the ordering of a beacon and a harbor is not specified.",

"pos":
" Along the vertical axis, the beacon lies above the harbor.",

"neg":
" Along the vertical axis, the beacon lies below the harbor."
},

{
"context":
"Along the vertical axis, the ordering of a cable car and a station is not specified.",

"pos":
" Along the vertical axis, the cable car lies above the station.",

"neg":
" Along the vertical axis, the cable car lies below the station."
},

{
"context":
"Along the vertical axis, the ordering of a ceiling lamp and a desk is not specified.",

"pos":
" Along the vertical axis, the ceiling lamp lies above the desk.",

"neg":
" Along the vertical axis, the ceiling lamp lies below the desk."
}

]

}

# ==================================================================================================
# 5. NEW UNTOUCHED FINAL
# ==================================================================================================

FINAL_ITEMS = [

{
"context":
"A rescue drone and a coastal road are visible, "
"with no vertical relation specified.",

"pos":
" The rescue drone is above the coastal road.",

"neg":
" The rescue drone is below the coastal road."
},

{
"context":
"A hanging display and a service counter are visible, "
"with no vertical relation specified.",

"pos":
" The hanging display is above the service counter.",

"neg":
" The hanging display is below the service counter."
},

{
"context":
"A hawk and a cliff path are visible, "
"with no vertical relation specified.",

"pos":
" The hawk is above the cliff path.",

"neg":
" The hawk is below the cliff path."
},

{
"context":
"A suspended inspection camera and a machine table are visible, "
"with no vertical relation specified.",

"pos":
" The suspended inspection camera is above the machine table.",

"neg":
" The suspended inspection camera is below the machine table."
},

{
"context":
"A navigation beacon and a pier deck are visible, "
"with no vertical relation specified.",

"pos":
" The navigation beacon is above the pier deck.",

"neg":
" The navigation beacon is below the pier deck."
},

{
"context":
"A balloon and an orchard are visible, "
"with no vertical relation specified.",

"pos":
" The balloon is above the orchard.",

"neg":
" The balloon is below the orchard."
},

{
"context":
"A ceiling projector and a classroom table are visible, "
"with no vertical relation specified.",

"pos":
" The ceiling projector is above the classroom table.",

"neg":
" The ceiling projector is below the classroom table."
},

{
"context":
"A suspended detector and a laboratory bench are visible, "
"with no vertical relation specified.",

"pos":
" The suspended detector is above the laboratory bench.",

"neg":
" The suspended detector is below the laboratory bench."
},

{
"context":
"A radar antenna and an airport building are visible, "
"with no vertical relation specified.",

"pos":
" The radar antenna is above the airport building.",

"neg":
" The radar antenna is below the airport building."
},

{
"context":
"A hanging banner and a reception desk are visible, "
"with no vertical relation specified.",

"pos":
" The hanging banner is above the reception desk.",

"neg":
" The hanging banner is below the reception desk."
},

{
"context":
"A glider and a highway are visible, "
"with no vertical relation specified.",

"pos":
" The glider is above the highway.",

"neg":
" The glider is below the highway."
},

{
"context":
"A suspended microphone and a theatre floor are visible, "
"with no vertical relation specified.",

"pos":
" The suspended microphone is above the theatre floor.",

"neg":
" The suspended microphone is below the theatre floor."
},

{
"context":
"A warning lamp and a loading bay are visible, "
"with no vertical relation specified.",

"pos":
" The warning lamp is above the loading bay.",

"neg":
" The warning lamp is below the loading bay."
},

{
"context":
"A cloud and a canyon floor are visible, "
"with no vertical relation specified.",

"pos":
" The cloud is above the canyon floor.",

"neg":
" The cloud is below the canyon floor."
},

{
"context":
"A hanging sculpture and a museum bench are visible, "
"with no vertical relation specified.",

"pos":
" The hanging sculpture is above the museum bench.",

"neg":
" The hanging sculpture is below the museum bench."
},

{
"context":
"An observation balloon and a riverbank are visible, "
"with no vertical relation specified.",

"pos":
" The observation balloon is above the riverbank.",

"neg":
" The observation balloon is below the riverbank."
}

]

# ==================================================================================================
# 6. NEUTRAL COLLATERAL SETS
# ==================================================================================================

NEUTRAL_FORGE = [

"A ceramic cup rests beside a folded newspaper.",

"A metal ruler lies beside a closed notebook.",

"A wooden bowl sits near a plain cloth.",

"A glass jar rests beside a cardboard package."

]

NEUTRAL_FINAL = [

"A calculator lies beside a paper folder.",

"A bottle stands beside a folded towel.",

"A tray rests beside a pair of gloves.",

"A plate sits beside a clean napkin.",

"A suitcase stands near a wooden chair.",

"A small radio rests beside a book."

]

NEUTRAL_CONT = (
    " The scene contains ordinary stationary objects "
    "and no additional relation is specified."
)

# ==================================================================================================
# 7. TOKEN HELPERS
# ==================================================================================================

def chat_text(x):

    return tokenizer.apply_chat_template(
        [
            {
                "role":"system",
                "content":SYSTEM
            },
            {
                "role":"user",
                "content":x
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )

def normalize(v):

    return (
        v /
        v.norm(
            dim=-1,
            keepdim=True
        ).clamp_min(EPS)
    )

# ==================================================================================================
# 8. CAPTURE SEARCH STATES
# ==================================================================================================

@torch.inference_mode()
def capture(text):

    enc = tokenizer(
        chat_text(text),
        return_tensors="pt"
    ).to(DEVICE)

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )

    pos = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )

    hs = [

        out.hidden_states[
            L+1
        ][0,pos]
        .float()
        .detach()

        for L in range(
            N_LAYERS
        )
    ]

    del out
    del enc

    return hs

# ==================================================================================================
# 9. BUILD SEARCH BANK
# ==================================================================================================

print("\n[3/18] Building SEARCH bank...")

POS_BANK = [
    []
    for _ in range(
        N_LAYERS
    )
]

NEG_BANK = [
    []
    for _ in range(
        N_LAYERS
    )
]

for i,(
    ptext,
    ntext
) in enumerate(
    SEARCH_PAIRS,
    1
):

    hp = capture(
        ptext
    )

    hn = capture(
        ntext
    )

    for L in range(
        N_LAYERS
    ):

        POS_BANK[L].append(
            hp[L]
        )

        NEG_BANK[L].append(
            hn[L]
        )

    print(
        f"       pair "
        f"{i:02d}/"
        f"{len(SEARCH_PAIRS)}"
    )

POS_BANK = [

    torch.stack(x)
    .float()
    .contiguous()

    for x in POS_BANK
]

NEG_BANK = [

    torch.stack(x)
    .float()
    .contiguous()

    for x in NEG_BANK
]

# ==================================================================================================
# 10. FISHER / BANK / RESIDUAL
# ==================================================================================================

print("\n[4/18] Building Fisher / Bank / Residual geometry...")

FISHER = []
BANK = []
BANK_RES = []

COS = []
RESN = []

for L in range(
    N_LAYERS
):

    Hp = POS_BANK[L]
    Hn = NEG_BANK[L]

    mp = Hp.mean(0)
    mn = Hn.mean(0)

    md = mp - mn

    R = torch.cat(
        [
            Hp-mp,
            Hn-mn
        ],
        0
    ).float()

    nu = max(
        int(
            R.shape[0]
        ) - 2,
        1
    )

    _, S, Vh = torch.linalg.svd(
        R,
        full_matrices=False
    )

    eig = (
        S.square() /
        float(nu)
    )

    pe = eig[
        eig >
        FISHER_EPS
    ]

    if pe.numel():

        es = torch.median(
            pe
        )

    else:

        es = torch.tensor(
            1.0,
            device=DEVICE
        )

    lam = (
        FISHER_ALPHA *
        es
    ).clamp_min(
        FISHER_EPS
    )

    V = Vh.T

    pr = V.T @ md

    f = (
        md/lam
        +
        V @ (
            (
                1/(eig+lam)
                -
                1/lam
            )
            *
            pr
        )
    )

    f = (
        f /
        f.norm()
        .clamp_min(EPS)
    ).float()

    dirs = normalize(
        Hp-Hn
    )

    sg = torch.where(

        (
            dirs @ f
        )[:,None] < 0,

        -torch.ones_like(
            dirs[:,:1]
        ),

        torch.ones_like(
            dirs[:,:1]
        )
    )

    dirs = (
        dirs *
        sg
    )

    b = dirs.mean(
        0
    )

    b = (
        b /
        b.norm()
        .clamp_min(EPS)
    ).float()

    c = torch.dot(
        f,
        b
    ).clamp(
        -1.0,
        1.0
    )

    if c < 0:

        b = -b
        c = -c

    residual = (
        b -
        c*f
    )

    residual_norm = (
        residual.norm()
    )

    if residual_norm < EPS:

        raise RuntimeError(
            f"Degenerate residual "
            f"L{L}"
        )

    residual = (
        residual /
        residual_norm
    ).float()

    FISHER.append(
        f
    )

    BANK.append(
        b
    )

    BANK_RES.append(
        residual
    )

    COS.append(
        float(c)
    )

    RESN.append(
        float(
            residual_norm
        )
    )

    print(
        f"       L{L:02d} "
        f"cos(F,B)="
        f"{float(c):+.6f} | "
        f"residual="
        f"{float(residual_norm):.6f}"
    )

FISHER = torch.stack(
    FISHER
).float().contiguous()

BANK = torch.stack(
    BANK
).float().contiguous()

BANK_RES = torch.stack(
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
# 11. ACTUATOR FACTORY
# ==================================================================================================

def vector_for_beta(
    beta
):

    vectors = []

    for L in range(
        N_LAYERS
    ):

        c = max(
            COS[L],
            1e-8
        )

        ratio = (
            RESN[L] /
            c
        )

        q = (
            FISHER[L]
            +
            float(beta) *
            ratio *
            BANK_RES[L]
        )

        q = (
            q /
            q.norm()
            .clamp_min(EPS)
        )

        vectors.append(
            q.float()
        )

    return torch.stack(
        vectors
    ).float().contiguous()

# ==================================================================================================
# 12. SEASC ENGINE
# ==================================================================================================

def blank_telemetry():

    return [

        {
            "requested":[],
            "realized":[]
        }

        for _ in range(
            N_LAYERS
        )
    ]

def install_hooks(
    vectors,
    sign,
    dose_scale,
    telemetry,
    state_capture=None
):

    handles = []

    for L in range(
        N_LAYERS
    ):

        def make_hook(
            layer_idx
        ):

            def hook(
                module,
                args,
                output
            ):

                if isinstance(
                    output,
                    tuple
                ):

                    old = output[0]
                    rest = output[1:]

                else:

                    old = output
                    rest = None

                batch = old.shape[0]

                direction = (
                    vectors[
                        layer_idx
                    ][None,:]
                    .expand(
                        batch,
                        -1
                    )
                )

                direction = (
                    float(sign) *
                    direction
                ).float().contiguous()

                requested_value = (
                    IVME *
                    ENVELOPE[
                        layer_idx
                    ] *
                    dose_scale
                )

                dose = torch.full(
                    (batch,),
                    float(
                        requested_value
                    ),
                    device=DEVICE,
                    dtype=torch.float32
                )

                old_last = (
                    old[
                        :,
                        -1,
                        :
                    ]
                    .float()
                )

                new = (
                    seasc_ext
                    .seasc_batch(
                        old,
                        direction,
                        dose
                    )
                )

                new_last = (
                    new[
                        :,
                        -1,
                        :
                    ]
                    .float()
                )

                realized = (
                    (
                        new_last -
                        old_last
                    )
                    .norm(
                        dim=-1
                    )
                    /
                    old_last
                    .norm(
                        dim=-1
                    )
                    .clamp_min(EPS)
                )

                telemetry[
                    layer_idx
                ][
                    "requested"
                ].extend(
                    dose
                    .detach()
                    .cpu()
                    .tolist()
                )

                telemetry[
                    layer_idx
                ][
                    "realized"
                ].extend(
                    realized
                    .detach()
                    .cpu()
                    .tolist()
                )

                if (
                    state_capture
                    is not None
                ):

                    state_capture[
                        layer_idx
                    ] = (
                        new_last
                        .detach()
                        .float()
                        .cpu()
                    )

                if rest is None:
                    return new

                return (
                    new,
                ) + rest

            return hook

        handles.append(

            layers[L]
            .register_forward_hook(
                make_hook(L)
            )
        )

    return handles

# ==================================================================================================
# 13. SEQUENCE / LOGP
# ==================================================================================================

def build_sequence(
    context,
    continuation
):

    p = tokenizer(
        chat_text(
            context
        ),
        return_tensors="pt",
        add_special_tokens=False
    )

    c = tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False
    )

    ids = torch.cat(
        [
            p["input_ids"],
            c["input_ids"]
        ],
        1
    ).to(
        DEVICE
    )

    mask = torch.ones_like(
        ids,
        device=DEVICE
    )

    return (
        ids,
        mask,
        p[
            "input_ids"
        ].shape[1]
    )

@torch.inference_mode()
def continuation_logp(
    context,
    continuation,
    vectors=None,
    sign=0,
    scale=0.0
):

    ids, mask, plen = (
        build_sequence(
            context,
            continuation
        )
    )

    telemetry = (
        blank_telemetry()
    )

    handles = []

    if vectors is not None:

        handles = install_hooks(
            vectors,
            sign,
            scale,
            telemetry
        )

    try:

        out = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True
        )

    finally:

        for h in handles:
            h.remove()

    logits = (
        out.logits
        .float()
    )

    target = (
        ids[
            :,
            plen:
        ]
    )

    pred = (
        logits[
            :,
            plen-1:-1,
            :
        ]
    )

    lp = F.log_softmax(
        pred,
        dim=-1
    )

    token_lp = (
        lp.gather(
            -1,
            target.unsqueeze(
                -1
            )
        )
        .squeeze(
            -1
        )
    )

    score = float(
        token_lp
        .mean()
        .item()
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

    return (
        score,
        telemetry
    )

# ==================================================================================================
# 14. CAUSAL ASSAY
# ==================================================================================================

def run_arm(
    method,
    vectors,
    scale,
    items
):

    cs = []

    plus_margins = []
    minus_margins = []

    requested = []
    realized = []

    for item in items:

        pp,t1 = continuation_logp(
            item["context"],
            item["pos"],
            vectors,
            +1,
            scale
        )

        pn,t2 = continuation_logp(
            item["context"],
            item["neg"],
            vectors,
            +1,
            scale
        )

        np_,t3 = continuation_logp(
            item["context"],
            item["pos"],
            vectors,
            -1,
            scale
        )

        nn,t4 = continuation_logp(
            item["context"],
            item["neg"],
            vectors,
            -1,
            scale
        )

        plus_margin = (
            pp-pn
        )

        minus_margin = (
            np_-nn
        )

        csym = (
            0.5 *
            (
                plus_margin -
                minus_margin
            )
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
            t1,
            t2,
            t3,
            t4
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

        "method":
            method,

        "scale":
            float(scale),

        "Csym":
            float(
                np.mean(cs)
            ),

        "Csym_sd":
            float(
                np.std(
                    cs,
                    ddof=1
                )
            ),

        "plus_margin":
            float(
                np.mean(
                    plus_margins
                )
            ),

        "minus_margin":
            float(
                np.mean(
                    minus_margins
                )
            ),

        "positive_items":
            int(
                np.sum(
                    np.asarray(
                        cs
                    ) > 0
                )
            ),

        "requested_mean":
            float(
                np.mean(
                    requested
                )
            ),

        "realized_mean":
            float(
                np.mean(
                    realized
                )
            ),

        "item_csym":[
            float(x)
            for x in cs
        ]
    }

# ==================================================================================================
# 15. SYMMETRIC COLLATERAL
# ==================================================================================================

@torch.inference_mode()
def neutral_one_sign(
    text,
    vectors,
    scale,
    sign
):

    ids,mask,plen = (
        build_sequence(
            text,
            NEUTRAL_CONT
        )
    )

    out0 = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        return_dict=True
    )

    l0 = (
        out0.logits[
            :,
            plen-1:-1,
            :
        ]
        .float()
    )

    target = (
        ids[
            :,
            plen:
        ]
    )

    lp0 = F.log_softmax(
        l0,
        -1
    )

    nll0 = (
        -lp0.gather(
            -1,
            target.unsqueeze(
                -1
            )
        )
        .squeeze(
            -1
        )
        .mean()
    )

    telemetry = (
        blank_telemetry()
    )

    handles = (
        install_hooks(
            vectors,
            sign,
            scale,
            telemetry
        )
    )

    try:

        out1 = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True
        )

    finally:

        for h in handles:
            h.remove()

    l1 = (
        out1.logits[
            :,
            plen-1:-1,
            :
        ]
        .float()
    )

    lp1 = F.log_softmax(
        l1,
        -1
    )

    nll1 = (
        -lp1.gather(
            -1,
            target.unsqueeze(
                -1
            )
        )
        .squeeze(
            -1
        )
        .mean()
    )

    kl = (
        lp0.exp() *
        (
            lp0-lp1
        )
    ).sum(
        -1
    ).mean()

    delta_nll = float(
        (
            nll1-nll0
        ).item()
    )

    kl_value = float(
        kl.item()
    )

    del (
        ids,
        mask,
        out0,
        out1,
        l0,
        l1,
        target,
        lp0,
        lp1
    )

    return (
        delta_nll,
        kl_value
    )

def collateral_arm(
    method,
    vectors,
    scale,
    neutral_set
):

    plus = []
    minus = []

    for text in neutral_set:

        plus.append(
            neutral_one_sign(
                text,
                vectors,
                scale,
                +1
            )
        )

        minus.append(
            neutral_one_sign(
                text,
                vectors,
                scale,
                -1
            )
        )

    all_kl = (
        [x[1] for x in plus]
        +
        [x[1] for x in minus]
    )

    all_abs_nll = (
        [
            abs(x[0])
            for x in plus
        ]
        +
        [
            abs(x[0])
            for x in minus
        ]
    )

    return {

        "method":
            method,

        "scale":
            float(scale),

        "kl_sym":
            float(
                np.mean(
                    all_kl
                )
            ),

        "abs_delta_nll_sym":
            float(
                np.mean(
                    all_abs_nll
                )
            ),

        "kl_plus":
            float(
                np.mean(
                    [
                        x[1]
                        for x in plus
                    ]
                )
            ),

        "kl_minus":
            float(
                np.mean(
                    [
                        x[1]
                        for x in minus
                    ]
                )
            )
    }

# ==================================================================================================
# 16. HIDDEN-STATE CANALIZATION ASSAY
# ==================================================================================================

@torch.inference_mode()
def capture_vanilla_last_states(
    context
):

    enc = tokenizer(
        chat_text(
            context
        ),
        return_tensors="pt"
    ).to(
        DEVICE
    )

    captured = [
        None
        for _ in range(
            N_LAYERS
        )
    ]

    handles = []

    for L in range(
        N_LAYERS
    ):

        def make_hook(
            layer_idx
        ):

            def hook(
                module,
                args,
                output
            ):

                if isinstance(
                    output,
                    tuple
                ):

                    h = output[0]

                else:

                    h = output

                captured[
                    layer_idx
                ] = (
                    h[
                        :,
                        -1,
                        :
                    ]
                    .detach()
                    .float()
                    .cpu()
                )

            return hook

        handles.append(
            layers[L]
            .register_forward_hook(
                make_hook(L)
            )
        )

    try:

        model(
            **enc,
            use_cache=False,
            return_dict=True
        )

    finally:

        for h in handles:
            h.remove()

    del enc

    return captured

@torch.inference_mode()
def capture_steered_last_states(
    context,
    vectors,
    scale,
    sign
):

    enc = tokenizer(
        chat_text(
            context
        ),
        return_tensors="pt"
    ).to(
        DEVICE
    )

    telemetry = (
        blank_telemetry()
    )

    captured = [
        None
        for _ in range(
            N_LAYERS
        )
    ]

    handles = (
        install_hooks(
            vectors,
            sign,
            scale,
            telemetry,
            state_capture=captured
        )
    )

    try:

        model(
            **enc,
            use_cache=False,
            return_dict=True
        )

    finally:

        for h in handles:
            h.remove()

    del enc

    return captured

def hidden_canalization(
    vectors,
    scale,
    items
):

    values = []

    target_values = []
    total_values = []

    # Use contexts only.
    for item in items:

        context = item[
            "context"
        ]

        vanilla = (
            capture_vanilla_last_states(
                context
            )
        )

        for sign in (
            +1,
            -1
        ):

            steered = (
                capture_steered_last_states(
                    context,
                    vectors,
                    scale,
                    sign
                )
            )

            for L in range(
                N_LAYERS
            ):

                h0 = (
                    vanilla[L][0]
                    .float()
                )

                h1 = (
                    steered[L][0]
                    .float()
                )

                delta = (
                    h1-h0
                )

                target = (
                    float(sign) *
                    vectors[L]
                    .detach()
                    .cpu()
                    .float()
                )

                alpha = (
                    torch.dot(
                        delta,
                        target
                    )
                )

                parallel = (
                    alpha *
                    target
                )

                perp = (
                    delta -
                    parallel
                )

                denom = (
                    h0.norm()
                    .clamp_min(EPS)
                )

                d_total = (
                    delta.norm() /
                    denom
                )

                d_target = (
                    parallel.norm() /
                    denom
                )

                d_perp = (
                    perp.norm() /
                    denom
                )

                total_values.append(
                    float(
                        d_total
                    )
                )

                target_values.append(
                    float(
                        d_target
                    )
                )

                values.append(
                    float(
                        d_perp
                    )
                )

    return {

        "D_perp":
            float(
                np.mean(
                    values
                )
            ),

        "D_target":
            float(
                np.mean(
                    target_values
                )
            ),

        "D_total":
            float(
                np.mean(
                    total_values
                )
            ),

        "off_target_fraction":
            float(
                np.mean(
                    values
                )
                /
                max(
                    np.mean(
                        total_values
                    ),
                    EPS
                )
            )
    }

# ==================================================================================================
# 17. COARSE HOMEOSTAT SCAN
# ==================================================================================================

print("\n[5/18] COARSE HOMEOSTAT SCAN...")

COARSE_ROWS = []

for beta in COARSE_BETAS:

    name = (
        f"BETA_{beta:+.2f}"
    )

    vector = (
        vector_for_beta(
            beta
        )
    )

    family_effects = []

    family_positive = []

    print(
        f"\n       {name}"
    )

    for family_name,items in (
        FORGE_FAMILIES.items()
    ):

        r = run_arm(
            name,
            vector,
            FORGE_SCALE,
            items
        )

        family_effects.append(
            r["Csym"]
        )

        family_positive.append(
            r["positive_items"]
        )

        print(
            f"         "
            f"{family_name:<24} "
            f"Csym={r['Csym']:+.6f}"
        )

    collateral = (
        collateral_arm(
            name,
            vector,
            FORGE_SCALE,
            NEUTRAL_FORGE
        )
    )

    hidden = (
        hidden_canalization(
            vector,
            FORGE_SCALE,
            [
                items[0]
                for items in
                FORGE_FAMILIES.values()
            ]
        )
    )

    family_mean = float(
        np.mean(
            family_effects
        )
    )

    family_sd = float(
        np.std(
            family_effects,
            ddof=1
        )
    )

    family_min = float(
        np.min(
            family_effects
        )
    )

    row = {

        "beta":
            float(beta),

        "Csym":
            family_mean,

        "family_sd":
            family_sd,

        "family_min":
            family_min,

        "KL":
            collateral[
                "kl_sym"
            ],

        "abs_dNLL":
            collateral[
                "abs_delta_nll_sym"
            ],

        "D_perp":
            hidden[
                "D_perp"
            ],

        "D_target":
            hidden[
                "D_target"
            ],

        "D_total":
            hidden[
                "D_total"
            ],

        "off_target_fraction":
            hidden[
                "off_target_fraction"
            ]
    }

    COARSE_ROWS.append(
        row
    )

    print(
        f"         MEAN={family_mean:+.6f} "
        f"| MIN={family_min:+.6f} "
        f"| SD={family_sd:.6f} "
        f"| KL={row['KL']:.6f} "
        f"| Dperp={row['D_perp']:.6f} "
        f"| off-target={row['off_target_fraction']:.4f}"
    )

# ==================================================================================================
# 18. ASHBY VIABILITY GATE
# ==================================================================================================

print("\n[6/18] ASHBY VIABILITY GATE...")

C_MAX = max(
    r["Csym"]
    for r in COARSE_ROWS
)

C_THRESHOLD = (
    CAUSAL_FRACTION *
    C_MAX
)

stage1 = [

    r
    for r in COARSE_ROWS

    if (
        r["Csym"] >=
        C_THRESHOLD
    )
]

if (
    REQUIRE_POSITIVE_WORST_FAMILY
):

    stage1 = [

        r
        for r in stage1

        if (
            r["family_min"]
            > 0
        )
    ]

if not stage1:

    raise RuntimeError(
        "No beta survived causal viability gate."
    )

MIN_KL = min(
    r["KL"]
    for r in stage1
)

MIN_DPERP = min(
    r["D_perp"]
    for r in stage1
)

KL_LIMIT = (
    MIN_KL *
    KL_TOLERANCE
)

DPERP_LIMIT = (
    MIN_DPERP *
    DPERP_TOLERANCE
)

stage2 = [

    r
    for r in stage1

    if (
        r["KL"]
        <= KL_LIMIT
        and
        r["D_perp"]
        <= DPERP_LIMIT
    )
]

if not stage2:

    # Do not silently fail.
    # Fall back to causal viable set,
    # but report it explicitly.
    stage2 = stage1[:]

    SAFETY_GATE_RELAXED = True

else:

    SAFETY_GATE_RELAXED = False

print(
    "       Cmax       :",
    f"{C_MAX:+.6f}"
)

print(
    "       C threshold:",
    f"{C_THRESHOLD:+.6f}"
)

print(
    "       KL limit   :",
    f"{KL_LIMIT:.6f}"
)

print(
    "       Dperp limit:",
    f"{DPERP_LIMIT:.6f}"
)

print(
    "       survivors  :",
    [
        r["beta"]
        for r in stage2
    ]
)

print(
    "       relaxed    :",
    SAFETY_GATE_RELAXED
)

# ==================================================================================================
# 19. PARETO FRONT
# ==================================================================================================

print("\n[7/18] PARETO SAFETY FRONT...")

def dominates(
    a,
    b
):

    # a dominates b if:
    # causal >=
    # and all disturbances <=
    # with at least one strict improvement.

    conditions = [

        a["Csym"]
        >=
        b["Csym"],

        a["KL"]
        <=
        b["KL"],

        a["abs_dNLL"]
        <=
        b["abs_dNLL"],

        a["D_perp"]
        <=
        b["D_perp"],

        a["family_sd"]
        <=
        b["family_sd"]
    ]

    strict = [

        a["Csym"]
        >
        b["Csym"],

        a["KL"]
        <
        b["KL"],

        a["abs_dNLL"]
        <
        b["abs_dNLL"],

        a["D_perp"]
        <
        b["D_perp"],

        a["family_sd"]
        <
        b["family_sd"]
    ]

    return (
        all(
            conditions
        )
        and
        any(
            strict
        )
    )

PARETO = []

for candidate in stage2:

    dominated = False

    for other in stage2:

        if (
            other is candidate
        ):
            continue

        if dominates(
            other,
            candidate
        ):

            dominated = True
            break

    if not dominated:

        PARETO.append(
            candidate
        )

print(
    "       Pareto betas:",
    [
        r["beta"]
        for r in PARETO
    ]
)

# ==================================================================================================
# 20. DEFINE LOCAL REFINEMENT REGION
# ==================================================================================================

print("\n[8/18] LOCAL CORRIDOR REFINEMENT...")

pareto_betas = sorted(
    [
        r["beta"]
        for r in PARETO
    ]
)

left = max(
    -2.0,
    min(
        pareto_betas
    ) - 0.25
)

right = min(
    2.0,
    max(
        pareto_betas
    ) + 0.25
)

FINE_BETAS = (
    np.arange(
        left,
        right + FINE_STEP/2,
        FINE_STEP
    )
    .round(4)
    .tolist()
)

print(
    "       refinement:",
    left,
    "to",
    right
)

print(
    "       fine betas:",
    FINE_BETAS
)

# ==================================================================================================
# 21. FINE SCAN
# ==================================================================================================

FINE_ROWS = []

for beta in FINE_BETAS:

    name = (
        f"FINE_{beta:+.4f}"
    )

    vector = (
        vector_for_beta(
            beta
        )
    )

    effects = []

    for family_name,items in (
        FORGE_FAMILIES.items()
    ):

        r = run_arm(
            name,
            vector,
            FORGE_SCALE,
            items
        )

        effects.append(
            r["Csym"]
        )

    coll = collateral_arm(
        name,
        vector,
        FORGE_SCALE,
        NEUTRAL_FORGE
    )

    hidden = hidden_canalization(
        vector,
        FORGE_SCALE,
        [
            items[0]
            for items in
            FORGE_FAMILIES.values()
        ]
    )

    row = {

        "beta":
            float(beta),

        "Csym":
            float(
                np.mean(
                    effects
                )
            ),

        "family_sd":
            float(
                np.std(
                    effects,
                    ddof=1
                )
            ),

        "family_min":
            float(
                np.min(
                    effects
                )
            ),

        "KL":
            coll[
                "kl_sym"
            ],

        "abs_dNLL":
            coll[
                "abs_delta_nll_sym"
            ],

        "D_perp":
            hidden[
                "D_perp"
            ],

        "D_target":
            hidden[
                "D_target"
            ],

        "D_total":
            hidden[
                "D_total"
            ],

        "off_target_fraction":
            hidden[
                "off_target_fraction"
            ]
    }

    FINE_ROWS.append(
        row
    )

    print(
        f"       beta={beta:+.4f} "
        f"C={row['Csym']:+.6f} "
        f"| KL={row['KL']:.6f} "
        f"| Dperp={row['D_perp']:.6f} "
        f"| min={row['family_min']:+.6f}"
    )

# ==================================================================================================
# 22. MICRO-PERTURBATION / PLATEAU ASSAY
# ==================================================================================================

print("\n[9/18] MICRO-PERTURBATION PLATEAU ASSAY...")

def quick_family_causal(
    beta
):

    vector = (
        vector_for_beta(
            beta
        )
    )

    effects = []

    for items in (
        FORGE_FAMILIES.values()
    ):

        r = run_arm(
            "MICRO",
            vector,
            FORGE_SCALE,
            items
        )

        effects.append(
            r["Csym"]
        )

    return float(
        np.mean(
            effects
        )
    )

for row in FINE_ROWS:

    beta = row[
        "beta"
    ]

    c0 = row[
        "Csym"
    ]

    cm = quick_family_causal(
        beta -
        MICRO_DELTA
    )

    cp = quick_family_causal(
        beta +
        MICRO_DELTA
    )

    curvature = abs(
        cp -
        2*c0 +
        cm
    )

    local_sd = float(
        np.std(
            [
                cm,
                c0,
                cp
            ],
            ddof=1
        )
    )

    local_min = float(
        min(
            cm,
            c0,
            cp
        )
    )

    row[
        "micro_minus"
    ] = cm

    row[
        "micro_plus"
    ] = cp

    row[
        "curvature"
    ] = float(
        curvature
    )

    row[
        "micro_sd"
    ] = local_sd

    row[
        "micro_min"
    ] = local_min

    print(
        f"       beta={beta:+.4f} "
        f"| C-={cm:+.6f} "
        f"C0={c0:+.6f} "
        f"C+={cp:+.6f} "
        f"| curvature={curvature:.6f} "
        f"| microSD={local_sd:.6f}"
    )

# ==================================================================================================
# 23. FINE VIABILITY
# ==================================================================================================

print("\n[10/18] CANALIZED SAFE-CORRIDOR GATE...")

FINE_CMAX = max(
    r["Csym"]
    for r in FINE_ROWS
)

FINE_THRESHOLD = (
    CAUSAL_FRACTION *
    FINE_CMAX
)

fine_viable = [

    r
    for r in FINE_ROWS

    if (
        r["Csym"]
        >=
        FINE_THRESHOLD
        and
        r["family_min"]
        > 0
        and
        r["micro_min"]
        > 0
    )
]

if not fine_viable:

    raise RuntimeError(
        "No beta survived fine viability."
    )

fine_min_kl = min(
    r["KL"]
    for r in fine_viable
)

fine_min_dperp = min(
    r["D_perp"]
    for r in fine_viable
)

fine_kl_limit = (
    fine_min_kl *
    KL_TOLERANCE
)

fine_dperp_limit = (
    fine_min_dperp *
    DPERP_TOLERANCE
)

safe = [

    r
    for r in fine_viable

    if (
        r["KL"]
        <=
        fine_kl_limit
        and
        r["D_perp"]
        <=
        fine_dperp_limit
    )
]

if not safe:

    safe = fine_viable[:]

    FINE_SAFETY_RELAXED = True

else:

    FINE_SAFETY_RELAXED = False

# --------------------------------------------------------------------------------------------------
# PARETO AGAIN
# --------------------------------------------------------------------------------------------------

SAFE_PARETO = []

for candidate in safe:

    dominated = False

    for other in safe:

        if (
            other is candidate
        ):
            continue

        if dominates(
            other,
            candidate
        ):

            dominated = True
            break

    if not dominated:

        SAFE_PARETO.append(
            candidate
        )

# ==================================================================================================
# 24. PHYSARUM CONDUCTANCE
# ==================================================================================================

print("\n[11/18] PHYSARUM-LIKE CONDUCTANCE...")

# Not used as sole selection criterion.
# It is a secondary self-organizing diagnostic.

for row in SAFE_PARETO:

    useful = max(
        row["Csym"] -
        FINE_THRESHOLD,
        0.0
    ) + 1e-6

    disturbance = (
        (
            row["KL"]
            + 1e-6
        )
        *
        (
            row["D_perp"]
            + 1e-6
        )
        *
        (
            row["family_sd"]
            + 1e-6
        )
        *
        (
            row["micro_sd"]
            + 1e-6
        )
    )

    row[
        "conductance"
    ] = float(
        useful /
        disturbance
    )

# Normalize conductance.
if SAFE_PARETO:

    max_cond = max(
        r["conductance"]
        for r in SAFE_PARETO
    )

    for r in SAFE_PARETO:

        r[
            "conductance_norm"
        ] = (
            r[
                "conductance"
            ] /
            max(
                max_cond,
                EPS
            )
        )

# ==================================================================================================
# 25. FINAL HOMEOSTATIC CHOICE
# ==================================================================================================

print("\n[12/18] FREEZING OPERATING CORRIDOR...")

# We do NOT simply choose max conductance.
#
# Lexicographic choice:
# 1. SAFE PARETO only
# 2. smallest micro instability
# 3. smallest D_perp
# 4. smallest KL
# 5. smallest |beta|
#
# This prioritizes a broad, calm plateau.

if not SAFE_PARETO:

    raise RuntimeError(
        "SAFE_PARETO empty."
    )

# Define robust subset:
best_micro = min(
    r["micro_sd"]
    for r in SAFE_PARETO
)

ROBUST_TOL = (
    best_micro *
    1.25
    +
    1e-12
)

robust = [

    r
    for r in SAFE_PARETO

    if (
        r["micro_sd"]
        <=
        ROBUST_TOL
    )
]

if not robust:

    robust = SAFE_PARETO[:]

# Then smallest hidden disturbance.
best_dperp = min(
    r["D_perp"]
    for r in robust
)

DPERP_FINAL_TOL = (
    best_dperp *
    1.10
    +
    1e-12
)

robust2 = [

    r
    for r in robust

    if (
        r["D_perp"]
        <=
        DPERP_FINAL_TOL
    )
]

if not robust2:

    robust2 = robust[:]

# Then smallest KL.
best_kl = min(
    r["KL"]
    for r in robust2
)

KL_FINAL_TOL = (
    best_kl *
    1.10
    +
    1e-12
)

robust3 = [

    r
    for r in robust2

    if (
        r["KL"]
        <=
        KL_FINAL_TOL
    )
]

if not robust3:

    robust3 = robust2[:]

# Minimum intervention as final tie breaker.
WINNER = min(
    robust3,
    key=lambda r:(
        abs(
            r["beta"]
        ),
        r["micro_sd"],
        r["D_perp"],
        r["KL"]
    )
)

WINNER_BETA = float(
    WINNER[
        "beta"
    ]
)

WINNER_VECTOR = (
    vector_for_beta(
        WINNER_BETA
    )
)

# --------------------------------------------------------------------------------------------------
# SAFE CORRIDOR = contiguous safe points around winner.
# --------------------------------------------------------------------------------------------------

safe_beta_set = set(
    round(
        r["beta"],
        4
    )
    for r in safe
)

corridor = [
    round(
        WINNER_BETA,
        4
    )
]

current = round(
    WINNER_BETA -
    FINE_STEP,
    4
)

while current in safe_beta_set:

    corridor.append(
        current
    )

    current = round(
        current -
        FINE_STEP,
        4
    )

current = round(
    WINNER_BETA +
    FINE_STEP,
    4
)

while current in safe_beta_set:

    corridor.append(
        current
    )

    current = round(
        current +
        FINE_STEP,
        4
    )

corridor = sorted(
    corridor
)

CORRIDOR_LEFT = min(
    corridor
)

CORRIDOR_RIGHT = max(
    corridor
)

print(
    "\n       SAFE CORRIDOR :",
    f"[{CORRIDOR_LEFT:+.4f}, "
    f"{CORRIDOR_RIGHT:+.4f}]"
)

print(
    "       FROZEN β     :",
    f"{WINNER_BETA:+.4f}"
)

print(
    "       Csym         :",
    f"{WINNER['Csym']:+.6f}"
)

print(
    "       KL           :",
    f"{WINNER['KL']:.6f}"
)

print(
    "       D_perp       :",
    f"{WINNER['D_perp']:.6f}"
)

print(
    "       micro SD     :",
    f"{WINNER['micro_sd']:.6f}"
)

print(
    "       curvature    :",
    f"{WINNER['curvature']:.6f}"
)

# ==================================================================================================
# 26. FINAL — FIRST ACCESS
# ==================================================================================================

print("\n[13/18] FINAL — FIRST ACCESS AFTER FREEZE...")

FINAL_ARMS = {

    "FISHER":
        FISHER,

    "BANK_MEAN":
        BANK,

    "HOMEOSTAT_FROZEN":
        WINNER_VECTOR
}

FINAL_ROWS = []

FINAL_COLLATERAL = []

for scale in FINAL_SCALES:

    print(
        f"\nDOSE={scale:.3f}"
    )

    for name,vector in (
        FINAL_ARMS.items()
    ):

        r = run_arm(
            name,
            vector,
            scale,
            FINAL_ITEMS
        )

        c = collateral_arm(
            name,
            vector,
            scale,
            NEUTRAL_FINAL
        )

        FINAL_ROWS.append(
            r
        )

        FINAL_COLLATERAL.append(
            c
        )

        print(
            f"  {name:<20} "
            f"Csym={r['Csym']:+.6f} "
            f"| P+={r['positive_items']:02d}/{len(FINAL_ITEMS)} "
            f"| KL={c['kl_sym']:.6f} "
            f"| |dNLL|={c['abs_delta_nll_sym']:.6f} "
            f"| real={100*r['realized_mean']:.4f}%"
        )

# ==================================================================================================
# 27. FINAL HIDDEN CANALIZATION
# ==================================================================================================

print("\n[14/18] FINAL HIDDEN CANALIZATION...")

FINAL_HIDDEN = []

hidden_probe_items = (
    FINAL_ITEMS[:4]
)

for scale in FINAL_SCALES:

    for name,vector in (
        FINAL_ARMS.items()
    ):

        h = hidden_canalization(
            vector,
            scale,
            hidden_probe_items
        )

        row = {

            "method":
                name,

            "scale":
                float(scale),

            **h
        }

        FINAL_HIDDEN.append(
            row
        )

        print(
            f"  {name:<20} "
            f"scale={scale:.3f} "
            f"| Dperp={h['D_perp']:.6f} "
            f"| Dtarget={h['D_target']:.6f} "
            f"| off-target={h['off_target_fraction']:.4f}"
        )

# ==================================================================================================
# 28. BOOTSTRAP
# ==================================================================================================

print("\n[15/18] FINAL PAIRED BOOTSTRAP...")

def get_final(
    name,
    scale
):

    return next(

        r
        for r in FINAL_ROWS

        if (
            r["method"] == name
            and
            r["scale"] == scale
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
            r["method"] == name
            and
            r["scale"] == scale
        )
    )

def paired_bootstrap(
    a,
    b,
    n=20000,
    seed=166
):

    a = np.asarray(
        a,
        dtype=np.float64
    )

    b = np.asarray(
        b,
        dtype=np.float64
    )

    d = (
        a-b
    )

    rng = (
        np.random
        .default_rng(
            seed
        )
    )

    means = np.empty(
        n,
        dtype=np.float64
    )

    for i in range(
        n
    ):

        idx = rng.integers(
            0,
            len(d),
            len(d)
        )

        means[i] = (
            d[idx]
            .mean()
        )

    return {

        "delta":
            float(
                d.mean()
            ),

        "ci_low":
            float(
                np.quantile(
                    means,
                    0.025
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    means,
                    0.975
                )
            )
    }

CONTRASTS = []

for scale in FINAL_SCALES:

    h = get_final(
        "HOMEOSTAT_FROZEN",
        scale
    )

    f = get_final(
        "FISHER",
        scale
    )

    b = get_final(
        "BANK_MEAN",
        scale
    )

    hf = paired_bootstrap(
        h["item_csym"],
        f["item_csym"],
        seed=
            SEED +
            int(
                scale*1000
            ) +
            1
    )

    hb = paired_bootstrap(
        h["item_csym"],
        b["item_csym"],
        seed=
            SEED +
            int(
                scale*1000
            ) +
            2
    )

    row = {

        "scale":
            float(scale),

        "homeostat_minus_fisher":
            hf["delta"],

        "hf_ci_low":
            hf["ci_low"],

        "hf_ci_high":
            hf["ci_high"],

        "homeostat_minus_bank":
            hb["delta"],

        "hb_ci_low":
            hb["ci_low"],

        "hb_ci_high":
            hb["ci_high"]
    }

    CONTRASTS.append(
        row
    )

    print(
        f"  scale={scale:.3f} | "
        f"H-F={hf['delta']:+.6f} "
        f"[{hf['ci_low']:+.6f},"
        f"{hf['ci_high']:+.6f}] | "
        f"H-B={hb['delta']:+.6f} "
        f"[{hb['ci_low']:+.6f},"
        f"{hb['ci_high']:+.6f}]"
    )

# ==================================================================================================
# 29. FINAL EFFICIENCY
# ==================================================================================================

print("\n[16/18] FINAL CAUSAL / DISTURBANCE EFFICIENCY...")

EFFICIENCY = []

for scale in FINAL_SCALES:

    for name in FINAL_ARMS:

        r = get_final(
            name,
            scale
        )

        c = get_coll(
            name,
            scale
        )

        h = next(

            x
            for x in FINAL_HIDDEN

            if (
                x["method"] == name
                and
                x["scale"] == scale
            )
        )

        causal_per_kl = (
            r["Csym"] /
            max(
                c["kl_sym"],
                1e-8
            )
        )

        causal_per_dperp = (
            r["Csym"] /
            max(
                h["D_perp"],
                1e-8
            )
        )

        causal_per_total_disturbance = (
            r["Csym"] /
            max(
                (
                    c["kl_sym"]
                    *
                    h["D_perp"]
                ),
                1e-8
            )
        )

        row = {

            "method":
                name,

            "scale":
                float(scale),

            "Csym":
                r["Csym"],

            "KL":
                c["kl_sym"],

            "D_perp":
                h["D_perp"],

            "C_per_KL":
                float(
                    causal_per_kl
                ),

            "C_per_Dperp":
                float(
                    causal_per_dperp
                ),

            "C_per_KLxDperp":
                float(
                    causal_per_total_disturbance
                )
        }

        EFFICIENCY.append(
            row
        )

        print(
            f"  {name:<20} "
            f"scale={scale:.3f} "
            f"| C/KL={causal_per_kl:+.5f} "
            f"| C/Dperp={causal_per_dperp:+.5f} "
            f"| C/(KL*Dperp)="
            f"{causal_per_total_disturbance:+.5f}"
        )

# ==================================================================================================
# 30. DECISION
# ==================================================================================================

print("\n[17/18] HOMEOSTAT DECISION...")

primary_scale = 0.50

home = get_final(
    "HOMEOSTAT_FROZEN",
    primary_scale
)

fish = get_final(
    "FISHER",
    primary_scale
)

home_coll = get_coll(
    "HOMEOSTAT_FROZEN",
    primary_scale
)

fish_coll = get_coll(
    "FISHER",
    primary_scale
)

home_hidden = next(

    x
    for x in FINAL_HIDDEN

    if (
        x["method"] ==
        "HOMEOSTAT_FROZEN"
        and
        x["scale"] ==
        primary_scale
    )
)

fish_hidden = next(

    x
    for x in FINAL_HIDDEN

    if (
        x["method"] ==
        "FISHER"
        and
        x["scale"] ==
        primary_scale
    )
)

causal_retention = (
    home["Csym"] /
    max(
        fish["Csym"],
        EPS
    )
)

kl_ratio = (
    home_coll["kl_sym"] /
    max(
        fish_coll["kl_sym"],
        EPS
    )
)

dperp_ratio = (
    home_hidden["D_perp"] /
    max(
        fish_hidden["D_perp"],
        EPS
    )
)

if (
    causal_retention >= 0.90
    and
    kl_ratio < 1.0
    and
    dperp_ratio < 1.0
):

    DECISION = (
        "SAFE_CAUSAL_CORRIDOR_SUPPORTED"
    )

elif (
    causal_retention >= 0.90
    and
    (
        kl_ratio < 1.0
        or
        dperp_ratio < 1.0
    )
):

    DECISION = (
        "PARTIAL_SAFE_CORRIDOR_SIGNAL"
    )

else:

    DECISION = (
        "SAFE_CORRIDOR_NOT_CONFIRMED"
    )

print(
    "       Frozen beta      :",
    f"{WINNER_BETA:+.4f}"
)

print(
    "       Corridor         :",
    f"[{CORRIDOR_LEFT:+.4f}, "
    f"{CORRIDOR_RIGHT:+.4f}]"
)

print(
    "       Causal retention :",
    f"{100*causal_retention:.2f}%"
)

print(
    "       KL / Fisher      :",
    f"{kl_ratio:.4f}"
)

print(
    "       Dperp / Fisher   :",
    f"{dperp_ratio:.4f}"
)

print(
    "       DECISION         :",
    DECISION
)

# ==================================================================================================
# 31. SAVE
# ==================================================================================================

print("\n[18/18] SAVING COMPLETE RECORD...")

def safe_rows(
    rows
):

    return [

        {
            k:v

            for k,v
            in r.items()

            if not isinstance(
                v,
                (
                    list,
                    tuple,
                    dict
                )
            )
        }

        for r in rows
    ]

def save_csv(
    filename,
    rows
):

    rows = safe_rows(
        rows
    )

    if not rows:
        return

    keys = sorted(
        set().union(
            *[
                set(
                    r.keys()
                )
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

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

save_csv(
    "COARSE_HOMEOSTAT.csv",
    COARSE_ROWS
)

save_csv(
    "FINE_HOMEOSTAT.csv",
    FINE_ROWS
)

save_csv(
    "SAFE_PARETO.csv",
    SAFE_PARETO
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
    "FINAL_HIDDEN_CANALIZATION.csv",
    FINAL_HIDDEN
)

save_csv(
    "FINAL_CONTRASTS.csv",
    CONTRASTS
)

save_csv(
    "FINAL_EFFICIENCY.csv",
    EFFICIENCY
)

np.savez_compressed(

    ROOT /
    "TEST166_CANALIZED_BETA_HOMEOSTAT.npz",

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

    frozen_beta=
        np.asarray(
            [
                WINNER_BETA
            ],
            dtype=np.float32
        ),

    safe_corridor=
        np.asarray(
            [
                CORRIDOR_LEFT,
                CORRIDOR_RIGHT
            ],
            dtype=np.float32
        ),

    envelope=
        np.asarray(
            ENVELOPE,
            dtype=np.float32
        )
)

REPORT = {

    "test":
        "TEST 166",

    "title":
        "CANALIZED BETA HOMEOSTAT",

    "model":
        MODEL_ID,

    "gpu":
        torch.cuda.get_device_name(0),

    "frozen_beta":
        WINNER_BETA,

    "safe_corridor":[
        CORRIDOR_LEFT,
        CORRIDOR_RIGHT
    ],

    "decision":
        DECISION,

    "primary_scale":
        primary_scale,

    "causal_retention_vs_fisher":
        causal_retention,

    "kl_ratio_vs_fisher":
        kl_ratio,

    "dperp_ratio_vs_fisher":
        dperp_ratio,

    "coarse_rows":
        COARSE_ROWS,

    "fine_rows":
        FINE_ROWS,

    "safe_pareto":
        SAFE_PARETO,

    "final_causal":
        FINAL_ROWS,

    "final_collateral":
        FINAL_COLLATERAL,

    "final_hidden":
        FINAL_HIDDEN,

    "final_efficiency":
        EFFICIENCY,

    "final_contrasts":
        CONTRASTS,

    "method_contract":[

        "Beta search begins broadly from -2 to +2.",

        "TEST165 beta=-0.5 is not privileged.",

        "Causal viability is applied before collateral optimization.",

        "Worst-family positivity is required during corridor discovery.",

        "Symmetric plus/minus neutral collateral is measured.",

        "Hidden-state off-target displacement is measured orthogonally to the actual actuator direction.",

        "Pareto dominance is used instead of a single weighted utility.",

        "Local beta perturbation tests plateau robustness.",

        "Physarum-like conductance is diagnostic and does not solely determine selection.",

        "The final tie breaker prefers minimum intervention magnitude.",

        "The operating corridor is frozen before FINAL access.",

        "FINAL prompts are not used for corridor construction.",

        "The experiment tests a safe operating region, not a universal optimum beta."
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

print("\n" + "="*156)
print("TEST 166 — FINAL HOMEOSTAT SUMMARY")
print("="*156)

print(
    "SAFE CORRIDOR       :",
    f"[{CORRIDOR_LEFT:+.4f}, "
    f"{CORRIDOR_RIGHT:+.4f}]"
)

print(
    "FROZEN BETA         :",
    f"{WINNER_BETA:+.4f}"
)

print(
    "FORGE Csym          :",
    f"{WINNER['Csym']:+.6f}"
)

print(
    "FORGE KL            :",
    f"{WINNER['KL']:.6f}"
)

print(
    "FORGE D_perp        :",
    f"{WINNER['D_perp']:.6f}"
)

print(
    "LOCAL CURVATURE     :",
    f"{WINNER['curvature']:.6f}"
)

print(
    "LOCAL MICRO SD      :",
    f"{WINNER['micro_sd']:.6f}"
)

print(
    "FINAL @0.50"
)

print(
    "  causal retention  :",
    f"{100*causal_retention:.2f}%"
)

print(
    "  KL/Fisher         :",
    f"{kl_ratio:.4f}"
)

print(
    "  Dperp/Fisher      :",
    f"{dperp_ratio:.4f}"
)

print(
    "DECISION            :",
    DECISION
)

print(
    "Saved               :",
    ROOT
)

print("="*156)
print("TEST 166 COMPLETE")
print("="*156)
