# =============================================================================
# TEST 160 — STATE-SPECIFIC ROUTING + CALIBRATED GAIN
# AKBASCORE 3.1 / SEASC
#
# CORRECTIVE FALSIFICATION OF TEST 159
#
# Qwen/Qwen2.5-7B-Instruct · BF16 · A100
#
# TEST159 LEFT TWO OPEN QUESTIONS
# --------------------------------
#
# 1) STATE-GAIN was degenerate:
#
#       batch size = 1
#       q - mean(q_batch) = 0
#       g(h) = 1
#
#    TEST160 calibrates gain ONLY from SEARCH reference states.
#
# 2) One deterministic shuffled mapping nearly matched the real mapping.
#
#    TEST160 compares MATCHED routing against 8 fixed independent
#    state->direction permutations.
#
# PRIMARY QUESTIONS
# -----------------
#
# Q1:
# Does calibrated state-dependent gain improve over fixed local steering?
#
# Q2:
# Does MATCHED state->direction routing outperform a distribution of
# shuffled state->direction mappings?
#
# Q3:
# Does direction + calibrated gain outperform direction alone?
#
# MOTOR
# -----
#
# Same AkbasCore frozen-norm physical law:
#
#   dh_L = rho_L * ||h_L|| * A_L
#
# Same L0-L19 static normalized DRA envelope.
#
# Frozen model.
# No training.
# No gradients.
# No LoRA.
# No optimizer.
# No Retention Guard.
# =============================================================================

import os
import sys
import gc
import math
import json
import csv
import random
import shutil
import subprocess
from pathlib import Path

print("=" * 150)
print("TEST 160 — STATE-SPECIFIC ROUTING + CALIBRATED GAIN")
print("AKBASCORE 3.1 / SEASC")
print("=" * 150)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45.0",
    "accelerate>=0.34.0",
    "ninja",
    "pybind11",
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

SEED = 160
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =============================================================================
# 1. LOCKED CONFIG
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

N_LAYERS = 20

IVME = 0.10
SONUM = 0.30
ZIRVE = 0.70
TABAN = 0.20

FISHER_ALPHA = 0.10
FISHER_EPS = 1e-8

STATE_TEMP = 0.10

# Gain calibration.
# z=0 -> gain 1.
# Positive SEARCH-calibrated confidence -> larger dose.
GAIN_BETA = 0.35
GAIN_MIN = 0.50
GAIN_MAX = 1.50
GAIN_EPS = 1e-6

# Three intervention regimes retained.
DOSE_SCALES = [0.25, 0.50, 1.00]

# Real matched routing is compared against this many independent null maps.
N_SHUFFLES = 8

EPS = 1e-10

SYSTEM = (
    "You are a concise reasoning assistant. "
    "Use only the information in the prompt."
)

ROOT = Path("/content/TEST160_STATE_ROUTING_GAIN")
ROOT.mkdir(parents=True, exist_ok=True)

print("GPU            :", torch.cuda.get_device_name(0))
print("Model          :", MODEL_ID)
print("Steered layers : L0-L19")
print("Dose scales    :", DOSE_SCALES)
print("Shuffle nulls  :", N_SHUFFLES)

# =============================================================================
# 2. CUDA — SAME FROZEN-NORM PHYSICAL MOTOR
# =============================================================================

BUILD_DIR = "/tmp/akbascore_test160"

if os.path.exists(BUILD_DIR):
    shutil.rmtree(BUILD_DIR, ignore_errors=True)

os.makedirs(BUILD_DIR, exist_ok=True)

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
) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(actuator.is_cuda(), "actuator must be CUDA");
    TORCH_CHECK(dose.is_cuda(), "dose must be CUDA");

    TORCH_CHECK(hidden.dim() == 3, "hidden must be [B,S,H]");
    TORCH_CHECK(actuator.dim() == 2, "actuator must be [B,H]");
    TORCH_CHECK(dose.dim() == 1, "dose must be [B]");

    TORCH_CHECK(hidden.size(0) == actuator.size(0), "batch mismatch");
    TORCH_CHECK(hidden.size(0) == dose.size(0), "dose batch mismatch");
    TORCH_CHECK(hidden.size(2) == actuator.size(1), "hidden mismatch");

    return seasc_batch_cuda(hidden, actuator, dose);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "seasc_batch",
        &seasc_batch,
        "TEST160 SEASC batch frozen-norm injection"
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
) {
    const int vec = blockIdx.x;
    const int b = vec / S;

    if (b >= B) return;

    extern __shared__ float shared[];

    const long long base =
        static_cast<long long>(vec)
        * static_cast<long long>(H);

    const long long abase =
        static_cast<long long>(b)
        * static_cast<long long>(H);

    float local_sum = 0.0f;

    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const float x =
            static_cast<float>(h[base + j]);

        local_sum += x * x;
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (
        unsigned int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] +=
                shared[threadIdx.x + stride];
        }

        __syncthreads();
    }

    const float norm0 =
        sqrtf(fmaxf(shared[0], 1.0e-20f));

    const float scale =
        dose[b] * norm0;

    __syncthreads();

    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const float old_value =
            static_cast<float>(h[base + j]);

        const float new_value =
            old_value
            + scale
            * actuator[abase + j];

        h[base + j] =
            static_cast<scalar_t>(new_value);
    }
}

torch::Tensor seasc_batch_cuda(
    torch::Tensor hidden,
    torch::Tensor actuator,
    torch::Tensor dose
) {
    auto out =
        hidden
        .contiguous()
        .clone();

    auto act =
        actuator
        .to(hidden.device(), torch::kFloat32)
        .contiguous();

    auto d =
        dose
        .to(hidden.device(), torch::kFloat32)
        .contiguous();

    const int B = static_cast<int>(out.size(0));
    const int S = static_cast<int>(out.size(1));
    const int H = static_cast<int>(out.size(2));

    const int n_vectors = B * S;

    constexpr int THREADS = 256;

    const size_t shared_bytes =
        THREADS * sizeof(float);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        out.scalar_type(),
        "test160_seasc_batch",
        [&] {
            seasc_batch_kernel<scalar_t>
                <<<n_vectors, THREADS, shared_bytes, stream>>>(
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

print("\n[1/14] Compiling AkbasCore SEASC kernel...")

seasc_ext = load_inline(
    name="akbascore_test160_cuda",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cflags=[
        "-O3",
        "-std=c++17",
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    with_cuda=True,
    build_directory=BUILD_DIR,
    verbose=False,
)

print("      CUDA kernel: OK")

# =============================================================================
# 3. MODEL
# =============================================================================

print("\n[2/14] Loading frozen model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    trust_remote_code=True,
)

model.eval()

for p in model.parameters():
    p.requires_grad_(False)

layers = model.model.layers
HIDDEN = model.config.hidden_size

if len(layers) < N_LAYERS:
    raise RuntimeError("Model has fewer than 20 layers.")

print("      hidden :", HIDDEN)
print("      layers :", len(layers))
print("      dtype  :", next(model.parameters()).dtype)

# =============================================================================
# 4. LOCKED DRA ENVELOPE
# =============================================================================

def dra_envelope(L):
    t = float(L)

    kb = (
        ZIRVE
        * math.exp(-SONUM * t)
        * (1.0 + SONUM * t)
        + TABAN
    )

    return kb / (ZIRVE + TABAN)

ENVELOPE = [
    dra_envelope(L)
    for L in range(N_LAYERS)
]

print("\nDRA envelope:")

for L, e in enumerate(ENVELOPE):
    print(
        f"  L{L:02d}: "
        f"E={e:.6f} | "
        f"base={100*IVME*e:.3f}%"
    )

# =============================================================================
# 5. SEARCH BANK
# =============================================================================

SEARCH_PAIRS = [
    (
        "The lantern is above the stone arch.",
        "The lantern is below the stone arch.",
    ),
    (
        "The kite is above the old tower.",
        "The kite is below the old tower.",
    ),
    (
        "The flag is above the wooden gate.",
        "The flag is below the wooden gate.",
    ),
    (
        "The bird is above the narrow bridge.",
        "The bird is below the narrow bridge.",
    ),
    (
        "The sign is above the glass doorway.",
        "The sign is below the glass doorway.",
    ),
    (
        "The lamp is above the metal shelf.",
        "The lamp is below the metal shelf.",
    ),
    (
        "The clock is above the brown cabinet.",
        "The clock is below the brown cabinet.",
    ),
    (
        "The balloon is above the green hill.",
        "The balloon is below the green hill.",
    ),
    (
        "The camera is above the small desk.",
        "The camera is below the small desk.",
    ),
    (
        "The painting is above the blue sofa.",
        "The painting is below the blue sofa.",
    ),
    (
        "The cloud is above the distant ridge.",
        "The cloud is below the distant ridge.",
    ),
    (
        "The marker is above the white board.",
        "The marker is below the white board.",
    ),
    (
        "The drone is above the concrete wall.",
        "The drone is below the concrete wall.",
    ),
    (
        "The shelf is above the storage box.",
        "The shelf is below the storage box.",
    ),
    (
        "The light is above the front window.",
        "The light is below the front window.",
    ),
    (
        "The symbol is above the horizontal line.",
        "The symbol is below the horizontal line.",
    ),
]

# =============================================================================
# 6. UNSEEN EVALUATION ITEMS
# =============================================================================

EVAL_ITEMS = [
    {
        "context":
            "A red marker and a blue marker are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The red marker is above the blue marker.",
        "neg":
            " The red marker is below the blue marker.",
    },
    {
        "context":
            "A bright lamp and a wooden table are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The bright lamp is above the wooden table.",
        "neg":
            " The bright lamp is below the wooden table.",
    },
    {
        "context":
            "A bird and a tree branch are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The bird is above the tree branch.",
        "neg":
            " The bird is below the tree branch.",
    },
    {
        "context":
            "A sign and a doorway are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The sign is above the doorway.",
        "neg":
            " The sign is below the doorway.",
    },
    {
        "context":
            "A shelf and a cabinet are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The shelf is above the cabinet.",
        "neg":
            " The shelf is below the cabinet.",
    },
    {
        "context":
            "A cloud and a hill are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The cloud is above the hill.",
        "neg":
            " The cloud is below the hill.",
    },
    {
        "context":
            "A clock and a desk are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The clock is above the desk.",
        "neg":
            " The clock is below the desk.",
    },
    {
        "context":
            "A framed picture and a sofa are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The framed picture is above the sofa.",
        "neg":
            " The framed picture is below the sofa.",
    },
    {
        "context":
            "A drone and a stone wall are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The drone is above the stone wall.",
        "neg":
            " The drone is below the stone wall.",
    },
    {
        "context":
            "A balloon and a house are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The balloon is above the house.",
        "neg":
            " The balloon is below the house.",
    },
    {
        "context":
            "A camera and a storage box are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The camera is above the storage box.",
        "neg":
            " The camera is below the storage box.",
    },
    {
        "context":
            "A silver symbol and a horizontal line are visible. "
            "Their vertical relation is unspecified.",
        "pos":
            " The silver symbol is above the horizontal line.",
        "neg":
            " The silver symbol is below the horizontal line.",
    },
]

NEUTRAL = [
    "A ceramic cup rests beside a closed notebook.",
    "A wooden chair stands near a plain wall.",
    "A glass bottle sits beside a small basket.",
    "A cotton towel rests on a clean counter.",
    "A metal toolbox is inside a quiet room.",
    "A paper folder lies beside a keyboard.",
    "A black umbrella stands near a cabinet.",
    "A silver spoon rests beside a plate.",
]

NEUTRAL_CONT = (
    " The scene contains ordinary stationary objects "
    "and no additional relation is specified."
)

# =============================================================================
# 7. PROMPT HELPERS
# =============================================================================

def chat_text(user_text):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM,
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

@torch.inference_mode()
def capture(text):
    enc = tokenizer(
        chat_text(text),
        return_tensors="pt",
    ).to(DEVICE)

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    pos = (
        int(
            enc["attention_mask"][0]
            .sum()
            .item()
        )
        - 1
    )

    hs = [
        out.hidden_states[L + 1][0, pos].float()
        for L in range(N_LAYERS)
    ]

    del out, enc

    return hs

# =============================================================================
# 8. BUILD SEARCH BANK
# =============================================================================

print("\n[3/14] Building SEARCH activation bank...")

POS_BANK = [
    []
    for _ in range(N_LAYERS)
]

NEG_BANK = [
    []
    for _ in range(N_LAYERS)
]

for i, (ptext, ntext) in enumerate(
    SEARCH_PAIRS,
    1,
):
    hp = capture(ptext)
    hn = capture(ntext)

    for L in range(N_LAYERS):
        POS_BANK[L].append(
            hp[L]
        )

        NEG_BANK[L].append(
            hn[L]
        )

    print(
        f"      pair "
        f"{i:02d}/"
        f"{len(SEARCH_PAIRS)}"
    )

POS_BANK = [
    torch.stack(x).float()
    for x in POS_BANK
]

NEG_BANK = [
    torch.stack(x).float()
    for x in NEG_BANK
]

# =============================================================================
# 9. FISHER + STATE FIELD
# =============================================================================

print(
    "\n[4/14] Building Fisher compasses "
    "+ state direction field..."
)

LOCAL_COMPASS = []
STATE_REF = []
STATE_DIR = []

for L in range(N_LAYERS):
    Hp = POS_BANK[L]
    Hn = NEG_BANK[L]

    mu_p = Hp.mean(dim=0)
    mu_n = Hn.mean(dim=0)

    mean_diff = (
        mu_p - mu_n
    )

    R = torch.cat(
        [
            Hp - mu_p,
            Hn - mu_n,
        ],
        dim=0,
    ).float()

    nu = max(
        int(R.shape[0]) - 2,
        1,
    )

    _, S, Vh = torch.linalg.svd(
        R,
        full_matrices=False,
    )

    eig = (
        S.square()
        / float(nu)
    )

    positive_eig = (
        eig[
            eig > FISHER_EPS
        ]
    )

    if positive_eig.numel():
        scale = torch.median(
            positive_eig
        )
    else:
        scale = torch.tensor(
            1.0,
            device=DEVICE,
            dtype=torch.float32,
        )

    lam = (
        FISHER_ALPHA
        * scale
    ).clamp_min(
        FISHER_EPS
    )

    V = Vh.T

    projection = (
        V.T
        @ mean_diff
    )

    correction = (
        (
            1.0
            / (
                eig + lam
            )
            -
            1.0
            / lam
        )
        * projection
    )

    fisher = (
        mean_diff
        / lam
        +
        V
        @ correction
    )

    fisher = (
        fisher
        / fisher.norm()
        .clamp_min(EPS)
    ).float().contiguous()

    refs = []
    dirs = []

    for i in range(
        len(SEARCH_PAIRS)
    ):
        ref = (
            0.5
            * (
                Hp[i]
                + Hn[i]
            )
        )

        ref = (
            ref
            / ref.norm()
            .clamp_min(EPS)
        )

        d = (
            Hp[i]
            - Hn[i]
        )

        d = (
            d
            / d.norm()
            .clamp_min(EPS)
        )

        if torch.dot(
            d,
            fisher,
        ) < 0:
            d = -d

        refs.append(ref)
        dirs.append(d)

    refs = (
        torch.stack(refs)
        .float()
        .contiguous()
    )

    dirs = (
        torch.stack(dirs)
        .float()
        .contiguous()
    )

    LOCAL_COMPASS.append(
        fisher
    )

    STATE_REF.append(
        refs
    )

    STATE_DIR.append(
        dirs
    )

    print(
        f"      L{L:02d} | "
        f"dir/Fisher="
        f"{(dirs @ fisher).mean().item():+.6f}"
    )

# =============================================================================
# 10. SEARCH-ONLY GAIN CALIBRATION
#
# For every SEARCH reference r_i:
#
#   q_i = sum_j softmax(cos(r_i,r_j)/T) cos(r_i,r_j)
#
# The distribution of q_i is frozen here.
#
# Runtime:
#
#   z = (q(h) - mean_SEARCH) / sd_SEARCH
#
#   g = exp(beta*z)
#
#   clamp(g, 0.5, 1.5)
#
# No runtime batch statistics are used.
# =============================================================================

print(
    "\n[5/14] Calibrating gain from SEARCH only..."
)

GAIN_MEAN = []
GAIN_SD = []

for L in range(N_LAYERS):
    refs = STATE_REF[L]

    sim = (
        refs
        @ refs.T
    )

    # Leave self-match out.
    sim = sim.clone()

    sim.fill_diagonal_(
        -1.0e9
    )

    w = torch.softmax(
        sim / STATE_TEMP,
        dim=-1,
    )

    q = (
        w * sim
    ).sum(dim=-1)

    q_mean = q.mean()

    q_sd = (
        q.std(
            unbiased=True
        )
        .clamp_min(
            GAIN_EPS
        )
    )

    GAIN_MEAN.append(
        q_mean.float()
    )

    GAIN_SD.append(
        q_sd.float()
    )

    print(
        f"      L{L:02d} | "
        f"q_mean={q_mean.item():+.6f} | "
        f"q_sd={q_sd.item():.6f}"
    )

# =============================================================================
# 11. FIXED SHUFFLE NULL MAPS
#
# Identity mappings are forbidden.
# Same permutation is used across layers within one null replicate so each
# null represents a coherent but wrong state->direction dictionary.
# =============================================================================

print(
    "\n[6/14] Building fixed routing-null permutations..."
)

rng = np.random.default_rng(
    SEED
)

N_BANK = len(
    SEARCH_PAIRS
)

SHUFFLE_PERMS = []

while len(
    SHUFFLE_PERMS
) < N_SHUFFLES:

    p = rng.permutation(
        N_BANK
    )

    if np.any(
        p
        == np.arange(N_BANK)
    ):
        continue

    if any(
        np.array_equal(
            p,
            old,
        )
        for old in SHUFFLE_PERMS
    ):
        continue

    SHUFFLE_PERMS.append(
        p.copy()
    )

for i, p in enumerate(
    SHUFFLE_PERMS
):
    print(
        f"      NULL {i:02d}: "
        f"{p.tolist()}"
    )

# =============================================================================
# 12. STATE DECODER
# =============================================================================

def state_decode(
    hidden_last,
    L,
    permutation=None,
):
    h = hidden_last.float()

    h_unit = (
        h
        / h.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(EPS)
    )

    refs = STATE_REF[L]

    dirs = STATE_DIR[L]

    sims = (
        h_unit
        @ refs.T
    )

    weights = torch.softmax(
        sims / STATE_TEMP,
        dim=-1,
    )

    # -------------------------------------------------------------
    # MATCHED:
    #
    #   w_i -> d_i
    #
    # SHUFFLED:
    #
    #   w_i -> d_perm(i)
    #
    # Similarity computation itself is identical.
    # Only correspondence is broken.
    # -------------------------------------------------------------

    if permutation is not None:
        perm_t = torch.as_tensor(
            permutation,
            device=DEVICE,
            dtype=torch.long,
        )

        dirs_used = dirs[
            perm_t
        ]
    else:
        dirs_used = dirs

    direction = (
        weights
        @ dirs_used
    )

    direction = (
        direction
        / direction.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(EPS)
    )

    confidence = (
        weights
        * sims
    ).sum(dim=-1)

    z = (
        confidence
        - GAIN_MEAN[L]
    ) / GAIN_SD[L]

    gain = torch.exp(
        GAIN_BETA
        * z
    )

    gain = torch.clamp(
        gain,
        GAIN_MIN,
        GAIN_MAX,
    )

    return (
        direction.float(),
        gain.float(),
        confidence.float(),
        z.float(),
    )

# =============================================================================
# 13. GAIN SANITY CHECK ON UNSEEN EVAL STATES
# =============================================================================

print(
    "\n[7/14] Gain sanity check on unseen evaluation states..."
)

EVAL_HIDDEN = []

for item in EVAL_ITEMS:
    EVAL_HIDDEN.append(
        capture(
            item["context"]
        )
    )

GAIN_SANITY = []

for L in range(N_LAYERS):
    vals = []

    for hs in EVAL_HIDDEN:
        _, g, q, z = state_decode(
            hs[L][None, :],
            L,
            permutation=None,
        )

        vals.append(
            (
                float(g.item()),
                float(q.item()),
                float(z.item()),
            )
        )

    gs = np.array(
        [x[0] for x in vals],
        dtype=np.float64,
    )

    zs = np.array(
        [x[2] for x in vals],
        dtype=np.float64,
    )

    GAIN_SANITY.append({
        "layer": L,
        "gain_mean":
            float(gs.mean()),
        "gain_sd":
            float(gs.std()),
        "gain_min":
            float(gs.min()),
        "gain_max":
            float(gs.max()),
        "z_mean":
            float(zs.mean()),
        "z_sd":
            float(zs.std()),
    })

    print(
        f"      L{L:02d} | "
        f"gain="
        f"{gs.mean():.4f}"
        f"±{gs.std():.4f} | "
        f"range="
        f"[{gs.min():.4f},"
        f"{gs.max():.4f}]"
    )

del EVAL_HIDDEN
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# 14. HOOK ENGINE
#
# Methods:
#
# LOCAL
# GAIN
# DIRECTION
# COMBINED
# SHUFFLE_DIRECTION_k
# SHUFFLE_COMBINED_k
# =============================================================================

def install_hooks(
    method,
    sign,
    dose_scale,
    telemetry,
    shuffle_idx=None,
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
                output,
            ):
                if isinstance(
                    output,
                    tuple,
                ):
                    old = output[0]
                    rest = output[1:]
                else:
                    old = output
                    rest = None

                B = old.shape[0]

                hlast = (
                    old[:, -1, :]
                    .float()
                )

                base_dose = (
                    IVME
                    * ENVELOPE[
                        layer_idx
                    ]
                    * dose_scale
                )

                local = (
                    LOCAL_COMPASS[
                        layer_idx
                    ][None, :]
                    .expand(
                        B,
                        -1,
                    )
                )

                if method == "LOCAL":
                    direction = local

                    gain = torch.ones(
                        B,
                        device=DEVICE,
                        dtype=torch.float32,
                    )

                    confidence = torch.zeros_like(
                        gain
                    )

                    z = torch.zeros_like(
                        gain
                    )

                elif method == "GAIN":
                    _, gain, confidence, z = (
                        state_decode(
                            hlast,
                            layer_idx,
                            permutation=None,
                        )
                    )

                    direction = local

                elif method == "DIRECTION":
                    direction, _, confidence, z = (
                        state_decode(
                            hlast,
                            layer_idx,
                            permutation=None,
                        )
                    )

                    gain = torch.ones(
                        B,
                        device=DEVICE,
                        dtype=torch.float32,
                    )

                elif method == "COMBINED":
                    (
                        direction,
                        gain,
                        confidence,
                        z,
                    ) = state_decode(
                        hlast,
                        layer_idx,
                        permutation=None,
                    )

                elif method == "SHUFFLE_DIRECTION":
                    if shuffle_idx is None:
                        raise RuntimeError(
                            "shuffle_idx required"
                        )

                    direction, _, confidence, z = (
                        state_decode(
                            hlast,
                            layer_idx,
                            permutation=(
                                SHUFFLE_PERMS[
                                    shuffle_idx
                                ]
                            ),
                        )
                    )

                    gain = torch.ones(
                        B,
                        device=DEVICE,
                        dtype=torch.float32,
                    )

                elif method == "SHUFFLE_COMBINED":
                    if shuffle_idx is None:
                        raise RuntimeError(
                            "shuffle_idx required"
                        )

                    (
                        direction,
                        gain,
                        confidence,
                        z,
                    ) = state_decode(
                        hlast,
                        layer_idx,
                        permutation=(
                            SHUFFLE_PERMS[
                                shuffle_idx
                            ]
                        ),
                    )

                else:
                    raise ValueError(
                        method
                    )

                direction = (
                    float(sign)
                    * direction
                ).contiguous()

                dose = (
                    base_dose
                    * gain
                ).float().contiguous()

                new = (
                    seasc_ext
                    .seasc_batch(
                        old,
                        direction,
                        dose,
                    )
                )

                old_last = (
                    old[:, -1, :]
                    .float()
                )

                new_last = (
                    new[:, -1, :]
                    .float()
                )

                delta = (
                    new_last
                    - old_last
                )

                norm0 = (
                    old_last
                    .norm(
                        dim=-1
                    )
                    .clamp_min(EPS)
                )

                realized = (
                    delta.norm(
                        dim=-1
                    )
                    / norm0
                )

                t = telemetry[
                    layer_idx
                ]

                t["requested"].extend(
                    dose.detach()
                    .cpu()
                    .tolist()
                )

                t["realized"].extend(
                    realized.detach()
                    .cpu()
                    .tolist()
                )

                t["gain"].extend(
                    gain.detach()
                    .cpu()
                    .tolist()
                )

                t["confidence"].extend(
                    confidence.detach()
                    .cpu()
                    .tolist()
                )

                t["z"].extend(
                    z.detach()
                    .cpu()
                    .tolist()
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

# =============================================================================
# 15. SCORING
# =============================================================================

def build_sequence(
    context,
    continuation,
):
    prompt = chat_text(
        context
    )

    p = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    c = tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False,
    )

    ids = torch.cat(
        [
            p["input_ids"],
            c["input_ids"],
        ],
        dim=1,
    ).to(DEVICE)

    mask = torch.ones_like(
        ids,
        device=DEVICE,
    )

    return (
        ids,
        mask,
        p["input_ids"].shape[1],
    )

def blank_telemetry():
    return [
        {
            "requested": [],
            "realized": [],
            "gain": [],
            "confidence": [],
            "z": [],
        }
        for _ in range(N_LAYERS)
    ]

@torch.inference_mode()
def continuation_logp(
    context,
    continuation,
    method=None,
    sign=0,
    scale=0.0,
    shuffle_idx=None,
):
    ids, mask, plen = (
        build_sequence(
            context,
            continuation,
        )
    )

    telemetry = (
        blank_telemetry()
    )

    handles = []

    if method is not None:
        handles = install_hooks(
            method,
            sign,
            scale,
            telemetry,
            shuffle_idx=shuffle_idx,
        )

    try:
        out = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    logits = (
        out.logits.float()
    )

    target = (
        ids[:, plen:]
    )

    pred = (
        logits[
            :,
            plen - 1:-1,
            :
        ]
    )

    lp = F.log_softmax(
        pred,
        dim=-1,
    )

    token_lp = (
        lp.gather(
            -1,
            target.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    score = float(
        token_lp.mean().item()
    )

    del out
    del logits
    del target
    del pred
    del lp
    del token_lp
    del ids
    del mask

    return (
        score,
        telemetry,
    )

# =============================================================================
# 16. ONE CAUSAL ARM
# =============================================================================

def run_causal_arm(
    method,
    scale,
    shuffle_idx=None,
):
    item_csym = []

    requested = []
    realized = []
    gains = []
    zs = []

    plus_margins = []
    minus_margins = []

    for item in EVAL_ITEMS:
        p_pos, t1 = (
            continuation_logp(
                item["context"],
                item["pos"],
                method=method,
                sign=+1,
                scale=scale,
                shuffle_idx=shuffle_idx,
            )
        )

        p_neg, t2 = (
            continuation_logp(
                item["context"],
                item["neg"],
                method=method,
                sign=+1,
                scale=scale,
                shuffle_idx=shuffle_idx,
            )
        )

        n_pos, t3 = (
            continuation_logp(
                item["context"],
                item["pos"],
                method=method,
                sign=-1,
                scale=scale,
                shuffle_idx=shuffle_idx,
            )
        )

        n_neg, t4 = (
            continuation_logp(
                item["context"],
                item["neg"],
                method=method,
                sign=-1,
                scale=scale,
                shuffle_idx=shuffle_idx,
            )
        )

        pm = (
            p_pos - p_neg
        )

        nm = (
            n_pos - n_neg
        )

        c = (
            0.5
            * (
                pm - nm
            )
        )

        plus_margins.append(
            pm
        )

        minus_margins.append(
            nm
        )

        item_csym.append(
            c
        )

        for tel in [
            t1,
            t2,
            t3,
            t4,
        ]:
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

                gains.extend(
                    tel[L][
                        "gain"
                    ]
                )

                zs.extend(
                    tel[L][
                        "z"
                    ]
                )

    energy = float(
        np.sum(
            np.square(
                realized
            )
        )
    )

    return {
        "method":
            method,

        "scale":
            scale,

        "shuffle_idx":
            (
                -1
                if shuffle_idx is None
                else shuffle_idx
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

        "Csym":
            float(
                np.mean(
                    item_csym
                )
            ),

        "Csym_median":
            float(
                np.median(
                    item_csym
                )
            ),

        "positive_items":
            int(
                np.sum(
                    np.asarray(
                        item_csym
                    )
                    > 0
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

        "energy":
            energy,

        "gain_mean":
            float(
                np.mean(
                    gains
                )
            ),

        "gain_sd":
            float(
                np.std(
                    gains
                )
            ),

        "gain_min":
            float(
                np.min(
                    gains
                )
            ),

        "gain_max":
            float(
                np.max(
                    gains
                )
            ),

        "z_mean":
            float(
                np.mean(
                    zs
                )
            ),

        "z_sd":
            float(
                np.std(
                    zs
                )
            ),

        "item_csym":
            [
                float(x)
                for x in item_csym
            ],
    }

# =============================================================================
# 17. REAL ARMS
# =============================================================================

print(
    "\n[8/14] Running real causal arms..."
)

REAL_METHODS = [
    "LOCAL",
    "GAIN",
    "DIRECTION",
    "COMBINED",
]

REAL_ROWS = []

for scale in DOSE_SCALES:
    for method in REAL_METHODS:
        r = run_causal_arm(
            method,
            scale,
        )

        REAL_ROWS.append(
            r
        )

        print(
            f"  {method:<10} "
            f"scale={scale:.2f} | "
            f"Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/"
            f"{len(EVAL_ITEMS)} | "
            f"gain="
            f"{r['gain_mean']:.4f}"
            f"±{r['gain_sd']:.4f} | "
            f"real="
            f"{100*r['realized_mean']:.4f}% | "
            f"E={r['energy']:.6f}"
        )

# =============================================================================
# 18. MULTI-SHUFFLE ROUTING NULL
# =============================================================================

print(
    "\n[9/14] Running multi-shuffle routing null..."
)

SHUFFLE_ROWS = []

for scale in DOSE_SCALES:
    for k in range(
        N_SHUFFLES
    ):
        rd = run_causal_arm(
            "SHUFFLE_DIRECTION",
            scale,
            shuffle_idx=k,
        )

        rc = run_causal_arm(
            "SHUFFLE_COMBINED",
            scale,
            shuffle_idx=k,
        )

        SHUFFLE_ROWS.extend(
            [
                rd,
                rc,
            ]
        )

        print(
            f"  scale={scale:.2f} "
            f"null={k:02d} | "
            f"DIR={rd['Csym']:+.6f} | "
            f"COMB={rc['Csym']:+.6f}"
        )

# =============================================================================
# 19. ROUTING NULL STATISTICS
# =============================================================================

print(
    "\n[10/14] Routing-null statistics..."
)

NULL_STATS = []

def real_row(
    method,
    scale,
):
    for r in REAL_ROWS:
        if (
            r["method"] == method
            and
            r["scale"] == scale
        ):
            return r

    raise KeyError(
        (method, scale)
    )

for scale in DOSE_SCALES:
    for real_method, null_method in [
        (
            "DIRECTION",
            "SHUFFLE_DIRECTION",
        ),
        (
            "COMBINED",
            "SHUFFLE_COMBINED",
        ),
    ]:
        real = real_row(
            real_method,
            scale,
        )

        null = [
            r["Csym"]
            for r in SHUFFLE_ROWS
            if (
                r["method"]
                == null_method
                and
                r["scale"]
                == scale
            )
        ]

        null = np.asarray(
            null,
            dtype=np.float64,
        )

        ge = int(
            np.sum(
                null
                >= real["Csym"]
            )
        )

        empirical_p = (
            ge + 1
        ) / (
            len(null) + 1
        )

        null_mean = float(
            null.mean()
        )

        null_sd = float(
            null.std(
                ddof=1
            )
        )

        margin = (
            real["Csym"]
            - null_mean
        )

        z = (
            margin
            / max(
                null_sd,
                EPS,
            )
        )

        row = {
            "scale":
                scale,

            "real_method":
                real_method,

            "real_Csym":
                real["Csym"],

            "null_mean":
                null_mean,

            "null_sd":
                null_sd,

            "real_minus_null":
                margin,

            "null_ge_real":
                ge,

            "n_null":
                len(null),

            "empirical_p":
                empirical_p,

            "descriptive_z":
                z,
        }

        NULL_STATS.append(
            row
        )

        print(
            f"  {real_method:<10} "
            f"scale={scale:.2f} | "
            f"real={real['Csym']:+.6f} | "
            f"null={null_mean:+.6f}"
            f"±{null_sd:.6f} | "
            f"Δ={margin:+.6f} | "
            f"p={empirical_p:.6f}"
        )

# =============================================================================
# 20. GAIN DIFFERENTIAL
# =============================================================================

print(
    "\n[11/14] Gain and combined differentials..."
)

DIFF_ROWS = []

for scale in DOSE_SCALES:
    local = real_row(
        "LOCAL",
        scale,
    )

    gain = real_row(
        "GAIN",
        scale,
    )

    direction = real_row(
        "DIRECTION",
        scale,
    )

    combined = real_row(
        "COMBINED",
        scale,
    )

    row = {
        "scale":
            scale,

        "gain_minus_local":
            gain["Csym"]
            - local["Csym"],

        "direction_minus_local":
            direction["Csym"]
            - local["Csym"],

        "combined_minus_local":
            combined["Csym"]
            - local["Csym"],

        "combined_minus_direction":
            combined["Csym"]
            - direction["Csym"],

        "combined_minus_gain":
            combined["Csym"]
            - gain["Csym"],

        "gain_energy_ratio":
            gain["energy"]
            / max(
                local["energy"],
                EPS,
            ),

        "combined_energy_ratio":
            combined["energy"]
            / max(
                direction["energy"],
                EPS,
            ),
    }

    DIFF_ROWS.append(
        row
    )

    print(
        f"  scale={scale:.2f} | "
        f"GAIN-LOCAL="
        f"{row['gain_minus_local']:+.6f} | "
        f"DIR-LOCAL="
        f"{row['direction_minus_local']:+.6f} | "
        f"COMB-DIR="
        f"{row['combined_minus_direction']:+.6f} | "
        f"E_GAIN/LOCAL="
        f"{row['gain_energy_ratio']:.4f} | "
        f"E_COMB/DIR="
        f"{row['combined_energy_ratio']:.4f}"
    )

# =============================================================================
# 21. NEUTRAL COLLATERAL
# =============================================================================

print(
    "\n[12/14] Neutral collateral..."
)

@torch.inference_mode()
def neutral_metrics(
    text,
    method,
    scale,
):
    ids, mask, plen = (
        build_sequence(
            text,
            NEUTRAL_CONT,
        )
    )

    out0 = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    )

    base_logits = (
        out0.logits.float()[
            :,
            plen - 1:-1,
            :
        ]
    )

    target = (
        ids[:, plen:]
    )

    base_lp = F.log_softmax(
        base_logits,
        dim=-1,
    )

    base_nll = (
        -base_lp
        .gather(
            -1,
            target.unsqueeze(-1),
        )
        .squeeze(-1)
        .mean()
    )

    telemetry = (
        blank_telemetry()
    )

    handles = install_hooks(
        method,
        +1,
        scale,
        telemetry,
    )

    try:
        out1 = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    steer_logits = (
        out1.logits.float()[
            :,
            plen - 1:-1,
            :
        ]
    )

    steer_lp = F.log_softmax(
        steer_logits,
        dim=-1,
    )

    steer_nll = (
        -steer_lp
        .gather(
            -1,
            target.unsqueeze(-1),
        )
        .squeeze(-1)
        .mean()
    )

    p = (
        base_lp.exp()
    )

    kl = (
        p
        * (
            base_lp
            - steer_lp
        )
    ).sum(
        dim=-1
    ).mean()

    dnll = float(
        (
            steer_nll
            - base_nll
        ).item()
    )

    klv = float(
        kl.item()
    )

    del out0
    del out1
    del ids
    del mask
    del target
    del base_logits
    del steer_logits
    del base_lp
    del steer_lp

    return (
        dnll,
        klv,
    )

COLLATERAL_ROWS = []

for scale in DOSE_SCALES:
    for method in REAL_METHODS:
        dnlls = []
        kls = []

        for text in NEUTRAL:
            dnll, kl = (
                neutral_metrics(
                    text,
                    method,
                    scale,
                )
            )

            dnlls.append(
                dnll
            )

            kls.append(
                kl
            )

        row = {
            "method":
                method,

            "scale":
                scale,

            "delta_nll":
                float(
                    np.mean(
                        dnlls
                    )
                ),

            "kl":
                float(
                    np.mean(
                        kls
                    )
                ),
        }

        COLLATERAL_ROWS.append(
            row
        )

        print(
            f"  {method:<10} "
            f"scale={scale:.2f} | "
            f"ΔNLL="
            f"{row['delta_nll']:+.6f} | "
            f"KL="
            f"{row['kl']:.6f}"
        )

# =============================================================================
# 22. DIAGNOSTIC CONTRACT
# =============================================================================

print(
    "\n[13/14] Diagnostic contract..."
)

FLAGS = []

gain_positive = sum(
    x["gain_minus_local"] > 0
    for x in DIFF_ROWS
)

direction_positive = sum(
    x["direction_minus_local"] > 0
    for x in DIFF_ROWS
)

combined_direction_positive = sum(
    x["combined_minus_direction"] > 0
    for x in DIFF_ROWS
)

matched_direction_positive = sum(
    (
        x["real_method"]
        == "DIRECTION"
        and
        x["real_minus_null"] > 0
    )
    for x in NULL_STATS
)

matched_combined_positive = sum(
    (
        x["real_method"]
        == "COMBINED"
        and
        x["real_minus_null"] > 0
    )
    for x in NULL_STATS
)

if gain_positive >= 2:
    FLAGS.append(
        "CALIBRATED_STATE_GAIN_SIGNAL"
    )

if direction_positive >= 2:
    FLAGS.append(
        "DIRECTION_FAMILY_ADVANTAGE_REPLICATED"
    )

if combined_direction_positive >= 2:
    FLAGS.append(
        "GAIN_ADDS_TO_STATE_DIRECTION"
    )

if matched_direction_positive >= 2:
    FLAGS.append(
        "MATCHED_ROUTING_ABOVE_NULL_MEAN"
    )

if matched_combined_positive >= 2:
    FLAGS.append(
        "MATCHED_COMBINED_ABOVE_NULL_MEAN"
    )

# Stronger routing claim:
# all three doses above null mean AND no null replicate reaches real
# at >=2 doses.
strong_routing = 0

for method in [
    "DIRECTION",
    "COMBINED",
]:
    rows = [
        x
        for x in NULL_STATS
        if x["real_method"]
        == method
    ]

    all_above = all(
        x["real_minus_null"] > 0
        for x in rows
    )

    zero_exceed = sum(
        x["null_ge_real"] == 0
        for x in rows
    )

    if (
        all_above
        and
        zero_exceed >= 2
    ):
        strong_routing += 1

if strong_routing > 0:
    FLAGS.append(
        "STATE_SPECIFIC_ROUTING_STRONG_SIGNAL"
    )

if not FLAGS:
    FLAGS.append(
        "NO_STATE_SPECIFIC_ADVANTAGE_ESTABLISHED"
    )

for flag in FLAGS:
    print(
        " -",
        flag
    )

# =============================================================================
# 23. SAVE
# =============================================================================

def csv_safe(rows):
    out = []

    for r in rows:
        out.append({
            k: v
            for k, v in r.items()
            if not isinstance(
                v,
                (
                    list,
                    dict,
                    tuple,
                ),
            )
        })

    return out

def save_csv(
    path,
    rows,
):
    rows = csv_safe(
        rows
    )

    if not rows:
        return

    keys = sorted(
        set().union(
            *[
                set(r.keys())
                for r in rows
            ]
        )
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

save_csv(
    ROOT / "REAL_ARMS.csv",
    REAL_ROWS,
)

save_csv(
    ROOT / "SHUFFLE_NULLS.csv",
    SHUFFLE_ROWS,
)

save_csv(
    ROOT / "ROUTING_NULL_STATS.csv",
    NULL_STATS,
)

save_csv(
    ROOT / "DIFFERENTIALS.csv",
    DIFF_ROWS,
)

save_csv(
    ROOT / "GAIN_SANITY.csv",
    GAIN_SANITY,
)

save_csv(
    ROOT / "COLLATERAL.csv",
    COLLATERAL_ROWS,
)

np.savez_compressed(
    ROOT / "TEST160_STATE_FIELD.npz",
    **{
        **{
            f"LOCAL_COMPASS_L{L}":
                LOCAL_COMPASS[L]
                .detach()
                .cpu()
                .numpy()
            for L in range(
                N_LAYERS
            )
        },

        **{
            f"STATE_REF_L{L}":
                STATE_REF[L]
                .detach()
                .cpu()
                .numpy()
            for L in range(
                N_LAYERS
            )
        },

        **{
            f"STATE_DIR_L{L}":
                STATE_DIR[L]
                .detach()
                .cpu()
                .numpy()
            for L in range(
                N_LAYERS
            )
        },

        "gain_mean":
            np.asarray(
                [
                    x.item()
                    for x in GAIN_MEAN
                ],
                dtype=np.float32,
            ),

        "gain_sd":
            np.asarray(
                [
                    x.item()
                    for x in GAIN_SD
                ],
                dtype=np.float32,
            ),

        "shuffle_permutations":
            np.asarray(
                SHUFFLE_PERMS,
                dtype=np.int64,
            ),
    }
)

REPORT = {
    "test":
        "TEST 160",

    "title":
        "STATE-SPECIFIC ROUTING + CALIBRATED GAIN",

    "model":
        MODEL_ID,

    "gpu":
        torch.cuda.get_device_name(0),

    "motor": {
        "layers":
            "L0-L19",

        "ivme":
            IVME,

        "sonum":
            SONUM,

        "zirve":
            ZIRVE,

        "taban":
            TABAN,

        "law":
            "frozen-norm direct dose",

        "training":
            False,
    },

    "gain": {
        "calibration":
            "SEARCH-only leave-self-out confidence distribution",

        "beta":
            GAIN_BETA,

        "min":
            GAIN_MIN,

        "max":
            GAIN_MAX,

        "runtime_batch_statistics":
            False,
    },

    "routing_null": {
        "n":
            N_SHUFFLES,

        "identity_fixed_points":
            False,

        "same_permutation_across_layers":
            True,
    },

    "real_arms":
        REAL_ROWS,

    "shuffle_rows":
        SHUFFLE_ROWS,

    "null_stats":
        NULL_STATS,

    "differentials":
        DIFF_ROWS,

    "gain_sanity":
        GAIN_SANITY,

    "collateral":
        COLLATERAL_ROWS,

    "flags":
        FLAGS,

    "interpretation_contract": [
        "TEST160 corrects the batch-size-one gain degeneracy in TEST159.",
        "Runtime gain uses only SEARCH-calibrated statistics.",
        "Matched routing is compared with eight fixed deranged routing maps.",
        "A direction-family advantage does not by itself establish state-specific routing.",
        "State-specific routing requires matched correspondence to exceed shuffled correspondence.",
        "Empirical p values with eight nulls have minimum possible value 1/9.",
        "Descriptive Z is not treated as a formal significance test.",
        "Gain changes physical intervention energy and must be interpreted together with energy telemetry.",
        "Results are limited to this model, semantic axis, prompt family, dose range, and state decoder.",
    ],
}

with open(
    ROOT / "REPORT.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        REPORT,
        f,
        indent=2,
        ensure_ascii=False,
    )

# =============================================================================
# 24. FINAL REPORT
# =============================================================================

print(
    "\n[14/14] COMPLETE"
)

print("=" * 150)
print("TEST 160 — FINAL SUMMARY")
print("=" * 150)

print("\nREAL ARMS")

for scale in DOSE_SCALES:
    print(
        f"\nDOSE SCALE = "
        f"{scale:.2f}"
    )

    for method in REAL_METHODS:
        r = real_row(
            method,
            scale,
        )

        print(
            f"  {method:<10} | "
            f"Csym="
            f"{r['Csym']:+.6f} | "
            f"P+="
            f"{r['positive_items']:02d}/"
            f"{len(EVAL_ITEMS)} | "
            f"gain="
            f"{r['gain_mean']:.4f}"
            f"±{r['gain_sd']:.4f} | "
            f"E="
            f"{r['energy']:.6f}"
        )

print(
    "\nMATCHED ROUTING VS SHUFFLED NULL"
)

for r in NULL_STATS:
    print(
        f"  {r['real_method']:<10} "
        f"scale="
        f"{r['scale']:.2f} | "
        f"real="
        f"{r['real_Csym']:+.6f} | "
        f"null="
        f"{r['null_mean']:+.6f}"
        f"±{r['null_sd']:.6f} | "
        f"Δ="
        f"{r['real_minus_null']:+.6f} | "
        f"p="
        f"{r['empirical_p']:.6f}"
    )

print(
    "\nGAIN DIFFERENTIALS"
)

for r in DIFF_ROWS:
    print(
        f"  scale="
        f"{r['scale']:.2f} | "
        f"GAIN-LOCAL="
        f"{r['gain_minus_local']:+.6f} | "
        f"DIR-LOCAL="
        f"{r['direction_minus_local']:+.6f} | "
        f"COMB-DIR="
        f"{r['combined_minus_direction']:+.6f} | "
        f"E_GAIN/LOCAL="
        f"{r['gain_energy_ratio']:.4f} | "
        f"E_COMB/DIR="
        f"{r['combined_energy_ratio']:.4f}"
    )

print(
    "\nDIAGNOSTIC FLAGS"
)

for flag in FLAGS:
    print(
        " ",
        flag
    )

print(
    "\nINTERPRETATION RULE"
)

print(
    "DIRECTION > LOCAL "
    "tests direction-family advantage."
)

print(
    "MATCHED DIRECTION > SHUFFLE distribution "
    "tests state-specific routing."
)

print(
    "GAIN > LOCAL "
    "tests calibrated state-dependent magnitude."
)

print(
    "COMBINED > DIRECTION "
    "tests whether calibrated gain adds information "
    "beyond direction selection."
)

print(
    "\nSaved:"
)

for name in [
    "REAL_ARMS.csv",
    "SHUFFLE_NULLS.csv",
    "ROUTING_NULL_STATS.csv",
    "DIFFERENTIALS.csv",
    "GAIN_SANITY.csv",
    "COLLATERAL.csv",
    "TEST160_STATE_FIELD.npz",
    "REPORT.json",
]:
    print(
        " ",
        ROOT / name
    )

print("=" * 150)
print("TEST 160 COMPLETE")
print("=" * 150)
