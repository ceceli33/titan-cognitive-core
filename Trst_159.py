# =============================================================================
# TEST 159 — LOCAL STATE-CONDITIONED STEERING DIFFERENTIAL
# AKBASCORE 3.1 / SEASC
#
# GLOBAL vs LAYER-LOCAL vs STATE-GAIN vs STATE-DIRECTION
# vs STATE-GAIN+DIRECTION vs SHUFFLED-STATE
#
# Qwen/Qwen2.5-7B-Instruct · BF16 · A100
#
# QUESTION
# --------
# TEST158 showed:
#
#   matched context transport > mismatched context transport
#
# across every tested axis × segment.
#
# TEST159 asks causally:
#
#   Is successful steering obtained by:
#
#   H1 — using the correct layer-local semantic coordinate?
#   H2 — adapting only steering magnitude to current hidden state?
#   H3 — adapting semantic direction itself to current hidden state?
#   H4 — adapting BOTH direction and magnitude?
#
# Core equations
# --------------
#
# LOCAL:
#
#   dh_L = s * rho_L * ||h_L|| * A_L
#
# STATE-GAIN:
#
#   dh_L = s * rho_L * g_L(h_L) * ||h_L|| * A_L
#
# STATE-DIRECTION:
#
#   A_L(h) = normalize(sum_i w_i(h) d_L,i)
#
#   dh_L = s * rho_L * ||h_L|| * A_L(h)
#
# COMBINED:
#
#   dh_L = s * rho_L * g_L(h_L) * ||h_L|| * A_L(h)
#
# s ∈ {-1,+1}
#
# STRICT CONTRACT
# ---------------
# Frozen model.
# No training.
# No optimizer.
# No gradients.
# No JVP/VJP.
# No LoRA.
# No Retention Guard.
#
# State bank is SEARCH-only.
# Evaluation prompts are unseen by state-bank construction.
#
# =============================================================================

import os
import sys
import gc
import math
import json
import csv
import time
import shutil
import random
import subprocess
from pathlib import Path

print("=" * 150)
print("TEST 159 — LOCAL STATE-CONDITIONED STEERING DIFFERENTIAL")
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

SEED = 159
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
GAIN_MIN = 0.50
GAIN_MAX = 1.50
EPS = 1e-10

# Cheap falsification first.
# Full AkbasCore envelope is retained, but steering strength is swept.
DOSE_SCALES = [0.25, 0.50, 1.00]

METHODS = [
    "GLOBAL_STATIC",
    "LAYER_LOCAL",
    "STATE_GAIN",
    "STATE_DIRECTION",
    "COMBINED",
    "SHUFFLED_STATE",
]

SYSTEM = (
    "You are a concise reasoning assistant. "
    "Use only the information in the prompt."
)

ROOT = Path("/content/TEST159_LOCAL_STATE_STEERING")
ROOT.mkdir(parents=True, exist_ok=True)

print("GPU            :", torch.cuda.get_device_name(0))
print("Model          :", MODEL_ID)
print("Steered layers : L0-L19")
print("Dose scales    :", DOSE_SCALES)
print("Methods        :", METHODS)

# =============================================================================
# 2. CUDA — AKBASCORE FROZEN-NORM INJECTION
#
# Same physical law:
#
#   h' = h + rho * ||h|| * A
#
# Direction and effective rho may vary per sequence row.
# This extension therefore accepts:
#
#   actuator [B,H]
#   relative_dose [B]
#
# allowing state-conditioned decisions without changing the physical motor.
# =============================================================================

BUILD_DIR = "/tmp/akbascore_test159"

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
    m.def("seasc_batch", &seasc_batch, "TEST159 SEASC batch frozen-norm injection");
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
        static_cast<long long>(vec) * static_cast<long long>(H);

    const long long abase =
        static_cast<long long>(b) * static_cast<long long>(H);

    float local_sum = 0.0f;

    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        float x = static_cast<float>(h[base + j]);
        local_sum += x * x;
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float norm0 = sqrtf(fmaxf(shared[0], 1.0e-20f));
    const float scale = dose[b] * norm0;

    __syncthreads();

    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        float old_value = static_cast<float>(h[base + j]);
        float new_value =
            old_value
            + scale * actuator[abase + j];

        h[base + j] = static_cast<scalar_t>(new_value);
    }
}

torch::Tensor seasc_batch_cuda(
    torch::Tensor hidden,
    torch::Tensor actuator,
    torch::Tensor dose
) {
    auto out = hidden.contiguous().clone();

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
    const size_t shared_bytes = THREADS * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        out.scalar_type(),
        "test159_seasc_batch",
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

print("\n[1/13] Compiling AkbasCore batch SEASC kernel...")

seasc_ext = load_inline(
    name="akbascore_test159_cuda",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    build_directory=BUILD_DIR,
    verbose=False,
)

print("      CUDA kernel: OK")

# =============================================================================
# 3. MODEL
# =============================================================================

print("\n[2/13] Loading frozen model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    trust_remote_code=True,
)

model.eval()

for p in model.parameters():
    p.requires_grad_(False)

layers = model.model.layers
HIDDEN = model.config.hidden_size

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
        f"E={e:.6f} "
        f"base-dose={100*IVME*e:.3f}%"
    )

# =============================================================================
# 5. SEARCH-ONLY MATCHED STATE BANK
#
# Evaluation sentences below do NOT occur here.
#
# Each pair defines:
#
#   reference state r_L,i
#   local semantic difference d_L,i
#
# r is deliberately label-symmetric:
#
#   r = normalize((h_pos + h_neg)/2)
#
# Therefore nearest-state selection cannot obtain the desired steering sign
# from POS/NEG identity.
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
# 6. EVALUATION — AMBIGUOUS / LABEL-NEUTRAL PROMPTS
#
# The prompt does not state ABOVE or BELOW.
#
# Candidate scoring asks:
#
#   + steering -> ABOVE candidate
#   - steering -> BELOW candidate
#
# Symmetric causal effect:
#
#   Csym = 0.5 * [
#       margin(+ steering)
#       -
#       margin(- steering)
#   ]
#
# where:
#
#   margin = logP(ABOVE completion) - logP(BELOW completion)
#
# Thus baseline task correctness cannot saturate the metric.
# =============================================================================

EVAL_ITEMS = [
    {
        "context":
            "A red marker and a blue marker are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The red marker is above the blue marker.",
        "neg": " The red marker is below the blue marker.",
    },
    {
        "context":
            "A bright lamp and a wooden table are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The bright lamp is above the wooden table.",
        "neg": " The bright lamp is below the wooden table.",
    },
    {
        "context":
            "A bird and a tree branch are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The bird is above the tree branch.",
        "neg": " The bird is below the tree branch.",
    },
    {
        "context":
            "A sign and a doorway are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The sign is above the doorway.",
        "neg": " The sign is below the doorway.",
    },
    {
        "context":
            "A shelf and a cabinet are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The shelf is above the cabinet.",
        "neg": " The shelf is below the cabinet.",
    },
    {
        "context":
            "A cloud and a hill are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The cloud is above the hill.",
        "neg": " The cloud is below the hill.",
    },
    {
        "context":
            "A clock and a desk are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The clock is above the desk.",
        "neg": " The clock is below the desk.",
    },
    {
        "context":
            "A framed picture and a sofa are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The framed picture is above the sofa.",
        "neg": " The framed picture is below the sofa.",
    },
    {
        "context":
            "A drone and a stone wall are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The drone is above the stone wall.",
        "neg": " The drone is below the stone wall.",
    },
    {
        "context":
            "A balloon and a house are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The balloon is above the house.",
        "neg": " The balloon is below the house.",
    },
    {
        "context":
            "A camera and a storage box are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The camera is above the storage box.",
        "neg": " The camera is below the storage box.",
    },
    {
        "context":
            "A silver symbol and a horizontal line are visible. "
            "Their vertical relation is unspecified.",
        "pos": " The silver symbol is above the horizontal line.",
        "neg": " The silver symbol is below the horizontal line.",
    },
]

# Neutral collateral prompts.
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

# =============================================================================
# 7. PROMPT / ACTIVATION HELPERS
# =============================================================================

def chat_text(user_text):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
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

    pos = int(
        enc["attention_mask"][0].sum().item()
    ) - 1

    hs = [
        out.hidden_states[L + 1][0, pos].float()
        for L in range(N_LAYERS)
    ]

    del out, enc

    return hs

# =============================================================================
# 8. BUILD SEARCH STATE BANK + SUBSPACE-FISHER LOCAL COMPASSES
# =============================================================================

print("\n[3/13] Building SEARCH-only state bank...")

POS_BANK = [[] for _ in range(N_LAYERS)]
NEG_BANK = [[] for _ in range(N_LAYERS)]

for i, (pos_text, neg_text) in enumerate(SEARCH_PAIRS, 1):
    hp = capture(pos_text)
    hn = capture(neg_text)

    for L in range(N_LAYERS):
        POS_BANK[L].append(hp[L])
        NEG_BANK[L].append(hn[L])

    print(f"      pair {i:02d}/{len(SEARCH_PAIRS)}")

POS_BANK = [
    torch.stack(x).float()
    for x in POS_BANK
]

NEG_BANK = [
    torch.stack(x).float()
    for x in NEG_BANK
]

LOCAL_COMPASS = []
STATE_REF = []
STATE_DIR = []

print("\n[4/13] Building local Fisher compasses + state-conditioned directions...")

for L in range(N_LAYERS):
    Hp = POS_BANK[L]
    Hn = NEG_BANK[L]

    mu_p = Hp.mean(dim=0)
    mu_n = Hn.mean(dim=0)

    mean_diff = mu_p - mu_n

    Rp = Hp - mu_p
    Rn = Hn - mu_n

    R = torch.cat([Rp, Rn], dim=0).float()

    nu = max(int(R.shape[0]) - 2, 1)

    U, S, Vh = torch.linalg.svd(
        R,
        full_matrices=False,
    )

    eig = S.square() / float(nu)

    positive_eig = eig[eig > FISHER_EPS]

    if positive_eig.numel():
        scale = torch.median(positive_eig)
    else:
        scale = torch.tensor(
            1.0,
            device=DEVICE,
            dtype=torch.float32,
        )

    lam = (
        FISHER_ALPHA * scale
    ).clamp_min(FISHER_EPS)

    V = Vh.T

    projection = V.T @ mean_diff

    correction = (
        (
            1.0 / (eig + lam)
            - 1.0 / lam
        )
        * projection
    )

    fisher = (
        mean_diff / lam
        + V @ correction
    )

    fisher = (
        fisher
        / fisher.norm().clamp_min(EPS)
    ).float()

    LOCAL_COMPASS.append(
        fisher.contiguous()
    )

    # -------------------------------------------------------------
    # State reference:
    # label-symmetric midpoint of each matched pair.
    #
    # State direction:
    # matched POS-NEG difference for each SEARCH context.
    #
    # Align every local sample direction to Fisher compass polarity.
    # -------------------------------------------------------------

    refs = []
    dirs = []

    for i in range(len(SEARCH_PAIRS)):
        ref = 0.5 * (
            Hp[i] + Hn[i]
        )

        ref = (
            ref
            / ref.norm().clamp_min(EPS)
        )

        d = Hp[i] - Hn[i]

        d = (
            d
            / d.norm().clamp_min(EPS)
        )

        if torch.dot(d, fisher) < 0:
            d = -d

        refs.append(ref)
        dirs.append(d)

    refs = torch.stack(refs).float()
    dirs = torch.stack(dirs).float()

    STATE_REF.append(refs)
    STATE_DIR.append(dirs)

    coherence = float(
        (dirs @ fisher).mean().item()
    )

    print(
        f"      L{L:02d} | "
        f"Fisher norm={fisher.norm().item():.6f} | "
        f"state-dir coherence={coherence:+.6f}"
    )

GLOBAL_COMPASS = LOCAL_COMPASS[0]

del POS_BANK, NEG_BANK
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# 9. STATE DECODER
#
# For current hidden state h:
#
# similarities:
#
#   s_i = cos(h, r_i)
#
# weights:
#
#   w_i = softmax(s_i / T)
#
# direction:
#
#   A(h) = normalize(sum_i w_i d_i)
#
# confidence:
#
#   q = sum_i w_i s_i
#
# gain:
#
#   raw g = exp((q - mean(q_batch))/temperature)
#
# then clamp and renormalize mean(g)=1.
#
# The final normalization is important:
# STATE-GAIN cannot win simply by receiving more average dose.
# =============================================================================

def state_decode(hidden_last, L, shuffled=False):
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

    sims = h_unit @ refs.T

    if shuffled:
        # Deterministic wrong context correspondence.
        dirs = torch.roll(
            dirs,
            shifts=5,
            dims=0,
        )

    weights = torch.softmax(
        sims / STATE_TEMP,
        dim=-1,
    )

    direction = weights @ dirs

    direction = (
        direction
        / direction.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(EPS)
    )

    confidence = (
        weights * sims
    ).sum(dim=-1)

    # Relative confidence only.
    centered = (
        confidence
        - confidence.mean()
    )

    gain = torch.exp(
        centered / 0.10
    )

    gain = torch.clamp(
        gain,
        GAIN_MIN,
        GAIN_MAX,
    )

    gain = (
        gain
        / gain.mean().clamp_min(EPS)
    )

    return (
        direction.float(),
        gain.float(),
        confidence.float(),
        weights.float(),
    )

# =============================================================================
# 10. HOOK ENGINE
# =============================================================================

def install_hooks(
    method,
    sign,
    dose_scale,
    telemetry,
):
    handles = []

    for L in range(N_LAYERS):

        def make_hook(layer_idx):

            def hook(module, args, output):
                if isinstance(output, tuple):
                    old = output[0]
                    rest = output[1:]
                else:
                    old = output
                    rest = None

                B = old.shape[0]

                old_last = old[:, -1, :].float()

                base_dose = (
                    IVME
                    * ENVELOPE[layer_idx]
                    * dose_scale
                )

                if method == "GLOBAL_STATIC":
                    direction = (
                        GLOBAL_COMPASS[None, :]
                        .expand(B, -1)
                    )

                    gain = torch.ones(
                        B,
                        device=DEVICE,
                        dtype=torch.float32,
                    )

                    confidence = torch.zeros_like(gain)

                elif method == "LAYER_LOCAL":
                    direction = (
                        LOCAL_COMPASS[layer_idx][None, :]
                        .expand(B, -1)
                    )

                    gain = torch.ones(
                        B,
                        device=DEVICE,
                        dtype=torch.float32,
                    )

                    confidence = torch.zeros_like(gain)

                elif method == "STATE_GAIN":
                    _, gain, confidence, _ = state_decode(
                        old_last,
                        layer_idx,
                        shuffled=False,
                    )

                    direction = (
                        LOCAL_COMPASS[layer_idx][None, :]
                        .expand(B, -1)
                    )

                elif method == "STATE_DIRECTION":
                    direction, _, confidence, _ = state_decode(
                        old_last,
                        layer_idx,
                        shuffled=False,
                    )

                    gain = torch.ones(
                        B,
                        device=DEVICE,
                        dtype=torch.float32,
                    )

                elif method == "COMBINED":
                    direction, gain, confidence, _ = state_decode(
                        old_last,
                        layer_idx,
                        shuffled=False,
                    )

                elif method == "SHUFFLED_STATE":
                    direction, gain, confidence, _ = state_decode(
                        old_last,
                        layer_idx,
                        shuffled=True,
                    )

                else:
                    raise ValueError(method)

                direction = (
                    float(sign)
                    * direction
                ).contiguous()

                dose = (
                    base_dose
                    * gain
                ).float().contiguous()

                new = seasc_ext.seasc_batch(
                    old,
                    direction,
                    dose,
                )

                new_last = new[:, -1, :].float()

                delta = new_last - old_last

                norm0 = (
                    old_last.norm(dim=-1)
                    .clamp_min(EPS)
                )

                realized = (
                    delta.norm(dim=-1)
                    / norm0
                )

                telemetry[layer_idx]["calls"] += 1
                telemetry[layer_idx]["requested"].extend(
                    dose.detach().cpu().tolist()
                )
                telemetry[layer_idx]["realized"].extend(
                    realized.detach().cpu().tolist()
                )
                telemetry[layer_idx]["gain"].extend(
                    gain.detach().cpu().tolist()
                )
                telemetry[layer_idx]["confidence"].extend(
                    confidence.detach().cpu().tolist()
                )

                if rest is None:
                    return new

                return (new,) + rest

            return hook

        handles.append(
            layers[L].register_forward_hook(
                make_hook(L)
            )
        )

    return handles

# =============================================================================
# 11. TEACHER-FORCED SEQUENCE SCORING
#
# Scores continuation log probability conditioned on prompt.
# Hooks operate during the full forward.
# =============================================================================

def build_sequence(user_context, continuation):
    prompt = chat_text(user_context)

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

    input_ids = torch.cat(
        [
            p["input_ids"],
            c["input_ids"],
        ],
        dim=1,
    ).to(DEVICE)

    attention_mask = torch.ones_like(
        input_ids,
        device=DEVICE,
    )

    prompt_len = p["input_ids"].shape[1]

    return (
        input_ids,
        attention_mask,
        prompt_len,
    )

@torch.inference_mode()
def continuation_logp(
    user_context,
    continuation,
    method=None,
    sign=0,
    dose_scale=0.0,
):
    ids, mask, prompt_len = build_sequence(
        user_context,
        continuation,
    )

    telemetry = [
        {
            "calls": 0,
            "requested": [],
            "realized": [],
            "gain": [],
            "confidence": [],
        }
        for _ in range(N_LAYERS)
    ]

    handles = []

    if method is not None:
        handles = install_hooks(
            method,
            sign,
            dose_scale,
            telemetry,
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

    logits = out.logits.float()

    # token t predicted by logits t-1
    target = ids[:, prompt_len:]

    pred_logits = logits[
        :,
        prompt_len - 1:-1,
        :
    ]

    lp = F.log_softmax(
        pred_logits,
        dim=-1,
    )

    token_lp = lp.gather(
        -1,
        target.unsqueeze(-1),
    ).squeeze(-1)

    mean_lp = float(
        token_lp.mean().item()
    )

    del out, logits, lp, token_lp
    del ids, mask, target, pred_logits

    return mean_lp, telemetry

# =============================================================================
# 12. BASELINE MARGINS
# =============================================================================

print("\n[5/13] Baseline symmetric task calibration...")

BASE_ROWS = []

for i, item in enumerate(EVAL_ITEMS):
    lp_pos, _ = continuation_logp(
        item["context"],
        item["pos"],
    )

    lp_neg, _ = continuation_logp(
        item["context"],
        item["neg"],
    )

    margin = lp_pos - lp_neg

    BASE_ROWS.append({
        "item": i,
        "lp_pos": lp_pos,
        "lp_neg": lp_neg,
        "margin": margin,
    })

    print(
        f"      item {i:02d} "
        f"baseline margin={margin:+.6f}"
    )

BASE_MARGIN = float(
    np.mean([
        x["margin"]
        for x in BASE_ROWS
    ])
)

print(
    f"      mean baseline margin = "
    f"{BASE_MARGIN:+.6f}"
)

# =============================================================================
# 13. CAUSAL ± STEERING MATRIX
# =============================================================================

print("\n[6/13] Running ± causal steering matrix...")

ROWS = []

for method in METHODS:
    for scale in DOSE_SCALES:

        plus_margins = []
        minus_margins = []

        requested = []
        realized = []
        gains = []
        confidences = []

        for i, item in enumerate(EVAL_ITEMS):

            p_pos, tel1 = continuation_logp(
                item["context"],
                item["pos"],
                method=method,
                sign=+1,
                dose_scale=scale,
            )

            p_neg, tel2 = continuation_logp(
                item["context"],
                item["neg"],
                method=method,
                sign=+1,
                dose_scale=scale,
            )

            m_pos = p_pos - p_neg

            n_pos, tel3 = continuation_logp(
                item["context"],
                item["pos"],
                method=method,
                sign=-1,
                dose_scale=scale,
            )

            n_neg, tel4 = continuation_logp(
                item["context"],
                item["neg"],
                method=method,
                sign=-1,
                dose_scale=scale,
            )

            m_neg = n_pos - n_neg

            plus_margins.append(m_pos)
            minus_margins.append(m_neg)

            for tel in [tel1, tel2, tel3, tel4]:
                for L in range(N_LAYERS):
                    requested.extend(
                        tel[L]["requested"]
                    )
                    realized.extend(
                        tel[L]["realized"]
                    )
                    gains.extend(
                        tel[L]["gain"]
                    )
                    confidences.extend(
                        tel[L]["confidence"]
                    )

        plus_mean = float(
            np.mean(plus_margins)
        )

        minus_mean = float(
            np.mean(minus_margins)
        )

        csym = 0.5 * (
            plus_mean - minus_mean
        )

        req_mean = float(
            np.mean(requested)
        )

        real_mean = float(
            np.mean(realized)
        )

        # Physical energy proxy.
        # Squared realized relative displacement.
        energy = float(
            np.sum(
                np.square(realized)
            )
        )

        row = {
            "method": method,
            "dose_scale": scale,
            "plus_margin": plus_mean,
            "minus_margin": minus_mean,
            "Csym": csym,
            "requested_mean": req_mean,
            "realized_mean": real_mean,
            "energy": energy,
            "gain_mean": float(np.mean(gains)),
            "gain_sd": float(np.std(gains)),
            "confidence_mean": float(np.mean(confidences)),
        }

        ROWS.append(row)

        print(
            f"  {method:<18} "
            f"scale={scale:.2f} | "
            f"+M={plus_mean:+.6f} "
            f"-M={minus_mean:+.6f} | "
            f"Csym={csym:+.6f} | "
            f"req={100*req_mean:.4f}% "
            f"real={100*real_mean:.4f}% | "
            f"E={energy:.6f}"
        )

# =============================================================================
# 14. COLLATERAL — NEUTRAL ΔNLL + KL
#
# We evaluate + steering.
#
# Baseline and steered logits are compared only over neutral continuation
# tokens. This is separate from the causal ± steering score.
# =============================================================================

print("\n[7/13] Neutral collateral assay...")

NEUTRAL_CONT = (
    " The scene contains ordinary stationary objects "
    "and no additional relation is specified."
)

@torch.inference_mode()
def neutral_metrics(text, method, scale):
    ids, mask, prompt_len = build_sequence(
        text,
        NEUTRAL_CONT,
    )

    out0 = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    )

    base_logits = out0.logits.float()[
        :,
        prompt_len - 1:-1,
        :
    ]

    target = ids[:, prompt_len:]

    base_lp = F.log_softmax(
        base_logits,
        dim=-1,
    )

    base_nll = -base_lp.gather(
        -1,
        target.unsqueeze(-1),
    ).squeeze(-1).mean()

    telemetry = [
        {
            "calls": 0,
            "requested": [],
            "realized": [],
            "gain": [],
            "confidence": [],
        }
        for _ in range(N_LAYERS)
    ]

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

    steer_logits = out1.logits.float()[
        :,
        prompt_len - 1:-1,
        :
    ]

    steer_lp = F.log_softmax(
        steer_logits,
        dim=-1,
    )

    steer_nll = -steer_lp.gather(
        -1,
        target.unsqueeze(-1),
    ).squeeze(-1).mean()

    p = base_lp.exp()

    kl = (
        p * (
            base_lp - steer_lp
        )
    ).sum(dim=-1).mean()

    dnll = float(
        (steer_nll - base_nll).item()
    )

    klv = float(
        kl.item()
    )

    del out0, out1
    del base_logits, steer_logits
    del base_lp, steer_lp
    del ids, mask, target

    return dnll, klv

COLLATERAL = {}

for method in METHODS:
    for scale in DOSE_SCALES:
        dnlls = []
        kls = []

        for text in NEUTRAL:
            dnll, kl = neutral_metrics(
                text,
                method,
                scale,
            )

            dnlls.append(dnll)
            kls.append(kl)

        key = (method, scale)

        COLLATERAL[key] = {
            "dnll": float(np.mean(dnlls)),
            "kl": float(np.mean(kls)),
        }

        print(
            f"  {method:<18} "
            f"scale={scale:.2f} | "
            f"ΔNLL={np.mean(dnlls):+.6f} | "
            f"KL={np.mean(kls):.6f}"
        )

for row in ROWS:
    c = COLLATERAL[
        (
            row["method"],
            row["dose_scale"],
        )
    ]

    row["delta_nll"] = c["dnll"]
    row["kl"] = c["kl"]

# =============================================================================
# 15. ENERGY-NORMALIZED VIEW
# =============================================================================

print("\n[8/13] Causal effect / energy view...")

for row in ROWS:
    row["Csym_per_energy"] = (
        row["Csym"]
        / max(row["energy"], EPS)
    )

SORTED = sorted(
    ROWS,
    key=lambda x: x["Csym"],
    reverse=True,
)

print(
    f"{'METHOD':<19}"
    f"{'SCALE':>8}"
    f"{'CSYM':>13}"
    f"{'ENERGY':>14}"
    f"{'CSYM/E':>14}"
    f"{'DNLL':>13}"
    f"{'KL':>13}"
)

print("-" * 94)

for r in SORTED:
    print(
        f"{r['method']:<19}"
        f"{r['dose_scale']:>8.2f}"
        f"{r['Csym']:>13.6f}"
        f"{r['energy']:>14.6f}"
        f"{r['Csym_per_energy']:>14.6f}"
        f"{r['delta_nll']:>13.6f}"
        f"{r['kl']:>13.6f}"
    )

# =============================================================================
# 16. PRIMARY DIFFERENTIALS
#
# Compare at each equal requested dose scale.
# =============================================================================

print("\n[9/13] Primary mechanistic differentials...")

DIFF_ROWS = []

def get_row(method, scale):
    for r in ROWS:
        if (
            r["method"] == method
            and r["dose_scale"] == scale
        ):
            return r
    raise KeyError((method, scale))

for scale in DOSE_SCALES:
    G = get_row("GLOBAL_STATIC", scale)
    L = get_row("LAYER_LOCAL", scale)
    SG = get_row("STATE_GAIN", scale)
    SD = get_row("STATE_DIRECTION", scale)
    C = get_row("COMBINED", scale)
    SH = get_row("SHUFFLED_STATE", scale)

    row = {
        "dose_scale": scale,

        "local_minus_global":
            L["Csym"] - G["Csym"],

        "state_gain_minus_local":
            SG["Csym"] - L["Csym"],

        "state_direction_minus_local":
            SD["Csym"] - L["Csym"],

        "combined_minus_local":
            C["Csym"] - L["Csym"],

        "combined_minus_state_gain":
            C["Csym"] - SG["Csym"],

        "combined_minus_state_direction":
            C["Csym"] - SD["Csym"],

        "combined_minus_shuffled":
            C["Csym"] - SH["Csym"],
    }

    DIFF_ROWS.append(row)

    print(
        f"scale={scale:.2f} | "
        f"LOCAL-GLOBAL={row['local_minus_global']:+.6f} | "
        f"GAIN-LOCAL={row['state_gain_minus_local']:+.6f} | "
        f"DIR-LOCAL={row['state_direction_minus_local']:+.6f} | "
        f"COMB-LOCAL={row['combined_minus_local']:+.6f} | "
        f"COMB-SHUFFLE={row['combined_minus_shuffled']:+.6f}"
    )

# =============================================================================
# 17. PER-ITEM SIGN CONSISTENCY AT BASE DOSE SCALE 1.0
#
# This avoids interpreting a mean driven by one or two prompts.
# =============================================================================

print("\n[10/13] Prompt-level consistency at scale=1.00...")

CONSISTENCY = []

for method in METHODS:
    cs = []

    for item in EVAL_ITEMS:
        p_pos, _ = continuation_logp(
            item["context"],
            item["pos"],
            method=method,
            sign=+1,
            dose_scale=1.0,
        )

        p_neg, _ = continuation_logp(
            item["context"],
            item["neg"],
            method=method,
            sign=+1,
            dose_scale=1.0,
        )

        n_pos, _ = continuation_logp(
            item["context"],
            item["pos"],
            method=method,
            sign=-1,
            dose_scale=1.0,
        )

        n_neg, _ = continuation_logp(
            item["context"],
            item["neg"],
            method=method,
            sign=-1,
            dose_scale=1.0,
        )

        csym_i = 0.5 * (
            (p_pos - p_neg)
            - (n_pos - n_neg)
        )

        cs.append(csym_i)

    positive = int(
        np.sum(
            np.asarray(cs) > 0
        )
    )

    CONSISTENCY.append({
        "method": method,
        "positive_prompts": positive,
        "total_prompts": len(cs),
        "median_Csym": float(np.median(cs)),
        "mean_Csym": float(np.mean(cs)),
    })

    print(
        f"  {method:<18} "
        f"P(Csym>0)={positive:02d}/{len(cs)} | "
        f"median={np.median(cs):+.6f} | "
        f"mean={np.mean(cs):+.6f}"
    )

# =============================================================================
# 18. DIAGNOSIS
#
# Descriptive mechanism labels only.
# We require consistency across >=2/3 dose scales.
# =============================================================================

print("\n[11/13] Mechanistic diagnosis...")

def positive_count(key, threshold=0.0):
    return sum(
        x[key] > threshold
        for x in DIFF_ROWS
    )

FLAGS = []

if positive_count("local_minus_global") >= 2:
    FLAGS.append(
        "LAYER_LOCAL_RECONSTRUCTION_SUPPORTED"
    )

if positive_count("state_gain_minus_local") >= 2:
    FLAGS.append(
        "STATE_DEPENDENT_GAIN_SUPPORTED"
    )

if positive_count("state_direction_minus_local") >= 2:
    FLAGS.append(
        "STATE_DEPENDENT_DIRECTION_SUPPORTED"
    )

if positive_count("combined_minus_local") >= 2:
    FLAGS.append(
        "STATE_CONDITIONED_COMBINED_SUPPORTED"
    )

if positive_count("combined_minus_shuffled") >= 2:
    FLAGS.append(
        "MATCHED_STATE_SPECIFICITY_SUPPORTED"
    )

if not FLAGS:
    FLAGS.append(
        "STATE_CONDITIONING_NOT_CAUSALLY_SUPPORTED"
    )

for x in FLAGS:
    print(" -", x)

print("""
Interpretation:

LAYER_LOCAL_RECONSTRUCTION_SUPPORTED
    Rebuilding the semantic compass independently at each layer outperforms
    carrying one source-layer vector through depth.

STATE_DEPENDENT_GAIN_SUPPORTED
    The current hidden state contains useful information about HOW MUCH
    steering should be applied, even with direction fixed to A_L.

STATE_DEPENDENT_DIRECTION_SUPPORTED
    The current hidden state contains useful information about WHICH local
    semantic direction should be used.

STATE_CONDITIONED_COMBINED_SUPPORTED
    Direction and magnitude adaptation together improve causal control over
    the ordinary layer-local AkbasCore compass.

MATCHED_STATE_SPECIFICITY_SUPPORTED
    Correct state-to-direction correspondence matters; the effect cannot be
    explained merely by mixing the SEARCH direction bank.

STATE_CONDITIONING_NOT_CAUSALLY_SUPPORTED
    TEST158's observational matched-context structure did not translate into
    superior causal steering under this decoder and intervention regime.
""")

# =============================================================================
# 19. SAVE
# =============================================================================

print("\n[12/13] Saving artifacts...")

def save_csv(path, rows):
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
        w = csv.DictWriter(
            f,
            fieldnames=keys,
        )
        w.writeheader()
        w.writerows(rows)

save_csv(
    ROOT / "METHOD_MATRIX.csv",
    ROWS,
)

save_csv(
    ROOT / "MECHANISTIC_DIFFERENTIALS.csv",
    DIFF_ROWS,
)

save_csv(
    ROOT / "PROMPT_CONSISTENCY.csv",
    CONSISTENCY,
)

save_csv(
    ROOT / "BASELINE.csv",
    BASE_ROWS,
)

npz = {}

for L in range(N_LAYERS):
    npz[f"LOCAL_COMPASS_L{L}"] = (
        LOCAL_COMPASS[L]
        .detach()
        .cpu()
        .numpy()
    )

    npz[f"STATE_REF_L{L}"] = (
        STATE_REF[L]
        .detach()
        .cpu()
        .numpy()
    )

    npz[f"STATE_DIR_L{L}"] = (
        STATE_DIR[L]
        .detach()
        .cpu()
        .numpy()
    )

np.savez_compressed(
    ROOT / "TEST159_STATE_BANK.npz",
    **npz,
)

REPORT = {
    "test":
        "TEST 159",

    "title":
        "LOCAL STATE-CONDITIONED STEERING DIFFERENTIAL",

    "model":
        MODEL_ID,

    "gpu":
        torch.cuda.get_device_name(0),

    "akbascore_motor": {
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

        "physical_law":
            "frozen-norm direct dose",

        "envelope":
            "static normalized DRA",

        "training":
            False,

        "gradients":
            False,
    },

    "state_decoder": {
        "search_pairs":
            len(SEARCH_PAIRS),

        "temperature":
            STATE_TEMP,

        "gain_min":
            GAIN_MIN,

        "gain_max":
            GAIN_MAX,

        "label_symmetric_reference":
            True,

        "evaluation_items_in_search":
            False,
    },

    "baseline_margin":
        BASE_MARGIN,

    "method_matrix":
        ROWS,

    "mechanistic_differentials":
        DIFF_ROWS,

    "prompt_consistency":
        CONSISTENCY,

    "diagnostic_flags":
        FLAGS,

    "interpretation_contract": [
        "TEST159 is a frozen-weight runtime intervention assay.",
        "The AkbasCore frozen-norm physical injection law is retained.",
        "The AkbasCore static normalized DRA layer envelope is retained.",
        "State references are label-symmetric matched-pair midpoints.",
        "Evaluation sentences are excluded from the SEARCH state bank.",
        "State gain is normalized to mean one within each forward batch.",
        "Causal control is measured symmetrically using positive and negative steering.",
        "Neutral delta-NLL and KL are reported separately from causal steering.",
        "Shuffled-state is a negative correspondence control.",
        "A positive result supports only the tested model, axis, prompt family, decoder, dose range and runtime configuration.",
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
# 20. FINAL REPORT
# =============================================================================

print("\n[13/13] COMPLETE")
print("=" * 150)
print("TEST 159 — FINAL SUMMARY")
print("=" * 150)

print(
    f"Baseline ambiguous margin : "
    f"{BASE_MARGIN:+.6f}"
)

print("\nMECHANISTIC DIFFERENTIALS")

for r in DIFF_ROWS:
    print(
        f"scale={r['dose_scale']:.2f} | "
        f"LOCAL-GLOBAL={r['local_minus_global']:+.6f} | "
        f"GAIN-LOCAL={r['state_gain_minus_local']:+.6f} | "
        f"DIR-LOCAL={r['state_direction_minus_local']:+.6f} | "
        f"COMBINED-LOCAL={r['combined_minus_local']:+.6f} | "
        f"COMBINED-SHUFFLED={r['combined_minus_shuffled']:+.6f}"
    )

print("\nDIAGNOSTIC FLAGS")

for x in FLAGS:
    print(" ", x)

print("\nSaved:")
print(" ", ROOT / "METHOD_MATRIX.csv")
print(" ", ROOT / "MECHANISTIC_DIFFERENTIALS.csv")
print(" ", ROOT / "PROMPT_CONSISTENCY.csv")
print(" ", ROOT / "BASELINE.csv")
print(" ", ROOT / "TEST159_STATE_BANK.npz")
print(" ", ROOT / "REPORT.json")

print("=" * 150)
print("TEST 159 COMPLETE")
print("=" * 150)
