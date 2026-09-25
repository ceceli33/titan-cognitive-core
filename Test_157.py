# =============================================================================
# TEST 157 — HELD-OUT SEMANTIC RECODING OPERATOR
# DOES A SHARED R_{a->b} GENERALIZE TO AN UNSEEN SEMANTIC AXIS?
#
# Qwen/Qwen2.5-7B-Instruct · Frozen · BF16 forward · FP32 analysis
#
# TEST156 NATURAL ATLAS IS PRESERVED EXACTLY:
#   same natural corpus
#   same seed=42 corpus shuffle
#   same chat template
#   same last-token capture
#   same centering
#   same sample-space QR/SVD
#   same TEST154/156 fingerprint
#
# TRAIN semantic axes:
#   BEFORE_AFTER
#   INSIDE_OUTSIDE
#   LEFT_RIGHT
#   NEAR_FAR
#   EARLIER_LATER
#   FRONT_BEHIND
#
# VAL semantic axes:
#   BIGGER_SMALLER
#   HOTTER_COLDER
#
# FINAL held-out semantic axis:
#   ABOVE_BELOW
#
# QUESTION:
#
#   TEST156 showed that the natural covariance correspondence
#
#       U_i  <->  V_i
#
#   is not itself a reliable semantic coordinate map.
#
#   TEST157 asks whether a shared semantic recoding operator
#
#       R_{a->b}
#
#   can be learned from OTHER semantic axes and generalize to a
#   completely unseen semantic axis.
#
# For a natural rank-K atlas:
#
#       c_a = U^T A_a
#       c_b = V^T A_b
#
# TRAIN learns:
#
#       c_b ~= R c_a
#
# Ridge solution:
#
#       R = C_b C_a^T (C_a C_a^T + lambda I)^-1
#
# Natural TEST156 mapping is the identity:
#
#       R = I
#
# Prediction:
#
#       Ahat_b = normalize(V R U^T A_a)
#
# IMPORTANT:
#
#   FINAL ABOVE/BELOW is NEVER used for:
#       K selection
#       ridge selection
#       R fitting
#       sign correction
#       calibration
#       operator selection
#
# No steering
# No intervention
# No LoRA
# No gradients
# No optimizer
# No JVP/VJP
# =============================================================================

import os
import sys
import subprocess
import json
import csv
import random
import warnings
from pathlib import Path

for pkg, mod in [
    ("transformers>=4.46.0", "transformers"),
    ("accelerate", "accelerate"),
]:
    try:
        __import__(mod)
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg]
        )

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# CONFIG
# =============================================================================

SEED = 157
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

LAYERS = [3, 6, 10, 14, 19]
SEGMENTS = [(3, 6), (6, 10), (10, 14), (14, 19)]

SEARCH_N = 192

K_VALUES = [4, 8, 16, 32]

RIDGES = [
    0.0,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
]

TRAIN_AXES = [
    "BEFORE_AFTER",
    "INSIDE_OUTSIDE",
    "LEFT_RIGHT",
    "NEAR_FAR",
    "EARLIER_LATER",
    "FRONT_BEHIND",
]

VAL_AXES = [
    "BIGGER_SMALLER",
    "HOTTER_COLDER",
]

FINAL_AXES = [
    "ABOVE_BELOW",
]

EPS = 1e-10
FP_TOL = 0.015

ROOT = Path(
    "/content/TEST157_HELDOUT_SEMANTIC_RECODING_OPERATOR"
)
ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type != "cuda":
    raise RuntimeError("TEST157 requires CUDA.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 154)
print("TEST 157 — HELD-OUT SEMANTIC RECODING OPERATOR")
print("DOES A SHARED R_{a->b} GENERALIZE TO AN UNSEEN SEMANTIC AXIS?")
print("=" * 154)

# =============================================================================
# MODEL
# =============================================================================

print("\n[1/16] Loading frozen model...")

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

HIDDEN = model.config.hidden_size

print("GPU            :", torch.cuda.get_device_name(0))
print("Model          :", MODEL_ID)
print("Hidden         :", HIDDEN)
print("Decoder layers :", len(model.model.layers))
print("Layers         :", LAYERS)
print("Segments       :", SEGMENTS)
print("K values       :", K_VALUES)
print("TRAIN axes     :", TRAIN_AXES)
print("VAL axes       :", VAL_AXES)
print("FINAL axis     :", FINAL_AXES)
print("Ridges         :", RIDGES)

# =============================================================================
# CHAT — IDENTICAL TO TEST156
# =============================================================================

SYSTEM_EN = "You are a concise reasoning assistant."

def chat_text(text):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_EN,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def encode_prompt(text):
    rendered = chat_text(text)

    tok = tokenizer(
        rendered,
        return_tensors="pt",
    ).to(DEVICE)

    pos = (
        int(
            tok["attention_mask"][0]
            .sum()
            .item()
        )
        - 1
    )

    return tok, pos

# =============================================================================
# SEMANTIC AXES
#
# Each axis uses 8 matched positive / negative sentence pairs.
# FINAL ABOVE/BELOW is identical to TEST156.
# =============================================================================

AXES = {
    "BEFORE_AFTER": {
        "POS": [
            "The bell rang before the door opened.",
            "The light flashed before the alarm sounded.",
            "The letter arrived before the meeting started.",
            "The rain stopped before the sun appeared.",
            "The engine started before the vehicle moved.",
            "The guest arrived before dinner began.",
            "The signal appeared before the machine responded.",
            "The announcement came before the crowd moved.",
        ],
        "NEG": [
            "The bell rang after the door opened.",
            "The light flashed after the alarm sounded.",
            "The letter arrived after the meeting started.",
            "The rain stopped after the sun appeared.",
            "The engine started after the vehicle moved.",
            "The guest arrived after dinner began.",
            "The signal appeared after the machine responded.",
            "The announcement came after the crowd moved.",
        ],
    },

    "INSIDE_OUTSIDE": {
        "POS": [
            "The key is inside the wooden box.",
            "The coin is inside the small container.",
            "The book is inside the blue bag.",
            "The tool is inside the metal cabinet.",
            "The bottle is inside the cardboard crate.",
            "The letter is inside the envelope.",
            "The toy is inside the storage basket.",
            "The cup is inside the kitchen cupboard.",
        ],
        "NEG": [
            "The key is outside the wooden box.",
            "The coin is outside the small container.",
            "The book is outside the blue bag.",
            "The tool is outside the metal cabinet.",
            "The bottle is outside the cardboard crate.",
            "The letter is outside the envelope.",
            "The toy is outside the storage basket.",
            "The cup is outside the kitchen cupboard.",
        ],
    },

    "LEFT_RIGHT": {
        "POS": [
            "The red marker is left of the blue marker.",
            "The lamp is left of the wooden table.",
            "The bird is left of the branch.",
            "The sign is left of the doorway.",
            "The shelf is left of the cabinet.",
            "The tree is left of the hill.",
            "The clock is left of the desk.",
            "The picture is left of the sofa.",
        ],
        "NEG": [
            "The red marker is right of the blue marker.",
            "The lamp is right of the wooden table.",
            "The bird is right of the branch.",
            "The sign is right of the doorway.",
            "The shelf is right of the cabinet.",
            "The tree is right of the hill.",
            "The clock is right of the desk.",
            "The picture is right of the sofa.",
        ],
    },

    "NEAR_FAR": {
        "POS": [
            "The red marker is near the blue marker.",
            "The lamp is near the wooden table.",
            "The bird is near the branch.",
            "The sign is near the doorway.",
            "The shelf is near the cabinet.",
            "The tree is near the hill.",
            "The clock is near the desk.",
            "The picture is near the sofa.",
        ],
        "NEG": [
            "The red marker is far from the blue marker.",
            "The lamp is far from the wooden table.",
            "The bird is far from the branch.",
            "The sign is far from the doorway.",
            "The shelf is far from the cabinet.",
            "The tree is far from the hill.",
            "The clock is far from the desk.",
            "The picture is far from the sofa.",
        ],
    },

    "EARLIER_LATER": {
        "POS": [
            "The first event happened earlier than the second event.",
            "The train arrived earlier than the bus.",
            "The message appeared earlier than the warning.",
            "The first measurement was recorded earlier than the second.",
            "The blue signal occurred earlier than the red signal.",
            "The first session ended earlier than the second session.",
            "The package arrived earlier than the letter.",
            "The first sound occurred earlier than the second sound.",
        ],
        "NEG": [
            "The first event happened later than the second event.",
            "The train arrived later than the bus.",
            "The message appeared later than the warning.",
            "The first measurement was recorded later than the second.",
            "The blue signal occurred later than the red signal.",
            "The first session ended later than the second session.",
            "The package arrived later than the letter.",
            "The first sound occurred later than the second sound.",
        ],
    },

    "FRONT_BEHIND": {
        "POS": [
            "The red marker is in front of the blue marker.",
            "The lamp is in front of the wooden table.",
            "The bird is in front of the branch.",
            "The sign is in front of the doorway.",
            "The shelf is in front of the cabinet.",
            "The tree is in front of the hill.",
            "The clock is in front of the desk.",
            "The picture is in front of the sofa.",
        ],
        "NEG": [
            "The red marker is behind the blue marker.",
            "The lamp is behind the wooden table.",
            "The bird is behind the branch.",
            "The sign is behind the doorway.",
            "The shelf is behind the cabinet.",
            "The tree is behind the hill.",
            "The clock is behind the desk.",
            "The picture is behind the sofa.",
        ],
    },

    "BIGGER_SMALLER": {
        "POS": [
            "The red circle is bigger than the blue circle.",
            "The wooden box is bigger than the metal box.",
            "The first ball is bigger than the second ball.",
            "The green square is bigger than the yellow square.",
            "The first container is bigger than the second container.",
            "The black disk is bigger than the white disk.",
            "The first stone is bigger than the second stone.",
            "The orange shape is bigger than the purple shape.",
        ],
        "NEG": [
            "The red circle is smaller than the blue circle.",
            "The wooden box is smaller than the metal box.",
            "The first ball is smaller than the second ball.",
            "The green square is smaller than the yellow square.",
            "The first container is smaller than the second container.",
            "The black disk is smaller than the white disk.",
            "The first stone is smaller than the second stone.",
            "The orange shape is smaller than the purple shape.",
        ],
    },

    "HOTTER_COLDER": {
        "POS": [
            "The red object is hotter than the blue object.",
            "The first cup is hotter than the second cup.",
            "The metal plate is hotter than the wooden plate.",
            "The first surface is hotter than the second surface.",
            "The black container is hotter than the white container.",
            "The first liquid is hotter than the second liquid.",
            "The orange block is hotter than the green block.",
            "The first sample is hotter than the second sample.",
        ],
        "NEG": [
            "The red object is colder than the blue object.",
            "The first cup is colder than the second cup.",
            "The metal plate is colder than the wooden plate.",
            "The first surface is colder than the second surface.",
            "The black container is colder than the white container.",
            "The first liquid is colder than the second liquid.",
            "The orange block is colder than the green block.",
            "The first sample is colder than the second sample.",
        ],
    },

    "ABOVE_BELOW": {
        "POS": [
            "The red marker is above the blue marker.",
            "The lamp is above the wooden table.",
            "The bird is above the branch.",
            "The sign is above the doorway.",
            "The shelf is above the cabinet.",
            "The cloud is above the hill.",
            "The clock is above the desk.",
            "The picture is above the sofa.",
        ],
        "NEG": [
            "The red marker is below the blue marker.",
            "The lamp is below the wooden table.",
            "The bird is below the branch.",
            "The sign is below the doorway.",
            "The shelf is below the cabinet.",
            "The cloud is below the hill.",
            "The clock is below the desk.",
            "The picture is below the sofa.",
        ],
    },
}

assert set(AXES.keys()) == set(
    TRAIN_AXES + VAL_AXES + FINAL_AXES
)

# =============================================================================
# TEST156 NATURAL CORPUS — EXACTLY PRESERVED
# =============================================================================

SUBJECTS = [
    "A wooden bowl",
    "A ceramic cup",
    "A blue folder",
    "A metal toolbox",
    "A paper notebook",
    "A glass bottle",
    "A cotton towel",
    "A small basket",
    "A desk lamp",
    "A cardboard box",
    "A kitchen plate",
    "A plastic container",
    "A leather wallet",
    "A silver spoon",
    "A garden hose",
    "A black umbrella",
]

VERBS = [
    "rests",
    "sits",
    "remains",
    "stands",
]

LOCATIONS = [
    "near a plain wall",
    "beside a wooden table",
    "inside a quiet room",
    "near a closed cabinet",
    "beside a simple chair",
    "near a clean counter",
    "inside a storage area",
    "near a small window",
]

DETAILS = [
    "The room is quiet.",
    "The surface is clean.",
    "The area is well lit.",
    "Nothing unusual is happening.",
    "The scene is ordinary.",
    "The setting is simple.",
    "The objects are stationary.",
    "The surroundings are calm.",
]

natural_pool = [
    f"{s} {v} {loc}. {d}"
    for s in SUBJECTS
    for v in VERBS
    for loc in LOCATIONS
    for d in DETAILS
]

# CRITICAL:
# TEST156 / TEST155 corpus construction.
# Do NOT replace this with SEED=157.
rng = np.random.default_rng(42)
rng.shuffle(natural_pool)

natural_pool = list(
    dict.fromkeys(natural_pool)
)

SEARCH_TEXTS = natural_pool[:SEARCH_N]

assert len(SEARCH_TEXTS) == SEARCH_N

# =============================================================================
# ACTIVATION CAPTURE — EXACT TEST156 LOGIC
# =============================================================================

@torch.inference_mode()
def capture_layers(text):
    tok, pos = encode_prompt(text)

    out = model(
        **tok,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    result = {
        L:
        out.hidden_states[L + 1][0, pos]
        .float()
        .cpu()
        for L in LAYERS
    }

    del out, tok

    return result

def capture_bank(texts, label):
    bank = {
        L: []
        for L in LAYERS
    }

    for i, text in enumerate(texts, 1):
        row = capture_layers(text)

        for L in LAYERS:
            bank[L].append(
                row[L]
            )

        if (
            i == 1
            or i % 32 == 0
            or i == len(texts)
        ):
            print(
                f"  {label}: "
                f"{i:03d}/{len(texts)}"
            )

    return {
        L:
        torch.stack(bank[L])
        .float()
        for L in LAYERS
    }

# =============================================================================
# NATURAL SEARCH ATLAS
# =============================================================================

print("\n[2/16] Capturing natural SEARCH atlas...")

SEARCH = capture_bank(
    SEARCH_TEXTS,
    "SEARCH",
)

MEAN = {
    L:
    SEARCH[L].mean(
        dim=0,
        keepdim=True,
    )
    for L in LAYERS
}

CENTERED = {
    L:
    SEARCH[L] - MEAN[L]
    for L in LAYERS
}

# =============================================================================
# SAMPLE-SPACE CROSS-COVARIANCE SVD — EXACT TEST156
#
# X^T = Qx Rx
# Y^T = Qy Ry
#
# Cxy = X^T Y/(N-1)
#     = Qx [Rx Ry^T/(N-1)] Qy^T
# =============================================================================

def cross_cov_svd(X_cpu, Y_cpu):
    X = X_cpu.to(
        DEVICE,
        dtype=torch.float32,
    )

    Y = Y_cpu.to(
        DEVICE,
        dtype=torch.float32,
    )

    n = X.shape[0]

    Qx, Rx = torch.linalg.qr(
        X.T,
        mode="reduced",
    )

    Qy, Ry = torch.linalg.qr(
        Y.T,
        mode="reduced",
    )

    small = (
        Rx @ Ry.T
    ) / float(n - 1)

    Uk, S, Vhk = torch.linalg.svd(
        small,
        full_matrices=False,
    )

    U = Qx @ Uk
    V = Qy @ Vhk.T

    return (
        U.detach(),
        S.detach(),
        V.detach(),
    )

print("\n[3/16] Building natural transport atlas...")

ATLAS = {}

for a, b in SEGMENTS:
    U, S, V = cross_cov_svd(
        CENTERED[a],
        CENTERED[b],
    )

    ATLAS[(a, b)] = {
        "U": U,
        "S": S,
        "V": V,
    }

    print(
        f"L{a}->L{b} | "
        f"s1={S[0].item():.6f} "
        f"s4={S[3].item():.6f} "
        f"s8={S[7].item():.6f} "
        f"s16={S[15].item():.6f} "
        f"s32={S[31].item():.6f}"
    )

# =============================================================================
# TEST156 FINGERPRINT — EXACTLY PRESERVED
# =============================================================================

print("\n[4/16] Fingerprint check...")

EXPECTED = {
    (3, 6): {
        1: 0.623879,
        4: 0.225252,
        16: 0.048689,
        32: 0.012134,
    },
    (6, 10): {
        1: 2.785877,
        4: 1.007360,
        16: 0.193126,
        32: 0.039315,
    },
    (10, 14): {
        1: 7.172801,
        4: 2.884923,
        16: 0.496274,
        32: 0.120100,
    },
    (14, 19): {
        1: 23.883511,
        4: 7.743535,
        16: 0.987475,
        32: 0.288235,
    },
}

fingerprint_max_rel = 0.0

for pair in SEGMENTS:
    S = ATLAS[pair]["S"]

    local_max = 0.0

    for idx, expected in EXPECTED[pair].items():
        actual = float(
            S[idx - 1].item()
        )

        rel = abs(
            actual - expected
        ) / (
            abs(expected) + EPS
        )

        local_max = max(
            local_max,
            rel,
        )

        fingerprint_max_rel = max(
            fingerprint_max_rel,
            rel,
        )

    print(
        f"L{pair[0]}->L{pair[1]} "
        f"max_rel_error="
        f"{100.0 * local_max:.4f}%"
    )

print(
    "GLOBAL fingerprint max rel error = "
    f"{100.0 * fingerprint_max_rel:.4f}%"
)

FP_PASS = (
    fingerprint_max_rel <= FP_TOL
)

print(
    "Fingerprint:",
    "PASS" if FP_PASS else "FAIL",
)

if not FP_PASS:
    raise RuntimeError(
        f"Atlas fingerprint failed: "
        f"{100.0 * fingerprint_max_rel:.4f}% "
        f"> {100.0 * FP_TOL:.2f}%"
    )

# =============================================================================
# SEMANTIC COMPASS EXTRACTION
# =============================================================================

print("\n[5/16] Capturing semantic axes...")

SEMANTIC_BANKS = {}

for axis_name in (
    TRAIN_AXES
    + VAL_AXES
    + FINAL_AXES
):
    print(
        f"\n  AXIS: {axis_name}"
    )

    pos_bank = capture_bank(
        AXES[axis_name]["POS"],
        f"{axis_name}:POS",
    )

    neg_bank = capture_bank(
        AXES[axis_name]["NEG"],
        f"{axis_name}:NEG",
    )

    SEMANTIC_BANKS[
        axis_name
    ] = {
        "POS": pos_bank,
        "NEG": neg_bank,
    }

# =============================================================================
# BUILD UNIT LAYER-LOCAL COMPASSES
# =============================================================================

print("\n[6/16] Building layer-local semantic compasses...")

COMPASS = {}

for axis_name in (
    TRAIN_AXES
    + VAL_AXES
    + FINAL_AXES
):
    COMPASS[axis_name] = {}

    for L in LAYERS:
        z = (
            SEMANTIC_BANKS[
                axis_name
            ]["POS"][L].mean(
                dim=0
            )
            -
            SEMANTIC_BANKS[
                axis_name
            ]["NEG"][L].mean(
                dim=0
            )
        ).to(
            DEVICE,
            dtype=torch.float32,
        )

        z = z / (
            z.norm() + EPS
        )

        COMPASS[
            axis_name
        ][L] = z

    adjacent = []

    for a, b in SEGMENTS:
        c = float(
            F.cosine_similarity(
                COMPASS[
                    axis_name
                ][a][None],
                COMPASS[
                    axis_name
                ][b][None],
            ).item()
        )

        adjacent.append(c)

    print(
        f"{axis_name:<18} "
        f"adjacent cos = "
        + " ".join(
            f"{x:+.4f}"
            for x in adjacent
        )
    )

# =============================================================================
# COEFFICIENT EXTRACTION
#
# For pair a->b:
#
#   source semantic coordinate:
#       c_a = U^T A_a
#
#   target semantic coordinate:
#       c_b = V^T A_b
#
# These are the coordinates on which R_{a->b} acts.
# =============================================================================

def semantic_coefficients(
    axis_name,
    pair,
    K,
):
    a, b = pair

    U = (
        ATLAS[pair]["U"]
        [:, :K]
    )

    V = (
        ATLAS[pair]["V"]
        [:, :K]
    )

    A_a = COMPASS[
        axis_name
    ][a]

    A_b = COMPASS[
        axis_name
    ][b]

    c_a = U.T @ A_a
    c_b = V.T @ A_b

    return c_a, c_b

def normalized_or_zero(x):
    n = x.norm()

    if n <= 1e-12:
        return torch.zeros_like(x)

    return x / n

def cosine_safe(a, b):
    if (
        a.norm() <= 1e-12
        or b.norm() <= 1e-12
    ):
        return 0.0

    return float(
        F.cosine_similarity(
            a[None],
            b[None],
        ).item()
    )

# =============================================================================
# TRAIN R_{a->b}
#
# Csrc : K x N_axes
# Ctgt : K x N_axes
#
# lambda = 0:
#     minimum-norm least squares via pseudoinverse
#
# lambda > 0:
#     Ctgt Csrc^T (Csrc Csrc^T + lambda I)^-1
#
# No VAL or FINAL axis enters fitting.
# =============================================================================

def fit_operator(
    pair,
    K,
    ridge,
):
    src_cols = []
    tgt_cols = []

    for axis_name in TRAIN_AXES:
        c_src, c_tgt = (
            semantic_coefficients(
                axis_name,
                pair,
                K,
            )
        )

        src_cols.append(c_src)
        tgt_cols.append(c_tgt)

    Csrc = torch.stack(
        src_cols,
        dim=1,
    )

    Ctgt = torch.stack(
        tgt_cols,
        dim=1,
    )

    if ridge == 0.0:
        R = (
            Ctgt
            @ torch.linalg.pinv(
                Csrc
            )
        )

    else:
        gram = (
            Csrc
            @ Csrc.T
        )

        eye = torch.eye(
            K,
            device=DEVICE,
            dtype=torch.float32,
        )

        R = (
            Ctgt
            @ Csrc.T
            @ torch.linalg.solve(
                gram
                + float(ridge) * eye,
                eye,
            )
        )

    train_pred = R @ Csrc

    train_mse = float(
        (
            train_pred
            - Ctgt
        )
        .square()
        .mean()
        .item()
    )

    R_norm = float(
        torch.linalg.matrix_norm(
            R
        ).item()
    )

    R_spectral = float(
        torch.linalg.matrix_norm(
            R,
            ord=2,
        ).item()
    )

    R_rank = int(
        torch.linalg.matrix_rank(
            R,
            tol=1e-6,
        ).item()
    )

    return {
        "R": R,
        "Csrc": Csrc,
        "Ctgt": Ctgt,
        "train_mse": train_mse,
        "fro_norm": R_norm,
        "spectral_norm": R_spectral,
        "rank": R_rank,
    }

# =============================================================================
# PREDICTION
#
# NATURAL:
#     V I U^T A_a
#
# RECODED:
#     V R U^T A_a
#
# ORACLE_V_CEILING:
#     normalized projection of true A_b into V_K
#
# STATIC:
#     A_a itself compared directly with A_b
# =============================================================================

def evaluate_axis(
    axis_name,
    pair,
    K,
    R=None,
):
    a, b = pair

    U = (
        ATLAS[pair]["U"]
        [:, :K]
    )

    V = (
        ATLAS[pair]["V"]
        [:, :K]
    )

    A_a = COMPASS[
        axis_name
    ][a]

    A_b = COMPASS[
        axis_name
    ][b]

    c_src = U.T @ A_a
    c_tgt = V.T @ A_b

    source_capture = float(
        c_src.square()
        .sum()
        .item()
    )

    target_capture = float(
        c_tgt.square()
        .sum()
        .item()
    )

    static_cos = cosine_safe(
        A_a,
        A_b,
    )

    natural_raw = (
        V @ c_src
    )

    natural_pred = (
        normalized_or_zero(
            natural_raw
        )
    )

    natural_cos = cosine_safe(
        natural_pred,
        A_b,
    )

    oracle_raw = (
        V @ c_tgt
    )

    oracle_pred = (
        normalized_or_zero(
            oracle_raw
        )
    )

    ceiling_cos = cosine_safe(
        oracle_pred,
        A_b,
    )

    if R is None:
        recoded_raw = None
        recoded_pred = None
        recoded_cos = None
        coeff_cos = None
        coeff_rel_error = None
        recoded_raw_norm = None

    else:
        pred_coeff = (
            R @ c_src
        )

        recoded_raw = (
            V @ pred_coeff
        )

        recoded_pred = (
            normalized_or_zero(
                recoded_raw
            )
        )

        recoded_cos = cosine_safe(
            recoded_pred,
            A_b,
        )

        coeff_cos = cosine_safe(
            pred_coeff,
            c_tgt,
        )

        coeff_rel_error = float(
            (
                pred_coeff
                - c_tgt
            ).norm().item()
            /
            (
                c_tgt.norm().item()
                + EPS
            )
        )

        recoded_raw_norm = float(
            recoded_raw.norm()
            .item()
        )

    return {
        "axis": axis_name,
        "source_layer": a,
        "target_layer": b,
        "K": K,

        "static_cos":
            static_cos,

        "natural_cos":
            natural_cos,

        "recoded_cos":
            recoded_cos,

        "V_ceiling_cos":
            ceiling_cos,

        "source_capture_U":
            source_capture,

        "target_capture_V":
            target_capture,

        "coefficient_cos":
            coeff_cos,

        "coefficient_rel_error":
            coeff_rel_error,

        "natural_raw_norm":
            float(
                natural_raw.norm()
                .item()
            ),

        "recoded_raw_norm":
            recoded_raw_norm,
    }

# =============================================================================
# TRAIN ALL CANDIDATE OPERATORS
# =============================================================================

print("\n[7/16] Fitting TRAIN-only semantic recoding operators...")

OPERATORS = {}
OPERATOR_ROWS = []

for K in K_VALUES:
    for ridge in RIDGES:
        print(
            f"\nK={K} ridge={ridge:g}"
        )

        for pair in SEGMENTS:
            fit = fit_operator(
                pair,
                K,
                ridge,
            )

            OPERATORS[
                (K, ridge, pair)
            ] = fit

            OPERATOR_ROWS.append({
                "K": K,
                "ridge": ridge,
                "source_layer":
                    pair[0],
                "target_layer":
                    pair[1],
                "train_mse":
                    fit["train_mse"],
                "R_fro_norm":
                    fit["fro_norm"],
                "R_spectral_norm":
                    fit["spectral_norm"],
                "R_rank":
                    fit["rank"],
            })

            print(
                f"  L{pair[0]}->L{pair[1]} | "
                f"MSE={fit['train_mse']:.8f} | "
                f"||R||F={fit['fro_norm']:.4f} | "
                f"||R||2={fit['spectral_norm']:.4f} | "
                f"rank={fit['rank']}"
            )

# =============================================================================
# VALIDATION
#
# ONLY BIGGER/SMALLER + HOTTER/COLDER.
#
# Selection criterion:
#
#   primary:
#       mean recoded cosine across
#       2 VAL axes x 4 segments
#
#   tie break 1:
#       mean delta over natural
#
#   tie break 2:
#       lower ridge complexity / smaller K
#
# FINAL ABOVE/BELOW remains untouched.
# =============================================================================

print("\n[8/16] VAL tournament...")

VAL_ROWS = []
VAL_SUMMARY = []

for K in K_VALUES:
    for ridge in RIDGES:
        rows = []

        for axis_name in VAL_AXES:
            for pair in SEGMENTS:
                R = OPERATORS[
                    (
                        K,
                        ridge,
                        pair,
                    )
                ]["R"]

                r = evaluate_axis(
                    axis_name,
                    pair,
                    K,
                    R,
                )

                r["ridge"] = ridge

                r[
                    "delta_recoded_vs_natural"
                ] = (
                    r["recoded_cos"]
                    - r["natural_cos"]
                )

                r[
                    "delta_recoded_vs_static"
                ] = (
                    r["recoded_cos"]
                    - r["static_cos"]
                )

                r[
                    "ceiling_utilization"
                ] = (
                    r["recoded_cos"]
                    /
                    (
                        r["V_ceiling_cos"]
                        + EPS
                    )
                    if r["V_ceiling_cos"] > 1e-8
                    else 0.0
                )

                rows.append(r)
                VAL_ROWS.append(r)

        mean_recoded = float(
            np.mean([
                r["recoded_cos"]
                for r in rows
            ])
        )

        mean_natural = float(
            np.mean([
                r["natural_cos"]
                for r in rows
            ])
        )

        mean_static = float(
            np.mean([
                r["static_cos"]
                for r in rows
            ])
        )

        mean_ceiling = float(
            np.mean([
                r["V_ceiling_cos"]
                for r in rows
            ])
        )

        delta_natural = (
            mean_recoded
            - mean_natural
        )

        delta_static = (
            mean_recoded
            - mean_static
        )

        coeff_cos = float(
            np.mean([
                r["coefficient_cos"]
                for r in rows
            ])
        )

        VAL_SUMMARY.append({
            "K": K,
            "ridge": ridge,

            "mean_recoded_cos":
                mean_recoded,

            "mean_natural_cos":
                mean_natural,

            "mean_static_cos":
                mean_static,

            "mean_V_ceiling":
                mean_ceiling,

            "delta_vs_natural":
                delta_natural,

            "delta_vs_static":
                delta_static,

            "mean_coefficient_cos":
                coeff_cos,
        })

print(
    f"\n{'K':>4} "
    f"{'RIDGE':>10} "
    f"{'RECODED':>11} "
    f"{'NATURAL':>11} "
    f"{'STATIC':>11} "
    f"{'ΔNAT':>11} "
    f"{'COEFF':>11}"
)

for r in VAL_SUMMARY:
    print(
        f"{r['K']:>4d} "
        f"{r['ridge']:>10g} "
        f"{r['mean_recoded_cos']:>+11.6f} "
        f"{r['mean_natural_cos']:>+11.6f} "
        f"{r['mean_static_cos']:>+11.6f} "
        f"{r['delta_vs_natural']:>+11.6f} "
        f"{r['mean_coefficient_cos']:>+11.6f}"
    )

# =============================================================================
# LOCK
#
# FINAL cannot influence this.
# =============================================================================

print("\n[9/16] LOCKING K and ridge from VAL only...")

VAL_SORTED = sorted(
    VAL_SUMMARY,
    key=lambda r: (
        r["mean_recoded_cos"],
        r["delta_vs_natural"],
        -r["K"],
        -r["ridge"],
    ),
    reverse=True,
)

WINNER = VAL_SORTED[0]

LOCKED_K = int(
    WINNER["K"]
)

LOCKED_RIDGE = float(
    WINNER["ridge"]
)

print(
    "LOCKED K     :",
    LOCKED_K,
)

print(
    "LOCKED ridge :",
    LOCKED_RIDGE,
)

print(
    "VAL recoded  : "
    f"{WINNER['mean_recoded_cos']:+.6f}"
)

print(
    "VAL natural  : "
    f"{WINNER['mean_natural_cos']:+.6f}"
)

print(
    "VAL Δnatural : "
    f"{WINNER['delta_vs_natural']:+.6f}"
)

# =============================================================================
# FINAL — FIRST ACCESS TO ABOVE/BELOW AS EVALUATION TARGET
#
# The compass was captured earlier for computational efficiency, but no
# FINAL value has entered fitting, model selection, ridge selection,
# K selection, sign selection, or calibration.
# =============================================================================

print("\n[10/16] FINAL held-out ABOVE/BELOW evaluation...")

FINAL_ROWS = []

for pair in SEGMENTS:
    R = OPERATORS[
        (
            LOCKED_K,
            LOCKED_RIDGE,
            pair,
        )
    ]["R"]

    r = evaluate_axis(
        "ABOVE_BELOW",
        pair,
        LOCKED_K,
        R,
    )

    r["ridge"] = (
        LOCKED_RIDGE
    )

    r[
        "delta_recoded_vs_natural"
    ] = (
        r["recoded_cos"]
        - r["natural_cos"]
    )

    r[
        "delta_recoded_vs_static"
    ] = (
        r["recoded_cos"]
        - r["static_cos"]
    )

    r[
        "ceiling_utilization"
    ] = (
        r["recoded_cos"]
        /
        (
            r["V_ceiling_cos"]
            + EPS
        )
        if r["V_ceiling_cos"] > 1e-8
        else 0.0
    )

    FINAL_ROWS.append(r)

    print(
        f"L{pair[0]}->L{pair[1]} | "
        f"STATIC={r['static_cos']:+.6f} | "
        f"NATURAL={r['natural_cos']:+.6f} | "
        f"R={r['recoded_cos']:+.6f} | "
        f"ΔR-NAT={r['delta_recoded_vs_natural']:+.6f} | "
        f"V-ceiling={r['V_ceiling_cos']:+.6f} | "
        f"coeff={r['coefficient_cos']:+.6f}"
    )

FINAL_MEAN_RECODED = float(
    np.mean([
        r["recoded_cos"]
        for r in FINAL_ROWS
    ])
)

FINAL_MEAN_NATURAL = float(
    np.mean([
        r["natural_cos"]
        for r in FINAL_ROWS
    ])
)

FINAL_MEAN_STATIC = float(
    np.mean([
        r["static_cos"]
        for r in FINAL_ROWS
    ])
)

FINAL_MEAN_CEILING = float(
    np.mean([
        r["V_ceiling_cos"]
        for r in FINAL_ROWS
    ])
)

FINAL_DELTA_NATURAL = (
    FINAL_MEAN_RECODED
    - FINAL_MEAN_NATURAL
)

FINAL_DELTA_STATIC = (
    FINAL_MEAN_RECODED
    - FINAL_MEAN_STATIC
)

print("\nFINAL MEANS")

print(
    "STATIC      : "
    f"{FINAL_MEAN_STATIC:+.6f}"
)

print(
    "NATURAL I   : "
    f"{FINAL_MEAN_NATURAL:+.6f}"
)

print(
    "LEARNED R   : "
    f"{FINAL_MEAN_RECODED:+.6f}"
)

print(
    "V CEILING   : "
    f"{FINAL_MEAN_CEILING:+.6f}"
)

print(
    "R - NATURAL : "
    f"{FINAL_DELTA_NATURAL:+.6f}"
)

print(
    "R - STATIC  : "
    f"{FINAL_DELTA_STATIC:+.6f}"
)

# =============================================================================
# CHAINED FINAL
#
# Start only from true A3.
#
# Each adjacent operator:
#
#   current hidden direction
#       -> source U coordinates
#       -> learned R
#       -> target V coordinates
#
# No true downstream ABOVE/BELOW compass is used to construct the chain.
# True downstream compasses are read only for evaluation.
# =============================================================================

print("\n[11/16] FINAL chained held-out transport...")

CHAIN_ROWS = []

z_natural = (
    COMPASS["ABOVE_BELOW"][3]
    .clone()
)

z_recoded = (
    COMPASS["ABOVE_BELOW"][3]
    .clone()
)

for pair in SEGMENTS:
    a, b = pair

    U = (
        ATLAS[pair]["U"]
        [:, :LOCKED_K]
    )

    V = (
        ATLAS[pair]["V"]
        [:, :LOCKED_K]
    )

    R = OPERATORS[
        (
            LOCKED_K,
            LOCKED_RIDGE,
            pair,
        )
    ]["R"]

    # TEST156 natural identity chain.
    natural_coeff = (
        U.T @ z_natural
    )

    natural_raw = (
        V @ natural_coeff
    )

    z_natural = (
        normalized_or_zero(
            natural_raw
        )
    )

    # TEST157 learned recoding chain.
    recoded_coeff = (
        U.T @ z_recoded
    )

    recoded_target_coeff = (
        R @ recoded_coeff
    )

    recoded_raw = (
        V @ recoded_target_coeff
    )

    z_recoded = (
        normalized_or_zero(
            recoded_raw
        )
    )

    true_target = (
        COMPASS[
            "ABOVE_BELOW"
        ][b]
    )

    natural_cos = cosine_safe(
        z_natural,
        true_target,
    )

    recoded_cos = cosine_safe(
        z_recoded,
        true_target,
    )

    source_capture_natural = float(
        natural_coeff.square()
        .sum()
        .item()
    )

    source_capture_recoded = float(
        recoded_coeff.square()
        .sum()
        .item()
    )

    row = {
        "source_layer": a,
        "target_layer": b,
        "K": LOCKED_K,
        "ridge": LOCKED_RIDGE,

        "natural_chain_cos":
            natural_cos,

        "recoded_chain_cos":
            recoded_cos,

        "delta_recoded_vs_natural":
            recoded_cos
            - natural_cos,

        "natural_source_capture":
            source_capture_natural,

        "recoded_source_capture":
            source_capture_recoded,

        "natural_raw_norm":
            float(
                natural_raw.norm()
                .item()
            ),

        "recoded_raw_norm":
            float(
                recoded_raw.norm()
                .item()
            ),
    }

    CHAIN_ROWS.append(row)

    print(
        f"L{a}->L{b} | "
        f"NATURAL chain="
        f"{natural_cos:+.6f} | "
        f"R chain="
        f"{recoded_cos:+.6f} | "
        f"Δ="
        f"{recoded_cos-natural_cos:+.6f} | "
        f"capR="
        f"{source_capture_recoded:.6f}"
    )

FINAL_CHAIN_NATURAL = float(
    np.mean([
        r["natural_chain_cos"]
        for r in CHAIN_ROWS
    ])
)

FINAL_CHAIN_RECODED = float(
    np.mean([
        r["recoded_chain_cos"]
        for r in CHAIN_ROWS
    ])
)

FINAL_CHAIN_L19_NATURAL = float(
    CHAIN_ROWS[-1][
        "natural_chain_cos"
    ]
)

FINAL_CHAIN_L19_RECODED = float(
    CHAIN_ROWS[-1][
        "recoded_chain_cos"
    ]
)

# =============================================================================
# LEAVE-ONE-TRAIN-AXIS-OUT
#
# This is not used for selection.
#
# It checks whether R is merely memorizing the six TRAIN directions.
# For each TRAIN axis:
#   fit R using the other five
#   predict the omitted axis.
# =============================================================================

print("\n[12/16] Leave-one-TRAIN-axis-out recoding check...")

LOO_ROWS = []

def fit_operator_from_axes(
    pair,
    K,
    ridge,
    fit_axes,
):
    src_cols = []
    tgt_cols = []

    for axis_name in fit_axes:
        c_src, c_tgt = (
            semantic_coefficients(
                axis_name,
                pair,
                K,
            )
        )

        src_cols.append(c_src)
        tgt_cols.append(c_tgt)

    Csrc = torch.stack(
        src_cols,
        dim=1,
    )

    Ctgt = torch.stack(
        tgt_cols,
        dim=1,
    )

    if ridge == 0.0:
        R = (
            Ctgt
            @ torch.linalg.pinv(
                Csrc
            )
        )

    else:
        gram = (
            Csrc
            @ Csrc.T
        )

        eye = torch.eye(
            K,
            device=DEVICE,
            dtype=torch.float32,
        )

        R = (
            Ctgt
            @ Csrc.T
            @ torch.linalg.solve(
                gram
                + float(ridge) * eye,
                eye,
            )
        )

    return R

for held_axis in TRAIN_AXES:
    fit_axes = [
        x
        for x in TRAIN_AXES
        if x != held_axis
    ]

    vals = []

    for pair in SEGMENTS:
        R_loo = (
            fit_operator_from_axes(
                pair,
                LOCKED_K,
                LOCKED_RIDGE,
                fit_axes,
            )
        )

        r = evaluate_axis(
            held_axis,
            pair,
            LOCKED_K,
            R_loo,
        )

        delta = (
            r["recoded_cos"]
            - r["natural_cos"]
        )

        vals.append(
            r["recoded_cos"]
        )

        LOO_ROWS.append({
            "held_axis":
                held_axis,

            "source_layer":
                pair[0],

            "target_layer":
                pair[1],

            "K":
                LOCKED_K,

            "ridge":
                LOCKED_RIDGE,

            "natural_cos":
                r["natural_cos"],

            "recoded_cos":
                r["recoded_cos"],

            "delta_vs_natural":
                delta,
        })

    print(
        f"{held_axis:<18} "
        f"mean held-out cos="
        f"{np.mean(vals):+.6f}"
    )

LOO_MEAN = float(
    np.mean([
        r["recoded_cos"]
        for r in LOO_ROWS
    ])
)

LOO_NATURAL_MEAN = float(
    np.mean([
        r["natural_cos"]
        for r in LOO_ROWS
    ])
)

# =============================================================================
# OPERATOR X-RAY
#
# Compare learned R to identity.
#
# R != I is not by itself evidence of semantic generalization.
# It is only a geometric diagnostic.
# =============================================================================

print("\n[13/16] Operator X-ray...")

XRAY_ROWS = []

for pair in SEGMENTS:
    R = OPERATORS[
        (
            LOCKED_K,
            LOCKED_RIDGE,
            pair,
        )
    ]["R"]

    I = torch.eye(
        LOCKED_K,
        device=DEVICE,
        dtype=torch.float32,
    )

    delta = (
        R - I
    )

    sv = torch.linalg.svdvals(
        R
    )

    identity_distance = float(
        torch.linalg.matrix_norm(
            delta
        ).item()
        /
        (
            torch.linalg.matrix_norm(
                I
            ).item()
            + EPS
        )
    )

    trace_norm = float(
        torch.trace(R).item()
        / float(LOCKED_K)
    )

    det_sign = None

    if LOCKED_K <= 32:
        sign, logabsdet = (
            torch.linalg.slogdet(
                R
            )
        )

        det_sign = float(
            sign.item()
        )

        logabsdet = float(
            logabsdet.item()
        )
    else:
        logabsdet = None

    row = {
        "source_layer":
            pair[0],

        "target_layer":
            pair[1],

        "K":
            LOCKED_K,

        "ridge":
            LOCKED_RIDGE,

        "relative_distance_from_identity":
            identity_distance,

        "trace_over_K":
            trace_norm,

        "largest_singular":
            float(
                sv[0].item()
            ),

        "smallest_singular":
            float(
                sv[-1].item()
            ),

        "effective_rank":
            int(
                torch.linalg.matrix_rank(
                    R,
                    tol=1e-6,
                ).item()
            ),

        "det_sign":
            det_sign,

        "logabsdet":
            logabsdet,
    }

    XRAY_ROWS.append(row)

    print(
        f"L{pair[0]}->L{pair[1]} | "
        f"||R-I||/||I||="
        f"{identity_distance:.6f} | "
        f"trace/K="
        f"{trace_norm:+.6f} | "
        f"smax="
        f"{sv[0].item():.6f} | "
        f"smin="
        f"{sv[-1].item():.6f} | "
        f"rank="
        f"{row['effective_rank']}"
    )

# =============================================================================
# DIAGNOSIS
#
# Descriptive classification only.
# No threshold below was used to choose K/ridge.
# =============================================================================

print("\n[14/16] Mechanistic diagnosis...")

VAL_GAIN = float(
    WINNER[
        "delta_vs_natural"
    ]
)

FINAL_GAIN = float(
    FINAL_DELTA_NATURAL
)

if (
    VAL_GAIN > 0.10
    and FINAL_GAIN > 0.10
):
    DIAGNOSIS = (
        "HELD_OUT_SEMANTIC_RECODING_SUPPORTED"
    )

    DIAGNOSIS_TEXT = (
        "A semantic recoding operator learned only from other semantic "
        "axes improves reconstruction of the unseen ABOVE/BELOW axis "
        "relative to the natural identity correspondence. This supports "
        "the existence of a cross-axis shared recoding structure inside "
        "the tested natural transport coordinates."
    )

elif (
    VAL_GAIN > 0.05
    and FINAL_GAIN <= 0.0
):
    DIAGNOSIS = (
        "VALIDATION_RECODING_DID_NOT_GENERALIZE"
    )

    DIAGNOSIS_TEXT = (
        "The learned recoding operator improved the validation axes but "
        "did not improve the untouched ABOVE/BELOW axis. The tested R "
        "therefore does not demonstrate held-out semantic generalization."
    )

elif (
    FINAL_GAIN > 0.0
    and FINAL_GAIN <= 0.10
):
    DIAGNOSIS = (
        "WEAK_OR_MIXED_HELD_OUT_RECODING"
    )

    DIAGNOSIS_TEXT = (
        "The learned operator shows a positive but limited held-out gain "
        "over the natural identity correspondence. The result is "
        "descriptive and does not yet establish a strong shared semantic "
        "recoding law."
    )

else:
    DIAGNOSIS = (
        "SHARED_RECODING_NOT_SUPPORTED"
    )

    DIAGNOSIS_TEXT = (
        "The learned cross-axis recoding operator does not improve the "
        "untouched ABOVE/BELOW semantic mapping sufficiently relative "
        "to the natural identity correspondence. A single shared linear "
        "R in the tested low-rank coordinates is therefore not supported "
        "as the missing semantic transport object."
    )

print(
    "VAL gain over natural   : "
    f"{VAL_GAIN:+.6f}"
)

print(
    "FINAL gain over natural : "
    f"{FINAL_GAIN:+.6f}"
)

print(
    "LOO TRAIN R mean        : "
    f"{LOO_MEAN:+.6f}"
)

print(
    "LOO natural mean        : "
    f"{LOO_NATURAL_MEAN:+.6f}"
)

print(
    "FINAL chain natural     : "
    f"{FINAL_CHAIN_NATURAL:+.6f}"
)

print(
    "FINAL chain recoded     : "
    f"{FINAL_CHAIN_RECODED:+.6f}"
)

print(
    "FINAL L19 chain natural : "
    f"{FINAL_CHAIN_L19_NATURAL:+.6f}"
)

print(
    "FINAL L19 chain recoded : "
    f"{FINAL_CHAIN_L19_RECODED:+.6f}"
)

print(
    "\nDIAGNOSIS:",
    DIAGNOSIS,
)

print(
    "\n",
    DIAGNOSIS_TEXT,
)

# =============================================================================
# SAVE
# =============================================================================

print("\n[15/16] Saving CSV / JSON / NPZ...")

OPERATOR_CSV = (
    ROOT / "OPERATOR_TRAIN.csv"
)

VAL_CSV = (
    ROOT / "VAL_TOURNAMENT.csv"
)

VAL_SUMMARY_CSV = (
    ROOT / "VAL_SUMMARY.csv"
)

FINAL_CSV = (
    ROOT / "FINAL_HELDOUT.csv"
)

CHAIN_CSV = (
    ROOT / "FINAL_CHAIN.csv"
)

LOO_CSV = (
    ROOT / "TRAIN_LOO.csv"
)

XRAY_CSV = (
    ROOT / "OPERATOR_XRAY.csv"
)

REPORT_JSON = (
    ROOT / "REPORT.json"
)

NPZ_PATH = (
    ROOT
    / "TEST157_HELDOUT_SEMANTIC_RECODING_OPERATOR.npz"
)

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
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(rows)

save_csv(
    OPERATOR_CSV,
    OPERATOR_ROWS,
)

save_csv(
    VAL_CSV,
    VAL_ROWS,
)

save_csv(
    VAL_SUMMARY_CSV,
    VAL_SUMMARY,
)

save_csv(
    FINAL_CSV,
    FINAL_ROWS,
)

save_csv(
    CHAIN_CSV,
    CHAIN_ROWS,
)

save_csv(
    LOO_CSV,
    LOO_ROWS,
)

save_csv(
    XRAY_CSV,
    XRAY_ROWS,
)

NPZ = {}

for pair in SEGMENTS:
    a, b = pair

    NPZ[
        f"U_L{a}_L{b}"
    ] = (
        ATLAS[pair]["U"]
        .cpu()
        .numpy()
    )

    NPZ[
        f"S_L{a}_L{b}"
    ] = (
        ATLAS[pair]["S"]
        .cpu()
        .numpy()
    )

    NPZ[
        f"V_L{a}_L{b}"
    ] = (
        ATLAS[pair]["V"]
        .cpu()
        .numpy()
    )

    R = OPERATORS[
        (
            LOCKED_K,
            LOCKED_RIDGE,
            pair,
        )
    ]["R"]

    NPZ[
        f"R_LOCKED_L{a}_L{b}"
    ] = (
        R.detach()
        .cpu()
        .numpy()
    )

for axis_name in (
    TRAIN_AXES
    + VAL_AXES
    + FINAL_AXES
):
    for L in LAYERS:
        NPZ[
            f"COMPASS_{axis_name}_L{L}"
        ] = (
            COMPASS[
                axis_name
            ][L]
            .detach()
            .cpu()
            .numpy()
        )

np.savez_compressed(
    NPZ_PATH,
    **NPZ,
)

REPORT = {
    "test":
        "TEST 157",

    "title":
        "HELD-OUT SEMANTIC RECODING OPERATOR",

    "model":
        MODEL_ID,

    "gpu":
        torch.cuda.get_device_name(0),

    "hidden":
        HIDDEN,

    "layers":
        LAYERS,

    "segments":
        SEGMENTS,

    "search_n":
        SEARCH_N,

    "fingerprint_max_relative_error":
        fingerprint_max_rel,

    "fingerprint_pass":
        FP_PASS,

    "train_axes":
        TRAIN_AXES,

    "val_axes":
        VAL_AXES,

    "final_axes":
        FINAL_AXES,

    "k_values":
        K_VALUES,

    "ridges":
        RIDGES,

    "locked": {
        "K":
            LOCKED_K,

        "ridge":
            LOCKED_RIDGE,

        "val_mean_recoded_cos":
            WINNER[
                "mean_recoded_cos"
            ],

        "val_mean_natural_cos":
            WINNER[
                "mean_natural_cos"
            ],

        "val_delta_vs_natural":
            WINNER[
                "delta_vs_natural"
            ],
    },

    "final": {
        "mean_static_cos":
            FINAL_MEAN_STATIC,

        "mean_natural_cos":
            FINAL_MEAN_NATURAL,

        "mean_recoded_cos":
            FINAL_MEAN_RECODED,

        "mean_V_ceiling":
            FINAL_MEAN_CEILING,

        "delta_recoded_vs_natural":
            FINAL_DELTA_NATURAL,

        "delta_recoded_vs_static":
            FINAL_DELTA_STATIC,

        "chain_natural_mean":
            FINAL_CHAIN_NATURAL,

        "chain_recoded_mean":
            FINAL_CHAIN_RECODED,

        "chain_L19_natural":
            FINAL_CHAIN_L19_NATURAL,

        "chain_L19_recoded":
            FINAL_CHAIN_L19_RECODED,
    },

    "loo": {
        "mean_recoded_cos":
            LOO_MEAN,

        "mean_natural_cos":
            LOO_NATURAL_MEAN,

        "delta":
            LOO_MEAN
            - LOO_NATURAL_MEAN,
    },

    "diagnosis":
        DIAGNOSIS,

    "diagnosis_text":
        DIAGNOSIS_TEXT,

    "interpretation_contract": [
        "The TEST156 natural SEARCH corpus and fingerprint are preserved.",
        "Natural transport U/S/V is constructed without semantic targets.",
        "R is fitted only from TRAIN semantic axes.",
        "VAL axes select K and ridge.",
        "ABOVE/BELOW does not participate in R fitting or hyperparameter selection.",
        "FINAL ABOVE/BELOW is used only after K and ridge are locked.",
        "The natural TEST156 semantic correspondence is represented by R=I in U/V coordinates.",
        "A positive FINAL R-vs-natural gain is evidence for held-out generalization only within this model, corpus family, layer set, semantic-axis family, and linear low-rank operator class.",
        "A negative result rejects this tested shared linear R; it does not prove that no nonlinear, state-dependent, axis-conditioned, or higher-rank semantic recoding exists.",
        "No steering or behavioral intervention occurs in TEST157.",
    ],
}

with open(
    REPORT_JSON,
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
# COMPLETE
# =============================================================================

print("\n[16/16] COMPLETE")
print("=" * 154)

print(
    "Atlas fingerprint max error : "
    f"{100.0 * fingerprint_max_rel:.4f}%"
)

print(
    "Locked K                    : "
    f"{LOCKED_K}"
)

print(
    "Locked ridge                : "
    f"{LOCKED_RIDGE:g}"
)

print(
    "VAL natural mean            : "
    f"{WINNER['mean_natural_cos']:+.6f}"
)

print(
    "VAL learned-R mean          : "
    f"{WINNER['mean_recoded_cos']:+.6f}"
)

print(
    "VAL R - natural             : "
    f"{VAL_GAIN:+.6f}"
)

print(
    "FINAL static mean           : "
    f"{FINAL_MEAN_STATIC:+.6f}"
)

print(
    "FINAL natural mean          : "
    f"{FINAL_MEAN_NATURAL:+.6f}"
)

print(
    "FINAL learned-R mean        : "
    f"{FINAL_MEAN_RECODED:+.6f}"
)

print(
    "FINAL R - natural           : "
    f"{FINAL_GAIN:+.6f}"
)

print(
    "FINAL V ceiling             : "
    f"{FINAL_MEAN_CEILING:+.6f}"
)

print(
    "FINAL chain natural         : "
    f"{FINAL_CHAIN_NATURAL:+.6f}"
)

print(
    "FINAL chain learned-R       : "
    f"{FINAL_CHAIN_RECODED:+.6f}"
)

print(
    "FINAL L19 natural chain     : "
    f"{FINAL_CHAIN_L19_NATURAL:+.6f}"
)

print(
    "FINAL L19 learned-R chain   : "
    f"{FINAL_CHAIN_L19_RECODED:+.6f}"
)

print(
    "TRAIN LOO natural           : "
    f"{LOO_NATURAL_MEAN:+.6f}"
)

print(
    "TRAIN LOO learned-R         : "
    f"{LOO_MEAN:+.6f}"
)

print(
    "\nDiagnosis                    : "
    f"{DIAGNOSIS}"
)

print("\nSaved:")
print(" ", OPERATOR_CSV)
print(" ", VAL_CSV)
print(" ", VAL_SUMMARY_CSV)
print(" ", FINAL_CSV)
print(" ", CHAIN_CSV)
print(" ", LOO_CSV)
print(" ", XRAY_CSV)
print(" ", REPORT_JSON)
print(" ", NPZ_PATH)

print("=" * 154)
print("TEST 157 COMPLETE")
print("=" * 154)
