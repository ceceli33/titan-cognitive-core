# =============================================================================
# TEST 156 — SEMANTIC TRANSPORT X-RAY
# NATURAL TRANSPORT vs TRUE LAYER-LOCAL SEMANTIC COORDINATES
#
# Qwen/Qwen2.5-7B-Instruct · Frozen · BF16 · FP32 analysis
#
# LOCKED AXIS:
#   ABOVE <-> BELOW
#
# QUESTION:
#   Does the natural cross-layer transport atlas discovered in TEST152/154
#   actually map a semantic compass at layer a into the independently
#   extracted semantic compass at layer b?
#
# DIAGNOSIS:
#
#   A) high transport->true-compass cosine
#      -> atlas maps semantic coordinates; TEST155 failure is downstream
#         of direction construction (injection/dose/composition).
#
#   B) low transport->true-compass cosine
#      -> natural covariance transport is not a semantic transport operator
#         for this axis.
#
#   C) direct high, chained low
#      -> sequential composition accumulates transport error.
#
#   D) SIGMA weighting > unweighted
#      -> transport spectrum contains useful semantic weighting information.
#
# No steering · No intervention · No training · No LoRA
# No gradients · No JVP/VJP · No optimizer
# =============================================================================

import os, sys, subprocess, math, json, csv, random, warnings
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

SEED = 156
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

LAYERS = [3, 6, 10, 14, 19]
SEGMENTS = [(3, 6), (6, 10), (10, 14), (14, 19)]

SEARCH_N = 192

K_VALUES = [4, 8, 16, 32]

METHODS = [
    "UNWEIGHTED",
    "SQRT_SIGMA",
    "SIGMA",
    "PROCRUSTES",
]

EPS = 1e-10

ROOT = Path("/content/TEST156_SEMANTIC_TRANSPORT_XRAY")
ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type != "cuda":
    raise RuntimeError("TEST156 requires CUDA.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 150)
print("TEST 156 — SEMANTIC TRANSPORT X-RAY")
print("NATURAL TRANSPORT vs TRUE LAYER-LOCAL SEMANTIC COORDINATES")
print("=" * 150)

# =============================================================================
# MODEL
# =============================================================================

print("\n[1/14] Loading frozen model...")

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
print("K              :", K_VALUES)
print("Methods        :", METHODS)

# =============================================================================
# CHAT
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
# LOCKED ABOVE / BELOW COMPASS
# =============================================================================

POS = [
    "The red marker is above the blue marker.",
    "The lamp is above the wooden table.",
    "The bird is above the branch.",
    "The sign is above the doorway.",
    "The shelf is above the cabinet.",
    "The cloud is above the hill.",
    "The clock is above the desk.",
    "The picture is above the sofa.",
]

NEG = [
    "The red marker is below the blue marker.",
    "The lamp is below the wooden table.",
    "The bird is below the branch.",
    "The sign is below the doorway.",
    "The shelf is below the cabinet.",
    "The cloud is below the hill.",
    "The clock is below the desk.",
    "The picture is below the sofa.",
]

# =============================================================================
# TEST152/154 NATURAL CORPUS FAMILY
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

# Preserve TEST155 corpus construction.
rng = np.random.default_rng(42)
rng.shuffle(natural_pool)

natural_pool = list(
    dict.fromkeys(natural_pool)
)

SEARCH_TEXTS = natural_pool[:SEARCH_N]

assert len(SEARCH_TEXTS) == SEARCH_N

# =============================================================================
# ACTIVATION CAPTURE
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
            bank[L].append(row[L])

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
# NATURAL ATLAS
# =============================================================================

print("\n[2/14] Capturing natural SEARCH atlas...")

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
# SAMPLE-SPACE CROSS-COVARIANCE SVD
#
# X^T = Qx Rx
# Y^T = Qy Ry
#
# Cxy = X^T Y/(N-1)
#     = Qx [Rx Ry^T/(N-1)] Qy^T
#
# No 3584 x 3584 SVD.
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

print("\n[3/14] Reconstructing natural transport atlas...")

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
# TEST154/155 FINGERPRINT CHECK
# =============================================================================

print("\n[4/14] Fingerprint check...")

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

if fingerprint_max_rel > 0.015:
    raise RuntimeError(
        "Atlas fingerprint drift > 1.5%. "
        "TEST156 aborted before semantic diagnosis."
    )

print("Fingerprint: PASS")

# =============================================================================
# TRUE LAYER-LOCAL COMPASSES
# =============================================================================

print("\n[5/14] Extracting immutable layer-local compasses...")

POS_BANK = capture_bank(
    POS,
    "POS",
)

NEG_BANK = capture_bank(
    NEG,
    "NEG",
)

COMPASS = {}

for L in LAYERS:
    a = (
        POS_BANK[L].mean(dim=0)
        - NEG_BANK[L].mean(dim=0)
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    a = a / (
        a.norm() + EPS
    )

    COMPASS[L] = a

print("\nTRUE COMPASS ADJACENT COSINES")

for a, b in SEGMENTS:
    c = F.cosine_similarity(
        COMPASS[a][None],
        COMPASS[b][None],
    ).item()

    print(
        f"cos(A{a}, A{b}) = "
        f"{c:+.6f}"
    )

# =============================================================================
# TRANSPORT
#
# UNWEIGHTED:
#   z_b = V U^T z_a
#
# SQRT_SIGMA:
#   z_b = V sqrt(S/S1) U^T z_a
#
# SIGMA:
#   z_b = V (S/S1) U^T z_a
#
# PROCRUSTES:
#   For orthonormal paired cross-cov modes this has the same V U^T
#   action inside the selected rank-K support. It is retained explicitly
#   as a named geometric reference and reported rather than hidden.
#
# All outputs normalized only AFTER raw projection diagnostics are stored.
# =============================================================================

def transport_raw(
    z,
    pair,
    K,
    method,
):
    U = ATLAS[pair]["U"][:, :K]
    S = ATLAS[pair]["S"][:K]
    V = ATLAS[pair]["V"][:, :K]

    coeff = U.T @ z

    source_projection_energy = float(
        coeff.square()
        .sum()
        .item()
    )

    if method == "UNWEIGHTED":
        weighted_coeff = coeff

    elif method == "SQRT_SIGMA":
        w = torch.sqrt(
            S / (
                S[0] + EPS
            )
        )

        weighted_coeff = (
            coeff * w
        )

    elif method == "SIGMA":
        w = (
            S / (
                S[0] + EPS
            )
        )

        weighted_coeff = (
            coeff * w
        )

    elif method == "PROCRUSTES":
        weighted_coeff = coeff

    else:
        raise ValueError(method)

    raw = V @ weighted_coeff

    raw_norm = float(
        raw.norm().item()
    )

    if raw_norm > 1e-12:
        normalized = (
            raw / raw.norm()
        )
    else:
        normalized = raw.clone()

    return {
        "raw": raw,
        "normalized": normalized,
        "coeff": coeff,
        "weighted_coeff": weighted_coeff,
        "source_projection_energy":
            source_projection_energy,
        "source_projection_fraction":
            source_projection_energy,
        "raw_transport_norm":
            raw_norm,
    }

# =============================================================================
# DIRECT SEGMENT X-RAY
#
# Each segment starts from the TRUE source compass A_a.
#
# This asks:
#
#   if I know the correct semantic coordinate at layer a,
#   does natural transport reconstruct the independently extracted A_b?
#
# =============================================================================

print("\n[6/14] Direct semantic transport X-ray...")

DIRECT_ROWS = []

for a, b in SEGMENTS:
    true_source = COMPASS[a]
    true_target = COMPASS[b]

    print(
        f"\n--- DIRECT L{a} -> L{b} ---"
    )

    for K in K_VALUES:
        U = ATLAS[(a, b)]["U"][:, :K]
        V = ATLAS[(a, b)]["V"][:, :K]

        source_proj = U @ (
            U.T @ true_source
        )

        target_proj = V @ (
            V.T @ true_target
        )

        source_capture = float(
            source_proj.square()
            .sum()
            .item()
        )

        target_capture = float(
            target_proj.square()
            .sum()
            .item()
        )

        for method in METHODS:
            result = transport_raw(
                true_source,
                (a, b),
                K,
                method,
            )

            pred = result[
                "normalized"
            ]

            cos_true = float(
                F.cosine_similarity(
                    pred[None],
                    true_target[None],
                ).item()
            ) if pred.norm() > 0 else 0.0

            # Does transported semantic vector land inside the natural
            # target response support regardless of exact semantic alignment?
            target_support_capture = float(
                (
                    V.T @ pred
                ).square()
                .sum()
                .item()
            ) if pred.norm() > 0 else 0.0

            # Compare prediction to simple same-coordinate source vector.
            static_cos_true = float(
                F.cosine_similarity(
                    true_source[None],
                    true_target[None],
                ).item()
            )

            improvement_over_static = (
                cos_true
                - static_cos_true
            )

            row = {
                "mode": "DIRECT",
                "source_layer": a,
                "target_layer": b,
                "K": K,
                "method": method,
                "cos_pred_true": cos_true,
                "cos_static_true":
                    static_cos_true,
                "delta_vs_static":
                    improvement_over_static,
                "source_compass_capture_U":
                    source_capture,
                "target_compass_capture_V":
                    target_capture,
                "transported_capture_V":
                    target_support_capture,
                "raw_transport_norm":
                    result[
                        "raw_transport_norm"
                    ],
            }

            DIRECT_ROWS.append(row)

            print(
                f"K={K:<3d} "
                f"{method:<12} | "
                f"cos(pred,A{b})="
                f"{cos_true:+.6f} | "
                f"static="
                f"{static_cos_true:+.6f} | "
                f"Δ="
                f"{improvement_over_static:+.6f} | "
                f"capU="
                f"{source_capture:.4f} | "
                f"capV(A{b})="
                f"{target_capture:.4f} | "
                f"|raw|="
                f"{result['raw_transport_norm']:.4f}"
            )

# =============================================================================
# CHAINED X-RAY
#
# Starts only from true A3.
# No later semantic compass is used for transport.
#
# A3 -> Ahat6 -> Ahat10 -> Ahat14 -> Ahat19
# =============================================================================

print("\n[7/14] Chained semantic transport X-ray...")

CHAIN_ROWS = []

CHAIN_VECTORS = {}

for K in K_VALUES:
    for method in METHODS:
        z = COMPASS[3].clone()

        CHAIN_VECTORS[
            (K, method, 3)
        ] = z.detach().cpu()

        cumulative_product = 1.0

        print(
            f"\nCHAIN K={K} "
            f"METHOD={method}"
        )

        for a, b in SEGMENTS:
            result = transport_raw(
                z,
                (a, b),
                K,
                method,
            )

            pred = result[
                "normalized"
            ]

            cos_true = float(
                F.cosine_similarity(
                    pred[None],
                    COMPASS[b][None],
                ).item()
            ) if pred.norm() > 0 else 0.0

            source_capture = float(
                result[
                    "source_projection_fraction"
                ]
            )

            cumulative_product *= (
                max(
                    source_capture,
                    0.0,
                )
            )

            row = {
                "mode": "CHAIN",
                "start_layer": 3,
                "source_layer": a,
                "target_layer": b,
                "K": K,
                "method": method,
                "cos_pred_true":
                    cos_true,
                "source_capture":
                    source_capture,
                "raw_transport_norm":
                    result[
                        "raw_transport_norm"
                    ],
                "capture_product":
                    cumulative_product,
            }

            CHAIN_ROWS.append(row)

            CHAIN_VECTORS[
                (K, method, b)
            ] = (
                pred.detach()
                .cpu()
            )

            print(
                f"L{a}->L{b} | "
                f"cos(chain,A{b})="
                f"{cos_true:+.6f} | "
                f"source_capture="
                f"{source_capture:.6f} | "
                f"|raw|="
                f"{result['raw_transport_norm']:.6f} | "
                f"capture_product="
                f"{cumulative_product:.8f}"
            )

            if pred.norm() <= 1e-12:
                z = pred
                break

            z = pred

# =============================================================================
# DIRECT vs CHAIN ERROR ACCUMULATION
# =============================================================================

print("\n[8/14] Direct vs chained error accumulation...")

COMPARE_ROWS = []

for K in K_VALUES:
    for method in METHODS:
        for a, b in SEGMENTS:
            direct = next(
                r
                for r in DIRECT_ROWS
                if (
                    r["source_layer"] == a
                    and r["target_layer"] == b
                    and r["K"] == K
                    and r["method"] == method
                )
            )

            chain = next(
                r
                for r in CHAIN_ROWS
                if (
                    r["source_layer"] == a
                    and r["target_layer"] == b
                    and r["K"] == K
                    and r["method"] == method
                )
            )

            gap = (
                chain["cos_pred_true"]
                - direct["cos_pred_true"]
            )

            COMPARE_ROWS.append({
                "source_layer": a,
                "target_layer": b,
                "K": K,
                "method": method,
                "direct_cos":
                    direct["cos_pred_true"],
                "chain_cos":
                    chain["cos_pred_true"],
                "chain_minus_direct":
                    gap,
            })

            print(
                f"K={K:<3d} "
                f"{method:<12} "
                f"L{a}->L{b} | "
                f"direct="
                f"{direct['cos_pred_true']:+.6f} "
                f"chain="
                f"{chain['cos_pred_true']:+.6f} "
                f"gap={gap:+.6f}"
            )

# =============================================================================
# SEMANTIC CAPTURE X-RAY
#
# Critical distinction:
#
# 1. Does source semantic compass even live in U_K?
# 2. Does target semantic compass live in V_K?
# 3. If both do, does the U_i <-> V_i mode correspondence map them correctly?
#
# A low transport cosine can arise from different mechanisms.
# =============================================================================

print("\n[9/14] Semantic support decomposition...")

SUPPORT_ROWS = []

for a, b in SEGMENTS:
    print(
        f"\n--- SUPPORT L{a}->L{b} ---"
    )

    for K in K_VALUES:
        U = ATLAS[(a, b)]["U"][:, :K]
        V = ATLAS[(a, b)]["V"][:, :K]

        A_src = COMPASS[a]
        A_tgt = COMPASS[b]

        c_src = U.T @ A_src
        c_tgt = V.T @ A_tgt

        src_capture = float(
            c_src.square()
            .sum()
            .item()
        )

        tgt_capture = float(
            c_tgt.square()
            .sum()
            .item()
        )

        # Coefficient agreement tests whether semantic coordinates use
        # corresponding natural modes similarly.
        if (
            c_src.norm() > 1e-12
            and c_tgt.norm() > 1e-12
        ):
            coeff_cos = float(
                F.cosine_similarity(
                    c_src[None],
                    c_tgt[None],
                ).item()
            )
        else:
            coeff_cos = 0.0

        # Best possible target alignment obtainable from V_K alone.
        tgt_proj = V @ c_tgt

        if tgt_proj.norm() > 1e-12:
            best_v_cos = float(
                F.cosine_similarity(
                    (
                        tgt_proj
                        / tgt_proj.norm()
                    )[None],
                    A_tgt[None],
                ).item()
            )
        else:
            best_v_cos = 0.0

        # Unweighted actual semantic map.
        pred = V @ c_src

        if pred.norm() > 1e-12:
            actual_cos = float(
                F.cosine_similarity(
                    (
                        pred
                        / pred.norm()
                    )[None],
                    A_tgt[None],
                ).item()
            )
        else:
            actual_cos = 0.0

        # Mapping utilization:
        # how much of the theoretical V_K semantic ceiling is achieved?
        utilization = (
            actual_cos
            / (
                best_v_cos + EPS
            )
            if best_v_cos > 1e-8
            else 0.0
        )

        SUPPORT_ROWS.append({
            "source_layer": a,
            "target_layer": b,
            "K": K,
            "source_capture_U":
                src_capture,
            "target_capture_V":
                tgt_capture,
            "coefficient_cosine":
                coeff_cos,
            "best_possible_V_cos":
                best_v_cos,
            "actual_unweighted_cos":
                actual_cos,
            "mapping_utilization":
                utilization,
        })

        print(
            f"K={K:<3d} | "
            f"A{a} in U="
            f"{src_capture:.6f} | "
            f"A{b} in V="
            f"{tgt_capture:.6f} | "
            f"coeff_cos="
            f"{coeff_cos:+.6f} | "
            f"V-ceiling="
            f"{best_v_cos:+.6f} | "
            f"actual="
            f"{actual_cos:+.6f} | "
            f"util="
            f"{utilization:+.4f}"
        )

# =============================================================================
# MODE-BY-MODE SEMANTIC SPECTRUM
#
# For each natural singular mode i:
#
#   source semantic coefficient = <U_i, A_a>
#   target semantic coefficient = <V_i, A_b>
#
# If signs/magnitudes disagree, covariance modes are not preserving the
# semantic coordinate even when both compasses overlap the subspaces.
# =============================================================================

print("\n[10/14] Mode-by-mode semantic spectrum...")

MODE_ROWS = []

MAX_MODE = 32

for a, b in SEGMENTS:
    U = ATLAS[(a, b)]["U"][:, :MAX_MODE]
    V = ATLAS[(a, b)]["V"][:, :MAX_MODE]
    S = ATLAS[(a, b)]["S"][:MAX_MODE]

    src = U.T @ COMPASS[a]
    tgt = V.T @ COMPASS[b]

    print(
        f"\n--- MODES L{a}->L{b} ---"
    )

    print(
        f"{'i':>3} "
        f"{'sigma':>11} "
        f"{'<U,Aa>':>12} "
        f"{'<V,Ab>':>12} "
        f"{'product':>12}"
    )

    for i in range(MAX_MODE):
        u = float(
            src[i].item()
        )

        v = float(
            tgt[i].item()
        )

        s = float(
            S[i].item()
        )

        product = u * v

        MODE_ROWS.append({
            "source_layer": a,
            "target_layer": b,
            "mode": i + 1,
            "sigma": s,
            "source_semantic_coeff": u,
            "target_semantic_coeff": v,
            "coefficient_product":
                product,
        })

        if i < 16:
            print(
                f"{i + 1:>3d} "
                f"{s:>11.6f} "
                f"{u:>+12.6f} "
                f"{v:>+12.6f} "
                f"{product:>+12.6f}"
            )

# =============================================================================
# SUMMARY MATRIX
# =============================================================================

print("\n[11/14] Summary matrix...")

SUMMARY_ROWS = []

for K in K_VALUES:
    for method in METHODS:
        direct_vals = [
            r["cos_pred_true"]
            for r in DIRECT_ROWS
            if (
                r["K"] == K
                and r["method"] == method
            )
        ]

        chain_vals = [
            r["cos_pred_true"]
            for r in CHAIN_ROWS
            if (
                r["K"] == K
                and r["method"] == method
            )
        ]

        final_chain = next(
            r["cos_pred_true"]
            for r in CHAIN_ROWS
            if (
                r["K"] == K
                and r["method"] == method
                and r["target_layer"] == 19
            )
        )

        mean_direct = float(
            np.mean(direct_vals)
        )

        mean_chain = float(
            np.mean(chain_vals)
        )

        SUMMARY_ROWS.append({
            "K": K,
            "method": method,
            "mean_direct_cos":
                mean_direct,
            "mean_chain_cos":
                mean_chain,
            "final_L19_chain_cos":
                final_chain,
        })

print(
    f"{'K':>4} "
    f"{'METHOD':<13}"
    f"{'DIRECT':>12}"
    f"{'CHAIN':>12}"
    f"{'L19':>12}"
)

for r in SUMMARY_ROWS:
    print(
        f"{r['K']:>4d} "
        f"{r['method']:<13}"
        f"{r['mean_direct_cos']:>+12.6f}"
        f"{r['mean_chain_cos']:>+12.6f}"
        f"{r['final_L19_chain_cos']:>+12.6f}"
    )

# =============================================================================
# DIAGNOSTIC CLASSIFICATION
#
# Descriptive thresholds only.
# They are NOT significance thresholds and do not select a winner.
# =============================================================================

print("\n[12/14] Mechanistic diagnosis...")

# Primary diagnostic uses K16 unweighted because TEST155 FINAL reference
# used RELAY_K16 unweighted. Weighted K32 is separately reported because
# it was the strongest low-energy RELAY configuration on TEST155 VAL.

PRIMARY_K = 16
PRIMARY_METHOD = "UNWEIGHTED"

primary_direct = [
    r
    for r in DIRECT_ROWS
    if (
        r["K"] == PRIMARY_K
        and r["method"] == PRIMARY_METHOD
    )
]

primary_chain = [
    r
    for r in CHAIN_ROWS
    if (
        r["K"] == PRIMARY_K
        and r["method"] == PRIMARY_METHOD
    )
]

mean_direct = float(
    np.mean([
        r["cos_pred_true"]
        for r in primary_direct
    ])
)

mean_chain = float(
    np.mean([
        r["cos_pred_true"]
        for r in primary_chain
    ])
)

final_chain = next(
    r["cos_pred_true"]
    for r in primary_chain
    if r["target_layer"] == 19
)

weighted32 = [
    r
    for r in DIRECT_ROWS
    if (
        r["K"] == 32
        and r["method"] == "SIGMA"
    )
]

weighted32_mean = float(
    np.mean([
        r["cos_pred_true"]
        for r in weighted32
    ])
)

unweighted32 = [
    r
    for r in DIRECT_ROWS
    if (
        r["K"] == 32
        and r["method"] == "UNWEIGHTED"
    )
]

unweighted32_mean = float(
    np.mean([
        r["cos_pred_true"]
        for r in unweighted32
    ])
)

if mean_direct >= 0.70:
    direct_status = "HIGH"
elif mean_direct >= 0.40:
    direct_status = "MODERATE"
else:
    direct_status = "LOW"

if (
    mean_direct >= 0.60
    and mean_chain < mean_direct - 0.20
):
    chain_status = "CHAIN_ERROR_ACCUMULATION"
elif mean_direct < 0.40:
    chain_status = "DIRECT_MAP_MISMATCH"
else:
    chain_status = "MIXED"

sigma_delta = (
    weighted32_mean
    - unweighted32_mean
)

print(
    f"Primary K16 unweighted mean direct cosine : "
    f"{mean_direct:+.6f}"
)

print(
    f"Primary K16 unweighted mean chain cosine  : "
    f"{mean_chain:+.6f}"
)

print(
    f"Primary K16 chain cosine at L19           : "
    f"{final_chain:+.6f}"
)

print(
    f"K32 unweighted mean direct cosine         : "
    f"{unweighted32_mean:+.6f}"
)

print(
    f"K32 SIGMA mean direct cosine              : "
    f"{weighted32_mean:+.6f}"
)

print(
    f"SIGMA - UNWEIGHTED                        : "
    f"{sigma_delta:+.6f}"
)

print(
    f"\nDIRECT STATUS : {direct_status}"
)

print(
    f"CHAIN STATUS  : {chain_status}"
)

if chain_status == "DIRECT_MAP_MISMATCH":
    diagnosis = (
        "Natural cross-covariance transport does not directly reconstruct "
        "the independently extracted layer-local ABOVE/BELOW semantic "
        "coordinates well enough under the primary TEST155 mapping. "
        "TEST155 RELAY failure is therefore consistent with a semantic-map "
        "mismatch rather than disappearance of the underlying perturbation."
    )

elif chain_status == "CHAIN_ERROR_ACCUMULATION":
    diagnosis = (
        "Adjacent semantic transport is individually substantial, but "
        "sequential composition loses alignment. TEST155 RELAY failure is "
        "consistent with accumulated chaining error."
    )

else:
    diagnosis = (
        "Semantic transport shows a mixed pattern. Direction construction "
        "alone does not fully explain TEST155 and injection/composition "
        "effects remain plausible."
    )

print("\nDIAGNOSIS")
print(diagnosis)

# =============================================================================
# SAVE
# =============================================================================

print("\n[13/14] Saving CSV / JSON / NPZ...")

DIRECT_CSV = ROOT / "DIRECT_TRANSPORT.csv"
CHAIN_CSV = ROOT / "CHAIN_TRANSPORT.csv"
SUPPORT_CSV = ROOT / "SEMANTIC_SUPPORT.csv"
MODES_CSV = ROOT / "MODE_SPECTRUM.csv"
SUMMARY_CSV = ROOT / "SUMMARY.csv"
REPORT_JSON = ROOT / "REPORT.json"
NPZ_PATH = ROOT / "TEST156_SEMANTIC_TRANSPORT_XRAY.npz"

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
    DIRECT_CSV,
    DIRECT_ROWS,
)

save_csv(
    CHAIN_CSV,
    CHAIN_ROWS,
)

save_csv(
    SUPPORT_CSV,
    SUPPORT_ROWS,
)

save_csv(
    MODES_CSV,
    MODE_ROWS,
)

save_csv(
    SUMMARY_CSV,
    SUMMARY_ROWS,
)

NPZ = {}

for pair in SEGMENTS:
    a, b = pair

    NPZ[f"U_L{a}_L{b}"] = (
        ATLAS[pair]["U"]
        .cpu()
        .numpy()
    )

    NPZ[f"S_L{a}_L{b}"] = (
        ATLAS[pair]["S"]
        .cpu()
        .numpy()
    )

    NPZ[f"V_L{a}_L{b}"] = (
        ATLAS[pair]["V"]
        .cpu()
        .numpy()
    )

for L in LAYERS:
    NPZ[f"COMPASS_L{L}"] = (
        COMPASS[L]
        .cpu()
        .numpy()
    )

for key, value in CHAIN_VECTORS.items():
    K, method, L = key

    NPZ[
        f"CHAIN_{method}_K{K}_L{L}"
    ] = value.numpy()

np.savez_compressed(
    NPZ_PATH,
    **NPZ,
)

REPORT = {
    "test": "TEST 156",
    "title":
        "SEMANTIC TRANSPORT X-RAY",

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

    "axis":
        "ABOVE <-> BELOW",

    "search_n":
        SEARCH_N,

    "k_values":
        K_VALUES,

    "methods":
        METHODS,

    "fingerprint_max_relative_error":
        fingerprint_max_rel,

    "primary_diagnostic": {
        "K":
            PRIMARY_K,

        "method":
            PRIMARY_METHOD,

        "mean_direct_cos":
            mean_direct,

        "mean_chain_cos":
            mean_chain,

        "L19_chain_cos":
            final_chain,

        "direct_status":
            direct_status,

        "chain_status":
            chain_status,
    },

    "weighted_K32": {
        "unweighted_mean_direct_cos":
            unweighted32_mean,

        "sigma_mean_direct_cos":
            weighted32_mean,

        "sigma_minus_unweighted":
            sigma_delta,
    },

    "diagnosis":
        diagnosis,

    "summary":
        SUMMARY_ROWS,

    "direct":
        DIRECT_ROWS,

    "chain":
        CHAIN_ROWS,

    "semantic_support":
        SUPPORT_ROWS,

    "interpretation_contract": [
        "TEST156 contains no steering and no behavioral optimization.",
        "The natural atlas is reconstructed without using semantic compass targets.",
        "True downstream layer compasses are evaluation targets only.",
        "Direct transport starts from the true semantic compass at each source layer.",
        "Chained transport starts only from A3 and never re-extracts semantics downstream.",
        "Low source capture means the semantic compass lies mostly outside the selected natural source transport subspace.",
        "High source and target capture with low coefficient agreement indicates mode-correspondence mismatch.",
        "High direct alignment with poor chained alignment indicates composition error.",
        "SIGMA weighting is diagnostic only and is not selected post hoc as a new production rule.",
        "Results are restricted to Qwen2.5-7B-Instruct, the ABOVE/BELOW axis, this corpus family, and these layers.",
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

print("\n[14/14] COMPLETE")
print("=" * 150)

print(
    "Atlas fingerprint max error : "
    f"{100.0 * fingerprint_max_rel:.4f}%"
)

print(
    "K16 direct mean cosine      : "
    f"{mean_direct:+.6f}"
)

print(
    "K16 chained mean cosine     : "
    f"{mean_chain:+.6f}"
)

print(
    "K16 chained L19 cosine      : "
    f"{final_chain:+.6f}"
)

print(
    "K32 sigma advantage         : "
    f"{sigma_delta:+.6f}"
)

print(
    "Diagnosis                   : "
    f"{chain_status}"
)

print("\nSaved:")
print(" ", DIRECT_CSV)
print(" ", CHAIN_CSV)
print(" ", SUPPORT_CSV)
print(" ", MODES_CSV)
print(" ", SUMMARY_CSV)
print(" ", REPORT_JSON)
print(" ", NPZ_PATH)

print("=" * 150)
print("TEST 156 COMPLETE")
print("=" * 150)
