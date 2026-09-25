# =============================================================================
# TEST 158 — SEMANTIC OBJECT X-RAY
# GAUGE · CONSTELLATION · HELD-OUT ALIGNMENT · CONTEXT · DEFORMATION SUBSPACE
#
# Qwen/Qwen2.5-7B-Instruct · Frozen · BF16 forward · FP32 analysis
#
# PURPOSE
# -------
# TEST157 rejected a single concept-independent linear recoding operator:
#
#       c_b ~= R_{a->b} c_a
#
# TEST158 asks a more fundamental question:
#
#   WHAT mathematical object carrying semantic identity survives depth?
#
# Five hypotheses are tested in ONE frozen-model capsule:
#
# H1 — CONSTELLATION / RELATIONAL GEOMETRY
#      Individual semantic vectors may rotate while pairwise relations between
#      semantic axes remain stable.
#
# H2 — HELD-OUT GAUGE / ORTHOGONAL ALIGNMENT
#      Layer spaces may differ mainly by a coordinate-frame transformation.
#      Q is learned ONLY from TRAIN semantic axes and evaluated on VAL/FINAL.
#
# H3 — PAIRWISE-DIFFERENCE GEOMETRY
#      Absolute semantic vectors may drift while relative semantic differences
#      between concepts remain stable.
#
# H4 — CONCEPT x CONTEXT / MIXED SELECTIVITY
#      A semantic axis may not be one vector. Its direction may depend on the
#      lexical/context family in which it is expressed.
#
# H5 — SEMANTIC DEFORMATION SUBSPACE
#      Mean-difference compass may collapse a richer semantic object.
#      Sample-wise POS-NEG differences may preserve a subspace across depth
#      even when the mean compass does not.
#
# STRICT CONTRACT
# ---------------
# Frozen model.
# No steering.
# No intervention.
# No training.
# No gradients.
# No JVP/VJP.
# No LoRA.
# No optimizer.
#
# TRAIN / VAL / FINAL split is at SEMANTIC-AXIS level.
# FINAL ABOVE/BELOW is untouched by alignment fitting and selection.
#
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

SEED = 158
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

LAYERS = [3, 6, 10, 14, 19]
SEGMENTS = [(3, 6), (6, 10), (10, 14), (14, 19)]

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

ALL_AXES = TRAIN_AXES + VAL_AXES + FINAL_AXES

N_PER_SIDE = 8
EPS = 1e-10

# Exact permutation null for semantic labels is small here.
N_PERM = 2000

ROOT = Path("/content/TEST158_SEMANTIC_OBJECT_XRAY")
ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type != "cuda":
    raise RuntimeError("TEST158 requires CUDA.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 156)
print("TEST 158 — SEMANTIC OBJECT X-RAY")
print("GAUGE · CONSTELLATION · HELD-OUT ALIGNMENT · CONTEXT · DEFORMATION SUBSPACE")
print("=" * 156)

# =============================================================================
# MODEL
# =============================================================================

print("\n[1/15] Loading frozen model...")

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
print("TRAIN axes     :", TRAIN_AXES)
print("VAL axes       :", VAL_AXES)
print("FINAL axes     :", FINAL_AXES)

# =============================================================================
# CHAT
# =============================================================================

SYSTEM_EN = "You are a concise reasoning assistant."

def encode_prompt(text):
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    tok = tokenizer(
        rendered,
        return_tensors="pt",
    ).to(DEVICE)

    pos = int(
        tok["attention_mask"][0].sum().item()
    ) - 1

    return tok, pos

# =============================================================================
# SEMANTIC AXES
#
# IMPORTANT:
# Each POS[k] and NEG[k] is matched by object/context family.
# Therefore:
#
#       d_k = h(POS_k) - h(NEG_k)
#
# is a matched semantic deformation sample.
#
# =============================================================================

AXES = {
    "BEFORE_AFTER": {
        "POS": [
            "The bell rings before the light flashes.",
            "The door opens before the alarm sounds.",
            "The bird lands before the branch moves.",
            "The engine starts before the gate opens.",
            "The lamp turns on before the clock chimes.",
            "The cup falls before the plate moves.",
            "The signal appears before the screen changes.",
            "The rain begins before the wind strengthens.",
        ],
        "NEG": [
            "The bell rings after the light flashes.",
            "The door opens after the alarm sounds.",
            "The bird lands after the branch moves.",
            "The engine starts after the gate opens.",
            "The lamp turns on after the clock chimes.",
            "The cup falls after the plate moves.",
            "The signal appears after the screen changes.",
            "The rain begins after the wind strengthens.",
        ],
    },

    "INSIDE_OUTSIDE": {
        "POS": [
            "The coin is inside the wooden box.",
            "The key is inside the blue bag.",
            "The apple is inside the basket.",
            "The notebook is inside the drawer.",
            "The bottle is inside the cabinet.",
            "The toy is inside the container.",
            "The spoon is inside the cup.",
            "The card is inside the envelope.",
        ],
        "NEG": [
            "The coin is outside the wooden box.",
            "The key is outside the blue bag.",
            "The apple is outside the basket.",
            "The notebook is outside the drawer.",
            "The bottle is outside the cabinet.",
            "The toy is outside the container.",
            "The spoon is outside the cup.",
            "The card is outside the envelope.",
        ],
    },

    "LEFT_RIGHT": {
        "POS": [
            "The red marker is left of the blue marker.",
            "The lamp is left of the wooden table.",
            "The cup is left of the plate.",
            "The chair is left of the cabinet.",
            "The notebook is left of the bottle.",
            "The tree is left of the house.",
            "The sign is left of the doorway.",
            "The box is left of the basket.",
        ],
        "NEG": [
            "The red marker is right of the blue marker.",
            "The lamp is right of the wooden table.",
            "The cup is right of the plate.",
            "The chair is right of the cabinet.",
            "The notebook is right of the bottle.",
            "The tree is right of the house.",
            "The sign is right of the doorway.",
            "The box is right of the basket.",
        ],
    },

    "NEAR_FAR": {
        "POS": [
            "The chair is near the window.",
            "The bottle is near the cabinet.",
            "The tree is near the house.",
            "The lamp is near the table.",
            "The bicycle is near the gate.",
            "The cup is near the plate.",
            "The notebook is near the keyboard.",
            "The basket is near the wall.",
        ],
        "NEG": [
            "The chair is far from the window.",
            "The bottle is far from the cabinet.",
            "The tree is far from the house.",
            "The lamp is far from the table.",
            "The bicycle is far from the gate.",
            "The cup is far from the plate.",
            "The notebook is far from the keyboard.",
            "The basket is far from the wall.",
        ],
    },

    "EARLIER_LATER": {
        "POS": [
            "The first meeting happened earlier than the second meeting.",
            "The blue signal appeared earlier than the red signal.",
            "The first train arrived earlier than the second train.",
            "The morning call occurred earlier than the evening call.",
            "The first message arrived earlier than the second message.",
            "The opening event happened earlier than the closing event.",
            "The first alarm sounded earlier than the second alarm.",
            "The first observation occurred earlier than the second observation.",
        ],
        "NEG": [
            "The first meeting happened later than the second meeting.",
            "The blue signal appeared later than the red signal.",
            "The first train arrived later than the second train.",
            "The morning call occurred later than the evening call.",
            "The first message arrived later than the second message.",
            "The opening event happened later than the closing event.",
            "The first alarm sounded later than the second alarm.",
            "The first observation occurred later than the second observation.",
        ],
    },

    "FRONT_BEHIND": {
        "POS": [
            "The bicycle is in front of the gate.",
            "The chair is in front of the table.",
            "The car is in front of the building.",
            "The box is in front of the cabinet.",
            "The tree is in front of the wall.",
            "The lamp is in front of the curtain.",
            "The basket is in front of the door.",
            "The sign is in front of the window.",
        ],
        "NEG": [
            "The bicycle is behind the gate.",
            "The chair is behind the table.",
            "The car is behind the building.",
            "The box is behind the cabinet.",
            "The tree is behind the wall.",
            "The lamp is behind the curtain.",
            "The basket is behind the door.",
            "The sign is behind the window.",
        ],
    },

    "BIGGER_SMALLER": {
        "POS": [
            "The red circle is bigger than the blue circle.",
            "The wooden box is bigger than the metal box.",
            "The first ball is bigger than the second ball.",
            "The green square is bigger than the yellow square.",
            "The first cup is bigger than the second cup.",
            "The black bag is bigger than the white bag.",
            "The first plate is bigger than the second plate.",
            "The left container is bigger than the right container.",
        ],
        "NEG": [
            "The red circle is smaller than the blue circle.",
            "The wooden box is smaller than the metal box.",
            "The first ball is smaller than the second ball.",
            "The green square is smaller than the yellow square.",
            "The first cup is smaller than the second cup.",
            "The black bag is smaller than the white bag.",
            "The first plate is smaller than the second plate.",
            "The left container is smaller than the right container.",
        ],
    },

    "HOTTER_COLDER": {
        "POS": [
            "The red cup is hotter than the blue cup.",
            "The first pan is hotter than the second pan.",
            "The metal plate is hotter than the wooden plate.",
            "The left bottle is hotter than the right bottle.",
            "The first room is hotter than the second room.",
            "The black stone is hotter than the white stone.",
            "The first surface is hotter than the second surface.",
            "The green container is hotter than the yellow container.",
        ],
        "NEG": [
            "The red cup is colder than the blue cup.",
            "The first pan is colder than the second pan.",
            "The metal plate is colder than the wooden plate.",
            "The left bottle is colder than the right bottle.",
            "The first room is colder than the second room.",
            "The black stone is colder than the white stone.",
            "The first surface is colder than the second surface.",
            "The green container is colder than the yellow container.",
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

for name in ALL_AXES:
    assert len(AXES[name]["POS"]) == N_PER_SIDE
    assert len(AXES[name]["NEG"]) == N_PER_SIDE

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
    bank = {L: [] for L in LAYERS}

    for i, text in enumerate(texts, 1):
        row = capture_layers(text)

        for L in LAYERS:
            bank[L].append(row[L])

        if i == 1 or i == len(texts):
            print(f"    {label}: {i:02d}/{len(texts)}")

    return {
        L: torch.stack(bank[L]).float()
        for L in LAYERS
    }

# =============================================================================
# CAPTURE ALL SEMANTIC AXES
# =============================================================================

print("\n[2/15] Capturing semantic activation banks...")

BANK = {}

for axis in ALL_AXES:
    print(f"\n  AXIS: {axis}")

    BANK[axis] = {
        "POS": capture_bank(
            AXES[axis]["POS"],
            f"{axis}:POS",
        ),
        "NEG": capture_bank(
            AXES[axis]["NEG"],
            f"{axis}:NEG",
        ),
    }

# =============================================================================
# SEMANTIC OBJECTS
#
# COMPASS:
#   mean(POS) - mean(NEG)
#
# DIFF:
#   matched sample-wise POS_i - NEG_i
#
# RESIDUAL DIFF:
#   DIFF_i - mean(DIFF)
#
# =============================================================================

print("\n[3/15] Building semantic objects...")

COMPASS = {}
DIFF = {}
DIFF_UNIT = {}
DIFF_CENTERED = {}

for axis in ALL_AXES:
    COMPASS[axis] = {}
    DIFF[axis] = {}
    DIFF_UNIT[axis] = {}
    DIFF_CENTERED[axis] = {}

    for L in LAYERS:
        pos = BANK[axis]["POS"][L]
        neg = BANK[axis]["NEG"][L]

        d = pos - neg
        mean_d = d.mean(dim=0)

        compass = mean_d / (
            mean_d.norm() + EPS
        )

        d_unit = d / (
            d.norm(dim=1, keepdim=True) + EPS
        )

        d_centered = (
            d - d.mean(dim=0, keepdim=True)
        )

        COMPASS[axis][L] = compass
        DIFF[axis][L] = d
        DIFF_UNIT[axis][L] = d_unit
        DIFF_CENTERED[axis][L] = d_centered

print("\nAdjacent compass cosines:")

for axis in ALL_AXES:
    vals = []

    for a, b in SEGMENTS:
        vals.append(
            float(
                F.cosine_similarity(
                    COMPASS[axis][a][None],
                    COMPASS[axis][b][None],
                ).item()
            )
        )

    print(
        f"{axis:<18} "
        + " ".join(
            f"{v:+.4f}"
            for v in vals
        )
    )

# =============================================================================
# HELPERS
# =============================================================================

def cosine(a, b):
    if a.norm() <= EPS or b.norm() <= EPS:
        return 0.0

    return float(
        F.cosine_similarity(
            a[None],
            b[None],
        ).item()
    )

def upper_triangle_vector(M):
    n = M.shape[0]

    idx = torch.triu_indices(
        n,
        n,
        offset=1,
    )

    return M[
        idx[0],
        idx[1],
    ]

def pearson_torch(x, y):
    x = x.float()
    y = y.float()

    x = x - x.mean()
    y = y - y.mean()

    denom = (
        x.norm() * y.norm()
    )

    if denom <= EPS:
        return 0.0

    return float(
        (x @ y / denom).item()
    )

def principal_cosines(X, Y, rank=None):
    # X/Y rows = samples/vectors.
    # Return principal cosines between row spans.

    X = X.float().to(DEVICE)
    Y = Y.float().to(DEVICE)

    if X.shape[0] == 0 or Y.shape[0] == 0:
        return torch.empty(0)

    _, sx, Vhx = torch.linalg.svd(
        X,
        full_matrices=False,
    )

    _, sy, Vhy = torch.linalg.svd(
        Y,
        full_matrices=False,
    )

    tol_x = (
        sx.max() * max(X.shape) * 1e-6
        if sx.numel()
        else 0.0
    )

    tol_y = (
        sy.max() * max(Y.shape) * 1e-6
        if sy.numel()
        else 0.0
    )

    rx = int(
        (sx > tol_x).sum().item()
    )

    ry = int(
        (sy > tol_y).sum().item()
    )

    r = min(rx, ry)

    if rank is not None:
        r = min(r, rank)

    if r == 0:
        return torch.empty(0)

    Bx = Vhx[:r].T
    By = Vhy[:r].T

    s = torch.linalg.svdvals(
        Bx.T @ By
    )

    return s.detach().cpu()

def orthogonal_fit(source_matrix, target_matrix):
    # rows = semantic axes
    # Find Q such that source @ Q ~= target.
    #
    # Thin low-rank orthogonal map:
    # source^T target = U S V^T
    # Q action is represented implicitly as U V^T inside semantic support.
    #
    # For held-out vector x:
    # x Q = (x U) V^T
    #
    X = source_matrix.float().to(DEVICE)
    Y = target_matrix.float().to(DEVICE)

    C = X.T @ Y

    # C is 3584x3584 -> avoid full SVD.
    #
    # X^T Y has rank <= number of TRAIN axes.
    #
    # X^T = Qx Rx
    # Y^T = Qy Ry
    # C = Qx Rx Ry^T Qy^T

    Qx, Rx = torch.linalg.qr(
        X.T,
        mode="reduced",
    )

    Qy, Ry = torch.linalg.qr(
        Y.T,
        mode="reduced",
    )

    small = Rx @ Ry.T

    U0, S, Vh0 = torch.linalg.svd(
        small,
        full_matrices=False,
    )

    U = Qx @ U0
    V = Qy @ Vh0.T

    return U, S, V

def apply_orthogonal_fit(x, U, V):
    x = x.float().to(DEVICE)

    pred = V @ (
        U.T @ x
    )

    if pred.norm() > EPS:
        pred = pred / pred.norm()

    return pred

def semantic_matrix(axis_names, L):
    return torch.stack([
        COMPASS[a][L]
        for a in axis_names
    ]).float()

def gram_cosine(axis_names, L):
    A = semantic_matrix(
        axis_names,
        L,
    )

    A = A / (
        A.norm(
            dim=1,
            keepdim=True,
        ) + EPS
    )

    return A @ A.T

# =============================================================================
# H1 — CONSTELLATION / RELATIONAL GEOMETRY
#
# Does the relationship between semantic axes survive depth even when the
# individual vectors themselves rotate?
# =============================================================================

print("\n[4/15] H1 — Semantic constellation geometry...")

CONSTELLATION_ROWS = []

for split_name, axes in [
    ("TRAIN", TRAIN_AXES),
    ("ALL", ALL_AXES),
]:
    print(f"\n  SPLIT: {split_name}")

    for a, b in SEGMENTS:
        Ga = gram_cosine(axes, a)
        Gb = gram_cosine(axes, b)

        va = upper_triangle_vector(Ga)
        vb = upper_triangle_vector(Gb)

        corr = pearson_torch(
            va,
            vb,
        )

        mae = float(
            (va - vb)
            .abs()
            .mean()
            .item()
        )

        rmse = float(
            torch.sqrt(
                (
                    (va - vb) ** 2
                ).mean()
            ).item()
        )

        individual = np.mean([
            cosine(
                COMPASS[x][a],
                COMPASS[x][b],
            )
            for x in axes
        ])

        CONSTELLATION_ROWS.append({
            "split": split_name,
            "source_layer": a,
            "target_layer": b,
            "mean_individual_cos":
                float(individual),
            "gram_correlation":
                corr,
            "gram_mae":
                mae,
            "gram_rmse":
                rmse,
        })

        print(
            f"L{a}->L{b} | "
            f"individual={individual:+.6f} | "
            f"Gram-r={corr:+.6f} | "
            f"MAE={mae:.6f}"
        )

# =============================================================================
# H2 — HELD-OUT ORTHOGONAL / GAUGE ALIGNMENT
#
# Fit only on TRAIN semantic axes.
#
# No VAL/FINAL vector is used to construct Q.
#
# Compare:
#   STATIC: source vector unchanged
#   ORTHO : TRAIN-fitted coordinate alignment
#
# =============================================================================

print("\n[5/15] H2 — TRAIN-only held-out orthogonal alignment...")

ORTHO = {}
ORTHO_ROWS = []

for a, b in SEGMENTS:
    X = semantic_matrix(
        TRAIN_AXES,
        a,
    )

    Y = semantic_matrix(
        TRAIN_AXES,
        b,
    )

    U, S, V = orthogonal_fit(
        X,
        Y,
    )

    ORTHO[(a, b)] = {
        "U": U,
        "S": S,
        "V": V,
    }

    print(
        f"\n  L{a}->L{b} "
        f"fit rank={len(S)}"
    )

    for split_name, axes in [
        ("TRAIN", TRAIN_AXES),
        ("VAL", VAL_AXES),
        ("FINAL", FINAL_AXES),
    ]:
        for axis in axes:
            src = (
                COMPASS[axis][a]
                .to(DEVICE)
            )

            tgt = (
                COMPASS[axis][b]
                .to(DEVICE)
            )

            pred = apply_orthogonal_fit(
                src,
                U,
                V,
            )

            static_cos = cosine(
                src,
                tgt,
            )

            aligned_cos = cosine(
                pred,
                tgt,
            )

            ORTHO_ROWS.append({
                "split": split_name,
                "axis": axis,
                "source_layer": a,
                "target_layer": b,
                "static_cos":
                    static_cos,
                "aligned_cos":
                    aligned_cos,
                "delta":
                    aligned_cos - static_cos,
            })

            print(
                f"    {split_name:<5} "
                f"{axis:<18} | "
                f"static={static_cos:+.6f} "
                f"ortho={aligned_cos:+.6f} "
                f"Δ={aligned_cos-static_cos:+.6f}"
            )

# =============================================================================
# H3 — PAIRWISE DIFFERENCE GEOMETRY
#
# If absolute semantic vectors are unstable but relative concept geometry is
# preserved, pairwise semantic differences should remain more stable.
#
# D_L(i,j) = normalize(A_i - A_j)
#
# We compare:
#
#   mean absolute vector persistence
#   mean pairwise-difference persistence
#
# =============================================================================

print("\n[6/15] H3 — Pairwise semantic-difference geometry...")

PAIR_ROWS = []

for split_name, axes in [
    ("TRAIN", TRAIN_AXES),
    ("ALL", ALL_AXES),
]:
    print(f"\n  SPLIT: {split_name}")

    for a, b in SEGMENTS:
        abs_vals = [
            cosine(
                COMPASS[x][a],
                COMPASS[x][b],
            )
            for x in axes
        ]

        pair_vals = []

        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                da = (
                    COMPASS[axes[i]][a]
                    - COMPASS[axes[j]][a]
                )

                db = (
                    COMPASS[axes[i]][b]
                    - COMPASS[axes[j]][b]
                )

                pair_vals.append(
                    cosine(
                        da,
                        db,
                    )
                )

        abs_mean = float(
            np.mean(abs_vals)
        )

        pair_mean = float(
            np.mean(pair_vals)
        )

        delta = (
            pair_mean - abs_mean
        )

        PAIR_ROWS.append({
            "split": split_name,
            "source_layer": a,
            "target_layer": b,
            "absolute_mean_cos":
                abs_mean,
            "pair_difference_mean_cos":
                pair_mean,
            "pair_minus_absolute":
                delta,
            "n_pairs":
                len(pair_vals),
        })

        print(
            f"L{a}->L{b} | "
            f"absolute={abs_mean:+.6f} | "
            f"pair-diff={pair_mean:+.6f} | "
            f"Δ={delta:+.6f}"
        )

# =============================================================================
# H4 — CONTEXT / MIXED SELECTIVITY
#
# Each matched POS_i - NEG_i gives a context-specific semantic direction.
#
# Questions:
#
# 1. Within a layer, how coherent are these 8 directions?
# 2. Across layers, does SAME-context matching preserve direction better than
#    CROSS-context matching?
#
# If:
#
#   same_context >> cross_context
#
# then semantic transport contains context-dependent structure that the
# mean compass destroys.
# =============================================================================

print("\n[7/15] H4 — Concept × context interaction...")

CONTEXT_ROWS = []

for axis in ALL_AXES:
    print(f"\n  AXIS: {axis}")

    for a, b in SEGMENTS:
        Da = DIFF_UNIT[axis][a]
        Db = DIFF_UNIT[axis][b]

        # same matched context
        same = torch.sum(
            Da * Db,
            dim=1,
        )

        # all cross-context combinations
        cross_matrix = Da @ Db.T

        mask = ~torch.eye(
            N_PER_SIDE,
            dtype=torch.bool,
        )

        cross = cross_matrix[mask]

        same_mean = float(
            same.mean().item()
        )

        cross_mean = float(
            cross.mean().item()
        )

        context_advantage = (
            same_mean - cross_mean
        )

        # Within-layer semantic consistency.
        Wa = Da @ Da.T
        Wb = Db @ Db.T

        tri = torch.triu_indices(
            N_PER_SIDE,
            N_PER_SIDE,
            offset=1,
        )

        within_a = float(
            Wa[
                tri[0],
                tri[1],
            ].mean().item()
        )

        within_b = float(
            Wb[
                tri[0],
                tri[1],
            ].mean().item()
        )

        compass_cos = cosine(
            COMPASS[axis][a],
            COMPASS[axis][b],
        )

        CONTEXT_ROWS.append({
            "axis": axis,
            "source_layer": a,
            "target_layer": b,
            "compass_cos":
                compass_cos,
            "same_context_cos":
                same_mean,
            "cross_context_cos":
                cross_mean,
            "context_advantage":
                context_advantage,
            "within_source_context_coherence":
                within_a,
            "within_target_context_coherence":
                within_b,
        })

        print(
            f"L{a}->L{b} | "
            f"compass={compass_cos:+.5f} | "
            f"same={same_mean:+.5f} | "
            f"cross={cross_mean:+.5f} | "
            f"same-cross={context_advantage:+.5f}"
        )

# =============================================================================
# H4 PERMUTATION NULL
#
# Destroy context correspondence at target layer while preserving all vectors.
#
# Test statistic:
#
#       mean diag(Da Db^T) - mean offdiag(Da Db^T)
#
# =============================================================================

print("\n[8/15] H4 — Context correspondence permutation null...")

PERM_ROWS = []

rng_perm = np.random.default_rng(SEED)

for axis in ALL_AXES:
    for a, b in SEGMENTS:
        Da = DIFF_UNIT[axis][a]
        Db = DIFF_UNIT[axis][b]

        M = Da @ Db.T

        eye = torch.eye(
            N_PER_SIDE,
            dtype=torch.bool,
        )

        real = float(
            M.diag().mean().item()
            - M[~eye].mean().item()
        )

        nulls = []

        for _ in range(N_PERM):
            perm = rng_perm.permutation(
                N_PER_SIDE
            )

            Mp = M[:, perm]

            stat = float(
                Mp.diag().mean().item()
                - Mp[~eye].mean().item()
            )

            nulls.append(stat)

        nulls = np.asarray(
            nulls,
            dtype=np.float64,
        )

        p = (
            1.0
            + float(
                np.sum(
                    nulls >= real
                )
            )
        ) / (
            N_PERM + 1.0
        )

        z = (
            real - nulls.mean()
        ) / (
            nulls.std(ddof=1) + EPS
        )

        PERM_ROWS.append({
            "axis": axis,
            "source_layer": a,
            "target_layer": b,
            "real_context_advantage":
                real,
            "null_mean":
                float(nulls.mean()),
            "null_sd":
                float(nulls.std(ddof=1)),
            "z":
                float(z),
            "p_one_sided":
                float(p),
        })

        print(
            f"{axis:<18} "
            f"L{a}->L{b} | "
            f"real={real:+.6f} | "
            f"null={nulls.mean():+.6f}"
            f"±{nulls.std(ddof=1):.6f} | "
            f"Z={z:+.3f} | "
            f"p={p:.6f}"
        )

# =============================================================================
# H5 — SEMANTIC DEFORMATION SUBSPACE
#
# The mean compass uses:
#
#       mean(d_i)
#
# but the full semantic object may be:
#
#       span{d_1, ..., d_8}
#
# Two versions:
#
# RAW:
#   span of matched semantic differences.
#
# CENTERED:
#   span of context-dependent deviations around the mean semantic compass.
#
# Principal cosines across layers test whether these subspaces survive even
# when the mean vector itself rotates.
# =============================================================================

print("\n[9/15] H5 — Semantic deformation subspaces...")

SUBSPACE_ROWS = []

for axis in ALL_AXES:
    print(f"\n  AXIS: {axis}")

    for a, b in SEGMENTS:
        raw_pc = principal_cosines(
            DIFF[axis][a],
            DIFF[axis][b],
            rank=N_PER_SIDE,
        )

        centered_pc = principal_cosines(
            DIFF_CENTERED[axis][a],
            DIFF_CENTERED[axis][b],
            rank=N_PER_SIDE - 1,
        )

        raw_mean = (
            float(raw_pc.mean().item())
            if raw_pc.numel()
            else 0.0
        )

        centered_mean = (
            float(centered_pc.mean().item())
            if centered_pc.numel()
            else 0.0
        )

        raw_min = (
            float(raw_pc.min().item())
            if raw_pc.numel()
            else 0.0
        )

        centered_min = (
            float(centered_pc.min().item())
            if centered_pc.numel()
            else 0.0
        )

        compass_cos = cosine(
            COMPASS[axis][a],
            COMPASS[axis][b],
        )

        SUBSPACE_ROWS.append({
            "axis": axis,
            "source_layer": a,
            "target_layer": b,
            "compass_cos":
                compass_cos,
            "raw_subspace_mean_principal_cos":
                raw_mean,
            "raw_subspace_min_principal_cos":
                raw_min,
            "centered_subspace_mean_principal_cos":
                centered_mean,
            "centered_subspace_min_principal_cos":
                centered_min,
            "raw_minus_compass":
                raw_mean - compass_cos,
            "centered_minus_compass":
                centered_mean - compass_cos,
        })

        print(
            f"L{a}->L{b} | "
            f"compass={compass_cos:+.5f} | "
            f"RAW-sub={raw_mean:+.5f} | "
            f"CENTERED-sub={centered_mean:+.5f} | "
            f"RAW-compass={raw_mean-compass_cos:+.5f}"
        )

# =============================================================================
# H5 MATCHED-SUBSPACE NULL
#
# Shuffle semantic pairing POS_i <-> NEG_j at target layer.
#
# This preserves the marginal activation banks while destroying the original
# matched semantic deformation correspondence.
#
# We compare mean principal cosine of RAW difference subspaces.
# =============================================================================

print("\n[10/15] H5 — Deformation-subspace pairing null...")

SUBSPACE_NULL_ROWS = []

rng_sub = np.random.default_rng(
    SEED + 1000
)

for axis in ALL_AXES:
    for a, b in SEGMENTS:
        real_pc = principal_cosines(
            DIFF[axis][a],
            DIFF[axis][b],
            rank=N_PER_SIDE,
        )

        real = (
            float(real_pc.mean().item())
            if real_pc.numel()
            else 0.0
        )

        pos_b = BANK[axis]["POS"][b]
        neg_b = BANK[axis]["NEG"][b]

        nulls = []

        for _ in range(N_PERM):
            perm = rng_sub.permutation(
                N_PER_SIDE
            )

            shuffled_diff = (
                pos_b
                - neg_b[perm]
            )

            pc = principal_cosines(
                DIFF[axis][a],
                shuffled_diff,
                rank=N_PER_SIDE,
            )

            nulls.append(
                float(pc.mean().item())
                if pc.numel()
                else 0.0
            )

        nulls = np.asarray(
            nulls,
            dtype=np.float64,
        )

        p = (
            1.0
            + float(
                np.sum(
                    nulls >= real
                )
            )
        ) / (
            N_PERM + 1.0
        )

        z = (
            real - nulls.mean()
        ) / (
            nulls.std(ddof=1) + EPS
        )

        SUBSPACE_NULL_ROWS.append({
            "axis": axis,
            "source_layer": a,
            "target_layer": b,
            "real_mean_principal_cos":
                real,
            "null_mean":
                float(nulls.mean()),
            "null_sd":
                float(nulls.std(ddof=1)),
            "z":
                float(z),
            "p_one_sided":
                float(p),
        })

        print(
            f"{axis:<18} "
            f"L{a}->L{b} | "
            f"real={real:.6f} | "
            f"null={nulls.mean():.6f}"
            f"±{nulls.std(ddof=1):.6f} | "
            f"Z={z:+.3f} | "
            f"p={p:.6f}"
        )

# =============================================================================
# HELD-OUT CONSTELLATION PREDICTION
#
# Does TRAIN relational geometry predict how a held-out semantic axis relates
# to TRAIN axes downstream?
#
# For held-out axis x:
#
# source relational signature:
#
#   r_a(x) = [cos(A_x, A_train1), ...]
#
# target relational signature:
#
#   r_b(x)
#
# If these correlate strongly, the held-out concept may preserve its position
# in the semantic constellation even when its absolute vector rotates.
# =============================================================================

print("\n[11/15] Held-out relational signatures...")

RELATIONAL_ROWS = []

for split_name, axes in [
    ("VAL", VAL_AXES),
    ("FINAL", FINAL_AXES),
]:
    for axis in axes:
        print(f"\n  {split_name}: {axis}")

        for a, b in SEGMENTS:
            ra = torch.tensor([
                cosine(
                    COMPASS[axis][a],
                    COMPASS[t][a],
                )
                for t in TRAIN_AXES
            ])

            rb = torch.tensor([
                cosine(
                    COMPASS[axis][b],
                    COMPASS[t][b],
                )
                for t in TRAIN_AXES
            ])

            corr = pearson_torch(
                ra,
                rb,
            )

            rmse = float(
                torch.sqrt(
                    (
                        (ra - rb) ** 2
                    ).mean()
                ).item()
            )

            abs_cos = cosine(
                COMPASS[axis][a],
                COMPASS[axis][b],
            )

            RELATIONAL_ROWS.append({
                "split": split_name,
                "axis": axis,
                "source_layer": a,
                "target_layer": b,
                "absolute_compass_cos":
                    abs_cos,
                "relational_signature_correlation":
                    corr,
                "relational_signature_rmse":
                    rmse,
            })

            print(
                f"L{a}->L{b} | "
                f"absolute={abs_cos:+.6f} | "
                f"relational-r={corr:+.6f} | "
                f"RMSE={rmse:.6f}"
            )

# =============================================================================
# SUMMARY
# =============================================================================

print("\n[12/15] Integrated summary...")

def mean_field(rows, key, filt=None):
    vals = []

    for r in rows:
        if filt is None or filt(r):
            vals.append(
                float(r[key])
            )

    return (
        float(np.mean(vals))
        if vals
        else float("nan")
    )

train_constellation = mean_field(
    CONSTELLATION_ROWS,
    "gram_correlation",
    lambda r: r["split"] == "TRAIN",
)

all_constellation = mean_field(
    CONSTELLATION_ROWS,
    "gram_correlation",
    lambda r: r["split"] == "ALL",
)

val_ortho_delta = mean_field(
    ORTHO_ROWS,
    "delta",
    lambda r: r["split"] == "VAL",
)

final_ortho_delta = mean_field(
    ORTHO_ROWS,
    "delta",
    lambda r: r["split"] == "FINAL",
)

pair_adv = mean_field(
    PAIR_ROWS,
    "pair_minus_absolute",
    lambda r: r["split"] == "ALL",
)

context_adv = mean_field(
    CONTEXT_ROWS,
    "context_advantage",
)

context_p_fraction = float(
    np.mean([
        r["p_one_sided"] < 0.05
        for r in PERM_ROWS
    ])
)

raw_sub_adv = mean_field(
    SUBSPACE_ROWS,
    "raw_minus_compass",
)

centered_sub_adv = mean_field(
    SUBSPACE_ROWS,
    "centered_minus_compass",
)

subspace_p_fraction = float(
    np.mean([
        r["p_one_sided"] < 0.05
        for r in SUBSPACE_NULL_ROWS
    ])
)

val_relational = mean_field(
    RELATIONAL_ROWS,
    "relational_signature_correlation",
    lambda r: r["split"] == "VAL",
)

final_relational = mean_field(
    RELATIONAL_ROWS,
    "relational_signature_correlation",
    lambda r: r["split"] == "FINAL",
)

print(
    f"TRAIN constellation Gram-r        : "
    f"{train_constellation:+.6f}"
)

print(
    f"ALL constellation Gram-r          : "
    f"{all_constellation:+.6f}"
)

print(
    f"VAL held-out ORTHO Δ              : "
    f"{val_ortho_delta:+.6f}"
)

print(
    f"FINAL held-out ORTHO Δ            : "
    f"{final_ortho_delta:+.6f}"
)

print(
    f"Pair-difference advantage         : "
    f"{pair_adv:+.6f}"
)

print(
    f"Mean same-context advantage       : "
    f"{context_adv:+.6f}"
)

print(
    f"Context permutation p<.05 fraction: "
    f"{context_p_fraction:.3f}"
)

print(
    f"RAW subspace - compass            : "
    f"{raw_sub_adv:+.6f}"
)

print(
    f"CENTERED subspace - compass       : "
    f"{centered_sub_adv:+.6f}"
)

print(
    f"Subspace pairing p<.05 fraction   : "
    f"{subspace_p_fraction:.3f}"
)

print(
    f"VAL relational signature r        : "
    f"{val_relational:+.6f}"
)

print(
    f"FINAL relational signature r      : "
    f"{final_relational:+.6f}"
)

# =============================================================================
# MECHANISTIC DIAGNOSIS
#
# These are descriptive routing labels, NOT significance claims.
# =============================================================================

print("\n[13/15] Mechanistic diagnosis...")

FLAGS = []

# Relational geometry survives more strongly than absolute vector geometry.
mean_abs_all = mean_field(
    CONSTELLATION_ROWS,
    "mean_individual_cos",
    lambda r: r["split"] == "ALL",
)

if (
    all_constellation >= 0.60
    and all_constellation > mean_abs_all + 0.20
):
    FLAGS.append(
        "RELATIONAL_GEOMETRY_PRESERVED"
    )

if (
    val_ortho_delta > 0.05
    and final_ortho_delta > 0.05
):
    FLAGS.append(
        "HELDOUT_GAUGE_ALIGNMENT_SUPPORTED"
    )

if pair_adv > 0.10:
    FLAGS.append(
        "PAIRWISE_DIFFERENCE_GEOMETRY_SUPPORTED"
    )

if (
    context_adv > 0.05
    and context_p_fraction >= 0.50
):
    FLAGS.append(
        "CONTEXT_DEPENDENT_SEMANTICS_SUPPORTED"
    )

if (
    raw_sub_adv > 0.15
    and subspace_p_fraction >= 0.50
):
    FLAGS.append(
        "SEMANTIC_DEFORMATION_SUBSPACE_SUPPORTED"
    )

if (
    centered_sub_adv > 0.15
    and subspace_p_fraction >= 0.50
):
    FLAGS.append(
        "CONTEXT_RESIDUAL_SUBSPACE_SUPPORTED"
    )

if (
    final_relational >= 0.60
    and final_relational > mean_abs_all + 0.20
):
    FLAGS.append(
        "HELDOUT_RELATIONAL_IDENTITY_SUPPORTED"
    )

if not FLAGS:
    FLAGS.append(
        "NO_SINGLE_OBJECT_DOMINATES"
    )

print("\nDIAGNOSTIC FLAGS")

for x in FLAGS:
    print(" -", x)

print("\nInterpretation map:")

print("""
RELATIONAL_GEOMETRY_PRESERVED
    Individual semantic arrows rotate, but the semantic constellation survives.

HELDOUT_GAUGE_ALIGNMENT_SUPPORTED
    A TRAIN-fitted low-rank orthogonal coordinate transformation improves
    unseen semantic axes. Layer coordinate-frame change is plausible.

PAIRWISE_DIFFERENCE_GEOMETRY_SUPPORTED
    Relative semantic differences survive better than absolute semantic arrows.

CONTEXT_DEPENDENT_SEMANTICS_SUPPORTED
    Matched lexical/context instances preserve cross-layer identity better than
    mismatched instances. A single averaged compass discards useful structure.

SEMANTIC_DEFORMATION_SUBSPACE_SUPPORTED
    The sample-wise semantic deformation span survives depth substantially
    better than the mean compass. The semantic object is better represented
    as a subspace than as one vector.

CONTEXT_RESIDUAL_SUBSPACE_SUPPORTED
    Even after removing the mean semantic direction, structured context-
    dependent semantic variation persists across depth.

HELDOUT_RELATIONAL_IDENTITY_SUPPORTED
    An unseen concept preserves its relationships to TRAIN concepts even when
    its absolute hidden-space direction changes.

NO_SINGLE_OBJECT_DOMINATES
    None of these simple objects explains semantic persistence alone.
""")

# =============================================================================
# SAVE
# =============================================================================

print("\n[14/15] Saving CSV / JSON / NPZ...")

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

paths = {
    "constellation":
        ROOT / "CONSTELLATION.csv",

    "ortho":
        ROOT / "HELDOUT_ORTHO.csv",

    "pair":
        ROOT / "PAIRWISE_GEOMETRY.csv",

    "context":
        ROOT / "CONTEXT_INTERACTION.csv",

    "context_null":
        ROOT / "CONTEXT_PERMUTATION_NULL.csv",

    "subspace":
        ROOT / "DEFORMATION_SUBSPACE.csv",

    "subspace_null":
        ROOT / "DEFORMATION_SUBSPACE_NULL.csv",

    "relational":
        ROOT / "HELDOUT_RELATIONAL_SIGNATURE.csv",

    "report":
        ROOT / "REPORT.json",

    "npz":
        ROOT / "TEST158_SEMANTIC_OBJECT_XRAY.npz",
}

save_csv(
    paths["constellation"],
    CONSTELLATION_ROWS,
)

save_csv(
    paths["ortho"],
    ORTHO_ROWS,
)

save_csv(
    paths["pair"],
    PAIR_ROWS,
)

save_csv(
    paths["context"],
    CONTEXT_ROWS,
)

save_csv(
    paths["context_null"],
    PERM_ROWS,
)

save_csv(
    paths["subspace"],
    SUBSPACE_ROWS,
)

save_csv(
    paths["subspace_null"],
    SUBSPACE_NULL_ROWS,
)

save_csv(
    paths["relational"],
    RELATIONAL_ROWS,
)

NPZ = {}

for axis in ALL_AXES:
    for L in LAYERS:
        NPZ[
            f"COMPASS_{axis}_L{L}"
        ] = (
            COMPASS[axis][L]
            .numpy()
        )

        NPZ[
            f"DIFF_{axis}_L{L}"
        ] = (
            DIFF[axis][L]
            .numpy()
        )

for pair in SEGMENTS:
    a, b = pair

    NPZ[
        f"ORTHO_U_L{a}_L{b}"
    ] = (
        ORTHO[pair]["U"]
        .detach()
        .cpu()
        .numpy()
    )

    NPZ[
        f"ORTHO_S_L{a}_L{b}"
    ] = (
        ORTHO[pair]["S"]
        .detach()
        .cpu()
        .numpy()
    )

    NPZ[
        f"ORTHO_V_L{a}_L{b}"
    ] = (
        ORTHO[pair]["V"]
        .detach()
        .cpu()
        .numpy()
    )

np.savez_compressed(
    paths["npz"],
    **NPZ,
)

REPORT = {
    "test":
        "TEST 158",

    "title":
        "SEMANTIC OBJECT X-RAY",

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

    "train_axes":
        TRAIN_AXES,

    "val_axes":
        VAL_AXES,

    "final_axes":
        FINAL_AXES,

    "summary": {
        "mean_absolute_axis_persistence":
            mean_abs_all,

        "train_constellation_gram_r":
            train_constellation,

        "all_constellation_gram_r":
            all_constellation,

        "val_heldout_ortho_delta":
            val_ortho_delta,

        "final_heldout_ortho_delta":
            final_ortho_delta,

        "pair_difference_advantage":
            pair_adv,

        "mean_context_advantage":
            context_adv,

        "context_p_lt_005_fraction":
            context_p_fraction,

        "raw_subspace_minus_compass":
            raw_sub_adv,

        "centered_subspace_minus_compass":
            centered_sub_adv,

        "subspace_p_lt_005_fraction":
            subspace_p_fraction,

        "val_relational_signature_r":
            val_relational,

        "final_relational_signature_r":
            final_relational,

        "diagnostic_flags":
            FLAGS,
    },

    "interpretation_contract": [
        "TEST158 contains no steering, intervention, training, gradients, optimizer, JVP or VJP.",
        "TRAIN, VAL and FINAL separation is performed at semantic-axis level.",
        "ABOVE/BELOW is never used to fit the orthogonal alignment.",
        "Compass persistence and relational-geometry persistence are distinct measurements.",
        "Pairwise-difference geometry tests relative semantic organization rather than absolute coordinates.",
        "Context analysis uses matched POS_i/NEG_i sentence families.",
        "Context permutation destroys sentence-family correspondence while preserving the activation vectors.",
        "Semantic deformation subspace uses the complete matched POS-minus-NEG bank rather than only its mean.",
        "Centered deformation analysis removes the mean semantic direction before comparing context-dependent residual spans.",
        "High principal-angle similarity alone does not establish causal transport.",
        "Orthogonal alignment improvement alone does not establish a universal gauge symmetry.",
        "Permutation p-values describe the specific finite semantic banks used here.",
        "Results are restricted to this model, prompt construction, layers and semantic-axis collection.",
    ],

    "constellation":
        CONSTELLATION_ROWS,

    "heldout_ortho":
        ORTHO_ROWS,

    "pairwise_geometry":
        PAIR_ROWS,

    "context_interaction":
        CONTEXT_ROWS,

    "context_null":
        PERM_ROWS,

    "deformation_subspace":
        SUBSPACE_ROWS,

    "deformation_subspace_null":
        SUBSPACE_NULL_ROWS,

    "heldout_relational_signature":
        RELATIONAL_ROWS,
}

with open(
    paths["report"],
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

print("\n[15/15] COMPLETE")
print("=" * 156)

print(
    "Mean absolute axis persistence      : "
    f"{mean_abs_all:+.6f}"
)

print(
    "ALL constellation Gram-r            : "
    f"{all_constellation:+.6f}"
)

print(
    "FINAL held-out ORTHO delta          : "
    f"{final_ortho_delta:+.6f}"
)

print(
    "Pair-difference advantage           : "
    f"{pair_adv:+.6f}"
)

print(
    "Mean context advantage              : "
    f"{context_adv:+.6f}"
)

print(
    "Context p<.05 fraction              : "
    f"{context_p_fraction:.3f}"
)

print(
    "RAW subspace minus compass          : "
    f"{raw_sub_adv:+.6f}"
)

print(
    "CENTERED subspace minus compass     : "
    f"{centered_sub_adv:+.6f}"
)

print(
    "Subspace p<.05 fraction             : "
    f"{subspace_p_fraction:.3f}"
)

print(
    "FINAL relational signature r        : "
    f"{final_relational:+.6f}"
)

print("\nDIAGNOSTIC FLAGS:")

for x in FLAGS:
    print(" ", x)

print("\nSaved:")

for p in paths.values():
    print(" ", p)

print("=" * 156)
print("TEST 158 COMPLETE")
print("=" * 156)
