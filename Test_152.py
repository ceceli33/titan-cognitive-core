# =============================================================================
# TEST 152 — SPONTANEOUS FLUCTUATION ATLAS
# ENDOGENOUS POTENT TRANSPORT SUBSPACE
# NATURAL CROSS-LAYER COVARIANCE · SHUFFLE NULL · PERSISTENCE X-RAY
#
# Qwen/Qwen2.5-7B-Instruct · Frozen Model · BF16 · A100
#
# ZERO STEERING
# ZERO PERTURBATION
# ZERO GRADIENT
# ZERO JVP / VJP
# ZERO TRAINING
#
# Question:
# Does the frozen transformer naturally use a low-rank, reproducible
# source subspace whose spontaneous L3 fluctuations covary with
# downstream fluctuations at L6/L10/L14/L19?
#
# IMPORTANT:
# This test does NOT assume that high variance = downstream potency.
#
# For centered natural activations:
#
#     X = H_L3 - mean(H_L3)
#     Y = H_L  - mean(H_L)
#
# we study the cross-layer covariance:
#
#     C_3,L = X^T Y / (N-1)
#
# without materializing the 3584 x 3584 matrix.
#
# If:
#
#     C_3,L = U Sigma V^T
#
# then:
#
#     U = source / forcing modes at L3
#     V = response modes at downstream layer L
#
# Source and response directions are NOT forced to be the same.
#
# Primary falsification:
# Keep X and Y exactly unchanged, but destroy sample correspondence:
#
#     C_null = X^T P Y / (N-1)
#
# using repeated row permutations P.
#
# A useful result requires:
#   1. real singular spectrum > shuffle spectrum,
#   2. low effective transport rank,
#   3. stable source modes across horizons,
#   4. stability on an untouched FINAL split.
#
# =============================================================================

import sys, subprocess, os, gc, math, warnings

for pkg, name in [
    ("transformers>=4.46.0", "transformers"),
    ("accelerate", "accelerate"),
]:
    try:
        __import__(name)
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

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU bulunamadı. A100 çalışma zamanı seçin.")

# =============================================================================
# 1. CONFIG
# =============================================================================

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SOURCE_LAYER = 3
TARGET_LAYERS = [6, 10, 14, 19]
CAPTURE_LAYERS = [SOURCE_LAYER] + TARGET_LAYERS

N_TOTAL = 256
N_SEARCH = 192
N_FINAL = 64

N_SHUFFLE = 32

REPORT_K = [1, 2, 4, 8, 16, 32, 48, 64]
MAX_MODES = 64

ENERGY_LEVELS = [0.50, 0.80, 0.90, 0.95]

EPS = 1e-8

SYSTEM = "You are a concise reasoning assistant."

print("=" * 128)
print("TEST 152 — SPONTANEOUS FLUCTUATION ATLAS")
print("ENDOGENOUS POTENT TRANSPORT SUBSPACE")
print("NATURAL CROSS-LAYER COVARIANCE · SHUFFLE NULL · PERSISTENCE X-RAY")
print("=" * 128)

# =============================================================================
# 2. NATURAL NEUTRAL CORPUS
#
# Procedurally generated neutral declarative sentences.
# No ABOVE/BELOW.
# No BEFORE/AFTER.
# No steering labels.
# No synthetic activation vectors.
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


def build_corpus():

    candidates = []

    for s in SUBJECTS:
        for v in VERBS:
            for loc in LOCATIONS:
                for d in DETAILS:

                    text = f"{s} {v} {loc}. {d}"

                    candidates.append(text)

    rng = np.random.default_rng(SEED)
    rng.shuffle(candidates)

    unique = list(dict.fromkeys(candidates))

    if len(unique) < N_TOTAL:
        raise RuntimeError("Yeterli benzersiz doğal prompt üretilemedi.")

    return unique[:N_TOTAL]


CORPUS = build_corpus()

assert len(CORPUS) == N_TOTAL
assert N_SEARCH + N_FINAL == N_TOTAL

SEARCH_TEXTS = CORPUS[:N_SEARCH]
FINAL_TEXTS = CORPUS[N_SEARCH:]

print("Total prompts    :", N_TOTAL)
print("SEARCH           :", N_SEARCH)
print("FINAL            :", N_FINAL)
print("Shuffle nulls    :", N_SHUFFLE)
print("Source layer     :", f"L{SOURCE_LAYER}")
print("Target layers    :", TARGET_LAYERS)

# =============================================================================
# 3. MODEL
# =============================================================================

print("\n[1/10] Loading frozen model...")

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

if max(CAPTURE_LAYERS) >= len(layers):
    raise RuntimeError("İstenen capture layer model sınırını aşıyor.")

print("GPU              :", torch.cuda.get_device_name(0))
print("Model            :", MODEL_ID)
print("Hidden           :", HIDDEN)
print("Decoder layers   :", len(layers))
print("dtype            :", next(model.parameters()).dtype)

# =============================================================================
# 4. NATURAL ACTIVATION CAPTURE
# =============================================================================

def prepare_text(text):

    chat = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    return tokenizer(
        chat,
        return_tensors="pt",
    ).to(DEVICE)


@torch.inference_mode()
def capture_one(text):

    x = prepare_text(text)

    out = model(
        **x,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    pos = int(
        x["attention_mask"][0].sum().item()
    ) - 1

    row = {}

    for layer_idx in CAPTURE_LAYERS:

        row[layer_idx] = (
            out.hidden_states[
                layer_idx + 1
            ][0, pos]
            .float()
            .cpu()
        )

    del x, out

    return row


def capture_set(texts, tag):

    bank = {
        L: []
        for L in CAPTURE_LAYERS
    }

    for i, text in enumerate(texts, 1):

        row = capture_one(text)

        for L in CAPTURE_LAYERS:
            bank[L].append(
                row[L]
            )

        if (
            i == 1
            or i % 32 == 0
            or i == len(texts)
        ):
            print(
                f"  {tag}: "
                f"{i:>3}/{len(texts)}"
            )

    for L in CAPTURE_LAYERS:

        bank[L] = torch.stack(
            bank[L],
            dim=0,
        ).float()

    return bank


print("\n[2/10] Capturing SEARCH spontaneous activations...")

SEARCH = capture_set(
    SEARCH_TEXTS,
    "SEARCH",
)

print("\n[3/10] Capturing untouched FINAL spontaneous activations...")

FINAL = capture_set(
    FINAL_TEXTS,
    "FINAL",
)

gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# 5. CENTER USING SEARCH MEANS ONLY
#
# FINAL never defines its own coordinate system.
# SEARCH means are frozen and applied to FINAL.
# =============================================================================

print("\n[4/10] Centering natural fluctuations...")

SEARCH_MEAN = {}

SEARCH_C = {}
FINAL_C = {}

for L in CAPTURE_LAYERS:

    mu = SEARCH[L].mean(
        dim=0,
        keepdim=True,
    )

    SEARCH_MEAN[L] = mu

    SEARCH_C[L] = (
        SEARCH[L] - mu
    ).contiguous()

    FINAL_C[L] = (
        FINAL[L] - mu
    ).contiguous()

    search_rms = float(
        torch.sqrt(
            SEARCH_C[L]
            .square()
            .sum(dim=1)
            .mean()
        ).item()
    )

    final_rms = float(
        torch.sqrt(
            FINAL_C[L]
            .square()
            .sum(dim=1)
            .mean()
        ).item()
    )

    print(
        f" L{L:02d} | "
        f"SEARCH fluct RMS={search_rms:.6f} | "
        f"FINAL={final_rms:.6f}"
    )

# =============================================================================
# 6. SAMPLE-SPACE CROSS-COVARIANCE SVD
#
# X: N x D source fluctuations
# Y: N x D target fluctuations
#
# C = X^T Y / (N-1)
#
# rank(C) <= N-1.
#
# Instead of forming D x D C:
#
# X = Qx Rx
# Y = Qy Ry
#
# C = Rx^T (Qx^T Qy) Ry / (N-1)
#
# The middle problem is at most N x N.
# =============================================================================

def cross_cov_svd(
    X_cpu,
    Y_cpu,
    max_modes=MAX_MODES,
):

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
        X,
        mode="reduced",
    )

    Qy, Ry = torch.linalg.qr(
        Y,
        mode="reduced",
    )

    middle = (
        Rx.T
        @ (Qx.T @ Qy)
        @ Ry
    ) / float(n - 1)

    U, S, Vh = torch.linalg.svd(
        middle,
        full_matrices=False,
    )

    k = min(
        max_modes,
        S.numel(),
    )

    U = U[:, :k]
    S = S[:k]
    V = Vh[:k].T

    del X, Y, Qx, Rx, Qy, Ry, middle, Vh

    return (
        U.detach().cpu(),
        S.detach().cpu(),
        V.detach().cpu(),
    )


# =============================================================================
# 7. SEARCH ATLAS
# =============================================================================

print("\n[5/10] Building SEARCH cross-layer transport atlas...")

ATLAS = {}

for target in TARGET_LAYERS:

    print(
        f"  SVD L{SOURCE_LAYER} -> L{target} ..."
    )

    U, S, V = cross_cov_svd(
        SEARCH_C[SOURCE_LAYER],
        SEARCH_C[target],
    )

    ATLAS[target] = {
        "U": U,
        "S": S,
        "V": V,
    }

    print(
        f"    sigma1={S[0].item():.6f} | "
        f"sigma2={S[1].item():.6f} | "
        f"sigma8={S[min(7, len(S)-1)].item():.6f}"
    )

# =============================================================================
# 8. TRANSPORT ENERGY / EFFECTIVE RANK
# =============================================================================

def cumulative_energy(S):

    e = S.double().square()

    total = float(
        e.sum().item()
    )

    if total <= EPS:
        return np.zeros(
            len(S),
            dtype=np.float64,
        )

    return (
        torch.cumsum(
            e,
            dim=0,
        )
        / e.sum()
    ).numpy()


def k_for_energy(cum, level):

    idx = np.where(
        cum >= level
    )[0]

    if len(idx) == 0:
        return len(cum)

    return int(
        idx[0] + 1
    )


def participation_rank(S):

    x = S.double().square()

    s1 = float(
        x.sum().item()
    )

    s2 = float(
        x.square().sum().item()
    )

    if s2 <= EPS:
        return 0.0

    return (
        s1 * s1
    ) / s2


print("\n[6/10] Transport spectrum / effective rank...")

for target in TARGET_LAYERS:

    S = ATLAS[target]["S"]

    cum = cumulative_energy(S)

    ATLAS[target]["cum_energy"] = cum

    ATLAS[target]["participation_rank"] = (
        participation_rank(S)
    )

    print(
        f"\nL{SOURCE_LAYER}->L{target}"
    )

    print(
        f"  participation rank: "
        f"{ATLAS[target]['participation_rank']:.3f}"
    )

    for level in ENERGY_LEVELS:

        print(
            f"  k@{int(level*100):02d}% energy: "
            f"{k_for_energy(cum, level)}"
        )

    for k in REPORT_K:

        if k <= len(cum):

            print(
                f"  E({k:02d}) = "
                f"{100.0*cum[k-1]:.3f}%"
            )

# =============================================================================
# 9. SHUFFLE NULL
#
# Same X.
# Same Y.
# Same marginal distributions.
# Only sample correspondence is destroyed.
#
# Primary statistic:
#
#   singular-value energy captured in first k modes
#
# and individual top singular values.
# =============================================================================

print("\n[7/10] Running sample-shuffle null...")

rng = np.random.default_rng(
    SEED + 152
)

NULL = {
    target: []
    for target in TARGET_LAYERS
}

for b in range(N_SHUFFLE):

    perm = torch.tensor(
        rng.permutation(N_SEARCH),
        dtype=torch.long,
    )

    for target in TARGET_LAYERS:

        _, S_null, _ = cross_cov_svd(
            SEARCH_C[SOURCE_LAYER],
            SEARCH_C[target][perm],
        )

        NULL[target].append(
            S_null.numpy()
        )

    if (
        b == 0
        or (b + 1) % 8 == 0
        or b + 1 == N_SHUFFLE
    ):
        print(
            f"  shuffle "
            f"{b+1:02d}/{N_SHUFFLE}"
        )

for target in TARGET_LAYERS:

    NULL[target] = np.stack(
        NULL[target],
        axis=0,
    )

# =============================================================================
# 10. FINAL GENERALIZATION
#
# SEARCH U/V are frozen.
#
# For each mode:
#
# source score:
#     a = X_final U
#
# response score:
#     b = Y_final V
#
# A genuine cross-layer mode should preserve covariance/correlation on
# untouched FINAL examples.
# =============================================================================

print("\n[8/10] Untouched FINAL validation...")

FINAL_REPORT = {}

for target in TARGET_LAYERS:

    U = ATLAS[target]["U"]
    V = ATLAS[target]["V"]

    Xf = FINAL_C[
        SOURCE_LAYER
    ].float()

    Yf = FINAL_C[
        target
    ].float()

    source_scores = Xf @ U
    response_scores = Yf @ V

    mode_corr = []
    mode_cov = []

    for j in range(
        U.shape[1]
    ):

        a = source_scores[:, j]
        b = response_scores[:, j]

        cov = float(
            (
                (a - a.mean())
                * (b - b.mean())
            ).sum().item()
            / max(
                len(a) - 1,
                1,
            )
        )

        denom = float(
            (
                (a - a.mean()).norm()
                * (b - b.mean()).norm()
            ).item()
        )

        corr = (
            float(
                torch.dot(
                    a - a.mean(),
                    b - b.mean(),
                ).item()
            )
            / (denom + EPS)
        )

        mode_cov.append(cov)
        mode_corr.append(corr)

    FINAL_REPORT[target] = {
        "corr": np.asarray(
            mode_corr,
            dtype=np.float64,
        ),
        "cov": np.asarray(
            mode_cov,
            dtype=np.float64,
        ),
    }

    print(
        f"L{SOURCE_LAYER}->L{target} | "
        f"FINAL corr M1={mode_corr[0]:+.6f} | "
        f"M2={mode_corr[1]:+.6f} | "
        f"M4={mode_corr[3]:+.6f} | "
        f"M8={mode_corr[7]:+.6f}"
    )

# =============================================================================
# 11. SOURCE-SUBSPACE PERSISTENCE ACROSS HORIZONS
#
# Principal angles between U_k source subspaces.
#
# singular values of U_A^T U_B are cosines of principal angles.
#
# We report:
#   mean cos
#   minimum cos
#   RMS cos
# =============================================================================

print("\n[9/10] Cross-horizon source-subspace persistence...")

PERSISTENCE_K = [
    k for k in [4, 8, 16, 32]
    if k <= MAX_MODES
]

PERSISTENCE = {}

for k in PERSISTENCE_K:

    print(
        f"\nSOURCE SUBSPACE k={k}"
    )

    for i, h1 in enumerate(
        TARGET_LAYERS
    ):

        for h2 in TARGET_LAYERS[
            i + 1:
        ]:

            U1 = ATLAS[h1][
                "U"
            ][:, :k]

            U2 = ATLAS[h2][
                "U"
            ][:, :k]

            s = torch.linalg.svdvals(
                U1.T @ U2
            ).numpy()

            mean_cos = float(
                s.mean()
            )

            min_cos = float(
                s.min()
            )

            rms_cos = float(
                np.sqrt(
                    np.mean(
                        s ** 2
                    )
                )
            )

            PERSISTENCE[
                (k, h1, h2)
            ] = (
                mean_cos,
                min_cos,
                rms_cos,
            )

            print(
                f"  L{h1} vs L{h2} | "
                f"mean={mean_cos:.6f} | "
                f"min={min_cos:.6f} | "
                f"RMS={rms_cos:.6f}"
            )

# =============================================================================
# 12. PERSISTENT POTENT SOURCE SUBSPACE
#
# Rather than arbitrarily declaring k=32:
#
# For each horizon take the smallest SEARCH subspace reaching 90%
# transport energy, capped at MAX_MODES.
#
# Form projectors:
#
#     P_L = U_L U_L^T
#
# We do NOT materialize D x D projectors.
#
# Concatenate weighted U bases:
#
#     W = [U_6/sqrt(H), U_10/sqrt(H), ...]
#
# Then left singular vectors of W identify source directions that receive
# support from multiple downstream horizons.
# =============================================================================

print("\n[10/10] Building persistent source potent subspace...")

bases = []
horizon_k90 = {}

for target in TARGET_LAYERS:

    cum = ATLAS[
        target
    ]["cum_energy"]

    k90 = min(
        k_for_energy(
            cum,
            0.90,
        ),
        MAX_MODES,
    )

    horizon_k90[target] = k90

    U = ATLAS[
        target
    ]["U"][:, :k90]

    bases.append(
        U / math.sqrt(
            len(TARGET_LAYERS)
        )
    )

    print(
        f"  L{SOURCE_LAYER}->L{target}: "
        f"k90={k90}"
    )

W = torch.cat(
    bases,
    dim=1,
).float()

Up, Sp, _ = torch.linalg.svd(
    W,
    full_matrices=False,
)

persistent_strength = (
    Sp.square()
).numpy()

# Eigenvalues of mean projector lie in [0,1].
persistent_strength = np.clip(
    persistent_strength,
    0.0,
    1.0,
)

print("\nPersistent source support spectrum:")

for i in range(
    min(
        32,
        len(persistent_strength),
    )
):

    print(
        f" P{i+1:02d} | "
        f"support={persistent_strength[i]:.6f}"
    )

# Thresholds are descriptive, not hard scientific truth.
for threshold in [
    0.90,
    0.75,
    0.50,
    0.25,
]:

    count = int(
        np.sum(
            persistent_strength
            >= threshold
        )
    )

    print(
        f" support >= {threshold:.2f}: "
        f"{count} dimensions"
    )

# =============================================================================
# 13. SHUFFLE FALSIFICATION REPORT
# =============================================================================

print("\n" + "=" * 140)
print("REAL vs SAMPLE-SHUFFLE CROSS-LAYER SPECTRUM")
print("=" * 140)

for target in TARGET_LAYERS:

    S_real = ATLAS[
        target
    ]["S"].numpy()

    null = NULL[
        target
    ]

    print(
        f"\nL{SOURCE_LAYER}->L{target}"
    )

    for mode_idx in [
        0, 1, 3, 7, 15, 31
    ]:

        if mode_idx >= len(S_real):
            continue

        real = float(
            S_real[
                mode_idx
            ]
        )

        vals = null[
            :,
            mode_idx
        ]

        exceed = int(
            np.sum(
                vals >= real
            )
        )

        p = (
            exceed + 1
        ) / (
            N_SHUFFLE + 1
        )

        mu = float(
            vals.mean()
        )

        sd = float(
            vals.std(
                ddof=1
            )
        )

        z = (
            real - mu
        ) / (
            sd + EPS
        )

        print(
            f" M{mode_idx+1:02d} | "
            f"REAL={real:.6f} | "
            f"NULL={mu:.6f}±{sd:.6f} | "
            f"null>={exceed:02d}/{N_SHUFFLE} | "
            f"emp.p={p:.6f} | "
            f"desc.Z={z:+.3f}"
        )

# =============================================================================
# 14. ENERGY CONCENTRATION REPORT
# =============================================================================

print("\n" + "=" * 140)
print("TRANSPORT ENERGY CONCENTRATION")
print("=" * 140)

print(
    f"{'HORIZON':>10} "
    f"{'PRANK':>10} "
    f"{'K50':>7} "
    f"{'K80':>7} "
    f"{'K90':>7} "
    f"{'K95':>7} "
    f"{'E8%':>10} "
    f"{'E16%':>10} "
    f"{'E32%':>10}"
)

for target in TARGET_LAYERS:

    cum = ATLAS[
        target
    ]["cum_energy"]

    pr = ATLAS[
        target
    ][
        "participation_rank"
    ]

    def energy_at(k):
        if k > len(cum):
            return float("nan")
        return (
            100.0
            * float(
                cum[k - 1]
            )
        )

    print(
        f"{('L3->L'+str(target)):>10} "
        f"{pr:>10.3f} "
        f"{k_for_energy(cum,0.50):>7} "
        f"{k_for_energy(cum,0.80):>7} "
        f"{k_for_energy(cum,0.90):>7} "
        f"{k_for_energy(cum,0.95):>7} "
        f"{energy_at(8):>10.3f} "
        f"{energy_at(16):>10.3f} "
        f"{energy_at(32):>10.3f}"
    )

# =============================================================================
# 15. FINAL GENERALIZATION REPORT
# =============================================================================

print("\n" + "=" * 140)
print("UNTOUCHED FINAL — SEARCH-LOCKED SOURCE/RESPONSE MODES")
print("=" * 140)

for target in TARGET_LAYERS:

    corr = FINAL_REPORT[
        target
    ]["corr"]

    print(
        f"\nL{SOURCE_LAYER}->L{target}"
    )

    for j in range(
        min(
            16,
            len(corr),
        )
    ):

        print(
            f" M{j+1:02d} | "
            f"FINAL score correlation="
            f"{corr[j]:+.6f}"
        )

# =============================================================================
# 16. PERSISTENCE SUMMARY
# =============================================================================

print("\n" + "=" * 140)
print("CROSS-HORIZON SOURCE-SUBSPACE PERSISTENCE")
print("=" * 140)

for k in PERSISTENCE_K:

    values = []

    for i, h1 in enumerate(
        TARGET_LAYERS
    ):

        for h2 in TARGET_LAYERS[
            i + 1:
        ]:

            values.append(
                PERSISTENCE[
                    (k, h1, h2)
                ][0]
            )

    print(
        f"k={k:02d} | "
        f"mean pairwise principal-cos="
        f"{np.mean(values):.6f} | "
        f"min pair mean="
        f"{np.min(values):.6f} | "
        f"max pair mean="
        f"{np.max(values):.6f}"
    )

# =============================================================================
# 17. SAVE NUMERICAL ARTIFACT
# =============================================================================

save_dict = {}

for target in TARGET_LAYERS:

    save_dict[
        f"U_L3_to_L{target}"
    ] = ATLAS[
        target
    ]["U"].numpy()

    save_dict[
        f"S_L3_to_L{target}"
    ] = ATLAS[
        target
    ]["S"].numpy()

    save_dict[
        f"V_L3_to_L{target}"
    ] = ATLAS[
        target
    ]["V"].numpy()

    save_dict[
        f"NULL_S_L3_to_L{target}"
    ] = NULL[
        target
    ]

    save_dict[
        f"FINAL_CORR_L3_to_L{target}"
    ] = FINAL_REPORT[
        target
    ]["corr"]

save_dict[
    "PERSISTENT_SOURCE_U"
] = Up.numpy()

save_dict[
    "PERSISTENT_SUPPORT"
] = persistent_strength

np.savez_compressed(
    "/content/TEST152_SPONTANEOUS_FLUCTUATION_ATLAS.npz",
    **save_dict,
)

# =============================================================================
# 18. LOCKED INTERPRETATION
# =============================================================================

print("\n" + "=" * 140)
print("LOCKED INTERPRETATION")
print("=" * 140)

print(
"""
1. Model weights remained frozen.

2. No steering vector was injected.

3. No perturbation, epsilon derivative, JVP, VJP, gradient, LoRA or
   optimizer participated in discovery.

4. TEST 152 measures naturally occurring hidden-state fluctuations.

5. High source-layer variance alone is NOT called "potent".

6. A source direction becomes a transport-mode candidate only through
   cross-layer covariance with naturally occurring downstream
   fluctuations.

7. Source and response modes are explicitly separated:

       U = source / forcing side at L3
       V = downstream response side

   They are not required to be the same vector.

8. The 3584 x 3584 cross-covariance matrix is never explicitly
   materialized. The decomposition is solved through the low-rank
   sample-space factorization.

9. k=32 is NOT assumed to be correct. The singular spectrum determines
   how many modes are required to explain 50%, 80%, 90% and 95% of the
   measured cross-layer transport energy.

10. Sample-shuffle nulls preserve the marginal L3 and downstream
    activation distributions while destroying only example-level
    correspondence.

11. A real spectrum that does not exceed the shuffled spectrum is not
    evidence for an endogenous cross-layer transport channel.

12. SEARCH discovers U/S/V. FINAL is untouched during discovery and
    tests whether the frozen source/response modes preserve score
    correlation on unseen examples.

13. Principal-angle analysis tests whether source transport subspaces
    are shared across L6/L10/L14/L19 rather than existing at only one
    downstream horizon.

14. The persistent source spectrum is derived from the mean support of
    horizon-specific source subspaces. It is a candidate map of
    repeatedly downstream-coupled L3 directions.

15. "Potent" here is an operational cross-layer covariance definition.
    This experiment does NOT yet prove causal downstream readability.

16. Likewise, dimensions outside the recovered subspace must NOT yet be
    called a causal null space. Causal potent/null status requires a
    later intervention experiment.

17. If a compact real spectrum survives shuffle, FINAL, and
    cross-horizon persistence, the next experiment may restrict
    synthetic AkbasCore vectors to that empirically recovered source
    subspace instead of searching blindly in all 3584 dimensions.
"""
)

print("=" * 140)
print("Saved: /content/TEST152_SPONTANEOUS_FLUCTUATION_ATLAS.npz")
print("TEST 152 COMPLETE")
print("=" * 140)
