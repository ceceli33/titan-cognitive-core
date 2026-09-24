# =============================================================================
# TEST 153 — CAUSAL POTENT / NULL TRANSPORT ASSAY
# SELF-CONTAINED TEST152 ATLAS RECONSTRUCTION
# FROZEN DISCOVERY -> LOCK -> CAUSAL INTERVENTION
#
# Qwen/Qwen2.5-7B-Instruct · A100 · BF16
#
# PHASE A
#   Reconstruct TEST152 atlas from its original SEARCH corpus.
#
# PHASE B
#   LOCK:
#       persistent source subspace
#       downstream response subspaces
#
# PHASE C
#   New unseen prompt family.
#   Equal-energy intervention at L3:
#
#       POTENT: z in locked persistent source span
#       NULL:   z orthogonal to broader locked persistent span
#
# Primary:
#
#       G_L = ||delta h_L|| / ||realized delta h_3||
#
#       R_PN = G_P / G_N
#
# Secondary:
#
#       Q_L = ||V_L^T delta h_L||^2 / ||delta h_L||^2
#
# Strong causal result requires BOTH:
#
#       G_P > G_N
#       Q_P > Q_N
#
# No training
# No gradient
# No JVP/VJP
# No semantic steering axis
# No TEST153 data used to discover the subspace
# =============================================================================

import os, sys, subprocess, math, gc, warnings

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
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU bulunamadı.")

# =============================================================================
# CONFIG
# =============================================================================

SEED = 153
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SOURCE_LAYER = 3
TARGET_LAYERS = [6, 10, 14, 19]
CAPTURE_LAYERS = [3, 6, 10, 14, 19]

N_ATLAS = 192

POTENT_K = 4
NULL_EXCLUDE_K = 7

RESPONSE_K = {
    6: 7,
    10: 8,
    14: 7,
    19: 6,
}

N_POTENT_DIRS = 8
N_NULL_DIRS = 8

DOSES = [
    0.00125,
    0.00250,
    0.00500,
]

PRIMARY_DOSE = 0.00250

EPS = 1e-10

DEVICE = torch.device("cuda")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

print("=" * 140)
print("TEST 153 — CAUSAL POTENT / NULL TRANSPORT ASSAY")
print("SELF-CONTAINED TEST152 ATLAS RECONSTRUCTION · LOCKED CAUSAL INTERVENTION")
print("=" * 140)

# =============================================================================
# MODEL
# =============================================================================

print("\n[1/12] Loading frozen model...")

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

print("GPU             :", torch.cuda.get_device_name(0))
print("Model           :", MODEL_ID)
print("Hidden          :", HIDDEN)
print("Layers          :", len(layers))
print("Source          : L3")
print("Targets         :", TARGET_LAYERS)

# =============================================================================
# TEXT PREPARATION
# =============================================================================

SYSTEM = "You are a concise reasoning assistant."

def encode(text):

    chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    x = tokenizer(
        chat,
        return_tensors="pt",
    ).to(DEVICE)

    pos = int(
        x["attention_mask"][0].sum().item()
    ) - 1

    return x, pos

# =============================================================================
# PHASE A — RECONSTRUCT TEST152 SEARCH CORPUS
#
# Exactly same procedural family and SEED=42 selection logic as TEST152.
# =============================================================================

print("\n[2/12] Reconstructing TEST152 SEARCH corpus...")

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

atlas_candidates = []

for s in SUBJECTS:
    for v in VERBS:
        for loc in LOCATIONS:
            for d in DETAILS:
                atlas_candidates.append(
                    f"{s} {v} {loc}. {d}"
                )

rng152 = np.random.default_rng(42)
rng152.shuffle(atlas_candidates)

atlas_candidates = list(
    dict.fromkeys(atlas_candidates)
)

ATLAS_TEXTS = atlas_candidates[:N_ATLAS]

assert len(ATLAS_TEXTS) == 192

print("Atlas SEARCH prompts:", len(ATLAS_TEXTS))

# =============================================================================
# NATURAL ACTIVATION CAPTURE
# =============================================================================

@torch.inference_mode()
def natural_capture(text):

    x, pos = encode(text)

    out = model(
        **x,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    result = {}

    for L in CAPTURE_LAYERS:

        result[L] = (
            out.hidden_states[L + 1][0, pos]
            .float()
            .cpu()
        )

    del out, x

    return result


print("\n[3/12] Rebuilding spontaneous activation atlas...")

BANK = {
    L: []
    for L in CAPTURE_LAYERS
}

for i, text in enumerate(
    ATLAS_TEXTS,
    1,
):

    row = natural_capture(text)

    for L in CAPTURE_LAYERS:
        BANK[L].append(row[L])

    if (
        i == 1
        or i % 32 == 0
        or i == N_ATLAS
    ):
        print(
            f"  atlas {i:03d}/{N_ATLAS}"
        )

for L in CAPTURE_LAYERS:

    BANK[L] = torch.stack(
        BANK[L]
    ).float()

    BANK[L] -= BANK[L].mean(
        dim=0,
        keepdim=True,
    )

# =============================================================================
# CROSS-COVARIANCE SVD
#
# C = X^T Y / (N-1)
#
# Direct 3584x3584 SVD is unnecessary.
#
# Thin QR:
#
# X = Qx Rx
# Y = Qy Ry
#
# C = Rx^T (Qx^T Qy) Ry / (N-1)
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
        X,
        mode="reduced",
    )

    Qy, Ry = torch.linalg.qr(
        Y,
        mode="reduced",
    )

    M = (
        Rx.T
        @ (Qx.T @ Qy)
        @ Ry
    ) / float(n - 1)

    U, S, Vh = torch.linalg.svd(
        M,
        full_matrices=False,
    )

    return (
        U.detach(),
        S.detach(),
        Vh.T.detach(),
    )

# =============================================================================
# BUILD HORIZON-SPECIFIC ATLAS
# =============================================================================

print("\n[4/12] Reconstructing locked source/response modes...")

ATLAS = {}

for target in TARGET_LAYERS:

    U, S, V = cross_cov_svd(
        BANK[SOURCE_LAYER],
        BANK[target],
    )

    ATLAS[target] = {
        "U": U,
        "S": S,
        "V": V,
    }

    print(
        f"L3->L{target} | "
        f"sigma1={S[0].item():.6f} | "
        f"sigma2={S[1].item():.6f} | "
        f"sigma8={S[7].item():.6f}"
    )

# =============================================================================
# VERIFY TEST152 RECONSTRUCTION
# =============================================================================

EXPECTED = {
    6:  (0.625086, 0.465954, 0.125546),
    10: (0.923193, 0.864209, 0.241059),
    14: (1.238007, 1.135841, 0.320666),
    19: (3.393379, 2.675814, 0.729173),
}

print("\nTEST152 fingerprint check:")

for target in TARGET_LAYERS:

    S = ATLAS[target]["S"]

    actual = np.array([
        S[0].item(),
        S[1].item(),
        S[7].item(),
    ])

    expected = np.array(
        EXPECTED[target]
    )

    err = np.max(
        np.abs(
            actual - expected
        )
    )

    print(
        f"L3->L{target} | "
        f"max abs error={err:.8f}"
    )

    if err > 5e-3:
        raise RuntimeError(
            f"TEST152 fingerprint uyuşmadı: L{target}"
        )

# =============================================================================
# PERSISTENT SOURCE SUBSPACE
#
# Exact TEST152 construction:
#
# k90:
# L6  = 7
# L10 = 8
# L14 = 7
# L19 = 6
#
# W = concat(U_h / sqrt(4))
#
# left singular vectors of W.
# =============================================================================

print("\n[5/12] Locking persistent source subspace...")

K90 = {
    6: 7,
    10: 8,
    14: 7,
    19: 6,
}

bases = []

for target in TARGET_LAYERS:

    bases.append(
        ATLAS[target]["U"][
            :,
            :K90[target]
        ] / math.sqrt(
            len(TARGET_LAYERS)
        )
    )

W = torch.cat(
    bases,
    dim=1,
)

PERSISTENT_U, PERSISTENT_S, _ = torch.linalg.svd(
    W,
    full_matrices=False,
)

SUPPORT = (
    PERSISTENT_S.square()
    .detach()
    .cpu()
    .numpy()
)

print("Persistent support:")

for i in range(
    min(10, len(SUPPORT))
):
    print(
        f" P{i+1:02d} = "
        f"{SUPPORT[i]:.6f}"
    )

EXPECTED_SUPPORT = np.array([
    0.998238,
    0.996707,
    0.992332,
    0.951680,
    0.865170,
    0.743055,
    0.678956,
])

support_error = np.max(
    np.abs(
        SUPPORT[:7]
        - EXPECTED_SUPPORT
    )
)

print(
    "TEST152 support fingerprint error:",
    f"{support_error:.8f}"
)

if support_error > 5e-3:
    raise RuntimeError(
        "Persistent subspace TEST152 ile uyuşmadı."
    )

# LOCK

U4 = PERSISTENT_U[
    :,
    :POTENT_K
].contiguous()

U7 = PERSISTENT_U[
    :,
    :NULL_EXCLUDE_K
].contiguous()

RESPONSE = {}

for target in TARGET_LAYERS:

    RESPONSE[target] = ATLAS[
        target
    ]["V"][
        :,
        :RESPONSE_K[target]
    ].contiguous()

print("\nATLAS LOCKED.")
print("Primary POTENT dimensions :", POTENT_K)
print("NULL exclusion dimensions :", NULL_EXCLUDE_K)

# Discovery activations are no longer needed.

del BANK
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# PHASE B — NEW CAUSAL PROMPTS
#
# Completely different sentence family.
# =============================================================================

CAUSAL_PROMPTS = [
    "A visitor places a ticket on the counter while waiting quietly.",
    "The technician checks a display and records a numerical reading.",
    "A student opens a notebook and copies a sentence from the board.",
    "The gardener carries a bucket across the yard during the morning.",
    "A passenger holds a small bag near the station entrance.",
    "The clerk arranges several envelopes beside a keyboard.",
    "A mechanic wipes a tool and returns it to a drawer.",
    "The cook places an empty pan beside the sink.",
    "A reader turns a page and continues reading the printed text.",
    "The worker moves a package onto a shelf in the storage room.",
    "A musician closes a case and places it beside a chair.",
    "The customer examines a receipt near the doorway.",
    "A child sets a pencil on the desk and looks toward the window.",
    "The driver checks a mirror while the vehicle remains stationary.",
    "A librarian carries several books toward a cart.",
    "The painter rinses a brush in a container of water.",
    "A traveler folds a map and places it inside a backpack.",
    "The operator reads a gauge and writes the value on a form.",
    "A shopper places a small item into a reusable bag.",
    "The cleaner moves a chair while preparing the room.",
    "A photographer adjusts a strap while holding a camera.",
    "The baker sets a tray on the counter.",
    "A teacher collects several sheets of paper from a table.",
    "The cyclist places a bottle into a holder.",
    "A carpenter measures a board and marks a point with a pencil.",
    "The receptionist checks a calendar and writes a note.",
    "A researcher closes a folder after reviewing several pages.",
    "The cashier places coins into sections of a drawer.",
    "A technician carries a cable toward a workbench.",
    "A visitor places a coat over the back of a chair.",
    "The assistant moves documents beside a printer.",
    "A worker checks a label before placing a container on a cart.",
]

N_PROMPTS = len(CAUSAL_PROMPTS)

PREPARED = [
    encode(x)
    for x in CAUSAL_PROMPTS
]

print(
    "\nCausal prompts:",
    N_PROMPTS
)

# =============================================================================
# DIRECTION BANK
#
# POTENT:
# random unit combinations inside U4.
#
# NULL:
# random ambient directions with U7 removed.
#
# Null directions mutually orthogonalized.
# =============================================================================

print("\n[6/12] Building matched direction banks...")

gen = torch.Generator(
    device=DEVICE
)

gen.manual_seed(
    SEED + 1000
)

POTENT = []

for _ in range(
    N_POTENT_DIRS
):

    c = torch.randn(
        POTENT_K,
        generator=gen,
        device=DEVICE,
        dtype=torch.float32,
    )

    z = U4 @ c

    z = z / (
        z.norm() + EPS
    )

    POTENT.append(z)

POTENT = torch.stack(
    POTENT
)

NULL = []

while len(NULL) < N_NULL_DIRS:

    z = torch.randn(
        HIDDEN,
        generator=gen,
        device=DEVICE,
        dtype=torch.float32,
    )

    # Remove TEST152 persistent candidate space.

    z = z - U7 @ (
        U7.T @ z
    )

    # Mutually orthogonalize null controls.

    if len(NULL) > 0:

        N = torch.stack(
            NULL,
            dim=1,
        )

        z = z - N @ (
            N.T @ z
        )

    n = z.norm()

    if n > 1e-6:

        NULL.append(
            z / n
        )

NULL = torch.stack(
    NULL
)

null_leak = torch.max(
    torch.abs(
        NULL @ U7
    )
).item()

print("POTENT bank :", POTENT.shape)
print("NULL bank   :", NULL.shape)
print(
    "Max NULL overlap with U7:",
    f"{null_leak:.9f}"
)

# =============================================================================
# FORWARD CAPTURE
#
# L3 block output is intercepted.
#
# Only the last prompt token is modified.
#
# Baseline:
# direction=None
#
# Intervention:
#
#   h3' = BF16(
#       h3 + dose * ||h3|| * z
#   )
#
# We measure the ACTUAL BF16 write:
#
#   delta3_real =
#       float(h3'_BF16) - float(h3_BF16)
# =============================================================================

@torch.inference_mode()
def run_forward(
    prepared,
    direction=None,
    dose=0.0,
):

    x, pos = prepared

    captured = {}

    def source_hook(
        module,
        inputs,
        output,
    ):

        if isinstance(
            output,
            tuple
        ):
            hidden = output[0]
        else:
            hidden = output

        natural = hidden[
            0,
            pos
        ].float()

        captured[
            "source_natural"
        ] = natural.detach().clone()

        if direction is None:

            captured[
                "source_written"
            ] = natural.detach().clone()

            return output

        requested = (
            dose
            * natural.norm()
            * direction
        )

        modified = hidden.clone()

        modified[
            0,
            pos
        ] = (
            natural
            + requested
        ).to(
            hidden.dtype
        )

        captured[
            "requested_norm"
        ] = float(
            requested.norm().item()
        )

        captured[
            "source_written"
        ] = (
            modified[
                0,
                pos
            ]
            .float()
            .detach()
            .clone()
        )

        if isinstance(
            output,
            tuple
        ):
            return (
                modified,
                *output[1:]
            )

        return modified

    hooks = []

    hooks.append(
        layers[
            SOURCE_LAYER
        ].register_forward_hook(
            source_hook
        )
    )

    for target in TARGET_LAYERS:

        def make_hook(L):

            def hook(
                module,
                inputs,
                output,
            ):

                hidden = (
                    output[0]
                    if isinstance(
                        output,
                        tuple
                    )
                    else output
                )

                captured[
                    f"L{L}"
                ] = (
                    hidden[
                        0,
                        pos
                    ]
                    .float()
                    .detach()
                    .clone()
                )

            return hook

        hooks.append(
            layers[
                target
            ].register_forward_hook(
                make_hook(
                    target
                )
            )
        )

    try:

        model(
            **x,
            use_cache=False,
            return_dict=True,
        )

    finally:

        for h in hooks:
            h.remove()

    return captured

# =============================================================================
# BASELINES
# =============================================================================

print("\n[7/12] Capturing causal baselines...")

BASE = []

for i, prepared in enumerate(
    PREPARED,
    1,
):

    BASE.append(
        run_forward(
            prepared
        )
    )

    if (
        i == 1
        or i % 8 == 0
        or i == N_PROMPTS
    ):
        print(
            f"  baseline "
            f"{i:02d}/{N_PROMPTS}"
        )

# =============================================================================
# RESPONSE FRACTION
# =============================================================================

def response_fraction(
    delta,
    target,
):

    V = RESPONSE[
        target
    ]

    projected = (
        V.T @ delta
    )

    return float(
        (
            projected.square().sum()
            / (
                delta.square().sum()
                + EPS
            )
        ).item()
    )

# =============================================================================
# INTERVENTIONS
# =============================================================================

print("\n[8/12] Running POTENT / NULL causal interventions...")

ROWS = []

TOTAL = (
    len(DOSES)
    * (
        N_POTENT_DIRS
        + N_NULL_DIRS
    )
    * N_PROMPTS
)

counter = 0

for dose in DOSES:

    for group, bank in [
        ("POTENT", POTENT),
        ("NULL", NULL),
    ]:

        for direction_id in range(
            bank.shape[0]
        ):

            z = bank[
                direction_id
            ]

            for prompt_id, prepared in enumerate(
                PREPARED
            ):

                baseline = BASE[
                    prompt_id
                ]

                intervention = run_forward(
                    prepared,
                    direction=z,
                    dose=dose,
                )

                delta3 = (
                    intervention[
                        "source_written"
                    ]
                    - baseline[
                        "source_written"
                    ]
                )

                realized = float(
                    delta3.norm().item()
                )

                requested = float(
                    intervention[
                        "requested_norm"
                    ]
                )

                row = {
                    "dose": dose,
                    "group": group,
                    "direction": direction_id,
                    "prompt": prompt_id,
                    "requested": requested,
                    "realized": realized,
                    "write_ratio":
                        realized
                        / (
                            requested
                            + EPS
                        ),
                }

                for target in TARGET_LAYERS:

                    delta = (
                        intervention[
                            f"L{target}"
                        ]
                        - baseline[
                            f"L{target}"
                        ]
                    )

                    dn = float(
                        delta.norm().item()
                    )

                    row[
                        f"G{target}"
                    ] = (
                        dn
                        / (
                            realized
                            + EPS
                        )
                    )

                    row[
                        f"Q{target}"
                    ] = response_fraction(
                        delta,
                        target,
                    )

                    row[
                        f"D{target}"
                    ] = dn

                ROWS.append(row)

                counter += 1

                if (
                    counter == 1
                    or counter % 128 == 0
                    or counter == TOTAL
                ):

                    print(
                        f"  {counter:04d}/{TOTAL} | "
                        f"dose={dose:.5f} | "
                        f"{group} | "
                        f"dir={direction_id+1}"
                    )

# =============================================================================
# AGGREGATION
#
# Direction = causal replication unit.
# Prompts are averaged inside each direction.
# =============================================================================

print("\n[9/12] Aggregating causal results...")

def direction_means(
    group,
    dose,
    key,
):

    n_dirs = (
        N_POTENT_DIRS
        if group == "POTENT"
        else N_NULL_DIRS
    )

    result = []

    for d in range(
        n_dirs
    ):

        values = [
            r[key]
            for r in ROWS
            if (
                r["group"] == group
                and r["direction"] == d
                and abs(
                    r["dose"] - dose
                ) < 1e-12
            )
        ]

        result.append(
            np.mean(values)
        )

    return np.asarray(
        result,
        dtype=np.float64,
    )


def permutation_p(
    A,
    B,
    seed,
    n_perm=20000,
):

    observed = (
        A.mean()
        - B.mean()
    )

    pooled = np.concatenate(
        [A, B]
    )

    rng = np.random.default_rng(
        seed
    )

    exceed = 0

    for _ in range(
        n_perm
    ):

        idx = rng.permutation(
            len(pooled)
        )

        pa = pooled[
            idx[:len(A)]
        ]

        pb = pooled[
            idx[len(A):]
        ]

        if (
            pa.mean()
            - pb.mean()
        ) >= observed:

            exceed += 1

    return (
        exceed + 1
    ) / (
        n_perm + 1
    )

# =============================================================================
# PRIMARY RESULTS
# =============================================================================

print("\n" + "=" * 150)
print("PRIMARY CAUSAL TEST — DOSE 0.00250")
print("=" * 150)

PRIMARY = {}

print(
    f"{'HORIZON':>10} "
    f"{'G_P':>11} "
    f"{'G_N':>11} "
    f"{'R_PN':>9} "
    f"{'pG':>10} "
    f"{'Q_P':>10} "
    f"{'Q_N':>10} "
    f"{'Qratio':>10} "
    f"{'pQ':>10}"
)

for target in TARGET_LAYERS:

    GP = direction_means(
        "POTENT",
        PRIMARY_DOSE,
        f"G{target}",
    )

    GN = direction_means(
        "NULL",
        PRIMARY_DOSE,
        f"G{target}",
    )

    QP = direction_means(
        "POTENT",
        PRIMARY_DOSE,
        f"Q{target}",
    )

    QN = direction_means(
        "NULL",
        PRIMARY_DOSE,
        f"Q{target}",
    )

    gp = GP.mean()
    gn = GN.mean()

    qp = QP.mean()
    qn = QN.mean()

    rpn = gp / (
        gn + EPS
    )

    qr = qp / (
        qn + EPS
    )

    pg = permutation_p(
        GP,
        GN,
        SEED + target,
    )

    pq = permutation_p(
        QP,
        QN,
        SEED + 100 + target,
    )

    PRIMARY[target] = {
        "GP": GP,
        "GN": GN,
        "QP": QP,
        "QN": QN,
        "R": rpn,
        "QR": qr,
        "pG": pg,
        "pQ": pq,
    }

    print(
        f"{('L3->L'+str(target)):>10} "
        f"{gp:>11.6f} "
        f"{gn:>11.6f} "
        f"{rpn:>9.4f} "
        f"{pg:>10.6f} "
        f"{qp:>10.6f} "
        f"{qn:>10.6f} "
        f"{qr:>10.4f} "
        f"{pq:>10.6f}"
    )

# =============================================================================
# DOSE ROBUSTNESS
# =============================================================================

print("\n" + "=" * 150)
print("DOSE ROBUSTNESS")
print("=" * 150)

for dose in DOSES:

    print(
        f"\nDOSE={dose:.5f}"
    )

    for target in TARGET_LAYERS:

        GP = direction_means(
            "POTENT",
            dose,
            f"G{target}",
        )

        GN = direction_means(
            "NULL",
            dose,
            f"G{target}",
        )

        QP = direction_means(
            "POTENT",
            dose,
            f"Q{target}",
        )

        QN = direction_means(
            "NULL",
            dose,
            f"Q{target}",
        )

        print(
            f"L{target:02d} | "
            f"G_P={GP.mean():.6f} | "
            f"G_N={GN.mean():.6f} | "
            f"R={GP.mean()/(GN.mean()+EPS):.4f} | "
            f"Q_P={QP.mean():.6f} | "
            f"Q_N={QN.mean():.6f}"
        )

# =============================================================================
# BF16 WRITE CONTROL
# =============================================================================

print("\n" + "=" * 150)
print("REALIZED BF16 SOURCE WRITE")
print("=" * 150)

for dose in DOSES:

    for group in [
        "POTENT",
        "NULL",
    ]:

        vals = np.array([
            r["write_ratio"]
            for r in ROWS
            if (
                r["group"] == group
                and abs(
                    r["dose"] - dose
                ) < 1e-12
            )
        ])

        realized = np.array([
            r["realized"]
            for r in ROWS
            if (
                r["group"] == group
                and abs(
                    r["dose"] - dose
                ) < 1e-12
            )
        ])

        print(
            f"dose={dose:.5f} | "
            f"{group:>6} | "
            f"realized/requested="
            f"{vals.mean():.6f}±{vals.std():.6f} | "
            f"realized norm="
            f"{realized.mean():.6f}"
        )

# =============================================================================
# PROMPT CONSISTENCY
# =============================================================================

print("\n" + "=" * 150)
print("PROMPT-LEVEL CONSISTENCY — PRIMARY DOSE")
print("=" * 150)

for target in TARGET_LAYERS:

    wins_G = 0
    wins_Q = 0

    ratios_G = []
    ratios_Q = []

    for prompt in range(
        N_PROMPTS
    ):

        gp = np.mean([
            r[f"G{target}"]
            for r in ROWS
            if (
                r["group"] == "POTENT"
                and r["prompt"] == prompt
                and abs(
                    r["dose"]
                    - PRIMARY_DOSE
                ) < 1e-12
            )
        ])

        gn = np.mean([
            r[f"G{target}"]
            for r in ROWS
            if (
                r["group"] == "NULL"
                and r["prompt"] == prompt
                and abs(
                    r["dose"]
                    - PRIMARY_DOSE
                ) < 1e-12
            )
        ])

        qp = np.mean([
            r[f"Q{target}"]
            for r in ROWS
            if (
                r["group"] == "POTENT"
                and r["prompt"] == prompt
                and abs(
                    r["dose"]
                    - PRIMARY_DOSE
                ) < 1e-12
            )
        ])

        qn = np.mean([
            r[f"Q{target}"]
            for r in ROWS
            if (
                r["group"] == "NULL"
                and r["prompt"] == prompt
                and abs(
                    r["dose"]
                    - PRIMARY_DOSE
                ) < 1e-12
            )
        ])

        wins_G += int(
            gp > gn
        )

        wins_Q += int(
            qp > qn
        )

        ratios_G.append(
            gp / (
                gn + EPS
            )
        )

        ratios_Q.append(
            qp / (
                qn + EPS
            )
        )

    print(
        f"L3->L{target} | "
        f"G P>N={wins_G:02d}/{N_PROMPTS} | "
        f"median R={np.median(ratios_G):.4f} | "
        f"Q P>N={wins_Q:02d}/{N_PROMPTS} | "
        f"median Qratio={np.median(ratios_Q):.4f}"
    )

# =============================================================================
# SAVE
# =============================================================================

print("\n[10/12] Saving atlas and causal result...")

save = {
    "persistent_U":
        PERSISTENT_U.detach().cpu().numpy(),

    "persistent_support":
        SUPPORT,

    "potent_dirs":
        POTENT.detach().cpu().numpy(),

    "null_dirs":
        NULL.detach().cpu().numpy(),
}

for target in TARGET_LAYERS:

    save[
        f"source_U_L{target}"
    ] = ATLAS[
        target
    ]["U"].detach().cpu().numpy()

    save[
        f"response_V_L{target}"
    ] = ATLAS[
        target
    ]["V"].detach().cpu().numpy()

    save[
        f"sigma_L{target}"
    ] = ATLAS[
        target
    ]["S"].detach().cpu().numpy()

    save[
        f"GP_L{target}"
    ] = PRIMARY[
        target
    ]["GP"]

    save[
        f"GN_L{target}"
    ] = PRIMARY[
        target
    ]["GN"]

    save[
        f"QP_L{target}"
    ] = PRIMARY[
        target
    ]["QP"]

    save[
        f"QN_L{target}"
    ] = PRIMARY[
        target
    ]["QN"]

OUT_PATH = (
    "/content/"
    "TEST153_CAUSAL_POTENT_NULL_TRANSPORT.npz"
)

np.savez_compressed(
    OUT_PATH,
    **save,
)

# =============================================================================
# LOCKED VERDICT RULE
# =============================================================================

print("\n[11/12] Locked falsification rule...")

print("\n" + "=" * 150)
print("LOCKED INTERPRETATION")
print("=" * 150)

print(
"""
TEST153 is a causal follow-up to TEST152.

TEST152:
    observational natural cross-layer covariance.

TEST153:
    atlas-locked intervention.

No TEST153 prompt or intervention is used to discover U or V.

Primary source space:
    first 4 TEST152 persistent modes.

NULL:
    random directions projected outside the broader first-7 persistent
    source candidate space.

The causal claim is NOT supported merely because POTENT produces a
large downstream delta.

Two predictions are tested separately:

    1. TRANSMISSION

       G_POTENT > G_NULL

    2. RESPONSE SPECIFICITY

       Q_POTENT > Q_NULL

where:

       G = ||delta downstream|| / ||realized BF16 delta L3||

and:

       Q = energy fraction of downstream delta inside TEST152's
           frozen response subspace.

Interpretation:

A)
    G_P > G_N
    Q_P > Q_N
    stable across prompts and doses

    -> strongest evidence that TEST152 recovered a causally privileged
       source-to-response transport structure.

B)
    G_P > G_N
    Q_P ~= Q_N

    -> source subspace may be generally high-transmission, but the
       specific TEST152 response mapping is not causally confirmed.

C)
    G_P ~= G_N
    Q_P > Q_N

    -> atlas predicts downstream orientation but not stronger total
       transmission.

D)
    G_P ~= G_N
    Q_P ~= Q_N

    -> observational TEST152 covariance does not survive causal
       intervention.

Even outcome A does NOT prove that all remaining 3577+ dimensions are
universally null.

The conclusion remains restricted to:
    this model,
    this source layer,
    these downstream horizons,
    these prompt distributions,
    and this intervention regime.
"""
)

# =============================================================================
# FINAL
# =============================================================================

print("\n[12/12] Complete.")

print("=" * 150)
print("Saved:", OUT_PATH)
print("TEST 153 COMPLETE")
print("=" * 150)
