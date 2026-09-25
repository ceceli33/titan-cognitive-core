# =============================================================================
# TEST 154 — TRANSPORT MECHANISM DIFFERENTIAL DIAGNOSIS
# UNIFIED CAUSAL X-RAY OF DEEP REPRESENTATIONAL TRANSPORT
#
# Qwen/Qwen2.5-7B-Instruct · Frozen · BF16
#
# TESTED MECHANISMS
#
# H1 — FIXED HIGHWAY
#      Does the original source geometry remain privileged downstream?
#
# H2 — RELAY / HAND-OFF
#      Does response geometry at one segment overlap / feed the source
#      geometry of the next segment?
#
# H3 — SELF-HEALING / ATTENUATION
#      Does the absolute perturbation collapse with depth?
#
# H4 — NONLINEAR FOLDING
#      Does linear predictivity fail while local nonlinear neighborhood
#      predictivity survives?
#
# H5 — DISTRIBUTED SUPERPOSITION / DIFFUSION
#      Does perturbation energy spread from low-K to broader natural
#      transport subspaces?
#
# H6 — DYNAMIC / CONTEXT-CONDITIONAL ROUTING
#      Does local prompt-neighborhood transport outperform the global
#      pooled transport map?
#
# H7 — NORM SINK / RELATIVE DILUTION
#      Does absolute perturbation survive while perturbation / natural
#      residual norm collapses?
#
# DESIGN
#
#   PHASE A  : SEARCH natural atlas
#   PHASE B  : untouched FINAL validation
#   PHASE C  : lock all global/local transport geometry
#   PHASE D  : one shared causal intervention bank
#   PHASE E  : reuse the SAME perturbation trajectories for all diagnoses
#
# No training
# No LoRA
# No gradients
# No JVP/VJP
# No semantic steering axis
# No causal prompt used for discovery
# =============================================================================

import os, sys, subprocess, math, gc, warnings, itertools

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

SEED = 154
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

LAYERS = [3, 6, 10, 14, 19]

SEGMENTS = [
    (3, 6),
    (6, 10),
    (10, 14),
    (14, 19),
]

N_SEARCH = 192
N_FINAL = 64

K_PRIMARY = 4
K_EXCLUDE = 8

K_SWEEP = [4, 8, 16, 32, 64, 128]

LOCAL_K = 24

N_POTENT_DIRS = 8
N_NULL_DIRS = 8

DOSE = 0.00250

N_GEOM_NULL = 256

EPS = 1e-10

DEVICE = torch.device("cuda")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

print("=" * 156)
print("TEST 154 — TRANSPORT MECHANISM DIFFERENTIAL DIAGNOSIS")
print("FIXED HIGHWAY · RELAY · SELF-HEALING · FOLDING · DIFFUSION · DYNAMIC ROUTING · NORM SINK")
print("=" * 156)

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

blocks = model.model.layers
HIDDEN = model.config.hidden_size

print("GPU            :", torch.cuda.get_device_name(0))
print("Model          :", MODEL_ID)
print("Hidden         :", HIDDEN)
print("Decoder layers :", len(blocks))
print("Layers         :", LAYERS)
print("Segments       :", SEGMENTS)
print("Primary K      :", K_PRIMARY)
print("K sweep        :", K_SWEEP)
print("Local KNN      :", LOCAL_K)
print("Dose           :", DOSE)

# =============================================================================
# TOKENIZATION
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
# NATURAL CORPUS
# =============================================================================

print("\n[2/14] Building SEARCH / FINAL natural corpus...")

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

candidates = [
    f"{s} {v} {loc}. {d}"
    for s in SUBJECTS
    for v in VERBS
    for loc in LOCATIONS
    for d in DETAILS
]

rng_data = np.random.default_rng(42)
rng_data.shuffle(candidates)

candidates = list(dict.fromkeys(candidates))

SEARCH_TEXTS = candidates[:N_SEARCH]
FINAL_TEXTS = candidates[N_SEARCH:N_SEARCH + N_FINAL]

assert len(SEARCH_TEXTS) == N_SEARCH
assert len(FINAL_TEXTS) == N_FINAL

print("SEARCH :", len(SEARCH_TEXTS))
print("FINAL  :", len(FINAL_TEXTS))

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

    for L in LAYERS:

        result[L] = (
            out.hidden_states[L + 1][0, pos]
            .float()
            .cpu()
        )

    del out, x

    return result


def capture_bank(texts, label):

    bank = {
        L: []
        for L in LAYERS
    }

    for i, text in enumerate(texts, 1):

        row = natural_capture(text)

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

    for L in LAYERS:

        bank[L] = torch.stack(
            bank[L]
        ).float()

    return bank


print("\n[3/14] Capturing natural activations...")

SEARCH = capture_bank(
    SEARCH_TEXTS,
    "SEARCH",
)

FINAL = capture_bank(
    FINAL_TEXTS,
    "FINAL",
)

MEAN = {
    L: SEARCH[L].mean(
        dim=0,
        keepdim=True,
    )
    for L in LAYERS
}

XS = {
    L: SEARCH[L] - MEAN[L]
    for L in LAYERS
}

XF = {
    L: FINAL[L] - MEAN[L]
    for L in LAYERS
}

# =============================================================================
# CROSS-COVARIANCE SVD
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
# GLOBAL NATURAL ATLAS
#
# Pairwise source -> target atlas for every ordered forward pair.
#
# This lets us test:
#
#   L3->L6
#   L3->L10
#   L3->L14
#   L3->L19
#
# AND local relay segments:
#
#   L3->L6
#   L6->L10
#   L10->L14
#   L14->L19
# =============================================================================

print("\n[4/14] Building global natural transport atlas...")

PAIRS = []

for i, a in enumerate(LAYERS[:-1]):

    for b in LAYERS[i + 1:]:

        PAIRS.append(
            (a, b)
        )

ATLAS = {}

for a, b in PAIRS:

    U, S, V = cross_cov_svd(
        XS[a],
        XS[b],
    )

    ATLAS[(a, b)] = {
        "U": U,
        "S": S,
        "V": V,
    }

    print(
        f"L{a}->L{b} | "
        f"sigma1={S[0].item():.6f} | "
        f"sigma4={S[3].item():.6f} | "
        f"sigma8={S[7].item():.6f}"
    )

# =============================================================================
# FINAL VALIDATION
# =============================================================================

print("\n[5/14] Untouched FINAL validation...")

FINAL_CORR = {}

for a, b in PAIRS:

    U = (
        ATLAS[(a, b)]["U"][
            :,
            :K_PRIMARY
        ]
        .cpu()
    )

    V = (
        ATLAS[(a, b)]["V"][
            :,
            :K_PRIMARY
        ]
        .cpu()
    )

    A = XF[a] @ U
    B = XF[b] @ V

    corrs = []

    for j in range(K_PRIMARY):

        x = A[:, j].numpy()
        y = B[:, j].numpy()

        c = float(
            np.corrcoef(
                x,
                y,
            )[0, 1]
        )

        corrs.append(c)

    FINAL_CORR[(a, b)] = corrs

    print(
        f"L{a}->L{b} | "
        + " ".join(
            f"M{j+1}={corrs[j]:+.4f}"
            for j in range(K_PRIMARY)
        )
    )

# =============================================================================
# H2 — RELAY GEOMETRY
#
# V_previous versus U_next.
# =============================================================================

print("\n[6/14] Relay geometry...")

def principal_cosines(A, B):

    return torch.linalg.svdvals(
        A.T @ B
    ).clamp(
        0.0,
        1.0,
    )


GEN_GEOM = torch.Generator(
    device=DEVICE
)

GEN_GEOM.manual_seed(
    SEED + 5000
)

RELAY_GEOM = {}

for i in range(
    len(SEGMENTS) - 1
):

    prev_seg = SEGMENTS[i]
    next_seg = SEGMENTS[i + 1]

    mid = prev_seg[1]

    Vprev = ATLAS[
        prev_seg
    ]["V"][
        :,
        :K_PRIMARY
    ]

    Unext = ATLAS[
        next_seg
    ]["U"][
        :,
        :K_PRIMARY
    ]

    pcs = principal_cosines(
        Vprev,
        Unext,
    )

    real = float(
        pcs.mean().item()
    )

    null = []

    for _ in range(N_GEOM_NULL):

        R = torch.randn(
            HIDDEN,
            K_PRIMARY,
            generator=GEN_GEOM,
            device=DEVICE,
            dtype=torch.float32,
        )

        R, _ = torch.linalg.qr(
            R,
            mode="reduced",
        )

        null.append(
            float(
                principal_cosines(
                    Vprev,
                    R,
                )
                .mean()
                .item()
            )
        )

    null = np.asarray(
        null,
        dtype=np.float64,
    )

    p = (
        1
        + np.sum(
            null >= real
        )
    ) / (
        len(null) + 1
    )

    RELAY_GEOM[mid] = {
        "pcs":
            pcs.cpu().numpy(),
        "mean":
            real,
        "null_mean":
            float(null.mean()),
        "null_sd":
            float(null.std()),
        "p":
            float(p),
    }

    print(
        f"L{prev_seg[0]}->L{mid}->L{next_seg[1]} | "
        f"PC={np.round(pcs.cpu().numpy(),4)} | "
        f"mean={real:.6f} | "
        f"null={null.mean():.6f}±{null.std():.6f} | "
        f"p={p:.6f}"
    )

# =============================================================================
# CAUSAL PROMPTS
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

PREPARED = [
    encode(x)
    for x in CAUSAL_PROMPTS
]

N_CAUSAL = len(PREPARED)

# =============================================================================
# CAUSAL FORWARD
# =============================================================================

@torch.inference_mode()
def run_forward(
    prepared,
    source_layer=None,
    direction=None,
    dose=0.0,
):

    x, pos = prepared

    captured = {}

    hooks = []

    if source_layer is not None:

        def source_hook(
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

            natural = (
                hidden[
                    0,
                    pos
                ]
                .float()
            )

            captured[
                "source_natural"
            ] = (
                natural
                .detach()
                .clone()
            )

            if direction is None:

                captured[
                    "source_written"
                ] = (
                    natural
                    .detach()
                    .clone()
                )

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

        hooks.append(
            blocks[
                source_layer
            ].register_forward_hook(
                source_hook
            )
        )

    for L in LAYERS:

        if L == source_layer:
            continue

        def make_hook(layer_id):

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
                    f"L{layer_id}"
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
            blocks[
                L
            ].register_forward_hook(
                make_hook(L)
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

print("\n[7/14] Capturing causal baselines...")

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
        or i == N_CAUSAL
    ):
        print(
            f"  baseline {i:02d}/{N_CAUSAL}"
        )

# =============================================================================
# BUILD POTENT / NULL BANK FOR EACH SEGMENT
# =============================================================================

print("\n[8/14] Building segment direction banks...")

GEN = torch.Generator(
    device=DEVICE
)

GEN.manual_seed(
    SEED + 9000
)

DIRS = {}

for seg in SEGMENTS:

    U = ATLAS[
        seg
    ]["U"]

    Up = U[
        :,
        :K_PRIMARY
    ]

    Ux = U[
        :,
        :K_EXCLUDE
    ]

    potent = []

    for _ in range(
        N_POTENT_DIRS
    ):

        c = torch.randn(
            K_PRIMARY,
            generator=GEN,
            device=DEVICE,
            dtype=torch.float32,
        )

        z = Up @ c

        z = z / (
            z.norm() + EPS
        )

        potent.append(z)

    potent = torch.stack(
        potent
    )

    null = []

    while len(null) < N_NULL_DIRS:

        z = torch.randn(
            HIDDEN,
            generator=GEN,
            device=DEVICE,
            dtype=torch.float32,
        )

        z = z - Ux @ (
            Ux.T @ z
        )

        if null:

            N = torch.stack(
                null,
                dim=1,
            )

            z = z - N @ (
                N.T @ z
            )

        if z.norm() > 1e-6:

            null.append(
                z / z.norm()
            )

    null = torch.stack(
        null
    )

    DIRS[seg] = {
        "POTENT": potent,
        "NULL": null,
    }

    leak = torch.max(
        torch.abs(
            null @ Ux
        )
    ).item()

    print(
        f"L{seg[0]}->L{seg[1]} | "
        f"null overlap={leak:.9f}"
    )

# =============================================================================
# LOCAL ROUTING GEOMETRY
#
# For each causal prompt:
#
# 1. find LOCAL_K nearest SEARCH examples at source layer
# 2. build local cross-covariance source/target SVD
#
# No causal intervention result participates in this construction.
# =============================================================================

print("\n[9/14] Locking prompt-local transport maps...")

LOCAL_ATLAS = {}

for prompt_id, base in enumerate(BASE):

    LOCAL_ATLAS[prompt_id] = {}

    for a, b in SEGMENTS:

        query = (
            base[f"L{a}"]
            .float()
            .cpu()
        )

        natural = SEARCH[a]

        q = (
            query
            - MEAN[a].squeeze(0)
        )

        Xn = XS[a]

        qn = q / (
            q.norm() + EPS
        )

        xn = Xn / (
            Xn.norm(
                dim=1,
                keepdim=True,
            )
            + EPS
        )

        similarity = (
            xn @ qn
        )

        idx = torch.topk(
            similarity,
            k=LOCAL_K,
            largest=True,
        ).indices

        Xlocal = SEARCH[a][idx]
        Ylocal = SEARCH[b][idx]

        Xlocal = (
            Xlocal
            - Xlocal.mean(
                dim=0,
                keepdim=True,
            )
        )

        Ylocal = (
            Ylocal
            - Ylocal.mean(
                dim=0,
                keepdim=True,
            )
        )

        Uloc, Sloc, Vloc = cross_cov_svd(
            Xlocal,
            Ylocal,
        )

        LOCAL_ATLAS[
            prompt_id
        ][(a, b)] = {
            "U":
                Uloc[:, :K_PRIMARY],
            "V":
                Vloc[:, :K_PRIMARY],
            "S":
                Sloc[:K_PRIMARY],
        }

    if (
        prompt_id == 0
        or (prompt_id + 1) % 8 == 0
    ):
        print(
            f"  local atlas "
            f"{prompt_id+1:02d}/{N_CAUSAL}"
        )

# =============================================================================
# CAUSAL TRAJECTORIES
#
# One shared intervention bank.
#
# Each trajectory captures all later diagnostic layers.
# =============================================================================

print("\n[10/14] Running shared causal trajectories...")

ROWS = []

TOTAL = (
    len(SEGMENTS)
    * (
        N_POTENT_DIRS
        + N_NULL_DIRS
    )
    * N_CAUSAL
)

counter = 0

for seg_idx, seg in enumerate(
    SEGMENTS
):

    source, immediate_target = seg

    source_position = LAYERS.index(
        source
    )

    downstream_layers = LAYERS[
        source_position + 1:
    ]

    for group in [
        "POTENT",
        "NULL",
    ]:

        bank = DIRS[
            seg
        ][group]

        for d in range(
            bank.shape[0]
        ):

            z = bank[d]

            for prompt_id, prepared in enumerate(
                PREPARED
            ):

                base = BASE[
                    prompt_id
                ]

                steered = run_forward(
                    prepared,
                    source_layer=source,
                    direction=z,
                    dose=DOSE,
                )

                source_base = (
                    base[
                        f"L{source}"
                    ]
                )

                delta_source = (
                    steered[
                        "source_written"
                    ]
                    - source_base
                )

                source_delta_norm = float(
                    delta_source.norm().item()
                )

                source_natural_norm = float(
                    source_base.norm().item()
                )

                source_relative = (
                    source_delta_norm
                    / (
                        source_natural_norm
                        + EPS
                    )
                )

                row = {
                    "segment": seg,
                    "source": source,
                    "immediate_target":
                        immediate_target,
                    "group": group,
                    "direction": d,
                    "prompt": prompt_id,
                    "source_delta_norm":
                        source_delta_norm,
                    "source_natural_norm":
                        source_natural_norm,
                    "source_relative":
                        source_relative,
                }

                for L in downstream_layers:

                    delta = (
                        steered[
                            f"L{L}"
                        ]
                        - base[
                            f"L{L}"
                        ]
                    )

                    natural_norm = float(
                        base[
                            f"L{L}"
                        ].norm().item()
                    )

                    dn = float(
                        delta.norm().item()
                    )

                    relative = (
                        dn
                        / (
                            natural_norm
                            + EPS
                        )
                    )

                    # ---------------------------------------------------------
                    # H3 SELF-HEALING
                    # absolute propagation relative to realized source write
                    # ---------------------------------------------------------

                    absolute_gain = (
                        dn
                        / (
                            source_delta_norm
                            + EPS
                        )
                    )

                    # ---------------------------------------------------------
                    # H7 NORM SINK
                    # relative perturbation survival
                    # ---------------------------------------------------------

                    relative_survival = (
                        relative
                        / (
                            source_relative
                            + EPS
                        )
                    )

                    row[
                        f"D_L{L}"
                    ] = dn

                    row[
                        f"NORM_L{L}"
                    ] = natural_norm

                    row[
                        f"G_L{L}"
                    ] = absolute_gain

                    row[
                        f"REL_L{L}"
                    ] = relative

                    row[
                        f"RSURV_L{L}"
                    ] = relative_survival

                    # ---------------------------------------------------------
                    # H1 FIXED HIGHWAY
                    #
                    # Project downstream delta into the RESPONSE side of the
                    # direct source->L atlas.
                    # ---------------------------------------------------------

                    direct_pair = (
                        source,
                        L,
                    )

                    Vdirect = ATLAS[
                        direct_pair
                    ]["V"][
                        :,
                        :K_PRIMARY
                    ]

                    fixed_coeff = (
                        Vdirect.T
                        @ delta
                    )

                    fixed_q = float(
                        (
                            fixed_coeff
                            .square()
                            .sum()
                            / (
                                delta
                                .square()
                                .sum()
                                + EPS
                            )
                        ).item()
                    )

                    row[
                        f"FIXED_Q_L{L}"
                    ] = fixed_q

                    # ---------------------------------------------------------
                    # H5 DIFFUSION / SUPERPOSITION
                    #
                    # Multi-K captured response energy.
                    # ---------------------------------------------------------

                    Vfull = ATLAS[
                        direct_pair
                    ]["V"]

                    for K in K_SWEEP:

                        k_use = min(
                            K,
                            Vfull.shape[1],
                        )

                        coeff = (
                            Vfull[
                                :,
                                :k_use
                            ].T
                            @ delta
                        )

                        qk = float(
                            (
                                coeff
                                .square()
                                .sum()
                                / (
                                    delta
                                    .square()
                                    .sum()
                                    + EPS
                                )
                            ).item()
                        )

                        row[
                            f"K{K}_L{L}"
                        ] = qk

                # -------------------------------------------------------------
                # H2 RELAY
                #
                # Immediate response -> next source.
                # -------------------------------------------------------------

                b = immediate_target

                delta_b = (
                    steered[
                        f"L{b}"
                    ]
                    - base[
                        f"L{b}"
                    ]
                )

                Vcurrent = ATLAS[
                    seg
                ]["V"][
                    :,
                    :K_PRIMARY
                ]

                response_component = (
                    Vcurrent
                    @ (
                        Vcurrent.T
                        @ delta_b
                    )
                )

                q_response = float(
                    (
                        response_component
                        .square()
                        .sum()
                        / (
                            delta_b
                            .square()
                            .sum()
                            + EPS
                        )
                    ).item()
                )

                row[
                    "RELAY_Q_RESPONSE"
                ] = q_response

                if seg_idx < (
                    len(SEGMENTS) - 1
                ):

                    next_seg = SEGMENTS[
                        seg_idx + 1
                    ]

                    Unext = ATLAS[
                        next_seg
                    ]["U"][
                        :,
                        :K_PRIMARY
                    ]

                    next_component = (
                        Unext.T
                        @ delta_b
                    )

                    q_next = float(
                        (
                            next_component
                            .square()
                            .sum()
                            / (
                                delta_b
                                .square()
                                .sum()
                                + EPS
                            )
                        ).item()
                    )

                    handoff = float(
                        (
                            (
                                Unext.T
                                @ response_component
                            )
                            .square()
                            .sum()
                            / (
                                response_component
                                .square()
                                .sum()
                                + EPS
                            )
                        ).item()
                    )

                    row[
                        "RELAY_Q_NEXT"
                    ] = q_next

                    row[
                        "RELAY_H"
                    ] = handoff

                # -------------------------------------------------------------
                # H6 DYNAMIC ROUTING
                #
                # Global response projection vs prompt-local response map.
                # -------------------------------------------------------------

                local_V = LOCAL_ATLAS[
                    prompt_id
                ][seg]["V"]

                global_proj = (
                    Vcurrent.T
                    @ delta_b
                )

                local_proj = (
                    local_V.T
                    @ delta_b
                )

                row[
                    "GLOBAL_Q"
                ] = float(
                    (
                        global_proj
                        .square()
                        .sum()
                        / (
                            delta_b
                            .square()
                            .sum()
                            + EPS
                        )
                    ).item()
                )

                row[
                    "LOCAL_Q"
                ] = float(
                    (
                        local_proj
                        .square()
                        .sum()
                        / (
                            delta_b
                            .square()
                            .sum()
                            + EPS
                        )
                    ).item()
                )

                # -------------------------------------------------------------
                # H4 NONLINEAR FOLDING
                #
                # Local-neighborhood displacement compatibility.
                #
                # We do not train an autoencoder.
                #
                # Compare delta direction against natural displacement
                # directions from the LOCAL_K neighborhood at target layer.
                #
                # If pooled linear Q is low but this local nonlinear /
                # neighborhood score survives, folding becomes plausible.
                # -------------------------------------------------------------

                query_source = (
                    base[
                        f"L{source}"
                    ]
                    .float()
                    .cpu()
                )

                centered_query = (
                    query_source
                    - MEAN[
                        source
                    ].squeeze(0)
                )

                Xsrc = XS[source]

                sims = (
                    (
                        Xsrc
                        / (
                            Xsrc.norm(
                                dim=1,
                                keepdim=True,
                            )
                            + EPS
                        )
                    )
                    @ (
                        centered_query
                        / (
                            centered_query.norm()
                            + EPS
                        )
                    )
                )

                idx = torch.topk(
                    sims,
                    k=LOCAL_K,
                ).indices

                target_neighbors = (
                    SEARCH[b][idx]
                    - SEARCH[b][idx].mean(
                        dim=0,
                        keepdim=True,
                    )
                )

                dt = (
                    delta_b
                    .detach()
                    .cpu()
                )

                dt_u = (
                    dt
                    / (
                        dt.norm()
                        + EPS
                    )
                )

                tn = (
                    target_neighbors
                    / (
                        target_neighbors.norm(
                            dim=1,
                            keepdim=True,
                        )
                        + EPS
                    )
                )

                local_cos = torch.abs(
                    tn @ dt_u
                )

                row[
                    "FOLD_LOCAL_MAX"
                ] = float(
                    local_cos.max().item()
                )

                row[
                    "FOLD_LOCAL_MEAN"
                ] = float(
                    local_cos.mean().item()
                )

                ROWS.append(row)

                counter += 1

                if (
                    counter == 1
                    or counter % 128 == 0
                    or counter == TOTAL
                ):
                    print(
                        f"  {counter:04d}/{TOTAL} | "
                        f"L{source}->L{immediate_target} | "
                        f"{group} | dir={d+1}"
                    )

# =============================================================================
# STATISTICS
# =============================================================================

print("\n[11/14] Aggregating mechanism evidence...")

def direction_means(
    seg,
    group,
    key,
):

    n = (
        N_POTENT_DIRS
        if group == "POTENT"
        else N_NULL_DIRS
    )

    out = []

    for d in range(n):

        vals = [
            r[key]
            for r in ROWS
            if (
                r["segment"] == seg
                and r["group"] == group
                and r["direction"] == d
                and key in r
            )
        ]

        out.append(
            float(
                np.mean(vals)
            )
        )

    return np.asarray(
        out,
        dtype=np.float64,
    )


def exact_p(A, B):

    pooled = np.concatenate(
        [A, B]
    )

    nA = len(A)

    observed = (
        A.mean()
        - B.mean()
    )

    exceed = 0
    total = 0

    for combo in itertools.combinations(
        range(len(pooled)),
        nA,
    ):

        mask = np.zeros(
            len(pooled),
            dtype=bool,
        )

        mask[
            list(combo)
        ] = True

        diff = (
            pooled[mask].mean()
            - pooled[~mask].mean()
        )

        if diff >= (
            observed - 1e-15
        ):
            exceed += 1

        total += 1

    return (
        exceed / total
    )


def compare(seg, key):

    P = direction_means(
        seg,
        "POTENT",
        key,
    )

    N = direction_means(
        seg,
        "NULL",
        key,
    )

    return {
        "P": float(P.mean()),
        "N": float(N.mean()),
        "ratio": float(
            P.mean()
            / (
                N.mean()
                + EPS
            )
        ),
        "p": float(
            exact_p(
                P,
                N,
            )
        ),
    }

# =============================================================================
# MECHANISM EVIDENCE MATRIX
# =============================================================================

print("\n" + "=" * 168)
print("MECHANISM EVIDENCE MATRIX")
print("=" * 168)

SUMMARY = {}

for seg_idx, seg in enumerate(
    SEGMENTS
):

    a, b = seg

    SUMMARY[seg] = {}

    print(
        f"\n{'='*80}"
    )
    print(
        f"SEGMENT L{a}->L{b}"
    )
    print(
        f"{'='*80}"
    )

    # -------------------------------------------------------------------------
    # H2 RELAY
    # -------------------------------------------------------------------------

    x = compare(
        seg,
        "RELAY_Q_RESPONSE",
    )

    SUMMARY[seg][
        "relay_response"
    ] = x

    print(
        "RELAY response   | "
        f"P={x['P']:.6f} "
        f"N={x['N']:.6f} "
        f"R={x['ratio']:.4f} "
        f"p={x['p']:.6f}"
    )

    if seg_idx < (
        len(SEGMENTS) - 1
    ):

        x = compare(
            seg,
            "RELAY_Q_NEXT",
        )

        SUMMARY[seg][
            "relay_next"
        ] = x

        print(
            "RELAY next       | "
            f"P={x['P']:.6f} "
            f"N={x['N']:.6f} "
            f"R={x['ratio']:.4f} "
            f"p={x['p']:.6f}"
        )

        x = compare(
            seg,
            "RELAY_H",
        )

        SUMMARY[seg][
            "relay_handoff"
        ] = x

        print(
            "RELAY handoff    | "
            f"P={x['P']:.6f} "
            f"N={x['N']:.6f} "
            f"R={x['ratio']:.4f} "
            f"p={x['p']:.6f}"
        )

    # -------------------------------------------------------------------------
    # H3 / H7 DEPTH TRAJECTORY
    # -------------------------------------------------------------------------

    source_index = LAYERS.index(a)

    for L in LAYERS[
        source_index + 1:
    ]:

        g = compare(
            seg,
            f"G_L{L}",
        )

        r = compare(
            seg,
            f"RSURV_L{L}",
        )

        q = compare(
            seg,
            f"FIXED_Q_L{L}",
        )

        SUMMARY[seg][
            f"gain_L{L}"
        ] = g

        SUMMARY[seg][
            f"relative_L{L}"
        ] = r

        SUMMARY[seg][
            f"fixed_L{L}"
        ] = q

        print(
            f"L{L:02d} absolute G | "
            f"P={g['P']:.4f} "
            f"N={g['N']:.4f} "
            f"R={g['ratio']:.4f} "
            f"p={g['p']:.6f}"
        )

        print(
            f"L{L:02d} rel survival| "
            f"P={r['P']:.4f} "
            f"N={r['N']:.4f} "
            f"R={r['ratio']:.4f} "
            f"p={r['p']:.6f}"
        )

        print(
            f"L{L:02d} fixed Q     | "
            f"P={q['P']:.6f} "
            f"N={q['N']:.6f} "
            f"R={q['ratio']:.4f} "
            f"p={q['p']:.6f}"
        )

    # -------------------------------------------------------------------------
    # H5 DIFFUSION
    # -------------------------------------------------------------------------

    print("\nMULTI-K CAPTURE at immediate target:")

    for K in K_SWEEP:

        x = compare(
            seg,
            f"K{K}_L{b}",
        )

        SUMMARY[seg][
            f"K{K}"
        ] = x

        print(
            f"  K={K:03d} | "
            f"P={x['P']:.6f} "
            f"N={x['N']:.6f} "
            f"R={x['ratio']:.4f} "
            f"p={x['p']:.6f}"
        )

    # -------------------------------------------------------------------------
    # H6 DYNAMIC ROUTING
    # -------------------------------------------------------------------------

    global_x = compare(
        seg,
        "GLOBAL_Q",
    )

    local_x = compare(
        seg,
        "LOCAL_Q",
    )

    SUMMARY[seg][
        "global_route"
    ] = global_x

    SUMMARY[seg][
        "local_route"
    ] = local_x

    print(
        "\nDYNAMIC routing:"
    )

    print(
        f"  GLOBAL | "
        f"P={global_x['P']:.6f} "
        f"N={global_x['N']:.6f} "
        f"R={global_x['ratio']:.4f}"
    )

    print(
        f"  LOCAL  | "
        f"P={local_x['P']:.6f} "
        f"N={local_x['N']:.6f} "
        f"R={local_x['ratio']:.4f}"
    )

    # -------------------------------------------------------------------------
    # H4 FOLDING
    # -------------------------------------------------------------------------

    fold_max = compare(
        seg,
        "FOLD_LOCAL_MAX",
    )

    fold_mean = compare(
        seg,
        "FOLD_LOCAL_MEAN",
    )

    SUMMARY[seg][
        "fold_max"
    ] = fold_max

    SUMMARY[seg][
        "fold_mean"
    ] = fold_mean

    print(
        "\nNONLINEAR/local geometry:"
    )

    print(
        f"  local max | "
        f"P={fold_max['P']:.6f} "
        f"N={fold_max['N']:.6f} "
        f"R={fold_max['ratio']:.4f} "
        f"p={fold_max['p']:.6f}"
    )

    print(
        f"  local mean| "
        f"P={fold_mean['P']:.6f} "
        f"N={fold_mean['N']:.6f} "
        f"R={fold_mean['ratio']:.4f} "
        f"p={fold_mean['p']:.6f}"
    )

# =============================================================================
# DEPTH X-RAY — L3 SOURCE
#
# Gives one compact picture of what happened to the TEST153-type
# early potent perturbation as it travels through depth.
# =============================================================================

print("\n" + "=" * 168)
print("L3 POTENT TRAJECTORY X-RAY")
print("=" * 168)

SEG0 = (3, 6)

print(
    f"{'LAYER':>7} "
    f"{'ABS_G':>12} "
    f"{'REL_SURV':>12} "
    f"{'FIXED_Q':>12} "
    f"{'K4':>12} "
    f"{'K32':>12} "
    f"{'K64':>12} "
    f"{'K128':>12}"
)

for L in [6, 10, 14, 19]:

    g = direction_means(
        SEG0,
        "POTENT",
        f"G_L{L}",
    ).mean()

    rs = direction_means(
        SEG0,
        "POTENT",
        f"RSURV_L{L}",
    ).mean()

    fq = direction_means(
        SEG0,
        "POTENT",
        f"FIXED_Q_L{L}",
    ).mean()

    vals = {}

    for K in [
        4,
        32,
        64,
        128,
    ]:

        vals[K] = (
            direction_means(
                SEG0,
                "POTENT",
                f"K{K}_L{L}",
            ).mean()
        )

    print(
        f"L{L:02d} "
        f"{g:12.6f} "
        f"{rs:12.6f} "
        f"{fq:12.6f} "
        f"{vals[4]:12.6f} "
        f"{vals[32]:12.6f} "
        f"{vals[64]:12.6f} "
        f"{vals[128]:12.6f}"
    )

# =============================================================================
# SAVE
# =============================================================================

print("\n[12/14] Saving complete diagnostic capsule...")

SAVE = {}

for pair in PAIRS:

    a, b = pair

    SAVE[
        f"U_L{a}_L{b}"
    ] = (
        ATLAS[pair]["U"]
        .cpu()
        .numpy()
    )

    SAVE[
        f"S_L{a}_L{b}"
    ] = (
        ATLAS[pair]["S"]
        .cpu()
        .numpy()
    )

    SAVE[
        f"V_L{a}_L{b}"
    ] = (
        ATLAS[pair]["V"]
        .cpu()
        .numpy()
    )

    SAVE[
        f"FINAL_CORR_L{a}_L{b}"
    ] = np.asarray(
        FINAL_CORR[pair],
        dtype=np.float64,
    )

for mid, x in RELAY_GEOM.items():

    SAVE[
        f"RELAY_PCOS_L{mid}"
    ] = x["pcs"]

    SAVE[
        f"RELAY_GEOM_L{mid}"
    ] = np.asarray(
        [
            x["mean"],
            x["null_mean"],
            x["null_sd"],
            x["p"],
        ],
        dtype=np.float64,
    )

# Save raw causal rows as structured object-compatible arrays.

SAVE["row_segment_a"] = np.asarray(
    [
        r["segment"][0]
        for r in ROWS
    ],
    dtype=np.int16,
)

SAVE["row_segment_b"] = np.asarray(
    [
        r["segment"][1]
        for r in ROWS
    ],
    dtype=np.int16,
)

SAVE["row_group"] = np.asarray(
    [
        1 if r["group"] == "POTENT" else 0
        for r in ROWS
    ],
    dtype=np.int8,
)

SAVE["row_direction"] = np.asarray(
    [
        r["direction"]
        for r in ROWS
    ],
    dtype=np.int16,
)

SAVE["row_prompt"] = np.asarray(
    [
        r["prompt"]
        for r in ROWS
    ],
    dtype=np.int16,
)

# Numeric telemetry matrix.

NUMERIC_KEYS = sorted(
    set().union(
        *[
            {
                k
                for k, v in r.items()
                if isinstance(
                    v,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                )
            }
            for r in ROWS
        ]
    )
    - {
        "source",
        "immediate_target",
        "direction",
        "prompt",
    }
)

SAVE["telemetry_keys"] = np.asarray(
    NUMERIC_KEYS,
    dtype="U64",
)

TELEMETRY = np.full(
    (
        len(ROWS),
        len(NUMERIC_KEYS),
    ),
    np.nan,
    dtype=np.float32,
)

for i, r in enumerate(ROWS):

    for j, key in enumerate(
        NUMERIC_KEYS
    ):

        if key in r:
            TELEMETRY[
                i,
                j
            ] = float(
                r[key]
            )

SAVE["telemetry"] = TELEMETRY

OUT = (
    "/content/"
    "TEST154_TRANSPORT_MECHANISM_DIAGNOSIS.npz"
)

np.savez_compressed(
    OUT,
    **SAVE,
)

# =============================================================================
# INTERPRETATION CONTRACT
# =============================================================================

print("\n[13/14] Interpretation contract...")

print("\n" + "=" * 168)
print("LOCKED INTERPRETATION CONTRACT")
print("=" * 168)

print(
"""
TEST154 does NOT select a winner by construction.

Several mechanisms may coexist.

H1 FIXED HIGHWAY
----------------
Supported only if source-specific response capture remains privileged
across depth rather than only at the first downstream horizon.

H2 RELAY
--------
Supported when:
    V_previous geometrically overlaps U_next,
and causal perturbations preferentially enter:
    V_current
and:
    U_next.

H3 SELF-HEALING / ATTENUATION
-----------------------------
Suggested when absolute perturbation propagation itself falls strongly
with depth.

A falling projection alone is NOT self-healing if total perturbation
energy remains large.

H4 NONLINEAR FOLDING
--------------------
Suggested only when global linear transport capture becomes weak while
local neighborhood compatibility remains selectively stronger for
POTENT than NULL.

This assay is deliberately only a first nonlinear probe.
It does not prove a curved manifold.

H5 DISTRIBUTED SUPERPOSITION / DIFFUSION
----------------------------------------
Suggested when low-K capture falls but broader K=32/64/128 capture
recovers substantial perturbation energy.

If total perturbation remains large while energy escapes K=4, the
signal has not necessarily disappeared.

H6 DYNAMIC / CONTEXT-CONDITIONAL ROUTING
----------------------------------------
Suggested when prompt-local transport maps systematically outperform
the pooled global map, especially for POTENT perturbations.

Local improvement by itself is not sufficient: it must exceed the
corresponding NULL behavior.

H7 NORM SINK / RELATIVE DILUTION
--------------------------------
Suggested when absolute perturbation remains substantial but:

    ||delta h_L|| / ||h_L||

shrinks strongly relative to its source-layer value.

This distinguishes relative dilution from actual disappearance.

IMPORTANT
---------
These mechanisms are not mutually exclusive.

Possible outcome:

    early relay
        +
    mid-depth diffusion
        +
    late context-dependent routing

is fully allowed.

The experiment therefore reports a MECHANISM EVIDENCE MATRIX rather
than forcing a single winning theory.
"""
)

# =============================================================================
# COMPLETE
# =============================================================================

print("\n[14/14] Complete.")
print("=" * 168)
print("Saved:", OUT)
print("TEST 154 COMPLETE")
print("=" * 168)
