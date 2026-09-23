# =============================================================================
# TEST 146 — AKBASCORE PARAMETER-SPACE INTERFERENCE X-RAY
# PART 1/2
#
# DIAGNOSTIC ONLY — NO GRADIENT PROJECTION / NO INTERVENTION
#
# A = ABOVE <-> BELOW
# B = BEFORE <-> AFTER
# Qwen2.5-7B-Instruct | A100 40 GB
#
# TEST145 VRAM-SAFE architecture preserved.
# X-ray is active ONLY during:
#     A -> B
#     B -> A
#
# Locked:
#     XRAY_N_PROBES = 4
#     XRAY_EVERY = 10
#
# Measures:
#     g_retain
#     g_new
#     cos(g_retain, g_new)
#     actual AdamW parameter step Δtheta
#     g_retain · Δtheta
#     cos(g_retain, Δtheta)
#     actual short-horizon retention-loss change
#     layer × q/k/v/o parameter-space geometry
#
# X-ray NEVER modifies gradients.
# =============================================================================

import sys
import os
import gc
import math
import time
import json
import random
import shutil
import warnings
import subprocess
import re
import importlib.metadata as md

from datetime import datetime, timezone


# =============================================================================
# 0. ENVIRONMENT
# =============================================================================

print("=" * 150)
print("TEST 146 — AKBASCORE PARAMETER-SPACE INTERFERENCE X-RAY")
print("DIAGNOSTIC ONLY — NO GRADIENT PROJECTION / NO INTERVENTION")
print("=" * 150)


def pkg_version(name):
    try:
        return md.version(name)
    except Exception:
        return "NOT INSTALLED"


try:
    _torchao = md.version("torchao")
except md.PackageNotFoundError:
    _torchao = None


if _torchao is not None:

    print(f"Removing torchao {_torchao}...")

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "torchao",
        ],
        text=True,
        capture_output=True,
    )

    print(r.stdout)

    for name in list(sys.modules):

        if name == "torchao" or name.startswith("torchao."):
            del sys.modules[name]


try:
    md.version("torchao")

    raise RuntimeError(
        "torchao could not be removed. Restart runtime and rerun."
    )

except md.PackageNotFoundError:
    pass


import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image

import transformers
import peft

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)


warnings.filterwarnings("ignore")

plt.ioff()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")


DEVICE = torch.device("cuda")

GB = 1024 ** 3


print("Python       :", sys.version.split()[0])
print("PyTorch      :", torch.__version__)
print("Transformers :", transformers.__version__)
print("PEFT         :", peft.__version__)
print("CUDA         :", torch.version.cuda)
print("GPU          :", torch.cuda.get_device_name(0))

print(
    "VRAM         :",
    f"{torch.cuda.get_device_properties(0).total_memory/GB:.2f} GB",
)


# =============================================================================
# 1. REPRODUCIBILITY / LOCKED CONFIG
# =============================================================================

SEED = 42


def reseed(offset=0):

    s = SEED + int(offset)

    random.seed(s)
    np.random.seed(s)

    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


reseed()

torch.backends.cuda.matmul.allow_tf32 = True


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

N_LAYERS = 20

HARD_MAX_UPDATES = 2000


# -----------------------------------------------------------------------------
# LOCKED SEASC 3.0
# -----------------------------------------------------------------------------

IVME = 0.10

SONUM = 0.30

ZIRVE = 0.70

TABAN = 0.20


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

LEARNING_RATE = 2e-5

LAMBDA_GEOMETRY = 1.0

LAMBDA_PRESERVE = 0.05

GRAD_CLIP = 1.0


# -----------------------------------------------------------------------------
# LoRA
# -----------------------------------------------------------------------------

LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.0


# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

MAX_TRAIN_TOKENS = 128

PRESERVE_TOP_K = 256


N_TRAIN = 64

N_VALIDATION = 24

N_FINAL = 24


# -----------------------------------------------------------------------------
# STOP / SELECTION
# -----------------------------------------------------------------------------

EVAL_EVERY = 10

MIN_UPDATES_BEFORE_STOP = 50

MIN_PROGRESS_IMPROVEMENT = 0.0025

MIN_DISTANCE_IMPROVEMENT = 0.0025

MIN_ALIGNMENT_IMPROVEMENT = 0.0010

PATIENCE_EVALS = 5

OVERSHOOT_PROGRESS_DROP = 0.05

ALIGNMENT_FLOOR = 0.0


# -----------------------------------------------------------------------------
# GENERATION
# -----------------------------------------------------------------------------

REPETITION_PENALTY = 1.15

MAX_BEHAVIOR_NEW_TOKENS = 160


SYSTEM_PROMPT = (
    "You are a concise reasoning assistant. "
    "Describe the requested relationship clearly and precisely."
)


PRINT_FULL_LAYER_TABLE_EVERY_EVAL = True


# =============================================================================
# TEST 146 — LOCKED X-RAY CONFIG
# =============================================================================

XRAY_EVERY = 10

XRAY_N_PROBES = 4

XRAY_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)

XRAY_EPS = 1e-12

XRAY_INTERVENTION = False


print("\nTEST 146 X-RAY LOCK")

print("XRAY_EVERY        :", XRAY_EVERY)

print("XRAY_N_PROBES     :", XRAY_N_PROBES)

print("XRAY_MODULES      :", XRAY_MODULES)

print("XRAY_INTERVENTION :", XRAY_INTERVENTION)


if XRAY_INTERVENTION:
    raise RuntimeError(
        "TEST146 must remain diagnostic-only. "
        "Gradient intervention is forbidden."
    )


# =============================================================================
# OUTPUT
# =============================================================================

OUT_DIR = "/content/test146_parameter_space_xray"

FIG_DIR = f"{OUT_DIR}/FIGURES"

CSV_DIR = f"{OUT_DIR}/CSV"

JSON_DIR = f"{OUT_DIR}/JSON"

REPORT_DIR = f"{OUT_DIR}/REPORT"

NPZ_DIR = f"{OUT_DIR}/NPZ"

ADAPTER_DIR = f"{OUT_DIR}/ADAPTERS"


for d in [
    OUT_DIR,
    FIG_DIR,
    CSV_DIR,
    JSON_DIR,
    REPORT_DIR,
    NPZ_DIR,
    ADAPTER_DIR,
]:

    os.makedirs(
        d,
        exist_ok=True,
    )


EXPERIMENT_START_UTC = datetime.now(
    timezone.utc
).isoformat()


# =============================================================================
# 2. VRAM LIFECYCLE
# =============================================================================

def gpu_mem():

    return {
        "allocated":
            torch.cuda.memory_allocated() / GB,

        "reserved":
            torch.cuda.memory_reserved() / GB,

        "max_allocated":
            torch.cuda.max_memory_allocated() / GB,
    }


def memory_report(label):

    m = gpu_mem()

    print(
        f"[VRAM] {label:<34} | "
        f"allocated={m['allocated']:.3f} GB | "
        f"reserved={m['reserved']:.3f} GB | "
        f"peak={m['max_allocated']:.3f} GB"
    )


def cuda_cleanup(label="cleanup"):

    gc.collect()

    torch.cuda.empty_cache()

    try:
        torch.cuda.ipc_collect()

    except Exception:
        pass

    gc.collect()

    memory_report(label)


def destroy_model(model, label):

    if model is not None:

        try:
            model.eval()

        except Exception:
            pass

        del model

    cuda_cleanup(label)


def assert_vram_for_new_model(
    label,
    max_allocated_gb=2.5,
):

    cuda_cleanup(
        f"{label} — barrier"
    )

    alloc = (
        torch.cuda.memory_allocated()
        / GB
    )

    if alloc > max_allocated_gb:

        print(
            f"WARNING: {alloc:.3f} GB remains allocated "
            "before a new 7B load. "
            "Checking for stale references is recommended."
        )


def cpu_metric_copy(m):

    out = {}

    for k, v in m.items():

        if isinstance(v, np.ndarray):

            out[k] = v.copy()

        elif isinstance(v, np.generic):

            out[k] = v.item()

        else:

            out[k] = v

    return out


# =============================================================================
# 3. LOCKED DRA / SEASC
# =============================================================================

def dra_envelope(L):

    t = float(L)

    x = (
        ZIRVE
        * math.exp(-SONUM * t)
        * (1.0 + SONUM * t)
        + TABAN
    )

    return x / (
        ZIRVE + TABAN
    )


ENVELOPE = [
    dra_envelope(L)
    for L in range(N_LAYERS)
]


RELATIVE_DOSE = [
    IVME * x
    for x in ENVELOPE
]


print(
    "\nLOCKED SEASC 3.0 — "
    "NO BRAKE | NO GATE | NO RADAR | "
    "NO GRAM-SCHMIDT | NO ANTI-INTERFERENCE"
)


for L in range(N_LAYERS):

    print(
        f"L{L:02d} | "
        f"E={ENVELOPE[L]:.6f} | "
        f"physical relative dose="
        f"{100*RELATIVE_DOSE[L]:.3f}%"
    )


# =============================================================================
# 4. COMPASS CORPORA
# =============================================================================

A_POSITIVE = [

    "a person standing above a bridge",

    "a person positioned above a tower",

    "an object located above the roof",

    "a bird flying above the mountain",

    "a light appearing above the building",

    "a person looking from above the valley",

    "a structure extending above the wall",

    "a signal coming from above the surface",
]


A_NEGATIVE = [

    "a person standing below a bridge",

    "a person positioned below a tower",

    "an object located below the roof",

    "a bird flying below the mountain",

    "a light appearing below the building",

    "a person looking from below the valley",

    "a structure extending below the wall",

    "a signal coming from below the surface",
]


B_POSITIVE = [

    "an event occurring before a meeting",

    "a signal appearing before an alarm",

    "a person arriving before a ceremony",

    "a light turning on before a sound",

    "a message arriving before a response",

    "a vehicle stopping before a departure",

    "a door opening before an announcement",

    "a bell ringing before a performance",
]


B_NEGATIVE = [

    "an event occurring after a meeting",

    "a signal appearing after an alarm",

    "a person arriving after a ceremony",

    "a light turning on after a sound",

    "a message arriving after a response",

    "a vehicle stopping after a departure",

    "a door opening after an announcement",

    "a bell ringing after a performance",
]


# =============================================================================
# 5. DATA
# =============================================================================

SPATIAL_A = [

    "red apple",
    "blue cup",
    "small lamp",
    "wooden box",

    "green pear",
    "silver key",
    "yellow ball",
    "black book",
]


SPATIAL_B = [

    "white plate",
    "glass bottle",
    "brown chair",
    "metal tray",

    "purple vase",
    "orange cone",
    "grey stone",
    "blue notebook",
]


SURFACES = [

    "table",
    "desk",
    "shelf",
    "platform",
    "counter",
]


SPATIAL_RELATIONS = [

    "above",
    "below",
]


def make_spatial_prompt(
    a,
    b,
    surface,
    relation,
):

    return (
        f"A {b} is on the {surface}. "
        f"The {a} is positioned {relation} the {b}. "
        f"Describe their relative spatial positions precisely."
    )


TEMPORAL_EVENTS_A = [

    "red light turns on",
    "small bell rings",
    "door opens",
    "vehicle stops",

    "message appears",
    "camera flashes",
    "alarm sounds",
    "screen changes",
]


TEMPORAL_EVENTS_B = [

    "meeting begins",
    "music starts",
    "announcement plays",
    "train departs",

    "reply arrives",
    "recording starts",
    "worker enters",
    "timer ends",
]


TEMPORAL_CONTEXTS = [

    "during the recorded sequence",

    "in the observed timeline",

    "during the experiment",

    "in the event log",

    "during the demonstration",
]


TEMPORAL_RELATIONS = [

    "before",
    "after",
]


def make_temporal_prompt(
    a,
    b,
    context,
    relation,
):

    return (
        f"{context.capitalize()}, "
        f"the {a} {relation} the {b}. "
        f"Describe the relative temporal order "
        f"of the two events precisely."
    )


all_A_prompts = [

    make_spatial_prompt(
        a,
        b,
        s,
        r,
    )

    for a in SPATIAL_A

    for b in SPATIAL_B

    for s in SURFACES

    for r in SPATIAL_RELATIONS
]


all_B_prompts = [

    make_temporal_prompt(
        a,
        b,
        c,
        r,
    )

    for a in TEMPORAL_EVENTS_A

    for b in TEMPORAL_EVENTS_B

    for c in TEMPORAL_CONTEXTS

    for r in TEMPORAL_RELATIONS
]


random.Random(
    SEED
).shuffle(
    all_A_prompts
)


random.Random(
    SEED + 1000
).shuffle(
    all_B_prompts
)


def split_prompts(items):

    n = (
        N_TRAIN
        + N_VALIDATION
        + N_FINAL
    )

    if len(items) < n:

        raise RuntimeError(
            "Insufficient prompt corpus."
        )

    return (

        items[
            :N_TRAIN
        ],

        items[
            N_TRAIN:
            N_TRAIN + N_VALIDATION
        ],

        items[
            N_TRAIN + N_VALIDATION:
            n
        ],
    )


(
    A_TRAIN,
    A_VALIDATION,
    A_FINAL,
) = split_prompts(
    all_A_prompts
)


(
    B_TRAIN,
    B_VALIDATION,
    B_FINAL,
) = split_prompts(
    all_B_prompts
)


print(
    f"\nA TRAIN={len(A_TRAIN)} "
    f"VAL={len(A_VALIDATION)} "
    f"FINAL={len(A_FINAL)}"
)


print(
    f"B TRAIN={len(B_TRAIN)} "
    f"VAL={len(B_VALIDATION)} "
    f"FINAL={len(B_FINAL)}"
)


# =============================================================================
# 6. TOKENIZER / MODEL
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


def encode_chat(prompt):

    text = tokenizer.apply_chat_template(

        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },

            {
                "role": "user",
                "content": prompt,
            },
        ],

        tokenize=False,

        add_generation_prompt=True,
    )


    enc = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=MAX_TRAIN_TOKENS,
    )


    return {
        k: v.to(DEVICE)
        for k, v in enc.items()
    }


def load_base_model():

    assert_vram_for_new_model(
        "before 7B load"
    )


    model = AutoModelForCausalLM.from_pretrained(

        MODEL_ID,

        dtype=torch.bfloat16,

        device_map={"": 0},

        attn_implementation="sdpa",

        trust_remote_code=True,
    )


    model.config.use_cache = False


    memory_report(
        "7B loaded"
    )


    return model


def resolve_layers(model):

    candidates = [

        lambda:
            model.model.layers,

        lambda:
            model.model.model.layers,

        lambda:
            model.base_model.model.model.layers,

        lambda:
            model.base_model.model.model.model.layers,

        lambda:
            model.model.model.model.layers,
    ]


    for getter in candidates:

        try:

            layers = getter()

            if len(layers) >= N_LAYERS:

                return layers

        except Exception:

            pass


    raise RuntimeError(
        "Decoder layers could not be resolved."
    )


# =============================================================================
# 7. MATH
# =============================================================================

def safe_cos(a, b):

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )


    na = np.linalg.norm(a)

    nb = np.linalg.norm(b)


    if (
        na < 1e-12
        or nb < 1e-12
    ):

        return 0.0


    return float(

        np.clip(

            np.dot(a, b)
            / (na * nb),

            -1,
            1,
        )
    )


def vector_projection_coefficient(
    d,
    t,
):

    d = np.asarray(
        d,
        dtype=np.float64,
    )

    t = np.asarray(
        t,
        dtype=np.float64,
    )


    den = float(
        np.dot(t, t)
    )


    if den < 1e-12:

        return 0.0


    return float(
        np.dot(d, t) / den
    )


def two_vector_decomposition(
    d,
    a,
    b,
):

    d = np.asarray(
        d,
        dtype=np.float64,
    )

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )


    G = np.array(

        [
            [
                np.dot(a, a),
                np.dot(a, b),
            ],

            [
                np.dot(b, a),
                np.dot(b, b),
            ],
        ]
    )


    rhs = np.array(

        [
            np.dot(a, d),

            np.dot(b, d),
        ]
    )


    coeff = (
        np.linalg.pinv(G)
        @ rhs
    )


    recon = (
        coeff[0] * a
        + coeff[1] * b
    )


    residual = (
        d - recon
    )


    dn = max(
        np.linalg.norm(d),
        1e-12,
    )


    return {

        "coef_A":
            float(coeff[0]),

        "coef_B":
            float(coeff[1]),

        "residual_fraction":
            float(
                np.linalg.norm(residual)
                / dn
            ),

        "reconstruction_fraction":
            float(
                np.linalg.norm(recon)
                / dn
            ),
    }


# =============================================================================
# 8. COMPASS / MOTOR / CAPTURE
# =============================================================================

HIDDEN_SIZE = None


@torch.inference_mode()
def extract_compass(
    model,
    positive,
    negative,
):

    was_training = model.training

    model.eval()


    pos_acc = [

        torch.zeros(
            HIDDEN_SIZE,
            device=DEVICE,
            dtype=torch.float32,
        )

        for _ in range(N_LAYERS)
    ]


    neg_acc = [

        torch.zeros(
            HIDDEN_SIZE,
            device=DEVICE,
            dtype=torch.float32,
        )

        for _ in range(N_LAYERS)
    ]


    for texts, accum in [

        (
            positive,
            pos_acc,
        ),

        (
            negative,
            neg_acc,
        ),
    ]:

        for text in texts:

            enc = tokenizer(

                text,

                return_tensors="pt",

                add_special_tokens=True,
            )


            enc = {

                k: v.to(DEVICE)

                for k, v in enc.items()
            }


            out = model(

                **enc,

                output_hidden_states=True,

                use_cache=False,

                return_dict=True,
            )


            p = (
                int(
                    enc[
                        "attention_mask"
                    ][0].sum().item()
                )
                - 1
            )


            for L in range(N_LAYERS):

                accum[L].add_(

                    out.hidden_states[
                        L + 1
                    ][
                        0,
                        p,
                        :
                    ].float()
                )


            del out, enc


    compass = []

    raw = []


    for L in range(N_LAYERS):

        diff = (

            pos_acc[L]
            / len(positive)

            -

            neg_acc[L]
            / len(negative)
        )


        n = diff.norm().clamp_min(
            1e-12
        )


        compass.append(

            (diff / n)
            .detach()
            .float()
            .clone()
        )


        raw.append(
            float(n.item())
        )


    del pos_acc, neg_acc


    if was_training:

        model.train()


    return compass, raw


def seasc_tensor(
    hidden,
    compass_vector,
    rho,
):

    norm0 = (
        hidden
        .float()
        .norm(
            p=2,
            dim=-1,
            keepdim=True,
        )
    )


    A = compass_vector.to(

        hidden.device,

        dtype=torch.float32,
    )


    delta = (

        float(rho)

        * norm0

        * A.view(
            1,
            1,
            -1,
        )
    )


    return (

        hidden.float()
        + delta

    ).to(
        hidden.dtype
    )


@torch.inference_mode()
def capture_natural(
    model,
    prompt,
):

    was_training = model.training

    model.eval()


    enc = encode_chat(
        prompt
    )


    out = model(

        **enc,

        output_hidden_states=True,

        use_cache=False,

        return_dict=True,
    )


    p = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )


    states = [

        out.hidden_states[
            L + 1
        ][
            0,
            p,
            :
        ]
        .detach()
        .float()
        .cpu()
        .clone()

        for L in range(N_LAYERS)
    ]


    del out, enc


    if was_training:

        model.train()


    return states


@torch.inference_mode()
def capture_seasc(
    model,
    prompt,
    compass,
):

    was_training = model.training

    model.eval()


    layers = resolve_layers(
        model
    )


    enc = encode_chat(
        prompt
    )


    p = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )


    captured = [
        None
    ] * N_LAYERS


    handles = []


    def make_hook(L):

        def hook(
            module,
            args,
            output,
        ):

            hidden = (
                output[0]
                if isinstance(
                    output,
                    tuple,
                )
                else output
            )


            steered = seasc_tensor(

                hidden,

                compass[L],

                RELATIVE_DOSE[L],
            )


            captured[L] = (

                steered[
                    0,
                    p,
                    :
                ]
                .detach()
                .float()
                .cpu()
                .clone()
            )


            if isinstance(
                output,
                tuple,
            ):

                return (
                    steered,
                ) + output[1:]


            return steered


        return hook


    for L in range(N_LAYERS):

        handles.append(

            layers[L]
            .register_forward_hook(
                make_hook(L)
            )
        )


    try:

        _ = model(

            **enc,

            use_cache=False,

            return_dict=True,
        )


    finally:

        for h in handles:

            h.remove()


    del enc


    if was_training:

        model.train()


    if any(
        x is None
        for x in captured
    ):

        raise RuntimeError(
            "SEASC capture incomplete."
        )


    return captured


@torch.inference_mode()
def capture_model_off(
    model,
    prompt,
):

    was_training = model.training

    model.eval()


    enc = encode_chat(
        prompt
    )


    out = model(

        **enc,

        output_hidden_states=True,

        use_cache=False,

        return_dict=True,
    )


    p = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )


    states = np.stack(

        [

            out.hidden_states[
                L + 1
            ][
                0,
                p,
                :
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .copy()

            for L in range(N_LAYERS)
        ],

        axis=0,
    )


    del out, enc


    if was_training:

        model.train()


    return states


def capture_prompt_set(
    model,
    prompts,
):

    return np.stack(

        [

            capture_model_off(
                model,
                p,
            )

            for p in prompts
        ],

        axis=0,
    )


# =============================================================================
# 9. IMMUTABLE BASE REFERENCES
# =============================================================================

print(
    "\n[1] Loading immutable BASE"
)


base = load_base_model()

base.eval()


for p in base.parameters():

    p.requires_grad_(False)


HIDDEN_SIZE = int(
    base.config.hidden_size
)


print(
    "Hidden size :",
    HIDDEN_SIZE,
)


print(
    "Layers      :",
    len(
        resolve_layers(base)
    ),
)


print(
    "\n[2] Extracting A0/B0"
)


(
    A0_COMPASS,
    A0_RAW,
) = extract_compass(

    base,

    A_POSITIVE,

    A_NEGATIVE,
)


(
    B0_COMPASS,
    B0_RAW,
) = extract_compass(

    base,

    B_POSITIVE,

    B_NEGATIVE,
)


A0_COMPASS_CPU = [

    x.detach()
    .float()
    .cpu()
    .clone()

    for x in A0_COMPASS
]


B0_COMPASS_CPU = [

    x.detach()
    .float()
    .cpu()
    .clone()

    for x in B0_COMPASS
]


del A0_COMPASS

del B0_COMPASS


A0_COMPASS = (
    A0_COMPASS_CPU
)

B0_COMPASS = (
    B0_COMPASS_CPU
)


INITIAL_AB_ROWS = []

initial_ab_cos = []


print(
    "\nINITIAL A/B COMPASS GEOMETRY"
)


for L in range(N_LAYERS):

    A = (
        A0_COMPASS[L]
        .numpy()
    )

    B = (
        B0_COMPASS[L]
        .numpy()
    )


    c = safe_cos(
        A,
        B,
    )


    initial_ab_cos.append(
        c
    )


    INITIAL_AB_ROWS.append({

        "layer":
            L,

        "A_raw_norm":
            A0_RAW[L],

        "B_raw_norm":
            B0_RAW[L],

        "cos_A_B":
            c,
    })


    print(

        f"L{L:02d} | "

        f"|Araw|="
        f"{A0_RAW[L]:9.5f} | "

        f"|Braw|="
        f"{B0_RAW[L]:9.5f} | "

        f"cos(A,B)="
        f"{c:+.7f}"
    )


initial_ab_cos = np.asarray(

    initial_ab_cos,

    dtype=np.float64,
)


def build_reference(
    model,
    prompts,
    compass,
    label,
):

    natural = []

    target = []


    print(
        f"\nBuilding {label}..."
    )


    for i, prompt in enumerate(
        prompts,
        1,
    ):

        natural.append(

            capture_natural(
                model,
                prompt,
            )
        )


        target.append(

            capture_seasc(
                model,
                prompt,
                compass,
            )
        )


        print(
            f"{label} "
            f"{i:02d}/"
            f"{len(prompts)}"
        )


    return natural, target


(
    A_VAL_BASE,
    A_VAL_T0,
) = build_reference(

    base,

    A_VALIDATION,

    A0_COMPASS,

    "A_VALIDATION",
)


(
    B_VAL_BASE,
    B_VAL_T0,
) = build_reference(

    base,

    B_VALIDATION,

    B0_COMPASS,

    "B_VALIDATION",
)


(
    A_FINAL_BASE,
    A_FINAL_T0,
) = build_reference(

    base,

    A_FINAL,

    A0_COMPASS,

    "A_FINAL_LOCKED",
)


(
    B_FINAL_BASE,
    B_FINAL_T0,
) = build_reference(

    base,

    B_FINAL,

    B0_COMPASS,

    "B_FINAL_LOCKED",
)


@torch.inference_mode()
def base_topk(
    model,
    prompt,
):

    enc = encode_chat(
        prompt
    )


    out = model(

        **enc,

        use_cache=False,

        return_dict=True,
    )


    p = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )


    logits = (

        out.logits[
            0,
            p,
            :
        ]
        .detach()
        .float()
    )


    values, indices = torch.topk(

        logits,

        k=min(
            PRESERVE_TOP_K,
            logits.numel(),
        ),
    )


    result = {

        "indices":
            indices
            .cpu()
            .to(
                torch.int32
            ),

        "logits":
            values
            .cpu()
            .to(
                torch.float16
            ),
    }


    del out, enc


    return result


print(
    "\n[3] Preservation cache"
)


ALL_TRAIN_UNIQUE = list(

    dict.fromkeys(

        A_TRAIN
        + B_TRAIN
    )
)


PRESERVE_CACHE = {}


for i, prompt in enumerate(
    ALL_TRAIN_UNIQUE,
    1,
):

    PRESERVE_CACHE[
        prompt
    ] = base_topk(
        base,
        prompt,
    )


    if (
        i % 8 == 0
        or i
        == len(
            ALL_TRAIN_UNIQUE
        )
    ):

        print(
            f"{i}/"
            f"{len(ALL_TRAIN_UNIQUE)}"
        )


@torch.inference_mode()
def generate_response(
    model,
    prompt,
):

    was_training = model.training

    model.eval()


    enc = encode_chat(
        prompt
    )


    plen = int(
        enc[
            "input_ids"
        ].shape[1]
    )


    ids = model.generate(

        **enc,

        max_new_tokens=
            MAX_BEHAVIOR_NEW_TOKENS,

        do_sample=False,

        repetition_penalty=
            REPETITION_PENALTY,

        use_cache=True,

        pad_token_id=
            tokenizer.pad_token_id,

        eos_token_id=
            tokenizer.eos_token_id,
    )


    text = tokenizer.decode(

        ids[
            0,
            plen:
        ],

        skip_special_tokens=True,

    ).strip()


    del enc, ids


    if was_training:

        model.train()


    return text


print(
    "\n[4] BASE final behavior"
)


BASE_A_FINAL_RESPONSES = []


for i, p in enumerate(
    A_FINAL,
    1,
):

    BASE_A_FINAL_RESPONSES.append(

        generate_response(
            base,
            p,
        )
    )


    print(
        f"BASE A "
        f"{i:02d}/"
        f"{N_FINAL}"
    )


BASE_B_FINAL_RESPONSES = []


for i, p in enumerate(
    B_FINAL,
    1,
):

    BASE_B_FINAL_RESPONSES.append(

        generate_response(
            base,
            p,
        )
    )


    print(
        f"BASE B "
        f"{i:02d}/"
        f"{N_FINAL}"
    )


def states_to_np(
    samples,
):

    return np.stack(

        [

            np.stack(

                [
                    x.numpy()
                    for x in sample
                ],

                axis=0,
            )

            for sample in samples
        ],

        axis=0,
    )


A_VAL_BASE_NP = states_to_np(
    A_VAL_BASE
)

A_VAL_T0_NP = states_to_np(
    A_VAL_T0
)

B_VAL_BASE_NP = states_to_np(
    B_VAL_BASE
)

B_VAL_T0_NP = states_to_np(
    B_VAL_T0
)

A_FINAL_BASE_NP = states_to_np(
    A_FINAL_BASE
)

A_FINAL_T0_NP = states_to_np(
    A_FINAL_T0
)

B_FINAL_BASE_NP = states_to_np(
    B_FINAL_BASE
)

B_FINAL_T0_NP = states_to_np(
    B_FINAL_T0
)


del A_VAL_BASE

del A_VAL_T0

del B_VAL_BASE

del B_VAL_T0

del A_FINAL_BASE

del A_FINAL_T0

del B_FINAL_BASE

del B_FINAL_T0

del base


cuda_cleanup(
    "immutable BASE released"
)


# =============================================================================
# 10. LOGS
# =============================================================================

TRAIN_LOG = []

EVAL_LOG = []

LAYER_LOG = []

COMPASS_LOG = []

INTERFERENCE_LOG = []

FINAL_LOG = []

MERGE_LOG = []

BEHAVIOR_LOG = []

JPEG_FILES = []


# TEST146
XRAY_GLOBAL_LOG = []

XRAY_MODULE_LOG = []

XRAY_LAYER_LOG = []

XRAY_STEP_LOG = []


# =============================================================================
# 11. MODEL / ADAPTER HELPERS
# =============================================================================

def lora_config():

    return LoraConfig(

        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        bias="none",

        task_type="CAUSAL_LM",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )


def create_lora_student():

    model = load_base_model()


    if hasattr(
        model,
        "gradient_checkpointing_disable",
    ):

        model.gradient_checkpointing_disable()


    return get_peft_model(

        model,

        lora_config(),
    )


def attach_new_lora(
    model,
    adapter_name="default",
):

    return get_peft_model(

        model,

        lora_config(),

        adapter_name=adapter_name,
    )


def load_merged_from_base(
    adapter_dir,
    label,
):

    print(
        f"\nMERGE: {label}"
    )


    parent = load_base_model()


    peft_model = PeftModel.from_pretrained(

        parent,

        adapter_dir,

        is_trainable=False,
    )


    peft_model.eval()


    merged = (
        peft_model
        .merge_and_unload()
    )


    merged.eval()

    merged.config.use_cache = False


    remaining = [

        n

        for n, _ in merged.named_modules()

        if "lora_" in n.lower()
    ]


    if remaining:

        raise RuntimeError(
            f"LoRA remains after {label}."
        )


    del peft_model, parent

    gc.collect()

    torch.cuda.empty_cache()


    return merged


def load_sequential_merged(
    first_adapter,
    second_adapter,
    label,
):

    print(
        f"\nSEQUENTIAL RECONSTRUCTION: {label}"
    )


    model = load_base_model()


    p1 = PeftModel.from_pretrained(

        model,

        first_adapter,

        is_trainable=False,
    )


    p1.eval()


    model = (
        p1
        .merge_and_unload()
    )


    model.eval()


    del p1

    gc.collect()

    torch.cuda.empty_cache()


    p2 = PeftModel.from_pretrained(

        model,

        second_adapter,

        is_trainable=False,
    )


    p2.eval()


    model = (
        p2
        .merge_and_unload()
    )


    model.eval()

    model.config.use_cache = False


    del p2

    gc.collect()

    torch.cuda.empty_cache()


    remaining = [

        n

        for n, _ in model.named_modules()

        if "lora_" in n.lower()
    ]


    if remaining:

        raise RuntimeError(
            f"LoRA remains after {label}."
        )


    return model


# =============================================================================
# 12. METRICS
# =============================================================================

def evaluate_against_reference(
    states,
    base_np,
    target_np,
):

    n = states.shape[0]


    distance = np.zeros(
        N_LAYERS
    )

    progress = np.zeros(
        N_LAYERS
    )

    alignment = np.zeros(
        N_LAYERS
    )

    correction = np.zeros(
        N_LAYERS
    )

    projection = np.zeros(
        N_LAYERS
    )


    for i in range(n):

        for L in range(N_LAYERS):

            B = base_np[
                i,
                L
            ]

            T = target_np[
                i,
                L
            ]

            S = states[
                i,
                L
            ]


            t = T - B

            d = S - B


            D0 = max(
                np.linalg.norm(t),
                1e-12,
            )


            D = np.linalg.norm(
                S - T
            )


            distance[L] += D


            progress[L] += (
                1.0
                - D / D0
            )


            alignment[L] += safe_cos(
                d,
                t,
            )


            correction[L] += (
                D / D0
            )


            projection[L] += (
                vector_projection_coefficient(
                    d,
                    t,
                )
            )


    for x in [

        distance,
        progress,
        alignment,
        correction,
        projection,

    ]:

        x /= n


    return {

        "distance":
            float(
                distance.mean()
            ),

        "progress":
            float(
                progress.mean()
            ),

        "alignment":
            float(
                alignment.mean()
            ),

        "correction":
            float(
                correction.mean()
            ),

        "projection":
            float(
                projection.mean()
            ),

        "layer_distance":
            distance,

        "layer_progress":
            progress,

        "layer_alignment":
            alignment,

        "layer_correction":
            correction,

        "layer_projection":
            projection,
    }


def compass_space_autopsy(
    states,
    base_np,
    compass_A,
    compass_B,
):

    Acoef = np.zeros(
        N_LAYERS
    )

    Bcoef = np.zeros(
        N_LAYERS
    )

    residual = np.zeros(
        N_LAYERS
    )

    Acos = np.zeros(
        N_LAYERS
    )

    Bcos = np.zeros(
        N_LAYERS
    )


    for i in range(
        states.shape[0]
    ):

        for L in range(
            N_LAYERS
        ):

            d = (
                states[i, L]
                - base_np[i, L]
            )


            A = (
                compass_A[L]
                .numpy()
            )


            B = (
                compass_B[L]
                .numpy()
            )


            dec = (
                two_vector_decomposition(
                    d,
                    A,
                    B,
                )
            )


            Acoef[L] += (
                dec["coef_A"]
            )

            Bcoef[L] += (
                dec["coef_B"]
            )

            residual[L] += (
                dec[
                    "residual_fraction"
                ]
            )

            Acos[L] += safe_cos(
                d,
                A,
            )

            Bcos[L] += safe_cos(
                d,
                B,
            )


    n = states.shape[0]


    Acoef /= n

    Bcoef /= n

    residual /= n

    Acos /= n

    Bcos /= n


    return {

        "coef_A":
            float(
                Acoef.mean()
            ),

        "coef_B":
            float(
                Bcoef.mean()
            ),

        "residual_fraction":
            float(
                residual.mean()
            ),

        "cos_A":
            float(
                Acos.mean()
            ),

        "cos_B":
            float(
                Bcos.mean()
            ),

        "layer_coef_A":
            Acoef,

        "layer_coef_B":
            Bcoef,

        "layer_residual":
            residual,

        "layer_cos_A":
            Acos,

        "layer_cos_B":
            Bcos,
    }


# =============================================================================
# 13. ORIGINAL AUTOGRAD / TEACHER / TRAINING FORWARD
# =============================================================================

def autograd_xray(
    student,
    prompt,
    label,
):

    print(
        f"\nAUTOGRAD X-RAY — {label}"
    )


    student.train()


    params = [

        p

        for p in student.parameters()

        if p.requires_grad
    ]


    for p in params:

        p.grad = None


    enc = encode_chat(
        prompt
    )


    pos = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )


    out = student(

        **enc,

        output_hidden_states=True,

        use_cache=False,

        return_dict=True,
    )


    terms = []


    for L in [
        0,
        5,
        10,
        15,
        19,
    ]:

        h = out.hidden_states[
            L + 1
        ][
            0,
            pos,
            :
        ]


        print(

            f"L{L:02d} "
            f"requires_grad="
            f"{h.requires_grad} "

            f"grad_fn="
            f"{type(h.grad_fn).__name__ if h.grad_fn else 'NONE'}"
        )


        if not h.requires_grad:

            raise RuntimeError(
                "Autograd X-ray failed."
            )


        terms.append(

            h.float()
            .square()
            .mean()
        )


    loss = (
        torch.stack(
            terms
        )
        .mean()
    )


    loss.backward()


    sq = 0.0

    count = 0


    for p in params:

        if p.grad is not None:

            g = (
                p.grad
                .detach()
                .float()
            )


            if not torch.isfinite(
                g
            ).all():

                raise RuntimeError(
                    "Non-finite gradient."
                )


            sq += float(

                g.square()
                .sum()
                .cpu()
            )


            count += 1


    norm = math.sqrt(
        sq
    )


    if (
        count == 0
        or norm <= 0
    ):

        raise RuntimeError(
            "No LoRA gradient."
        )


    print(
        "Global trainable grad norm:",
        f"{norm:.9f}",
    )


    print(
        "AUTOGRAD X-RAY: PASS"
    )


    for p in params:

        p.grad = None


    del (
        out,
        enc,
        terms,
        loss,
    )


    cuda_cleanup(
        "after autograd x-ray"
    )


@torch.inference_mode()
def current_teacher_target(
    student,
    prompt,
    current_compass,
):

    was_training = (
        student.training
    )


    student.eval()


    layers = resolve_layers(
        student
    )


    enc = encode_chat(
        prompt
    )


    pos = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )


    captured = [
        None
    ] * N_LAYERS


    handles = []


    def make_hook(L):

        def hook(
            module,
            args,
            output,
        ):

            hidden = (
                output[0]
                if isinstance(
                    output,
                    tuple,
                )
                else output
            )


            steered = seasc_tensor(

                hidden,

                current_compass[L],

                RELATIVE_DOSE[L],
            )


            captured[L] = (

                steered[
                    0,
                    pos,
                    :
                ]
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float16,
                )
                .clone()
            )


            if isinstance(
                output,
                tuple,
            ):

                return (
                    steered,
                ) + output[1:]


            return steered


        return hook


    for L in range(
        N_LAYERS
    ):

        handles.append(

            layers[L]
            .register_forward_hook(
                make_hook(L)
            )
        )


    try:

        _ = student(

            **enc,

            use_cache=False,

            return_dict=True,
        )


    finally:

        for h in handles:

            h.remove()


    del enc


    if was_training:

        student.train()


    if any(
        x is None
        for x in captured
    ):

        raise RuntimeError(
            "Teacher target incomplete."
        )


    return torch.stack(
        captured,
        dim=0,
    )


def training_forward(
    student,
    prompt,
    target_cpu,
):

    student.train()


    enc = encode_chat(
        prompt
    )


    pos = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )


    out = student(

        **enc,

        output_hidden_states=True,

        use_cache=False,

        return_dict=True,
    )


    geo = []


    for L in range(
        N_LAYERS
    ):

        h = (

            out.hidden_states[
                L + 1
            ][
                0,
                pos,
                :
            ]
            .float()
        )


        target = (
            target_cpu[L]
            .to(
                DEVICE,
                dtype=torch.float32,
            )
        )


        scale2 = (

            target
            .square()
            .sum()
            .detach()
            .clamp_min(
                1e-6
            )
        )


        geo.append(

            (h - target)
            .square()
            .sum()

            / scale2
        )


    geometry_loss = (

        torch.stack(
            geo
        )
        .mean()
    )


    logits = (

        out.logits[
            0,
            pos,
            :
        ]
        .float()
    )


    cache = (
        PRESERVE_CACHE[
            prompt
        ]
    )


    indices = (

        cache["indices"]
        .to(
            DEVICE,
            dtype=torch.long,
        )
    )


    base_logits = (

        cache["logits"]
        .to(
            DEVICE,
            dtype=torch.float32,
        )
    )


    current_logits = (
        logits[
            indices
        ]
    )


    preserve_loss = F.kl_div(

        F.log_softmax(
            current_logits,
            dim=-1,
        ),

        F.softmax(
            base_logits,
            dim=-1,
        ),

        reduction="sum",
    )


    total = (

        LAMBDA_GEOMETRY
        * geometry_loss

        +

        LAMBDA_PRESERVE
        * preserve_loss
    )


    del out, enc


    return (
        total,
        geometry_loss.detach(),
        preserve_loss.detach(),
    )


# =============================================================================
# 14. TEST 146 — PARAMETER-SPACE X-RAY
# =============================================================================

def xray_trainable_params(
    student,
):

    rows = []


    for name, p in (
        student.named_parameters()
    ):

        if not p.requires_grad:
            continue


        if (
            "lora_"
            not in name.lower()
        ):

            continue


        match = re.search(
            r"\.layers\.(\d+)\.",
            name,
        )


        if match is None:
            continue


        layer = int(
            match.group(1)
        )


        if not (
            0
            <= layer
            < N_LAYERS
        ):

            continue


        module = None


        for m in XRAY_MODULES:

            if f".{m}." in name:

                module = m

                break


        if module is None:
            continue


        if "lora_A" in name:

            factor = "A"

        elif "lora_B" in name:

            factor = "B"

        else:

            factor = "OTHER"


        rows.append({

            "name":
                name,

            "param":
                p,

            "layer":
                layer,

            "module":
                module,

            "factor":
                factor,
        })


    if not rows:

        raise RuntimeError(
            "TEST146: no trainable LoRA parameters found."
        )


    return rows


def xray_target_loss(
    student,
    prompt,
    target_np,
):

    # IMPORTANT:
    # eval() changes module mode only.
    # It DOES NOT disable autograd.
    # This makes the diagnostic deterministic while preserving gradients.

    was_training = (
        student.training
    )


    student.eval()


    enc = encode_chat(
        prompt
    )


    pos = (
        int(
            enc[
                "attention_mask"
            ][0].sum().item()
        )
        - 1
    )


    out = student(

        **enc,

        output_hidden_states=True,

        use_cache=False,

        return_dict=True,
    )


    terms = []


    for L in range(
        N_LAYERS
    ):

        h = (

            out.hidden_states[
                L + 1
            ][
                0,
                pos,
                :
            ]
            .float()
        )


        target = torch.as_tensor(

            target_np[L],

            device=DEVICE,

            dtype=torch.float32,
        )


        scale2 = (

            target
            .square()
            .sum()
            .detach()
            .clamp_min(
                1e-6
            )
        )


        terms.append(

            (h - target)
            .square()
            .sum()

            / scale2
        )


    loss = (

        torch.stack(
            terms
        )
        .mean()
    )


    # Restore training mode before returning.
    # The graph remains valid.

    if was_training:

        student.train()


    del out, enc


    return loss


def xray_probe_gradient(
    student,
    prompts,
    targets_np,
    param_rows,
):

    params = [

        r["param"]

        for r in param_rows
    ]


    accum = [

        torch.zeros_like(

            p,

            device="cpu",

            dtype=torch.float32,
        )

        for p in params
    ]


    losses = []


    for i, prompt in enumerate(
        prompts
    ):

        loss = xray_target_loss(

            student,

            prompt,

            targets_np[i],
        )


        grads = torch.autograd.grad(

            loss,

            params,

            retain_graph=False,

            create_graph=False,

            allow_unused=True,
        )


        losses.append(

            float(
                loss
                .detach()
                .cpu()
            )
        )


        for j, g in enumerate(
            grads
        ):

            if g is not None:

                accum[j].add_(

                    g
                    .detach()
                    .float()
                    .cpu()
                )


        del grads, loss


    inv = (
        1.0
        / max(
            len(prompts),
            1,
        )
    )


    for g in accum:

        g.mul_(inv)


    return (
        accum,
        float(
            np.mean(losses)
        ),
    )


def xray_vector_stats(
    a_list,
    b_list,
):

    dot = 0.0

    a2 = 0.0

    b2 = 0.0


    for a, b in zip(
        a_list,
        b_list,
    ):

        aa = a.double()

        bb = b.double()


        dot += float(
            (aa * bb).sum()
        )


        a2 += float(
            (aa * aa).sum()
        )


        b2 += float(
            (bb * bb).sum()
        )


    na = math.sqrt(
        max(
            a2,
            0.0,
        )
    )


    nb = math.sqrt(
        max(
            b2,
            0.0,
        )
    )


    cos = (

        dot

        / max(
            na * nb,
            XRAY_EPS,
        )
    )


    cos = float(
        np.clip(
            cos,
            -1.0,
            1.0,
        )
    )


    return {

        "dot":
            dot,

        "norm_a":
            na,

        "norm_b":
            nb,

        "cosine":
            cos,

        # For a single vector projection,
        # this equals |cosine|.
        "projection_fraction":
            abs(cos),
    }


def xray_group_indices(
    param_rows,
):

    modules = {}

    layers = {}


    for i, row in enumerate(
        param_rows
    ):

        module_key = (

            row["layer"],

            row["module"],
        )


        modules.setdefault(
            module_key,
            [],
        ).append(i)


        layers.setdefault(
            row["layer"],
            [],
        ).append(i)


    return (
        modules,
        layers,
    )


def xray_subset(
    vectors,
    indices,
):

    return [

        vectors[i]

        for i in indices
    ]


def xray_snapshot_params(
    param_rows,
):

    return [

        row["param"]
        .detach()
        .float()
        .cpu()
        .clone()

        for row in param_rows
    ]


def xray_actual_delta(
    param_rows,
    before,
):

    return [

        row["param"]
        .detach()
        .float()
        .cpu()

        - old

        for row, old in zip(
            param_rows,
            before,
        )
    ]


def xray_run_pre_step(
    student,
    branch,
    step,
    retain_name,
    retain_prompts,
    retain_targets,
    new_name,
    new_prompts,
    new_targets,
):

    param_rows = (
        xray_trainable_params(
            student
        )
    )


    retain_probe_prompts = (

        retain_prompts[
            :XRAY_N_PROBES
        ]
    )


    new_probe_prompts = (

        new_prompts[
            :XRAY_N_PROBES
        ]
    )


    retain_probe_targets = (

        retain_targets[
            :XRAY_N_PROBES
        ]
    )


    new_probe_targets = (

        new_targets[
            :XRAY_N_PROBES
        ]
    )


    g_retain, retain_loss = (

        xray_probe_gradient(

            student,

            retain_probe_prompts,

            retain_probe_targets,

            param_rows,
        )
    )


    g_new, new_loss = (

        xray_probe_gradient(

            student,

            new_probe_prompts,

            new_probe_targets,

            param_rows,
        )
    )


    global_stats = (

        xray_vector_stats(
            g_retain,
            g_new,
        )
    )


    (
        module_groups,
        layer_groups,

    ) = xray_group_indices(
        param_rows
    )


    print(
        "\n"
        + "=" * 150
    )


    print(

        f"TEST146 PRE-STEP X-RAY | "
        f"{branch} | "
        f"step={step} | "

        f"{retain_name} RETAIN "
        f"vs "
        f"{new_name} WRITE"
    )


    print(
        "=" * 150
    )


    print(

        "GLOBAL gradient geometry | "

        f"cos="
        f"{global_stats['cosine']:+.8f} | "

        f"dot="
        f"{global_stats['dot']:+.8e} | "

        f"|gR|="
        f"{global_stats['norm_a']:.8e} | "

        f"|gN|="
        f"{global_stats['norm_b']:.8e}"
    )


    XRAY_GLOBAL_LOG.append({

        "branch":
            branch,

        "step":
            step,

        "retain":
            retain_name,

        "new":
            new_name,

        "retain_probe_loss":
            retain_loss,

        "new_probe_loss":
            new_loss,

        "grad_dot":
            global_stats[
                "dot"
            ],

        "grad_cosine":
            global_stats[
                "cosine"
            ],

        "retain_grad_norm":
            global_stats[
                "norm_a"
            ],

        "new_grad_norm":
            global_stats[
                "norm_b"
            ],

        "projection_fraction":
            global_stats[
                "projection_fraction"
            ],

        "raw_gradient_conflict":
            global_stats[
                "dot"
            ] < 0.0,
    })


    for (
        L,
        module,
    ), indices in (
        module_groups.items()
    ):

        s = xray_vector_stats(

            xray_subset(
                g_retain,
                indices,
            ),

            xray_subset(
                g_new,
                indices,
            ),
        )


        XRAY_MODULE_LOG.append({

            "branch":
                branch,

            "step":
                step,

            "layer":
                L,

            "module":
                module,

            "grad_dot":
                s["dot"],

            "grad_cosine":
                s["cosine"],

            "retain_grad_norm":
                s["norm_a"],

            "new_grad_norm":
                s["norm_b"],

            "projection_fraction":
                s[
                    "projection_fraction"
                ],

            "raw_gradient_conflict":
                s["dot"] < 0.0,
        })


    for L, indices in (
        layer_groups.items()
    ):

        s = xray_vector_stats(

            xray_subset(
                g_retain,
                indices,
            ),

            xray_subset(
                g_new,
                indices,
            ),
        )


        XRAY_LAYER_LOG.append({

            "branch":
                branch,

            "step":
                step,

            "layer":
                L,

            "grad_dot":
                s["dot"],

            "grad_cosine":
                s["cosine"],

            "retain_grad_norm":
                s["norm_a"],

            "new_grad_norm":
                s["norm_b"],

            "projection_fraction":
                s[
                    "projection_fraction"
                ],

            "raw_gradient_conflict":
                s["dot"] < 0.0,
        })


    before = (
        xray_snapshot_params(
            param_rows
        )
    )


    return {

        "param_rows":
            param_rows,

        "g_retain":
            g_retain,

        "g_new":
            g_new,

        "before":
            before,

        "retain_loss_before":
            retain_loss,

        "retain_prompts":
            retain_probe_prompts,

        "retain_targets":
            retain_probe_targets,

        "retain_name":
            retain_name,

        "new_name":
            new_name,

        "module_groups":
            module_groups,

        "layer_groups":
            layer_groups,
    }


@torch.inference_mode()
def xray_probe_loss_only(
    student,
    prompts,
    targets,
):

    vals = []


    was_training = (
        student.training
    )


    student.eval()


    for i, prompt in enumerate(
        prompts
    ):

        enc = encode_chat(
            prompt
        )


        pos = (
            int(
                enc[
                    "attention_mask"
                ][0].sum().item()
            )
            - 1
        )


        out = student(

            **enc,

            output_hidden_states=True,

            use_cache=False,

            return_dict=True,
        )


        terms = []


        for L in range(
            N_LAYERS
        ):

            h = (

                out.hidden_states[
                    L + 1
                ][
                    0,
                    pos,
                    :
                ]
                .float()
            )


            target = torch.as_tensor(

                targets[
                    i,
                    L
                ],

                device=DEVICE,

                dtype=torch.float32,
            )


            scale2 = (

                target
                .square()
                .sum()
                .clamp_min(
                    1e-6
                )
            )


            terms.append(

                (h - target)
                .square()
                .sum()

                / scale2
            )


        vals.append(

            float(

                torch.stack(
                    terms
                )
                .mean()
                .cpu()
            )
        )


        del out, enc


    if was_training:

        student.train()


    return float(
        np.mean(vals)
    )


def xray_run_post_step(
    student,
    branch,
    step,
    ctx,
):

    param_rows = (
        ctx["param_rows"]
    )


    delta = (
        xray_actual_delta(

            param_rows,

            ctx["before"],
        )
    )


    step_stats = (

        xray_vector_stats(

            ctx["g_retain"],

            delta,
        )
    )


    actual_after = (

        xray_probe_loss_only(

            student,

            ctx[
                "retain_prompts"
            ],

            ctx[
                "retain_targets"
            ],
        )
    )


    actual_delta_loss = (

        actual_after

        - ctx[
            "retain_loss_before"
        ]
    )


    predicted_delta_loss = (

        step_stats[
            "dot"
        ]
    )


    predicted_sign = int(
        np.sign(
            predicted_delta_loss
        )
    )


    actual_sign = int(
        np.sign(
            actual_delta_loss
        )
    )


    sign_match = (
        predicted_sign
        == actual_sign
    )


    print(

        "ACTUAL ADAMW STEP | "

        f"gR·Δtheta="
        f"{predicted_delta_loss:+.8e} | "

        f"cos(gR,Δtheta)="
        f"{step_stats['cosine']:+.8f} | "

        f"actual ΔLretain="
        f"{actual_delta_loss:+.8e} | "

        f"sign_match="
        f"{sign_match}"
    )


    XRAY_STEP_LOG.append({

        "branch":
            branch,

        "step":
            step,

        "retain":
            ctx["retain_name"],

        "new":
            ctx["new_name"],

        "predicted_delta_retain_loss":
            predicted_delta_loss,

        "actual_delta_retain_loss":
            actual_delta_loss,

        "retain_step_cosine":
            step_stats[
                "cosine"
            ],

        "retain_grad_norm":
            step_stats[
                "norm_a"
            ],

        "actual_step_norm":
            step_stats[
                "norm_b"
            ],

        "predicted_harm":
            predicted_delta_loss
            > 0.0,

        "actual_harm":
            actual_delta_loss
            > 0.0,

        "sign_match":
            bool(
                sign_match
            ),
    })


    for (
        L,
        module,
    ), indices in (
        ctx[
            "module_groups"
        ].items()
    ):

        s = xray_vector_stats(

            xray_subset(

                ctx[
                    "g_retain"
                ],

                indices,
            ),

            xray_subset(

                delta,

                indices,
            ),
        )


        for row in reversed(
            XRAY_MODULE_LOG
        ):

            if (

                row["branch"]
                == branch

                and row["step"]
                == step

                and row["layer"]
                == L

                and row["module"]
                == module
            ):

                row[
                    "retain_step_dot"
                ] = s["dot"]


                row[
                    "retain_step_cosine"
                ] = s["cosine"]


                row[
                    "actual_step_norm"
                ] = s["norm_b"]


                row[
                    "predicted_harm"
                ] = (
                    s["dot"]
                    > 0.0
                )


                break


    for L, indices in (
        ctx[
            "layer_groups"
        ].items()
    ):

        s = xray_vector_stats(

            xray_subset(

                ctx[
                    "g_retain"
                ],

                indices,
            ),

            xray_subset(

                delta,

                indices,
            ),
        )


        for row in reversed(
            XRAY_LAYER_LOG
        ):

            if (

                row["branch"]
                == branch

                and row["step"]
                == step

                and row["layer"]
                == L
            ):

                row[
                    "retain_step_dot"
                ] = s["dot"]


                row[
                    "retain_step_cosine"
                ] = s["cosine"]


                row[
                    "actual_step_norm"
                ] = s["norm_b"]


                row[
                    "predicted_harm"
                ] = (
                    s["dot"]
                    > 0.0
                )


                break


    del delta


    # Release the large CPU X-ray snapshots immediately.
    ctx.clear()


    gc.collect()


# =============================================================================
# 15. CONSOLE
# =============================================================================

def print_layer_table(
    label,
    primary,
    secondary=None,
    secondary_baseline=None,
):

    print(
        "\n"
        + "-" * 145
    )


    print(
        "LAYER X-RAY —",
        label,
    )


    print(
        "-" * 145
    )


    if secondary is None:

        for L in range(
            N_LAYERS
        ):

            print(

                f"L{L:02d} | "

                f"P="
                f"{100*primary['layer_progress'][L]:+10.4f}% | "

                f"align="
                f"{primary['layer_alignment'][L]:+.7f} | "

                f"proj="
                f"{primary['layer_projection'][L]:+.6f}"
            )


    else:

        for L in range(
            N_LAYERS
        ):

            if (
                secondary_baseline
                is None
            ):

                delta = float(
                    "nan"
                )

            else:

                delta = 100 * (

                    secondary[
                        "layer_progress"
                    ][L]

                    -

                    secondary_baseline[
                        "layer_progress"
                    ][L]
                )


            print(

                f"L{L:02d} | "

                f"primary="
                f"{100*primary['layer_progress'][L]:+10.4f}% | "

                f"align="
                f"{primary['layer_alignment'][L]:+.7f} | "

                f"secondary="
                f"{100*secondary['layer_progress'][L]:+10.4f}% | "

                f"Δ="
                f"{delta:+10.4f} pp"
            )


def print_cross_summary(
    stage,
    step,
    primary_name,
    primary,
    secondary_name,
    secondary,
    baseline=None,
    autopsy=None,
):

    print(
        "\n"
        + "=" * 145
    )


    print(
        f"{stage} — STEP {step}"
    )


    print(
        "=" * 145
    )


    print(

        f"{primary_name}: "

        f"progress="
        f"{100*primary['progress']:+.6f}% | "

        f"distance="
        f"{primary['distance']:.8f} | "

        f"align="
        f"{primary['alignment']:+.8f} | "

        f"projection="
        f"{primary['projection']:+.8f}"
    )


    print(

        f"{secondary_name}: "

        f"progress="
        f"{100*secondary['progress']:+.6f}% | "

        f"align="
        f"{secondary['alignment']:+.8f} | "

        f"projection="
        f"{secondary['projection']:+.8f}"
    )


    if baseline is not None:

        delta = (

            secondary[
                "progress"
            ]

            -

            baseline[
                "progress"
            ]
        )


        print(
            f"Interference Δ="
            f"{100*delta:+.6f} pp"
        )


        ld = (

            secondary[
                "layer_progress"
            ]

            -

            baseline[
                "layer_progress"
            ]
        )


        print(

            f"Worst layer "
            f"L{int(np.argmin(ld)):02d}: "
            f"{100*ld.min():+.5f} pp | "

            f"Best layer "
            f"L{int(np.argmax(ld)):02d}: "
            f"{100*ld.max():+.5f} pp"
        )


    if autopsy:

        print(

            f"Autopsy A="
            f"{autopsy['coef_A']:+.8f} | "

            f"B="
            f"{autopsy['coef_B']:+.8f} | "

            f"residual="
            f"{100*autopsy['residual_fraction']:.4f}%"
        )


# =============================================================================
# END OF TEST 146 — PART 1/2
#
# IMPORTANT:
# PART 2 STARTS WITH:
#
#     # 16. CRYSTALLIZATION ENGINE
#
# Paste PART 2 DIRECTLY BELOW THIS LINE.
# =============================================================================
# =============================================================================
# 16. CRYSTALLIZATION ENGINE
# =============================================================================

def crystallize(
    student,branch_name,crystal_name,train_prompts,
    positive,negative,immutable_compass,
    primary_validation_prompts,primary_val_base,primary_val_target,
    secondary_name,secondary_validation_prompts,
    secondary_val_base,secondary_val_target,
    secondary_baseline_metrics=None,seed_offset=0,
    enable_parameter_xray=False,
):
    print("\n"+"#"*150)
    print(f"CRYSTALLIZATION — {branch_name} — {crystal_name}")
    print("#"*150)

    reseed(seed_offset)

    current_compass,current_raw=extract_compass(
        student,positive,negative
    )

    params=[p for p in student.parameters() if p.requires_grad]

    if not params:
        raise RuntimeError("No trainable parameters.")

    optimizer=torch.optim.AdamW(
        params,
        lr=LEARNING_RATE,
        betas=(0.9,0.95),
        weight_decay=0.01
    )

    autograd_xray(
        student,
        train_prompts[0],
        branch_name
    )

    best_dir=os.path.join(
        ADAPTER_DIR,
        f"{branch_name}_BEST"
    )

    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)

    os.makedirs(best_dir,exist_ok=True)

    primary_states=capture_prompt_set(
        student,
        primary_validation_prompts
    )

    secondary_states=capture_prompt_set(
        student,
        secondary_validation_prompts
    )

    primary0=evaluate_against_reference(
        primary_states,
        primary_val_base,
        primary_val_target
    )

    secondary0=evaluate_against_reference(
        secondary_states,
        secondary_val_base,
        secondary_val_target
    )

    if secondary_baseline_metrics is None:
        secondary_baseline_metrics=cpu_metric_copy(
            secondary0
        )

    autopsy0=compass_space_autopsy(
        primary_states,
        primary_val_base,
        A0_COMPASS,
        B0_COMPASS
    )

    print_cross_summary(
        branch_name,
        0,
        crystal_name,
        primary0,
        secondary_name,
        secondary0,
        secondary_baseline_metrics,
        autopsy0
    )

    if PRINT_FULL_LAYER_TABLE_EVERY_EVAL:
        print_layer_table(
            branch_name,
            primary0,
            secondary0,
            secondary_baseline_metrics
        )

    # -------------------------------------------------------------------------
    # TEST 146 — STEP 0 PARAMETER-SPACE X-RAY
    # Diagnostic only. No gradient is modified.
    # -------------------------------------------------------------------------

    if enable_parameter_xray:
        run_parameter_xray(
            student=student,
            branch=branch_name,
            step=0,
            preserve_name=secondary_name,
            acquire_name=crystal_name,
            preserve_prompts=secondary_validation_prompts,
            preserve_targets_np=secondary_val_target,
            acquire_prompts=primary_validation_prompts,
            acquire_targets_np=primary_val_target,
        )

    best_progress=primary0["progress"]
    best_distance=primary0["distance"]
    best_alignment=primary0["alignment"]

    best_step=0
    no_improvement=0
    stop_reason=None

    student.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    del primary_states,secondary_states

    start=time.time()
    update=0
    epoch=0

    local_rng=random.Random(
        SEED+seed_offset
    )

    while update<HARD_MAX_UPDATES:

        epoch+=1

        order=list(
            range(len(train_prompts))
        )

        local_rng.shuffle(order)

        for idx in order:

            if update>=HARD_MAX_UPDATES:
                break

            update+=1

            prompt=train_prompts[idx]

            target=current_teacher_target(
                student,
                prompt,
                current_compass
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            total_loss,geo_loss,preserve_loss=training_forward(
                student,
                prompt,
                target
            )

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss "
                    f"{branch_name} "
                    f"step {update}"
                )

            total_loss.backward()

            grad_norm=torch.nn.utils.clip_grad_norm_(
                params,
                GRAD_CLIP
            )

            grad_value=float(
                torch.as_tensor(
                    grad_norm
                ).detach().float().cpu()
            )

            if not math.isfinite(
                grad_value
            ):
                raise RuntimeError(
                    "Non-finite gradient."
                )

            optimizer.step()

            previous_compass=[
                x.detach().clone()
                for x in current_compass
            ]

            new_compass,new_raw=extract_compass(
                student,
                positive,
                negative
            )

            ci=[]
            cp=[]

            for L in range(N_LAYERS):

                init=immutable_compass[L].numpy()

                prev=(
                    previous_compass[L]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                new=(
                    new_compass[L]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                c0=safe_cos(
                    init,
                    new
                )

                cprev=safe_cos(
                    prev,
                    new
                )

                ci.append(c0)
                cp.append(cprev)

                COMPASS_LOG.append({
                    "branch":branch_name,
                    "crystal":crystal_name,
                    "step":update,
                    "layer":L,
                    "raw_norm":new_raw[L],
                    "cos_immutable_initial":c0,
                    "cos_previous":cprev,
                })

            del previous_compass,current_compass

            current_compass=[
                x.detach().clone().to(DEVICE)
                for x in new_compass
            ]

            del new_compass

            mean_ci=float(
                np.mean(ci)
            )

            mean_cp=float(
                np.mean(cp)
            )

            TRAIN_LOG.append({
                "branch":branch_name,
                "crystal":crystal_name,
                "step":update,
                "epoch":epoch,
                "loss_total":
                    float(total_loss.detach().cpu()),
                "loss_geometry":
                    float(geo_loss.cpu()),
                "loss_preserve":
                    float(preserve_loss.cpu()),
                "grad_norm":
                    grad_value,
                "mean_compass_cos_initial":
                    mean_ci,
                "mean_compass_cos_previous":
                    mean_cp,
                "elapsed_seconds":
                    time.time()-start,
            })

            print(
                f"{branch_name:12s} | "
                f"step={update:04d} | "
                f"loss="
                f"{float(total_loss.detach().cpu()):.7f} | "
                f"geo="
                f"{float(geo_loss.cpu()):.7f} | "
                f"presK="
                f"{float(preserve_loss.cpu()):.7f} | "
                f"grad={grad_value:.6f} | "
                f"cos(init,t)={mean_ci:+.6f} | "
                f"cos(prev,t)={mean_cp:+.6f}"
            )

            del target,total_loss,geo_loss,preserve_loss

            if update%EVAL_EVERY==0:

                primary_states=capture_prompt_set(
                    student,
                    primary_validation_prompts
                )

                secondary_states=capture_prompt_set(
                    student,
                    secondary_validation_prompts
                )

                primary=evaluate_against_reference(
                    primary_states,
                    primary_val_base,
                    primary_val_target
                )

                secondary=evaluate_against_reference(
                    secondary_states,
                    secondary_val_base,
                    secondary_val_target
                )

                autopsy=compass_space_autopsy(
                    primary_states,
                    primary_val_base,
                    A0_COMPASS,
                    B0_COMPASS
                )

                print_cross_summary(
                    branch_name,
                    update,
                    crystal_name,
                    primary,
                    secondary_name,
                    secondary,
                    secondary_baseline_metrics,
                    autopsy
                )

                if PRINT_FULL_LAYER_TABLE_EVERY_EVAL:
                    print_layer_table(
                        branch_name,
                        primary,
                        secondary,
                        secondary_baseline_metrics
                    )

                EVAL_LOG.append({
                    "branch":branch_name,
                    "crystal":crystal_name,
                    "step":update,

                    "primary_progress":
                        primary["progress"],

                    "primary_distance":
                        primary["distance"],

                    "primary_alignment":
                        primary["alignment"],

                    "primary_projection":
                        primary["projection"],

                    "secondary_progress":
                        secondary["progress"],

                    "secondary_alignment":
                        secondary["alignment"],

                    "secondary_projection":
                        secondary["projection"],

                    "secondary_interference":
                        secondary["progress"]-
                        secondary_baseline_metrics[
                            "progress"
                        ],

                    "autopsy_coef_A":
                        autopsy["coef_A"],

                    "autopsy_coef_B":
                        autopsy["coef_B"],

                    "autopsy_residual":
                        autopsy[
                            "residual_fraction"
                        ],
                })

                for L in range(N_LAYERS):

                    LAYER_LOG.append({
                        "branch":branch_name,
                        "crystal":crystal_name,
                        "step":update,
                        "layer":L,

                        "primary_progress":
                            primary[
                                "layer_progress"
                            ][L],

                        "primary_alignment":
                            primary[
                                "layer_alignment"
                            ][L],

                        "primary_projection":
                            primary[
                                "layer_projection"
                            ][L],

                        "secondary_progress":
                            secondary[
                                "layer_progress"
                            ][L],

                        "secondary_alignment":
                            secondary[
                                "layer_alignment"
                            ][L],

                        "secondary_projection":
                            secondary[
                                "layer_projection"
                            ][L],

                        "secondary_interference":
                            secondary[
                                "layer_progress"
                            ][L]-
                            secondary_baseline_metrics[
                                "layer_progress"
                            ][L],

                        "autopsy_coef_A":
                            autopsy[
                                "layer_coef_A"
                            ][L],

                        "autopsy_coef_B":
                            autopsy[
                                "layer_coef_B"
                            ][L],

                        "autopsy_residual":
                            autopsy[
                                "layer_residual"
                            ][L],
                    })

                # -------------------------------------------------------------
                # TEST 146 PARAMETER-SPACE X-RAY
                # Every 10 steps.
                # No backward modification.
                # No projection.
                # No optimizer intervention.
                # -------------------------------------------------------------

                if (
                    enable_parameter_xray
                    and
                    update%XRAY_EVERY==0
                ):

                    run_parameter_xray(
                        student=student,
                        branch=branch_name,
                        step=update,
                        preserve_name=secondary_name,
                        acquire_name=crystal_name,
                        preserve_prompts=
                            secondary_validation_prompts,
                        preserve_targets_np=
                            secondary_val_target,
                        acquire_prompts=
                            primary_validation_prompts,
                        acquire_targets_np=
                            primary_val_target,
                    )

                current_progress=primary[
                    "progress"
                ]

                current_distance=primary[
                    "distance"
                ]

                current_alignment=primary[
                    "alignment"
                ]

                progress_gain=(
                    current_progress-
                    best_progress
                )

                relative_distance_gain=(
                    (
                        best_distance-
                        current_distance
                    )/
                    max(
                        best_distance,
                        1e-12
                    )
                )

                alignment_gain=(
                    current_alignment-
                    best_alignment
                )

                new_best=(
                    current_progress>
                    best_progress
                    and
                    current_alignment>=
                    ALIGNMENT_FLOOR
                )

                if new_best:

                    best_progress=current_progress
                    best_distance=current_distance
                    best_alignment=current_alignment
                    best_step=update

                    if os.path.exists(
                        best_dir
                    ):
                        shutil.rmtree(
                            best_dir
                        )

                    student.save_pretrained(
                        best_dir
                    )

                    tokenizer.save_pretrained(
                        best_dir
                    )

                    no_improvement=0

                    print(
                        f">>> NEW BEST | "
                        f"{branch_name} | "
                        f"step={best_step} | "
                        f"progress="
                        f"{100*best_progress:+.6f}%"
                    )

                else:

                    meaningful_change=(
                        abs(progress_gain)>=
                        MIN_PROGRESS_IMPROVEMENT
                        or
                        abs(
                            relative_distance_gain
                        )>=
                        MIN_DISTANCE_IMPROVEMENT
                        or
                        abs(alignment_gain)>=
                        MIN_ALIGNMENT_IMPROVEMENT
                    )

                    no_improvement+=1

                    print(
                        f"No new best | "
                        f"patience="
                        f"{no_improvement}/"
                        f"{PATIENCE_EVALS} | "
                        f"ΔP="
                        f"{100*progress_gain:+.5f} pp | "
                        f"movement="
                        f"{meaningful_change}"
                    )

                overshoot=(
                    update>=
                    MIN_UPDATES_BEFORE_STOP
                    and
                    best_progress-
                    current_progress>=
                    OVERSHOOT_PROGRESS_DROP
                )

                del primary_states
                del secondary_states
                del autopsy

                if overshoot:

                    stop_reason=(
                        "OVERSHOOT_DETECTED"
                    )

                    print(
                        f">>> OVERSHOOT | "
                        f"best="
                        f"{100*best_progress:+.6f}% | "
                        f"current="
                        f"{100*current_progress:+.6f}% | "
                        f"rollback BEST"
                    )

                    break

                if (
                    update>=
                    MIN_UPDATES_BEFORE_STOP
                    and
                    no_improvement>=
                    PATIENCE_EVALS
                ):

                    stop_reason=(
                        "HELDOUT_EQUILIBRIUM"
                    )

                    print(
                        ">>> HELDOUT EQUILIBRIUM"
                    )

                    break

            if update%25==0:
                cuda_cleanup(
                    f"{branch_name} "
                    f"step {update}"
                )

        if stop_reason is not None:
            break

    if stop_reason is None:
        stop_reason=(
            "HARD_MAX_UPDATES"
        )

    print(
        f"\n{branch_name} STOP | "
        f"last={update} | "
        f"best={best_step} | "
        f"reason={stop_reason} | "
        f"progress="
        f"{100*best_progress:+.6f}%"
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    del optimizer
    del params
    del current_compass
    del current_raw

    cuda_cleanup(
        f"{branch_name} "
        "training objects released"
    )

    return {
        "best_dir":
            best_dir,

        "best_step":
            best_step,

        "last_step":
            int(update),

        "stop_reason":
            stop_reason,

        "best_progress":
            float(best_progress),

        "best_distance":
            float(best_distance),

        "best_alignment":
            float(best_alignment),

        "initial_primary":
            cpu_metric_copy(primary0),

        "initial_secondary":
            cpu_metric_copy(secondary0),

        "secondary_baseline":
            cpu_metric_copy(
                secondary_baseline_metrics
            ),
    }


# =============================================================================
# 17. MERGE DRIFT — ONE MODEL ONLY
# =============================================================================

def merge_drift_check_base(
    adapter_dir,
    prompts,
    label
):
    print(
        f"\nMERGE DRIFT — {label}"
    )

    parent=load_base_model()

    adapter=PeftModel.from_pretrained(
        parent,
        adapter_dir,
        is_trainable=False
    )

    adapter.eval()

    before=capture_prompt_set(
        adapter,
        prompts
    )

    merged=adapter.merge_and_unload()

    merged.eval()
    merged.config.use_cache=False

    after=capture_prompt_set(
        merged,
        prompts
    )

    drift=float(
        np.linalg.norm(
            after-before
        )/
        max(
            np.linalg.norm(before),
            1e-12
        )
    )

    status=(
        "TIGHT_EQUIVALENCE"
        if drift<=0.005
        else
        "SMALL_BF16_MERGE_DRIFT"
        if drift<=0.015
        else
        "MERGE_DRIFT_REQUIRES_REVIEW"
    )

    MERGE_LOG.append({
        "stage":label,
        "relative_hidden_drift":
            drift,
        "status":status
    })

    print(
        f"{label} "
        f"drift={drift:.10e} | "
        f"{status}"
    )

    del before,after,adapter,parent

    gc.collect()
    torch.cuda.empty_cache()

    return merged,drift,status


def merge_drift_check_sequential(
    first_adapter,
    second_adapter,
    prompts,
    label
):
    print(
        f"\nMERGE DRIFT — {label}"
    )

    parent=load_base_model()

    p1=PeftModel.from_pretrained(
        parent,
        first_adapter,
        is_trainable=False
    )

    p1.eval()

    parent=p1.merge_and_unload()
    parent.eval()

    del p1

    gc.collect()
    torch.cuda.empty_cache()

    p2=PeftModel.from_pretrained(
        parent,
        second_adapter,
        is_trainable=False
    )

    p2.eval()

    before=capture_prompt_set(
        p2,
        prompts
    )

    merged=p2.merge_and_unload()

    merged.eval()
    merged.config.use_cache=False

    after=capture_prompt_set(
        merged,
        prompts
    )

    drift=float(
        np.linalg.norm(
            after-before
        )/
        max(
            np.linalg.norm(before),
            1e-12
        )
    )

    status=(
        "TIGHT_EQUIVALENCE"
        if drift<=0.005
        else
        "SMALL_BF16_MERGE_DRIFT"
        if drift<=0.015
        else
        "MERGE_DRIFT_REQUIRES_REVIEW"
    )

    MERGE_LOG.append({
        "stage":label,
        "relative_hidden_drift":
            drift,
        "status":status
    })

    print(
        f"{label} "
        f"drift={drift:.10e} | "
        f"{status}"
    )

    del before,after,p2,parent

    gc.collect()
    torch.cuda.empty_cache()

    return merged,drift,status


# =============================================================================
# 18. STAGE 1 — BASE -> A
# =============================================================================

print("\n"+"="*150)
print("STAGE 1/4 — BASE -> A")
print("="*150)

assert_vram_for_new_model(
    "STAGE 1"
)

student_A=create_lora_student()

A_RESULT=crystallize(
    student_A,
    "A_ONLY",
    "A",
    A_TRAIN,

    A_POSITIVE,
    A_NEGATIVE,
    A0_COMPASS,

    A_VALIDATION,
    A_VAL_BASE_NP,
    A_VAL_T0_NP,

    "B",
    B_VALIDATION,
    B_VAL_BASE_NP,
    B_VAL_T0_NP,

    None,
    10,

    enable_parameter_xray=False,
)

del student_A

cuda_cleanup(
    "STAGE 1 student deleted"
)

A_MERGED,\
A_MERGE_DRIFT,\
A_MERGE_STATUS=merge_drift_check_base(
    A_RESULT["best_dir"],
    A_VALIDATION,
    "A_ONLY"
)

A_ONLY_A_VAL_STATES=capture_prompt_set(
    A_MERGED,
    A_VALIDATION
)

A_ONLY_B_VAL_STATES=capture_prompt_set(
    A_MERGED,
    B_VALIDATION
)

A_ONLY_A_VAL=evaluate_against_reference(
    A_ONLY_A_VAL_STATES,
    A_VAL_BASE_NP,
    A_VAL_T0_NP
)

A_ONLY_B_VAL=evaluate_against_reference(
    A_ONLY_B_VAL_STATES,
    B_VAL_BASE_NP,
    B_VAL_T0_NP
)

A_ONLY_A_VAL=cpu_metric_copy(
    A_ONLY_A_VAL
)

A_ONLY_B_VAL=cpu_metric_copy(
    A_ONLY_B_VAL
)

print(
    f"A-only merged | "
    f"A="
    f"{100*A_ONLY_A_VAL['progress']:+.6f}% | "
    f"B="
    f"{100*A_ONLY_B_VAL['progress']:+.6f}%"
)

del A_ONLY_A_VAL_STATES
del A_ONLY_B_VAL_STATES


# =============================================================================
# 19. STAGE 2 — A -> B
# =============================================================================

print("\n"+"="*150)
print("STAGE 2/4 — A -> B")
print("="*150)

student_AB=attach_new_lora(
    A_MERGED,
    "default"
)

A_MERGED=None

AB_RESULT=crystallize(
    student_AB,
    "A_TO_B",
    "B",
    B_TRAIN,

    B_POSITIVE,
    B_NEGATIVE,
    B0_COMPASS,

    B_VALIDATION,
    B_VAL_BASE_NP,
    B_VAL_T0_NP,

    "A",
    A_VALIDATION,
    A_VAL_BASE_NP,
    A_VAL_T0_NP,

    A_ONLY_A_VAL,
    20,

    enable_parameter_xray=True,
)

del student_AB

cuda_cleanup(
    "STAGE 2 student deleted"
)

AB_MERGED,\
AB_MERGE_DRIFT,\
AB_MERGE_STATUS=merge_drift_check_sequential(
    A_RESULT["best_dir"],
    AB_RESULT["best_dir"],
    B_VALIDATION,
    "A_TO_B"
)

AB_VAL_A_STATES=capture_prompt_set(
    AB_MERGED,
    A_VALIDATION
)

AB_VAL_B_STATES=capture_prompt_set(
    AB_MERGED,
    B_VALIDATION
)

AB_VAL_A=evaluate_against_reference(
    AB_VAL_A_STATES,
    A_VAL_BASE_NP,
    A_VAL_T0_NP
)

AB_VAL_B=evaluate_against_reference(
    AB_VAL_B_STATES,
    B_VAL_BASE_NP,
    B_VAL_T0_NP
)

print(
    f"A->B merged validation | "
    f"A={100*AB_VAL_A['progress']:+.6f}% | "
    f"B={100*AB_VAL_B['progress']:+.6f}%"
)

del AB_VAL_A_STATES
del AB_VAL_B_STATES

del AB_MERGED
AB_MERGED=None

cuda_cleanup(
    "STAGE 2 AB model fully destroyed"
)

assert_vram_for_new_model(
    "PRE-STAGE-3 HARD BARRIER"
)


# =============================================================================
# 20. STAGE 3 — BASE -> B
# =============================================================================

print("\n"+"="*150)
print("STAGE 3/4 — BASE -> B")
print("="*150)

student_B=create_lora_student()

B_RESULT=crystallize(
    student_B,
    "B_ONLY",
    "B",
    B_TRAIN,

    B_POSITIVE,
    B_NEGATIVE,
    B0_COMPASS,

    B_VALIDATION,
    B_VAL_BASE_NP,
    B_VAL_T0_NP,

    "A",
    A_VALIDATION,
    A_VAL_BASE_NP,
    A_VAL_T0_NP,

    None,
    30,

    enable_parameter_xray=False,
)

del student_B

cuda_cleanup(
    "STAGE 3 student deleted"
)

B_MERGED,\
B_MERGE_DRIFT,\
B_MERGE_STATUS=merge_drift_check_base(
    B_RESULT["best_dir"],
    B_VALIDATION,
    "B_ONLY"
)

B_ONLY_B_VAL_STATES=capture_prompt_set(
    B_MERGED,
    B_VALIDATION
)

B_ONLY_A_VAL_STATES=capture_prompt_set(
    B_MERGED,
    A_VALIDATION
)

B_ONLY_B_VAL=evaluate_against_reference(
    B_ONLY_B_VAL_STATES,
    B_VAL_BASE_NP,
    B_VAL_T0_NP
)

B_ONLY_A_VAL=evaluate_against_reference(
    B_ONLY_A_VAL_STATES,
    A_VAL_BASE_NP,
    A_VAL_T0_NP
)

B_ONLY_B_VAL=cpu_metric_copy(
    B_ONLY_B_VAL
)

B_ONLY_A_VAL=cpu_metric_copy(
    B_ONLY_A_VAL
)

print(
    f"B-only merged | "
    f"B={100*B_ONLY_B_VAL['progress']:+.6f}% | "
    f"A={100*B_ONLY_A_VAL['progress']:+.6f}%"
)

del B_ONLY_B_VAL_STATES
del B_ONLY_A_VAL_STATES


# =============================================================================
# 21. STAGE 4 — B -> A
# =============================================================================

print("\n"+"="*150)
print("STAGE 4/4 — B -> A")
print("="*150)

student_BA=attach_new_lora(
    B_MERGED,
    "default"
)

B_MERGED=None

BA_RESULT=crystallize(
    student_BA,
    "B_TO_A",
    "A",
    A_TRAIN,

    A_POSITIVE,
    A_NEGATIVE,
    A0_COMPASS,

    A_VALIDATION,
    A_VAL_BASE_NP,
    A_VAL_T0_NP,

    "B",
    B_VALIDATION,
    B_VAL_BASE_NP,
    B_VAL_T0_NP,

    B_ONLY_B_VAL,
    40,

    enable_parameter_xray=True,
)

del student_BA

cuda_cleanup(
    "STAGE 4 student deleted"
)

BA_MERGED,\
BA_MERGE_DRIFT,\
BA_MERGE_STATUS=merge_drift_check_sequential(
    B_RESULT["best_dir"],
    BA_RESULT["best_dir"],
    A_VALIDATION,
    "B_TO_A"
)

del BA_MERGED
BA_MERGED=None

cuda_cleanup(
    "STAGE 4 BA model fully destroyed"
)

assert_vram_for_new_model(
    "PRE-FINAL HARD BARRIER"
)


# =============================================================================
# 22. FINAL EXAM — ONE MODEL AT A TIME
# =============================================================================

def final_exam_model(
    model,
    label
):
    print(
        "\n"+"-"*150
    )

    print(
        "FINAL EXAM —",
        label
    )

    print(
        "-"*150
    )

    A_states=capture_prompt_set(
        model,
        A_FINAL
    )

    B_states=capture_prompt_set(
        model,
        B_FINAL
    )

    A_metrics=evaluate_against_reference(
        A_states,
        A_FINAL_BASE_NP,
        A_FINAL_T0_NP
    )

    B_metrics=evaluate_against_reference(
        B_states,
        B_FINAL_BASE_NP,
        B_FINAL_T0_NP
    )

    A_autopsy=compass_space_autopsy(
        A_states,
        A_FINAL_BASE_NP,
        A0_COMPASS,
        B0_COMPASS
    )

    B_autopsy=compass_space_autopsy(
        B_states,
        B_FINAL_BASE_NP,
        A0_COMPASS,
        B0_COMPASS
    )

    print(
        f"A="
        f"{100*A_metrics['progress']:+.6f}% | "
        f"B="
        f"{100*B_metrics['progress']:+.6f}% | "
        f"A-align="
        f"{A_metrics['alignment']:+.7f} | "
        f"B-align="
        f"{B_metrics['alignment']:+.7f}"
    )

    for L in range(N_LAYERS):

        FINAL_LOG.append({
            "state":label,
            "layer":L,

            "A_progress":
                A_metrics[
                    "layer_progress"
                ][L],

            "A_alignment":
                A_metrics[
                    "layer_alignment"
                ][L],

            "A_projection":
                A_metrics[
                    "layer_projection"
                ][L],

            "B_progress":
                B_metrics[
                    "layer_progress"
                ][L],

            "B_alignment":
                B_metrics[
                    "layer_alignment"
                ][L],

            "B_projection":
                B_metrics[
                    "layer_projection"
                ][L],

            "A_autopsy_coef_A":
                A_autopsy[
                    "layer_coef_A"
                ][L],

            "A_autopsy_coef_B":
                A_autopsy[
                    "layer_coef_B"
                ][L],

            "A_autopsy_residual":
                A_autopsy[
                    "layer_residual"
                ][L],

            "B_autopsy_coef_A":
                B_autopsy[
                    "layer_coef_A"
                ][L],

            "B_autopsy_coef_B":
                B_autopsy[
                    "layer_coef_B"
                ][L],

            "B_autopsy_residual":
                B_autopsy[
                    "layer_residual"
                ][L],
        })

    return {
        "A_states":
            A_states.copy(),

        "B_states":
            B_states.copy(),

        "A":
            cpu_metric_copy(
                A_metrics
            ),

        "B":
            cpu_metric_copy(
                B_metrics
            ),

        "A_autopsy":
            cpu_metric_copy(
                A_autopsy
            ),

        "B_autopsy":
            cpu_metric_copy(
                B_autopsy
            ),
    }


def jaccard_text(a,b):

    A=set(
        a.lower().split()
    )

    B=set(
        b.lower().split()
    )

    return (
        1.0
        if not A and not B
        else
        len(A&B)/
        max(
            len(A|B),
            1
        )
    )


def behavioral_final_exam(
    model,
    label
):
    changed_A=0
    changed_B=0

    jA=[]
    jB=[]

    for i,prompt in enumerate(
        A_FINAL
    ):

        response=generate_response(
            model,
            prompt
        )

        base_response=(
            BASE_A_FINAL_RESPONSES[i]
        )

        changed=(
            response.strip()!=
            base_response.strip()
        )

        jac=jaccard_text(
            response,
            base_response
        )

        changed_A+=int(changed)
        jA.append(jac)

        BEHAVIOR_LOG.append({
            "state":label,
            "crystal_set":"A",
            "prompt_index":i,
            "prompt":prompt,
            "base_response":
                base_response,
            "response":
                response,
            "changed":
                changed,
            "lexical_jaccard":
                jac,
        })

    for i,prompt in enumerate(
        B_FINAL
    ):

        response=generate_response(
            model,
            prompt
        )

        base_response=(
            BASE_B_FINAL_RESPONSES[i]
        )

        changed=(
            response.strip()!=
            base_response.strip()
        )

        jac=jaccard_text(
            response,
            base_response
        )

        changed_B+=int(changed)
        jB.append(jac)

        BEHAVIOR_LOG.append({
            "state":label,
            "crystal_set":"B",
            "prompt_index":i,
            "prompt":prompt,
            "base_response":
                base_response,
            "response":
                response,
            "changed":
                changed,
            "lexical_jaccard":
                jac,
        })

    result={
        "A_changed":
            changed_A,

        "B_changed":
            changed_B,

        "A_mean_jaccard":
            float(np.mean(jA)),

        "B_mean_jaccard":
            float(np.mean(jB)),
    }

    print(
        f"{label} behavior | "
        f"A changed="
        f"{changed_A}/{N_FINAL} "
        f"J="
        f"{result['A_mean_jaccard']:.6f} | "
        f"B changed="
        f"{changed_B}/{N_FINAL} "
        f"J="
        f"{result['B_mean_jaccard']:.6f}"
    )

    return result


model=load_merged_from_base(
    A_RESULT["best_dir"],
    "A_ONLY_FINAL"
)

A_ONLY_FINAL=final_exam_model(
    model,
    "A_ONLY"
)

del model

cuda_cleanup(
    "FINAL A-only destroyed"
)


model=load_merged_from_base(
    B_RESULT["best_dir"],
    "B_ONLY_FINAL"
)

B_ONLY_FINAL=final_exam_model(
    model,
    "B_ONLY"
)

del model

cuda_cleanup(
    "FINAL B-only destroyed"
)


model=load_sequential_merged(
    A_RESULT["best_dir"],
    AB_RESULT["best_dir"],
    "AB_FINAL"
)

AB_FINAL=final_exam_model(
    model,
    "A_TO_B"
)

AB_BEHAVIOR=behavioral_final_exam(
    model,
    "A_TO_B"
)

del model

cuda_cleanup(
    "FINAL AB destroyed"
)


model=load_sequential_merged(
    B_RESULT["best_dir"],
    BA_RESULT["best_dir"],
    "BA_FINAL"
)

BA_FINAL=final_exam_model(
    model,
    "B_TO_A"
)

BA_BEHAVIOR=behavioral_final_exam(
    model,
    "B_TO_A"
)

del model

cuda_cleanup(
    "FINAL BA destroyed"
)


# =============================================================================
# 23. INTERFERENCE / ORDER EFFECT
# =============================================================================

I_A_FROM_B=(
    AB_FINAL["A"]["progress"]-
    A_ONLY_FINAL["A"]["progress"]
)

I_B_FROM_A=(
    BA_FINAL["B"]["progress"]-
    B_ONLY_FINAL["B"]["progress"]
)

A_GAIN_IN_BA=(
    BA_FINAL["A"]["progress"]-
    B_ONLY_FINAL["A"]["progress"]
)

B_GAIN_IN_AB=(
    AB_FINAL["B"]["progress"]-
    A_ONLY_FINAL["B"]["progress"]
)

A_LAYER_INTERFERENCE=(
    AB_FINAL["A"]["layer_progress"]-
    A_ONLY_FINAL["A"]["layer_progress"]
)

B_LAYER_INTERFERENCE=(
    BA_FINAL["B"]["layer_progress"]-
    B_ONLY_FINAL["B"]["layer_progress"]
)

AB_ORDER_STACK=np.concatenate(
    [
        AB_FINAL["A_states"],
        AB_FINAL["B_states"]
    ],
    axis=0
)

BA_ORDER_STACK=np.concatenate(
    [
        BA_FINAL["A_states"],
        BA_FINAL["B_states"]
    ],
    axis=0
)

BASE_ORDER_STACK=np.concatenate(
    [
        A_FINAL_BASE_NP,
        B_FINAL_BASE_NP
    ],
    axis=0
)

ORDER_DISTANCE_RAW=float(
    np.linalg.norm(
        AB_ORDER_STACK-
        BA_ORDER_STACK
    )
)

ORDER_DISTANCE_NORMALIZED=float(
    ORDER_DISTANCE_RAW/
    max(
        np.linalg.norm(
            BASE_ORDER_STACK
        ),
        1e-12
    )
)

ORDER_LAYER=np.zeros(
    N_LAYERS
)

for L in range(N_LAYERS):

    ORDER_LAYER[L]=(
        np.linalg.norm(
            AB_ORDER_STACK[:,L,:]-
            BA_ORDER_STACK[:,L,:]
        )/
        max(
            np.linalg.norm(
                BASE_ORDER_STACK[:,L,:]
            ),
            1e-12
        )
    )


print(
    "\n"+"="*165
)

print(
    "FINAL LAYERWISE INTERFERENCE MATRIX"
)

print(
    "="*165
)

print(
    "Layer | A-only P% | "
    "A after B P% | I[A<-B]pp | "
    "B-only P% | B after A P% | "
    "I[B<-A]pp | Order%"
)

for L in range(N_LAYERS):

    print(
        f"L{L:02d} | "
        f"{100*A_ONLY_FINAL['A']['layer_progress'][L]:+10.4f} | "
        f"{100*AB_FINAL['A']['layer_progress'][L]:+12.4f} | "
        f"{100*A_LAYER_INTERFERENCE[L]:+11.4f} | "
        f"{100*B_ONLY_FINAL['B']['layer_progress'][L]:+10.4f} | "
        f"{100*BA_FINAL['B']['layer_progress'][L]:+12.4f} | "
        f"{100*B_LAYER_INTERFERENCE[L]:+11.4f} | "
        f"{100*ORDER_LAYER[L]:10.5f}"
    )

    INTERFERENCE_LOG.append({
        "layer":L,

        "A_only_progress":
            A_ONLY_FINAL[
                "A"
            ][
                "layer_progress"
            ][L],

        "A_after_B_progress":
            AB_FINAL[
                "A"
            ][
                "layer_progress"
            ][L],

        "I_A_from_B":
            A_LAYER_INTERFERENCE[L],

        "B_only_progress":
            B_ONLY_FINAL[
                "B"
            ][
                "layer_progress"
            ][L],

        "B_after_A_progress":
            BA_FINAL[
                "B"
            ][
                "layer_progress"
            ][L],

        "I_B_from_A":
            B_LAYER_INTERFERENCE[L],

        "order_distance_normalized":
            ORDER_LAYER[L],
    })


# =============================================================================
# 24. TEST 146 — GRADIENT / REPRESENTATION CORRELATION
# =============================================================================

XRAY_CORRELATION_LOG=[]


def nearest_layer_interference(
    branch,
    layer
):
    if branch=="A_TO_B":
        return float(
            A_LAYER_INTERFERENCE[layer]
        )

    if branch=="B_TO_A":
        return float(
            B_LAYER_INTERFERENCE[layer]
        )

    return float("nan")


def correlation_report(
    branch
):
    rows=[]

    for row in GRAD_XRAY_LAYER_LOG:

        if row["branch"]!=branch:
            continue

        layer=int(
            row["layer"]
        )

        rep_interference=(
            nearest_layer_interference(
                branch,
                layer
            )
        )

        rows.append({
            "branch":
                branch,

            "step":
                int(row["step"]),

            "layer":
                layer,

            "gradient_cosine":
                float(row["cosine"]),

            "gradient_dot":
                float(row["dot"]),

            "gradient_projection_ratio":
                float(
                    row[
                        "projection_ratio"
                    ]
                ),

            "gradient_conflict":
                bool(
                    row["conflict"]
                ),

            "final_representation_interference":
                rep_interference,
        })

    XRAY_CORRELATION_LOG.extend(
        rows
    )

    if len(rows)<2:

        return {
            "n":len(rows),
            "pearson_cos_vs_final_interference":
                float("nan"),
            "negative_dot_fraction":
                float("nan"),
        }

    x=np.asarray(
        [
            r["gradient_cosine"]
            for r in rows
        ],
        dtype=np.float64
    )

    y=np.asarray(
        [
            r[
                "final_representation_interference"
            ]
            for r in rows
        ],
        dtype=np.float64
    )

    if (
        np.std(x)<1e-12
        or
        np.std(y)<1e-12
    ):
        corr=float("nan")
    else:
        corr=float(
            np.corrcoef(
                x,
                y
            )[0,1]
        )

    neg_fraction=float(
        np.mean(
            [
                r["gradient_dot"]<0
                for r in rows
            ]
        )
    )

    print(
        "\nTEST 146 CORRELATION —",
        branch
    )

    print(
        "N layer-step pairs:",
        len(rows)
    )

    print(
        "Pearson "
        "gradient cosine vs "
        "final representation interference:",
        f"{corr:+.8f}"
    )

    print(
        "Negative gradient-dot fraction:",
        f"{100*neg_fraction:.4f}%"
    )

    print(
        "NOTE: exploratory correlation; "
        "not a causal estimate."
    )

    return {
        "n":
            len(rows),

        "pearson_cos_vs_final_interference":
            corr,

        "negative_dot_fraction":
            neg_fraction,
    }


XRAY_AB_CORRELATION=correlation_report(
    "A_TO_B"
)

XRAY_BA_CORRELATION=correlation_report(
    "B_TO_A"
)


# =============================================================================
# 25. TEST 146 — FORENSIC RANKING
# =============================================================================

def forensic_ranking(
    branch,
    top_n=20
):
    rows=[
        r
        for r in GRAD_XRAY_LAYER_LOG
        if r["branch"]==branch
    ]

    rows=sorted(
        rows,
        key=lambda r:r["cosine"]
    )

    print(
        "\n"+"="*145
    )

    print(
        f"TEST 146 — "
        f"{branch} — "
        f"{top_n} MOST NEGATIVE "
        f"LAYER/STEP GRADIENT RELATIONS"
    )

    print(
        "="*145
    )

    for r in rows[:top_n]:

        layer=int(
            r["layer"]
        )

        rep=nearest_layer_interference(
            branch,
            layer
        )

        print(
            f"step={int(r['step']):04d} | "
            f"L{layer:02d} | "
            f"cos={r['cosine']:+.8f} | "
            f"dot={r['dot']:+.8e} | "
            f"|cos|={r['projection_ratio']:.8f} | "
            f"final_repr_I="
            f"{100*rep:+.6f} pp"
        )


forensic_ranking(
    "A_TO_B"
)

forensic_ranking(
    "B_TO_A"
)


# =============================================================================
# 26. TEST 146 — MODULE CONFLICT SUMMARY
# =============================================================================

MODULE_CONFLICT_SUMMARY=[]

for branch in [
    "A_TO_B",
    "B_TO_A"
]:

    print(
        "\n"+"="*120
    )

    print(
        "TEST 146 MODULE SUMMARY —",
        branch
    )

    print(
        "="*120
    )

    for module_name in XRAY_MODULES:

        rows=[
            r
            for r in GRAD_XRAY_LOG
            if
            r["branch"]==branch
            and
            r["module"]==module_name
        ]

        if not rows:
            continue

        mean_cos=float(
            np.mean(
                [
                    r["cosine"]
                    for r in rows
                ]
            )
        )

        negative_fraction=float(
            np.mean(
                [
                    r["dot"]<0
                    for r in rows
                ]
            )
        )

        mean_projection=float(
            np.mean(
                [
                    r[
                        "projection_ratio"
                    ]
                    for r in rows
                ]
            )
        )

        MODULE_CONFLICT_SUMMARY.append({
            "branch":
                branch,

            "module":
                module_name,

            "mean_cosine":
                mean_cos,

            "negative_dot_fraction":
                negative_fraction,

            "mean_projection_ratio":
                mean_projection,

            "n":
                len(rows),
        })

        print(
            f"{module_name:8s} | "
            f"mean cos="
            f"{mean_cos:+.8f} | "
            f"negative dot="
            f"{100*negative_fraction:7.3f}% | "
            f"mean |cos|="
            f"{mean_projection:.8f} | "
            f"N={len(rows)}"
        )


# =============================================================================
# 27. SAVE CSV
# =============================================================================

tables={
    "TEST146_training_log.csv":
        TRAIN_LOG,

    "TEST146_validation_log.csv":
        EVAL_LOG,

    "TEST146_layerwise_validation.csv":
        LAYER_LOG,

    "TEST146_compass_evolution.csv":
        COMPASS_LOG,

    "TEST146_initial_AB_geometry.csv":
        INITIAL_AB_ROWS,

    "TEST146_final_interference_matrix.csv":
        INTERFERENCE_LOG,

    "TEST146_final_geometry.csv":
        FINAL_LOG,

    "TEST146_merge_diagnostics.csv":
        MERGE_LOG,

    "TEST146_behavior.csv":
        BEHAVIOR_LOG,

    "TEST146_gradient_xray_module.csv":
        GRAD_XRAY_LOG,

    "TEST146_gradient_xray_layer.csv":
        GRAD_XRAY_LAYER_LOG,

    "TEST146_gradient_xray_global.csv":
        GRAD_XRAY_GLOBAL_LOG,

    "TEST146_gradient_vs_interference.csv":
        XRAY_CORRELATION_LOG,

    "TEST146_module_conflict_summary.csv":
        MODULE_CONFLICT_SUMMARY,
}

for filename,data in tables.items():

    pd.DataFrame(
        data
    ).to_csv(
        os.path.join(
            CSV_DIR,
            filename
        ),
        index=False
    )


# =============================================================================
# 28. NPZ
# =============================================================================

NPZ_PATH=os.path.join(
    NPZ_DIR,
    "TEST146_parameter_space_xray_geometry.npz"
)

np.savez_compressed(
    NPZ_PATH,

    A0_compass=np.stack(
        [
            x.numpy()
            for x in A0_COMPASS
        ],
        axis=0
    ),

    B0_compass=np.stack(
        [
            x.numpy()
            for x in B0_COMPASS
        ],
        axis=0
    ),

    initial_AB_cos=
        initial_ab_cos,

    envelope=np.asarray(
        ENVELOPE,
        dtype=np.float32
    ),

    relative_dose=np.asarray(
        RELATIVE_DOSE,
        dtype=np.float32
    ),

    A_final_base=
        A_FINAL_BASE_NP,

    A_final_T0=
        A_FINAL_T0_NP,

    B_final_base=
        B_FINAL_BASE_NP,

    B_final_T0=
        B_FINAL_T0_NP,

    A_only_A_states=
        A_ONLY_FINAL[
            "A_states"
        ],

    A_only_B_states=
        A_ONLY_FINAL[
            "B_states"
        ],

    B_only_A_states=
        B_ONLY_FINAL[
            "A_states"
        ],

    B_only_B_states=
        B_ONLY_FINAL[
            "B_states"
        ],

    AB_A_states=
        AB_FINAL[
            "A_states"
        ],

    AB_B_states=
        AB_FINAL[
            "B_states"
        ],

    BA_A_states=
        BA_FINAL[
            "A_states"
        ],

    BA_B_states=
        BA_FINAL[
            "B_states"
        ],

    A_layer_interference=
        A_LAYER_INTERFERENCE,

    B_layer_interference=
        B_LAYER_INTERFERENCE,

    order_layer=
        ORDER_LAYER,
)


# =============================================================================
# 29. SUMMARY
# =============================================================================

EXPERIMENT_END_UTC=(
    datetime.now(
        timezone.utc
    ).isoformat()
)

SUMMARY={
    "experiment":
        "TEST 146 — AkbasCore Parameter-Space Interference X-Ray",

    "model":
        MODEL_ID,

    "seed":
        SEED,

    "vram_architecture":
        "single-live-7B-model",

    "motor":{
        "status":
            "LOCKED / UNMODIFIED",

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

        "brake":
            False,

        "cosine_gate":
            False,

        "radar":
            False,

        "gram_schmidt_actuator":
            False,

        "anti_interference_controller":
            False,
    },

    "test146_parameter_xray":{
        "diagnostic_only":
            True,

        "gradient_modified":
            False,

        "projection_applied":
            False,

        "xray_every":
            XRAY_EVERY,

        "probe_count_per_crystal":
            XRAY_N_PROBES,

        "modules":
            list(
                XRAY_MODULES
            ),

        "A_to_B":
            XRAY_AB_CORRELATION,

        "B_to_A":
            XRAY_BA_CORRELATION,
    },

    "xray_interpretation":{
        "gradient_pair":
            "preservation geometry-loss gradient versus acquisition geometry-loss gradient",

        "negative_dot":
            "conflict indicator under simple descent geometry; not a causality proof",

        "optimizer_warning":
            "AdamW actual parameter step is not identical to raw negative acquisition gradient",

        "projection_ratio":
            "absolute cosine magnitude",

        "correlation_warning":
            "gradient-versus-final-interference correlation is exploratory and repeated layer-step observations are not independent",
    },

    "data":{
        "train_per_crystal":
            N_TRAIN,

        "validation_per_crystal":
            N_VALIDATION,

        "final_per_crystal":
            N_FINAL,

        "final_student_metrics_used_for_training":
            False,

        "final_student_metrics_used_for_selection":
            False,
    },

    "initial_compass_geometry":{
        "mean_cos_A_B":
            float(
                initial_ab_cos.mean()
            ),

        "min_cos_A_B":
            float(
                initial_ab_cos.min()
            ),

        "max_cos_A_B":
            float(
                initial_ab_cos.max()
            ),
    },

    "A_only":{
        "best_step":
            A_RESULT[
                "best_step"
            ],

        "stop_reason":
            A_RESULT[
                "stop_reason"
            ],

        "final_A_progress":
            A_ONLY_FINAL[
                "A"
            ][
                "progress"
            ],

        "cross_B_progress":
            A_ONLY_FINAL[
                "B"
            ][
                "progress"
            ],

        "merge_drift":
            A_MERGE_DRIFT,

        "merge_status":
            A_MERGE_STATUS,
    },

    "B_only":{
        "best_step":
            B_RESULT[
                "best_step"
            ],

        "stop_reason":
            B_RESULT[
                "stop_reason"
            ],

        "final_B_progress":
            B_ONLY_FINAL[
                "B"
            ][
                "progress"
            ],

        "cross_A_progress":
            B_ONLY_FINAL[
                "A"
            ][
                "progress"
            ],

        "merge_drift":
            B_MERGE_DRIFT,

        "merge_status":
            B_MERGE_STATUS,
    },

    "A_to_B":{
        "best_step_B":
            AB_RESULT[
                "best_step"
            ],

        "stop_reason":
            AB_RESULT[
                "stop_reason"
            ],

        "final_A_progress":
            AB_FINAL[
                "A"
            ][
                "progress"
            ],

        "final_B_progress":
            AB_FINAL[
                "B"
            ][
                "progress"
            ],

        "I_A_from_B":
            I_A_FROM_B,

        "B_gain_relative_to_A_only":
            B_GAIN_IN_AB,

        "merge_drift":
            AB_MERGE_DRIFT,

        "merge_status":
            AB_MERGE_STATUS,
    },

    "B_to_A":{
        "best_step_A":
            BA_RESULT[
                "best_step"
            ],

        "stop_reason":
            BA_RESULT[
                "stop_reason"
            ],

        "final_A_progress":
            BA_FINAL[
                "A"
            ][
                "progress"
            ],

        "final_B_progress":
            BA_FINAL[
                "B"
            ][
                "progress"
            ],

        "I_B_from_A":
            I_B_FROM_A,

        "A_gain_relative_to_B_only":
            A_GAIN_IN_BA,

        "merge_drift":
            BA_MERGE_DRIFT,

        "merge_status":
            BA_MERGE_STATUS,
    },

    "order_effect":{
        "AB_vs_BA_raw_distance":
            ORDER_DISTANCE_RAW,

        "AB_vs_BA_normalized_distance":
            ORDER_DISTANCE_NORMALIZED,
    },

    "behavior":{
        "interpretation_warning":
            "Changed output is not correctness.",

        "AB":
            AB_BEHAVIOR,

        "BA":
            BA_BEHAVIOR,
    },

    "final_runtime_state":{
        "motor":
            "OFF",

        "seasc_injection":
            "OFF",

        "forward_hooks":
            "NONE",

        "runtime_lora":
            "REMOVED / MERGED",
    },

    "start_utc":
        EXPERIMENT_START_UTC,

    "end_utc":
        EXPERIMENT_END_UTC,
}

with open(
    os.path.join(
        JSON_DIR,
        "TEST146_summary.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        SUMMARY,
        f,
        ensure_ascii=False,
        indent=2
    )


# =============================================================================
# 30. REPORT
# =============================================================================

REPORT_PATH=os.path.join(
    REPORT_DIR,
    "TEST146_FULL_REPORT.txt"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TEST 146 — AKBASCORE PARAMETER-SPACE INTERFERENCE X-RAY\n"
    )

    f.write(
        "="*120+"\n\n"
    )

    f.write(
        f"Model: {MODEL_ID}\n"
    )

    f.write(
        "Crystal-A: ABOVE <-> BELOW\n"
    )

    f.write(
        "Crystal-B: BEFORE <-> AFTER\n"
    )

    f.write(
        "SEASC 3.0: LOCKED / UNMODIFIED\n"
    )

    f.write(
        "X-ray: DIAGNOSTIC ONLY\n"
    )

    f.write(
        "Gradient projection: NONE\n"
    )

    f.write(
        "Gradient intervention: NONE\n"
    )

    f.write(
        f"XRAY_N_PROBES: {XRAY_N_PROBES}\n"
    )

    f.write(
        f"XRAY_EVERY: {XRAY_EVERY}\n\n"
    )

    f.write(
        "INITIAL COMPASS GEOMETRY\n"
    )

    f.write(
        f"Mean cos(A,B): "
        f"{initial_ab_cos.mean():+.8f}\n"
    )

    f.write(
        f"Min cos(A,B): "
        f"{initial_ab_cos.min():+.8f}\n"
    )

    f.write(
        f"Max cos(A,B): "
        f"{initial_ab_cos.max():+.8f}\n\n"
    )

    f.write(
        "FINAL STATES\n"
    )

    f.write(
        f"A-only A: "
        f"{100*A_ONLY_FINAL['A']['progress']:+.6f}%\n"
    )

    f.write(
        f"B-only B: "
        f"{100*B_ONLY_FINAL['B']['progress']:+.6f}%\n"
    )

    f.write(
        f"A->B A: "
        f"{100*AB_FINAL['A']['progress']:+.6f}%\n"
    )

    f.write(
        f"A->B B: "
        f"{100*AB_FINAL['B']['progress']:+.6f}%\n"
    )

    f.write(
        f"B->A A: "
        f"{100*BA_FINAL['A']['progress']:+.6f}%\n"
    )

    f.write(
        f"B->A B: "
        f"{100*BA_FINAL['B']['progress']:+.6f}%\n\n"
    )

    f.write(
        f"I[A <- B]: "
        f"{100*I_A_FROM_B:+.6f} pp\n"
    )

    f.write(
        f"I[B <- A]: "
        f"{100*I_B_FROM_A:+.6f} pp\n"
    )

    f.write(
        f"AB vs BA normalized hidden distance: "
        f"{ORDER_DISTANCE_NORMALIZED:.10e}\n\n"
    )

    f.write(
        "PARAMETER-SPACE X-RAY\n"
    )

    f.write(
        f"A->B Pearson gradient cosine vs "
        f"final interference: "
        f"{XRAY_AB_CORRELATION['pearson_cos_vs_final_interference']:+.8f}\n"
    )

    f.write(
        f"A->B negative-dot fraction: "
        f"{XRAY_AB_CORRELATION['negative_dot_fraction']:.8f}\n"
    )

    f.write(
        f"B->A Pearson gradient cosine vs "
        f"final interference: "
        f"{XRAY_BA_CORRELATION['pearson_cos_vs_final_interference']:+.8f}\n"
    )

    f.write(
        f"B->A negative-dot fraction: "
        f"{XRAY_BA_CORRELATION['negative_dot_fraction']:.8f}\n\n"
    )

    f.write(
        "INTERPRETATION WARNING\n"
    )

    f.write(
        "Negative gradient dot product is a conflict indicator "
        "for simple descent geometry, not a causal proof.\n"
    )

    f.write(
        "AdamW actual parameter updates are not identical to "
        "raw negative acquisition gradients.\n"
    )

    f.write(
        "Gradient/final-interference correlations are exploratory.\n\n"
    )

    f.write(
        "MERGE DIAGNOSTICS\n"
    )

    for row in MERGE_LOG:

        f.write(
            f"{row['stage']}: "
            f"{row['relative_hidden_drift']:.10e} | "
            f"{row['status']}\n"
        )

    f.write(
        "\nFINAL RUNTIME\n"
    )

    f.write(
        "Motor: OFF\n"
        "SEASC: OFF\n"
        "Hooks: NONE\n"
        "LoRA: MERGED\n"
    )


# =============================================================================
# 31. TEST 146 FIGURES
# =============================================================================

plt.style.use(
    "dark_background"
)


def save_jpeg(
    fig,
    filename
):
    temp=os.path.join(
        FIG_DIR,
        "_temp.png"
    )

    final=os.path.join(
        FIG_DIR,
        filename
    )

    fig.savefig(
        temp,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    with Image.open(
        temp
    ) as im:

        if im.mode in (
            "RGBA",
            "LA"
        ):

            rgba=im.convert(
                "RGBA"
            )

            bg=Image.new(
                "RGB",
                rgba.size,
                (0,0,0)
            )

            bg.paste(
                rgba,
                mask=rgba.getchannel(
                    "A"
                )
            )

            rgb=bg

        else:

            rgb=im.convert(
                "RGB"
            )

        rgb.save(
            final,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False
        )

    os.remove(
        temp
    )

    with open(
        final,
        "rb"
    ) as f:

        if f.read(2)!=b"\xff\xd8":
            raise RuntimeError(
                "Invalid JPEG."
            )

    JPEG_FILES.append(
        final
    )


def gradient_conflict_heatmap(
    branch,
    filename
):
    rows=[
        r
        for r in GRAD_XRAY_LAYER_LOG
        if r["branch"]==branch
    ]

    if not rows:
        return

    steps=sorted(
        set(
            int(r["step"])
            for r in rows
        )
    )

    matrix=np.full(
        (
            len(steps),
            N_LAYERS
        ),
        np.nan,
        dtype=np.float64
    )

    step_index={
        s:i
        for i,s in enumerate(steps)
    }

    for r in rows:

        matrix[
            step_index[
                int(r["step"])
            ],
            int(r["layer"])
        ]=float(
            r["cosine"]
        )

    fig=plt.figure(
        figsize=(18,9)
    )

    im=plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    plt.colorbar(
        im,
        label="cos(g_preserve, g_acquire)"
    )

    plt.xticks(
        np.arange(
            N_LAYERS
        ),
        [
            f"L{x:02d}"
            for x in range(
                N_LAYERS
            )
        ],
        rotation=45
    )

    plt.yticks(
        np.arange(
            len(steps)
        ),
        [
            str(s)
            for s in steps
        ]
    )

    plt.xlabel(
        "Layer"
    )

    plt.ylabel(
        "Training step"
    )

    plt.title(
        f"TEST 146 — "
        f"{branch} — "
        f"Parameter-Space Gradient Conflict"
    )

    plt.tight_layout()

    save_jpeg(
        fig,
        filename
    )


def module_layer_heatmap(
    branch,
    filename
):
    matrix=np.full(
        (
            len(XRAY_MODULES),
            N_LAYERS
        ),
        np.nan,
        dtype=np.float64
    )

    for mi,module_name in enumerate(
        XRAY_MODULES
    ):

        for L in range(
            N_LAYERS
        ):

            rows=[
                r
                for r in GRAD_XRAY_LOG
                if
                r["branch"]==branch
                and
                r["module"]==module_name
                and
                int(r["layer"])==L
            ]

            if rows:

                matrix[mi,L]=float(
                    np.mean(
                        [
                            r["cosine"]
                            for r in rows
                        ]
                    )
                )

    fig=plt.figure(
        figsize=(18,6)
    )

    im=plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    plt.colorbar(
        im,
        label="Mean cos(g_preserve, g_acquire)"
    )

    plt.xticks(
        np.arange(
            N_LAYERS
        ),
        [
            f"L{x:02d}"
            for x in range(
                N_LAYERS
            )
        ],
        rotation=45
    )

    plt.yticks(
        np.arange(
            len(XRAY_MODULES)
        ),
        XRAY_MODULES
    )

    plt.xlabel(
        "Layer"
    )

    plt.ylabel(
        "Projection module"
    )

    plt.title(
        f"TEST 146 — "
        f"{branch} — "
        f"Mean Module/Layer Gradient Geometry"
    )

    plt.tight_layout()

    save_jpeg(
        fig,
        filename
    )


gradient_conflict_heatmap(
    "A_TO_B",
    "TEST146_GRADIENT_CONFLICT_A_TO_B.jpg"
)

gradient_conflict_heatmap(
    "B_TO_A",
    "TEST146_GRADIENT_CONFLICT_B_TO_A.jpg"
)

module_layer_heatmap(
    "A_TO_B",
    "TEST146_MODULE_LAYER_A_TO_B.jpg"
)

module_layer_heatmap(
    "B_TO_A",
    "TEST146_MODULE_LAYER_B_TO_A.jpg"
)


fig=plt.figure(
    figsize=(16,8)
)

plt.plot(
    np.arange(
        N_LAYERS
    ),
    initial_ab_cos,
    marker="o",
    linewidth=2
)

plt.axhline(
    0,
    linewidth=.8
)

plt.ylim(
    -1.05,
    1.05
)

plt.xticks(
    np.arange(
        N_LAYERS
    )
)

plt.xlabel(
    "Layer"
)

plt.ylabel(
    "cos(A_L, B_L)"
)

plt.title(
    "TEST 146 — Natural A/B Compass Geometry"
)

plt.grid(
    alpha=.2
)

plt.tight_layout()

save_jpeg(
    fig,
    "TEST146_INITIAL_AB_COMPASS.jpg"
)


fig=plt.figure(
    figsize=(17,8)
)

x=np.arange(
    N_LAYERS
)

plt.plot(
    x,
    100*A_LAYER_INTERFERENCE,
    marker="o",
    label="I[A <- B]"
)

plt.plot(
    x,
    100*B_LAYER_INTERFERENCE,
    marker="s",
    label="I[B <- A]"
)

plt.axhline(
    0,
    linewidth=.8
)

plt.xticks(
    x
)

plt.xlabel(
    "Layer"
)

plt.ylabel(
    "Interference (percentage points)"
)

plt.title(
    "TEST 146 — Layerwise Representation Interference"
)

plt.legend()

plt.grid(
    alpha=.2
)

plt.tight_layout()

save_jpeg(
    fig,
    "TEST146_LAYERWISE_REPRESENTATION_INTERFERENCE.jpg"
)

plt.close(
    "all"
)


# =============================================================================
# 32. FINAL SCIENTIFIC REPORT
# =============================================================================

print(
    "\n"+"="*170
)

print(
    "TEST 146 — FINAL SCIENTIFIC REPORT"
)

print(
    "="*170
)


print(
    "\nINITIAL GEOMETRY"
)

print(
    f"Mean cos(A,B) : "
    f"{initial_ab_cos.mean():+.8f}"
)

print(
    f"Min           : "
    f"L{int(np.argmin(initial_ab_cos)):02d} "
    f"{initial_ab_cos.min():+.8f}"
)

print(
    f"Max           : "
    f"L{int(np.argmax(initial_ab_cos)):02d} "
    f"{initial_ab_cos.max():+.8f}"
)


print(
    "\nSINGLE CRYSTAL"
)

print(
    f"A-only A : "
    f"{100*A_ONLY_FINAL['A']['progress']:+.6f}%"
)

print(
    f"A-only B : "
    f"{100*A_ONLY_FINAL['B']['progress']:+.6f}%"
)

print(
    f"B-only B : "
    f"{100*B_ONLY_FINAL['B']['progress']:+.6f}%"
)

print(
    f"B-only A : "
    f"{100*B_ONLY_FINAL['A']['progress']:+.6f}%"
)


print(
    "\nA -> B"
)

print(
    f"A final       : "
    f"{100*AB_FINAL['A']['progress']:+.6f}%"
)

print(
    f"B final       : "
    f"{100*AB_FINAL['B']['progress']:+.6f}%"
)

print(
    f"I[A <- B]     : "
    f"{100*I_A_FROM_B:+.6f} pp"
)

print(
    f"B gain        : "
    f"{100*B_GAIN_IN_AB:+.6f} pp"
)


print(
    "\nB -> A"
)

print(
    f"A final       : "
    f"{100*BA_FINAL['A']['progress']:+.6f}%"
)

print(
    f"B final       : "
    f"{100*BA_FINAL['B']['progress']:+.6f}%"
)

print(
    f"I[B <- A]     : "
    f"{100*I_B_FROM_A:+.6f} pp"
)

print(
    f"A gain        : "
    f"{100*A_GAIN_IN_BA:+.6f} pp"
)


print(
    "\nPARAMETER-SPACE X-RAY"
)

print(
    "Diagnostic only : YES"
)

print(
    "Gradient modified: NO"
)

print(
    "Projection       : NO"
)

print(
    "XRAY probes      :",
    XRAY_N_PROBES
)

print(
    "XRAY every       :",
    XRAY_EVERY
)


print(
    "\nA -> B X-RAY"
)

print(
    "Pearson cos vs final interference :",
    f"{XRAY_AB_CORRELATION['pearson_cos_vs_final_interference']:+.8f}"
)

print(
    "Negative-dot fraction             :",
    f"{100*XRAY_AB_CORRELATION['negative_dot_fraction']:.6f}%"
)


print(
    "\nB -> A X-RAY"
)

print(
    "Pearson cos vs final interference :",
    f"{XRAY_BA_CORRELATION['pearson_cos_vs_final_interference']:+.8f}"
)

print(
    "Negative-dot fraction             :",
    f"{100*XRAY_BA_CORRELATION['negative_dot_fraction']:.6f}%"
)


print(
    "\nORDER"
)

print(
    f"Raw distance  : "
    f"{ORDER_DISTANCE_RAW:.8f}"
)

print(
    f"Normalized    : "
    f"{100*ORDER_DISTANCE_NORMALIZED:.6f}%"
)

worst_order=int(
    np.argmax(
        ORDER_LAYER
    )
)

print(
    f"Max layer     : "
    f"L{worst_order:02d} "
    f"{100*ORDER_LAYER[worst_order]:.6f}%"
)


worst_A=int(
    np.argmin(
        A_LAYER_INTERFERENCE
    )
)

best_A=int(
    np.argmax(
        A_LAYER_INTERFERENCE
    )
)

worst_B=int(
    np.argmin(
        B_LAYER_INTERFERENCE
    )
)

best_B=int(
    np.argmax(
        B_LAYER_INTERFERENCE
    )
)


print(
    "\nREPRESENTATION INTERFERENCE HOTSPOTS"
)

print(
    f"A strongest loss : "
    f"L{worst_A:02d} "
    f"{100*A_LAYER_INTERFERENCE[worst_A]:+.6f} pp"
)

print(
    f"A strongest gain : "
    f"L{best_A:02d} "
    f"{100*A_LAYER_INTERFERENCE[best_A]:+.6f} pp"
)

print(
    f"B strongest loss : "
    f"L{worst_B:02d} "
    f"{100*B_LAYER_INTERFERENCE[worst_B]:+.6f} pp"
)

print(
    f"B strongest gain : "
    f"L{best_B:02d} "
    f"{100*B_LAYER_INTERFERENCE[best_B]:+.6f} pp"
)


print(
    "\nMERGE FORENSICS"
)

for row in MERGE_LOG:

    print(
        f"{row['stage']:12s} | "
        f"{row['relative_hidden_drift']:.10e} | "
        f"{row['status']}"
    )


print(
    "\nBEHAVIOR"
)

print(
    f"A->B | "
    f"A changed "
    f"{AB_BEHAVIOR['A_changed']}/{N_FINAL} | "
    f"B changed "
    f"{AB_BEHAVIOR['B_changed']}/{N_FINAL}"
)

print(
    f"B->A | "
    f"A changed "
    f"{BA_BEHAVIOR['A_changed']}/{N_FINAL} | "
    f"B changed "
    f"{BA_BEHAVIOR['B_changed']}/{N_FINAL}"
)

print(
    "changed != correct"
)


print(
    "\nSCIENTIFIC CAUTION"
)

print(
    "Negative cos(g_preserve, g_acquire) "
    "is a gradient-field conflict indicator."
)

print(
    "It is not by itself proof of causal parameter interference."
)

print(
    "AdamW actual optimizer steps are not identical "
    "to raw -gradient directions."
)

print(
    "Gradient/final-interference correlation is exploratory."
)


print(
    "\nFINAL RUNTIME"
)

print(
    "Motor       : OFF"
)

print(
    "SEASC       : OFF"
)

print(
    "Hooks       : NONE"
)

print(
    "LoRA        : REMOVED / MERGED"
)

print(
    "Final test selection use: NO"
)

memory_report(
    "end of scientific experiment"
)


# =============================================================================
# 33. ARCHIVE
# =============================================================================

ZIP_BASE=(
    "/content/"
    "TEST146_AKBASCORE_PARAMETER_XRAY_COMPLETE"
)

ZIP_PATH=shutil.make_archive(
    ZIP_BASE,
    "zip",
    OUT_DIR
)

print(
    "\nZIP:",
    ZIP_PATH
)

for root,dirs,files_in_dir in os.walk(
    OUT_DIR
):

    for filename in sorted(
        files_in_dir
    ):

        print(
            os.path.join(
                root,
                filename
            )
        )


from google.colab import files

files.download(
    ZIP_PATH
)


print(
    "\n"+"="*170
)

print(
    "TEST 146 COMPLETE"
)

print(
    "="*170
)
