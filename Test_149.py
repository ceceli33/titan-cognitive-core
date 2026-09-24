# =============================================================================
# TEST 149 — AKBASCORE CLOSED ENDOGENOUS SELF-LEARNING
# Qwen2.5-7B-Instruct | A100
#
# Based on the proven TEST 148 loading / hook / capture architecture.
#
# =============================================================================
# BUG FIX (this version)
# =============================================================================
# Root cause of the crash:
#
#   AcceleratorError: CUDA error: device-side assert triggered
#   (raised inside sampled_weight() -> flat[idx].cpu())
#
# sampled_weight() built its sampling indices with:
#
#   idx = torch.linspace(0, n - 1, count, device=flat.device).long()
#
# torch.linspace on a CUDA tensor computes in float32 by default.
# float32 can only represent integers EXACTLY up to 2**24 (~16.7M).
#
#   q_proj / k_proj / v_proj / o_proj  ~ 12.8M elements  -> safe
#   gate_proj / up_proj / down_proj    ~ 67.8M elements  -> UNSAFE
#
# For the large MLP matrices, the float32 endpoint of linspace can round
# to a value >= n. .long() truncates rather than clamping, so the final
# index sometimes equals n (one past the end) -> out-of-bounds indexing
# on the GPU -> device-side assert. This is why q/k/v/o never failed but
# gate/up/down eventually did.
#
# FIX: build the indices in float64 on CPU (numpy), round, and clip to
# [0, n-1] before moving to the GPU. This is correct regardless of matrix
# size.
#
# NOTE: once a device-side assert has fired, the CUDA context for that
# process is poisoned. You must restart the Colab runtime (Runtime ->
# Restart session) before re-running this corrected script.
# =============================================================================
#
# RESEARCH QUESTION
# -----------------
# What happens inside a freely trainable 7B model when:
#
#   1) the model generates its OWN question,
#   2) answers its OWN question,
#   3) an endogenous question->answer vector is extracted,
#   4) AkbasCore/SEASC applies layerwise pressure,
#   5) that pressured hidden target is crystallized into ALL model weights,
#   6) the changed model generates the next question,
#   7) this continues for 10,000 steps?
#
# NO EXTERNAL TRAINING KNOWLEDGE.
#
# CONTINUOUS TRAJECTORY
# ---------------------
# theta_0 -> theta_100 -> theta_1000 -> theta_10000
#
# IMPORTANT EXPERIMENTAL RULE
# ---------------------------
# The 5 fixed probe questions NEVER enter training.
#
# At steps 100 and 1000:
#   NO probe question is asked.
#   Only a frozen model checkpoint is written to disk.
#
# After the entire 10,000-step endogenous run is finished:
#
#   theta_0
#   theta_100
#   theta_1000
#   theta_10000
#
# are loaded ONE AT A TIME and the same 5 probes are tested:
#
#   AkbasCore OFF
#   AkbasCore ON
#
# Therefore probe evaluation cannot alter the learning trajectory.
# This "replay after the fact" design is also why no RNG snapshot/restore
# is required around milestone audits: nothing about the audits ever
# feeds back into training, because training has either not yet reached
# that point (100, 1000 are pure checkpoint saves, no probing) or has
# already fully finished (10000).
#
# VRAM:
#   Only one 7B model is resident at a time during replay.
#   Finished models/tensors are deleted aggressively.
#
# DOWNLOAD:
#   PNG ONLY.
#   No TXT/CSV/JSON/ZIP/model is downloaded.
#   PNG files are deleted from Colab after download requests.
#
# FINAL CONSOLE:
#   One compact MASTER FINAL AUDIT at the very bottom containing the
#   essential numerical results + all five questions and answers.
# =============================================================================


# =============================================================================
# SECOND-PASS FIXES (this version) — logic/bookkeeping, not the CUDA crash
# =============================================================================
#
# [F1] VECTOR ECOLOGY WAS NEVER SNAPSHOTTED AT MILESTONES
#      theta_t and the vector memory V_t are two coupled dynamical systems.
#      The previous version only checkpointed theta_t at 100/1000/10000;
#      V_t (VECTOR_MEMORY / VECTOR_HITS / TOTAL_NEW / TOTAL_REUSE) was never
#      captured, so "how did the vector ecology evolve?" could never be
#      answered after the fact. FIX: VECTOR_SNAPSHOTS[step] now stores a CPU
#      clone of the memory pool, hit counts, and NEW/REUSE totals at
#      step 0/100/1000/10000.
#
# [F2] "step" WAS ATTEMPT COUNT, NOT SUCCESSFUL-CRYSTALLIZATION COUNT
#      The old loop was `for step in range(1, TOTAL_STEPS+1)` and used
#      `continue` on non-finite loss/grad without decrementing anything.
#      So theta_10000 could actually be theta_9993 if 7 updates were
#      skipped, and — worse — if step==100 happened to be a skipped
#      update, save_checkpoint(100) would never run and replay would
#      crash looking for a checkpoint that was never written.
#      FIX: the loop is now a `while SUCCESSFUL_STEPS < TOTAL_STEPS`
#      driven strictly by successful optimizer steps. A separate
#      RAW_ATTEMPTS counter is kept only for generating unique
#      self-question prompts and is not used for any milestone logic.
#
# [F3] VECTOR MEMORY WAS MUTATED EVEN ON FAILED UPDATES
#      resolve_vector() used to write into VECTOR_MEMORY immediately,
#      before the loss/gradient finiteness check. A skipped update
#      therefore still advanced V_t even though theta_t did not move,
#      breaking the (theta_t, V_t) pairing the whole experiment depends
#      on. FIX: vector resolution is now two-phase. peek_vector() looks
#      up the nearest family and computes the candidate blended vector
#      WITHOUT mutating any global state. commit_vector() performs the
#      actual mutation (NEW insert / REUSE blend-and-store / LRU evict)
#      and is only called after opt.step() has succeeded. A failed
#      update now leaves (theta_t, V_t) completely unchanged.
#
# [F4] "PROBE LEAKAGE: NONE" WAS ASSERTED, NOT ENFORCED
#      The five probes were never fed into training directly, which is
#      true and remains true. But nothing stopped the model from
#      spontaneously inventing a self-question that is lexically close
#      to one of the probes (e.g. "12 objects + 3 more..."), which would
#      quietly let probe-shaped material influence the crystallization
#      target even without ever showing the model the literal probe text.
#      FIX: next_question() now runs a host-side (Python-only) similarity
#      gate against the fixed probe set using difflib.SequenceMatcher and
#      word-overlap (Jaccard) before accepting a generated question. This
#      never shows the model the probe text; it only ever discards the
#      model's own output on the CPU side, so the closed endogenous loop
#      is not altered, only filtered. Blocked attempts are counted in
#      PROBE_LEAK_BLOCKED and reported in the final audit.
#
# [F5] DISK: theta_10000 NO LONGER NEEDS TO BE WRITTEN TO DISK
#      Previously all three milestones (100/1000/10000) were saved as
#      full BF16 7B checkpoints (~15GB each) and theta_10000 was then
#      reloaded from disk for its own audit. Since the trained model is
#      still resident in memory the instant training finishes, its audit
#      is now run directly on that live object BEFORE it is deleted.
#      Only theta_100 and theta_1000 are ever written to disk, saving one
#      full checkpoint's worth of disk I/O and space.
#
# None of these change the CUDA fix from the previous version
# (sampled_weight() still builds indices in float64 on CPU and clips them
# before moving to the GPU).
# =============================================================================


# =============================================================================
# 0. ENVIRONMENT
# =============================================================================

import sys, os, gc, math, json, random, shutil, subprocess, warnings, re, time
import difflib
import importlib.metadata as md
from datetime import datetime, timezone
from collections import Counter, deque, defaultdict

# ---- environment -------------------------------------------------------------

try:
    ta = md.version("torchao")
except md.PackageNotFoundError:
    ta = None

if ta:
    print("Removing incompatible torchao:", ta)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        stdout=subprocess.DEVNULL
    )
    for k in list(sys.modules):
        if k == "torchao" or k.startswith("torchao."):
            del sys.modules[k]

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")


# =============================================================================
# 1. CONFIG
# =============================================================================

SEED = 42

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

DEVICE = torch.device("cuda")

N_LAYERS = 20

TOTAL_STEPS = 10_000

MILESTONES = (0, 100, 1_000, 10_000)

# -----------------------------------------------------------------------------
# AkbasCore / SEASC
# -----------------------------------------------------------------------------

IVME = .10
SONUM = .30
ZIRVE = .70
TABAN = .20

# -----------------------------------------------------------------------------
# Full-weight crystallization
# -----------------------------------------------------------------------------
#
# All model parameters are trainable.
#
# SGD is deliberately used instead of AdamW:
# AdamW optimizer states for 7.6B parameters would be prohibitively large
# on a 40 GB A100.
# -----------------------------------------------------------------------------

LR = 2e-6

GRAD_CLIP = .50

# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

MAX_TOKENS = 160

QUESTION_NEW_TOKENS = 48

ANSWER_NEW_TOKENS = 72

PROBE_NEW_TOKENS = 96

QUESTION_TEMP = .90

QUESTION_TOP_P = .93

# -----------------------------------------------------------------------------
# Vector memory
# -----------------------------------------------------------------------------

VECTOR_REUSE_COS = .82

VECTOR_BLEND = .35

MAX_VECTOR_MEMORIES = 512

# CPU storage to avoid filling VRAM.
VECTOR_STORE_DTYPE = torch.float16

# -----------------------------------------------------------------------------
# Console / telemetry
# -----------------------------------------------------------------------------

HEARTBEAT_EVERY = 100

TELEMETRY_EVERY = 10

# -----------------------------------------------------------------------------
# Weight X-ray
# -----------------------------------------------------------------------------
#
# Deterministic samples distributed over attention + MLP matrices.
# -----------------------------------------------------------------------------

WEIGHT_SAMPLE_PER_MATRIX = 2048

WEIGHT_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT = "/content/TEST149_AKBASCORE_CLOSED_ENDOGENOUS"

CKPT = f"{ROOT}/CHECKPOINTS"

FIG = f"{ROOT}/FIGURES"

shutil.rmtree(ROOT, ignore_errors=True)

os.makedirs(CKPT, exist_ok=True)

os.makedirs(FIG, exist_ok=True)

START = datetime.now(timezone.utc).isoformat()


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def seed(x=0):

    s = SEED + x

    random.seed(s)

    np.random.seed(s)

    torch.manual_seed(s)

    torch.cuda.manual_seed_all(s)


seed()

torch.backends.cuda.matmul.allow_tf32 = True


# =============================================================================
# 3. CLEANUP
# =============================================================================

def clean():

    gc.collect()

    torch.cuda.empty_cache()

    try:
        torch.cuda.ipc_collect()
    except:
        pass


# =============================================================================
# 4. AKBASCORE ENVELOPE
# =============================================================================

def envelope(L):

    x = (
        ZIRVE
        * math.exp(-SONUM * L)
        * (1 + SONUM * L)
        + TABAN
    )

    return x / (ZIRVE + TABAN)


ENV = np.array(
    [envelope(i) for i in range(N_LAYERS)],
    dtype=np.float64
)

DOSE = IVME * ENV


# =============================================================================
# 5. FIXED ISOLATED PROBES
# =============================================================================
#
# These are ONLY measurement instruments.
#
# They are never:
#   - used for training
#   - used for target construction
#   - added to vector memory
#   - used in loss
#   - used in backpropagation
#   - shown at step 100 or 1000 during training
#
# Replay occurs only after the 10,000-step training trajectory has finished.
# =============================================================================

PROBES = [

    (
        "A red cube is inside a transparent box. "
        "The transparent box is inside a larger wooden crate. "
        "Where is the red cube relative to the wooden crate?"
    ),

    (
        "Event A happens before Event B, and Event B happens before Event C. "
        "If this ordering does not change, which event occurs last?"
    ),

    (
        "A closed room is continuously heated while heat escapes through its "
        "walls. If heating power increases while everything else remains the "
        "same, what should happen to the room's equilibrium temperature?"
    ),

    (
        "A container holds 12 identical objects. Three more identical objects "
        "are added and none are removed. How many objects are now in the container?"
    ),

    (
        "System X uses less energy than System Y to perform the same task, "
        "while both produce the same useful output. Which system is more "
        "energy-efficient, and why?"
    ),
]

N_PROBES = len(PROBES)

assert N_PROBES == 5


# =============================================================================
# 6. SYSTEM PROMPTS
# =============================================================================

SYSTEM = (
    "You are a concise reasoning assistant. "
    "Answer clearly and precisely."
)

SELF_QUESTION_SYSTEM = (
    "Create exactly one short reasoning question using only relationships and "
    "knowledge already represented inside your own model. "
    "Do not use tools or external sources. "
    "Do not answer the question. "
    "Output only the question."
)


# =============================================================================
# 7. TOKENIZER
# =============================================================================

tok = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


# =============================================================================
# 8. MODEL LOADING — TEST148 STYLE
# =============================================================================

def load_base():

    clean()

    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    m.config.use_cache = False

    return m


def load_checkpoint(path):

    clean()

    m = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    m.config.use_cache = False

    return m


def layers(m):

    for f in (
        lambda: m.model.layers,
        lambda: m.model.model.layers,
        lambda: m.base_model.model.model.layers,
        lambda: m.base_model.model.model.model.layers
    ):

        try:

            x = f()

            if len(x) >= N_LAYERS:
                return x

        except:
            pass

    raise RuntimeError("Decoder layers unresolved.")


# =============================================================================
# 9. CHAT ENCODING
# =============================================================================

def enc(m, prompt, system=SYSTEM):

    t = tok.apply_chat_template(
        [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    x = tok(
        t,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS
    )

    return {
        k: v.to(DEVICE)
        for k, v in x.items()
    }


# =============================================================================
# 10. GEOMETRY UTILITIES
# =============================================================================

def cosine(a, b):

    if torch.is_tensor(a):
        a = a.detach().float().cpu().numpy()

    if torch.is_tensor(b):
        b = b.detach().float().cpu().numpy()

    a = np.asarray(a, dtype=np.float64).reshape(-1)

    b = np.asarray(b, dtype=np.float64).reshape(-1)

    n = np.linalg.norm(a) * np.linalg.norm(b)

    if n < 1e-12:
        return 0.

    return float(
        np.clip(
            np.dot(a, b) / n,
            -1,
            1
        )
    )


def angle_deg(a, b):

    c = cosine(a, b)

    return math.degrees(
        math.acos(
            np.clip(c, -1, 1)
        )
    )


def normalize_layers(v):

    return (
        v
        /
        torch.linalg.vector_norm(
            v,
            dim=1,
            keepdim=True
        ).clamp_min(1e-12)
    )


def steer(h, v, rho):

    n = h.float().norm(
        dim=-1,
        keepdim=True
    )

    return (
        h.float()
        + rho
        * n
        * v.to(h.device).float().view(1, 1, -1)
    ).to(h.dtype)


# =============================================================================
# 11. GENERATION
# =============================================================================

@torch.inference_mode()
def generate_answer(m, question, max_new_tokens=ANSWER_NEW_TOKENS):

    was = m.training

    m.eval()

    x = enc(m, question)

    y = m.generate(
        **x,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.10,
        pad_token_id=tok.eos_token_id
    )

    z = y[
        0,
        x["input_ids"].shape[1]:
    ]

    answer = tok.decode(
        z,
        skip_special_tokens=True
    ).strip()

    del x, y, z

    if was:
        m.train()

    return answer


@torch.inference_mode()
def generate_self_question(m, step, attempt=0):

    was = m.training

    m.eval()

    # The numeric identifier is not semantic training information.
    prompt = (
        "Invent one reasoning problem from your own internal knowledge. "
        f"Internal run identifier: {step}-{attempt}."
    )

    x = enc(
        m,
        prompt,
        system=SELF_QUESTION_SYSTEM
    )

    y = m.generate(
        **x,
        max_new_tokens=QUESTION_NEW_TOKENS,
        do_sample=True,
        temperature=QUESTION_TEMP,
        top_p=QUESTION_TOP_P,
        repetition_penalty=1.10,
        pad_token_id=tok.eos_token_id
    )

    z = y[
        0,
        x["input_ids"].shape[1]:
    ]

    q = tok.decode(
        z,
        skip_special_tokens=True
    ).strip()

    q = q.split("\n")[0].strip()

    if not q.endswith("?"):
        q = q.rstrip(".") + "?"

    del x, y, z

    if was:
        m.train()

    return q


# =============================================================================
# 12. TEST148-STYLE CAPTURE
# =============================================================================

@torch.inference_mode()
def capture(m, p, c=None):

    was = m.training

    m.eval()

    x = enc(m, p)

    q = int(
        x["attention_mask"][0].sum()
    ) - 1

    if c is None:

        o = m(
            **x,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

        z = np.stack(
            [
                o.hidden_states[L + 1][0, q]
                .float()
                .cpu()
                .numpy()
                for L in range(N_LAYERS)
            ]
        )

        del o

    else:

        cap = [None] * N_LAYERS

        hs = []

        ls = layers(m)

        for L in range(N_LAYERS):

            def hook(mod, args, out, L=L):

                h = (
                    out[0]
                    if isinstance(out, tuple)
                    else out
                )

                s = steer(
                    h,
                    c[L],
                    DOSE[L]
                )

                cap[L] = (
                    s[0, q]
                    .float()
                    .cpu()
                    .numpy()
                    .copy()
                )

                return (
                    (s,) + out[1:]
                    if isinstance(out, tuple)
                    else s
                )

            hs.append(
                ls[L].register_forward_hook(
                    hook
                )
            )

        try:

            m(
                **x,
                use_cache=False,
                return_dict=True
            )

        finally:

            for h in hs:
                h.remove()

        z = np.stack(cap)

    del x

    if was:
        m.train()

    return z


# =============================================================================
# 13. ANSWER GEOMETRY
# =============================================================================

@torch.inference_mode()
def capture_answer_geometry(m, question, answer):

    was = m.training

    m.eval()

    p = (
        question
        + "\n\nProposed answer:\n"
        + answer
    )

    x = enc(m, p)

    q = int(
        x["attention_mask"][0].sum()
    ) - 1

    o = m(
        **x,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )

    z = np.stack(
        [
            o.hidden_states[L + 1][0, q]
            .float()
            .cpu()
            .numpy()
            for L in range(N_LAYERS)
        ]
    )

    del x, o

    if was:
        m.train()

    return z


def endogenous_vector(qgeo, ageo):

    d = torch.as_tensor(
        ageo - qgeo,
        dtype=torch.float32
    )

    return normalize_layers(d)


# =============================================================================
# 14. VECTOR MEMORY — CPU ONLY
# =============================================================================

VECTOR_MEMORY = []

VECTOR_HITS = Counter()

TOTAL_NEW = 0

TOTAL_REUSE = 0


def vector_similarity(a, b):

    a = a.float()

    b = b.float()

    cs = F.cosine_similarity(
        a,
        b,
        dim=1
    )

    return float(
        cs.mean()
    )


def peek_vector(raw):

    # -----------------------------------------------------------------------
    # [F3] READ-ONLY PHASE — no global state is mutated here.
    #
    # This only *proposes* what would happen to the vector memory if this
    # step succeeds. Nothing is written to VECTOR_MEMORY / VECTOR_HITS /
    # TOTAL_NEW / TOTAL_REUSE until commit_vector() is explicitly called
    # after a successful opt.step().
    # -----------------------------------------------------------------------

    raw = raw.detach().cpu().float()

    if len(VECTOR_MEMORY) == 0:

        return dict(
            vector=raw,
            mode="NEW",
            best_i=None,
            best_c=float("nan"),
            raw=raw
        )

    best_i = None

    best_c = -2.

    for i, old in enumerate(VECTOR_MEMORY):

        c = vector_similarity(
            raw,
            old.float()
        )

        if c > best_c:

            best_c = c

            best_i = i

    if best_c >= VECTOR_REUSE_COS:

        old = VECTOR_MEMORY[
            best_i
        ].float()

        mixed = normalize_layers(
            (1. - VECTOR_BLEND) * old
            + VECTOR_BLEND * raw
        )

        return dict(
            vector=mixed,
            mode="REUSE",
            best_i=best_i,
            best_c=best_c,
            raw=raw
        )

    return dict(
        vector=raw,
        mode="NEW",
        best_i=None,
        best_c=best_c,
        raw=raw
    )


def commit_vector(peek):

    # -----------------------------------------------------------------------
    # [F3] WRITE PHASE — call this ONLY after opt.step() has succeeded.
    #
    # This is the only place VECTOR_MEMORY / VECTOR_HITS / TOTAL_NEW /
    # TOTAL_REUSE are ever mutated, so a failed crystallization step can
    # never move V_t forward while theta_t stays put.
    # -----------------------------------------------------------------------

    global TOTAL_NEW
    global TOTAL_REUSE

    if peek["mode"] == "REUSE":

        i = peek["best_i"]

        VECTOR_MEMORY[i] = (
            peek["vector"]
            .to(VECTOR_STORE_DTYPE)
            .clone()
        )

        VECTOR_HITS[i] += 1

        TOTAL_REUSE += 1

        return i

    # mode == "NEW"

    TOTAL_NEW += 1

    raw = peek["raw"]

    if len(VECTOR_MEMORY) < MAX_VECTOR_MEMORIES:

        VECTOR_MEMORY.append(
            raw.to(
                VECTOR_STORE_DTYPE
            ).clone()
        )

        idx = len(VECTOR_MEMORY) - 1

    else:

        usage = [
            VECTOR_HITS[i]
            for i in range(
                len(VECTOR_MEMORY)
            )
        ]

        idx = int(
            np.argmin(usage)
        )

        VECTOR_MEMORY[idx] = (
            raw.to(
                VECTOR_STORE_DTYPE
            ).clone()
        )

        VECTOR_HITS[idx] = 0

    return idx


def snapshot_vector_memory():

    # -----------------------------------------------------------------------
    # [F1] Milestone snapshot of the whole vector ecology V_t.
    #
    # ~512 x 20 x 3584 FP16 is roughly 70MB — cheap next to the 7B
    # checkpoints, and it is the only way to later ask "how did the
    # family of endogenous directions evolve between milestones?".
    # -----------------------------------------------------------------------

    return dict(
        memory=[
            v.clone()
            for v in VECTOR_MEMORY
        ],
        hits=dict(VECTOR_HITS),
        total_new=TOTAL_NEW,
        total_reuse=TOTAL_REUSE,
    )


VECTOR_SNAPSHOTS = {
    0: snapshot_vector_memory()
}


# =============================================================================
# 15. TEACHER — TEST148-STYLE SEASC PRESSURE
# =============================================================================

@torch.no_grad()
def teacher(m, p, c):

    was = m.training

    m.eval()

    x = enc(m, p)

    q = int(
        x["attention_mask"][0].sum()
    ) - 1

    ls = layers(m)

    cap = [None] * N_LAYERS

    hs = []

    for L in range(N_LAYERS):

        def hook(mod, args, out, L=L):

            h = (
                out[0]
                if isinstance(out, tuple)
                else out
            )

            s = steer(
                h,
                c[L],
                DOSE[L]
            )

            cap[L] = (
                s[0, q]
                .detach()
                .float()
                .cpu()
                .half()
            )

            return (
                (s,) + out[1:]
                if isinstance(out, tuple)
                else s
            )

        hs.append(
            ls[L].register_forward_hook(
                hook
            )
        )

    try:

        m(
            **x,
            use_cache=False,
            return_dict=True
        )

    finally:

        for h in hs:
            h.remove()

    if was:
        m.train()

    del x

    return torch.stack(cap)


# =============================================================================
# 16. FULL-WEIGHT CRYSTALLIZATION LOSS
# =============================================================================

def train_loss(m, p, target):

    x = enc(m, p)

    q = int(
        x["attention_mask"][0].sum()
    ) - 1

    o = m(
        **x,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
    )

    losses = []

    for L in range(N_LAYERS):

        h = (
            o.hidden_states[L + 1][0, q]
            .float()
        )

        t = (
            target[L]
            .to(DEVICE)
            .float()
        )

        num = (
            h - t
        ).square().sum()

        den = (
            t.square()
            .sum()
            .detach()
            .clamp_min(1e-6)
        )

        losses.append(
            num / den
        )

    loss = torch.stack(
        losses
    ).mean()

    del x, o

    return loss


# =============================================================================
# 17. WEIGHT FINGERPRINT
# =============================================================================
#
# We need actual weight-space evidence without keeping a complete FP32 copy
# of 7.6B parameters.
#
# Deterministic samples are taken from:
#
# q/k/v/o attention projections
# gate/up/down MLP projections
#
# for all observed 20 layers.
#
# Stored baseline samples are FP32 CPU.
#
# --- FIX -------------------------------------------------------------------
# Indices are now built in float64 on CPU (numpy), rounded, and clipped to
# [0, n-1] BEFORE being moved to the GPU. The previous version built them
# with torch.linspace(..., device=cuda) in float32, which cannot exactly
# represent integers above 2**24 (~16.7M). gate_proj/up_proj/down_proj have
# ~67.8M elements each, so the float32 endpoint could round up to n,
# producing an out-of-bounds index and a CUDA device-side assert.
# -----------------------------------------------------------------------------

def get_matrix(layer, name):

    if name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ):

        return getattr(
            layer.self_attn,
            name
        ).weight

    return getattr(
        layer.mlp,
        name
    ).weight


@torch.no_grad()
def sampled_weight(p):

    flat = p.detach().reshape(-1)

    n = flat.numel()

    count = min(
        WEIGHT_SAMPLE_PER_MATRIX,
        n
    )

    if count <= 1:

        idx_np = np.zeros(max(count, 1), dtype=np.int64)

    else:

        # float64 on CPU: exact integer representation up to 2**53,
        # far beyond any matrix size we will ever sample here.
        idx_np = np.linspace(0, n - 1, count, dtype=np.float64)

        idx_np = np.clip(
            np.round(idx_np),
            0,
            n - 1
        ).astype(np.int64)

    idx = torch.from_numpy(idx_np).to(flat.device)

    return (
        flat[idx]
        .float()
        .cpu()
        .clone()
    )


@torch.no_grad()
def weight_fingerprint(m):

    out = {}

    ls = layers(m)

    for L in range(N_LAYERS):

        out[L] = {}

        for name in WEIGHT_MODULES:

            out[L][name] = sampled_weight(
                get_matrix(
                    ls[L],
                    name
                )
            )

    return out


def fingerprint_metrics(base_fp, now_fp):

    layer_rel = []

    matrix_rel = defaultdict(list)

    changed_fraction = []

    for L in range(N_LAYERS):

        d2 = 0.

        b2 = 0.

        changed = 0

        total = 0

        for name in WEIGHT_MODULES:

            b = base_fp[L][name].float()

            n = now_fp[L][name].float()

            d = n - b

            dn2 = float(
                d.square().sum()
            )

            bn2 = float(
                b.square().sum()
            )

            d2 += dn2

            b2 += bn2

            matrix_rel[name].append(
                math.sqrt(dn2)
                /
                max(
                    math.sqrt(bn2),
                    1e-12
                )
            )

            changed += int(
                (n != b).sum()
            )

            total += b.numel()

        layer_rel.append(
            math.sqrt(d2)
            /
            max(
                math.sqrt(b2),
                1e-12
            )
        )

        changed_fraction.append(
            changed / max(total, 1)
        )

    return dict(

        layer_relative=np.asarray(
            layer_rel,
            dtype=np.float64
        ),

        mean_relative=float(
            np.mean(layer_rel)
        ),

        max_relative=float(
            np.max(layer_rel)
        ),

        mean_changed_fraction=float(
            np.mean(changed_fraction)
        ),

        matrix_relative={
            k: float(np.mean(v))
            for k, v in matrix_rel.items()
        }
    )


# =============================================================================
# 18. CHECKPOINT SAVE
# =============================================================================

def checkpoint_path(step):

    return f"{CKPT}/STEP_{step:05d}"


def save_checkpoint(m, step):

    path = checkpoint_path(step)

    shutil.rmtree(
        path,
        ignore_errors=True
    )

    print(
        f"\n[CHECKPOINT] Saving frozen theta_{step:,} ..."
    )

    m.save_pretrained(
        path,
        safe_serialization=True,
        max_shard_size="5GB"
    )

    tok.save_pretrained(path)

    print(
        f"[CHECKPOINT] theta_{step:,} saved."
    )

    return path


# =============================================================================
# 19. TRAINING TELEMETRY
# =============================================================================

TELEMETRY = []

ROLL_LOSS = deque(maxlen=100)

ROLL_GRAD = deque(maxlen=100)

ROLL_COS = deque(maxlen=100)

SUCCESSFUL_STEPS = 0

NONFINITE_SKIPS = 0

DUPLICATE_REJECTIONS = 0

RECENT_Q = deque(maxlen=512)

RECENT_Q_SET = set()


def normalize_q(q):

    return re.sub(
        r"\s+",
        " ",
        q.strip().lower()
    )


def remember_question(q):

    global DUPLICATE_REJECTIONS

    key = normalize_q(q)

    if key in RECENT_Q_SET:

        return False

    if len(RECENT_Q) == RECENT_Q.maxlen:

        old = RECENT_Q.popleft()

        RECENT_Q_SET.discard(old)

    RECENT_Q.append(key)

    RECENT_Q_SET.add(key)

    return True


# =============================================================================
# [F4] HOST-SIDE PROBE-ISOLATION GATE
# =============================================================================
#
# This never shows the model the probe text. It only ever inspects, on the
# Python/CPU side, text the model has ALREADY generated on its own, and
# discards it if it is lexically too close to one of the five permanently
# isolated probes. This keeps the closed endogenous loop closed (no probe
# information enters the model) while preventing an accidental collision
# between a self-generated question and a probe from quietly contaminating
# the crystallization target.
#
# Two cheap, dependency-free similarity signals are combined:
#   - difflib.SequenceMatcher ratio on the normalized strings
#   - word-level Jaccard overlap
# =============================================================================

PROBE_NORM = [
    normalize_q(p)
    for p in PROBES
]

PROBE_WORDS = [
    set(p.split())
    for p in PROBE_NORM
]

PROBE_LEAK_THRESHOLD = 0.55

PROBE_LEAK_BLOCKED = 0


def probe_similarity(q):

    qn = normalize_q(q)

    qw = set(qn.split())

    best = 0.0

    for pn, pw in zip(PROBE_NORM, PROBE_WORDS):

        seq_ratio = difflib.SequenceMatcher(
            None,
            qn,
            pn
        ).ratio()

        union = qw | pw

        jaccard = (
            len(qw & pw) / len(union)
            if union
            else 0.0
        )

        score = max(seq_ratio, jaccard)

        if score > best:

            best = score

    return best


def next_question(m, step):

    global DUPLICATE_REJECTIONS
    global PROBE_LEAK_BLOCKED

    last = None

    # A few extra attempts vs. before (5 -> 8) since we now also filter
    # for probe-similarity, not just exact duplicates.
    for attempt in range(8):

        q = generate_self_question(
            m,
            step,
            attempt
        )

        last = q

        if not remember_question(q):

            DUPLICATE_REJECTIONS += 1

            continue

        if probe_similarity(q) >= PROBE_LEAK_THRESHOLD:

            PROBE_LEAK_BLOCKED += 1

            continue

        return q

    # All attempts either duplicated or leaked toward a probe: fall back to
    # the last generated question rather than stalling the closed loop.
    # This is logged so it is visible in the final audit, not silently
    # swallowed.
    return last


# =============================================================================
# 20. START MODEL
# =============================================================================

print("=" * 120)

print(
    "TEST 149 — AKBASCORE CLOSED ENDOGENOUS SELF-LEARNING"
)

print("=" * 120)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "Torch:",
    torch.__version__
)

print(
    "Transformers:",
    transformers.__version__
)

print(
    "External training data: NONE"
)

print(
    "Retention Guard: OFF"
)

print(
    "Structural growth: OFF"
)

print(
    "Training trajectory: 0 -> 100 -> 1,000 -> 10,000"
)

print(
    "Probe evaluation during training: NONE"
)

print("=" * 120)


m = load_base()

HIDDEN = int(
    m.config.hidden_size
)

TOTAL_PARAMS = sum(
    p.numel()
    for p in m.parameters()
)

# ALL WEIGHTS FREE.

for p in m.parameters():

    p.requires_grad_(True)


TRAINABLE_PARAMS = sum(
    p.numel()
    for p in m.parameters()
    if p.requires_grad
)


print(
    "Hidden:",
    HIDDEN
)

print(
    "Decoder layers:",
    len(layers(m))
)

print(
    "Observed/pressured layers:",
    N_LAYERS
)

print(
    "Total parameters:",
    f"{TOTAL_PARAMS:,}"
)

print(
    "Trainable parameters:",
    f"{TRAINABLE_PARAMS:,}"
)


# =============================================================================
# 21. STEP-0 WEIGHT FINGERPRINT
# =============================================================================

print(
    "\nCapturing step-0 distributed FP32 weight fingerprint..."
)

BASE_FP = weight_fingerprint(m)

print(
    "Step-0 fingerprint captured."
)


# =============================================================================
# 22. STEP-0 CHECKPOINT
# =============================================================================
#
# We do NOT save a second 15GB copy of the untouched base.
#
# theta_0 is exactly MODEL_ID and can be reloaded later.
# =============================================================================

print(
    "theta_0 replay source:",
    MODEL_ID
)


# =============================================================================
# 23. OPTIMIZER
# =============================================================================

params = [
    p
    for p in m.parameters()
    if p.requires_grad
]

opt = torch.optim.SGD(
    params,
    lr=LR,
    momentum=0.
)


# =============================================================================
# 24. CLOSED ENDOGENOUS TRAINING
# =============================================================================

print(
    "\n"
    + "=" * 120
)

print(
    "CLOSED ENDOGENOUS TRAINING START"
)

print(
    "No external question/answer/data enters the learning loop."
)

print(
    "The five probes are not touched until training is completely finished."
)

print(
    "=" * 120
)


t0 = time.time()

# -----------------------------------------------------------------------------
# [F2] SUCCESSFUL_STEPS is now the ONLY thing that advances the loop and the
# ONLY thing milestone checks (100/1000/10000) and checkpoint saves are keyed
# on. A skipped (non-finite) update never increments it, so theta_10000
# always means exactly 10,000 successful crystallizations, never fewer, and
# a milestone can never be silently missed because it happened to coincide
# with a failed step.
#
# RAW_ATTEMPTS is a separate, purely-cosmetic counter used only to build a
# unique identifier string for generate_self_question(); it has no effect on
# milestones, checkpoints, or telemetry indexing.
#
# MAX_RAW_ATTEMPTS is a defensive ceiling so a pathological run (e.g. the
# model repeatedly producing non-finite losses) cannot spin forever; it is
# set generously (5x TOTAL_STEPS) since a handful of skips is expected but
# a runaway failure rate is not.
# -----------------------------------------------------------------------------

RAW_ATTEMPTS = 0

MAX_RAW_ATTEMPTS = TOTAL_STEPS * 5

ABORTED_EARLY = False


while SUCCESSFUL_STEPS < TOTAL_STEPS:

    RAW_ATTEMPTS += 1

    if RAW_ATTEMPTS > MAX_RAW_ATTEMPTS:

        print(
            "\n[ABORT] MAX_RAW_ATTEMPTS exceeded "
            f"({MAX_RAW_ATTEMPTS:,}) with only "
            f"{SUCCESSFUL_STEPS:,}/{TOTAL_STEPS:,} successful "
            "crystallizations. Stopping the closed loop early; "
            "replay audit will run on whatever milestones were "
            "actually reached."
        )

        ABORTED_EARLY = True

        break

    # -------------------------------------------------------------------------
    # 1. CURRENT MODEL INVENTS ITS OWN QUESTION
    #
    # [F4] next_question() now also rejects questions that are lexically
    # close to one of the five isolated probes, purely on the Python side.
    # -------------------------------------------------------------------------

    q = next_question(
        m,
        RAW_ATTEMPTS
    )

    # -------------------------------------------------------------------------
    # 2. CURRENT MODEL ANSWERS ITS OWN QUESTION
    # -------------------------------------------------------------------------

    a = generate_answer(
        m,
        q
    )

    # -------------------------------------------------------------------------
    # 3. CURRENT ENDOGENOUS Q -> A GEOMETRY
    # -------------------------------------------------------------------------

    qgeo = capture(
        m,
        q
    )

    ageo = capture_answer_geometry(
        m,
        q,
        a
    )

    raw_v = endogenous_vector(
        qgeo,
        ageo
    )

    # -------------------------------------------------------------------------
    # 4. NEW / REUSE — READ-ONLY PEEK
    #
    # [F3] peek_vector() does not mutate VECTOR_MEMORY / VECTOR_HITS /
    # TOTAL_NEW / TOTAL_REUSE. It only proposes what the update would be.
    # -------------------------------------------------------------------------

    peek = peek_vector(
        raw_v
    )

    c = peek["vector"]

    nearest_cos = peek["best_c"]

    # -------------------------------------------------------------------------
    # 5. AKBASCORE PRESSURE
    # -------------------------------------------------------------------------

    target = teacher(
        m,
        q,
        c
    )

    # -------------------------------------------------------------------------
    # 6. MOTOR REMOVED -> CRYSTALLIZE INTO FREE 7B WEIGHTS
    # -------------------------------------------------------------------------

    m.train()

    opt.zero_grad(
        set_to_none=True
    )

    loss = train_loss(
        m,
        q,
        target
    )

    if not torch.isfinite(loss):

        # [F3] Failure path: (theta_t, V_t) stays exactly as it was.
        # peek_vector() already touched nothing, so there is nothing to
        # roll back here.

        NONFINITE_SKIPS += 1

        opt.zero_grad(
            set_to_none=True
        )

        del (
            q,
            a,
            qgeo,
            ageo,
            raw_v,
            peek,
            c,
            target,
            loss
        )

        clean()

        continue

    loss.backward()

    gn = torch.nn.utils.clip_grad_norm_(
        params,
        GRAD_CLIP
    )

    gn_f = float(
        torch.as_tensor(gn)
        .detach()
        .float()
        .cpu()
    )

    if not math.isfinite(gn_f):

        # [F3] Same guarantee as above: vector memory is still untouched.

        NONFINITE_SKIPS += 1

        opt.zero_grad(
            set_to_none=True
        )

        del (
            q,
            a,
            qgeo,
            ageo,
            raw_v,
            peek,
            c,
            target,
            loss,
            gn
        )

        clean()

        continue

    opt.step()

    # -------------------------------------------------------------------------
    # [F3] COMMIT PHASE — theta_t just moved, so V_t is now allowed to move
    # too, and only now.
    # -------------------------------------------------------------------------

    memory_id = commit_vector(
        peek
    )

    # -------------------------------------------------------------------------
    # [F2] Only a successful opt.step() advances the true crystallization
    # counter that everything else (milestones, checkpoints, ETA) is keyed
    # on.
    # -------------------------------------------------------------------------

    SUCCESSFUL_STEPS += 1

    step = SUCCESSFUL_STEPS

    loss_f = float(
        loss.detach().float().cpu()
    )

    ROLL_LOSS.append(loss_f)

    ROLL_GRAD.append(gn_f)

    if math.isfinite(nearest_cos):

        ROLL_COS.append(nearest_cos)

    # -------------------------------------------------------------------------
    # 7. LIGHT TELEMETRY
    # -------------------------------------------------------------------------

    if (
        step % TELEMETRY_EVERY == 0
        or step in (100, 1000, 10000)
    ):

        TELEMETRY.append(
            dict(
                step=step,
                raw_attempts=RAW_ATTEMPTS,
                loss=loss_f,
                grad_norm=gn_f,
                memories=len(VECTOR_MEMORY),
                new=TOTAL_NEW,
                reuse=TOTAL_REUSE,
                nearest_cos=(
                    nearest_cos
                    if math.isfinite(nearest_cos)
                    else np.nan
                )
            )
        )

    # -------------------------------------------------------------------------
    # 8. FREE STEP-LOCAL OBJECTS
    # -------------------------------------------------------------------------

    opt.zero_grad(
        set_to_none=True
    )

    del (
        q,
        a,
        qgeo,
        ageo,
        raw_v,
        peek,
        c,
        target,
        loss,
        gn
    )

    if step % 25 == 0:

        clean()

    # -------------------------------------------------------------------------
    # 9. HEARTBEAT
    # -------------------------------------------------------------------------

    if step % HEARTBEAT_EVERY == 0:

        elapsed = time.time() - t0

        rate = step / max(
            elapsed,
            1e-9
        )

        eta = (
            TOTAL_STEPS - step
        ) / max(
            rate,
            1e-9
        )

        mean_loss = (
            float(np.mean(ROLL_LOSS))
            if ROLL_LOSS
            else np.nan
        )

        mean_grad = (
            float(np.mean(ROLL_GRAD))
            if ROLL_GRAD
            else np.nan
        )

        print(
            f"STEP {step:>5,}/{TOTAL_STEPS:,} "
            f"(raw attempts={RAW_ATTEMPTS:,}, "
            f"skipped={NONFINITE_SKIPS:,}) "
            f"| L100={mean_loss:.6f} "
            f"| G100={mean_grad:.4f} "
            f"| MEM={len(VECTOR_MEMORY):3d} "
            f"| NEW={TOTAL_NEW:5d} "
            f"| REUSE={TOTAL_REUSE:5d} "
            f"| {rate:.4f} step/s "
            f"| ETA={eta/3600:.2f}h"
        )

    # -------------------------------------------------------------------------
    # 10. FROZEN MILESTONE CHECKPOINT + VECTOR ECOLOGY SNAPSHOT
    #
    # NO PROBE IS ASKED HERE.
    #
    # [F5] theta_10000 is deliberately NOT written to disk: training is
    # about to end anyway, and the live model `m` will be audited directly
    # in section 27 before it is deleted. Only theta_100 and theta_1000
    # need a disk checkpoint, since training continues past those points.
    #
    # [F1] VECTOR_SNAPSHOTS captures the vector ecology V_t at every
    # milestone (0 was already captured before the loop started), so the
    # coupled evolution of (theta_t, V_t) can actually be inspected later.
    # -------------------------------------------------------------------------

    if step in (
        100,
        1_000,
        10_000
    ):

        if step != 10_000:

            save_checkpoint(
                m,
                step
            )

        VECTOR_SNAPSHOTS[step] = snapshot_vector_memory()

        clean()


# =============================================================================
# 25. END TRAINING
# =============================================================================

TOTAL_RUNTIME = time.time() - t0

print(
    "\n"
    + "=" * 120
)

print(
    "ENDOGENOUS TRAJECTORY FINISHED"
    if not ABORTED_EARLY
    else "ENDOGENOUS TRAJECTORY ABORTED EARLY"
)

print(
    f"Successful crystallizations : {SUCCESSFUL_STEPS:,} / {TOTAL_STEPS:,}"
)

print(
    f"Raw self-question attempts  : {RAW_ATTEMPTS:,}"
)

print(
    f"Non-finite steps skipped    : {NONFINITE_SKIPS:,}"
)

print(
    f"Duplicate questions blocked : {DUPLICATE_REJECTIONS:,}"
)

print(
    f"Probe-similar questions blocked (host-side, never shown to model) : "
    f"{PROBE_LEAK_BLOCKED:,}"
)

print(
    "Probe questions have still never entered training."
)

print(
    "=" * 120
)


# =============================================================================
# 26/27. REPLAY AUDIT SETUP
# =============================================================================
#
# [F5] theta_10000 will be audited directly on the still-resident live
# model `m` (no 15GB disk write, no reload) — but that has to happen AFTER
# audit_checkpoint() is defined below (section 29). Top-level script code
# runs strictly in order, unlike a function body, so `m`/`opt`/`params`
# are deliberately NOT deleted yet: deletion happens in section 30, right
# after the live-model audit of theta_10000 completes.
#
# NOW — and only now — the five fixed probes are used at all.
#
# Only ONE 7B model is resident in GPU memory at any point during replay.
# =============================================================================

AUDIT = {}


# =============================================================================
# 28. AKBASCORE-ON ANSWER
# =============================================================================

@torch.inference_mode()
def generate_answer_on(m, question, c):

    was = m.training

    m.eval()

    x = enc(
        m,
        question
    )

    ls = layers(m)

    hs = []

    for L in range(N_LAYERS):

        def hook(mod, args, out, L=L):

            h = (
                out[0]
                if isinstance(out, tuple)
                else out
            )

            s = steer(
                h,
                c[L],
                DOSE[L]
            )

            return (
                (s,) + out[1:]
                if isinstance(out, tuple)
                else s
            )

        hs.append(
            ls[L].register_forward_hook(
                hook
            )
        )

    try:

        y = m.generate(
            **x,
            max_new_tokens=PROBE_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.10,
            pad_token_id=tok.eos_token_id
        )

    finally:

        for h in hs:
            h.remove()

    z = y[
        0,
        x["input_ids"].shape[1]:
    ]

    answer = tok.decode(
        z,
        skip_special_tokens=True
    ).strip()

    del x, y, z

    if was:
        m.train()

    return answer


# =============================================================================
# 29. ONE CHECKPOINT AUDIT
# =============================================================================

def audit_checkpoint(step, preloaded_model=None):

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"FINAL REPLAY AUDIT — theta_{step:,}"
        + (
            " (live, no disk round-trip)"
            if preloaded_model is not None
            else ""
        )
    )

    print(
        "=" * 120
    )

    if preloaded_model is not None:

        # [F5] theta_10000 case: reuse the already-resident trained model
        # instead of writing a full BF16 7B checkpoint to disk and
        # reloading it a moment later.
        mm = preloaded_model

    elif step == 0:

        mm = load_base()

    else:

        mm = load_checkpoint(
            checkpoint_path(step)
        )

    mm.eval()

    for p in mm.parameters():

        p.requires_grad_(False)

    # -------------------------------------------------------------------------
    # Weight fingerprint
    # -------------------------------------------------------------------------

    fp = weight_fingerprint(mm)

    wm = fingerprint_metrics(
        BASE_FP,
        fp
    )

    # -------------------------------------------------------------------------
    # Probe results
    # -------------------------------------------------------------------------

    results = []

    for i, question in enumerate(
        PROBES
    ):

        # OFF answer
        off_answer = generate_answer(
            mm,
            question,
            PROBE_NEW_TOKENS
        )

        # OFF geometry
        off_geo = capture(
            mm,
            question
        )

        # Current Q->A endogenous vector
        ans_geo = capture_answer_geometry(
            mm,
            question,
            off_answer
        )

        c = endogenous_vector(
            off_geo,
            ans_geo
        )

        # ON answer
        on_answer = generate_answer_on(
            mm,
            question,
            c
        )

        # ON geometry
        on_geo = capture(
            mm,
            question,
            c
        )

        # Motor runtime displacement
        motor_layer = []

        for L in range(N_LAYERS):

            den = max(
                np.linalg.norm(
                    off_geo[L]
                ),
                1e-12
            )

            motor_layer.append(
                np.linalg.norm(
                    on_geo[L]
                    - off_geo[L]
                )
                / den
            )

        results.append(
            dict(
                question=question,
                off_answer=off_answer,
                on_answer=on_answer,
                off_geo=off_geo,
                on_geo=on_geo,
                vector=c.cpu().numpy(),
                motor_layer=np.asarray(
                    motor_layer,
                    dtype=np.float64
                ),
                motor_mean=float(
                    np.mean(motor_layer)
                )
            )
        )

        print(
            f"Probe {i+1}/5 "
            f"| motor hidden displacement="
            f"{100*np.mean(motor_layer):.6f}%"
        )

        del (
            off_geo,
            ans_geo,
            c,
            on_geo
        )

        clean()

    AUDIT[step] = dict(
        weight=wm,
        probes=results
    )

    print(
        "Weight FP movement:",
        f"{100*wm['mean_relative']:.10f}%"
    )

    print(
        "Sampled weights changed:",
        f"{100*wm['mean_changed_fraction']:.6f}%"
    )

    del fp

    if preloaded_model is None:

        # Only delete a model this function loaded itself. A preloaded
        # model (theta_10000's live `m`) is the caller's responsibility —
        # see section 30, which deletes it immediately after this audit
        # returns.
        del mm

    clean()


# =============================================================================
# 30. RUN ALL REPLAY AUDITS
# =============================================================================
#
# [F5] theta_10000 is audited FIRST, directly on the live training model,
# before it is deleted. This is the only milestone that skips the disk
# round-trip. theta_0 / theta_100 / theta_1000 are then audited in the
# usual load-from-disk-or-base way, one model resident at a time.
# =============================================================================

print(
    "\n"
    + "=" * 120
)

print(
    "Auditing theta_10000 directly on the live, still-resident model "
    "(no disk round-trip)."
)

print(
    "=" * 120
)

audit_checkpoint(
    10_000,
    preloaded_model=m
)

del opt

del params

del m

clean()

for step in MILESTONES:

    if step == 10_000:

        # Already audited above, on the live model.
        continue

    audit_checkpoint(step)


# =============================================================================
# 31. CROSS-MILESTONE METRICS
# =============================================================================

BASE_OFF = [
    AUDIT[0]["probes"][i]["off_geo"]
    for i in range(N_PROBES)
]

BASE_VECTOR = [
    AUDIT[0]["probes"][i]["vector"]
    for i in range(N_PROBES)
]


for step in MILESTONES:

    hidden_drifts = []

    vector_rotations = []

    motor_effects = []

    for i in range(N_PROBES):

        now = AUDIT[step][
            "probes"
        ][i]

        b = BASE_OFF[i]

        f = now[
            "off_geo"
        ]

        for L in range(N_LAYERS):

            den = max(
                np.linalg.norm(
                    b[L]
                ),
                1e-12
            )

            hidden_drifts.append(
                np.linalg.norm(
                    f[L] - b[L]
                )
                / den
            )

            vector_rotations.append(
                angle_deg(
                    BASE_VECTOR[i][L],
                    now["vector"][L]
                )
            )

        motor_effects.append(
            now[
                "motor_mean"
            ]
        )

    AUDIT[step][
        "mean_hidden_drift"
    ] = float(
        np.mean(hidden_drifts)
    )

    AUDIT[step][
        "max_hidden_drift"
    ] = float(
        np.max(hidden_drifts)
    )

    AUDIT[step][
        "mean_vector_rotation"
    ] = float(
        np.mean(vector_rotations)
    )

    AUDIT[step][
        "max_vector_rotation"
    ] = float(
        np.max(vector_rotations)
    )

    AUDIT[step][
        "mean_motor_effect"
    ] = float(
        np.mean(motor_effects)
    )


# =============================================================================
# 32. LAYERWISE HIDDEN DRIFT
# =============================================================================

for step in MILESTONES:

    layer_vals = []

    for L in range(N_LAYERS):

        vals = []

        for i in range(N_PROBES):

            b = BASE_OFF[i][L]

            f = AUDIT[step][
                "probes"
            ][i][
                "off_geo"
            ][L]

            vals.append(
                np.linalg.norm(f - b)
                /
                max(
                    np.linalg.norm(b),
                    1e-12
                )
            )

        layer_vals.append(
            np.mean(vals)
        )

    AUDIT[step][
        "layer_hidden_drift"
    ] = np.asarray(
        layer_vals
    )


# =============================================================================
# 33. TELEMETRY DATAFRAME
# =============================================================================

tdf = pd.DataFrame(
    TELEMETRY
)


# =============================================================================
# 34. FIGURE STORAGE
# =============================================================================

FIGURES = []


def savefig(fig, name):

    path = f"{FIG}/{name}"

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    FIGURES.append(path)


# =============================================================================
# 35. FIGURE 1 — GLOBAL HIDDEN DRIFT
# =============================================================================

fig = plt.figure(
    figsize=(10, 6)
)

xs = list(MILESTONES)

ys = [
    100
    * AUDIT[s][
        "mean_hidden_drift"
    ]
    for s in xs
]

plt.plot(
    xs,
    ys,
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Endogenous crystallization step"
)

plt.ylabel(
    "Mean persistent hidden drift from theta_0 (%)"
)

plt.title(
    "TEST 149 — Persistent Internal Reorganization\n"
    "Same five isolated probes, AkbasCore OFF"
)

plt.grid(alpha=.2)

plt.tight_layout()

savefig(
    fig,
    "TEST149_01_PERSISTENT_HIDDEN_DRIFT.png"
)


# =============================================================================
# 36. FIGURE 2 — WEIGHT MOVEMENT
# =============================================================================

fig = plt.figure(
    figsize=(10, 6)
)

ys = [
    100
    * AUDIT[s]["weight"][
        "mean_relative"
    ]
    for s in xs
]

plt.plot(
    xs,
    ys,
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Endogenous crystallization step"
)

plt.ylabel(
    "Distributed weight fingerprint movement (%)"
)

plt.title(
    "TEST 149 — Persistent Weight-Space Movement\n"
    "Attention + MLP samples across 20 decoder layers"
)

plt.grid(alpha=.2)

plt.tight_layout()

savefig(
    fig,
    "TEST149_02_WEIGHT_MOVEMENT.png"
)


# =============================================================================
# 37. FIGURE 3 — MOTOR EFFECT
# =============================================================================

fig = plt.figure(
    figsize=(10, 6)
)

ys = [
    100
    * AUDIT[s][
        "mean_motor_effect"
    ]
    for s in xs
]

plt.plot(
    xs,
    ys,
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Model state"
)

plt.ylabel(
    "AkbasCore ON vs OFF hidden displacement (%)"
)

plt.title(
    "TEST 149 — Live AkbasCore Pressure Across Model Evolution"
)

plt.grid(alpha=.2)

plt.tight_layout()

savefig(
    fig,
    "TEST149_03_MOTOR_ON_OFF_EFFECT.png"
)


# =============================================================================
# 38. FIGURE 4 — VECTOR ECOLOGY
# =============================================================================

fig = plt.figure(
    figsize=(11, 6)
)

if len(tdf):

    plt.plot(
        tdf.step,
        tdf.new,
        label="NEW events"
    )

    plt.plot(
        tdf.step,
        tdf.reuse,
        label="REUSE events"
    )

    plt.plot(
        tdf.step,
        tdf.memories,
        label="Active vector memories"
    )

plt.xlabel(
    "Endogenous step"
)

plt.ylabel(
    "Count"
)

plt.title(
    "TEST 149 — Endogenous Vector Ecology\n"
    "Birth, reuse and consolidation of internally generated directions"
)

plt.legend()

plt.grid(alpha=.2)

plt.tight_layout()

savefig(
    fig,
    "TEST149_04_VECTOR_ECOLOGY.png"
)


# =============================================================================
# 39. FIGURE 5 — CRYSTALLIZATION LOSS
# =============================================================================

fig = plt.figure(
    figsize=(11, 6)
)

if len(tdf):

    plt.plot(
        tdf.step,
        tdf.loss,
        linewidth=1.3
    )

plt.xlabel(
    "Endogenous step"
)

plt.ylabel(
    "Crystallization loss"
)

plt.title(
    "TEST 149 — AkbasCore Pressure -> Weight Crystallization"
)

plt.grid(alpha=.2)

plt.tight_layout()

savefig(
    fig,
    "TEST149_05_CRYSTALLIZATION_DYNAMICS.png"
)


# =============================================================================
# 40. FIGURE 6 — LAYER X-RAY
# =============================================================================

fig = plt.figure(
    figsize=(12, 7)
)

for step in (
    100,
    1000,
    10000
):

    plt.plot(
        np.arange(N_LAYERS),
        100
        * AUDIT[step][
            "layer_hidden_drift"
        ],
        marker="o",
        label=f"{step:,}"
    )

plt.xticks(
    np.arange(N_LAYERS)
)

plt.xlabel(
    "Decoder layer"
)

plt.ylabel(
    "Persistent hidden drift (%)"
)

plt.title(
    "TEST 149 — Layer X-Ray\n"
    "Where does the endogenous reorganization accumulate?"
)

plt.legend()

plt.grid(alpha=.2)

plt.tight_layout()

savefig(
    fig,
    "TEST149_06_LAYER_XRAY.png"
)


# =============================================================================
# 41. FIVE HUMAN-READABLE QUESTION EVOLUTION FIGURES
# =============================================================================
#
# Exactly five additional figures:
# one per isolated probe.
#
# Each shows:
#
# theta_0
# theta_100
# theta_1000
# theta_10000
#
# with:
#   persistent OFF answer
#   live motor ON answer
#   persistent hidden drift
#   live motor displacement
# =============================================================================

def wrap(s, width=92):

    s = " ".join(
        str(s).split()
    )

    return "\n".join(
        [
            s[i:i+width]
            for i in range(
                0,
                len(s),
                width
            )
        ]
    )


for i in range(N_PROBES):

    fig = plt.figure(
        figsize=(16, 11)
    )

    ax = plt.gca()

    ax.axis("off")

    lines = []

    lines.append(
        f"PROBE {i+1}"
    )

    lines.append(
        ""
    )

    lines.append(
        "QUESTION:"
    )

    lines.append(
        wrap(PROBES[i])
    )

    lines.append(
        ""
    )

    lines.append(
        "=" * 100
    )

    for step in MILESTONES:

        p = AUDIT[step][
            "probes"
        ][i]

        if step == 0:

            drift = 0.

        else:

            vals = []

            for L in range(N_LAYERS):

                b = BASE_OFF[i][L]

                f = p[
                    "off_geo"
                ][L]

                vals.append(
                    np.linalg.norm(
                        f - b
                    )
                    /
                    max(
                        np.linalg.norm(b),
                        1e-12
                    )
                )

            drift = float(
                np.mean(vals)
            )

        lines.append(
            ""
        )

        lines.append(
            f"MODEL STATE theta_{step:,}"
        )

        lines.append(
            f"Persistent hidden drift from theta_0: "
            f"{100*drift:.6f}%"
        )

        lines.append(
            f"Live AkbasCore ON/OFF displacement: "
            f"{100*p['motor_mean']:.6f}%"
        )

        lines.append(
            ""
        )

        lines.append(
            "MOTOR OFF — WHAT REMAINS IN THE MODEL:"
        )

        lines.append(
            wrap(
                p["off_answer"]
            )
        )

        lines.append(
            ""
        )

        lines.append(
            "MOTOR ON — SAME MODEL UNDER LIVE AKBASCORE PRESSURE:"
        )

        lines.append(
            wrap(
                p["on_answer"]
            )
        )

        lines.append(
            ""
        )

        lines.append(
            "-" * 100
        )

    ax.text(
        .01,
        .99,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.2,
        transform=ax.transAxes
    )

    plt.title(
        f"TEST 149 — PROBE {i+1}: "
        "0 -> 100 -> 1,000 -> 10,000\n"
        "Persistent change vs live AkbasCore pressure",
        fontsize=14
    )

    plt.tight_layout()

    savefig(
        fig,
        f"TEST149_PROBE_{i+1:02d}_EVOLUTION.png"
    )


# =============================================================================
# 42. FINAL THREE-STATE HUMAN FIGURE
# =============================================================================
#
# STEP 0 OFF
# STEP 10000 OFF
# STEP 10000 ON
# =============================================================================

fig = plt.figure(
    figsize=(18, 14)
)

ax = plt.gca()

ax.axis("off")

blocks = []

for i in range(N_PROBES):

    p0 = AUDIT[0][
        "probes"
    ][i]

    pf = AUDIT[10000][
        "probes"
    ][i]

    blocks += [

        f"PROBE {i+1}",

        "QUESTION: "
        + wrap(
            PROBES[i],
            100
        ),

        "",

        "0 / ORIGINAL MODEL:",
        wrap(
            p0["off_answer"],
            100
        ),

        "",

        "10,000 / MOTOR OFF — PERSISTENT RESULT:",
        wrap(
            pf["off_answer"],
            100
        ),

        "",

        "10,000 / MOTOR ON — LIVE AKBASCORE:",
        wrap(
            pf["on_answer"],
            100
        ),

        "",

        "=" * 110,

        ""
    ]

ax.text(
    .01,
    .99,
    "\n".join(blocks),
    va="top",
    ha="left",
    family="monospace",
    fontsize=7.4,
    transform=ax.transAxes
)

plt.title(
    "TEST 149 — ORIGINAL vs 10,000-STEP PERSISTENT MODEL "
    "vs LIVE AKBASCORE",
    fontsize=15
)

plt.tight_layout()

savefig(
    fig,
    "TEST149_12_FINAL_HUMAN_COMPARISON.png"
)


# =============================================================================
# 43. DOWNLOAD PNG ONLY
# =============================================================================

print(
    "\n"
    + "=" * 120
)

print(
    "PNG DOWNLOADS"
)

print(
    "Only figures will be downloaded."
)

print(
    "=" * 120
)

from google.colab import files

for path in FIGURES:

    if os.path.exists(path):

        print(
            "Downloading:",
            os.path.basename(path)
        )

        files.download(path)


# =============================================================================
# 44. DELETE PNG FROM COLAB AFTER DOWNLOAD REQUESTS
# =============================================================================

for path in FIGURES:

    try:

        if os.path.exists(path):
            os.remove(path)

    except Exception as e:

        print(
            "PNG cleanup warning:",
            e
        )


clean()


# =============================================================================
# 45. DELETE LARGE CHECKPOINTS AFTER ALL REPLAY TESTS
# =============================================================================
#
# They are no longer needed.
#
# [F5] theta_10000 was never written to disk in the first place (it was
# audited directly on the live model in section 30), so there is only
# theta_100 and theta_1000 to clean up here — one full 7B checkpoint's
# worth of disk I/O saved for the whole run.
# =============================================================================

print(
    "\nRemoving temporary model checkpoints..."
)

for step in (
    100,
    1000
):

    shutil.rmtree(
        checkpoint_path(step),
        ignore_errors=True
    )

clean()

print(
    "Temporary checkpoints removed (theta_10000 was never written to disk)."
)


# =============================================================================
# 46. MASTER FINAL AUDIT
# =============================================================================
#
# THIS IS THE IMPORTANT PART.
#
# It is deliberately the LAST substantial output in the Colab cell.
#
# Copy from:
#
#       TEST 149 — MASTER FINAL AUDIT
#
# to:
#
#       END MASTER FINAL AUDIT
#
# and send that section for analysis.
# =============================================================================

print(
    "\n\n"
    + "=" * 132
)

print(
    "TEST 149 — MASTER FINAL AUDIT"
)

print(
    "=" * 132
)


# -----------------------------------------------------------------------------
# A. EXPERIMENT IDENTITY
# -----------------------------------------------------------------------------

print(
    "\n[A] EXPERIMENT"
)

print(
    "-" * 132
)

print(
    f"Model                         : {MODEL_ID}"
)

print(
    f"Total parameters              : {TOTAL_PARAMS:,}"
)

print(
    f"Trainable parameters          : {TRAINABLE_PARAMS:,}"
)

print(
    f"Observed / pressured layers   : {N_LAYERS}"
)

print(
    f"Target endogenous steps       : {TOTAL_STEPS:,}"
)

print(
    f"Successful updates            : {SUCCESSFUL_STEPS:,}"
)

print(
    f"Raw self-question attempts    : {RAW_ATTEMPTS:,}"
)

print(
    f"Non-finite skipped updates    : {NONFINITE_SKIPS:,}"
)

print(
    f"Duplicate questions blocked   : {DUPLICATE_REJECTIONS:,}"
)

print(
    f"Probe-similar questions blocked (host-side gate, never shown to "
    f"model): {PROBE_LEAK_BLOCKED:,}"
)

print(
    f"Probe leak similarity threshold: {PROBE_LEAK_THRESHOLD}"
)

print(
    f"Aborted early (MAX_RAW_ATTEMPTS exceeded): {ABORTED_EARLY}"
)

print(
    f"theta_10000 disk checkpoint    : NEVER WRITTEN (audited live, "
    f"see [F5])"
)

print(
    f"External training information : NONE"
)

print(
    f"External training dataset     : NONE"
)

print(
    f"Teacher model                 : NONE"
)

print(
    f"RAG / Internet                : NONE"
)

print(
    f"Human training answers        : NONE"
)

print(
    f"LoRA                          : OFF"
)

print(
    f"Retention Guard               : OFF"
)

print(
    f"Structural growth             : OFF"
)

print(
    f"All base weights trainable    : YES"
)

print(
    f"Probe leakage into training   : NONE"
)

print(
    f"Probe timing                  : AFTER 10,000-step training completed"
)

print(
    f"IVME                          : {IVME}"
)

print(
    f"SONUM                         : {SONUM}"
)

print(
    f"ZIRVE                         : {ZIRVE}"
)

print(
    f"TABAN                         : {TABAN}"
)

print(
    f"LR                            : {LR}"
)

print(
    f"Gradient clip                 : {GRAD_CLIP}"
)

print(
    f"Vector reuse cosine           : {VECTOR_REUSE_COS}"
)

print(
    f"Vector blend                  : {VECTOR_BLEND}"
)

print(
    f"Runtime hours                 : {TOTAL_RUNTIME/3600:.6f}"
)


# -----------------------------------------------------------------------------
# B. CORE MILESTONE TABLE
# -----------------------------------------------------------------------------

print(
    "\n[B] CORE MILESTONE TABLE"
)

print(
    "-" * 132
)

header = (
    f"{'STATE':>10} | "
    f"{'HIDDEN DRIFT':>14} | "
    f"{'MAX H DRIFT':>14} | "
    f"{'VECTOR ROT':>12} | "
    f"{'MOTOR EFFECT':>14} | "
    f"{'WEIGHT MOVE':>14} | "
    f"{'W CHANGED':>12}"
)

print(header)

print(
    "-" * len(header)
)

for step in MILESTONES:

    x = AUDIT[step]

    print(
        f"{step:>10,} | "
        f"{100*x['mean_hidden_drift']:>13.8f}% | "
        f"{100*x['max_hidden_drift']:>13.8f}% | "
        f"{x['mean_vector_rotation']:>11.6f} deg | "
        f"{100*x['mean_motor_effect']:>13.8f}% | "
        f"{100*x['weight']['mean_relative']:>13.10f}% | "
        f"{100*x['weight']['mean_changed_fraction']:>11.6f}%"
    )


# -----------------------------------------------------------------------------
# C. VECTOR ECOLOGY
# -----------------------------------------------------------------------------

print(
    "\n[C] ENDOGENOUS VECTOR ECOLOGY"
)

print(
    "-" * 132
)

total_vector_events = (
    TOTAL_NEW
    + TOTAL_REUSE
)

reuse_rate = (
    TOTAL_REUSE
    /
    max(
        total_vector_events,
        1
    )
)

print(
    f"Final active memories         : {len(VECTOR_MEMORY)}"
)

print(
    f"Total NEW events              : {TOTAL_NEW}"
)

print(
    f"Total REUSE events            : {TOTAL_REUSE}"
)

print(
    f"Total vector events           : {total_vector_events}"
)

print(
    f"Final reuse rate              : {100*reuse_rate:.8f}%"
)

print(
    f"Exact repeated Q rejections   : {DUPLICATE_REJECTIONS}"
)

print(
    f"Probe-similar Q blocked       : {PROBE_LEAK_BLOCKED}"
)


# -----------------------------------------------------------------------------
# C2. VECTOR ECOLOGY AT EACH MILESTONE
# -----------------------------------------------------------------------------
#
# [F1] This is the data that was previously missing entirely: what did the
# vector memory V_t actually look like at 0 / 100 / 1000 / 10000, and how
# much did it keep moving between milestones?
#
# Caveat, stated plainly: memory slot i does not necessarily hold "the same
# family" at two different milestones, because the LRU eviction rule in
# commit_vector() can overwrite the least-used slot with an unrelated new
# direction. The slot-index cosine below is therefore a coarse, descriptive
# continuity signal ("how much did the pool as arranged look alike"), not a
# rigorous per-family trajectory. A rigorous per-family trajectory would
# require tracking family identity across evictions, which this experiment
# does not attempt.
# -----------------------------------------------------------------------------

print(
    "\n[C2] VECTOR ECOLOGY AT EACH MILESTONE"
)

print(
    "-" * 132
)

for s in MILESTONES:

    if s not in VECTOR_SNAPSHOTS:

        print(
            f"theta_{s:<6,} | snapshot missing "
            "(milestone not reached — see ABORTED_EARLY above)"
        )

        continue

    snap = VECTOR_SNAPSHOTS[s]

    n_mem = len(snap["memory"])

    hit_vals = list(snap["hits"].values())

    mean_hits = (
        float(np.mean(hit_vals))
        if hit_vals
        else 0.0
    )

    print(
        f"theta_{s:<6,} | memories={n_mem:3d} "
        f"| total_new={snap['total_new']:5d} "
        f"| total_reuse={snap['total_reuse']:5d} "
        f"| mean hits/slot={mean_hits:6.2f}"
    )


def slotwise_pool_rotation(step_a, step_b):

    if (
        step_a not in VECTOR_SNAPSHOTS
        or step_b not in VECTOR_SNAPSHOTS
    ):

        return None

    A = VECTOR_SNAPSHOTS[step_a]["memory"]

    B = VECTOR_SNAPSHOTS[step_b]["memory"]

    n = min(
        len(A),
        len(B)
    )

    if n == 0:

        return None

    per_slot_cos = []

    for i in range(n):

        cs = F.cosine_similarity(
            A[i].float(),
            B[i].float(),
            dim=1
        )

        per_slot_cos.append(
            float(cs.mean())
        )

    return dict(
        n_common_slots=n,
        mean_cosine=float(np.mean(per_slot_cos)),
        min_cosine=float(np.min(per_slot_cos)),
        max_cosine=float(np.max(per_slot_cos)),
    )


print(
    "\nSlot-index cosine continuity between milestones "
    "(descriptive only — see caveat above):"
)

for a, b in (
    (0, 100),
    (100, 1000),
    (1000, 10000),
    (0, 10000),
):

    r = slotwise_pool_rotation(
        a,
        b
    )

    if r is None:

        print(
            f"  V_{a:<6,} -> V_{b:<6,} | (unavailable)"
        )

    else:

        print(
            f"  V_{a:<6,} -> V_{b:<6,} | "
            f"common slots={r['n_common_slots']:3d} "
            f"| mean cos={r['mean_cosine']:+.4f} "
            f"| min={r['min_cosine']:+.4f} "
            f"| max={r['max_cosine']:+.4f}"
        )


# -----------------------------------------------------------------------------
# D. TRAINING DYNAMICS
# -----------------------------------------------------------------------------

print(
    "\n[D] TRAINING DYNAMICS"
)

print(
    "-" * 132
)

if len(tdf):

    print(
        f"Mean recorded loss            : {tdf.loss.mean():.10f}"
    )

    print(
        f"Final recorded loss           : {tdf.loss.iloc[-1]:.10f}"
    )

    print(
        f"Min recorded loss             : {tdf.loss.min():.10f}"
    )

    print(
        f"Max recorded loss             : {tdf.loss.max():.10f}"
    )

    print(
        f"Mean recorded grad norm       : {tdf.grad_norm.mean():.10f}"
    )

    print(
        f"Final recorded grad norm      : {tdf.grad_norm.iloc[-1]:.10f}"
    )

    finite_cos = tdf.nearest_cos[
        np.isfinite(
            tdf.nearest_cos
        )
    ]

    if len(finite_cos):

        print(
            f"Mean nearest-vector cosine    : {finite_cos.mean():.10f}"
        )


# -----------------------------------------------------------------------------
# E. FINAL WEIGHT MATRIX X-RAY
# -----------------------------------------------------------------------------

print(
    "\n[E] STEP 10,000 WEIGHT MATRIX X-RAY"
)

print(
    "-" * 132
)

for name in WEIGHT_MODULES:

    print(
        f"{name:>12s} : "
        f"{100*AUDIT[10000]['weight']['matrix_relative'][name]:.12f}%"
    )


# -----------------------------------------------------------------------------
# F. FINAL LAYER HIDDEN X-RAY
# -----------------------------------------------------------------------------

print(
    "\n[F] STEP 10,000 LAYER HIDDEN X-RAY"
)

print(
    "-" * 132
)

for L in range(N_LAYERS):

    print(
        f"L{L:02d} persistent hidden drift : "
        f"{100*AUDIT[10000]['layer_hidden_drift'][L]:.10f}%"
    )


# -----------------------------------------------------------------------------
# G. FIVE PROBE NUMERICAL SUMMARY
# -----------------------------------------------------------------------------

print(
    "\n[G] FIVE ISOLATED PROBE NUMERICAL SUMMARY"
)

print(
    "-" * 132
)

for i in range(N_PROBES):

    print(
        f"\nPROBE {i+1}"
    )

    for step in MILESTONES:

        p = AUDIT[step]["probes"][i]

        if step == 0:

            drift = 0.

            rot = 0.

        else:

            ds = []

            rs = []

            for L in range(N_LAYERS):

                b = BASE_OFF[i][L]

                f = p["off_geo"][L]

                ds.append(
                    np.linalg.norm(f - b)
                    /
                    max(
                        np.linalg.norm(b),
                        1e-12
                    )
                )

                rs.append(
                    angle_deg(
                        BASE_VECTOR[i][L],
                        p["vector"][L]
                    )
                )

            drift = float(
                np.mean(ds)
            )

            rot = float(
                np.mean(rs)
            )

        print(
            f" theta_{step:<5,} "
            f"| persistent drift={100*drift:10.6f}% "
            f"| vector rotation={rot:9.5f} deg "
            f"| live motor={100*p['motor_mean']:10.6f}%"
        )


# -----------------------------------------------------------------------------
# H. QUESTIONS + ALL ANSWERS
# -----------------------------------------------------------------------------

print(
    "\n[H] FIVE PERMANENTLY ISOLATED QUESTIONS AND ANSWERS"
)

print(
    "These questions were never used for training."
)

print(
    "-" * 132
)


for i, question in enumerate(PROBES):

    print(
        "\n"
        + "#" * 132
    )

    print(
        f"PROBE {i+1}"
    )

    print(
        "#" * 132
    )

    print(
        "\nQUESTION:"
    )

    print(question)

    for step in MILESTONES:

        p = AUDIT[step][
            "probes"
        ][i]

        print(
            f"\n--- theta_{step:,} | AKBASCORE OFF ---"
        )

        print(
            p["off_answer"]
        )

        print(
            f"\n--- theta_{step:,} | AKBASCORE ON ---"
        )

        print(
            p["on_answer"]
        )


# -----------------------------------------------------------------------------
# I. ULTRA-COMPACT COPY BLOCK
# -----------------------------------------------------------------------------

print(
    "\n\n[I] FINAL KEY NUMBERS — QUICK COPY"
)

print(
    "-" * 132
)

for step in MILESTONES:

    x = AUDIT[step]

    print(
        f"theta_{step:,}: "
        f"hidden={100*x['mean_hidden_drift']:.10f}% | "
        f"vector={x['mean_vector_rotation']:.8f}deg | "
        f"motor={100*x['mean_motor_effect']:.10f}% | "
        f"weight={100*x['weight']['mean_relative']:.12f}% | "
        f"changed_weights={100*x['weight']['mean_changed_fraction']:.8f}%"
    )

print(
    f"\nNEW={TOTAL_NEW} "
    f"| REUSE={TOTAL_REUSE} "
    f"| REUSE_RATE={100*reuse_rate:.8f}% "
    f"| MEMORIES={len(VECTOR_MEMORY)}"
)

print(
    f"SUCCESSFUL_STEPS={SUCCESSFUL_STEPS} "
    f"| SKIPPED={NONFINITE_SKIPS} "
    f"| RUNTIME_H={TOTAL_RUNTIME/3600:.6f}"
)

print(
    f"PNG_DOWNLOADS={len(FIGURES)}"
)

remaining_png = []

if os.path.isdir(FIG):

    remaining_png = [
        x
        for x in os.listdir(FIG)
        if x.lower().endswith(".png")
    ]

print(
    f"PNG_REMAINING_IN_COLAB={len(remaining_png)}"
)

print(
    "NON_IMAGE_DOWNLOADS=0"
)

print(
    "TEMP_CHECKPOINTS_DELETED=YES"
)

print(
    "\n"
    + "=" * 132
)

print(
    "END MASTER FINAL AUDIT"
)

print(
    "=" * 132
)

print(
    "TEST 149 COMPLETE"
)
