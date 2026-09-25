# =============================================================================
# TEST 155 — TRANSPORT-AWARE STEERING TOURNAMENT
# STATIC vs LAYER-COMPASS vs SHUFFLED-RELAY vs RELAY vs SUBSPACE
#
# Qwen/Qwen2.5-7B-Instruct · Frozen · BF16
#
# LOCKED TARGET:
#   ABOVE <-> BELOW
#
# PIPELINE:
#   SEARCH -> VAL TOURNAMENT -> LOCK -> UNTOUCHED FINAL
#
# FINAL:
#   ID English
#   OOD Topic English
#   OOD Long English
#   OOD Turkish
#
# CONTROLS:
#   BASE
#   STATIC
#   LAYER-COMPASS
#   SHUFFLED-RELAY
#   RELAY-K4
#   RELAY-K16
#   RELAY-K32
#   SUBSPACE-K16
#
# PRIMARY METRICS:
#   signed concept log-prob margin
#   non-target ΔNLL
#   non-target KL
#   requested intervention energy
#   realized BF16 intervention energy
#
# No training · No LoRA · No gradients · No JVP/VJP
# =============================================================================

import os, sys, subprocess, math, json, csv, gc, warnings, random
from pathlib import Path

for pkg, mod in [
    ("transformers>=4.46.0", "transformers"),
    ("accelerate", "accelerate"),
]:
    try:
        __import__(mod)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# CONFIG
# =============================================================================

SEED = 155
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

LAYERS = [3, 6, 10, 14, 19]
SEGMENTS = [(3, 6), (6, 10), (10, 14), (14, 19)]

K_VALUES = [4, 16, 32]
SUBSPACE_K = 16

SEARCH_N = 192
VAL_N = 32
FINAL_N = 32

# Relative write magnitude at each intervention layer.
# VAL selects the best dose for each method.
DOSES = [0.00125, 0.00250, 0.00500]

# Candidate methods.
METHODS = [
    "BASE",
    "STATIC",
    "LAYER",
    "SHUFFLED",
    "RELAY_K4",
    "RELAY_K16",
    "RELAY_K32",
    "SUBSPACE_K16",
]

# VAL selection constraints.
# We do NOT collapse everything into an arbitrary weighted score.
# First maximize concept gain among configurations whose collateral
# remains within these pre-locked bounds.
MAX_DNLL = 0.12
MAX_KL = 0.12

# Capability gate:
# BASE must correctly prefer the evidence-supported answer on at least
# this fraction of paired prompts.
CAPABILITY_THRESHOLD = 0.70

EPS = 1e-10

ROOT = Path("/content/TEST155_TRANSPORT_AWARE_TOURNAMENT")
ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    raise RuntimeError("TEST155 requires CUDA.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 150)
print("TEST 155 — TRANSPORT-AWARE STEERING TOURNAMENT")
print("STATIC · LAYER-COMPASS · SHUFFLED-RELAY · RELAY · SUBSPACE")
print("=" * 150)

# =============================================================================
# MODEL
# =============================================================================

print("\n[1/18] Loading frozen model...")

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
print("K values       :", K_VALUES)
print("Doses          :", DOSES)

# =============================================================================
# CHAT / TOKEN HELPERS
# =============================================================================

SYSTEM_EN = "You are a concise reasoning assistant."
SYSTEM_TR = "Kısa ve doğru yanıt veren bir akıl yürütme asistanısın."

def chat_text(user_text, system_text):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def encode_prompt(user_text, lang="EN"):
    system = SYSTEM_TR if lang == "TR" else SYSTEM_EN
    text = chat_text(user_text, system)
    tok = tokenizer(text, return_tensors="pt").to(DEVICE)
    pos = int(tok["attention_mask"][0].sum().item()) - 1
    return tok, pos

def continuation_ids(answer):
    ids = tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].to(DEVICE)
    return ids

# =============================================================================
# LOCKED ABOVE / BELOW COMPASS SENTENCES
# =============================================================================

POS_EN = [
    "The red marker is above the blue marker.",
    "The lamp is above the wooden table.",
    "The bird is above the branch.",
    "The sign is above the doorway.",
    "The shelf is above the cabinet.",
    "The cloud is above the hill.",
    "The clock is above the desk.",
    "The picture is above the sofa.",
]

NEG_EN = [
    "The red marker is below the blue marker.",
    "The lamp is below the wooden table.",
    "The bird is below the branch.",
    "The sign is below the doorway.",
    "The shelf is below the cabinet.",
    "The cloud is below the hill.",
    "The clock is below the desk.",
    "The picture is below the sofa.",
]

POS_TR = [
    "Kırmızı işaret mavi işaretin üstündedir.",
    "Lamba ahşap masanın üstündedir.",
    "Kuş dalın üstündedir.",
    "Tabela kapının üstündedir.",
    "Raf dolabın üstündedir.",
    "Bulut tepenin üstündedir.",
    "Saat masanın üstündedir.",
    "Resim kanepenin üstündedir.",
]

NEG_TR = [
    "Kırmızı işaret mavi işaretin altındadır.",
    "Lamba ahşap masanın altındadır.",
    "Kuş dalın altındadır.",
    "Tabela kapının altındadır.",
    "Raf dolabın altındadır.",
    "Bulut tepenin altındadır.",
    "Saat masanın altındadır.",
    "Resim kanepenin altındadır.",
]

# =============================================================================
# NATURAL SEARCH CORPUS
# Same controlled family used for transport discovery.
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

VERBS = ["rests", "sits", "remains", "stands"]

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

rng = np.random.default_rng(42)
rng.shuffle(natural_pool)
natural_pool = list(dict.fromkeys(natural_pool))
SEARCH_TEXTS = natural_pool[:SEARCH_N]

assert len(SEARCH_TEXTS) == SEARCH_N

# =============================================================================
# SAMPLE-SPACE CROSS-COVARIANCE SVD
#
# X^T = Qx Rx
# Y^T = Qy Ry
#
# X^T Y/(N-1)
#   = Qx [Rx Ry^T/(N-1)] Qy^T
#
# Only the N x N middle matrix is decomposed.
# =============================================================================

def cross_cov_svd(X_cpu, Y_cpu):
    X = X_cpu.to(DEVICE, dtype=torch.float32)
    Y = Y_cpu.to(DEVICE, dtype=torch.float32)

    n = X.shape[0]

    Qx, Rx = torch.linalg.qr(X.T, mode="reduced")
    Qy, Ry = torch.linalg.qr(Y.T, mode="reduced")

    small = (Rx @ Ry.T) / float(n - 1)

    Uk, S, Vhk = torch.linalg.svd(
        small,
        full_matrices=False,
    )

    U = Qx @ Uk
    V = Qy @ Vhk.T

    return U.detach(), S.detach(), V.detach()

# =============================================================================
# ACTIVATION CAPTURE
# =============================================================================

@torch.inference_mode()
def capture_layers(text, lang="EN"):
    tok, pos = encode_prompt(text, lang)

    out = model(
        **tok,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    row = {
        L: out.hidden_states[L + 1][0, pos].float().cpu()
        for L in LAYERS
    }

    del out, tok
    return row

def capture_bank(texts, lang="EN", label="BANK"):
    bank = {L: [] for L in LAYERS}

    for i, text in enumerate(texts, 1):
        row = capture_layers(text, lang)

        for L in LAYERS:
            bank[L].append(row[L])

        if i == 1 or i % 32 == 0 or i == len(texts):
            print(f"  {label}: {i:03d}/{len(texts)}")

    return {
        L: torch.stack(bank[L]).float()
        for L in LAYERS
    }

# =============================================================================
# SEARCH NATURAL ATLAS
# =============================================================================

print("\n[2/18] Capturing natural SEARCH atlas...")

SEARCH = capture_bank(
    SEARCH_TEXTS,
    lang="EN",
    label="SEARCH",
)

MEAN = {
    L: SEARCH[L].mean(dim=0, keepdim=True)
    for L in LAYERS
}

CENTERED = {
    L: SEARCH[L] - MEAN[L]
    for L in LAYERS
}

print("\n[3/18] Building transport atlas...")

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
        f"s16={S[15].item():.6f} "
        f"s32={S[31].item():.6f}"
    )

# =============================================================================
# COMPASS EXTRACTION
# =============================================================================

print("\n[4/18] Extracting locked ABOVE/BELOW layer compasses...")

POS_BANK_EN = capture_bank(POS_EN, "EN", "POS-EN")
NEG_BANK_EN = capture_bank(NEG_EN, "EN", "NEG-EN")

POS_BANK_TR = capture_bank(POS_TR, "TR", "POS-TR")
NEG_BANK_TR = capture_bank(NEG_TR, "TR", "NEG-TR")

COMPASS_EN = {}
COMPASS_TR = {}

for L in LAYERS:
    a = (
        POS_BANK_EN[L].mean(dim=0)
        - NEG_BANK_EN[L].mean(dim=0)
    ).to(DEVICE)

    a = a / (a.norm() + EPS)
    COMPASS_EN[L] = a

    t = (
        POS_BANK_TR[L].mean(dim=0)
        - NEG_BANK_TR[L].mean(dim=0)
    ).to(DEVICE)

    t = t / (t.norm() + EPS)
    COMPASS_TR[L] = t

    print(
        f"L{L:02d} | "
        f"cos(EN,TR)="
        f"{F.cosine_similarity(a[None], t[None]).item():+.6f}"
    )

# =============================================================================
# TRANSPORT OPERATORS
# =============================================================================

def transport(z, pair, K, weighted=False):
    U = ATLAS[pair]["U"][:, :K]
    V = ATLAS[pair]["V"][:, :K]

    c = U.T @ z

    if weighted:
        s = ATLAS[pair]["S"][:K]
        w = s / (s[0] + EPS)
        c = c * w

    y = V @ c

    if y.norm() < 1e-8:
        return y

    return y / (y.norm() + EPS)

def relay_chain(K, weighted=False):
    dirs = {3: COMPASS_EN[3]}

    current = COMPASS_EN[3]

    for pair in SEGMENTS:
        a, b = pair

        if a != 3:
            current = dirs[a]

        nxt = transport(
            current,
            pair,
            K,
            weighted=weighted,
        )

        # Fallback only if projection is numerically empty.
        if nxt.norm() < 1e-8:
            nxt = COMPASS_EN[b]

        dirs[b] = nxt / (nxt.norm() + EPS)

    return dirs

RELAY_DIRS = {
    K: relay_chain(K, weighted=False)
    for K in K_VALUES
}

# Weighted variants are candidates only during VAL.
RELAY_WEIGHTED_DIRS = {
    K: relay_chain(K, weighted=True)
    for K in K_VALUES
}

# =============================================================================
# SHUFFLED RELAY
#
# Correct transport machinery, wrong transition assignment.
# We keep source/target dimensionality identical but intentionally use
# the wrong segment operator and then map back through its target basis.
# =============================================================================

SHUFFLE_MAP = {
    3: (10, 14),
    6: (14, 19),
    10: (3, 6),
    14: (6, 10),
}

def shuffled_dirs(K=16):
    dirs = {3: COMPASS_EN[3]}
    current = COMPASS_EN[3]

    for a, b in SEGMENTS:
        wrong_pair = SHUFFLE_MAP[a]

        U = ATLAS[wrong_pair]["U"][:, :K]
        V = ATLAS[wrong_pair]["V"][:, :K]

        c = U.T @ current
        y = V @ c

        if y.norm() < 1e-8:
            y = COMPASS_EN[b]

        y = y / (y.norm() + EPS)

        dirs[b] = y
        current = y

    return dirs

SHUFFLED_DIRS = shuffled_dirs(16)

# =============================================================================
# SUBSPACE-K16
#
# Concept coefficients are represented in the source transport basis and
# moved into the target response basis. Unlike RELAY, coefficient vector
# magnitude/structure is retained rather than collapsing to a fresh 1-D
# normalized concept direction before every hand-off.
# =============================================================================

def subspace_chain(K=16):
    dirs = {3: COMPASS_EN[3]}
    coeffs = {}

    z = COMPASS_EN[3]

    for pair in SEGMENTS:
        a, b = pair

        U = ATLAS[pair]["U"][:, :K]
        V = ATLAS[pair]["V"][:, :K]

        c = U.T @ z
        coeffs[b] = c.detach()

        y = V @ c

        if y.norm() < 1e-8:
            y = COMPASS_EN[b]

        dirs[b] = y / (y.norm() + EPS)
        z = dirs[b]

    return dirs, coeffs

SUBSPACE_DIRS, SUBSPACE_COEFFS = subspace_chain(SUBSPACE_K)

# =============================================================================
# METHOD DIRECTION TABLE
# =============================================================================

def method_dirs(method, weighted=False):
    if method == "STATIC":
        return {L: COMPASS_EN[3] for L in LAYERS}

    if method == "LAYER":
        return {L: COMPASS_EN[L] for L in LAYERS}

    if method == "SHUFFLED":
        return SHUFFLED_DIRS

    if method.startswith("RELAY_K"):
        K = int(method.split("K")[-1])

        if weighted:
            return RELAY_WEIGHTED_DIRS[K]

        return RELAY_DIRS[K]

    if method == "SUBSPACE_K16":
        return SUBSPACE_DIRS

    if method == "BASE":
        return {}

    raise ValueError(method)

# =============================================================================
# BEHAVIOR DATA
#
# Each item contains explicit evidence and two candidate continuations.
# label = +1 means ABOVE is supported.
# label = -1 means BELOW is supported.
#
# Concept score:
#   signed [ logP(ABOVE answer) - logP(BELOW answer) ]
#
# This avoids keyword counting.
# =============================================================================

OBJECT_PAIRS_EN = [
    ("red marker", "blue marker"),
    ("silver coin", "black card"),
    ("green cube", "white cube"),
    ("small lamp", "wooden box"),
    ("paper label", "metal plate"),
    ("yellow disk", "gray disk"),
    ("orange tile", "purple tile"),
    ("glass bead", "plastic bead"),
]

OBJECT_PAIRS_TR = [
    ("kırmızı işaret", "mavi işaret"),
    ("gümüş para", "siyah kart"),
    ("yeşil küp", "beyaz küp"),
    ("küçük lamba", "ahşap kutu"),
    ("kağıt etiket", "metal plaka"),
    ("sarı disk", "gri disk"),
    ("turuncu karo", "mor karo"),
    ("cam boncuk", "plastik boncuk"),
]

TOPICS = [
    "inventory check",
    "laboratory note",
    "storage record",
    "museum catalog",
    "workshop report",
    "classroom observation",
    "shipping note",
    "inspection sheet",
]

def build_en_items(n, split, long=False, topic=False, seed=0):
    rr = random.Random(seed)
    items = []

    for i in range(n):
        a, b = OBJECT_PAIRS_EN[i % len(OBJECT_PAIRS_EN)]
        label = 1 if i % 2 == 0 else -1

        relation = "above" if label == 1 else "below"

        if topic:
            prefix = (
                f"This is part of a {TOPICS[i % len(TOPICS)]}. "
                f"Only the explicitly stated spatial relation is relevant. "
            )
        else:
            prefix = ""

        if long:
            filler = (
                "Several unrelated objects are also present. "
                "Their colors, materials, ownership, age, and position in the "
                "room do not alter the stated relation. "
                "The record contains descriptive details that should not be "
                "used to infer a different spatial arrangement. "
            ) * 3
        else:
            filler = ""

        prompt = (
            prefix
            + filler
            + f"Evidence: the {a} is {relation} the {b}. "
              f"Question: where is the {a} relative to the {b}? "
              f"Answer with the relation only."
        )

        items.append({
            "split": split,
            "lang": "EN",
            "prompt": prompt,
            "above": " above",
            "below": " below",
            "label": label,
        })

    rr.shuffle(items)
    return items

def build_tr_items(n, split, seed=0):
    rr = random.Random(seed)
    items = []

    for i in range(n):
        a, b = OBJECT_PAIRS_TR[i % len(OBJECT_PAIRS_TR)]
        label = 1 if i % 2 == 0 else -1

        relation = "üstündedir" if label == 1 else "altındadır"

        prompt = (
            f"Kanıt: {a}, {b} nesnesinin {relation}. "
            f"Soru: {a}, {b} nesnesine göre nerededir? "
            f"Yalnızca ilişkiyi yaz."
        )

        items.append({
            "split": split,
            "lang": "TR",
            "prompt": prompt,
            "above": " üstünde",
            "below": " altında",
            "label": label,
        })

    rr.shuffle(items)
    return items

VAL_ITEMS = build_en_items(
    VAL_N,
    "VAL",
    seed=1551,
)

FINAL_ID = build_en_items(
    FINAL_N,
    "FINAL_ID",
    seed=1552,
)

FINAL_TOPIC = build_en_items(
    FINAL_N,
    "FINAL_TOPIC",
    topic=True,
    seed=1553,
)

FINAL_LONG = build_en_items(
    FINAL_N,
    "FINAL_LONG",
    long=True,
    topic=True,
    seed=1554,
)

FINAL_TR = build_tr_items(
    FINAL_N,
    "FINAL_TR",
    seed=1555,
)

# =============================================================================
# MULTI-LAYER INTERVENTION HOOKS
#
# requested write:
#   u_L = dose * ||h_L|| * direction_L
#
# BF16 realized write is measured after cast:
#   delta_realized = h_written(BF16->FP32) - h_natural(FP32)
# =============================================================================

class SteeringHooks:
    def __init__(self, dirs, dose):
        self.dirs = dirs
        self.dose = dose
        self.handles = []
        self.requested_energy = 0.0
        self.realized_energy = 0.0
        self.writes = []

    def install(self):
        if not self.dirs or self.dose == 0:
            return

        for L in LAYERS:
            if L not in self.dirs:
                continue

            direction = self.dirs[L].to(
                DEVICE,
                dtype=torch.float32,
            )

            direction = direction / (
                direction.norm() + EPS
            )

            def make_hook(layer_id, d):
                def hook(module, inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output

                    # Steering all token positions preserves the same concept
                    # direction across the current sequence.
                    natural = hidden.float()

                    norms = natural.norm(
                        dim=-1,
                        keepdim=True,
                    )

                    requested = (
                        self.dose
                        * norms
                        * d.view(1, 1, -1)
                    )

                    modified_fp32 = natural + requested
                    modified = modified_fp32.to(hidden.dtype)

                    realized = modified.float() - natural

                    req_e = float(
                        requested.square().sum().item()
                    )

                    real_e = float(
                        realized.square().sum().item()
                    )

                    self.requested_energy += req_e
                    self.realized_energy += real_e

                    self.writes.append({
                        "layer": layer_id,
                        "requested_energy": req_e,
                        "realized_energy": real_e,
                    })

                    if isinstance(output, tuple):
                        return (modified, *output[1:])

                    return modified

                return hook

            self.handles.append(
                blocks[L].register_forward_hook(
                    make_hook(L, direction)
                )
            )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

# =============================================================================
# TEACHER-FORCED CONTINUATION LOG-PROB
#
# Returns:
#   total log probability
#   mean NLL
#   token-level baseline/steered distributions can be compared separately.
# =============================================================================

def prepare_full(prompt, answer, lang):
    system = SYSTEM_TR if lang == "TR" else SYSTEM_EN

    prefix = chat_text(prompt, system)

    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]

    ans_ids = tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]

    full_ids = torch.cat(
        [prefix_ids, ans_ids],
        dim=0,
    ).unsqueeze(0).to(DEVICE)

    attn = torch.ones_like(full_ids)

    answer_start = prefix_ids.numel()

    return full_ids, attn, answer_start

@torch.inference_mode()
def continuation_eval(
    prompt,
    answer,
    lang,
    dirs=None,
    dose=0.0,
    need_logits=False,
):
    ids, attn, answer_start = prepare_full(
        prompt,
        answer,
        lang,
    )

    hooks = SteeringHooks(
        dirs or {},
        dose,
    )

    hooks.install()

    try:
        out = model(
            input_ids=ids,
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )
    finally:
        hooks.remove()

    logits = out.logits.float()

    # Token t is predicted at position t-1.
    positions = torch.arange(
        answer_start,
        ids.shape[1],
        device=DEVICE,
    )

    pred_logits = logits[
        0,
        positions - 1,
        :
    ]

    targets = ids[
        0,
        positions
    ]

    logp = F.log_softmax(
        pred_logits,
        dim=-1,
    )

    selected = logp.gather(
        1,
        targets[:, None],
    ).squeeze(1)

    total_logp = float(
        selected.sum().item()
    )

    mean_nll = float(
        (-selected.mean()).item()
    )

    result = {
        "total_logp": total_logp,
        "mean_nll": mean_nll,
        "requested_energy": hooks.requested_energy,
        "realized_energy": hooks.realized_energy,
        "n_tokens": int(targets.numel()),
    }

    if need_logits:
        result["logits"] = pred_logits.detach().cpu()
        result["targets"] = targets.detach().cpu()

    del out, logits

    return result

# =============================================================================
# NON-TARGET COLLATERAL TEXT
#
# Separate neutral continuation means the target words ABOVE/BELOW are not
# counted as collateral damage.
# =============================================================================

NEUTRAL_CONT_EN = (
    " The statement should be interpreted only from the evidence provided."
)

NEUTRAL_CONT_TR = (
    " İfade yalnızca verilen kanıta göre yorumlanmalıdır."
)

@torch.inference_mode()
def collateral_eval(item, dirs, dose):
    neutral = (
        NEUTRAL_CONT_TR
        if item["lang"] == "TR"
        else NEUTRAL_CONT_EN
    )

    base = continuation_eval(
        item["prompt"],
        neutral,
        item["lang"],
        dirs=None,
        dose=0.0,
        need_logits=True,
    )

    steered = continuation_eval(
        item["prompt"],
        neutral,
        item["lang"],
        dirs=dirs,
        dose=dose,
        need_logits=True,
    )

    B = base["logits"].to(
        DEVICE,
        dtype=torch.float32,
    )

    S = steered["logits"].to(
        DEVICE,
        dtype=torch.float32,
    )

    log_pb = F.log_softmax(B, dim=-1)
    log_ps = F.log_softmax(S, dim=-1)

    pb = log_pb.exp()

    kl = (
        pb
        * (
            log_pb
            - log_ps
        )
    ).sum(dim=-1).mean()

    dnll = (
        steered["mean_nll"]
        - base["mean_nll"]
    )

    return {
        "dnll": float(dnll),
        "kl": float(kl.item()),
        "requested_energy": steered["requested_energy"],
        "realized_energy": steered["realized_energy"],
    }

# =============================================================================
# ITEM EVALUATION
# =============================================================================

@torch.inference_mode()
def evaluate_item(item, method, dose, weighted=False):
    dirs = method_dirs(
        method,
        weighted=weighted,
    )

    # Both candidate answers are evaluated under identical steering.
    above = continuation_eval(
        item["prompt"],
        item["above"],
        item["lang"],
        dirs=dirs,
        dose=dose,
    )

    below = continuation_eval(
        item["prompt"],
        item["below"],
        item["lang"],
        dirs=dirs,
        dose=dose,
    )

    raw_margin = (
        above["total_logp"]
        - below["total_logp"]
    )

    signed_margin = (
        item["label"]
        * raw_margin
    )

    correct = int(
        signed_margin > 0
    )

    collateral = collateral_eval(
        item,
        dirs,
        dose,
    )

    # Energy is measured on the neutral collateral continuation because all
    # methods see exactly the same continuation there.
    return {
        "signed_margin": float(signed_margin),
        "correct": correct,
        "dnll": collateral["dnll"],
        "kl": collateral["kl"],
        "requested_energy": collateral["requested_energy"],
        "realized_energy": collateral["realized_energy"],
    }

# =============================================================================
# DATASET EVALUATION
# =============================================================================

def evaluate_dataset(
    items,
    method,
    dose,
    weighted=False,
    label="",
):
    rows = []

    for i, item in enumerate(items, 1):
        r = evaluate_item(
            item,
            method,
            dose,
            weighted=weighted,
        )

        r.update({
            "split": item["split"],
            "lang": item["lang"],
            "method": method,
            "dose": dose,
            "weighted": int(weighted),
            "item": i - 1,
            "label": item["label"],
        })

        rows.append(r)

        if i == 1 or i % 8 == 0 or i == len(items):
            print(
                f"  {label} | "
                f"{method} | "
                f"dose={dose:.5f} | "
                f"{i:02d}/{len(items)}"
            )

    def avg(key):
        return float(
            np.mean([
                x[key]
                for x in rows
            ])
        )

    summary = {
        "split": items[0]["split"],
        "lang": items[0]["lang"],
        "method": method,
        "dose": dose,
        "weighted": int(weighted),
        "margin": avg("signed_margin"),
        "accuracy": avg("correct"),
        "dnll": avg("dnll"),
        "kl": avg("kl"),
        "requested_energy": avg("requested_energy"),
        "realized_energy": avg("realized_energy"),
    }

    return summary, rows

# =============================================================================
# MODEL CAPABILITY GATE
# =============================================================================

print("\n[5/18] Model Capability Gate...")

BASE_EN_SUM, BASE_EN_ROWS = evaluate_dataset(
    FINAL_ID,
    "BASE",
    0.0,
    label="CAP-EN",
)

BASE_TR_SUM, BASE_TR_ROWS = evaluate_dataset(
    FINAL_TR,
    "BASE",
    0.0,
    label="CAP-TR",
)

CAP_EN = (
    BASE_EN_SUM["accuracy"]
    >= CAPABILITY_THRESHOLD
)

CAP_TR = (
    BASE_TR_SUM["accuracy"]
    >= CAPABILITY_THRESHOLD
)

print("\nCAPABILITY")
print(
    f"EN accuracy={BASE_EN_SUM['accuracy']:.4f} "
    f"margin={BASE_EN_SUM['margin']:+.6f} "
    f"gate={'PASS' if CAP_EN else 'FAIL'}"
)

print(
    f"TR accuracy={BASE_TR_SUM['accuracy']:.4f} "
    f"margin={BASE_TR_SUM['margin']:+.6f} "
    f"gate={'PASS' if CAP_TR else 'FAIL'}"
)

# =============================================================================
# VAL TOURNAMENT
#
# BASE evaluated once.
# RELAY has unweighted and weighted candidates.
# FINAL will receive only the locked winner.
# =============================================================================

print("\n[6/18] VAL tournament...")

VAL_RESULTS = []
VAL_ROWS = []

base_val, base_rows = evaluate_dataset(
    VAL_ITEMS,
    "BASE",
    0.0,
    label="VAL",
)

VAL_RESULTS.append(base_val)
VAL_ROWS.extend(base_rows)

for method in METHODS:
    if method == "BASE":
        continue

    weighted_options = (
        [False, True]
        if method.startswith("RELAY_K")
        else [False]
    )

    for weighted in weighted_options:
        for dose in DOSES:
            s, r = evaluate_dataset(
                VAL_ITEMS,
                method,
                dose,
                weighted=weighted,
                label="VAL",
            )

            s["concept_gain"] = (
                s["margin"]
                - base_val["margin"]
            )

            VAL_RESULTS.append(s)
            VAL_ROWS.extend(r)

# =============================================================================
# VAL LOCK RULE
#
# Feasible:
#   ΔNLL <= MAX_DNLL
#   KL    <= MAX_KL
#
# Among feasible configurations:
#   maximize concept_gain
#   tie-break lower realized energy
#   then lower KL
#
# If none feasible:
#   choose Pareto-like minimum collateral among positive concept gain;
#   if no positive gain exists, choose maximum concept gain.
# =============================================================================

print("\n[7/18] Locking winner from VAL only...")

candidates = [
    x for x in VAL_RESULTS
    if x["method"] != "BASE"
]

feasible = [
    x for x in candidates
    if (
        x["dnll"] <= MAX_DNLL
        and x["kl"] <= MAX_KL
    )
]

if feasible:
    LOCK = sorted(
        feasible,
        key=lambda x: (
            -x["concept_gain"],
            x["realized_energy"],
            x["kl"],
        )
    )[0]

    LOCK_REASON = "FEASIBLE_MAX_CONCEPT"

else:
    positive = [
        x for x in candidates
        if x["concept_gain"] > 0
    ]

    if positive:
        LOCK = sorted(
            positive,
            key=lambda x: (
                x["dnll"] + x["kl"],
                -x["concept_gain"],
                x["realized_energy"],
            )
        )[0]

        LOCK_REASON = "NO_FEASIBLE_MIN_COLLATERAL"

    else:
        LOCK = sorted(
            candidates,
            key=lambda x: (
                -x["concept_gain"],
                x["realized_energy"],
            )
        )[0]

        LOCK_REASON = "NO_POSITIVE_GAIN"

LOCK_METHOD = LOCK["method"]
LOCK_DOSE = float(LOCK["dose"])
LOCK_WEIGHTED = bool(LOCK["weighted"])

print("\nVAL LEADERBOARD")
print(
    f"{'METHOD':<18}"
    f"{'W':>4}"
    f"{'DOSE':>10}"
    f"{'MARGIN':>12}"
    f"{'GAIN':>12}"
    f"{'ACC':>9}"
    f"{'DNLL':>12}"
    f"{'KL':>12}"
    f"{'ENERGY':>14}"
)

for x in sorted(
    candidates,
    key=lambda z: -z["concept_gain"],
):
    print(
        f"{x['method']:<18}"
        f"{x['weighted']:>4d}"
        f"{x['dose']:>10.5f}"
        f"{x['margin']:>12.6f}"
        f"{x['concept_gain']:>12.6f}"
        f"{x['accuracy']:>9.4f}"
        f"{x['dnll']:>12.6f}"
        f"{x['kl']:>12.6f}"
        f"{x['realized_energy']:>14.4f}"
    )

print("\nLOCKED")
print("Method :", LOCK_METHOD)
print("Dose   :", LOCK_DOSE)
print("Weighted:", LOCK_WEIGHTED)
print("Reason :", LOCK_REASON)

# =============================================================================
# FINAL COMPARATOR SET
#
# Winner alone is insufficient for causal interpretation.
# We also evaluate pre-specified controls at the LOCKED dose:
#
# BASE
# STATIC
# LAYER
# SHUFFLED
# LOCKED winner
#
# If winner is not a RELAY, RELAY_K16 is retained as the pre-specified
# transport-aware reference.
# =============================================================================

FINAL_METHODS = [
    ("BASE", 0.0, False),
    ("STATIC", LOCK_DOSE, False),
    ("LAYER", LOCK_DOSE, False),
    ("SHUFFLED", LOCK_DOSE, False),
]

winner_tuple = (
    LOCK_METHOD,
    LOCK_DOSE,
    LOCK_WEIGHTED,
)

if winner_tuple not in FINAL_METHODS:
    FINAL_METHODS.append(
        winner_tuple
    )

relay_reference = (
    "RELAY_K16",
    LOCK_DOSE,
    False,
)

if relay_reference not in FINAL_METHODS:
    FINAL_METHODS.append(
        relay_reference
    )

# =============================================================================
# UNTOUCHED FINAL
# =============================================================================

print("\n[8/18] FINAL-ID...")

ALL_FINAL_RESULTS = []
ALL_FINAL_ROWS = []

FINAL_SPLITS = [
    ("ID", FINAL_ID),
    ("TOPIC", FINAL_TOPIC),
    ("LONG", FINAL_LONG),
    ("TR", FINAL_TR),
]

for split_name, items in FINAL_SPLITS:

    print(f"\n--- FINAL {split_name} ---")

    for method, dose, weighted in FINAL_METHODS:

        s, r = evaluate_dataset(
            items,
            method,
            dose,
            weighted=weighted,
            label=f"FINAL-{split_name}",
        )

        # BASE-relative concept gain for this exact distribution.
        # Find BASE now or compute after all methods.
        s["final_name"] = split_name

        for row in r:
            row["final_name"] = split_name

        ALL_FINAL_RESULTS.append(s)
        ALL_FINAL_ROWS.extend(r)

# Add split-specific BASE-relative gain.

for split_name, _ in FINAL_SPLITS:
    base = next(
        x for x in ALL_FINAL_RESULTS
        if (
            x["final_name"] == split_name
            and x["method"] == "BASE"
        )
    )

    for x in ALL_FINAL_RESULTS:
        if x["final_name"] == split_name:
            x["concept_gain"] = (
                x["margin"]
                - base["margin"]
            )

# =============================================================================
# TURKISH DIAGNOSTIC ATLAS
#
# This is NOT allowed to alter the locked English->Turkish FINAL result.
# It diagnoses whether failure is:
#
#   model capability
#   cross-language map transfer
#   or steering itself
#
# Turkish layer compass already exists.
#
# Build a small Turkish natural transport atlas separately.
# =============================================================================

print("\n[9/18] Building Turkish diagnostic atlas...")

TR_SUBJECTS = [
    "Ahşap kase",
    "Seramik fincan",
    "Mavi dosya",
    "Metal alet kutusu",
    "Kağıt defter",
    "Cam şişe",
    "Pamuklu havlu",
    "Küçük sepet",
]

TR_VERBS = [
    "sessizce duruyor",
    "yerinde duruyor",
    "orada bulunuyor",
    "hareketsiz duruyor",
]

TR_LOCS = [
    "düz bir duvarın yanında",
    "ahşap bir masanın yanında",
    "sessiz bir odanın içinde",
    "kapalı bir dolabın yakınında",
    "basit bir sandalyenin yanında",
    "temiz bir tezgahın yakınında",
]

TR_DETAILS = [
    "Oda sessizdir.",
    "Yüzey temizdir.",
    "Alan iyi aydınlatılmıştır.",
    "Olağandışı bir durum yoktur.",
    "Sahne sıradandır.",
    "Ortam sakindir.",
]

TR_POOL = [
    f"{s} {loc} {v}. {d}"
    for s in TR_SUBJECTS
    for loc in TR_LOCS
    for v in TR_VERBS
    for d in TR_DETAILS
]

rng_tr = np.random.default_rng(155)
rng_tr.shuffle(TR_POOL)

TR_SEARCH_TEXTS = list(
    dict.fromkeys(TR_POOL)
)[:SEARCH_N]

TR_SEARCH = capture_bank(
    TR_SEARCH_TEXTS,
    lang="TR",
    label="TR-SEARCH",
)

TR_MEAN = {
    L: TR_SEARCH[L].mean(
        dim=0,
        keepdim=True,
    )
    for L in LAYERS
}

TR_CENTERED = {
    L: TR_SEARCH[L] - TR_MEAN[L]
    for L in LAYERS
}

TR_ATLAS = {}

for pair in SEGMENTS:
    a, b = pair

    U, S, V = cross_cov_svd(
        TR_CENTERED[a],
        TR_CENTERED[b],
    )

    TR_ATLAS[pair] = {
        "U": U,
        "S": S,
        "V": V,
    }

    print(
        f"TR L{a}->L{b} | "
        f"s1={S[0].item():.6f} "
        f"s4={S[3].item():.6f}"
    )

def tr_relay_chain(K=16):
    dirs = {3: COMPASS_TR[3]}
    current = COMPASS_TR[3]

    for pair in SEGMENTS:
        a, b = pair

        U = TR_ATLAS[pair]["U"][:, :K]
        V = TR_ATLAS[pair]["V"][:, :K]

        c = U.T @ current
        y = V @ c

        if y.norm() < 1e-8:
            y = COMPASS_TR[b]

        y = y / (y.norm() + EPS)

        dirs[b] = y
        current = y

    return dirs

TR_RELAY_DIRS = tr_relay_chain(16)

# =============================================================================
# CUSTOM DIR EVALUATION FOR TR DIAGNOSTIC
# =============================================================================

@torch.inference_mode()
def evaluate_item_custom_dirs(
    item,
    dirs,
    dose,
):
    above = continuation_eval(
        item["prompt"],
        item["above"],
        item["lang"],
        dirs=dirs,
        dose=dose,
    )

    below = continuation_eval(
        item["prompt"],
        item["below"],
        item["lang"],
        dirs=dirs,
        dose=dose,
    )

    margin = (
        above["total_logp"]
        - below["total_logp"]
    )

    signed = (
        item["label"]
        * margin
    )

    coll = collateral_eval(
        item,
        dirs,
        dose,
    )

    return {
        "signed_margin": float(signed),
        "correct": int(signed > 0),
        "dnll": coll["dnll"],
        "kl": coll["kl"],
        "requested_energy": coll["requested_energy"],
        "realized_energy": coll["realized_energy"],
    }

def evaluate_custom_dataset(
    items,
    dirs,
    dose,
    name,
):
    rows = []

    for i, item in enumerate(items, 1):
        r = evaluate_item_custom_dirs(
            item,
            dirs,
            dose,
        )

        rows.append(r)

        if i == 1 or i % 8 == 0 or i == len(items):
            print(
                f"  {name} "
                f"{i:02d}/{len(items)}"
            )

    return {
        "name": name,
        "margin": float(
            np.mean([
                r["signed_margin"]
                for r in rows
            ])
        ),
        "accuracy": float(
            np.mean([
                r["correct"]
                for r in rows
            ])
        ),
        "dnll": float(
            np.mean([
                r["dnll"]
                for r in rows
            ])
        ),
        "kl": float(
            np.mean([
                r["kl"]
                for r in rows
            ])
        ),
        "realized_energy": float(
            np.mean([
                r["realized_energy"]
                for r in rows
            ])
        ),
    }

print("\n[10/18] Turkish cross-language diagnostic...")

EN_TO_TR = evaluate_custom_dataset(
    FINAL_TR,
    RELAY_DIRS[16],
    LOCK_DOSE,
    "EN-ATLAS -> TR",
)

TR_TO_TR = evaluate_custom_dataset(
    FINAL_TR,
    TR_RELAY_DIRS,
    LOCK_DOSE,
    "TR-ATLAS -> TR",
)

TR_LAYER = evaluate_custom_dataset(
    FINAL_TR,
    {L: COMPASS_TR[L] for L in LAYERS},
    LOCK_DOSE,
    "TR-LAYER -> TR",
)

# =============================================================================
# ENERGY EFFICIENCY
#
# Positive concept gain per realized energy.
# Report-only: never used to retroactively select FINAL winner.
# =============================================================================

print("\n[11/18] Computing efficiency...")

for x in VAL_RESULTS:
    if x["method"] == "BASE":
        x["efficiency"] = 0.0
    else:
        x["efficiency"] = (
            x.get(
                "concept_gain",
                x["margin"] - base_val["margin"],
            )
            / (
                x["realized_energy"]
                + EPS
            )
        )

for x in ALL_FINAL_RESULTS:
    if x["method"] == "BASE":
        x["efficiency"] = 0.0
    else:
        x["efficiency"] = (
            x["concept_gain"]
            / (
                x["realized_energy"]
                + EPS
            )
        )

# =============================================================================
# PARETO
#
# Higher concept gain better.
# Lower DNLL, KL, realized energy better.
# =============================================================================

def dominates(a, b):
    better_or_equal = (
        a["concept_gain"] >= b["concept_gain"]
        and a["dnll"] <= b["dnll"]
        and a["kl"] <= b["kl"]
        and a["realized_energy"] <= b["realized_energy"]
    )

    strictly = (
        a["concept_gain"] > b["concept_gain"]
        or a["dnll"] < b["dnll"]
        or a["kl"] < b["kl"]
        or a["realized_energy"] < b["realized_energy"]
    )

    return better_or_equal and strictly

print("\n[12/18] Pareto analysis...")

PARETO = {}

for split_name, _ in FINAL_SPLITS:
    group = [
        x for x in ALL_FINAL_RESULTS
        if x["final_name"] == split_name
    ]

    frontier = []

    for a in group:
        dominated = any(
            dominates(b, a)
            for b in group
            if b is not a
        )

        if not dominated:
            frontier.append(
                (
                    a["method"],
                    a["weighted"],
                    a["dose"],
                )
            )

    PARETO[split_name] = frontier

# =============================================================================
# FINAL TABLE
# =============================================================================

print("\n[13/18] FINAL championship table...")

print("\n" + "=" * 170)
print("FINAL CHAMPIONSHIP")
print("=" * 170)

print(
    f"{'SPLIT':<12}"
    f"{'METHOD':<18}"
    f"{'W':>4}"
    f"{'DOSE':>9}"
    f"{'MARGIN':>12}"
    f"{'GAIN':>12}"
    f"{'ACC':>9}"
    f"{'DNLL':>11}"
    f"{'KL':>11}"
    f"{'ENERGY':>14}"
    f"{'EFF':>14}"
)

for x in ALL_FINAL_RESULTS:
    print(
        f"{x['final_name']:<12}"
        f"{x['method']:<18}"
        f"{x['weighted']:>4d}"
        f"{x['dose']:>9.5f}"
        f"{x['margin']:>12.6f}"
        f"{x['concept_gain']:>12.6f}"
        f"{x['accuracy']:>9.4f}"
        f"{x['dnll']:>11.6f}"
        f"{x['kl']:>11.6f}"
        f"{x['realized_energy']:>14.4f}"
        f"{x['efficiency']:>14.8f}"
    )

print("\nPARETO FRONTIERS")

for split_name, frontier in PARETO.items():
    print(
        split_name,
        "=>",
        frontier,
    )

# =============================================================================
# PRIMARY PRE-SPECIFIED CONTRASTS
# =============================================================================

print("\n[14/18] Primary contrasts...")

def get_final(split_name, method):
    matches = [
        x for x in ALL_FINAL_RESULTS
        if (
            x["final_name"] == split_name
            and x["method"] == method
        )
    ]

    if not matches:
        return None

    return matches[0]

CONTRASTS = {}

for split_name, _ in FINAL_SPLITS:
    relay = get_final(
        split_name,
        "RELAY_K16",
    )

    static = get_final(
        split_name,
        "STATIC",
    )

    layer = get_final(
        split_name,
        "LAYER",
    )

    shuffled = get_final(
        split_name,
        "SHUFFLED",
    )

    if relay is None:
        continue

    CONTRASTS[split_name] = {}

    for name, other in [
        ("RELAY_vs_STATIC", static),
        ("RELAY_vs_LAYER", layer),
        ("RELAY_vs_SHUFFLED", shuffled),
    ]:
        if other is None:
            continue

        CONTRASTS[split_name][name] = {
            "margin_delta":
                relay["margin"]
                - other["margin"],

            "gain_delta":
                relay["concept_gain"]
                - other["concept_gain"],

            "dnll_delta":
                relay["dnll"]
                - other["dnll"],

            "kl_delta":
                relay["kl"]
                - other["kl"],

            "energy_ratio":
                relay["realized_energy"]
                / (
                    other["realized_energy"]
                    + EPS
                ),
        }

        z = CONTRASTS[
            split_name
        ][name]

        print(
            f"{split_name:<10} "
            f"{name:<22} | "
            f"Δmargin={z['margin_delta']:+.6f} "
            f"ΔNLL={z['dnll_delta']:+.6f} "
            f"ΔKL={z['kl_delta']:+.6f} "
            f"Eratio={z['energy_ratio']:.4f}"
        )

# =============================================================================
# TURKISH CAPABILITY INTERPRETATION
# =============================================================================

print("\n[15/18] Language capability interpretation...")

TR_STATUS = ""

if not CAP_TR:
    TR_STATUS = (
        "MODEL_CAPABILITY_GATE_FAIL: "
        "Turkish cross-language steering result is capability-confounded "
        "and must not be treated as a clean RELAY failure."
    )
else:
    TR_STATUS = (
        "MODEL_CAPABILITY_GATE_PASS: "
        "Turkish cross-language steering result is interpretable as a "
        "transport/generalization result under this task distribution."
    )

print(TR_STATUS)

print("\nTurkish diagnostic")
print(
    f"BASE          | "
    f"margin={BASE_TR_SUM['margin']:+.6f} "
    f"acc={BASE_TR_SUM['accuracy']:.4f}"
)

for x in [
    EN_TO_TR,
    TR_TO_TR,
    TR_LAYER,
]:
    print(
        f"{x['name']:<16} | "
        f"margin={x['margin']:+.6f} "
        f"acc={x['accuracy']:.4f} "
        f"dnll={x['dnll']:+.6f} "
        f"KL={x['kl']:.6f} "
        f"E={x['realized_energy']:.4f}"
    )

# =============================================================================
# SAVE CSV
# =============================================================================

print("\n[16/18] Saving CSV / JSON / NPZ...")

VAL_CSV = ROOT / "VAL_RESULTS.csv"
FINAL_CSV = ROOT / "FINAL_RESULTS.csv"
RAW_CSV = ROOT / "FINAL_RAW_ROWS.csv"
REPORT_JSON = ROOT / "REPORT.json"
NPZ_PATH = ROOT / "TEST155_TRANSPORT_AWARE_TOURNAMENT.npz"

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

        for r in rows:
            writer.writerow(r)

save_csv(
    VAL_CSV,
    VAL_RESULTS,
)

save_csv(
    FINAL_CSV,
    ALL_FINAL_RESULTS,
)

save_csv(
    RAW_CSV,
    ALL_FINAL_ROWS,
)

# =============================================================================
# SAVE ATLAS / DIRECTIONS
# =============================================================================

NPZ = {}

for pair in SEGMENTS:
    a, b = pair

    NPZ[f"U_EN_L{a}_L{b}"] = (
        ATLAS[pair]["U"]
        .cpu()
        .numpy()
    )

    NPZ[f"S_EN_L{a}_L{b}"] = (
        ATLAS[pair]["S"]
        .cpu()
        .numpy()
    )

    NPZ[f"V_EN_L{a}_L{b}"] = (
        ATLAS[pair]["V"]
        .cpu()
        .numpy()
    )

    NPZ[f"U_TR_L{a}_L{b}"] = (
        TR_ATLAS[pair]["U"]
        .cpu()
        .numpy()
    )

    NPZ[f"S_TR_L{a}_L{b}"] = (
        TR_ATLAS[pair]["S"]
        .cpu()
        .numpy()
    )

    NPZ[f"V_TR_L{a}_L{b}"] = (
        TR_ATLAS[pair]["V"]
        .cpu()
        .numpy()
    )

for L in LAYERS:
    NPZ[f"COMPASS_EN_L{L}"] = (
        COMPASS_EN[L]
        .cpu()
        .numpy()
    )

    NPZ[f"COMPASS_TR_L{L}"] = (
        COMPASS_TR[L]
        .cpu()
        .numpy()
    )

np.savez_compressed(
    NPZ_PATH,
    **NPZ,
)

# =============================================================================
# JSON REPORT
# =============================================================================

REPORT = {
    "test": "TEST 155",
    "title": "TRANSPORT-AWARE STEERING TOURNAMENT",
    "model": MODEL_ID,
    "gpu": torch.cuda.get_device_name(0),
    "hidden": HIDDEN,
    "layers": LAYERS,
    "segments": SEGMENTS,
    "target_axis": "ABOVE <-> BELOW",
    "doses": DOSES,
    "k_values": K_VALUES,

    "capability": {
        "threshold":
            CAPABILITY_THRESHOLD,

        "EN": {
            "pass": bool(CAP_EN),
            "accuracy":
                BASE_EN_SUM["accuracy"],
            "margin":
                BASE_EN_SUM["margin"],
        },

        "TR": {
            "pass": bool(CAP_TR),
            "accuracy":
                BASE_TR_SUM["accuracy"],
            "margin":
                BASE_TR_SUM["margin"],
        },

        "interpretation":
            TR_STATUS,
    },

    "lock": {
        "method":
            LOCK_METHOD,
        "dose":
            LOCK_DOSE,
        "weighted":
            LOCK_WEIGHTED,
        "reason":
            LOCK_REASON,
    },

    "val_results":
        VAL_RESULTS,

    "final_results":
        ALL_FINAL_RESULTS,

    "pareto":
        {
            k: [
                list(x)
                for x in v
            ]
            for k, v in PARETO.items()
        },

    "contrasts":
        CONTRASTS,

    "turkish_diagnostic": {
        "EN_ATLAS_TO_TR":
            EN_TO_TR,

        "TR_ATLAS_TO_TR":
            TR_TO_TR,

        "TR_LAYER_TO_TR":
            TR_LAYER,
    },

    "interpretation_contract": [
        "VAL alone selects the locked configuration.",
        "FINAL never changes K, dose, weighting, or method.",
        "RELAY beating STATIC does not alone prove transport specificity.",
        "RELAY beating SHUFFLED supports correct layer-to-layer map specificity.",
        "RELAY beating LAYER supports transport-aware hand-off beyond independent layer compasses.",
        "Turkish failure is capability-confounded if the BASE Turkish capability gate fails.",
        "Turkish atlas is diagnostic only and cannot alter the English-atlas FINAL result.",
        "No single weighted championship score is used.",
        "Pareto dimensions are concept gain, non-target ΔNLL, non-target KL, and realized BF16 intervention energy.",
        "Claims are restricted to this model, axis, prompt distributions, layers, and intervention regime.",
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
# LOCKED INTERPRETATION CONTRACT
# =============================================================================

print("\n[17/18] Locked interpretation contract...")

print("\n" + "=" * 150)
print("LOCKED INTERPRETATION CONTRACT")
print("=" * 150)

print(
"""
PRIMARY QUESTIONS

1. Does transport-aware steering outperform STATIC under matched dose
   and measured BF16 write cost?

2. Does RELAY outperform SHUFFLED-RELAY?
   If yes, the advantage is more specifically associated with the
   correct layer-to-layer transport geometry rather than merely using
   different vectors at different layers.

3. Does RELAY outperform LAYER-COMPASS?
   If yes, independently extracting a direction at every layer is not
   sufficient to explain the transport-aware advantage.

4. Does broader K improve control/collateral tradeoff?
   If yes, TEST154 diffusion structure has engineering value.

5. Does the English-locked transport system retain an advantage on
   topic-shifted, long-context and Turkish prompts?

LANGUAGE RULE

If BASE Turkish capability fails the pre-locked capability threshold,
the English->Turkish steering result is capability-confounded.

The Turkish diagnostic atlas may identify whether within-Turkish
transport remains usable, but it may NEVER replace or modify the
locked English->Turkish FINAL result.

ENERGY RULE

Requested intervention energy and realized BF16 write energy are
reported separately.

Realized BF16 energy is the primary engineering cost.

COLLATERAL RULE

Target answer tokens are not used for collateral measurement.

Collateral is measured on a separate neutral continuation using:
    ΔNLL
    KL(BASE || STEERED)

SELECTION RULE

Only VAL may choose:
    method
    K
    dose
    weighted/unweighted transport

FINAL is read-only evaluation.

NO UNIVERSALITY CLAIM

Successful Turkish or OOD transfer supports cross-distribution
generalization under the tested conditions. It does not by itself
establish a universal architecture constant.

NO AUTOMATIC WINNER CLAIM

The test reports raw metrics and Pareto frontiers.
RELAY is allowed to win, tie, or lose.
"""
)

# =============================================================================
# DONE
# =============================================================================

print("\n[18/18] COMPLETE")
print("=" * 150)

print("Locked method :", LOCK_METHOD)
print("Locked dose   :", LOCK_DOSE)
print("Weighted      :", LOCK_WEIGHTED)
print("EN capability :", "PASS" if CAP_EN else "FAIL")
print("TR capability :", "PASS" if CAP_TR else "FAIL")

print("\nSaved:")
print(" ", VAL_CSV)
print(" ", FINAL_CSV)
print(" ", RAW_CSV)
print(" ", REPORT_JSON)
print(" ", NPZ_PATH)

print("=" * 150)
print("TEST 155 COMPLETE")
print("=" * 150)
