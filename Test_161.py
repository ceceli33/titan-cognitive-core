# ==================================================================================================
# TEST 161 — SEMANTIC DIRECTION SUBSPACE CAUSAL ASSAY
# AKBASCORE 3.1 / SEASC
# ==================================================================================================

import os, sys, gc, math, json, csv, random, shutil, subprocess
from pathlib import Path

print("=" * 150)
print("TEST 161 — SEMANTIC DIRECTION SUBSPACE CAUSAL ASSAY")
print("AKBASCORE 3.1 / SEASC")
print("=" * 150)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45.0", "accelerate>=0.34.0", "ninja", "pybind11"
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
SEED = 161
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================================================================================================
# 1. CONFIG
# ==================================================================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
N_LAYERS = 20

IVME, SONUM, ZIRVE, TABAN = 0.10, 0.30, 0.70, 0.20
FISHER_ALPHA, FISHER_EPS = 0.10, 1e-8

DOSE_SCALES = [0.25, 0.50, 1.00]
IN_SPAN_DRAWS = 8
OUT_SPAN_DRAWS = 8
AMBIENT_DRAWS = 8
PERMUTATIONS = 20000
EPS = 1e-10

SYSTEM = "You are a concise reasoning assistant. Use only the information in the prompt."
ROOT = Path("/content/TEST161_SEMANTIC_SUBSPACE")
ROOT.mkdir(parents=True, exist_ok=True)

print("GPU            :", torch.cuda.get_device_name(0))
print("Model          :", MODEL_ID)
print("Steered layers : L0-L19")
print("Dose scales    :", DOSE_SCALES)
print("In-span draws  :", IN_SPAN_DRAWS)
print("Out-span draws :", OUT_SPAN_DRAWS)
print("Ambient draws  :", AMBIENT_DRAWS)

# ==================================================================================================
# 2. CUDA — SAME FROZEN-NORM SEASC MOTOR
# ==================================================================================================

BUILD_DIR = "/tmp/akbascore_test161"
shutil.rmtree(BUILD_DIR, ignore_errors=True)
os.makedirs(BUILD_DIR, exist_ok=True)

CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor seasc_batch_cuda(torch::Tensor hidden, torch::Tensor actuator, torch::Tensor dose);

torch::Tensor seasc_batch(torch::Tensor hidden, torch::Tensor actuator, torch::Tensor dose) {
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
    m.def("seasc_batch", &seasc_batch, "TEST161 SEASC batch frozen-norm injection");
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
    const int B, const int S, const int H
) {
    const int vec = blockIdx.x;
    const int b = vec / S;
    if (b >= B) return;

    extern __shared__ float shared[];
    const long long base = static_cast<long long>(vec) * static_cast<long long>(H);
    const long long abase = static_cast<long long>(b) * static_cast<long long>(H);

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const float x = static_cast<float>(h[base + j]);
        local_sum += x * x;
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    const float norm0 = sqrtf(fmaxf(shared[0], 1.0e-20f));
    const float scale = dose[b] * norm0;
    __syncthreads();

    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        const float old_value = static_cast<float>(h[base + j]);
        h[base + j] = static_cast<scalar_t>(old_value + scale * actuator[abase + j]);
    }
}

torch::Tensor seasc_batch_cuda(torch::Tensor hidden, torch::Tensor actuator, torch::Tensor dose) {
    auto out = hidden.contiguous().clone();
    auto act = actuator.to(hidden.device(), torch::kFloat32).contiguous();
    auto d = dose.to(hidden.device(), torch::kFloat32).contiguous();

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
        "test161_seasc_batch",
        [&] {
            seasc_batch_kernel<scalar_t><<<n_vectors, THREADS, shared_bytes, stream>>>(
                out.data_ptr<scalar_t>(), act.data_ptr<float>(), d.data_ptr<float>(), B, S, H
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

print("\n[1/15] Compiling AkbasCore SEASC kernel...")

seasc_ext = load_inline(
    name="akbascore_test161_cuda",
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=None,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    build_directory=BUILD_DIR,
    verbose=False
)

print("      CUDA kernel: OK")

# ==================================================================================================
# 3. MODEL
# ==================================================================================================

print("\n[2/15] Loading frozen model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    trust_remote_code=True
)

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

layers = model.model.layers
HIDDEN = model.config.hidden_size

if len(layers) < N_LAYERS:
    raise RuntimeError("Model has fewer than 20 layers.")

print("      hidden :", HIDDEN)
print("      layers :", len(layers))
print("      dtype  :", next(model.parameters()).dtype)

# ==================================================================================================
# 4. DRA ENVELOPE
# ==================================================================================================

def dra_envelope(L):
    t = float(L)
    kb = ZIRVE * math.exp(-SONUM * t) * (1.0 + SONUM * t) + TABAN
    return kb / (ZIRVE + TABAN)

ENVELOPE = [dra_envelope(L) for L in range(N_LAYERS)]

print("\nDRA envelope:")
for L, e in enumerate(ENVELOPE):
    print(f"  L{L:02d}: E={e:.6f} | base={100 * IVME * e:.3f}%")

# ==================================================================================================
# 5. SEARCH BANK
# ==================================================================================================

SEARCH_PAIRS = [
    ("The lantern is above the stone arch.", "The lantern is below the stone arch."),
    ("The kite is above the old tower.", "The kite is below the old tower."),
    ("The flag is above the wooden gate.", "The flag is below the wooden gate."),
    ("The bird is above the narrow bridge.", "The bird is below the narrow bridge."),
    ("The sign is above the glass doorway.", "The sign is below the glass doorway."),
    ("The lamp is above the metal shelf.", "The lamp is below the metal shelf."),
    ("The clock is above the brown cabinet.", "The clock is below the brown cabinet."),
    ("The balloon is above the green hill.", "The balloon is below the green hill."),
    ("The camera is above the small desk.", "The camera is below the small desk."),
    ("The painting is above the blue sofa.", "The painting is below the blue sofa."),
    ("The cloud is above the distant ridge.", "The cloud is below the distant ridge."),
    ("The marker is above the white board.", "The marker is below the white board."),
    ("The drone is above the concrete wall.", "The drone is below the concrete wall."),
    ("The shelf is above the storage box.", "The shelf is below the storage box."),
    ("The light is above the front window.", "The light is below the front window."),
    ("The symbol is above the horizontal line.", "The symbol is below the horizontal line.")
]

# ==================================================================================================
# 6. EVALUATION / NEUTRAL SET
# ==================================================================================================

EVAL_ITEMS = [
    {"context":"A red marker and a blue marker are visible. Their vertical relation is unspecified.",
     "pos":" The red marker is above the blue marker.", "neg":" The red marker is below the blue marker."},
    {"context":"A bright lamp and a wooden table are visible. Their vertical relation is unspecified.",
     "pos":" The bright lamp is above the wooden table.", "neg":" The bright lamp is below the wooden table."},
    {"context":"A bird and a tree branch are visible. Their vertical relation is unspecified.",
     "pos":" The bird is above the tree branch.", "neg":" The bird is below the tree branch."},
    {"context":"A sign and a doorway are visible. Their vertical relation is unspecified.",
     "pos":" The sign is above the doorway.", "neg":" The sign is below the doorway."},
    {"context":"A shelf and a cabinet are visible. Their vertical relation is unspecified.",
     "pos":" The shelf is above the cabinet.", "neg":" The shelf is below the cabinet."},
    {"context":"A cloud and a hill are visible. Their vertical relation is unspecified.",
     "pos":" The cloud is above the hill.", "neg":" The cloud is below the hill."},
    {"context":"A clock and a desk are visible. Their vertical relation is unspecified.",
     "pos":" The clock is above the desk.", "neg":" The clock is below the desk."},
    {"context":"A framed picture and a sofa are visible. Their vertical relation is unspecified.",
     "pos":" The framed picture is above the sofa.", "neg":" The framed picture is below the sofa."},
    {"context":"A drone and a stone wall are visible. Their vertical relation is unspecified.",
     "pos":" The drone is above the stone wall.", "neg":" The drone is below the stone wall."},
    {"context":"A balloon and a house are visible. Their vertical relation is unspecified.",
     "pos":" The balloon is above the house.", "neg":" The balloon is below the house."},
    {"context":"A camera and a storage box are visible. Their vertical relation is unspecified.",
     "pos":" The camera is above the storage box.", "neg":" The camera is below the storage box."},
    {"context":"A silver symbol and a horizontal line are visible. Their vertical relation is unspecified.",
     "pos":" The silver symbol is above the horizontal line.", "neg":" The silver symbol is below the horizontal line."}
]

NEUTRAL = [
    "A ceramic cup rests beside a closed notebook.",
    "A wooden chair stands near a plain wall.",
    "A glass bottle sits beside a small basket.",
    "A cotton towel rests on a clean counter.",
    "A metal toolbox is inside a quiet room.",
    "A paper folder lies beside a keyboard.",
    "A black umbrella stands near a cabinet.",
    "A silver spoon rests beside a plate."
]

NEUTRAL_CONT = " The scene contains ordinary stationary objects and no additional relation is specified."

# ==================================================================================================
# 7. HELPERS
# ==================================================================================================

def chat_text(user_text):
    return tokenizer.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":user_text}],
        tokenize=False,
        add_generation_prompt=True
    )

@torch.inference_mode()
def capture(text):
    enc = tokenizer(chat_text(text), return_tensors="pt").to(DEVICE)
    out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    pos = int(enc["attention_mask"][0].sum().item()) - 1
    hs = [out.hidden_states[L + 1][0, pos].float().detach() for L in range(N_LAYERS)]
    del out, enc
    return hs

def normalize(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(EPS)

# ==================================================================================================
# 8. SEARCH ACTIVATIONS
# ==================================================================================================

print("\n[3/15] Building SEARCH activation bank...")

POS_BANK = [[] for _ in range(N_LAYERS)]
NEG_BANK = [[] for _ in range(N_LAYERS)]

for i, (ptext, ntext) in enumerate(SEARCH_PAIRS, 1):
    hp, hn = capture(ptext), capture(ntext)
    for L in range(N_LAYERS):
        POS_BANK[L].append(hp[L])
        NEG_BANK[L].append(hn[L])
    print(f"      pair {i:02d}/{len(SEARCH_PAIRS)}")

POS_BANK = [torch.stack(x).float().contiguous() for x in POS_BANK]
NEG_BANK = [torch.stack(x).float().contiguous() for x in NEG_BANK]

# ==================================================================================================
# 9. FISHER + SEMANTIC DIRECTION SUBSPACE
# ==================================================================================================

print("\n[4/15] Building Fisher compasses + semantic direction subspaces...")

LOCAL_COMPASS, BANK_MEAN, PCA1, SPAN_BASIS = [], [], [], []
SPAN_RANK, FISHER_CAPTURE, MEAN_FISHER_COS, PCA1_FISHER_COS, TOP4_ENERGY = [], [], [], [], []

for L in range(N_LAYERS):
    Hp, Hn = POS_BANK[L], NEG_BANK[L]
    mu_p, mu_n = Hp.mean(0), Hn.mean(0)
    mean_diff = mu_p - mu_n

    R = torch.cat([Hp - mu_p, Hn - mu_n], dim=0).float()
    nu = max(int(R.shape[0]) - 2, 1)

    _, S_f, Vh_f = torch.linalg.svd(R, full_matrices=False)
    eig = S_f.square() / float(nu)
    positive_eig = eig[eig > FISHER_EPS]

    eig_scale = torch.median(positive_eig) if positive_eig.numel() else torch.tensor(
        1.0, device=DEVICE, dtype=torch.float32
    )

    lam = (FISHER_ALPHA * eig_scale).clamp_min(FISHER_EPS)
    V = Vh_f.T
    projection = V.T @ mean_diff
    correction = ((1.0 / (eig + lam) - 1.0 / lam) * projection)
    fisher = mean_diff / lam + V @ correction
    fisher = (fisher / fisher.norm().clamp_min(EPS)).float().contiguous()

    pair_dirs = normalize(Hp - Hn)
    orientation = pair_dirs @ fisher
    pair_dirs = pair_dirs * torch.where(
        orientation[:, None] < 0,
        -torch.ones_like(orientation[:, None]),
        torch.ones_like(orientation[:, None])
    )

    bank_mean = pair_dirs.mean(0)
    bank_mean = (bank_mean / bank_mean.norm().clamp_min(EPS)).float().contiguous()

    centered = pair_dirs - pair_dirs.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    pc1 = Vh[0].float().contiguous()
    if torch.dot(pc1, fisher) < 0:
        pc1 = -pc1
    pc1 = (pc1 / pc1.norm().clamp_min(EPS)).contiguous()

    _, S_span, Vh_span = torch.linalg.svd(pair_dirs, full_matrices=False)

    if S_span.numel():
        tol = max(pair_dirs.shape) * torch.finfo(torch.float32).eps * S_span.max()
    else:
        tol = torch.tensor(0.0, device=DEVICE)

    rank = max(int((S_span > tol).sum().item()), 1)
    basis = Vh_span[:rank].T.float().contiguous()
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    basis = basis.float().contiguous()

    fisher_capture = float((basis.T @ fisher).square().sum().item())
    mean_fisher = float(torch.dot(bank_mean, fisher).item())
    pc_fisher = float(torch.dot(pc1, fisher).item())

    total_energy = S.square().sum().clamp_min(EPS)
    k4 = min(4, int(S.numel()))
    top4 = float((S[:k4].square().sum() / total_energy).item())

    LOCAL_COMPASS.append(fisher)
    BANK_MEAN.append(bank_mean)
    PCA1.append(pc1)
    SPAN_BASIS.append(basis)
    SPAN_RANK.append(rank)
    FISHER_CAPTURE.append(fisher_capture)
    MEAN_FISHER_COS.append(mean_fisher)
    PCA1_FISHER_COS.append(pc_fisher)
    TOP4_ENERGY.append(top4)

    print(
        f"      L{L:02d} | rank={rank:02d} | Fisher-in-span={fisher_capture:.6f} | "
        f"mean/Fisher={mean_fisher:+.6f} | PC1/Fisher={pc_fisher:+.6f} | E4={top4:.6f}"
    )

LOCAL_COMPASS = torch.stack(LOCAL_COMPASS).float().contiguous()
BANK_MEAN = torch.stack(BANK_MEAN).float().contiguous()
PCA1 = torch.stack(PCA1).float().contiguous()

# ==================================================================================================
# 10. RANDOM DIRECTION SETS
# ==================================================================================================

print("\n[5/15] Building fixed in-span, out-span and ambient direction draws...")

GEN = torch.Generator(device=DEVICE)
GEN.manual_seed(SEED)

def orient_to_fisher(z, fisher):
    return -z if torch.dot(z, fisher) < 0 else z

IN_SPAN = []
for d in range(IN_SPAN_DRAWS):
    per_layer = []
    for L in range(N_LAYERS):
        B = SPAN_BASIS[L]
        coeff = torch.randn(B.shape[1], generator=GEN, device=DEVICE, dtype=torch.float32)
        z = B @ coeff
        z = z / z.norm().clamp_min(EPS)
        per_layer.append(orient_to_fisher(z, LOCAL_COMPASS[L]))
    IN_SPAN.append(torch.stack(per_layer).float().contiguous())

OUT_SPAN = []
for d in range(OUT_SPAN_DRAWS):
    per_layer = []
    for L in range(N_LAYERS):
        B = SPAN_BASIS[L]
        success = False
        for _ in range(32):
            z = torch.randn(HIDDEN, generator=GEN, device=DEVICE, dtype=torch.float32)
            z = z - B @ (B.T @ z)
            n = z.norm()
            if n > 1e-7:
                z = z / n
                success = True
                break
        if not success:
            raise RuntimeError(f"Failed to construct out-span direction at L{L}.")
        per_layer.append(orient_to_fisher(z, LOCAL_COMPASS[L]))
    OUT_SPAN.append(torch.stack(per_layer).float().contiguous())

AMBIENT = []
for d in range(AMBIENT_DRAWS):
    per_layer = []
    for L in range(N_LAYERS):
        z = torch.randn(HIDDEN, generator=GEN, device=DEVICE, dtype=torch.float32)
        z = z / z.norm().clamp_min(EPS)
        per_layer.append(orient_to_fisher(z, LOCAL_COMPASS[L]))
    AMBIENT.append(torch.stack(per_layer).float().contiguous())

def mean_span_projection(vector_set):
    vals = []
    for vecs in vector_set:
        for L in range(N_LAYERS):
            vals.append(float((SPAN_BASIS[L].T @ vecs[L]).square().sum().item()))
    return float(np.mean(vals))

print("      in-span capture :", f"{mean_span_projection(IN_SPAN):.8f}")
print("      out-span capture:", f"{mean_span_projection(OUT_SPAN):.8f}")
print("      ambient capture :", f"{mean_span_projection(AMBIENT):.8f}")

# ==================================================================================================
# 11. HOOK ENGINE
# ==================================================================================================

def blank_telemetry():
    return [{"requested":[], "realized":[]} for _ in range(N_LAYERS)]

def install_hooks(vectors, sign, dose_scale, telemetry):
    handles = []

    for L in range(N_LAYERS):
        def make_hook(layer_idx):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    old, rest = output[0], output[1:]
                else:
                    old, rest = output, None

                batch = old.shape[0]
                direction = vectors[layer_idx][None, :].expand(batch, -1)
                direction = (float(sign) * direction).float().contiguous()

                requested_value = IVME * ENVELOPE[layer_idx] * dose_scale
                dose = torch.full(
                    (batch,), float(requested_value), device=DEVICE, dtype=torch.float32
                )

                old_last = old[:, -1, :].float()
                new = seasc_ext.seasc_batch(old, direction, dose)
                new_last = new[:, -1, :].float()

                realized = (new_last - old_last).norm(dim=-1) / old_last.norm(dim=-1).clamp_min(EPS)

                telemetry[layer_idx]["requested"].extend(dose.detach().cpu().tolist())
                telemetry[layer_idx]["realized"].extend(realized.detach().cpu().tolist())

                return new if rest is None else (new,) + rest
            return hook

        handles.append(layers[L].register_forward_hook(make_hook(L)))

    return handles

# ==================================================================================================
# 12. SCORING
# ==================================================================================================

def build_sequence(context, continuation):
    prompt = chat_text(context)
    p = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    c = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)

    ids = torch.cat([p["input_ids"], c["input_ids"]], dim=1).to(DEVICE)
    mask = torch.ones_like(ids, device=DEVICE)
    return ids, mask, p["input_ids"].shape[1]

@torch.inference_mode()
def continuation_logp(context, continuation, vectors=None, sign=0, scale=0.0):
    ids, mask, plen = build_sequence(context, continuation)
    telemetry = blank_telemetry()
    handles = []

    if vectors is not None:
        handles = install_hooks(vectors, sign, scale, telemetry)

    try:
        out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    logits = out.logits.float()
    target = ids[:, plen:]
    pred = logits[:, plen - 1:-1, :]
    lp = F.log_softmax(pred, dim=-1)
    token_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    score = float(token_lp.mean().item())

    del out, logits, target, pred, lp, token_lp, ids, mask
    return score, telemetry

# ==================================================================================================
# 13. BASELINE
# ==================================================================================================

print("\n[6/15] Measuring untouched baseline...")

BASE_ROWS = []

for i, item in enumerate(EVAL_ITEMS):
    pos, _ = continuation_logp(item["context"], item["pos"])
    neg, _ = continuation_logp(item["context"], item["neg"])
    margin = pos - neg
    BASE_ROWS.append({"item":i, "margin":margin})
    print(f"      item {i:02d} | margin={margin:+.6f}")

BASE_MEAN = float(np.mean([x["margin"] for x in BASE_ROWS]))
print(f"      baseline mean={BASE_MEAN:+.6f}")

# ==================================================================================================
# 14. CAUSAL ARM
# ==================================================================================================

def run_causal_arm(method, vectors, scale, draw=-1):
    item_csym, plus_margins, minus_margins = [], [], []
    requested, realized = [], []

    for item in EVAL_ITEMS:
        p_pos, t1 = continuation_logp(item["context"], item["pos"], vectors, +1, scale)
        p_neg, t2 = continuation_logp(item["context"], item["neg"], vectors, +1, scale)
        n_pos, t3 = continuation_logp(item["context"], item["pos"], vectors, -1, scale)
        n_neg, t4 = continuation_logp(item["context"], item["neg"], vectors, -1, scale)

        pm, nm = p_pos - p_neg, n_pos - n_neg
        c = 0.5 * (pm - nm)

        plus_margins.append(pm)
        minus_margins.append(nm)
        item_csym.append(c)

        for tel in (t1, t2, t3, t4):
            for L in range(N_LAYERS):
                requested.extend(tel[L]["requested"])
                realized.extend(tel[L]["realized"])

    return {
        "method":method, "draw":draw, "scale":scale,
        "plus_margin":float(np.mean(plus_margins)),
        "minus_margin":float(np.mean(minus_margins)),
        "Csym":float(np.mean(item_csym)),
        "Csym_median":float(np.median(item_csym)),
        "positive_items":int(np.sum(np.asarray(item_csym) > 0)),
        "requested_mean":float(np.mean(requested)),
        "realized_mean":float(np.mean(realized)),
        "energy":float(np.sum(np.square(realized))),
        "item_csym":[float(x) for x in item_csym]
    }

# ==================================================================================================
# 15. REFERENCE ARMS
# ==================================================================================================

print("\n[7/15] Running deterministic reference arms...")

REAL_ROWS = []
REFERENCE_ARMS = [
    ("FISHER", LOCAL_COMPASS),
    ("BANK_MEAN", BANK_MEAN),
    ("PCA1", PCA1)
]

for scale in DOSE_SCALES:
    for method, vectors in REFERENCE_ARMS:
        r = run_causal_arm(method, vectors, scale)
        REAL_ROWS.append(r)
        print(
            f"  {method:<12} scale={scale:.2f} | Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/{len(EVAL_ITEMS)} | "
            f"real={100*r['realized_mean']:.4f}% | E={r['energy']:.6f}"
        )

# ==================================================================================================
# 16. RANDOM TOURNAMENTS
# ==================================================================================================

print("\n[8/15] Running random IN-SPAN directions...")
IN_ROWS = []
for scale in DOSE_SCALES:
    for d, vectors in enumerate(IN_SPAN):
        r = run_causal_arm("IN_SPAN", vectors, scale, d)
        IN_ROWS.append(r)
        print(
            f"  scale={scale:.2f} draw={d:02d} | Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/{len(EVAL_ITEMS)} | E={r['energy']:.6f}"
        )

print("\n[9/15] Running random OUT-SPAN directions...")
OUT_ROWS = []
for scale in DOSE_SCALES:
    for d, vectors in enumerate(OUT_SPAN):
        r = run_causal_arm("OUT_SPAN", vectors, scale, d)
        OUT_ROWS.append(r)
        print(
            f"  scale={scale:.2f} draw={d:02d} | Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/{len(EVAL_ITEMS)} | E={r['energy']:.6f}"
        )

print("\n[10/15] Running AMBIENT random directions...")
AMBIENT_ROWS = []
for scale in DOSE_SCALES:
    for d, vectors in enumerate(AMBIENT):
        r = run_causal_arm("AMBIENT", vectors, scale, d)
        AMBIENT_ROWS.append(r)
        print(
            f"  scale={scale:.2f} draw={d:02d} | Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/{len(EVAL_ITEMS)} | E={r['energy']:.6f}"
        )

# ==================================================================================================
# 17. STATISTICS
# ==================================================================================================

print("\n[11/15] Semantic-subspace statistics...")

def values_at(rows, scale):
    return np.asarray([r["Csym"] for r in rows if r["scale"] == scale], dtype=np.float64)

def permutation_p_greater(a, b, seed):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    exceed = 0

    for _ in range(PERMUTATIONS):
        perm = rng.permutation(pooled)
        if perm[:n_a].mean() - perm[n_a:].mean() >= observed:
            exceed += 1

    return (exceed + 1) / (PERMUTATIONS + 1)

def pairwise_auc(a, b):
    wins, total = 0.0, 0
    for x in a:
        for y in b:
            total += 1
            wins += 1.0 if x > y else 0.5 if x == y else 0.0
    return wins / total

STAT_ROWS = []

for scale in DOSE_SCALES:
    ins, outs, amb = values_at(IN_ROWS, scale), values_at(OUT_ROWS, scale), values_at(AMBIENT_ROWS, scale)

    row = {
        "scale":scale,
        "in_mean":float(ins.mean()), "in_sd":float(ins.std(ddof=1)),
        "out_mean":float(outs.mean()), "out_sd":float(outs.std(ddof=1)),
        "ambient_mean":float(amb.mean()), "ambient_sd":float(amb.std(ddof=1)),
        "in_minus_out":float(ins.mean() - outs.mean()),
        "in_minus_ambient":float(ins.mean() - amb.mean()),
        "auc_in_vs_out":float(pairwise_auc(ins, outs)),
        "auc_in_vs_ambient":float(pairwise_auc(ins, amb)),
        "p_in_gt_out":float(permutation_p_greater(ins, outs, SEED + int(1000 * scale))),
        "p_in_gt_ambient":float(permutation_p_greater(ins, amb, SEED + 10000 + int(1000 * scale)))
    }

    STAT_ROWS.append(row)

    print(
        f"  scale={scale:.2f} | IN={row['in_mean']:+.6f}±{row['in_sd']:.6f} | "
        f"OUT={row['out_mean']:+.6f}±{row['out_sd']:.6f} | "
        f"Δ={row['in_minus_out']:+.6f} | AUC={row['auc_in_vs_out']:.4f} | p={row['p_in_gt_out']:.6f}"
    )
    print(
        f"             AMB={row['ambient_mean']:+.6f}±{row['ambient_sd']:.6f} | "
        f"IN-AMB={row['in_minus_ambient']:+.6f} | "
        f"AUC={row['auc_in_vs_ambient']:.4f} | p={row['p_in_gt_ambient']:.6f}"
    )

# ==================================================================================================
# 18. ENERGY
# ==================================================================================================

print("\n[12/15] Physical-energy matching...")

ENERGY_ROWS = []

for scale in DOSE_SCALES:
    in_e = np.asarray([r["energy"] for r in IN_ROWS if r["scale"] == scale])
    out_e = np.asarray([r["energy"] for r in OUT_ROWS if r["scale"] == scale])
    amb_e = np.asarray([r["energy"] for r in AMBIENT_ROWS if r["scale"] == scale])

    row = {
        "scale":scale,
        "in_energy":float(in_e.mean()),
        "out_energy":float(out_e.mean()),
        "ambient_energy":float(amb_e.mean()),
        "out_over_in":float(out_e.mean() / max(in_e.mean(), EPS)),
        "ambient_over_in":float(amb_e.mean() / max(in_e.mean(), EPS))
    }

    ENERGY_ROWS.append(row)
    print(
        f"  scale={scale:.2f} | IN={row['in_energy']:.6f} | OUT={row['out_energy']:.6f} | "
        f"OUT/IN={row['out_over_in']:.6f} | AMB/IN={row['ambient_over_in']:.6f}"
    )

# ==================================================================================================
# 19. NEUTRAL COLLATERAL
# ==================================================================================================

print("\n[13/15] Neutral collateral...")

@torch.inference_mode()
def neutral_metrics(text, vectors, scale):
    ids, mask, plen = build_sequence(text, NEUTRAL_CONT)

    out0 = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    base_logits = out0.logits.float()[:, plen - 1:-1, :]
    target = ids[:, plen:]
    base_lp = F.log_softmax(base_logits, dim=-1)
    base_nll = -base_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()

    telemetry = blank_telemetry()
    handles = install_hooks(vectors, +1, scale, telemetry)

    try:
        out1 = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()

    steer_logits = out1.logits.float()[:, plen - 1:-1, :]
    steer_lp = F.log_softmax(steer_logits, dim=-1)
    steer_nll = -steer_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean()

    p = base_lp.exp()
    kl = (p * (base_lp - steer_lp)).sum(-1).mean()

    dnll, klv = float((steer_nll - base_nll).item()), float(kl.item())

    del out0, out1, ids, mask, target, base_logits, steer_logits, base_lp, steer_lp
    return dnll, klv

COLLATERAL_ROWS = []

for scale in DOSE_SCALES:
    for method, vectors in REFERENCE_ARMS:
        dnlls, kls = [], []

        for text in NEUTRAL:
            dnll, kl = neutral_metrics(text, vectors, scale)
            dnlls.append(dnll)
            kls.append(kl)

        row = {
            "method":method,
            "scale":scale,
            "delta_nll":float(np.mean(dnlls)),
            "kl":float(np.mean(kls))
        }

        COLLATERAL_ROWS.append(row)
        print(
            f"  {method:<12} scale={scale:.2f} | "
            f"ΔNLL={row['delta_nll']:+.6f} | KL={row['kl']:.6f}"
        )

# ==================================================================================================
# 20. DIAGNOSTIC CONTRACT
# ==================================================================================================

print("\n[14/15] Diagnostic contract...")

FLAGS = []

directional_out = sum(r["in_minus_out"] > 0 and r["auc_in_vs_out"] > 0.5 for r in STAT_ROWS)
significant_out = sum(r["in_minus_out"] > 0 and r["p_in_gt_out"] < 0.05 for r in STAT_ROWS)
directional_ambient = sum(r["in_minus_ambient"] > 0 and r["auc_in_vs_ambient"] > 0.5 for r in STAT_ROWS)
significant_ambient = sum(r["in_minus_ambient"] > 0 and r["p_in_gt_ambient"] < 0.05 for r in STAT_ROWS)

if directional_out >= 2:
    FLAGS.append("SEMANTIC_SPAN_DIRECTIONAL_ADVANTAGE")
if significant_out >= 2:
    FLAGS.append("SEMANTIC_SPAN_CAUSAL_SUPPORT")
if directional_ambient >= 2:
    FLAGS.append("SEMANTIC_SPAN_ABOVE_AMBIENT")
if significant_ambient >= 2:
    FLAGS.append("SEMANTIC_SPAN_ABOVE_AMBIENT_STRONG")
if not FLAGS:
    FLAGS.append("SEMANTIC_SPAN_ADVANTAGE_NOT_ESTABLISHED")

for flag in FLAGS:
    print(" -", flag)

# ==================================================================================================
# 21. SAVE
# ==================================================================================================

def csv_safe(rows):
    return [
        {k:v for k,v in r.items() if not isinstance(v, (list, dict, tuple))}
        for r in rows
    ]

def save_csv(path, rows):
    rows = csv_safe(rows)
    if not rows:
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

GEOMETRY_ROWS = [
    {
        "layer":L,
        "span_rank":SPAN_RANK[L],
        "fisher_capture":FISHER_CAPTURE[L],
        "mean_fisher_cos":MEAN_FISHER_COS[L],
        "pca1_fisher_cos":PCA1_FISHER_COS[L],
        "top4_variation_energy":TOP4_ENERGY[L]
    }
    for L in range(N_LAYERS)
]

save_csv(ROOT / "BASELINE.csv", BASE_ROWS)
save_csv(ROOT / "REFERENCE_ARMS.csv", REAL_ROWS)
save_csv(ROOT / "IN_SPAN.csv", IN_ROWS)
save_csv(ROOT / "OUT_SPAN.csv", OUT_ROWS)
save_csv(ROOT / "AMBIENT.csv", AMBIENT_ROWS)
save_csv(ROOT / "SPAN_STATISTICS.csv", STAT_ROWS)
save_csv(ROOT / "ENERGY_MATCH.csv", ENERGY_ROWS)
save_csv(ROOT / "COLLATERAL.csv", COLLATERAL_ROWS)
save_csv(ROOT / "SUBSPACE_GEOMETRY.csv", GEOMETRY_ROWS)

np.savez_compressed(
    ROOT / "TEST161_SEMANTIC_SUBSPACE.npz",
    local_compass=LOCAL_COMPASS.detach().cpu().numpy(),
    bank_mean=BANK_MEAN.detach().cpu().numpy(),
    pca1=PCA1.detach().cpu().numpy(),
    envelope=np.asarray(ENVELOPE, dtype=np.float32),
    span_rank=np.asarray(SPAN_RANK, dtype=np.int64),
    fisher_capture=np.asarray(FISHER_CAPTURE, dtype=np.float32),
    in_span=np.stack([x.detach().cpu().numpy() for x in IN_SPAN]),
    out_span=np.stack([x.detach().cpu().numpy() for x in OUT_SPAN]),
    ambient=np.stack([x.detach().cpu().numpy() for x in AMBIENT])
)

REPORT = {
    "test":"TEST 161",
    "title":"SEMANTIC DIRECTION SUBSPACE CAUSAL ASSAY",
    "model":MODEL_ID,
    "gpu":torch.cuda.get_device_name(0),
    "motor":{
        "layers":"L0-L19",
        "ivme":IVME,
        "sonum":SONUM,
        "zirve":ZIRVE,
        "taban":TABAN,
        "law":"frozen-norm direct dose",
        "training":False
    },
    "draws":{
        "in_span":IN_SPAN_DRAWS,
        "out_span":OUT_SPAN_DRAWS,
        "ambient":AMBIENT_DRAWS
    },
    "baseline_mean":BASE_MEAN,
    "geometry":GEOMETRY_ROWS,
    "reference_arms":REAL_ROWS,
    "in_span":IN_ROWS,
    "out_span":OUT_ROWS,
    "ambient":AMBIENT_ROWS,
    "statistics":STAT_ROWS,
    "energy_match":ENERGY_ROWS,
    "collateral":COLLATERAL_ROWS,
    "flags":FLAGS,
    "interpretation_contract":[
        "IN_SPAN versus OUT_SPAN is the primary causal contrast.",
        "All random directions are unit normalized before intervention.",
        "OUT_SPAN directions are explicitly projected outside the SEARCH semantic direction span.",
        "AMBIENT directions are unrestricted full-space random controls.",
        "All random directions are sign-oriented to the positive Fisher hemisphere.",
        "All arms use the same frozen-norm SEASC motor and L0-L19 DRA envelope.",
        "Physical intervention energy is measured separately.",
        "IN_SPAN greater than OUT_SPAN supports a causal advantage of the semantic direction subspace.",
        "IN_SPAN greater than AMBIENT tests against arbitrary high-dimensional perturbation.",
        "FISHER greater than IN_SPAN indicates that orientation inside the semantic span still matters.",
        "Failure of IN_SPAN to exceed OUT_SPAN rejects the simple semantic-subspace explanation.",
        "This assay does not test state-specific routing.",
        "Results are specific to this model, semantic axis, prompt family, dose range and SEARCH construction."
    ]
}

with open(ROOT / "REPORT.json", "w", encoding="utf-8") as f:
    json.dump(REPORT, f, indent=2, ensure_ascii=False)

# ==================================================================================================
# 22. FINAL REPORT
# ==================================================================================================

print("\n[15/15] COMPLETE")
print("=" * 150)
print("TEST 161 — FINAL SUMMARY")
print("=" * 150)

print(f"\nBaseline mean margin : {BASE_MEAN:+.6f}")

print("\nSEMANTIC SUBSPACE GEOMETRY")
print(f"  mean rank          : {np.mean(SPAN_RANK):.3f}")
print(f"  rank range         : {min(SPAN_RANK)}-{max(SPAN_RANK)}")
print(f"  Fisher capture     : {np.mean(FISHER_CAPTURE):.6f}")
print(f"  mean/Fisher cosine : {np.mean(MEAN_FISHER_COS):+.6f}")
print(f"  PC1/Fisher cosine  : {np.mean(PCA1_FISHER_COS):+.6f}")

print("\nDETERMINISTIC REFERENCES")
for scale in DOSE_SCALES:
    print(f"\nDOSE SCALE = {scale:.2f}")
    for method in ("FISHER", "BANK_MEAN", "PCA1"):
        r = next(x for x in REAL_ROWS if x["method"] == method and x["scale"] == scale)
        print(
            f"  {method:<12} | Csym={r['Csym']:+.6f} | "
            f"P+={r['positive_items']:02d}/{len(EVAL_ITEMS)} | E={r['energy']:.6f}"
        )

print("\nPRIMARY CAUSAL CONTRAST")
for r in STAT_ROWS:
    print(
        f"  scale={r['scale']:.2f} | IN={r['in_mean']:+.6f}±{r['in_sd']:.6f} | "
        f"OUT={r['out_mean']:+.6f}±{r['out_sd']:.6f} | "
        f"Δ={r['in_minus_out']:+.6f} | AUC={r['auc_in_vs_out']:.4f} | "
        f"p={r['p_in_gt_out']:.6f}"
    )

print("\nSECONDARY AMBIENT CONTRAST")
for r in STAT_ROWS:
    print(
        f"  scale={r['scale']:.2f} | IN={r['in_mean']:+.6f} | "
        f"AMBIENT={r['ambient_mean']:+.6f} | Δ={r['in_minus_ambient']:+.6f} | "
        f"AUC={r['auc_in_vs_ambient']:.4f} | p={r['p_in_gt_ambient']:.6f}"
    )

print("\nPHYSICAL ENERGY")
for r in ENERGY_ROWS:
    print(
        f"  scale={r['scale']:.2f} | OUT/IN={r['out_over_in']:.6f} | "
        f"AMBIENT/IN={r['ambient_over_in']:.6f}"
    )

print("\nDIAGNOSTIC FLAGS")
for flag in FLAGS:
    print(" ", flag)

print("\nINTERPRETATION RULE")
print("IN_SPAN > OUT_SPAN tests whether semantic-span membership itself has causal value.")
print("IN_SPAN > AMBIENT tests whether the result exceeds unrestricted random perturbation.")
print("FISHER > IN_SPAN means orientation inside the span still matters.")
print("IN_SPAN ~= OUT_SPAN rejects the simple semantic-subspace explanation.")

print("\nSaved:")
for name in [
    "BASELINE.csv",
    "REFERENCE_ARMS.csv",
    "IN_SPAN.csv",
    "OUT_SPAN.csv",
    "AMBIENT.csv",
    "SPAN_STATISTICS.csv",
    "ENERGY_MATCH.csv",
    "COLLATERAL.csv",
    "SUBSPACE_GEOMETRY.csv",
    "TEST161_SEMANTIC_SUBSPACE.npz",
    "REPORT.json"
]:
    print(" ", ROOT / name)

print("=" * 150)
print("TEST 161 COMPLETE")
print("=" * 150)
