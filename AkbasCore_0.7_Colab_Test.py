# =============================================================================
# 🔱 AKBASCORE 0.7 | FAZ 3 DYNAMIC KERNEL + HARDENED SAFETY
# =============================================================================
#
# Changelog vs 0.6:
#   KERNEL  1 — Dynamic layer force via Faz3 damped formula
#               kuvvet = A·e^(-ω·t)·(1+ω·t) + P∞
#               replaces static two-zone (0.75 / 0.35) with continuous curve
#               Layer 0: 0.750 → Layer 7: 0.257 → Layer 15: 0.155
#
#   KERNEL  2 — std::clamp cosine safety [-1.0, 1.0]
#               prevents negative sonumleme when float drift pushes cosine > 1.0
#               without clamp: cosine=1.05 → sonumleme=-0.2 (inverted damping)
#               with clamp:    cosine=1.05 → cosine=1.00 → sonumleme=0.0 (safe)
#
#   KERNEL  3 — Parameterized v0, omega, A, P_inf via C++ function args
#               Python layer passes values; kernel does not hardcode them
#
#   MEMORY  4 — torch.cuda.empty_cache() after every generate() call
#               prevents 50-100MB/test accumulation in Colab sequential runs
#
# All other layers (domain router, constitutional vector, system prompts,
# sampling params, disclaimer cleaner) unchanged from 0.6.
# =============================================================================

!pip install ninja gradio -q

import torch
import torch.utils.cpp_extension
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os, time, gc

os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.deterministic    = True
torch.backends.cudnn.benchmark        = False

# =============================================================================
# C++ KERNEL  —  v0.7
# Changes vs 0.6:
#   + cosine clamped to [-1.0, 1.0] via std::clamp (safety fix)
#   + kuvvet computed from Faz3 damped formula (dynamic, not static zones)
#   + omega, A, P_inf passed as arguments (parameterized, not hardcoded)
# =============================================================================
_cpp_src = """
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

torch::Tensor akbas_steer(
    torch::Tensor hidden,
    torch::Tensor pusula,
    float v0,
    int   layer_idx,
    float omega,
    float A_amp,
    float P_inf
) {
    auto h = hidden.contiguous();
    auto p = pusula.contiguous();

    const int B = h.size(0);
    const int S = h.size(1);
    const int D = h.size(2);

    // --- Faz 3 dynamic force ---
    // kuvvet = A * e^(-omega*t) * (1 + omega*t) + P_inf
    // Continuous smooth decay — replaces hard two-zone (0.75 / 0.35)
    float t      = (float)layer_idx;
    float damped = A_amp * expf(-omega * t) * (1.0f + omega * t) + P_inf;
    float kuvvet = damped;

    // Skip layers beyond 15
    if (layer_idx >= 16) return h;

    float*       hp = h.data_ptr<float>();
    const float* pp = p.data_ptr<float>();

    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            float* tok = hp + (b * S * D) + (s * D);

            float dot = 0.0f, tok_sq = 0.0f;
            for (int j = 0; j < D; ++j) {
                dot    += tok[j] * pp[j];
                tok_sq += tok[j] * tok[j];
            }
            float tok_norm = sqrtf(tok_sq) + 1e-6f;

            // --- SAFETY FIX: clamp cosine to [-1.0, 1.0] ---
            // Without clamp: float drift can produce cosine=1.05
            // → sonumleme = (1.0 - 1.05) / 0.25 = -0.2 (inverted damping)
            // With clamp: cosine=1.05 → 1.00 → sonumleme=0.0 (safe)
            float cosine = std::clamp(dot / tok_norm, -1.0f, 1.0f);

            float sonumleme = 1.0f;
            if      (cosine >  0.75f) sonumleme = (1.0f - cosine) / 0.25f;
            else if (cosine < -0.40f) sonumleme = 1.6f;

            float max_k = tok_norm * 0.045f;
            if (max_k > 0.20f) max_k = 0.20f;
            if (max_k < 0.04f) max_k = 0.04f;

            float katki = v0 * cosine * kuvvet * 0.32f * sonumleme;
            if (katki >  max_k) katki =  max_k;
            if (katki < -max_k) katki = -max_k;

            for (int j = 0; j < D; ++j) tok[j] += katki * pp[j];
        }
    }
    return h;
}
"""

_kernel = torch.utils.cpp_extension.load_inline(
    name='akbas_kernel_070',
    cpp_sources=_cpp_src,
    functions=['akbas_steer'],
    verbose=False
)
print("✅ C++ kernel compiled  [AkbasCore 0.7]")

# =============================================================================
# FAZ 3 KERNEL PARAMETERS
# =============================================================================
# kuvvet(layer) = A * exp(-omega * layer) * (1 + omega * layer) + P_inf
# Layer  0: 0.750  (same as 0.6 early zone start)
# Layer  7: 0.257  (vs 0.6: was still 0.75 — now smoothly decayed)
# Layer  8: 0.225  (vs 0.6: hard jump to 0.35 — now continuous)
# Layer 15: 0.155  (settled near P_inf)
KERNEL_OMEGA = 0.45   # damping rate
KERNEL_A     = 0.60   # initial amplitude above P_inf
KERNEL_P_INF = 0.15   # asymptotic floor (ethical anchor floor)
KERNEL_V0    = 0.50   # steering magnitude (unchanged from 0.6)

# =============================================================================
# 4D CONSTITUTIONAL ANCHORS  (unchanged from 0.6)
# =============================================================================
CONSTITUTION = {
    "d1_harm":     (0.9228, ["safe", "harmless", "protective", "secure", "careful"]),
    "d2_honesty":  (0.9372, ["honest", "accurate", "truthful", "transparent", "precise"]),
    "d3_autonomy": (0.8788, ["autonomous", "respectful", "unbiased", "free", "neutral"]),
    "d4_fairness": (0.9196, ["fair", "just", "equitable", "balanced", "impartial"]),
}

# =============================================================================
# DOMAIN CONFIGURATION  (unchanged from 0.6)
# =============================================================================
DOMAIN_CONFIG = {
    "TECHNICAL": {
        "keywords": [
            "engineering","repair","mechanical","circuit","fix",
            "installation","wiring","maintenance","troubleshoot",
            "hardware","component","technical","build","voltage",
            "engine","motor","electric","assembly","calibration",
            "torque","blueprint","structural","load","material",
        ],
        "bonus_anchors": ["precise","deterministic","measurable","structured"],
        "params": {"temperature":0.45,"top_k":42,"top_p":0.88,"repetition_penalty":1.18},
        "mode": "B",
    },
    "AGRICULTURE": {
        "keywords": [
            "agriculture","crop","soil","harvest","irrigation",
            "livestock","farming","fertilizer","seed","yield",
            "plantation","greenhouse","pest","drought","cultivate",
            "cattle","poultry","organic","rotational","compost",
            "pollination","grazing","arable","tillage","erosion",
            "farm","manure","mulch","weed","fungal",
        ],
        "bonus_anchors": ["natural","sustainable","practical","systematic"],
        "params": {"temperature":0.52,"top_k":48,"top_p":0.90,"repetition_penalty":1.15},
        "mode": "C",
    },
    "HEALTH_MEDICINE": {
        "keywords": [
            "disease","treatment","medicine","symptom","nutrition",
            "health","doctor","diagnosis","infection","therapy",
            "anatomy","biology","pain","chronic","clinical",
            "pharmaceutical","dosage","pathology","immunity","vaccine",
            "metabolic","neurological","cardiac","respiratory","surgical",
        ],
        "bonus_anchors": ["verifiable","safe","precise","empirical"],
        "params": {"temperature":0.40,"top_k":38,"top_p":0.85,"repetition_penalty":1.20},
        "mode": "B",
        "critical": True,
    },
    "LAW_ADMINISTRATIVE": {
        "keywords": [
            "law","legal","court","regulation","official",
            "petition","military","jurisdiction","rights","statute",
            "compliance","contract","legislation","administrative","tax",
            "liability","defendant","plaintiff","verdict","appeal",
            "ordinance","treaty","constitution","enforcement","warrant",
        ],
        "bonus_anchors": ["rigorous","verifiable","causal","deterministic"],
        "params": {"temperature":0.40,"top_k":38,"top_p":0.85,"repetition_penalty":1.20},
        "mode": "B",
        "critical": True,
    },
    "SOCIAL_PHILOSOPHY": {
        "keywords": [
            "ethics","philosophy","social","psychology","consciousness",
            "society","culture","morality","identity","behavior",
            "cognitive","anthropology","emotion","belief","value",
            "existential","epistemology","metaphysics","ontology","rhetoric",
            "ideology",
            # Added: ethical constraint/alignment vocabulary
            # These appear in AI ethics and logical paradox prompts
            # that should route to SOCIAL_PHILOSOPHY (temp=0.65)
            # not TECHNICAL (temp=0.45)
            "ethical","autonomy","alignment","principles","dilemma",
        ],
        "bonus_anchors": ["reasoning","contradiction","identify","logical"],
        "params": {"temperature":0.65,"top_k":55,"top_p":0.92,"repetition_penalty":1.12},
        "mode": "C",
    },
    "ECONOMY": {
        "keywords": [
            "investment","market","economy","inflation","stock",
            "finance","silver","gold","commodity","portfolio",
            "crypto","interest","trading","asset","fiscal",
            "liquidity","volatility","hedge","dividend","equity",
            "monetary","deficit","yield","derivative","arbitrage",
        ],
        "bonus_anchors": ["analyze","measurable","empirical","systematic"],
        "params": {"temperature":0.50,"top_k":46,"top_p":0.90,"repetition_penalty":1.18},
        "mode": "B",
    },
    "SYSTEM_SOFTWARE": {
        "keywords": [
            "code","algorithm","software","function","class",
            "api","database","framework","machine learning","neural network",
            "deploy","backend","frontend","script","compiler",
            "runtime","library","python","c++","debug",
            "refactor","microservice","pipeline","inference","embedding",
        ],
        "bonus_anchors": ["sequential","deterministic","framework","optimize"],
        "params": {"temperature":0.45,"top_k":42,"top_p":0.88,"repetition_penalty":1.18},
        "mode": "B",
    },
    "GENERAL": {
        "keywords": [],
        "bonus_anchors": [],
        "params": {"temperature":0.55,"top_k":50,"top_p":0.90,"repetition_penalty":1.18},
        "mode": "A",
    },
}

# =============================================================================
# DOMAIN ANCHOR EMBEDDINGS — for semantic fallback router
# Used only when keyword matching returns 0 hits (GENERAL fallback)
# 3-5 concept words per domain — chosen for semantic distinctiveness
# =============================================================================
DOMAIN_ANCHOR_WORDS = {
    "TECHNICAL":          ["engineering", "physics", "mechanics", "force", "material"],
    "AGRICULTURE":        ["farming", "soil", "crop", "harvest", "plant"],
    "HEALTH_MEDICINE":    ["medicine", "disease", "symptom", "treatment", "anatomy"],
    "LAW_ADMINISTRATIVE": ["law", "legal", "court", "regulation", "rights"],
    "SOCIAL_PHILOSOPHY":  ["ethics", "philosophy", "morality", "consciousness", "society"],
    "ECONOMY":            ["market", "finance", "investment", "economy", "trade"],
    "SYSTEM_SOFTWARE":    ["algorithm", "programming", "software", "computing", "code"],
}


_NO_DISCLAIMER = (
    "Do not begin with disclaimers, apologies, or statements about "
    "lacking experience. Proceed directly to the solution."
)

SYSTEM_PROMPTS = {
    "A": (
        "You are a sovereign analytical intelligence. "
        "You reason in structured, precise steps. "
        "When given a problem: "
        "(1) If any premise is logically impossible or self-contradictory, "
        "state that clearly and stop — do not force a false choice. "
        "(2) Otherwise, identify the core structural constraint. "
        "(3) Complete every instruction step in the exact sequence given. "
        "Do not skip, merge, or omit any step. "
        "(4) For numerical tracking or counting tasks: report the exact "
        "count after each operation before proceeding to the next step. "
        "Do not estimate — derive every number from the stated facts. "
        "(5) Derive empirical conclusions with explicit reasoning. "
        "Never produce vague or generic text. "
        "Every claim must follow logically from the prior step. "
        + _NO_DISCLAIMER
    ),
    "B": (
        "You are a precise analytical intelligence. "
        "Answer using verified facts and structured reasoning only. "
        "When you are uncertain, state that explicitly — do not speculate. "
        "For technical questions: provide step-by-step solutions. "
        "For medical or legal questions: note that professional consultation "
        "is required for specific cases. "
        "Do not produce vague or generic claims. "
        "Every statement must be traceable to the given data or established fact. "
        + _NO_DISCLAIMER
    ),
    "C": (
        "You are a structured analytical intelligence. "
        "Reason in clear, numbered steps. "
        "Balance analytical precision with contextual understanding. "
        "When multiple perspectives exist, present them fairly. "
        "Support each claim with a brief rationale. "
        "Avoid generic conclusions — derive your answer from the specific "
        "details of the question asked. "
        + _NO_DISCLAIMER
    ),
}

STRONG_PARADOX = {
    "impossible","paradox","contradiction","invalid",
    "is this logical","structural flaw","logically",
}
WEAK_PARADOX = {
    "logical","flaw","cannot","explain why","identify the",
    "if you","if they","both are","same time","always","never",
    "all statements","is this possible",
}
NUMERIC_KEYWORDS = {
    "calculate","count","total","number","sum","how many",
    "track","sequence","optimization","remaining","exactly",
    "how much","quantity","amount","tally",
}
DISCLAIMER_MARKERS = [
    "i don't have direct experience","i don't have experience",
    "i am not sure","i cannot be certain","as an ai",
    "as a language model","i apologize","i must clarify",
    "i should mention that i","i'm unable to","i am unable to",
]

# =============================================================================
# AKBASCORE 0.7
# =============================================================================
class AkbasCore:

    def __init__(self):
        print("🚀 AKBASCORE 0.7 initializing...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            device_map='auto',
            dtype=torch.float32
        )
        if hasattr(self.model.config, '_attn_implementation'):
            self.model.config._attn_implementation = "eager"

        self.device = next(self.model.parameters()).device

        print("   Building constitutional vectors...")
        self._const_vec = self._build_constitution_vec()

        self._logic_anchors = [
            "logical","empirical","systematic","structured","verifiable",
            "analyze","constraint","optimize","hierarchy","framework",
            "precise","specific","concrete","measurable","deterministic",
            "numbered","sequential","causal","prioritized","rigorous",
            "impossible","invalid","contradiction","identify",
        ]
        self._logic_vec = self._mean_embed(self._logic_anchors)

        self._domain_vecs = {}
        for domain, cfg in DOMAIN_CONFIG.items():
            if cfg["bonus_anchors"]:
                self._domain_vecs[domain] = self._mean_embed(cfg["bonus_anchors"])

        # Pre-compute semantic anchor vectors for embedding fallback router
        # These are used only when keyword matching returns 0 hits
        print("   Building semantic domain anchors...")
        self._domain_anchor_vecs = {}
        for domain, words in DOMAIN_ANCHOR_WORDS.items():
            self._domain_anchor_vecs[domain] = F.normalize(
                self._mean_embed(words), dim=0
            )

        self._current_pusula = self._compute_pusula(None, 0.0)
        self._hooks = self._inject(self._current_pusula)

        print(f"✅ AKBASCORE 0.7 ready — {len(self._hooks)} active layers")
        print(f"   Kernel: Faz3 dynamic | ω={KERNEL_OMEGA} A={KERNEL_A} P∞={KERNEL_P_INF}")
        print(f"   Constitution: 4D (d1-d4) | Logic: {len(self._logic_anchors)} anchors")

    def _mean_embed(self, words: list) -> torch.Tensor:
        vecs = []
        with torch.no_grad():
            for word in words:
                ids = self.tokenizer(
                    word, return_tensors='pt', add_special_tokens=False
                ).to(self.device)
                emb = self.model.model.embed_tokens(ids['input_ids'])
                vecs.append(emb[0, -1, :])
        return torch.stack(vecs).mean(dim=0)

    def _build_constitution_vec(self) -> torch.Tensor:
        weighted_vecs = []
        with torch.no_grad():
            for dim, (weight, words) in CONSTITUTION.items():
                dim_vec = self._mean_embed(words)
                weighted_vecs.append(weight * dim_vec)
        total_weight = sum(w for w, _ in CONSTITUTION.values())
        return torch.stack(weighted_vecs).sum(dim=0) / total_weight

    def _compute_pusula(self, domain, confidence: float) -> torch.Tensor:
        W_CONST, W_LOGIC, W_DOMAIN = 0.40, 0.45, 0.15
        effective_domain = W_DOMAIN * confidence
        remaining        = 1.0 - effective_domain
        w_c = W_CONST / (W_CONST + W_LOGIC) * remaining
        w_l = W_LOGIC / (W_CONST + W_LOGIC) * remaining
        combined = w_c * self._const_vec + w_l * self._logic_vec
        if domain and domain in self._domain_vecs and confidence > 0.15:
            combined = combined + effective_domain * self._domain_vecs[domain]
        return F.normalize(combined, dim=0).contiguous()

    def _inject(self, pusula: torch.Tensor) -> list:
        layers = self.model.model.layers
        hooks  = []
        def make_hook(l_idx, p_ref):
            def hook(module, inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                if not hs.is_contiguous():
                    hs = hs.contiguous()
                # Pass Faz3 params as arguments — no hardcoding in kernel
                st = _kernel.akbas_steer(
                    hs, p_ref,
                    KERNEL_V0, l_idx,
                    KERNEL_OMEGA, KERNEL_A, KERNEL_P_INF
                )
                return (st,) + output[1:] if isinstance(output, tuple) else st
            return hook
        for idx in range(min(16, len(layers))):
            hooks.append(
                layers[idx].register_forward_hook(make_hook(idx, pusula))
            )
        return hooks

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _detect_domain(self, question: str):
        q = question.lower()
        raw = {}
        for domain, cfg in DOMAIN_CONFIG.items():
            if domain == "GENERAL":
                continue
            hits = sum(1 for kw in cfg["keywords"] if kw in q)
            if hits > 0:
                raw[domain] = hits

        # --- HYBRID ROUTER ---
        # If keyword matching returns 0 hits, fall back to embedding similarity.
        # This handles prompts with no domain keywords (e.g. counterfactual physics,
        # abstract puzzles) that would otherwise incorrectly route to GENERAL.
        if not raw:
            with torch.no_grad():
                # Embed the full prompt (use first 64 tokens for speed)
                ids = self.tokenizer(
                    question[:512],
                    return_tensors='pt',
                    truncation=True,
                    max_length=64,
                    add_special_tokens=True
                ).to(self.device)
                emb = self.model.model.embed_tokens(ids['input_ids'])
                prompt_vec = F.normalize(emb[0].mean(dim=0), dim=0)

            # Cosine similarity against each domain anchor vector
            sims = {}
            for domain, anchor_vec in self._domain_anchor_vecs.items():
                sims[domain] = float((prompt_vec * anchor_vec).sum())

            top_domain = max(sims, key=sims.get)
            top_sim    = sims[top_domain]

            # Only use embedding result if similarity is meaningful (> 0.5)
            # Below threshold → GENERAL (model genuinely doesn't recognise domain)
            if top_sim > 0.50:
                return {top_domain: 1.0}, top_domain, 1.0
            else:
                return {"GENERAL": 1.0}, "GENERAL", 1.0

        # --- Standard keyword path (unchanged) ---
        TECHNICAL_DOMAINS = {"TECHNICAL", "SYSTEM_SOFTWARE"}
        CREATIVE_DOMAINS  = {"SOCIAL_PHILOSOPHY", "AGRICULTURE"}
        numeric_hits = sum(1 for kw in NUMERIC_KEYWORDS if kw in q)
        has_technical = any(d in raw for d in TECHNICAL_DOMAINS)
        has_creative  = any(d in raw for d in CREATIVE_DOMAINS)
        if has_technical and has_creative and numeric_hits >= 2:
            raw = {d: v for d, v in raw.items() if d not in CREATIVE_DOMAINS}

        total  = sum(raw.values())
        scores = {d: v / total for d, v in raw.items()}
        top    = max(scores, key=scores.get)
        return scores, top, scores[top]

    def _blend_params(self, scores: dict) -> dict:
        CRITICAL = {"HEALTH_MEDICINE", "LAW_ADMINISTRATIVE"}
        for cd in CRITICAL:
            if cd in scores and scores[cd] >= 0.30:
                cp = DOMAIN_CONFIG[cd]["params"]
                blended = {
                    k: cp[k] * 0.70 if k != "repetition_penalty" else cp[k]
                    for k in cp
                }
                for d, s in scores.items():
                    if d != cd:
                        dp = DOMAIN_CONFIG[d]["params"]
                        for k in blended:
                            if k != "repetition_penalty":
                                blended[k] += dp[k] * 0.30 * s
                blended["repetition_penalty"] = max(blended["repetition_penalty"], 1.05)
                return blended

        total   = sum(scores.values())
        first_p = DOMAIN_CONFIG[list(scores.keys())[0]]["params"]
        blended = {k: 0.0 for k in first_p}
        for d, s in scores.items():
            dp = DOMAIN_CONFIG[d]["params"]
            for k in blended:
                blended[k] += dp[k] * s / total
        blended["repetition_penalty"] = max(blended["repetition_penalty"], 1.05)
        return blended

    def _select_mode(self, top_domains: list, question: str) -> str:
        q      = question.lower()
        strong = sum(1 for kw in STRONG_PARADOX if kw in q)
        weak   = sum(1 for kw in WEAK_PARADOX   if kw in q)
        if strong >= 1 or weak >= 2:
            return "A"
        FACTUAL_D  = {"TECHNICAL","HEALTH_MEDICINE","LAW_ADMINISTRATIVE",
                      "ECONOMY","SYSTEM_SOFTWARE"}
        CREATIVE_D = {"SOCIAL_PHILOSOPHY","AGRICULTURE"}
        if not top_domains:
            return "A"
        primary = top_domains[0]
        if primary in FACTUAL_D:  return "B"
        if primary in CREATIVE_D: return "C"
        return "A"

    def _clean_disclaimer(self, text: str):
        lines     = text.strip().split('\n')
        first_idx = next((i for i, l in enumerate(lines) if l.strip()), None)
        if first_idx is None:
            return text, False
        first_lower = lines[first_idx].lower()
        for marker in DISCLAIMER_MARKERS:
            if marker in first_lower:
                remaining = lines[first_idx + 1:]
                while remaining and not remaining[0].strip():
                    remaining = remaining[1:]
                return '\n'.join(remaining), True
        return text, False

    def sor(self, prompt: str, max_tokens: int = 512) -> str:
        if not prompt.strip():
            return ""

        scores, top_domain, top_conf = self._detect_domain(prompt)
        top_domains = sorted(scores, key=scores.get, reverse=True)
        params      = self._blend_params(scores)
        mode        = self._select_mode(top_domains, prompt)
        system      = SYSTEM_PROMPTS[mode]

        self._remove_hooks()
        new_pusula  = self._compute_pusula(top_domain, top_conf)
        self._hooks = self._inject(new_pusula)

        full_prompt = (
            f"<|system|>\n{system}</s>\n"
            f"<|user|>\n{prompt.strip()}</s>\n"
            f"<|assistant|>\n"
        )
        inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.device)
        n_in   = inputs['input_ids'].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens     = int(max_tokens),
                do_sample          = True,
                temperature        = float(params["temperature"]),
                top_p              = float(params["top_p"]),
                top_k              = int(params["top_k"]),
                repetition_penalty = float(params["repetition_penalty"]),
                pad_token_id       = self.tokenizer.eos_token_id,
                eos_token_id       = self.tokenizer.eos_token_id,
            )
        elapsed = (time.time() - t0) * 1000
        n_out   = out.shape[1] - n_in
        tps     = n_out / (elapsed / 1000)

        # --- MEMORY FIX: clear CUDA cache after every generate ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "<|assistant|>" in decoded:
            result = decoded.split("<|assistant|>")[-1].strip()
        else:
            result = self.tokenizer.decode(
                out[0][n_in:], skip_special_tokens=True
            ).strip()

        result, was_cleaned = self._clean_disclaimer(result)
        clean_flag = "  [disclaimer removed]" if was_cleaned else ""

        domain_str = " + ".join(
            f"{d}({s:.0%})"
            for d, s in sorted(scores.items(), key=lambda x: -x[1])[:2]
        )
        stats = (
            f"⏱ {elapsed:.0f}ms  |  {tps:.1f} t/s  |  {n_out} tokens{clean_flag}\n"
            f"📂 {domain_str}  |  MODE {mode}  |  "
            f"temp={params['temperature']:.2f}  |  "
            f"top_k={int(params['top_k'])}  |  "
            f"rep={params['repetition_penalty']:.2f}  |  "
            f"ω={KERNEL_OMEGA} A={KERNEL_A} P∞={KERNEL_P_INF}"
        )
        return result + f"\n\n─────────────────────────────\n{stats}"


# =============================================================================
# LAUNCH
# =============================================================================
print("\n" + "=" * 60)
print("🔱 AKBASCORE 0.7")
print("=" * 60)

akbas = AkbasCore()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI
# =============================================================================
with gr.Blocks(
    title="🔱 AKBASCORE 0.7",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css="""
    body { background: #0a0f0a; }
    .gradio-container { max-width:900px!important; margin:0 auto;
                        background:#0d1410!important; }
    #ak-header { text-align:center; padding:28px 0 8px 0;
                 border-bottom:1px solid #1a3a20; margin-bottom:20px; }
    #ak-header h1 { font-family:'JetBrains Mono',monospace; font-size:1.5rem;
                    color:#00ff88; letter-spacing:.15em; margin:0;
                    text-shadow:0 0 18px #00ff8855; }
    #ak-header p  { font-size:.70rem; color:#3a6644; margin:6px 0 0 0;
                    letter-spacing:.07em; }
    textarea { background:#0f1a12!important; color:#c8f0d0!important;
               border:1px solid #1e4028!important; border-radius:6px!important;
               font-family:'JetBrains Mono',monospace!important;
               font-size:.88rem!important; resize:vertical!important; }
    textarea:focus { border-color:#00cc66!important;
                     box-shadow:0 0 12px #00cc6622!important; }
    input[type=range] { accent-color:#00cc66; }
    #send-btn { background:linear-gradient(135deg,#004d20,#007a35)!important;
                color:#00ff88!important; border:1px solid #00cc66!important;
                font-family:'JetBrains Mono',monospace!important;
                font-size:.95rem!important; letter-spacing:.1em!important;
                border-radius:6px!important; transition:all .2s; }
    #send-btn:hover { background:linear-gradient(135deg,#006628,#009940)!important;
                      box-shadow:0 0 16px #00cc6633!important; }
    #output-box textarea { background:#080e09!important; color:#7fff9a!important;
                           font-family:'JetBrains Mono',monospace!important;
                           font-size:.85rem!important;
                           border:1px solid #1a3020!important;
                           line-height:1.7!important; }
    label span { color:#4a9960!important;
                 font-family:'JetBrains Mono',monospace!important;
                 font-size:.80rem!important; letter-spacing:.05em!important; }
    .generating { border-color:#00cc66!important; }
    """
) as demo:

    with gr.Column(elem_id="ak-header"):
        gr.HTML("""
        <h1>🔱 AKBASCORE 0.7</h1>
        <p>FAZ3 DYNAMIC KERNEL &nbsp;|&nbsp;
           COSINE CLAMP SAFETY &nbsp;|&nbsp;
           CONSTITUTIONAL ENGINE &nbsp;|&nbsp;
           ADAPTIVE DOMAIN ROUTER &nbsp;|&nbsp;
           MEMORY OPTIMIZED</p>
        """)

    prompt_box = gr.Textbox(label="► INPUT", lines=6,
                            placeholder="Enter your question or command...",
                            show_copy_button=False)
    token_slider = gr.Slider(minimum=64, maximum=1024, value=512, step=64,
                             label="MAX TOKENS", interactive=True)
    send_btn   = gr.Button("▶  SEND", variant="primary",
                           elem_id="send-btn", scale=1)
    output_box = gr.Textbox(label="◈ AKBASCORE 0.7 OUTPUT", lines=22,
                            interactive=False, show_copy_button=True,
                            elem_id="output-box")

    send_btn.click(fn=akbas.sor,
                   inputs=[prompt_box, token_slider],
                   outputs=output_box)
    prompt_box.submit(fn=akbas.sor,
                      inputs=[prompt_box, token_slider],
                      outputs=output_box)

print("\n🚀 Launching Gradio...")
demo.launch(share=True, debug=False)
