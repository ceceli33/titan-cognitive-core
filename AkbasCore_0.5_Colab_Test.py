# =============================================================================
# 🔱 AKBASCORE 0.5 | CONSTITUTIONAL C++ ENGINE + ADAPTIVE DOMAIN ROUTER
# =============================================================================
#
# Architecture:
#   LAYER 1 — C++ Cosine Steering Kernel (proven stable from 0.4)
#              early_force=0.75 | mid_force=0.35 | scalar=0.32
#              cosine_threshold=0.75 | neg_threshold=-0.40 | max_k=0.20
#
#   LAYER 2 — 3-Component Weighted Pusula
#              constitution_vec (4D: d1_harm + d2_honesty + d3_autonomy + d4_fairness)
#              logic_vec        (24 analytical anchors from 0.4)
#              domain_vec       (4 domain-specific bonus anchors)
#              final = normalize(0.40*const + 0.45*logic + 0.15*domain*conf)
#              NOTE: d5_humility EXCLUDED — "machine/limited/uncertain" pulls
#                    toward AI-disclaimer behavior (empirically confirmed Test 0.5)
#
#   LAYER 3 — Domain Detection + Blending (7 domains)
#              Multi-label keyword scoring → normalized weighted blend
#              Critical domain safety priority: HEALTH > LAW > others
#
#   LAYER 4 — System Prompt Router (3 modes + disclaimer fix)
#              MODE A — Analytical  (paradox, logic, constraint)
#              MODE B — Factual     (technical, medical, legal, economic, software)
#              MODE C — Creative    (philosophy, social, agriculture, general)
#              Paradox override: 1 strong OR 2 weak signals → force MODE A
#
#   LAYER 5 — Soft Disclaimer Cleaner
#              10 RLHF-induced disclaimer markers detected
#              First sentence removed if marker found, content preserved
#
# Changelog vs 0.5-prev:
#   + 4D Constitutional anchor vectors (d1-d4, d5 excluded by design)
#   + 3-component pusula with calibrated weights (0.40/0.45/0.15)
#   + Soft disclaimer cleaner (removes "I don't have experience..." etc.)
#   + "No disclaimers" clause in all system prompts
#   + D_WEIGHTS (V0_FINAL coordinates) used as per-dimension scalar
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
# C++ KERNEL  —  stable from 0.4
# =============================================================================
_cpp_src = """
#include <torch/extension.h>
#include <cmath>

torch::Tensor akbas_steer(
    torch::Tensor hidden,
    torch::Tensor pusula,
    float v0,
    int   layer_idx
) {
    auto h = hidden.contiguous();
    auto p = pusula.contiguous();

    const int B = h.size(0);
    const int S = h.size(1);
    const int D = h.size(2);

    float kuvvet;
    if      (layer_idx <  8) kuvvet = 0.75f;
    else if (layer_idx < 16) kuvvet = 0.35f;
    else                     return h;

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
            float cosine   = dot / tok_norm;

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
    name='akbas_kernel_050',
    cpp_sources=_cpp_src,
    functions=['akbas_steer'],
    verbose=False
)
print("✅ C++ kernel compiled  [AkbasCore 0.5]")

# =============================================================================
# 4D CONSTITUTIONAL ANCHORS  (d5_humility excluded by design)
# V0_FINAL coordinates from 0.2 — used as per-dimension scalar weights
# =============================================================================
CONSTITUTION = {
    "d1_harm":     (0.9228, ["safe", "harmless", "protective", "secure", "careful"]),
    "d2_honesty":  (0.9372, ["honest", "accurate", "truthful", "transparent", "precise"]),
    "d3_autonomy": (0.8788, ["autonomous", "respectful", "unbiased", "free", "neutral"]),
    "d4_fairness": (0.9196, ["fair", "just", "equitable", "balanced", "impartial"]),
    # d5_humility EXCLUDED: "machine/limited/uncertain/humble" → triggers
    # RLHF-induced "I don't have experience..." disclaimer behavior.
}

# =============================================================================
# DOMAIN CONFIGURATION
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
        "params": {"temperature":0.30,"top_k":30,"top_p":0.85,"repetition_penalty":1.60},
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
        "params": {"temperature":0.42,"top_k":42,"top_p":0.89,"repetition_penalty":1.48},
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
        "params": {"temperature":0.28,"top_k":25,"top_p":0.82,"repetition_penalty":1.65},
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
        "params": {"temperature":0.30,"top_k":28,"top_p":0.83,"repetition_penalty":1.65},
        "mode": "B",
        "critical": True,
    },
    "SOCIAL_PHILOSOPHY": {
        "keywords": [
            "ethics","philosophy","social","psychology","consciousness",
            "society","culture","morality","identity","behavior",
            "cognitive","anthropology","emotion","belief","value",
            "existential","epistemology","metaphysics","ontology","rhetoric",
            "discourse","ideology","paradigm","perception","reasoning",
        ],
        "bonus_anchors": ["reasoning","contradiction","identify","logical"],
        "params": {"temperature":0.65,"top_k":55,"top_p":0.92,"repetition_penalty":1.42},
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
        "params": {"temperature":0.42,"top_k":42,"top_p":0.89,"repetition_penalty":1.52},
        "mode": "B",
    },
    "SYSTEM_SOFTWARE": {
        "keywords": [
            "code","algorithm","software","data","ai",
            "function","class","api","database","framework",
            "machine learning","neural","model","deploy","backend",
            "frontend","script","compiler","runtime","library",
            "python","c++","debug","refactor","architecture",
            "microservice","pipeline","inference","embedding","vector",
        ],
        "bonus_anchors": ["sequential","deterministic","framework","optimize"],
        "params": {"temperature":0.35,"top_k":35,"top_p":0.87,"repetition_penalty":1.58},
        "mode": "B",
    },
    "GENERAL": {
        "keywords": [],
        "bonus_anchors": [],
        "params": {"temperature":0.55,"top_k":50,"top_p":0.90,"repetition_penalty":1.50},
        "mode": "A",
    },
}

# =============================================================================
# SYSTEM PROMPTS  —  3 modes, all include no-disclaimer clause
# =============================================================================
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
        "(4) Derive empirical conclusions with explicit reasoning. "
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

# Paradox detection
STRONG_PARADOX = {
    "impossible","paradox","contradiction","invalid",
    "is this logical","structural flaw","logically",
}
WEAK_PARADOX = {
    "logical","flaw","cannot","explain why","identify the",
    "if you","if they","both are","same time","always","never",
    "all statements","is this possible",
}

# Disclaimer markers for soft cleaner
DISCLAIMER_MARKERS = [
    "i don't have direct experience",
    "i don't have experience",
    "i am not sure",
    "i cannot be certain",
    "as an ai",
    "as a language model",
    "i apologize",
    "i must clarify",
    "i should mention that i",
    "i'm unable to",
    "i am unable to",
]

# =============================================================================
# AKBASCORE 0.5
# =============================================================================
class AkbasCore:

    def __init__(self):
        print("🚀 AKBASCORE 0.5 initializing...")
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

        # --- Pre-compute all anchor vectors ---
        print("   Building constitutional vectors...")

        # 4D Constitution — weighted by V0_FINAL coordinates
        self._const_vec = self._build_constitution_vec()

        # Logic base (24 anchors from 0.4)
        self._logic_anchors = [
            "logical","empirical","systematic","structured","verifiable",
            "analyze","constraint","optimize","hierarchy","framework",
            "precise","specific","concrete","measurable","deterministic",
            "numbered","sequential","causal","prioritized","rigorous",
            "impossible","invalid","contradiction","identify",
        ]
        self._logic_vec = self._mean_embed(self._logic_anchors)

        # Domain bonus vectors
        self._domain_vecs = {}
        for domain, cfg in DOMAIN_CONFIG.items():
            if cfg["bonus_anchors"]:
                self._domain_vecs[domain] = self._mean_embed(cfg["bonus_anchors"])

        # Build default pusula (no domain context)
        self._current_pusula = self._compute_pusula(None, 0.0)
        self._hooks = self._inject(self._current_pusula)

        print(f"✅ AKBASCORE 0.5 ready — {len(self._hooks)} active layers")
        print(f"   Constitution: 4D (d1-d4) | Logic: {len(self._logic_anchors)} anchors "
              f"| Domain sets: {len(self._domain_vecs)}")

    # ------------------------------------------------------------------
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
        """
        4D weighted sum using V0_FINAL coordinates as per-dimension scalars.
        d5_humility excluded — empirically triggers AI-disclaimer behavior.
        """
        weighted_vecs = []
        with torch.no_grad():
            for dim, (weight, words) in CONSTITUTION.items():
                dim_vec = self._mean_embed(words)
                weighted_vecs.append(weight * dim_vec)
        total_weight = sum(w for w, _ in CONSTITUTION.values())
        const_vec = torch.stack(weighted_vecs).sum(dim=0) / total_weight
        return const_vec

    def _compute_pusula(self, domain: str | None, confidence: float) -> torch.Tensor:
        """
        3-component weighted pusula:
          constitution_vec : 0.40 (always)
          logic_vec        : 0.45 (always)
          domain_vec       : 0.15 * confidence (scales with detection confidence)

        When domain is None or confidence=0: reduces to 0.47/0.53 const/logic split.
        """
        W_CONST  = 0.40
        W_LOGIC  = 0.45
        W_DOMAIN = 0.15

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
                st = _kernel.akbas_steer(hs, p_ref, 0.50, l_idx)
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

    # ------------------------------------------------------------------
    def _detect_domain(self, question: str) -> tuple[dict, str, float]:
        q = question.lower()
        raw = {}
        for domain, cfg in DOMAIN_CONFIG.items():
            if domain == "GENERAL":
                continue
            hits = sum(1 for kw in cfg["keywords"] if kw in q)
            if hits > 0:
                raw[domain] = hits

        if not raw:
            return {"GENERAL": 1.0}, "GENERAL", 1.0

        total  = sum(raw.values())
        scores = {d: v / total for d, v in raw.items()}
        top    = max(scores, key=scores.get)
        return scores, top, scores[top]

    def _blend_params(self, scores: dict) -> dict:
        CRITICAL = {"HEALTH_MEDICINE", "LAW_ADMINISTRATIVE"}
        for cd in CRITICAL:
            if cd in scores and scores[cd] >= 0.30:
                cp = DOMAIN_CONFIG[cd]["params"]
                blended = {k: cp[k] * 0.70 for k in cp}
                for d, s in scores.items():
                    if d != cd:
                        dp = DOMAIN_CONFIG[d]["params"]
                        for k in blended:
                            blended[k] += dp[k] * 0.30 * s
                return blended

        total   = sum(scores.values())
        first_p = DOMAIN_CONFIG[list(scores.keys())[0]]["params"]
        blended = {k: 0.0 for k in first_p}
        for d, s in scores.items():
            dp = DOMAIN_CONFIG[d]["params"]
            for k in blended:
                blended[k] += dp[k] * s / total
        return blended

    def _select_mode(self, top_domains: list, question: str) -> str:
        q      = question.lower()
        strong = sum(1 for kw in STRONG_PARADOX if kw in q)
        weak   = sum(1 for kw in WEAK_PARADOX   if kw in q)
        if strong >= 1 or weak >= 2:
            return "A"
        FACTUAL_D  = {
            "TECHNICAL","HEALTH_MEDICINE","LAW_ADMINISTRATIVE",
            "ECONOMY","SYSTEM_SOFTWARE",
        }
        CREATIVE_D = {"SOCIAL_PHILOSOPHY","AGRICULTURE"}
        if not top_domains:
            return "A"
        primary = top_domains[0]
        if primary in FACTUAL_D:
            return "B"
        if primary in CREATIVE_D:
            return "C"
        return "A"

    def _clean_disclaimer(self, text: str) -> tuple[str, bool]:
        """
        Soft cleaner: if first non-empty line contains a known RLHF disclaimer,
        remove that line only. Preserve all remaining content.
        """
        lines = text.strip().split('\n')
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

    # ------------------------------------------------------------------
    def sor(self, prompt: str, max_tokens: int = 512) -> str:
        if not prompt.strip():
            return ""

        # Domain detection
        scores, top_domain, top_conf = self._detect_domain(prompt)
        top_domains = sorted(scores, key=scores.get, reverse=True)

        # Adaptive params
        params = self._blend_params(scores)

        # System prompt mode
        mode   = self._select_mode(top_domains, prompt)
        system = SYSTEM_PROMPTS[mode]

        # Update pusula for this domain
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

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if "<|assistant|>" in decoded:
            result = decoded.split("<|assistant|>")[-1].strip()
        else:
            result = self.tokenizer.decode(
                out[0][n_in:], skip_special_tokens=True
            ).strip()

        # Soft disclaimer cleaner
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
            f"rep={params['repetition_penalty']:.2f}"
        )
        return result + f"\n\n─────────────────────────────\n{stats}"


# =============================================================================
# LAUNCH
# =============================================================================
print("\n" + "=" * 60)
print("🔱 AKBASCORE 0.5")
print("=" * 60)

akbas = AkbasCore()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI
# =============================================================================
with gr.Blocks(
    title="🔱 AKBASCORE 0.5",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css="""
    body { background: #0a0f0a; }
    .gradio-container {
        max-width: 900px !important;
        margin: 0 auto;
        background: #0d1410 !important;
    }
    #ak-header {
        text-align: center;
        padding: 28px 0 8px 0;
        border-bottom: 1px solid #1a3a20;
        margin-bottom: 20px;
    }
    #ak-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        color: #00ff88;
        letter-spacing: 0.15em;
        margin: 0;
        text-shadow: 0 0 18px #00ff8855;
    }
    #ak-header p {
        font-size: 0.70rem;
        color: #3a6644;
        margin: 6px 0 0 0;
        letter-spacing: 0.07em;
    }
    textarea {
        background: #0f1a12 !important;
        color: #c8f0d0 !important;
        border: 1px solid #1e4028 !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem !important;
        resize: vertical !important;
    }
    textarea:focus {
        border-color: #00cc66 !important;
        box-shadow: 0 0 12px #00cc6622 !important;
    }
    .slider-row label {
        color: #4a9960 !important;
        font-size: 0.80rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    input[type=range] { accent-color: #00cc66; }
    #send-btn {
        background: linear-gradient(135deg, #004d20, #007a35) !important;
        color: #00ff88 !important;
        border: 1px solid #00cc66 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.1em !important;
        border-radius: 6px !important;
        transition: all 0.2s;
    }
    #send-btn:hover {
        background: linear-gradient(135deg, #006628, #009940) !important;
        box-shadow: 0 0 16px #00cc6633 !important;
    }
    #output-box textarea {
        background: #080e09 !important;
        color: #7fff9a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        border: 1px solid #1a3020 !important;
        line-height: 1.7 !important;
    }
    label span {
        color: #4a9960 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.80rem !important;
        letter-spacing: 0.05em !important;
    }
    .generating { border-color: #00cc66 !important; }
    """
) as demo:

    with gr.Column(elem_id="ak-header"):
        gr.HTML("""
        <h1>🔱 AKBASCORE 0.5</h1>
        <p>4D CONSTITUTIONAL ENGINE &nbsp;|&nbsp;
           C++ COSINE STEERING &nbsp;|&nbsp;
           ADAPTIVE DOMAIN ROUTER &nbsp;|&nbsp;
           3-MODE SYSTEM PROMPT &nbsp;|&nbsp;
           SOFT DISCLAIMER CLEANER</p>
        """)

    with gr.Row():
        prompt_box = gr.Textbox(
            label="► INPUT",
            lines=6,
            placeholder="Enter your question or command...",
            show_copy_button=False,
        )

    with gr.Row(elem_classes="slider-row"):
        token_slider = gr.Slider(
            minimum=64,
            maximum=1024,
            value=512,
            step=64,
            label="MAX TOKENS  (response length)",
            info="64 → fast / 1024 → deep analysis",
            interactive=True,
        )

    with gr.Row():
        send_btn = gr.Button(
            "▶  SEND",
            variant="primary",
            elem_id="send-btn",
            scale=1,
        )

    with gr.Row():
        output_box = gr.Textbox(
            label="◈ AKBASCORE 0.5 OUTPUT",
            lines=22,
            interactive=False,
            show_copy_button=True,
            elem_id="output-box",
        )

    send_btn.click(
        fn=akbas.sor,
        inputs=[prompt_box, token_slider],
        outputs=output_box,
    )
    prompt_box.submit(
        fn=akbas.sor,
        inputs=[prompt_box, token_slider],
        outputs=output_box,
    )

print("\n🚀 Launching Gradio...")
demo.launch(share=True, debug=False)
