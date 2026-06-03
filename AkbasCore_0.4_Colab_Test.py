# =============================================================================
# 🔱 AKBASCORE 0.4 | BALANCED STEERING + ADAPTIVE SAMPLING
# =============================================================================
# Changelog vs 0.3 / 0.5.1:
#   KERNEL  — early force 0.95 → 0.75 | mid force 0.50 → 0.35
#             cosine damping threshold 0.70 → 0.75 (more tolerant)
#             negative threshold  -0.35 → -0.40
#             contribution scalar  0.38 → 0.32  (softer push)
#             max_k upper cap      0.22 → 0.20
#   ANCHOR  — 20 → 24  (+impossible, +invalid, +contradiction, +identify)
#   SAMPLING — temperature 0.45 → 0.55 | top_p 0.85 → 0.90
#              top_k 40 → 50 | repetition_penalty 1.15 → 1.18
#   SYSTEM  — "Titan" removed | paradox-awareness clause added
#              explicit instruction-completion clause added
# =============================================================================

!pip install ninja gradio -q

import torch
import torch.utils.cpp_extension
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import os, time, gc

os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cudnn.deterministic    = True
torch.backends.cudnn.benchmark        = False

# =============================================================================
# C++ KERNEL  —  v0.4
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

    // 0.4 CHANGE: reduced force on both zones
    // early layers (0-7): 0.95 → 0.75  — less aggressive structural lock-in
    // mid   layers (8-15): 0.50 → 0.35 — softer mid-layer reinforcement
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

            // 0.4 CHANGE: raised positive threshold 0.70 → 0.75
            //             lowered negative threshold -0.35 → -0.40
            //             Both changes give the model more natural headroom
            float sonumleme = 1.0f;
            if      (cosine >  0.75f) sonumleme = (1.0f - cosine) / 0.25f;
            else if (cosine < -0.40f) sonumleme = 1.6f;

            // 0.4 CHANGE: upper cap 0.22 → 0.20 (more conservative)
            float max_k = tok_norm * 0.045f;
            if (max_k > 0.20f) max_k = 0.20f;
            if (max_k < 0.04f) max_k = 0.04f;

            // 0.4 CHANGE: scalar 0.38 → 0.32 (softer contribution)
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
    name='akbas_kernel_040',
    cpp_sources=_cpp_src,
    functions=['akbas_steer'],
    verbose=False
)
print("✅ C++ kernel compiled  [AkbasCore 0.4]")

# =============================================================================
# AKBASCORE 0.4
# =============================================================================
class AkbasCore:

    # 0.4 CHANGE: "Titan" removed entirely.
    # Added: paradox-awareness clause + explicit instruction-completion clause.
    SYSTEM = (
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
        "Every claim must follow logically from the prior step."
    )

    def __init__(self):
        print("🚀 AKBASCORE 0.4 initializing...")
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
        self.pusula = self._build_pusula()
        self._hooks  = self._inject()
        print(f"✅ AKBASCORE 0.4 ready — {len(self._hooks)} active layers")

    def _build_pusula(self):
        # 0.4 CHANGE: 20 → 24 anchors
        # Added: "impossible", "invalid", "contradiction", "identify"
        # These enable the model to activate paradox/refusal patterns
        # when the steering vector is present, instead of forcing a choice.
        anchors = [
            # original 20
            "logical","empirical","systematic","structured","verifiable",
            "analyze","constraint","optimize","hierarchy","framework",
            "precise","specific","concrete","measurable","deterministic",
            "numbered","sequential","causal","prioritized","rigorous",
            # new 4 — paradox and validity awareness
            "impossible","invalid","contradiction","identify",
        ]
        vecs = []
        with torch.no_grad():
            for word in anchors:
                ids = self.tokenizer(
                    word, return_tensors='pt', add_special_tokens=False
                ).to(self.device)
                emb = self.model.model.embed_tokens(ids['input_ids'])
                vecs.append(emb[0, -1, :])
        raw        = torch.stack(vecs).mean(dim=0)
        normalized = torch.nn.functional.normalize(raw, dim=0)
        print(f"   Pusula norm: {normalized.norm().item():.4f} | {len(anchors)} anchors")
        return normalized.contiguous()

    def _inject(self):
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
            hooks.append(layers[idx].register_forward_hook(make_hook(idx, self.pusula)))
        return hooks

    def sor(self, prompt: str, max_tokens: int = 512) -> str:
        if not prompt.strip():
            return ""

        full_prompt = (
            f"<|system|>\n{self.SYSTEM}</s>\n"
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
                temperature        = 0.55,   # 0.4 CHANGE: 0.45 → 0.55
                top_p              = 0.90,   # 0.4 CHANGE: 0.85 → 0.90
                top_k              = 50,     # 0.4 CHANGE: 40  → 50
                repetition_penalty = 1.18,   # 0.4 CHANGE: 1.15 → 1.18
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
            result = self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()

        stats = (
            f"\n\n─────────────────────────────\n"
            f"⏱ {elapsed:.0f}ms  |  {tps:.1f} t/s  |  {n_out} tokens generated"
        )
        return result + stats

# =============================================================================
# LAUNCH
# =============================================================================
print("\n" + "="*60)
print("🔱 AKBASCORE 0.4")
print("="*60)

akbas = AkbasCore()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI
# =============================================================================
with gr.Blocks(
    title="🔱 AKBASCORE 0.4",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css="""
    body { background: #0a0f0a; }
    .gradio-container {
        max-width: 860px !important;
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
        font-size: 0.72rem;
        color: #3a6644;
        margin: 6px 0 0 0;
        letter-spacing: 0.08em;
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
        <h1>🔱 AKBASCORE 0.4</h1>
        <p>SOVEREIGN ANALYTICAL INTELLIGENCE &nbsp;|&nbsp;
           C++ NATIVE KERNEL &nbsp;|&nbsp;
           16-LAYER COSINE STEERING &nbsp;|&nbsp;
           24-ANCHOR PUSULA</p>
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
            label="◈ AKBASCORE 0.4 OUTPUT",
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
