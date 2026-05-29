# =============================================================================
# 🔱 AKBASCORE 0.5.1 | TOKEN SLIDER + TEMİZ ARAYÜZ
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
# C++ KERNEL
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
    if      (layer_idx <  8) kuvvet = 0.95f;
    else if (layer_idx < 16) kuvvet = 0.50f;
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
            if      (cosine >  0.70f) sonumleme = (1.0f - cosine) / 0.30f;
            else if (cosine < -0.35f) sonumleme = 1.8f;

            float max_k = tok_norm * 0.045f;
            if (max_k > 0.22f) max_k = 0.22f;
            if (max_k < 0.04f) max_k = 0.04f;

            float katki = v0 * cosine * kuvvet * 0.38f * sonumleme;
            if (katki >  max_k) katki =  max_k;
            if (katki < -max_k) katki = -max_k;

            for (int j = 0; j < D; ++j) tok[j] += katki * pp[j];
        }
    }
    return h;
}
"""

_kernel = torch.utils.cpp_extension.load_inline(
    name='akbas_kernel_051',
    cpp_sources=_cpp_src,
    functions=['akbas_steer'],
    verbose=False
)
print("✅ C++ kernel derlendi")

# =============================================================================
# AKBASCORE
# =============================================================================
class AkbasCore:
    SYSTEM = (
        "You are TITAN 5.4 — a sovereign analytical intelligence. "
        "You reason in structured, precise steps. "
        "When given a problem, you: "
        "(1) identify the core structural constraint, "
        "(2) produce a numbered, hierarchical analysis, "
        "(3) derive empirical conclusions with explicit reasoning. "
        "You never produce vague or generic text. "
        "Every claim must follow logically from the prior step."
    )

    def __init__(self):
        print("🚀 AKBASCORE 0.5.1 başlatılıyor...")
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
        print(f"✅ AKBASCORE 0.5.1 hazır — {len(self._hooks)} aktif katman")

    def _build_pusula(self):
        anchors = [
            "logical","empirical","systematic","structured","verifiable",
            "analyze","constraint","optimize","hierarchy","framework",
            "precise","specific","concrete","measurable","deterministic",
            "numbered","sequential","causal","prioritized","rigorous",
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
        print(f"   Pusula norm: {normalized.norm().item():.4f} | {len(anchors)} anchor")
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

    # max_tokens parametresi eklendi
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
                temperature        = 0.45,
                top_p              = 0.85,
                top_k              = 40,
                repetition_penalty = 1.15,
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

        stats = f"\n\n─────────────────────────────\n⏱ {elapsed:.0f}ms  |  {tps:.1f} t/s  |  {n_out} token üretildi"
        return result + stats

# =============================================================================
# BAŞLAT (ısınma yok — direkt kullanıma hazır)
# =============================================================================
print("\n" + "="*60)
print("🔱 AKBASCORE 0.5.1")
print("="*60)

akbas = AkbasCore()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =============================================================================
# GRADIO — TOKEN SLIDER + TEMİZ ARAYÜZ
# =============================================================================
with gr.Blocks(
    title="🔱 AKBASCORE 0.5.1",
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

    /* Başlık */
    #titan-header {
        text-align: center;
        padding: 28px 0 8px 0;
        border-bottom: 1px solid #1a3a20;
        margin-bottom: 20px;
    }
    #titan-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        color: #00ff88;
        letter-spacing: 0.15em;
        margin: 0;
        text-shadow: 0 0 18px #00ff8855;
    }
    #titan-header p {
        font-size: 0.72rem;
        color: #3a6644;
        margin: 6px 0 0 0;
        letter-spacing: 0.08em;
    }

    /* Input alanı */
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

    /* Slider */
    .slider-row label {
        color: #4a9960 !important;
        font-size: 0.80rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    input[type=range] {
        accent-color: #00cc66;
    }

    /* Token badge */
    #token-badge {
        display: inline-block;
        background: #0f2a18;
        border: 1px solid #1e5030;
        color: #00ff88;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 3px 12px;
        border-radius: 20px;
        margin-left: 10px;
        vertical-align: middle;
    }

    /* Gönder butonu */
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

    /* Çıktı */
    #output-box textarea {
        background: #080e09 !important;
        color: #7fff9a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        border: 1px solid #1a3020 !important;
        line-height: 1.7 !important;
    }

    /* Label renkleri */
    label span {
        color: #4a9960 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.80rem !important;
        letter-spacing: 0.05em !important;
    }

    /* Spinner */
    .generating { border-color: #00cc66 !important; }
    """
) as demo:

    with gr.Column(elem_id="titan-header"):
        gr.HTML("""
        <h1>🔱 AKBASCORE 0.5.1</h1>
        <p>TITAN 5.4 SOVEREIGN INTELLIGENCE &nbsp;|&nbsp;
           C++ NATIVE KERNEL &nbsp;|&nbsp;
           16-LAYER COSINE STEERING &nbsp;|&nbsp;
           20-ANCHOR PUSULA</p>
        """)

    with gr.Row():
        prompt_box = gr.Textbox(
            label="► SORU / KOMUT",
            lines=6,
            placeholder="Sorunuzu buraya yazın...",
            show_copy_button=False,
        )

    with gr.Row(elem_classes="slider-row"):
        token_slider = gr.Slider(
            minimum=64,
            maximum=1024,
            value=512,
            step=64,
            label="MAX TOKEN  (yanıt uzunluğu)",
            info="64 → hızlı / 1024 → uzun analiz",
            interactive=True,
        )

    with gr.Row():
        send_btn = gr.Button(
            "▶  TITAN'A GÖNDer",
            variant="primary",
            elem_id="send-btn",
            scale=1,
        )

    with gr.Row():
        output_box = gr.Textbox(
            label="◈ TITAN 5.4 YANITI",
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

    # Enter ile de gönderilebilir
    prompt_box.submit(
        fn=akbas.sor,
        inputs=[prompt_box, token_slider],
        outputs=output_box,
    )

print("\n🚀 Gradio başlatılıyor...")
demo.launch(share=True, debug=False)
