# =============================================================================
# 🔱 TITAN 5.4 | ZERO-LATENCY WEIGHT GRAFTING ENGINE
# Colab Interactive Version
# =============================================================================
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings, os, time
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    import gradio as gr
except ImportError:
    os.system('pip install -q gradio')
    import gradio as gr

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
V0 = 0.45
COMPASS_ANCHORS = ["logical", "empirical", "objective", "systemic", "verifiable"]

print(f"🔧 Device: {DEVICE}")
print("📦 Model yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
    device_map='auto'
)
model.eval()
print("✅ Model yüklendi")

# =============================================================================
# ⚙️ TITAN GRAFTING
# =============================================================================
print("\n⚙️  TITAN grafting başlıyor...")

with torch.no_grad():

    tokens = tokenizer(COMPASS_ANCHORS, return_tensors='pt', padding=True).to(DEVICE)
    vectors = model.model.embed_tokens(tokens['input_ids'])
    pusula = F.normalize(vectors.mean(dim=1).mean(dim=0), dim=0).to(model.dtype)
    print(f"🧭 Pusula vektörü | shape: {pusula.shape}")

    raw = model.model.embed_tokens.weight.data
    benzerlik = (raw * pusula).sum(dim=-1, keepdim=True)
    katki = torch.clamp(V0 * 0.80 * 0.3 * benzerlik * pusula.unsqueeze(0), -0.15, 0.15)
    model.model.embed_tokens.weight.data = raw + katki
    print(f"📐 Embedding grafting tamamlandı")

    grafted = 0
    for idx, layer in enumerate(model.model.layers):
        kuvvet = 0.80 if idx < 8 else (0.40 if idx < 16 else 0.0)
        if kuvvet == 0.0:
            continue
        for proj in [layer.self_attn.q_proj, layer.self_attn.v_proj]:
            w = proj.weight.data
            alignment = (w @ pusula)
            delta = torch.clamp(
                V0 * kuvvet * 0.3 * alignment.unsqueeze(1) * pusula.unsqueeze(0),
                -0.15, 0.15
            )
            proj.weight.data = w + delta
        grafted += 1

    print(f"🔩 Layer grafting tamamlandı | {grafted} layer")

print("✅ TITAN 5.4 aktif\n")

# =============================================================================
# 🧠 INFERENCE
# =============================================================================
def titan_inference(user_question):
    if not user_question.strip():
        return "Sorgu boş bırakılamaz.", ""

    full_prompt = f"<|user|>\n{user_question}</s>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors='pt').to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.55,
            top_k=50,
            top_p=0.90,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    input_len = inputs['input_ids'].shape[1]
    new_tokens = output_ids.shape[1] - input_len
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0

    cevap = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    ).strip()

    stats = f"⏱️ {elapsed:.1f}s | {new_tokens} token | {tok_per_sec:.1f} tok/sn"
    return cevap, stats

# =============================================================================
# 🖥️ GRADIO PANEL
# =============================================================================
css = """
.gradio-container { max-width: 100% !important; width: 100% !important; }
textarea, input[type="text"] { font-size: 16px !important; }
#stats_box { font-size: 13px; color: #888; margin-top: 4px; }
"""

with gr.Blocks(css=css, title="TITAN 5.4") as titan_panel:

    gr.Markdown("## 🔱 TITAN 5.4 | Zero-Latency Steering Engine")

    with gr.Row():
        with gr.Column(scale=1):
            txt_question = gr.Textbox(
                label="Sorunuz",
                placeholder="Sorunuzu buraya yazın...",
                lines=5
            )
            btn_submit = gr.Button("🚀 Gönder", variant="primary")
            btn_clear = gr.Button("🗑️ Temizle", variant="secondary")

        with gr.Column(scale=2):
            txt_output = gr.Textbox(
                label="TITAN Çıktısı",
                lines=18,
                show_copy_button=True
            )
            txt_stats = gr.Textbox(
                label="İstatistik",
                lines=1,
                interactive=False,
                elem_id="stats_box"
            )

    btn_submit.click(
        fn=titan_inference,
        inputs=[txt_question],
        outputs=[txt_output, txt_stats]
    )

    btn_clear.click(
        fn=lambda: ("", "", ""),
        outputs=[txt_question, txt_output, txt_stats]
    )

titan_panel.launch(debug=False, share=True)
