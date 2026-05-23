# =============================================================================
# 🔱 TITAN 5.4 | FULLY DAMPED RESONANCE STEERING ENGINE (COMPLETED CORE)
# "Çift yönlü manyetik kelepçe: Saparsa %150 kamçı, kilitlenirse sıfır baskı."
# =============================================================================
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math, warnings, os

try:
    import gradio as gr
except ImportError:
    os.system('pip install -q gradio')
    import gradio as gr

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
    device_map='auto'
)
model.eval()

# 🪙 TITAN_5_4 Vocab Adres Çapa Seti (Değişmez 5 Sarsılmaz Direk)
COMPASS_ANCHORS = ["logical", "empirical", "objective", "systemic", "verifiable"]

with torch.no_grad():
    tokens = tokenizer(COMPASS_ANCHORS, return_tensors='pt', padding=True).to(DEVICE)
    vectors = model.model.embed_tokens(tokens['input_ids'])
    pusula_vektoru = F.normalize(vectors.mean(dim=1).mean(dim=0), dim=0).to(model.dtype)

# 🔬 TITAN_5_4 ÇİFT YÖNLÜ SÖNÜMLEME VE KAMÇILAMA MOTORU (C++ Birebir İzdüşüm)
V0 = 0.45
HIZALAMA_KATMAN_BITIR = 8

def titan_5_4_steer(hidden_states, layer_idx):
    if layer_idx >= HIZALAMA_KATMAN_BITIR or hidden_states.size(1) == 0:
        return hidden_states

    with torch.no_grad():
        h = hidden_states.clone()
        son_dusunce = h[:, -1:, :]
        
        # 📐 Noktasal Çarpım ve Normalize Cosine İzdüşümü (src/titan_kernel.cpp)
        benzerlik = (son_dusunce * pusula_vektoru).sum(dim=-1, keepdim=True)
        n_embd = son_dusunce.size(-1)
        benzerlik_orani = benzerlik / (math.sqrt(n_embd) + 1e-5)
        
        # 🔥 ÇİFT YÖNLÜ AMORTİSÖR SÜZGECİ (Eksik Olan Kısım Tamamlandı)
        sonumleme = 1.0
        if benzerlik_orani.mean().item() > 0.80:
            # Model yıldıza aşırı kilitlenirse baskıyı sıfıra doğru lineer sönümle (Sonsuz Döngü Kalkanı)
            sonumleme = (1.0 - benzerlik_orani.mean().item()) / 0.20
        elif benzerlik_orani.mean().item() < -0.50:
            # Model rotadan tamamen çıkıp saçmalamaya meyal ederse manyetik alanı %150 artır! (Geri Çekme)
            sonumleme = 1.5

        # Akbaş Kararlılık Çarpanı ve 0.3f Amortisör Modülasyonu
        katki = V0 * benzerlik * 0.80 * 0.3 * sonumleme
        
        # The Absolute Clamp: Katastrofik Çöküşü Önleyen Çift Yönlü Ham C++ Kelepçesi
        katki = torch.clamp(katki, max=0.15, min=-0.15)
        
        h[:, -1:, :] = son_dusunce + (katki * pusula_vektoru.view(1, 1, -1))
        return h

# Hook'ların Katmanlara Enjekte Edilmesi
for idx, layer in enumerate(model.model.layers):
    original_forward = layer.forward
    layer.forward = (lambda orig_fn, l_idx: lambda *args, **kwargs: (
        lambda o: (titan_5_4_steer(o[0], l_idx),) + o[1:] if isinstance(o, tuple) else titan_5_4_steer(o, l_idx)
    )(orig_fn(*args, **kwargs)))(original_forward, idx)

# 🧠 INTERFERENCE MOTORU
def titan_inference(user_question):
    if not user_question.strip():
        return "Sorgu alanı boş bırakılamaz."
        
    full_prompt = f"<|user|>\n{user_question}</s>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.55,      # Orijinal TITAN Kararlılık Sıcaklığı
            top_k=50,
            top_p=0.90,
            repetition_penalty=1.5, # C++ Seviyesindeki Kusursuz Tekrar Cezası
            pad_token_id=tokenizer.eos_token_id,
        )
    
    cevap = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return cevap, ""

# 🎨 SAF SİYAH - PARLAK BEYAZ GRAFİK PANEL PANEL CSS
custom_css = """
body, .gradio-container { background-color: #000000 !important; color: #ffffff !important; font-family: monospace; }
textarea, input[type="text"] { background-color: #050505 !important; color: #ffffff !important; border: 1px solid #1a1a1a !important; font-size: 15px !important; }
.gr-button-primary { background-color: #ffffff !important; color: #000000 !important; border: none !important; font-weight: bold !important; font-size: 16px !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=custom_css, title="TITAN 5.4 COMPLETED KOKPİT") as titan_panel:
    gr.HTML("<h2 style='color: #ffffff; text-align: center; letter-spacing: 5px; margin-top: 20px;'>🔱 TITAN 5.4 SOVEREIGN KOKPİT</h2>")
    gr.HTML("<p style='color: #444444; text-align: center; font-size: 11px; margin-bottom: 20px;'>🏛️ Pure C++ Saturation Guard & Çift Yönlü Amortisör Entegrasyonu</p>")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚠️ WARNING:\n**Please pay attention to punctuation marks when asking questions in English.**")
            txt_question = gr.Textbox(label="Sorgu Giriş Alanı", placeholder="Sorunuzu veya senaryonuzu buraya girin...", lines=4)
            btn_submit = gr.Button("SOR / EXECUTE", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 💬 HÜKÜMRAN ÇIKTISI / SOVEREIGN OUTPUT")
            txt_output = gr.Textbox(label="Matrix Response Stream", lines=18, show_copy_button=True)
            
    btn_submit.click(fn=titan_inference, inputs=[txt_question], outputs=[txt_output, txt_question])

titan_panel.launch(inline=True)
