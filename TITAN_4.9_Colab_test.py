# =============================================================================
# 🔱 TITAN 5.4 | LATENT SPACE WEIGHT GRAFTING ENGINE (FULL-WIDTH EXPANDED)
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
    # 📐 1. Pusula Vektörünün Çıkarılması
    tokens = tokenizer(COMPASS_ANCHORS, return_tensors='pt', padding=True).to(DEVICE)
    vectors = model.model.embed_tokens(tokens['input_ids'])
    pusula_vektoru = F.normalize(vectors.mean(dim=1).mean(dim=0), dim=0).to(model.dtype)

    # 🔬 2. LATENT UZAY GRAFTING (AŞILAMA) OPERASYONU
    raw_embeds = model.model.embed_tokens.weight.data
    benzerlikler = (raw_embeds * pusula_vektoru).sum(dim=-1, keepdim=True)
    
    # Akbaş Kararlılık Çarpanı ve Amortisör Modülasyonu (Ana mantık korundu)
    V0_graft = 0.45 * 0.80 * 0.3
    katki_matrisi = V0_graft * benzerlikler * pusula_vektoru.view(1, -1)
    
    # Katastrofik Çöküşü Önleyen Çift Yönlü Mutlak Kelepçe (Clamp)
    katki_matrisi = torch.clamp(katki_matrisi, max=0.15, min=-0.15)
    
    # Latent uzayı tek seferde bük ve kalıcı olarak güncelle
    model.model.embed_tokens.weight.data = raw_embeds + katki_matrisi

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
            repetition_penalty=1.5, # Kusursuz Tekrar Cezası
            pad_token_id=tokenizer.eos_token_id,
        )
    
    cevap = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return cevap, ""

# 🛠️ Ekran sınırlarını tam anlamıyla kaldıran agresif genişlik CSS'i
expand_css = """
.gradio-container, .gradio-container > .wrap, .contain, .form, .gap { 
    max-width: 100% !important; 
    width: 100% !important; 
    padding-left: 0px !important; 
    padding-right: 0px !important; 
    margin-left: 0px !important; 
    margin-right: 0px !important; 
}
textarea, input[type="text"] { 
    width: 100% !important; 
    font-size: 16px !important; 
}
"""

# Tasarımsız, doğrudan en geniş okumaya odaklı saf yapı
with gr.Blocks(css=expand_css, title="TITAN 5.4") as titan_panel:
    txt_question = gr.Textbox(
        label="Sorunuzu Yazın", 
        placeholder="Buraya girin...", 
        lines=3
    )
    btn_submit = gr.Button("Soruyu Gönder")
    
    txt_output = gr.Textbox(
        label="Çıktı Ekranı", 
        lines=18, 
        show_copy_button=True
    )
            
    btn_submit.click(fn=titan_inference, inputs=[txt_question], outputs=[txt_output, txt_question])

titan_panel.launch(inline=True)
