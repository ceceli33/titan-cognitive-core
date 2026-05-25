# =============================================================================
# 🧬 TITAN 5.5 | MONOLITHIC MATRIC EVOLUTION ENGINE (ULTRA-FAST PHASE 5)
# =============================================================================
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings, os

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

# 🪙 TITAN_5_5 Vocab Adres Çapa Seti (Değişmez Orijinal Omurga)
COMPASS_ANCHORS = ["logical", "empirical", "objective", "systemic", "verifiable"]

with torch.no_grad():
    # 📐 1. Sabit Pusula Vektörünün Çıkarılması
    tokens = tokenizer(COMPASS_ANCHORS, return_tensors='pt', padding=True).to(DEVICE)
    vectors = model.model.embed_tokens(tokens['input_ids'])
    pusula_vektoru = F.normalize(vectors.mean(dim=1).mean(dim=0), dim=0).to(model.dtype)

    # 🔱 PHASE 4: Orijinal Titan Omurgasını Yedekle ve İlk Hizalamayı Yap
    BASE_EMBEDS = model.model.embed_tokens.weight.data.clone()
    benzerlikler = (BASE_EMBEDS * pusula_vektoru).sum(dim=-1, keepdim=True)
    V0_graft = 0.45 * 0.80 * 0.3
    TITAN_CORE = BASE_EMBEDS + torch.clamp(V0_graft * benzerlikler * pusula_vektoru.view(1, -1), max=0.15, min=-0.15)

# 🧠 INTERFERENCE MOTORU (YAZILIM DÖNGÜSÜ TAMAMEN KALDIRILDI)
def titan_inference(user_question):
    if not user_question.strip():
        return "Sorgu alanı boş bırakılamaz."
        
    full_prompt = f"<|user|>\n{user_question}</s>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors='pt').to(DEVICE)
    
    # 📉 Faz 5: Tek Adımlı Dinamik Evrim ve Sıkıştırma Denklem Parametreleri
    alpha = 0.12   # Soru Enerjisiyle Hücre Genişleme Katsayısı
    beta = 0.05    # Tanh Tabanlı Doğal Buharlaşma/Budama Katsayısı
    gamma = 0.1    # Tanh Kararlılık Filtresi Hassasiyeti

    with torch.no_grad():
        # 1. Girdinin (Sorunun) Özgün Enerji Matrisini Çıkar
        input_vectors = model.model.embed_tokens(inputs['input_ids']).squeeze(0) # [Prompt_Uzunluğu, 2048]
        soru_enerjisi = F.normalize(input_vectors.mean(dim=0), dim=0).to(model.dtype) # [2048]

        # 2. DOĞAL HÜCRE ÜRETİMİ (Genişleme): Sorunun enerjisi ile pusulayı latent uzayda evrimleştir
        # Model, sorulan sorunun rasyonel ağırlığına göre yönünü anlık çizer
        genisleme = alpha * (pusula_vektoru * soru_enerjisi).sum() * pusula_vektoru.view(1, -1)

        # 3. DOĞAL BUHARLAŞMA (Sinaptik Budama): Tanh filtresiyle gereksiz gürültüyü tek hamlede erit
        # Sabit omurga değerlerini korurken zayıf frekansları sıfıra doğru büker
        budama = beta * torch.tanh(gamma * TITAN_CORE)

        # 4. TEK SEFERLİK ANLIK EVRİM: Ağırlık matrisini güncelle ve emniyet kilidini bas
        AGI_WEIGHTS = TITAN_CORE + genisleme - budama
        model.model.embed_tokens.weight.data = torch.max(torch.min(AGI_WEIGHTS, TITAN_CORE + 0.15), TITAN_CORE - 0.15)

        # 5. SAF HIZ: Üretimi doğrudan HuggingFace'in kendi optimize CUDA motoruna bırak
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
    
    # Çıkarım bittiğinde, bir sonraki soru temiz başlasın diye beyni orijinal TITAN Çekirdeğine sıfırla
    model.model.embed_tokens.weight.data = TITAN_CORE
    
    cevap = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return cevap

# 🛠️ Ekran sınırlarını kaldıran tam genişlik CSS'i
expand_css = """
.gradio-container, .gradio-container > .wrap, .contain, .form, .gap { 
    max-width: 100% !important; width: 100% !important; 
    padding-left: 0px !important; padding-right: 0px !important; 
}
textarea, input[type="text"] { font-size: 16px !important; }
"""

with gr.Blocks(css=expand_css, title="TITAN 5.5") as titan_panel:
    gr.Markdown("### 🔱 TITAN 5.5 | Monolitik Faz 5 Evrim Motoru")
    txt_question = gr.Textbox(label="Sorunuzu Yazın", placeholder="Buraya girin...", lines=3)
    btn_submit = gr.Button("Soruyu Gönder (Yıldırım Hızı)")
    txt_output = gr.Textbox(label="Çıktı Ekranı", lines=18, show_copy_button=True)
            
    btn_submit.click(fn=titan_inference, inputs=[txt_question], outputs=[txt_output])

# Kanca sistemi yok, doğrudan Titan 5.4 hızıyla çalışır
titan_panel.launch(share=True, inline=True)
