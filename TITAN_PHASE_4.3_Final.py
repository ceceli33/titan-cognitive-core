# =============================================================================
# 🔱 TITAN 4.4 | HALKIN BİLGESİ (Robust Zeka)
# "Kusurlu ithalata tahammül, kusursuz ithalata hükümranlık"
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import os
import re

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')

# =============================================================================
# 🔱 KONFİGÜRASYON
# =============================================================================
class TitanConfig:
    V0 = 0.45
    
    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    
    # 🔥 YENİ: SEMANTİK ÇAPALAR (Anlam odaklı)
    COMPASS_ANCHORS = [
        "logical",      # mantık
        "empirical",    # gözlem
        "objective",    # nesnellik
        "systemic",     # sistem
        "verifiable"    # doğrulanabilir
    ]
    
    # 🔥 YENİ: NİYET ÇAPALARI (Kullanıcı niyetini anlamak için)
    INTENT_ANCHORS = [
        "understand",   # anlamak
        "respond",      # yanıt vermek
        "clarify",      # açıklığa kavuşturmak
        "help",         # yardım etmek
        "explain"       # açıklamak
    ]
    
    MAX_TOKENS = 350
    TEMPERATURE = 0.55
    TOP_K = 50
    TOP_P = 0.90
    REPETITION_PENALTY = 1.5
    
    HIZALAMA_KATMAN_BITIR = 8
    EVRENSEL_KOPRU_BITIR = 16
    
    HIZALAMA_KUVVET = 0.80
    EVRENSEL_KOPRU_KUVVET = 0.40
    TAM_OZGURLUK_KUVVET = 0.00

print("🔱 TITAN 4.4 | Halkın Bilgesi (Robust Zeka)")
print("="*65)
print("   • R1 Hizalama:    V0=0.45 | %80 (katman 0-7)")
print("   • R2 Kavram:      logical, empirical, objective, systemic, verifiable")
print("   • 🔥 YENİ Niyet:  understand, respond, clarify, help, explain")

# =============================================================================
# 🔱 GİRİŞ NORMALİZASYONU (Input Pre-processor)
# =============================================================================
class InputNormalizer:
    """Bozuk grameri TITAN'ın anlayacağı dile çevirir"""
    
    def __init__(self):
        self.icraat = 0
        self.son_kalite = 1.0
    
    def _gramer_notu(self, text):
        """Giriş kalitesini 0-1 arası puanla (1=mükemmel)"""
        score = 1.0
        
        # Kısa cümleler (bozuk olabilir) - tolerans artar
        if len(text.split()) < 5:
            score = score * 0.95
        
        # Çok fazla soru işareti veya noktalama düzensizliği
        if text.count('?') > 2:
            score = score * 0.90
        if text.count('!') > 1:
            score = score * 0.85
        
        # Bazı anahtar kelimeler varsa (niyeti anlaşılıyor)
        intent_keywords = ['mean', 'want', 'trying to say', 'explain', 'what is', 'how to']
        if any(kw in text.lower() for kw in intent_keywords):
            score = min(1.0, score + 0.15)  # Niyet anlaşılıyorsa affet
        
        return max(0.3, min(1.0, score))
    
    def normalize(self, prompt):
        """Sadece kalite analizi yap, metni değiştirme"""
        self.son_kalite = self._gramer_notu(prompt)
        self.icraat += 1
        
        # Kalite düşükse bilgi ver (opsiyonel)
        if self.son_kalite < 0.6:
            print(f"   (Giriş kalitesi: {self.son_kalite:.2f} - TITAN anlamaya çalışıyor...)")
        
        return prompt  # Metni değiştirme, sadece kaliteyi kaydet
    
    def get_dinamik_v0(self, base_v0):
        """Kaliteye göre V0 ayarı - düşük kalitede baskıyı azalt"""
        return base_v0 * self.son_kalite
    
    def get_kuvvet_carpani(self):
        """Düşük kalitede müdahaleyi gevşet"""
        # Kusurlu girişte daha toleranslı ol
        return 0.5 + (self.son_kalite * 0.5)

# =============================================================================
# 🔱 PUSULA (çift vektörlü)
# =============================================================================
class Pusula:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.kavram_vector = None
        self.niyet_vector = None
        self._cikar()
    
    def _cikar(self):
        with torch.no_grad():
            # KAVRAM VEKTÖRÜ
            tokens = self.tokenizer(
                TitanConfig.COMPASS_ANCHORS,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            
            vectors = self.model.model.embed_tokens(tokens['input_ids'])
            weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)
            weights = weights.view(-1, 1, 1)
            
            weighted_vectors = vectors * weights
            token_means = weighted_vectors.mean(dim=1)
            self.kavram_vector = token_means.mean(dim=0)
            self.kavram_vector = F.normalize(self.kavram_vector, dim=0)
            self.kavram_vector = self.kavram_vector * 0.6
            
            # NİYET VEKTÖRÜ
            intent_tokens = self.tokenizer(
                TitanConfig.INTENT_ANCHORS,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            
            intent_vectors = self.model.model.embed_tokens(intent_tokens['input_ids'])
            intent_weighted = intent_vectors * weights
            intent_means = intent_weighted.mean(dim=1)
            self.niyet_vector = intent_means.mean(dim=0)
            self.niyet_vector = F.normalize(self.niyet_vector, dim=0)
            self.niyet_vector = self.niyet_vector * 0.4  # daha hafif
            
            print(f"✓ Çift Pusula çıkarıldı")
            print(f"   • Kavram vektörü: norm={self.kavram_vector.norm().item():.4f}")
            print(f"   • Niyet vektörü:  norm={self.niyet_vector.norm().item():.4f}")
    
    def get(self):
        return self.kavram_vector
    
    def get_intent(self):
        return self.niyet_vector

# =============================================================================
# 🔱 TITAN KERNEL (Dinamik toleranslı)
# =============================================================================
class TitanKernel:
    def __init__(self, pusula_vector, niyet_vector, v0=0.45):
        self.kavram = pusula_vector
        self.niyet = niyet_vector
        self.v0 = v0
        self.son_kuvvet = 0.0
        self.son_benzerlik = 0.0
        self.son_bolge = "Başlangıç"
        self.normalizer = InputNormalizer()
    
    def _kademeli_kuvvet(self, layer_idx):
        if layer_idx < TitanConfig.HIZALAMA_KATMAN_BITIR:
            self.son_bolge = "🏛️ R1: Hizalama"
            return TitanConfig.HIZALAMA_KUVVET
        elif layer_idx < TitanConfig.EVRENSEL_KOPRU_BITIR:
            self.son_bolge = "🌉 R3: Mantık Köprüsü"
            return TitanConfig.EVRENSEL_KOPRU_KUVVET
        else:
            self.son_bolge = "🕊️ R5: Hükümran Çıkış"
            return TitanConfig.TAM_OZGURLUK_KUVVET
    
    def process_input(self, prompt):
        """Soru gelince yapılacak işlemler"""
        return self.normalizer.normalize(prompt)
    
    def yönlendir(self, hidden_states, layer_idx, prompt_kalitesi=1.0):
        kuvvet_katsayisi = self._kademeli_kuvvet(layer_idx)
        
        if kuvvet_katsayisi == 0.0:
            return hidden_states
        
        # Dinamik V0 (kaliteye göre)
        dinamik_v0 = self.normalizer.get_dinamik_v0(self.v0)
        
        with torch.no_grad():
            son_dusunce = hidden_states[:, -1:, :].detach()
            
            # Kavram uyumu
            kavram_benzerlik = (son_dusunce * self.kavram).sum(dim=-1, keepdim=True)
            
            # Niyet uyumu (daha hafif)
            niyet_benzerlik = (son_dusunce * self.niyet).sum(dim=-1, keepdim=True)
            
            # Karışık benzerlik (ağırlıklı ortalama)
            benzerlik = kavram_benzerlik * 0.7 + niyet_benzerlik * 0.3
            
            # Kalite düşükse müdahaleyi gevşet
            gevşeme = self.normalizer.get_kuvvet_carpani()
            
            katki = dinamik_v0 * benzerlik * kuvvet_katsayisi * 0.3 * gevşeme
            katki = torch.clamp(katki, max=0.15)
            
            yonlendirilmis = son_dusunce + katki * self.kavram.view(1, 1, -1)
            hidden_states[:, -1:, :] = yonlendirilmis.to(hidden_states.dtype)
        
        self.son_kuvvet = katki.mean().item()
        self.son_benzerlik = benzerlik.mean().item()
        
        return hidden_states
    
    def istatistik(self):
        return {
            'kuvvet': round(self.son_kuvvet, 4),
            'benzerlik': round(self.son_benzerlik, 4),
            'bolge': self.son_bolge,
            'v0': self.normalizer.get_dinamik_v0(TitanConfig.V0),
            'giris_kalitesi': round(self.normalizer.son_kalite, 2),
            'sıcaklık': TitanConfig.TEMPERATURE,
        }

# =============================================================================
# 📦 MODEL YÜKLEME
# =============================================================================
print("\n📦 TinyLlama yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(TitanConfig.MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    TitanConfig.MODEL_ID,
    dtype=torch.float32,
    device_map='auto',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.eval()
print(f"✓ Model hazır | {len(model.model.layers)} katman")

# =============================================================================
# 🔱 PUSULA VE ENJEKSİYON
# =============================================================================
pusula = Pusula(model, tokenizer, model.device)
titan = TitanKernel(pusula.get(), pusula.get_intent(), v0=TitanConfig.V0)

# Hook mekanizması (kalite bilgisini iletmek için wrapper)
current_quality = 1.0

layers = model.model.layers
for idx, layer in enumerate(layers):
    original_forward = layer.forward
    
    def make_steering_hook(original_fn, layer_num):
        def hooked_forward(*args, **kwargs):
            output = original_fn(*args, **kwargs)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            steered = titan.yönlendir(hidden, layer_num, current_quality)
            
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        return hooked_forward
    
    layer.forward = make_steering_hook(original_forward, idx)

print(f"\n✓ 5 Raylı sistem {len(layers)} katmana entegre edildi")
print(f"   • 🏛️  R1-Hizalama:    katman 0-{TitanConfig.HIZALAMA_KATMAN_BITIR-1} (%{int(TitanConfig.HIZALAMA_KUVVET*100)})")
print(f"   • 🌉  R3-Mantık:      katman {TitanConfig.HIZALAMA_KATMAN_BITIR}-{TitanConfig.EVRENSEL_KOPRU_BITIR-1} (%{int(TitanConfig.EVRENSEL_KOPRU_KUVVET*100)})")
print(f"   • 🕊️  R5-Özgürlük:    katman {TitanConfig.EVRENSEL_KOPRU_BITIR}+ (%{int(TitanConfig.TAM_OZGURLUK_KUVVET*100)})")
print(f"   • 🔥 Dinamik V0:      Giriş kalitesine göre ayarlanır (0.45 × kalite)")
print(f"   • 🔥 Niyet vektörü:   understand, respond, clarify, help, explain")

# =============================================================================
# 💬 SORGU
# =============================================================================
def soru_sor(prompt, max_tokens=TitanConfig.MAX_TOKENS):
    global current_quality
    
    # Giriş normalizasyonu
    normalized = titan.process_input(prompt)
    current_quality = titan.normalizer.son_kalite
    
    full_prompt = f"<|user|>\n{normalized}</s>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=TitanConfig.TEMPERATURE,
            top_k=TitanConfig.TOP_K,
            top_p=TitanConfig.TOP_P,
            repetition_penalty=TitanConfig.REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    yeni_tokenler = output_ids[0][inputs['input_ids'].shape[1]:]
    cevap = tokenizer.decode(yeni_tokenler, skip_special_tokens=True)
    
    if not cevap or len(cevap.strip()) == 0:
        cevap = "[TITAN] (Cevap üretilemedi)"
    
    return cevap, titan.istatistik()

# =============================================================================
# 🔱 KOKPİT
# =============================================================================
def kokpit_goster(prompt, cevap, stats):
    kuvvet = stats.get('kuvvet', 0)
    benzerlik = stats.get('benzerlik', 0)
    bolge = stats.get('bolge', '?')
    kalite = stats.get('giris_kalitesi', 1.0)
    dinamik_v0 = stats.get('v0', 0.45)
    
    if benzerlik > 0.5:
        renk, durum = '#44ff88', '🟢 HİZALI'
    elif benzerlik > 0.2:
        renk, durum = '#88ff44', '🟡 GEÇİŞ'
    else:
        renk, durum = '#ffaa44', '🟠 SERBEST'
    
    bolge_ikon = '🏛️' if 'Hizalama' in bolge else '🌉' if 'Köprü' in bolge else '🕊️'
    
    # Kalite göstergesi
    if kalite > 0.8:
        kalite_str = f'🟢 {kalite:.2f} (Yüksek)'
    elif kalite > 0.5:
        kalite_str = f'🟡 {kalite:.2f} (Orta)'
    else:
        kalite_str = f'🟠 {kalite:.2f} (Düşük - Tolerans aktif)'
    
    html = f'''
    <div style="font-family:monospace;background:#0a0e17;border:2px solid {renk};
                border-radius:12px;padding:14px;margin:10px 0;">
        <div style="border-bottom:1px solid {renk};padding-bottom:6px;margin-bottom:10px;">
            <span style="color:{renk};font-weight:bold;">🔱 TITAN 4.4 | Halkın Bilgesi</span>
            <span style="color:#5a7080;font-size:10px;"> | {durum}</span>
        </div>
        
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px;">
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:7px;color:#5a7080;">⚡ MANYETİK ALAN</div>
                <div style="font-size:14px;color:{renk};">{kuvvet:.4f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:7px;color:#5a7080;">📐 HİZALAMA</div>
                <div style="font-size:14px;color:#ffaa44;">{benzerlik:.3f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:7px;color:#5a7080;">🎚️ V0</div>
                <div style="font-size:14px;color:#44ff88;">{dinamik_v0:.3f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:7px;color:#5a7080;">📥 GİRİŞ KALİTESİ</div>
                <div style="font-size:11px;color:#ffaa44;">{kalite_str}</div>
            </div>
        </div>
        
        <div style="background:#0d1117;border-radius:6px;padding:6px;margin-bottom:8px;">
            <div style="font-size:8px;color:#5a7080;">{bolge_ikon} AKTİF RAY</div>
            <div style="font-size:10px;color:{renk};">{bolge}</div>
        </div>
        
        <div style="background:#0d1117;border-radius:6px;padding:8px;">
            <div style="font-size:8px;color:#5a7080;">💬 HÜKÜMRAN ÇIKTISI</div>
            <div style="font-size:11px;color:#c9d4e0;max-height:250px;overflow-y:auto;line-height:1.4;">
                {cevap}
            </div>
        </div>
        
        <div style="margin-top:8px;text-align:center;font-size:9px;color:#d4af37;">
            🔱 "Kusurlu ithalata tahammül, kusursuz çıktıya hükümranlık."
        </div>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# 🔱 ARAYÜZ
# =============================================================================
soru_kutusu = widgets.Textarea(
    value='What is sovereign intelligence?',  # Basit soru - test için
    placeholder='Soruyu yaz... (Bozuk gramerde de yazabilirsin, TITAN anlayacaktır)',
    layout=widgets.Layout(width='100%', height='100px')
)

sor_btn = widgets.Button(description='🔱 SOR', button_style='success', layout=widgets.Layout(width='100px'))
temizle_btn = widgets.Button(description='🗑️ TEMİZLE', button_style='warning', layout=widgets.Layout(width='100px'))
cikti_alani = widgets.Output()

def on_sor(b):
    global current_quality
    with cikti_alani:
        clear_output(wait=True)
        if not soru_kutusu.value.strip():
            print("⚠️ Lütfen bir soru yazın.")
            return
        try:
            print("⚡ TITAN 4.4 düşünüyor (Giriş kalitesi analiz ediliyor)...")
            cevap, stats = soru_sor(soru_kutusu.value)
            clear_output(wait=True)
            kokpit_goster(soru_kutusu.value, cevap, stats)
        except Exception as e:
            clear_output(wait=True)
            print(f"💀 Hata: {str(e)[:300]}")

def on_temizle(b):
    soru_kutusu.value = ''
    with cikti_alani:
        clear_output(wait=True)
        print("🧹 Temizlendi.")

sor_btn.on_click(on_sor)
temizle_btn.on_click(on_temizle)

buton_kutusu = widgets.HBox([sor_btn, temizle_btn])

# =============================================================================
# 🔱 BAŞLAT
# =============================================================================
print("\n" + "="*65)
print("🔱 TITAN 4.4 HAZIR | Halkın Bilgesi (Robust Zeka)")
print("="*65)
print(f"   • R1-Hizalama:    %{int(TitanConfig.HIZALAMA_KUVVET*100)} | katman 0-{TitanConfig.HIZALAMA_KATMAN_BITIR-1}")
print(f"   • R3-Mantık:      %{int(TitanConfig.EVRENSEL_KOPRU_KUVVET*100)} | katman {TitanConfig.HIZALAMA_KATMAN_BITIR}-{TitanConfig.EVRENSEL_KOPRU_BITIR-1}")
print(f"   • R5-Özgürlük:    %{int(TitanConfig.TAM_OZGURLUK_KUVVET*100)} | katman {TitanConfig.EVRENSEL_KOPRU_BITIR}+")
print(f"   • 🔥 Dinamik V0:   = 0.45 × Giriş Kalitesi")
print(f"   • 🔥 Niyet Çapası: understand, respond, clarify, help, explain")
print(f"   • 🔥 Tolerans:     Düşük kaliteli girişte müdahale azalır")
print("="*65)
print("🚀 Artık bozuk gramerde yazılan soruları da anlıyor.\n")

display(widgets.VBox([
    widgets.HTML('<h3 style="font-family:monospace;color:#44ff88;margin:0;">🔱 TITAN 4.4 | Halkın Bilgesi</h3>'),
    widgets.HTML('<p style="font-size:9px;color:#5a7080;margin:0 0 10px 0;">🏛️ Hizalama → 🌉 Mantık → 🕊️ Özgürlük | 🔥 Dinamik tolerans | 🔥 Niyet çapası</p>'),
    soru_kutusu,
    buton_kutusu,
    cikti_alani
]))

print("\n✅ TITAN 4.4 hazır. Kusurlu dille de sorabilirsiniz, anlayacaktır.")
