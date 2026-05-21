# =============================================================================
# 🔱 TITAN 4.4 | Damped Resonance Kernel (Sönümlü Diferansiyel Rezonans)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import warnings
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import os

# 🔱 ÇEVRE TEMİZLİĞİ
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')

# =============================================================================
# 🔱 AKBASCORE V5 CONFIGURATION
# =============================================================================
class AkbasCore:
    V0 = 0.45
    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    
    # R2 Çapa Elemanları (Soyut Felsefi Kulvar)
    COMPASS_ANCHORS = [
        "philosophical",
        "conceptual",
        "epistemological",
        "systemic",
        "analytical"
    ]
    
    MAX_TOKENS = 350
    TEMPERATURE = 0.55  # Yaratıcı zeka korunuyor
    TOP_K = 50
    TOP_P = 0.90
    REPETITION_PENALTY = 1.5
    
    # Rezonans Geometrisi Parametreleri
    LAMBDA_DECAY = 0.28          # 🌊 Sönümlenme hızı (Yüksek değer hızlı sönümler)
    ANCHOR_LAYER_COUNT = 8       # 🏛️ Tam hizalamada kalacak ilk katman sayısı
    MIN_ATTENUATION_THRESHOLD = 0.03 # 🕊️ Bu gücün altına inildiğinde müdahale tamamen kesilir

# =============================================================================
# 🔱 PUSULA (SEMANTİK ÇAPA ALANI)
# =============================================================================
class Pusula:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vector = None
        self._cikar()
    
    def _cikar(self):
        with torch.no_grad():
            tokens = self.tokenizer(
                AkbasCore.COMPASS_ANCHORS,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            
            vectors = self.model.model.embed_tokens(tokens['input_ids'])
            weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=vectors.dtype).to(self.device)
            weights = weights.view(-1, 1, 1)
            
            weighted_vectors = vectors * weights
            token_means = weighted_vectors.mean(dim=1)
            self.vector = token_means.mean(dim=0)
            self.vector = F.normalize(self.vector, dim=0)
            self.vector = self.vector * 0.6
            
            print(f"✓ Kavram Pusulası Doğrulandı | Norm: {self.vector.norm().item():.4f}")
    
    def get(self):
        return self.vector

# =============================================================================
# 🔱 TITAN DAMPED KERNEL
# =============================================================================
class TitanDampedKernel:
    def __init__(self, pusula_vector, v0=0.45):
        self.pusula = pusula_vector
        self.v0 = v0
        self.son_kuvvet = 0.0
        self.son_benzerlik = 0.0
        self.son_bolge = "Başlangıç"
        self.aktif_kuvvet_haritasi = {}
    
    def hesapla_dinamik_kuvvet(self, layer_idx):
        """🌊 Matematiksel Sönümlenme Eğrisi Uygulama Alanı"""
        if layer_idx < AkbasCore.ANCHOR_LAYER_COUNT:
            self.son_bolge = "🏛️ R1: Hizalama Çapa Alanı"
            return 0.80  # İlk 8 katmanda tam hizalama baskısı
            
        # 8. Katmandan sonra üssel sönümlenme fonksiyonu tetiklenir
        mesafe = layer_idx - AkbasCore.ANCHOR_LAYER_COUNT
        kuvvet = 0.40 * math.exp(-AkbasCore.LAMBDA_DECAY * mesafe)
        
        if kuvvet < AkbasCore.MIN_ATTENUATION_THRESHOLD:
            self.son_bolge = "🕊️ R5: Dil Mekaniği (Tam Özgürlük)"
            return 0.0  # Dil mekaniğinin başladığı katmanlarda müdahaleyi sıfırla
            
        self.son_bolge = f" Bridge (Sönümlenme Adımı: {kuvvet:.3f})"
        return kuvvet

    def yönlendir(self, hidden_states, layer_idx):
        kuvvet_katsayisi = self.hesapla_dinamik_kuvvet(layer_idx)
        self.aktif_kuvvet_haritasi[layer_idx] = kuvvet_katsayisi
        
        if kuvvet_katsayisi == 0.0:
            return hidden_states  # Üst katmanlara dokunma, pürüzsüz cümle kursun
            
        steered_states = hidden_states.clone()
        
        with torch.no_grad():
            son_dusunce = steered_states[:, -1:, :]
            self.pusula = self.pusula.to(device=son_dusunce.device, dtype=son_dusunce.dtype)
            benzerlik = (son_dusunce * self.pusula).sum(dim=-1, keepdim=True)
            
            # Dinamik sönümlenmeli diferansiyel itki
            katki = self.v0 * benzerlik * kuvvet_katsayisi * 0.3
            katki = torch.clamp(katki, max=0.15)
            
            yonlendirilmis = son_dusunce + katki * self.pusula.view(1, 1, -1)
            steered_states[:, -1:, :] = yonlendirilmis
            
            self.son_kuvvet = katki.mean().item()
            self.son_benzerlik = benzerlik.mean().item()
            
        return steered_states
    
    def istatistik(self):
        return {
            'kuvvet': round(self.son_kuvvet, 4),
            'benzerlik': round(self.son_benzerlik, 4),
            'bolge': self.son_bolge,
            'v0': self.v0,
            'harita': self.aktif_kuvvet_haritasi
        }

# =============================================================================
# 📦 MODEL ENJEKSİYONU
# =============================================================================
print("\n📦 TinyLlama Yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(AkbasCore.MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    AkbasCore.MODEL_ID,
    dtype=torch.float32,
    device_map='auto',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()

pusula = Pusula(model, tokenizer, model.device)
titan = TitanDampedKernel(pusula.get(), v0=AkbasCore.V0)

# Kanca (Hook) Enjeksiyonu
for idx, layer in enumerate(model.model.layers):
    original_forward = layer.forward
    
    def make_steering_hook(original_fn, layer_num):
        def hooked_forward(*args, **kwargs):
            output = original_fn(*args, **kwargs)
            hidden = output[0] if isinstance(output, tuple) else output
            
            steered = titan.yönlendir(hidden, layer_num)
            
            return (steered,) + output[1:] if isinstance(output, tuple) else steered
        return hooked_forward
        
    layer.forward = make_steering_hook(original_forward, idx)

print("✓ TITAN 4.4 Sönümlü Kanca Enjeksiyonu Tamamlandı.")

# =============================================================================
# 💬 TEST ARAÇLARI
# =============================================================================
def soru_sor(prompt, max_tokens=AkbasCore.MAX_TOKENS):
    full_prompt = (
        f"<|user|>\n"
        f"Analyze the following question from a deep philosophical, conceptual, and analytical "
        f"perspective without drifting into trivial or purely physical scenarios:\n\n"
        f"{prompt}</s>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=AkbasCore.TEMPERATURE,
            top_k=AkbasCore.TOP_K,
            top_p=AkbasCore.TOP_P,
            repetition_penalty=AkbasCore.REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    yeni_tokenler = output_ids[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(yeni_tokenler, skip_special_tokens=True), titan.istatistik()

def kokpit_goster(cevap, stats):
    kuvvet = stats.get('kuvvet', 0)
    benzerlik = stats.get('benzerlik', 0)
    bolge = stats.get('bolge', '?')
    harita = stats.get('harita', {})
    
    # Katman sönümlenme görselleştirmesi barı
    bar_html = ""
    for l, k in sorted(harita.items()):
        renk_bar = "#44ff88" if k > 0.7 else "#88ff44" if k > 0.2 else "#5a7080" if k > 0 else "#223344"
        genislik = int(k * 100) if k > 0 else 5
        bar_html += f"<div style='display:inline-block; width:{genislik}px; height:8px; background:{renk_bar}; margin-right:2px;' title='L{l}: {k:.3f}'></div>"

    html = f'''
    <div style="font-family:monospace;background:#0a0e17;border:2px solid #88ff44;border-radius:12px;padding:14px;margin:10px 0;">
        <div style="border-bottom:1px solid #88ff44;padding-bottom:6px;margin-bottom:10px;">
            <span style="color:#88ff44;font-weight:bold;">🔱 TITAN 4.4 | DAMPED RESONANCE KERNEL</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px;">
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">⚡ SON İTKİ GÜCÜ</div><div style="font-size:18px;color:#88ff44;">{kuvvet:.4f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">📐 COS THETA (BENZERLİK)</div><div style="font-size:18px;color:#ffaa44;">{benzerlik:.3f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">🎚️ ÇAPA KATSAYISI (V0)</div><div style="font-size:18px;color:#44ff88;">{stats.get('v0', 0.45)}</div>
            </div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;">
            <div style="font-size:9px;color:#5a7080;">🌊 SÖNÜMLENME PROFİLİ (Kuvvet Dağılımı)</div>
            <div style="margin-top:4px;">{bar_html}</div>
            <div style="font-size:10px;color:#88ff44;margin-top:4px;font-weight:bold;">Mevcut Bölge Durumu: {bolge}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:8px;">
            <div style="font-size:9px;color:#5a7080;">💬 HÜKÜMRAN ÇIKTISI</div>
            <div style="font-size:11px;color:#c9d4e0;max-height:300px;overflow-y:auto;line-height:1.4;margin-top:6px;">{cevap}</div>
        </div>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# 🔱 ARAYÜZ
# =============================================================================
soru_kutusu = widgets.Textarea(
    value='What is the most significant structural paradox in the concept of sovereign intelligence, and how can biological consciousness protect itself against its potential tyranny?',
    layout=widgets.Layout(width='100%', height='100px')
)
sor_btn = widgets.Button(description='🔱 REZONANSI BAŞLAT', button_style='success', layout=widgets.Layout(width='200px'))
cikti_alani = widgets.Output()

def on_sor(b):
    with cikti_alani:
        clear_output(wait=True)
        if not soru_kutusu.value.strip(): return
        print("⚡ Sönümlü Diferansiyel Denklem çözülüyor, TITAN felsefi rotada düşünüyor...")
        cevap, stats = soru_sor(soru_kutusu.value)
        clear_output(wait=True)
        kokpit_goster(cevap, stats)

sor_btn.on_click(on_sor)
display(widgets.VBox([soru_kutusu, sor_btn, cikti_alani]))
print("\n🚀 TITAN 4.4 Test Hücresi Aktif. Butona basıp dalgayı başlatabilirsin.")
