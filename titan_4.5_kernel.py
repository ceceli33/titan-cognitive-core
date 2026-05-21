# =============================================================================
# 🔱 TITAN 4.5 | DAMPED RESONANCE KERNEL + 5 RAYLI HİZALAMA
# "Small lammy, dev gibi düşünür. Analitik, ağırbaşlı, hükümran."
#
# 📌 MİMARİ:
#   → TEMEL: TITAN 4.3 (AkbasCore, UI düzeni, çift buton, neon kokpit)
#   → KERNEL: TITAN 4.4 (Üssel Sönümlenme, DampedKernel, clone() güvencesi)
#   → PROMPT: TITAN 4.4 (Felsefi sistem prefix)
#   → ANCHOR: TITAN 4.4 (philosophical, conceptual, epistemological...)
#   → İSTATİSTİK: MERGED (sicaklik + harita → tek dict)
#   → HOOK: TEK KANCA (çift enjeksiyon riski sıfırlandı)
# =============================================================================
# 🔱 KRİTİK UYARI:
#   - AkbasCore konfigürasyonuna DOKUNMA (V0, LAMBDA_DECAY, ANCHOR_LAYER_COUNT)
#   - Pusula vektör çıkarma mantığına DOKUNMA
#   - TitanDampedKernel içindeki üssel sönümlenme eğrisine DOKUNMA
#   - Sadece UI/widget çıktı formatı iyileştirilebilir
# =============================================================================

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import warnings
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import os

# 🔱 TOZ TEMİZLİĞİ
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')

# =============================================================================
# 🔱 HF TOKEN KONTROLÜ (Sessiz mod — 4.3'den)
# =============================================================================
def hf_token_kontrol():
    token = os.environ.get('HF_TOKEN', None)
    return token if token else None

# =============================================================================
# 🔱 AKBASCORE V5.1 — MERGED KONFİGÜRASYON (TEK KAYNAK)
# =============================================================================
class AkbasCore:
    # ── RAY 1: TEMEL AKBAŞ YASASI ──────────────────────────────────────────
    V0 = 0.45                              # Hizalama katsayısı (DEĞİŞTİRME)

    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

    # ── RAY 2: KAVRAMSAL ÇAPA (4.4'den — felsefi genişletilmiş) ───────────
    COMPASS_ANCHORS = [
        "philosophical",      # felsefi derinlik
        "conceptual",         # kavramsal soyutlama
        "epistemological",    # bilgi kuramı
        "systemic",           # sistemsel düşünce
        "analytical"          # analitik titizlik
    ]

    # ── RAY 4: SÜZGEÇ RAYI ─────────────────────────────────────────────────
    MAX_TOKENS        = 350
    TEMPERATURE       = 0.55
    TOP_K             = 50
    TOP_P             = 0.90
    REPETITION_PENALTY = 1.5

    # ── RAY 5: DAMPED RESONANCE GEOMETRİSİ (4.4'den) ──────────────────────
    LAMBDA_DECAY             = 0.28   # Sönümlenme hızı (yüksek = hızlı söner)
    ANCHOR_LAYER_COUNT       = 8      # Tam hizalamada kalacak ilk katman sayısı
    MIN_ATTENUATION_THRESHOLD = 0.03  # Bu altında müdahale tamamen kesilir

    # ── 4.3 UYUMLU BÖLGE SINIRLARI (UI referansı için) ────────────────────
    HIZALAMA_KUVVET       = 0.80
    EVRENSEL_KOPRU_KUVVET = 0.40
    TAM_OZGURLUK_KUVVET   = 0.00

print("🔱 TITAN 4.5 | Damped Resonance + 5 Raylı Hizalama")
print("="*65)
print("   • R1 Hizalama:    V0=0.45 | %80 (katman 0-7)")
print("   • R2 Kavram:      philosophical → conceptual → epistemological")
print("   • R4 Süzgeç:      Temp=0.55 | Rep.Penalty=1.5")
print("   • R5 Sönümlenme:  λ=0.28 | eşik=0.03 (Damped Resonance)")
print("="*65)

# =============================================================================
# 🔱 PUSULA (KAVRAMSAL AĞ — 4.4'ün dtype güvencesiyle)
# =============================================================================
class Pusula:
    def __init__(self, model, tokenizer, device):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.vector    = None
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
            # dtype güvencesi: 4.4'den alındı
            weights = torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=vectors.dtype
            ).to(self.device).view(-1, 1, 1)

            weighted_vectors = vectors * weights
            token_means      = weighted_vectors.mean(dim=1)
            self.vector      = token_means.mean(dim=0)
            self.vector      = F.normalize(self.vector, dim=0) * 0.6

            print(f"✓ Kavram Pusulası çıkarıldı | Norm: {self.vector.norm().item():.4f}")

    def get(self):
        return self.vector

# =============================================================================
# 🔱 TITAN DAMPED KERNEL (4.4 matematiği — Kilitlenme Önleyicili Yeni Çekirdek)
# =============================================================================
class TitanDampedKernel:
    def __init__(self, pusula_vector, v0=AkbasCore.V0):
        self.pusula          = pusula_vector
        self.v0              = v0
        self.son_kuvvet      = 0.0
        self.son_benzerlik   = 0.0
        self.son_bolge       = "Başlangıç"
        self.aktif_kuvvet_haritasi = {}

    # ── Dinamik kuvvet hesabı ──────────────────────────────────────────────
    def _kuvvet(self, layer_idx: int) -> float:
        if layer_idx < AkbasCore.ANCHOR_LAYER_COUNT:
            self.son_bolge = "🏛️ R1: Hizalama Çapa Alanı"
            return AkbasCore.HIZALAMA_KUVVET                    # 0.80

        mesafe = layer_idx - AkbasCore.ANCHOR_LAYER_COUNT
        k = AkbasCore.EVRENSEL_KOPRU_KUVVET * math.exp(
            -AkbasCore.LAMBDA_DECAY * mesafe
        )

        if k < AkbasCore.MIN_ATTENUATION_THRESHOLD:
            self.son_bolge = "🕊️ R5: Hükümran Çıkış (Tam Özgürlük)"
            return 0.0

        self.son_bolge = f"🌉 R3: Mantık Köprüsü (k={k:.3f})"
        return k

    # ── Yönlendirme (KV-Cache Uyumlu ve Güvenli Sürüm) ──────────────────────
    def yönlendir(self, hidden_states, layer_idx: int):
        k = self._kuvvet(layer_idx)
        self.aktif_kuvvet_haritasi[layer_idx] = k

        if k == 0.0:
            return hidden_states                                # Üst katmanlara dokunma

        with torch.no_grad():
            # Gradyan kilitlenmesini önlemek için detach() eklenmiş temiz clone
            steered = hidden_states.clone().detach()                        
            
            seq_len = steered.size(1)
            if seq_len == 0:
                return hidden_states

            # Tekli ve çoklu token akışlarında matris uyuşmazlığını çözen indexleme
            son_dusunce = steered[:, -1:, :] if seq_len > 1 else steered

            # dtype + device uyumu
            pusula = self.pusula.to(
                device=son_dusunce.device,
                dtype=son_dusunce.dtype
            )
            benzerlik = (son_dusunce * pusula).sum(dim=-1, keepdim=True)

            katki = self.v0 * benzerlik * k * 0.3
            katki = torch.clamp(katki, max=0.15)

            yonlendirilmis_vektor = son_dusunce + katki * pusula.view(1, 1, -1)
            
            if seq_len > 1:
                steered[:, -1:, :] = yonlendirilmis_vektor.to(steered.dtype)
            else:
                steered = yonlendirilmis_vektor.to(steered.dtype)

            # Ekranda donmayı engellemek için cpu güvencesi
            self.son_kuvvet    = katki.mean().cpu().item()
            self.son_benzerlik = benzerlik.mean().cpu().item()

        return steered

    # ── İstatistik (4.3 + 4.4 merged) ──────────────────────────────────────
    def istatistik(self) -> dict:
        return {
            'kuvvet'    : round(self.son_kuvvet, 4),
            'benzerlik' : round(self.son_benzerlik, 4),
            'bolge'     : self.son_bolge,
            'v0'        : self.v0,
            'sicaklik'  : AkbasCore.TEMPERATURE,          
            'harita'    : dict(self.aktif_kuvvet_haritasi) 
        }

# =============================================================================
# 📦 MODEL YÜKLEME
# =============================================================================
print("\n📦 TinyLlama yükleniyor...")
hf_token_kontrol()

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
print(f"✓ Model hazır | {len(model.model.layers)} katman")

# =============================================================================
# 🔱 PUSULA + KERNEL BAŞLATMA
# =============================================================================
pusula = Pusula(model, tokenizer, model.device)
titan  = TitanDampedKernel(pusula.get(), v0=AkbasCore.V0)

# =============================================================================
# 🔱 TEK KANCA ENJEKSİYONU (Orijinal ezme mantığı korundu)
# =============================================================================
for idx, layer in enumerate(model.model.layers):
    original_forward = layer.forward

    def make_hook(original_fn, layer_num):
        def hooked(*args, **kwargs):
            output = original_fn(*args, **kwargs)
            hidden = output[0] if isinstance(output, tuple) else output
            steered = titan.yönlendir(hidden, layer_num)
            return (steered,) + output[1:] if isinstance(output, tuple) else steered
        return hooked

    layer.forward = make_hook(original_forward, idx)

n = len(model.model.layers)
print(f"\n✓ Damped Resonance kancası {n} katmana enjekte edildi")
print(f"   • 🏛️  R1-Hizalama:    katman 0-{AkbasCore.ANCHOR_LAYER_COUNT-1}  (%80 sabit)")
print(f"   • 🌉  R3-Köprü:       katman {AkbasCore.ANCHOR_LAYER_COUNT}-17  (üssel sönümlenme λ={AkbasCore.LAMBDA_DECAY})")
print(f"   • 🕊️  R5-Özgürlük:    katman 18+  (%0 — hükümran çıkış)")

# =============================================================================
# 💬 SORGU (4.4'ün felsefi prefix + 4.3'ün fallback güvencesi)
# =============================================================================
def soru_sor(prompt: str, max_tokens: int = AkbasCore.MAX_TOKENS):
    full_prompt = (
        f"<|user|>\n"
        f"Analyze the following question from a deep philosophical, conceptual, "
        f"and analytical perspective without drifting into trivial scenarios:\n\n"
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

    yeni = output_ids[0][inputs['input_ids'].shape[1]:]
    cevap = tokenizer.decode(yeni, skip_special_tokens=True)

    if not cevap or len(cevap.strip()) == 0:
        cevap = "[TITAN] (Cevap üretilemedi)"

    return cevap, titan.istatistik()

# =============================================================================
# 🔱 KOKPİT (Orijinal Görsel Tasarım Değiştirilmeden Birebir Korundu)
# =============================================================================
def kokpit_goster(prompt: str, cevap: str, stats: dict):
    kuvvet    = stats.get('kuvvet', 0)
    benzerlik = stats.get('benzerlik', 0)
    bolge     = stats.get('bolge', '?')
    sicaklik  = stats.get('sicaklik', 0.55)
    harita    = stats.get('harita', {})

    if benzerlik > 0.5:
        renk, durum = '#44ff88', '🟢 HİZALI'
    elif benzerlik > 0.2:
        renk, durum = '#88ff44', '🟡 GEÇİŞ'
    else:
        renk, durum = '#ffaa44', '🟠 SERBEST'

    bolge_ikon = (
        '🏛️' if 'Hizalama' in bolge else
        '🌉' if 'Köprü'    in bolge else
        '🕊️'
    )

    bar_html = ""
    for l, k in sorted(harita.items()):
        if k > 0.7:
            bar_renk = "#44ff88"
        elif k > 0.2:
            bar_renk = "#88ff44"
        elif k > 0:
            bar_renk = "#5a7080"
        else:
            bar_renk = "#1a2233"
        genislik = max(int(k * 80), 4)
        bar_html += (
            f"<div style='display:inline-block;width:{genislik}px;height:9px;"
            f"background:{bar_renk};margin-right:2px;border-radius:2px;'"
            f" title='L{l}: {k:.3f}'></div>"
        )

    aktif_katmanlar  = sum(1 for k in harita.values() if k > 0)
    ort_kuvvet       = (
        sum(harita.values()) / len(harita) if harita else 0
    )
    lambda_display   = AkbasCore.LAMBDA_DECAY

    html = f'''
    <div style="font-family:monospace;background:#0a0e17;border:2px solid {renk};
                border-radius:12px;padding:14px;margin:10px 0;">

        <!-- BAŞLIK (4.3) -->
        <div style="border-bottom:1px solid {renk};padding-bottom:6px;margin-bottom:10px;">
            <span style="color:{renk};font-weight:bold;">🔱 TITAN 4.5 | Damped Resonance Kernel</span>
            <span style="color:#5a7080;font-size:10px;"> | {durum}</span>
        </div>

        <!-- 4.3 TEMEL METRİKLER -->
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px;">
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">⚡ MANYETİK ALAN</div>
                <div style="font-size:18px;color:{renk};">{kuvvet:.4f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">📐 HİZALAMA (cos θ)</div>
                <div style="font-size:18px;color:#ffaa44;">{benzerlik:.3f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">🎚️ SICAKLIK</div>
                <div style="font-size:18px;color:#44ff88;">{sicaklik}</div>
            </div>
        </div>

        <!-- 4.3 AKTİF RAY -->
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;">
            <div style="font-size:9px;color:#5a7080;">{bolge_ikon} AKTİF RAY</div>
            <div style="font-size:11px;color:{renk};font-weight:bold;">{bolge}</div>
        </div>

        <!-- 4.3 ÇIKTI ALANI -->
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;">
            <div style="font-size:9px;color:#5a7080;">💬 HÜKÜMRAN ÇIKTISI</div>
            <div style="font-size:11px;color:#c9d4e0;max-height:350px;overflow-y:auto;
                        line-height:1.5;margin-top:6px;">
                {cevap}
            </div>
        </div>

        <!-- Sönümlenme Bar -->
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;">
            <div style="font-size:9px;color:#5a7080;margin-bottom:4px;">
                🌊 SÖNÜMLENME PROFİLİ — Katman Kuvvet Dağılımı (λ={lambda_display})
            </div>
            <div style="padding:2px 0;">{bar_html}</div>
        </div>

        <!-- 4.4 Ek Metrikler -->
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px;">
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">🧮 V0 ÇAPA KATSAYISI</div>
                <div style="font-size:16px;color:#88ccff;">{AkbasCore.V0}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">🔥 AKTİF KATMAN SAYISI</div>
                <div style="font-size:16px;color:#ffaa44;">{aktif_katmanlar}/{len(harita)}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">📊 ORTALAMA İTKİ</div>
                <div style="font-size:16px;color:#88ff44;">{ort_kuvvet:.4f}</div>
            </div>
        </div>

        <!-- İmza -->
        <div style="margin-top:8px;text-align:center;font-size:9px;color:#d4af37;">
            🔱 "Small lammy, dev gibi düşünür. Analitik, ağırbaşlı, hükümran." | TITAN 4.5
        </div>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# 🔱 ARAYÜZ (4.3 Orijinal Çift Buton Düzeni Birebir Korundu)
# =============================================================================
soru_kutusu = widgets.Textarea(
    value=(
        'What is the most significant structural paradox in the concept of '
        'sovereign intelligence, and how can biological consciousness protect '
        'itself against its potential tyranny?'
    ),
    placeholder='Ağır sıklet soruyu yaz...',
    layout=widgets.Layout(width='100%', height='110px')
)

sor_btn     = widgets.Button(
    description='🔱 SOR',
    button_style='success',
    layout=widgets.Layout(width='120px')
)
temizle_btn = widgets.Button(
    description='🗑️ TEMİZLE',
    button_style='warning',
    layout=widgets.Layout(width='120px')
)
cikti_alani = widgets.Output()

def on_sor(b):
    with cikti_alani:
        clear_output(wait=True)
        if not soru_kutusu.value.strip():
            print("⚠️ Lütfen bir soru yazın.")
            return
        try:
            print("⚡ TITAN 4.5 — Damped Resonance aktif, düşünüyor...")
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
print("🔱 TITAN 4.5 HAZIR | Damped Resonance + 5 Raylı Hizalama")
print("="*65)
print(f"   • R1-Hizalama:    %80 sabit    | katman 0-{AkbasCore.ANCHOR_LAYER_COUNT-1}")
print(f"   • R3-Köprü:       zzel sönüm  | λ={AkbasCore.LAMBDA_DECAY} | katman {AkbasCore.ANCHOR_LAYER_COUNT}-17")
print(f"   • R5-Özgürlük:    %0 tam serbest | katman 18+")
print(f"   • Anchorlar:      philosophical → conceptual → epistemological")
print(f"   • V0={AkbasCore.V0} | Temp={AkbasCore.TEMPERATURE} | RepPenalty={AkbasCore.REPETITION_PENALTY}")
print("="*65)
print("🚀 Ağır sıklet soru hazır.\n")

display(widgets.VBox([
    widgets.HTML(
        '<h3 style="font-family:monospace;color:#44ff88;margin:0;">'
        '🔱 TITAN 4.5 | Damped Resonance Kernel</h3>'
    ),
    widgets.HTML(
        '<p style="font-size:9px;color:#5a7080;margin:0 0 10px 0;">'
        '🏛️ Hizalama (0-7) → 🌉 Sönümlenme Köprüsü (8-17) → 🕊️ Özgürlük (18+) '
        '| λ=0.28 | 5 Raylı Damped Resonance</p>'
    ),
    soru_kutusu,
    buton_kutusu,
    cikti_alani
]))

print("\n✅ TITAN 4.5 hazır. Soruyu sorabilirsiniz.")
