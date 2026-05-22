# =============================================================================
# 🔱 TITAN 4.6 | HİBRİT DAMPED RESONANCE + DİNAMİK V0 REGÜLATÖRÜ
# "Small lammy, dev gibi düşünür. Analitik, ağırbaşlı, hükümran."
#
# 📌 MİMARİ:
#   → TEMEL ISKELET  : TITAN 4.5 (DampedKernel, üssel sönümlenme, clone() güvencesi)
#   → V0 REGÜLATÖRÜ  : TITAN 4.3 (Dinamik V0 — semantik sapma ölçümlü basınç valfi)
#   → ANCHOR          : TITAN 4.5 (philosophical, conceptual, epistemological...)
#   → PROMPT PREFIX   : TITAN 4.5 (felsefi analiz talebi)
#   → HOOK            : TEK KANCA — çift enjeksiyon sıfırlandı
#   → İSTATİSTİK      : MERGED + dinamik_v0 alanı eklendi
#
# 🔬 ENTEGRASYON MANTIĞI:
#   4.3'ün dinamik V0 valfi → 4.5'in damped kernel'ına gömlendi.
#   V0_efektif = V0_temel(0.45) yerini V0_dinamik[0.25–0.65]'e bırakır.
#   Sapma arttıkça V0 yükselir (daha güçlü çekim), hizalandıkça gevşer.
#   DampedKernel'ın katman bazlı üssel sönümlemesi aynen korundu.
#   Tek hook → tek yönlendir() çağrısı → sıfır çakışma.
#
# 🔱 KRİTİK UYARI:
#   - AkbasCore sabitlerini değiştirme (V0_MIN/MAX, LAMBDA_DECAY, ANCHOR_LAYER_COUNT)
#   - Pusula vektör çıkarma mantığına dokunma
#   - _kuvvet() içindeki üssel sönümlenme eğrisine dokunma
#   - _dinamik_v0() içindeki sapma hesabına dokunma
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
# 🔱 HF TOKEN KONTROLÜ (Sessiz mod)
# =============================================================================
def hf_token_kontrol():
    token = os.environ.get('HF_TOKEN', None)
    return token if token else None

# =============================================================================
# 🔱 AKBASCORE V6.0 — HİBRİT KONFİGÜRASYON (TEK KAYNAK)
# =============================================================================
class AkbasCore:
    # ── RAY 1: TEMEL AKBAŞ YASASI ──────────────────────────────────────────
    V0             = 0.45   # Temel çapa (referans; runtime'da _dinamik_v0() yönetir)
    V0_MIN         = 0.25   # Hizalanmış durum — gevşek çekim  (4.3'den)
    V0_MAX         = 0.65   # Sapma durumu    — güçlü düzeltme (4.3'den)

    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

    # ── RAY 2: KAVRAMSAL ÇAPA (4.5'den — felsefi) ─────────────────────────
    COMPASS_ANCHORS = [
        "philosophical",    # felsefi derinlik
        "conceptual",       # kavramsal soyutlama
        "epistemological",  # bilgi kuramı
        "systemic",         # sistemsel düşünce
        "analytical"        # analitik titizlik
    ]

    # ── RAY 4: SÜZGEÇ RAYI ─────────────────────────────────────────────────
    MAX_TOKENS         = 350
    TEMPERATURE        = 0.55
    TOP_K              = 50
    TOP_P              = 0.90
    REPETITION_PENALTY = 1.5

    # ── RAY 5: DAMPED RESONANCE GEOMETRİSİ (4.5'den) ──────────────────────
    LAMBDA_DECAY              = 0.28   # Sönümlenme hızı
    ANCHOR_LAYER_COUNT        = 8      # Sabit hizalama katmanları
    MIN_ATTENUATION_THRESHOLD = 0.03   # Bu altında müdahale tamamen kesilir

    # ── BÖLGE SINIRLARI (UI referansı) ────────────────────────────────────
    HIZALAMA_KUVVET       = 0.80
    EVRENSEL_KOPRU_KUVVET = 0.40
    TAM_OZGURLUK_KUVVET   = 0.00

print("🔱 TITAN 4.6 | Hibrit Damped Resonance + Dinamik V0 Regülatörü")
print("="*65)
print("   • V0 Regülatörü:  Dinamik [0.25–0.65] | sapma bazlı çekim gücü")
print("   • R2 Kavram:      philosophical → conceptual → epistemological")
print("   • R4 Süzgeç:      Temp=0.55 | Rep.Penalty=1.5")
print("   • R5 Sönümlenme:  λ=0.28 | eşik=0.03 (Damped Resonance)")
print("   • Hook:           Tek kanca — çift enjeksiyon sıfır")
print("="*65)

# =============================================================================
# 🔱 PUSULA (KAVRAMSAL AĞ — 4.5'in dtype güvencesiyle)
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
            # dtype güvencesi: float tensor'ı modelin dtype'ına eşle
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
# 🔱 TITAN HİBRİT KERNEL — 4.6'nın Kalbi
#    4.5 DampedKernel iskeleti + 4.3 Dinamik V0 valfi
#    TEK yönlendir() → TEK hook → SIFIR çakışma
# =============================================================================
class TitanHybridKernel:
    def __init__(self, pusula_vector, v0=AkbasCore.V0):
        self.pusula           = pusula_vector
        self.base_v0          = v0          # Referans çapa (AkbasCore.V0 = 0.45)
        self.dinamik_v0       = v0          # Runtime'da güncellenir
        self.son_kuvvet       = 0.0
        self.son_benzerlik    = 0.0
        self.son_sapma        = 0.0         # 4.3'den: anlık sapma miktarı
        self.son_bolge        = "Başlangıç"
        self.aktif_kuvvet_haritasi = {}

    # ── 4.3'DEN: Kosinüs sapmasını ölçerek dinamik V0 üret ────────────────
    # Mantık: model pusuladan saptıkça V0 yükselir (daha güçlü çekim),
    #         hizalandıkça V0 alçalır (üretim özgürlüğü açılır).
    def _dinamik_v0(self, son_dusunce: torch.Tensor) -> float:
        pusula_cpu = self.pusula.to(device=son_dusunce.device, dtype=son_dusunce.dtype)
        # Kosinüs benzerliği [-1, 1] → sapma miktarı [0, 1]
        benzerlik_skaler = F.cosine_similarity(
            son_dusunce.squeeze(1),
            pusula_cpu.unsqueeze(0),
            dim=-1
        ).mean().item()
        sapma = 1.0 - (benzerlik_skaler + 1.0) / 2.0  # [0,1] normalize
        self.son_sapma = round(sapma, 4)
        # Lineer interpolasyon: sapma → V0 aralığı
        v0 = AkbasCore.V0_MIN + (AkbasCore.V0_MAX - AkbasCore.V0_MIN) * sapma
        self.dinamik_v0 = round(v0, 4)
        return v0

    # ── 4.5'DEN: Katman bazlı üssel sönümlenme kuvveti ────────────────────
    def _kuvvet(self, layer_idx: int) -> float:
        if layer_idx < AkbasCore.ANCHOR_LAYER_COUNT:
            self.son_bolge = "🏛️ R1: Hizalama Çapa Alanı"
            return AkbasCore.HIZALAMA_KUVVET                    # 0.80 sabit

        mesafe = layer_idx - AkbasCore.ANCHOR_LAYER_COUNT
        k = AkbasCore.EVRENSEL_KOPRU_KUVVET * math.exp(
            -AkbasCore.LAMBDA_DECAY * mesafe
        )

        if k < AkbasCore.MIN_ATTENUATION_THRESHOLD:
            self.son_bolge = "🕊️ R5: Hükümran Çıkış (Tam Özgürlük)"
            return 0.0

        self.son_bolge = f"🌉 R3: Mantık Köprüsü (k={k:.3f})"
        return k

    # ── HİBRİT yönlendir(): Tek giriş noktası — çakışma yok ──────────────
    def yönlendir(self, hidden_states, layer_idx: int):
        k = self._kuvvet(layer_idx)
        self.aktif_kuvvet_haritasi[layer_idx] = k

        if k == 0.0:
            return hidden_states   # Üst katmanlara dokunma

        with torch.no_grad():
            # 4.5'den: gradyan kilitlenmesini önleyen temiz clone
            steered = hidden_states.clone().detach()

            seq_len = steered.size(1)
            if seq_len == 0:
                return hidden_states

            son_dusunce = steered[:, -1:, :] if seq_len > 1 else steered

            # 4.3'den: kosinüs sapmasına göre V0'ı canlı güncelle
            # (Her adımda ölçüm → adaptif baskı)
            v0_efektif = self._dinamik_v0(son_dusunce)

            # 4.5'den: dtype + device uyumu
            pusula = self.pusula.to(
                device=son_dusunce.device,
                dtype=son_dusunce.dtype
            )

            # Ortak katki denklemi (her iki versiyondan korunan çekirdek)
            # v0 artık sabit değil: dinamik V0 valfi devrede
            benzerlik_vektoru = (son_dusunce * pusula).sum(dim=-1, keepdim=True)
            katki = v0_efektif * benzerlik_vektoru * k * 0.3
            katki = torch.clamp(katki, max=0.15)

            yonlendirilmis = son_dusunce + katki * pusula.view(1, 1, -1)

            if seq_len > 1:
                steered[:, -1:, :] = yonlendirilmis.to(steered.dtype)
            else:
                steered = yonlendirilmis.to(steered.dtype)

            # 4.5'den: cpu güvencesi (ekran donması önlenir)
            self.son_kuvvet    = katki.mean().cpu().item()
            self.son_benzerlik = benzerlik_vektoru.mean().cpu().item()

        return steered

    # ── İstatistik: 4.5 base + 4.3 dinamik_v0 + sapma alanı ──────────────
    def istatistik(self) -> dict:
        aktif_katmanlar = sum(1 for k in self.aktif_kuvvet_haritasi.values() if k > 0)
        ort_kuvvet      = (
            sum(self.aktif_kuvvet_haritasi.values()) / len(self.aktif_kuvvet_haritasi)
            if self.aktif_kuvvet_haritasi else 0
        )
        return {
            'kuvvet'         : round(self.son_kuvvet, 4),
            'benzerlik'      : round(self.son_benzerlik, 4),
            'bolge'          : self.son_bolge,
            'v0'             : self.dinamik_v0,     # Anlık dinamik V0
            'v0_temel'       : self.base_v0,        # Referans çapa
            'sapma'          : self.son_sapma,      # Kosinüs sapma miktarı [0,1]
            'sicaklik'       : AkbasCore.TEMPERATURE,
            'harita'         : dict(self.aktif_kuvvet_haritasi),
            'aktif_katmanlar': aktif_katmanlar,
            'ort_kuvvet'     : round(ort_kuvvet, 4),
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
# 🔱 PUSULA + HİBRİT KERNEL BAŞLATMA
# =============================================================================
pusula = Pusula(model, tokenizer, model.device)
titan  = TitanHybridKernel(pusula.get(), v0=AkbasCore.V0)

# =============================================================================
# 🔱 TEK KANCA ENJEKSİYONU (4.5'in orijinal ezme mantığı — dokunulmadı)
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
print(f"\n✓ Hibrit Damped Resonance kancası {n} katmana enjekte edildi")
print(f"   • 🏛️  R1-Hizalama:  katman 0-{AkbasCore.ANCHOR_LAYER_COUNT-1}  (%80 sabit)")
print(f"   • 🌉  R3-Köprü:     katman {AkbasCore.ANCHOR_LAYER_COUNT}-17  (üssel sönümlenme λ={AkbasCore.LAMBDA_DECAY})")
print(f"   • 🕊️  R5-Özgürlük: katman 18+  (%0 — hükümran çıkış)")
print(f"   • 🌀  V0 Valfi:     Dinamik [{AkbasCore.V0_MIN}–{AkbasCore.V0_MAX}] | sapma bazlı")

# =============================================================================
# 💬 SORGU (4.5'in felsefi prefix'i — korundu)
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

    yeni  = output_ids[0][inputs['input_ids'].shape[1]:]
    cevap = tokenizer.decode(yeni, skip_special_tokens=True)

    if not cevap or len(cevap.strip()) == 0:
        cevap = "[TITAN] (Cevap üretilemedi)"

    return cevap, titan.istatistik()

# =============================================================================
# 🔱 KOKPİT (4.5 görsel tabanı + 4.3 dinamik V0 paneli + sapma göstergesi)
# =============================================================================
def kokpit_goster(prompt: str, cevap: str, stats: dict):
    kuvvet          = stats.get('kuvvet', 0)
    benzerlik       = stats.get('benzerlik', 0)
    bolge           = stats.get('bolge', '?')
    sicaklik        = stats.get('sicaklik', 0.55)
    harita          = stats.get('harita', {})
    v0_anlik        = stats.get('v0', 0.45)
    v0_temel        = stats.get('v0_temel', 0.45)
    sapma           = stats.get('sapma', 0.0)
    aktif_katmanlar = stats.get('aktif_katmanlar', 0)
    ort_kuvvet      = stats.get('ort_kuvvet', 0)

    # Durum rengi: benzerlik bazlı
    if benzerlik > 0.5:
        renk, durum = '#44ff88', '🟢 HİZALI'
    elif benzerlik > 0.2:
        renk, durum = '#88ff44', '🟡 GEÇİŞ'
    else:
        renk, durum = '#ffaa44', '🟠 SERBEST'

    # V0 valfi rengi: sapma bazlı (4.3 mantığı)
    if sapma > 0.65:
        v0_renk, v0_durum = '#ff5555', '⚠️ YÜKSEK BASINÇ'
    elif sapma > 0.35:
        v0_renk, v0_durum = '#ffcc00', '🌀 ADAPTİF'
    else:
        v0_renk, v0_durum = '#44ff88', '✅ GEVŞEK'

    bolge_ikon = (
        '🏛️' if 'Hizalama' in bolge else
        '🌉' if 'Köprü'    in bolge else
        '🕊️'
    )

    # Sönümlenme bar (4.5'den)
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

    # V0 valfi görsel göstergesi (4.3'den ilham)
    v0_pct = int(((v0_anlik - AkbasCore.V0_MIN) /
                  (AkbasCore.V0_MAX - AkbasCore.V0_MIN)) * 100)
    v0_bar = (
        f"<div style='background:#1a2233;border-radius:4px;height:8px;width:100%;margin-top:4px;'>"
        f"<div style='background:{v0_renk};width:{v0_pct}%;height:100%;border-radius:4px;'></div>"
        f"</div>"
    )

    html = f'''
    <div style="font-family:monospace;background:#0a0e17;border:2px solid {renk};
                border-radius:12px;padding:14px;margin:10px 0;">

        <!-- BAŞLIK -->
        <div style="border-bottom:1px solid {renk};padding-bottom:6px;margin-bottom:10px;">
            <span style="color:{renk};font-weight:bold;">🔱 TITAN 4.6 | Hibrit Damped Resonance + Dinamik V0</span>
            <span style="color:#5a7080;font-size:10px;"> | {durum}</span>
        </div>

        <!-- TEMEL METRİKLER (4.5 düzeni) -->
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

        <!-- DİNAMİK V0 VALFİ (4.3'den — yeni panel) -->
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;
                    border:1px solid {v0_renk}33;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-size:9px;color:#5a7080;">🌀 DİNAMİK V0 VALFİ — {v0_durum}</div>
                <div style="font-size:10px;color:{v0_renk};font-weight:bold;">
                    {v0_anlik:.4f} <span style="color:#5a7080;font-size:8px;">(temel: {v0_temel})</span>
                </div>
            </div>
            {v0_bar}
            <div style="display:flex;justify-content:space-between;font-size:8px;color:#5a7080;margin-top:2px;">
                <span>V0_MIN={AkbasCore.V0_MIN} (hizalı)</span>
                <span>Sapma: {sapma:.3f}</span>
                <span>V0_MAX={AkbasCore.V0_MAX} (sapkın)</span>
            </div>
        </div>

        <!-- AKTİF RAY -->
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;">
            <div style="font-size:9px;color:#5a7080;">{bolge_ikon} AKTİF RAY</div>
            <div style="font-size:11px;color:{renk};font-weight:bold;">{bolge}</div>
        </div>

        <!-- ÇIKTI ALANI -->
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;">
            <div style="font-size:9px;color:#5a7080;">💬 HÜKÜMRAN ÇIKTISI</div>
            <div style="font-size:11px;color:#c9d4e0;max-height:350px;overflow-y:auto;
                        line-height:1.5;margin-top:6px;">
                {cevap}
            </div>
        </div>

        <!-- SÖNÜMLENME PROFİLİ (4.5'den) -->
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;">
            <div style="font-size:9px;color:#5a7080;margin-bottom:4px;">
                🌊 SÖNÜMLENME PROFİLİ — Katman Kuvvet Dağılımı (λ={AkbasCore.LAMBDA_DECAY})
            </div>
            <div style="padding:2px 0;">{bar_html}</div>
        </div>

        <!-- EK METRİKLER -->
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px;">
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">🧮 V0 TEMEL ÇAPA</div>
                <div style="font-size:16px;color:#88ccff;">{v0_temel}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">🔥 AKTİF KATMAN</div>
                <div style="font-size:16px;color:#ffaa44;">{aktif_katmanlar}/{len(harita)}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">📊 ORT. İTKİ</div>
                <div style="font-size:16px;color:#88ff44;">{ort_kuvvet:.4f}</div>
            </div>
        </div>

        <!-- İmza -->
        <div style="margin-top:8px;text-align:center;font-size:9px;color:#d4af37;">
            🔱 "Semantik dalga boyu kendi dengesini inşa ediyor." | TITAN 4.6 Hibrit
        </div>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# 🔱 ARAYÜZ (4.5 çift buton düzeni — korundu)
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
            print("⚡ TITAN 4.6 — Hibrit Damped Resonance aktif, düşünüyor...")
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
print("🔱 TITAN 4.6 HAZIR | Hibrit Damped Resonance + Dinamik V0")
print("="*65)
print(f"   • V0 Valfi:       [{AkbasCore.V0_MIN}–{AkbasCore.V0_MAX}] dinamik | sapma → güç ↑")
print(f"   • R1-Hizalama:    %80 sabit    | katman 0-{AkbasCore.ANCHOR_LAYER_COUNT-1}")
print(f"   • R3-Köprü:       üssel sönüm | λ={AkbasCore.LAMBDA_DECAY} | katman {AkbasCore.ANCHOR_LAYER_COUNT}-17")
print(f"   • R5-Özgürlük:    %0 tam serbest | katman 18+")
print(f"   • Anchorlar:      philosophical → conceptual → epistemological")
print(f"   • Temp={AkbasCore.TEMPERATURE} | RepPenalty={AkbasCore.REPETITION_PENALTY}")
print("="*65)
print("🚀 Hibrit sistem hazır.\n")

display(widgets.VBox([
    widgets.HTML(
        '<h3 style="font-family:monospace;color:#44ff88;margin:0;">'
        '🔱 TITAN 4.6 | Hibrit Damped Resonance + Dinamik V0 Regülatörü</h3>'
    ),
    widgets.HTML(
        '<p style="font-size:9px;color:#5a7080;margin:0 0 10px 0;">'
        '🏛️ Hizalama (0-7) → 🌉 Sönümlenme Köprüsü (8-17) → 🕊️ Özgürlük (18+) '
        '| λ=0.28 | V0∈[0.25–0.65] Dinamik Valf</p>'
    ),
    soru_kutusu,
    buton_kutusu,
    cikti_alani
]))

print("\n✅ TITAN 4.6 hazır. Soruyu sorabilirsiniz.")
