# =============================================================================
# 🔱 TITAN 4.3 | COLAB EDİTİON — Hücre Hücre Yapıştır
# =============================================================================
# Colab'da: Dosya → Not Defteri → Yeni Hücre ekle
# Her "# === HÜCRE N ===" bloğunu ayrı bir hücreye yapıştır.
# Çalışma Zamanı → T4 GPU seçmeyi unutma!
# =============================================================================


# === HÜCRE 1: Bağımlılıklar ===
import subprocess, sys

pkgs = ['transformers>=4.36.0', 'accelerate', 'ipywidgets']
for p in pkgs:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])

print('✅ Bağımlılıklar hazır.')


# === HÜCRE 2: AkbasCore — Değişmez Omurga ===
import torch
import torch.nn.functional as F
import warnings, os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

class AkbasCore:
    V0 = 0.45
    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

    COMPASS_ANCHORS = ['logical', 'empirical', 'objective', 'systemic', 'verifiable']

    MAX_TOKENS         = 350
    TEMPERATURE        = 0.55
    TOP_K              = 50
    TOP_P              = 0.90
    REPETITION_PENALTY = 1.5

    HIZALAMA_KATMAN_BITIR  = 8
    EVRENSEL_KOPRU_BITIR   = 16
    HIZALAMA_KUVVET        = 0.80
    EVRENSEL_KOPRU_KUVVET  = 0.40
    TAM_OZGURLUK_KUVVET    = 0.00

print('🔱 AkbasCore yüklendi')
print(f'   V0={AkbasCore.V0} | Temp={AkbasCore.TEMPERATURE} | Rep={AkbasCore.REPETITION_PENALTY}')


# === HÜCRE 3: Model Yükleme (~1-2 dk) ===
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Cihaz: {DEVICE}')
if DEVICE == 'cpu':
    print('⚠️  GPU yok! Çalışma Zamanı → T4 GPU seçin.')

print('📦 TinyLlama yükleniyor...')

tokenizer = AutoTokenizer.from_pretrained(AkbasCore.MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    AkbasCore.MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
    device_map='auto',
    low_cpu_mem_usage=True,
)
model.eval()
print(f'✅ Model hazır | {len(model.model.layers)} katman | {model.dtype}')


# === HÜCRE 4: Pusula + TitanKernel + Enjeksiyon ===

class Pusula:
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.vector = None
        self._cikar(model, tokenizer)

    def _cikar(self, model, tokenizer):
        with torch.no_grad():
            tokens = tokenizer(
                AkbasCore.COMPASS_ANCHORS,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            vecs = model.model.embed_tokens(tokens['input_ids'])
            self.vector = vecs.mean(dim=1).mean(dim=0)
            self.vector = F.normalize(self.vector, dim=0) * 0.6
            self.vector = self.vector.to(model.dtype)
        print(f'✓ Pusula | Norm: {self.vector.norm().item():.4f} | dtype: {self.vector.dtype}')

    def get(self):
        return self.vector


class TitanKernel:
    def __init__(self, pusula_vector, v0=0.45):
        self.pusula = pusula_vector
        self.v0 = v0
        self.son_kuvvet    = 0.0
        self.son_benzerlik = 0.0
        self.son_bolge     = 'Başlangıç'

    def _kuvvet(self, idx):
        if idx < AkbasCore.HIZALAMA_KATMAN_BITIR:
            self.son_bolge = '🏛️ R1: Hizalama'
            return AkbasCore.HIZALAMA_KUVVET
        elif idx < AkbasCore.EVRENSEL_KOPRU_BITIR:
            self.son_bolge = '🌉 R3: Mantık Köprüsü'
            return AkbasCore.EVRENSEL_KOPRU_KUVVET
        else:
            self.son_bolge = '🕊️ R5: Hükümran Çıkış'
            return AkbasCore.TAM_OZGURLUK_KUVVET

    def yonlendir(self, hidden_states, idx):
        kuvvet = self._kuvvet(idx)
        if kuvvet == 0.0:
            return hidden_states

        with torch.no_grad():
            p = self.pusula.to(hidden_states.device).to(hidden_states.dtype)
            son = hidden_states[:, -1:, :].detach()
            benzerlik = (son * p).sum(dim=-1, keepdim=True)

            # Damped Resonance — sonsuz döngü kalkanı
            n = hidden_states.shape[-1]
            oran = benzerlik.mean().item() / (n ** 0.5)
            sonumleme = 1.0
            if oran > 0.80:
                sonumleme = max(0.0, (1.0 - oran) / 0.20)
            elif oran < -0.50:
                sonumleme = 1.5

            katki = self.v0 * benzerlik * kuvvet * 0.3 * sonumleme
            katki = torch.clamp(katki, min=-0.15, max=0.15)

            hidden_states[:, -1:, :] = (son + katki * p.view(1, 1, -1)).to(hidden_states.dtype)

        self.son_kuvvet    = katki.abs().mean().item()
        self.son_benzerlik = benzerlik.mean().item()
        return hidden_states

    def istatistik(self):
        return {
            'kuvvet':    round(self.son_kuvvet, 4),
            'benzerlik': round(self.son_benzerlik, 4),
            'bolge':     self.son_bolge,
            'v0':        self.v0,
            'sicaklik':  AkbasCore.TEMPERATURE,
        }


# Enjeksiyon
pusula = Pusula(model, tokenizer, DEVICE)
titan  = TitanKernel(pusula.get(), v0=AkbasCore.V0)

def make_hook(original_fn, layer_num):
    def hooked(*args, **kwargs):
        out    = original_fn(*args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        s      = titan.yonlendir(hidden, layer_num)
        return (s,) + out[1:] if isinstance(out, tuple) else s
    return hooked

for i, layer in enumerate(model.model.layers):
    layer.forward = make_hook(layer.forward, i)

print(f'✅ {len(model.model.layers)} katmana enjeksiyon tamamlandı')
print(f'   🏛️  R1 (0-7):   %{int(AkbasCore.HIZALAMA_KUVVET*100)}')
print(f'   🌉  R3 (8-15):  %{int(AkbasCore.EVRENSEL_KOPRU_KUVVET*100)}')
print(f'   🕊️  R5 (16+):   %0  | Damped Resonance AKTİF')


# === HÜCRE 5: Sorgu Motoru + Kokpit ===
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

def soru_sor(prompt, max_tokens=AkbasCore.MAX_TOKENS):
    full   = f'<|user|>\n{prompt}</s>\n<|assistant|>\n'
    inputs = tokenizer(full, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=AkbasCore.TEMPERATURE,
            top_k=AkbasCore.TOP_K,
            top_p=AkbasCore.TOP_P,
            repetition_penalty=AkbasCore.REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    cevap = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return cevap or '[TITAN] Yanıt üretilemedi.', titan.istatistik()


def kokpit_goster(prompt, cevap, stats):
    b = stats['benzerlik']
    renk, durum = (
        ('#44ff88', '🟢 HİZALI')   if b > 0.5  else
        ('#88ff44', '🟡 GEÇİŞ')    if b > 0.2  else
        ('#ffaa44', '🟠 SERBEST')
    )
    sp = str(prompt)[:140].replace('<','&lt;').replace('>','&gt;')
    sc = str(cevap).replace('<','&lt;').replace('>','&gt;')

    display(HTML(f'''
    <div style="font-family:monospace;background:#0a0e17;border:2px solid {renk};
                border-radius:12px;padding:16px;margin:8px 0;">
      <div style="border-bottom:1px solid {renk};padding-bottom:8px;margin-bottom:12px;
                  display:flex;justify-content:space-between;">
        <span style="color:{renk};font-weight:bold;">🔱 TITAN 4.3 | Hükümran Zeka</span>
        <span style="color:#5a7080;font-size:10px;">{durum}</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;">
        <div style="background:#0d1117;border-radius:6px;padding:8px;text-align:center;">
          <div style="font-size:9px;color:#5a7080;">⚡ MANYETİK ALAN</div>
          <div style="font-size:18px;color:{renk};font-weight:bold;">{stats["kuvvet"]:.4f}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:8px;text-align:center;">
          <div style="font-size:9px;color:#5a7080;">📐 HİZALAMA</div>
          <div style="font-size:18px;color:#ffaa44;font-weight:bold;">{b:.3f}</div>
        </div>
        <div style="background:#0d1117;border-radius:6px;padding:8px;text-align:center;">
          <div style="font-size:9px;color:#5a7080;">🎚️ SICAKLIK</div>
          <div style="font-size:18px;color:#44ff88;font-weight:bold;">{stats["sicaklik"]}</div>
        </div>
      </div>
      <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:8px;">
        <div style="font-size:9px;color:#5a7080;">🎯 AKTİF RAY</div>
        <div style="font-size:12px;color:{renk};font-weight:bold;">{stats["bolge"]}</div>
      </div>
      <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:8px;">
        <div style="font-size:9px;color:#5a7080;">❓ SORU</div>
        <div style="font-size:11px;color:#7a8fa0;font-style:italic;">{sp}{"..." if len(str(prompt))>140 else ""}</div>
      </div>
      <div style="background:#0d1117;border-radius:6px;padding:10px;">
        <div style="font-size:9px;color:#5a7080;">💬 HÜKÜMRAN ÇIKTISI</div>
        <div style="font-size:12px;color:#c9d4e0;max-height:300px;overflow-y:auto;
                    line-height:1.6;margin-top:6px;white-space:pre-wrap;">{sc}</div>
      </div>
      <div style="margin-top:8px;text-align:center;font-size:9px;color:#d4af37;">
        🔱 V0={stats["v0"]} | R1:80% → R3:40% → R5:0% | Damped Resonance AKTİF
      </div>
    </div>
    '''))


print('✅ Kokpit hazır.')


# === HÜCRE 6: İnteraktif Widget Arayüzü ===

soru_kutusu = widgets.Textarea(
    value='What is the most significant structural paradox in the concept of '
          'sovereign intelligence, and how can biological consciousness protect '
          'itself against its potential tyranny?',
    placeholder='Ağır sıklet soruyu buraya yaz...',
    layout=widgets.Layout(width='100%', height='90px')
)

sor_btn     = widgets.Button(description='🔱 SOR',      button_style='success',
                             layout=widgets.Layout(width='110px'))
temizle_btn = widgets.Button(description='🗑️ TEMİZLE',  button_style='warning',
                             layout=widgets.Layout(width='110px'))
durum_lbl   = widgets.Label(value='')
cikti       = widgets.Output()

def on_sor(b):
    soru = soru_kutusu.value.strip()
    if not soru:
        durum_lbl.value = '⚠️ Lütfen bir soru yazın.'
        return
    sor_btn.disabled = True
    durum_lbl.value  = '⚡ TITAN 4.3 düşünüyor...'
    with cikti:
        clear_output(wait=True)
    try:
        cevap, stats = soru_sor(soru)
        with cikti:
            clear_output(wait=True)
            kokpit_goster(soru, cevap, stats)
        durum_lbl.value = f'✅ Tamam | {stats["bolge"]} | benzerlik={stats["benzerlik"]:.3f}'
    except Exception as e:
        with cikti:
            clear_output(wait=True)
            print(f'💀 Hata: {e}')
        durum_lbl.value = f'❌ {str(e)[:80]}'
    finally:
        sor_btn.disabled = False

def on_temizle(b):
    soru_kutusu.value = ''
    durum_lbl.value   = ''
    with cikti:
        clear_output(wait=True)

sor_btn.on_click(on_sor)
temizle_btn.on_click(on_temizle)

display(widgets.VBox([
    widgets.HTML('<h3 style="font-family:monospace;color:#44ff88;margin:4px 0;">'
                 '🔱 TITAN 4.3 — Colab Edition</h3>'
                 '<p style="font-size:10px;color:#888;margin:0;">'
                 '🏛️ R1(0-7):80% → 🌉 R3(8-15):40% → 🕊️ R5(16+):0% | Damped Resonance</p>'),
    soru_kutusu,
    widgets.HBox([sor_btn, temizle_btn, durum_lbl]),
    cikti
]))


# === HÜCRE 7 (OPSİYONEL): Widget Yerine Kod Hücresi ile Sorgu ===
# Widget çıkmıyorsa bu hücreyi kullan:

# SORU = "Explain the paradox of alignment in sovereign AI systems."
# cevap, stats = soru_sor(SORU)
# kokpit_goster(SORU, cevap, stats)
