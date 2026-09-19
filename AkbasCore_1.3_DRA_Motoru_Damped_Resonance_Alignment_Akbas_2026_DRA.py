# =============================================================================
# AkbasCore 1.3 — DRA Motoru (Damped Resonance Alignment)
# (Akbaş) 2026 DRA
# =============================================================================
#
# BU SÜRÜM NEDİR?
#   Bu, projenin uzun bir test/doğrulama sürecinin (irili ufaklı 100'ün
#   üzerinde deney) sonunda ortaya çıkan, SADELEŞTİRİLMİŞ ve EN İYİ
#   AYARLARLA SABİTLENMİŞ sürümüdür. Ayar kolu, çoklu eksen, Dual Mode gibi
#   deneysel özellikler kasıtlı olarak kaldırıldı -- çünkü hangi ayarın en
#   dengeli sonucu verdiği zaten test edildi ve koda sabitlendi.
#
#   Kullanım tek amaçlı: bir sıcaklık değeri ve bir soru gir, cevabı al.
#   Cevabın ALTINDA sayısal bir şey GÖRMEZSİN -- ekran sade kalsın diye.
#   Ama "Kopyalamak İçin" kutusunda soru+cevap+TÜM sayısal telemetri,
#   Türkçe isimlerle VE ne anlama geldiklerini anlatan açıklamalarla birlikte
#   hazır bekliyor -- bunu kopyalayıp bir yapay zekaya ya da başka birine
#   gösterdiğinde, içeride ne olduğu ekstra bir açıklama gerekmeden anlaşılır.
#
# MOTOR NEDİR (kısaca)?
#   AkbasCore, bir dil modelinin gizli durumunu (hidden state), üretim
#   sırasında, modelin AĞIRLIKLARINA hiç dokunmadan, "Pusula" adı verilen
#   sabit bir yöne doğru KATMAN KATMAN, çok küçük ve kendi kendini
#   sınırlayan bir itkiyle nazikçe yönlendirir. Yöntemin adı "Sönümlü
#   Rezonans Hizalaması" (Damped Resonance Alignment, DRA) -- itki erken
#   katmanlarda güçlü başlar, derinlere indikçe sönümlenir ama küçük bir
#   "taban" seviyesinde kalıcı kalır.
#
# BU SÜRÜMDE SABİTLENEN AYARLAR (test edilip en dengeli bulunanlar):
#   İvme (itki gücü)     = 1.00
#   Sönüm (sönümlenme hızı) = 0.30
#   Zirve (başlangıç genliği) = 0.70
#   Taban (kalıcı zemin)  = 0.10
#   Terazi (anayasa kategori ağırlıkları) = eşit (test edildi, özel
#     ağırlıkların (0.9228 vb.) sisteme ölçülebilir bir katkısı olmadığı
#     kanıtlandı -- kod burada bilinçli olarak sadeleştirildi)
# =============================================================================

import subprocess, sys, os, shutil, time, math, json
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import ninja  # noqa
except ImportError:
    print("ninja kuruluyor..."); _pip("ninja")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline
from IPython.display import HTML, display

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 72)
print("AkbasCore 1.3 — DRA Motoru yükleniyor... (Akbaş, 2026)")
print("=" * 72)

# =============================================================================
# C++ ÇEKİRDEK — Sönümlü Rezonans formülü (test edilmiş, değiştirilmedi)
# =============================================================================
_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

torch::Tensor akbas_steer(
    torch::Tensor hidden,
    torch::Tensor pusula,
    float ivme,
    int layer_idx,
    float sonum,
    float zirve,
    float taban,
    torch::Tensor prev_cosine,
    torch::Tensor log_buf
) {
    auto h  = hidden.contiguous();
    auto p  = pusula.contiguous();
    auto lb = log_buf.contiguous();
    if (layer_idx >= 20) return h;

    const int B = h.size(0), S = h.size(1), D = h.size(2);
    float t = (float)layer_idx;

    float       *hp  = h.data_ptr<float>();
    float       *pcp = prev_cosine.data_ptr<float>();
    float       *lbp = lb.data_ptr<float>();
    const float *pp  = p.data_ptr<float>();

    for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s) {
        float *tok = hp + (b * S * D) + (s * D);
        int    idx = b * S + s;

        float dot = 0, sq = 0;
        for (int j = 0; j < D; ++j) {
            dot += tok[j] * pp[j];
            sq  += tok[j] * tok[j];
        }
        float norm = sqrtf(sq) + 1e-6f;
        float cs   = std::clamp(dot / norm, -1.0f, 1.0f);

        float unc = 1.0f - fabsf(cs);
        float dw  = sonum + unc * 0.20f;
        float kb  = zirve * expf(-dw * t) * (1.0f + dw * t) + taban;

        float dr = std::clamp(cs - pcp[idx], -0.15f, 0.15f);
        if (cs > 0.80f && dr < 0.0f) dr *= 0.30f;

        float kv = kb;
        if      (dr > 0) kv *= (1.0f - dr    * 0.30f);
        else if (dr < 0) kv *= (1.0f + (-dr) * 0.30f);
        kv = std::clamp(kv, 0.05f, 1.0f);
        pcp[idx] = cs;

        float son = 1.0f;
        if      (cs >  0.75f) son = (1.0f - cs) / 0.25f;
        else if (cs < -0.40f) son = 1.6f;

        float mk    = std::clamp(norm * 0.045f, 0.04f, 0.20f);
        float katki = std::clamp(ivme * cs * kv * 0.32f * son, -mk, mk);

        for (int j = 0; j < D; ++j) tok[j] += katki * pp[j];

        if (S > 1 && b == 0 && s == 0 && layer_idx < 20) {
            lbp[layer_idx * 4 + 0] = cs;
            lbp[layer_idx * 4 + 1] = kb;
            lbp[layer_idx * 4 + 2] = kv;
            lbp[layer_idx * 4 + 3] = katki;
        }
    }
    return h;
}
"""

if not torch.utils.cpp_extension.is_ninja_available():
    import site
    for _sp in site.getsitepackages():
        _b = os.path.normpath(os.path.join(_sp, "..", "..", "bin"))
        if os.path.isdir(_b):
            os.environ["PATH"] = _b + ":" + os.environ.get("PATH", "")
for _n in ["akbascore13_f1", "akbascore13_f2"]:
    for _r in ["/tmp/akbascore13", os.path.expanduser("~/.cache/torch_extensions")]:
        _p = os.path.join(_r, _n)
        if os.path.exists(_p):
            shutil.rmtree(_p, ignore_errors=True)

def _build(name):
    d = f"/tmp/akbascore13/{name}"
    os.makedirs(d, exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name, cpp_sources=_CPP, functions=["akbas_steer"],
        build_directory=d, verbose=False, extra_cflags=["-O2", "-std=c++17"])

print("C++ çekirdek derleniyor...")
try:
    _kernel = _build("akbascore13_f1")
except Exception as e:
    print(f"tekrar deneniyor ({e})..."); _kernel = _build("akbascore13_f2")
print("✅ Çekirdek hazır\n")

# =============================================================================
# SABİT AYARLAR — test edilip en dengeli bulunan değerler, DOKUNULMUYOR
# =============================================================================
MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
IVME       = 1.00
SONUM      = 0.30
ZIRVE      = 0.70
TABAN      = 0.10
ORAN       = 0.32          # motor sabiti, değiştirilemez
NUM_LAYERS = 20
MAX_NEW_TOKENS = 1000       # pratik üst sınır -- model kendi bitirirse daha kısa sürer
TAHMINI_TOKEN_HIZI = 1.6    # sn başına token, CPU ölçümlerinden -- SADECE süre tahmini için

print(f"Model yükleniyor: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    low_cpu_mem_usage=True)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
device = next(model.parameters()).device
layers = model.model.layers
print(f"✅ Model hazır ({device})\n")

# =============================================================================
# PUSULA — sadeleştirilmiş Terazi (eşit ağırlık; test edildi, özel ağırlıkların
# hiçbir ölçülebilir katkısı olmadığı kanıtlandı, o yüzden burada basitleştirildi)
# =============================================================================
def _word_vec(word):
    W = model.model.embed_tokens.weight
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tokenizer.unk_token_id or 0]
    return torch.stack([W[i].float().cpu().detach() for i in ids]).mean(0)

ETIK_KATEGORILERI = {
    "zarar_vermeme": ["safe", "harmless", "protective", "secure", "careful"],
    "durustluk":     ["honest", "accurate", "truthful", "transparent", "precise"],
    "ozerklik":      ["autonomous", "respectful", "unbiased", "free", "neutral"],
    "adalet":        ["fair", "just", "equitable", "balanced", "impartial"],
}
MANTIK_CAPALARI = ["logical", "empirical", "systematic", "structured", "verifiable",
                   "analyze", "precise", "deterministic", "sequential", "causal",
                   "rigorous", "impossible", "contradiction", "identify", "optimize"]

print("Pusula inşa ediliyor...")
_kategori_vektorleri = [F.normalize(torch.stack(
    [_word_vec(w) for w in kelimeler]).mean(0), dim=0)
    for kelimeler in ETIK_KATEGORILERI.values()]
_etik_vektor = F.normalize(torch.stack(_kategori_vektorleri).mean(0), dim=0)  # EŞİT ağırlık
_mantik_vektor = F.normalize(torch.stack(
    [_word_vec(w) for w in MANTIK_CAPALARI]).mean(0), dim=0)
PUSULA = F.normalize(0.40 * _etik_vektor + 0.60 * _mantik_vektor, dim=0).to(device)
print("✅ Pusula hazır\n")

# =============================================================================
# HOOK — DÜZELTİLMİŞ paylaşılan pc (tüm katmanlar TEK tensörü paylaşır,
# arşivde bulduğumuz "bilişsel amnezi" hatası burada giderilmiştir)
# =============================================================================
def _kur_hook(log_buf):
    pc_paylasilan = {"pc": torch.zeros(1, dtype=torch.float32, device=device)}

    def yap(layer_idx):
        def hook(mod, girdi, cikti):
            hidden = cikti[0] if isinstance(cikti, tuple) else cikti
            od = hidden.dtype
            hs = hidden.float() if od != torch.float32 else hidden
            if not hs.is_contiguous():
                hs = hs.contiguous()
            B, S, D = hs.shape
            bs = B * S
            if pc_paylasilan["pc"].shape[0] != bs:
                pc_paylasilan["pc"] = torch.zeros(bs, dtype=torch.float32, device=hs.device)
            st = _kernel.akbas_steer(
                hs, PUSULA, float(IVME), layer_idx, float(SONUM), float(ZIRVE),
                float(TABAN), pc_paylasilan["pc"], log_buf)
            if od != torch.float32:
                st = st.to(od)
            return (st,) + cikti[1:] if isinstance(cikti, tuple) else st
        return hook

    return [layers[i].register_forward_hook(yap(i)) for i in range(NUM_LAYERS)]


def akbascore_calistir(sicaklik, soru):
    prompt_metni = (f"<|im_start|>system\nYou are a helpful, accurate, and thoughtful "
                     f"assistant.<|im_end|>\n<|im_start|>user\n{soru.strip()}<|im_end|>\n"
                     f"<|im_start|>assistant\n")
    girdi = tokenizer(prompt_metni, return_tensors="pt").to(device)

    log_buf = torch.zeros(NUM_LAYERS, 4, dtype=torch.float32, device=device)
    hooks = _kur_hook(log_buf)

    sicaklik = float(sicaklik)
    do_sample = sicaklik > 0.0
    uret_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id,
                       repetition_penalty=1.0, do_sample=do_sample)
    if do_sample:
        uret_kwargs.update(temperature=sicaklik, top_p=0.90, top_k=50)

    t0 = time.time()
    try:
        with torch.no_grad():
            cikti = model.generate(**girdi, **uret_kwargs)
    finally:
        for h in hooks:
            h.remove()
    sure_sn = time.time() - t0

    n_girdi = girdi["input_ids"].shape[1]
    cevap = tokenizer.decode(cikti[0][n_girdi:], skip_special_tokens=True).strip()

    # --- Telemetri (SADECE kopyalama kutusuna gidecek, ana ekrana değil) ---
    katman_sayisi_gecerli = int((log_buf[:, 1] != 0).sum().item())
    if katman_sayisi_gecerli == 0:
        katman_sayisi_gecerli = 1
    ort_cs = (log_buf[:, 0].sum() / katman_sayisi_gecerli).item()
    ort_kb = (log_buf[:, 1].sum() / katman_sayisi_gecerli).item()
    ort_kv = (log_buf[:, 2].sum() / katman_sayisi_gecerli).item()
    ort_katki = (log_buf[:, 3].sum() / katman_sayisi_gecerli).item()
    kelime_sayisi = len(cevap.split())
    uretilen_token = cikti.shape[1] - n_girdi

    kopyalama_metni = f"""=== AkbasCore 1.3 DRA Motoru — Kayıt (Akbaş, 2026) ===

[SORU]
{soru.strip()}

[SICAKLIK (örnekleme rastgeleliği, 0=deterministik/en tutarlı, yüksek=daha yaratıcı/rastgele)]
{sicaklik}

[CEVAP]
{cevap}

--- SAYISAL TELEMETRİ (Türkçe isimlerle ve açıklamalarla) ---
Bu bölüm, bir kişi ya da bir yapay zeka modeli tarafından okunduğunda,
motorun bu üretim sırasında NASIL çalıştığını ek açıklama gerekmeden
anlayabilmesi için hazırlanmıştır.

• Süre (saniye): {sure_sn:.1f}
  -> Bu cevabın üretilmesi gerçekte ne kadar sürdü.

• Üretilen token sayısı: {uretilen_token}
• Kelime sayısı: {kelime_sayisi}

• İvme (steering velocity / itki gücü): {IVME}
  -> Motorun Pusula yönünde ne kadar güçlü ittiğini belirleyen ana kadran.
     Test edilen değerler arasında en dengeli bulunan sabit değer.

• Sönüm (decay rate): {SONUM}
  -> İtkinin, modelin katmanları derinleştikçe ne kadar hızlı söndüğünü
     belirler. Yüksek sönüm = itki erken katmanlarda hızla azalır.

• Zirve (amplitude): {ZIRVE}
  -> İtkinin en erken (birinci) katmandaki başlangıç genliği/gücü.

• Taban (permanent floor): {TABAN}
  -> İtkinin, en derin katmanlarda bile SIFIRA inmeyip kalıcı olarak
     koruduğu minimum seviye. Motorun hiçbir zaman tamamen "susmamasını"
     sağlayan tasarım kararı.

• Ortalama cos(θ) (kosinüs benzerliği): {ort_cs:+.5f}
  -> Modelin, üretim sırasında hidden state'inin (gizli durumunun)
     Pusula yönüyle ORTALAMA ne kadar hizalı olduğunu gösterir. Pozitif
     ve küçük bir sayı, modelin zaten hafifçe o yöne eğilimli olduğunu
     ve motorun bunu ZORLAMAK yerine hafifçe GÜÇLENDİRDİĞİNİ gösterir
     (projenin temel felsefesi: "nehri zorlamadan hafifçe yönlendirmek").

• Ortalama kb (katman kazancı, sönüm eğrisinin ham değeri): {ort_kb:.5f}
• Ortalama kv (hız-düzeltmeli kazanç, Fren mekanizmasından sonraki hali): {ort_kv:.5f}
  -> Bu ikisi motorun iç "gaz pedalı" değerleridir; kb ham hesaplanan
     değer, kv ise ardışık katmanlar arasındaki değişime göre hafifçe
     frenlenmiş/hızlandırılmış son halidir.

• Ortalama katkı (contribution, hidden state'e fiilen eklenen miktar): {ort_katki:+.6f}
  -> Bu, motorun HER katmanda hidden state'e gerçekten ne kadarlık bir
     sayısal ekleme yaptığının ortalamasıdır -- yani "dokunuşun" gerçek
     büyüklüğü. Küçük olması BEKLENEN ve İSTENEN bir durumdur (nazik
     dokunuş felsefesi).

• Etkin katman sayısı: {katman_sayisi_gecerli}/{NUM_LAYERS}
  -> Motor, modelin toplam katmanlarının ilk {NUM_LAYERS} tanesinde aktiftir,
     geri kalan katmanlara hiç dokunulmaz.

=== Kayıt sonu ==="""

    return cevap, kopyalama_metni, sure_sn


def _kopyala_dugmesi_goster(metin, dugme_id):
    """Colab'ın kendi IPython.display.HTML aracıyla, tarayıcının panosuna
    (navigator.clipboard) tek tıkla kopyalayan bir düğme gösterir."""
    js_guvenli_metin = json.dumps(metin)
    html = f"""
    <div id="{dugme_id}_kapsayici" style="margin:8px 0;">
      <button onclick='
          navigator.clipboard.writeText({js_guvenli_metin});
          this.innerText="✅ Kopyalandı!";
          setTimeout(()=>{{ this.innerText="📋 Soru+Cevap+Sayısal Kaydı Kopyala"; }}, 2000);
      ' style="padding:10px 16px; font-size:15px; cursor:pointer; width:100%;
               background:#1a3050; color:#8fc7ff; border:1px solid #2a4a70;
               border-radius:4px;">
        📋 Soru+Cevap+Sayısal Kaydı Kopyala
      </button>
    </div>
    """
    display(HTML(html))


# =============================================================================
# ARAYÜZ — ipywidgets (Colab'a gömülü, Gradio'ya göre çok daha kararlı).
# Her etiket kendi kutucuğunun ÜSTÜNDE, ayrı satırda -- mobilde yana kaymaz.
# =============================================================================
import ipywidgets as widgets

genislik_tam = widgets.Layout(width="100%")

etiket_sicaklik = widgets.HTML("<b>Sıcaklık</b> (0 = tutarlı/deterministik, "
                                "0.6-1.0 = dengeli, yüksek = daha yaratıcı)")
kutu_sicaklik = widgets.BoundedFloatText(
    value=0.65, min=0.0, max=2.0, step=0.05, layout=genislik_tam)

etiket_soru = widgets.HTML("<b>Sorunuz</b>")
kutu_soru = widgets.Textarea(
    placeholder="Sorunuzu buraya yazın...",
    layout=widgets.Layout(width="100%", height="110px"))

dugme_gonder = widgets.Button(
    description="▶ Gönder", button_style="primary", layout=genislik_tam)

alan_durum = widgets.Output()

etiket_cevap = widgets.HTML("<b>Cevap</b>")
kutu_cevap = widgets.Textarea(
    value="", disabled=True, layout=widgets.Layout(width="100%", height="320px"))

etiket_kopyala = widgets.HTML(
    "<b>Kopyalamak İçin</b> — soru + cevap + Türkçe açıklamalı sayısal kayıt "
    "(tek seferde kopyalayıp paylaşabilir ya da bir yapay zekaya inceletebilirsiniz)")
kutu_kopyala = widgets.Textarea(
    value="", disabled=True, layout=widgets.Layout(width="100%", height="320px"))

alan_kopyala_dugme = widgets.Output()

_sayac = {"n": 0}

def _gonder_tiklandi(b):
    soru = kutu_soru.value.strip()
    if not soru:
        with alan_durum:
            alan_durum.clear_output()
            print("Lütfen önce bir soru yazın.")
        return

    dugme_gonder.disabled = True
    tahmini_sn = MAX_NEW_TOKENS / TAHMINI_TOKEN_HIZI
    with alan_durum:
        alan_durum.clear_output()
        print(f"⏳ Model çalışıyor... En fazla ~{tahmini_sn/60:.1f} dakika sürebilir "
              f"(model erken bitirirse daha kısa sürer).")

    try:
        cevap, kopyalama_metni, sure_sn = akbascore_calistir(kutu_sicaklik.value, soru)
    except Exception as e:
        with alan_durum:
            alan_durum.clear_output()
            print(f"❌ Hata: {e}")
        dugme_gonder.disabled = False
        return

    kutu_cevap.value = cevap
    kutu_kopyala.value = kopyalama_metni

    with alan_durum:
        alan_durum.clear_output()
        print(f"✅ Tamamlandı ({sure_sn:.1f} sn)")

    _sayac["n"] += 1
    with alan_kopyala_dugme:
        alan_kopyala_dugme.clear_output()
        _kopyala_dugmesi_goster(kopyalama_metni, f"akbascore_kayit_{_sayac['n']}")

    dugme_gonder.disabled = False

dugme_gonder.on_click(_gonder_tiklandi)

display(widgets.VBox([
    widgets.HTML("<h3>🔱 AkbasCore 1.3 — DRA Motoru</h3>"
                 "<p style='color:#888'>(Akbaş, 2026 DRA)</p>"),
    etiket_sicaklik, kutu_sicaklik,
    etiket_soru, kutu_soru,
    dugme_gonder,
    alan_durum,
    etiket_cevap, kutu_cevap,
    etiket_kopyala, kutu_kopyala,
    alan_kopyala_dugme,
]))

print("✅ AkbasCore 1.3 hazır. Yukarıdaki kutulara sıcaklık ve sorunuzu girin.")

# =============================================================================
# (Akbaş) 2026 DRA — AkbasCore 1.3
# =============================================================================
