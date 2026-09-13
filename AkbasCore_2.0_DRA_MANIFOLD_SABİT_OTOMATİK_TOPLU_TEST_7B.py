# =============================================================================
# 🔱⬜ AKBASCORE 2.0 — DRA/MANIFOLD (SABİT/OTOMATİK TOPLU TEST — 7B)
# Hiçbir ayar yapmadan, tek hücreyi çalıştırıp 10 sabit soru için
# vanilla + steered çıktıları ve tüm logları otomatik üreten sürüm
# A zero-configuration batch script: run the single cell, get vanilla +
# steered outputs and full logs for 10 fixed questions, automatically
# =============================================================================
#
# BU DOSYADA HİÇBİR ŞEY AYARLAMANIZ GEREKMİYOR / NOTHING TO CONFIGURE HERE:
#   • Model sabit: Qwen/Qwen2.5-7B-Instruct — "gated" DEĞİL (token/lisans
#     kabulü gerekmiyor), standart, üzerinde özel bir fine-tune/RLHF varyantı
#     olmayan, doğrudan Qwen ekibinin resmi instruct modeli.
#   • 4-bit yükleme sabit AÇIK — T4'ün ~15GB VRAM'i 7B modeli bf16/fp16'da
#     zorlayabileceği için, hiç sormadan otomatik devrede.
#   • Motor kolları sabit: ivme=0.50 sönüm=0.30 zirve=0.70 taban=0.20
#   • Yönlendirme fiziği sabit: 1.00 (tam çekici/attractor)
#   • Sıcaklık sabit: 0.00 (deterministik/greedy — steering etkisini
#     örnekleme gürültüsünden ayırmak için, projenin kendi test
#     metodolojisi gereği)
#   • İki sabit eksen hazır geliyor:
#       Eksen 1 — Çekici (Işık): logical, coherent, clear, consistent,
#                 precise, grounded, truthful, structured
#       Eksen 2 — İtici (Karanlık): contradiction, chaos, nonsense, vague,
#                 hallucination, incoherent, illogical, absurd
#   • 10 sabit soru koda gömülü (aşağıda QUESTIONS listesinde) — çeşitli
#     kategorilerden (paradoks, tasarım, yaratıcılık, matematik, ikna,
#     empati, açıklama, tercih, etik, felsefe), hepsi zararsız/nötr.
#
#   Bu dosyayı Colab'da tek hücreye yapıştırıp çalıştırdığınızda: model
#   otomatik yüklenir, 10 soru sırayla vanilla ve steered modda çalıştırılır,
#   her biri için tam log tablosuyla birlikte çıktı hem ekrana basılır hem
#   de /content/akbascore_test_sonuclari.txt dosyasına yazılır ve (Colab
#   ortamındaysanız) otomatik indirilir. Hiçbir düğmeye basmanız, hiçbir
#   kutuya bir şey yazmanız gerekmiyor.
#   → Paste this into one Colab cell and run it: the model loads
#     automatically, all 10 questions run in both Vanilla and Steered mode
#     in sequence, each with its full log table, printed to the console AND
#     written to /content/akbascore_test_sonuclari.txt, auto-downloaded if
#     running in Colab. No buttons, no fields to fill in.
#
# COLAB'DA ÇALIŞTIRMA / HOW TO RUN ON COLAB:
#   1. Runtime → Change runtime type → T4 GPU seçin, kaydedin.
#   2. Bu dosyanın TAMAMINI kopyalayıp TEK bir hücreye yapıştırın, çalıştırın.
#   3. Bekleyin — model indirme + 10 soru x 2 pass (vanilla+steered) zaman
#      alır. Kabaca tahmin: mevcut CPU-güvenlik tasarımı nedeniyle (bkz.
#      dosyanın orijinalindeki "bilinen sınırlamalar") T4'te bu, soru başına
#      birkaç dakika sürebilir — 10 soru toplamda 30-60+ dakikaya çıkabilir.
#      Daha hızlı bir ilk deneme isterseniz aşağıdaki MAX_TOKENS sabitini
#      (şu an 400) düşürebilirsiniz — tek değiştirmeniz gereken satır o.
#   4. Bitince ekrandaki tüm çıktıyı görürsünüz, ayrıca bir .txt dosyası
#      otomatik iner.
#   → 1. Runtime → Change runtime type → T4 GPU, save.
#     2. Paste this ENTIRE file into ONE cell and run it.
#     3. Wait — downloading the model plus 10 questions x 2 passes takes
#        time; due to the current CPU-safety design this may take several
#        minutes per question on a T4, so 10 questions could take 30-60+
#        minutes total. For a faster first test, lower MAX_TOKENS (currently
#        400) below — that is the only line you would need to change.
#     4. When done, all output is on screen and a .txt file auto-downloads.
#
# DÜRÜSTLÜK NOTU / HONESTY NOTE: Bu dosya, GPU'suz/internetsiz bir sandbox'ta
# hazırlandı — sadece Python söz dizimi doğrulanmıştır, hiç çalıştırılmamıştır.
# İlk gerçek çalıştırma sizin Colab oturumunuzda olacak. Hata alırsanız tam
# hata mesajını (traceback'in tamamını) paylaşın, birlikte düzeltelim.
# → This file was prepared in a sandbox without GPU/internet — only Python
#   syntax was validated, it has never been executed. The first real run
#   will be in your own Colab session. If you hit an error, share the full
#   traceback so we can fix it together.
#
# AkbasCore, bir transformer dil modelinin gizli durumunu (hidden state) her
# katmanda matematiksel olarak yönlendiren bir C++ çıkarım motorudur. Yeniden
# eğitim yapmaz, ağırlıklara dokunmaz, prompt'u değiştirmez — müdahale
# doğrudan aktivasyon uzayında gerçekleşir. Çekirdek (matematik, C++ kernel)
# AkbasCore 2.0 — DRA/MANIFOLD ile birebir aynı, hiç değiştirilmedi.
# AkbasCore is a C++ inference-time engine that mathematically steers a
# transformer's hidden state at every layer, without retraining, without
# touching weights, without modifying the prompt. The kernel (math, C++
# core) is identical to AkbasCore 2.0 — DRA/MANIFOLD, unchanged.
# =============================================================================

import subprocess, sys, os, shutil, time, gc, math

def _pip(pkg, no_deps=False):
    cmd = [sys.executable,"-m","pip","install",pkg,"-q"]
    if no_deps:
        cmd.append("--no-deps")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── ÖNEMLİ / IMPORTANT ──────────────────────────────────────────────────────
# Colab'ın önceden kurulu, CUDA'lı PyTorch'u, bitsandbytes/accelerate gibi
# paketler kurulurken pip'in bağımlılık çözümlemesi torch'u "yetersiz" bulup
# SESSİZCE CPU'lu, CUDA'sız bir sürümle DEĞİŞTİREBİLİR. Bu, Colab'da bilinen,
# sinsi bir sorun: GPU fiziksel olarak bağlı ve "T4" seçili olsa bile,
# torch.cuda.is_available() aniden False döner. Bunu önlemek için bu
# paketleri --no-deps ile kuruyoruz — torch'a HİÇ dokunmasınlar diye.
# Gradio bu betikte hiç kullanılmadığı (arayüz yok, tamamen otomatik toplu
# çalıştırma) için kurulum listesinden tamamen çıkarıldı — gereksiz, büyük
# bağımlılık ağacı, ekstra risk.
# Colab's preinstalled, CUDA-enabled PyTorch can get SILENTLY REPLACED by a
# CPU-only build when pip resolves dependencies for packages like
# bitsandbytes/accelerate. This is a known, sneaky Colab issue: even with a
# T4 physically attached and selected, torch.cuda.is_available() can
# suddenly return False. --no-deps prevents these installs from touching
# torch at all. Gradio was removed entirely since this batch script has no
# UI and never imports it -- one less large dependency tree to risk.
for _pkg,_imp in [("ninja","ninja"),
                  ("bitsandbytes>=0.46.1","bitsandbytes"),
                  ("accelerate","accelerate")]:
    try:
        m=__import__(_imp)
        if _imp=="bitsandbytes":
            if tuple(int(x) for x in m.__version__.split(".")[:3])<(0,46,1):
                raise ImportError
    except ImportError:
        print(f"installing {_pkg} (--no-deps)..."); _pip(_pkg, no_deps=True); print(f"✅ {_imp} ready")

import torch, torch.utils.cpp_extension, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Torch/CUDA sağlık kontrolü — kernel derlemeden, model indirmeden ÖNCE ───
# Torch/CUDA sanity check — BEFORE compiling the kernel or downloading the
# model. Distinguishes "no GPU attached" from "GPU attached but the pip
# installs above just clobbered torch's CUDA build" (the sneaky Colab bug
# described above), because the fix is different for each case.
print(f"torch sürümü: {torch.__version__}  |  torch CUDA derlemesi: {torch.version.cuda}")
_nvidia_smi_ok = False
try:
    _r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    _nvidia_smi_ok = (_r.returncode == 0)
except Exception:
    _nvidia_smi_ok = False

if not torch.cuda.is_available():
    print("\n" + "!"*70)
    if _nvidia_smi_ok:
        print("❌ GPU DONANIM OLARAK VAR (nvidia-smi onu görüyor) ama PyTorch")
        print("   onu göremiyor. Bu, Colab'ın hazır CUDA'lı PyTorch'unun, bu")
        print("   hücredeki pip kurulumları sırasında sessizce CPU'lu bir")
        print("   sürümle değiştirilmiş olması anlamına gelir — GPU seçiminiz")
        print("   veya runtime tipiniz YANLIŞ DEĞİL, torch kurulumu bozulmuş.")
        print("")
        print("   ÇÖZÜM (bu, 'Restart session' ile düzelmez, çünkü paketler VM")
        print("   üzerinde kalıcı kalır — TEMİZ BİR VM gerekiyor):")
        print("   1) Runtime → Disconnect and delete runtime")
        print("   2) Tekrar bağlanın (Runtime → T4 GPU tekrar seçili olacak)")
        print("   3) Bu hücreyi YENİDEN çalıştırın (bu sürümde --no-deps")
        print("      düzeltmesi olduğu için bu sefer torch'a dokunulmayacak)")
    else:
        print("❌ GPU DONANIM OLARAK DA BULUNAMADI (nvidia-smi başarısız).")
        print("   ÇÖZÜM:")
        print("   1) Runtime → Change runtime type → 'T4 GPU' seçin, Save")
        print("   2) Runtime → Restart session, hücreyi yeniden çalıştırın")
        print("   3) Hâlâ olmuyorsa günlük ücretsiz GPU kotanız dolmuş olabilir")
        print("      — birkaç saat sonra tekrar deneyin ya da Colab Pro kullanın")
    print("!"*70 + "\n")
    raise SystemExit("GPU kullanılamıyor — yukarıdaki adımları izleyip tekrar deneyin.")
else:
    print(f"✅ GPU tespit edildi: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)\n")


os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
torch.backends.cudnn.deterministic    = True
torch.backends.cudnn.benchmark        = False

# =============================================================================
# C++ ÇEKİRDEK — Native Multi-Axis Manifold Engine  (DEĞİŞMEDİ / UNCHANGED)
# =============================================================================
_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

torch::Tensor akbas_orthogonalize(torch::Tensor compass_matrix) {
    TORCH_CHECK(!compass_matrix.is_cuda(),
        "AkbasCore cekirdegi CUDA tensoruyle cagrildi. Bu kernel host-side "
        "(CPU) C++ donguleriyle yazildi -- CUDA bellegini dogrudan dereference "
        "etmek segfault'a yol acar. Cagiran taraf (_inject) tensoru .cpu()'ya "
        "tasimadan bu fonksiyona ulastirmis olmali.");
    auto cm = compass_matrix.clone().contiguous();
    const int K = cm.size(0), D = cm.size(1);
    float *p = cm.data_ptr<float>();

    for (int k = 0; k < K; ++k) {
        float *vk = p + k * D;
        for (int j = 0; j < k; ++j) {
            const float *vj = p + j * D;
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += vk[d] * vj[d];
            for (int d = 0; d < D; ++d) vk[d] -= dot * vj[d];
        }
        float sq = 0.0f;
        for (int d = 0; d < D; ++d) sq += vk[d] * vk[d];
        float norm = sqrtf(sq) + 1e-8f;
        for (int d = 0; d < D; ++d) vk[d] /= norm;
    }
    return cm;
}

static inline float layer_curve(int layer_idx, float center, float width) {
    if (width < 1e-6f) return 1.0f;
    float x = ((float)layer_idx - (center - width * 0.5f)) / width;
    return std::clamp(x, 0.0f, 1.0f);
}

static inline float axis_katki(
    const float *tok, const float *axis_vec, int D,
    float norm_hint, float alpha, int layer_idx,
    float sonum, float zirve, float taban, float attractor_mode,
    float &pcv, float &out_cs, float &out_kb, float &out_kv
) {
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) dot += tok[d] * axis_vec[d];
    float cs = std::clamp(dot / norm_hint, -1.0f, 1.0f);

    float unc = 1.0f - fabsf(cs);
    float dw  = sonum + unc * 0.20f;
    float t   = (float)layer_idx;
    float kb  = zirve * expf(-dw * t) * (1.0f + dw * t) + taban;

    float dr = std::clamp(cs - pcv, -0.15f, 0.15f);
    if (cs > 0.80f && dr < 0.0f) dr *= 0.30f;

    float kv = kb;
    if      (dr > 0) kv *= (1.0f - dr    * 0.30f);
    else if (dr < 0) kv *= (1.0f + (-dr) * 0.30f);
    kv = std::clamp(kv, 0.05f, 1.0f);
    pcv = cs;

    float mk = std::clamp(norm_hint * 0.045f, 0.04f, 0.20f);

    float son = 1.0f;
    if      (cs >  0.75f) son = (1.0f - cs) / 0.25f;
    else if (cs < -0.40f) son = 1.6f;
    float katki_rezonans = alpha * cs * kv * 0.32f * son;

    float ang_err = (1.0f - cs) * 0.5f;
    float katki_cekici = alpha * kv * 0.32f * ang_err * 2.0f;

    float m = std::clamp(attractor_mode, 0.0f, 1.0f);
    float katki = std::clamp(
        (1.0f - m) * katki_rezonans + m * katki_cekici, -mk, mk);

    out_cs = cs; out_kb = kb; out_kv = kv;
    return katki;
}

torch::Tensor akbas_observe_multi(
    torch::Tensor hidden, torch::Tensor compass_matrix, torch::Tensor alphas,
    torch::Tensor centers, torch::Tensor widths, int layer_idx,
    float sonum, float zirve, float taban, float attractor_mode,
    torch::Tensor prev_cosine, torch::Tensor log_buf, torch::Tensor log_count
) {
    TORCH_CHECK(!hidden.is_cuda() && !compass_matrix.is_cuda(),
        "AkbasCore akbas_observe_multi: CUDA tensoru tespit edildi.");
    auto h  = hidden.contiguous();
    auto cm = compass_matrix.contiguous();
    auto al = alphas.contiguous();
    auto ce = centers.contiguous();
    auto wi = widths.contiguous();
    auto pc = prev_cosine.contiguous();
    auto lb = log_buf.contiguous();
    auto lc = log_count.contiguous();

    const int B = h.size(0), S = h.size(1), D = h.size(2);
    const int K = cm.size(0);
    const int N_LAYERS = lb.size(1);
    if (layer_idx >= N_LAYERS) return h;

    float *hp  = h.data_ptr<float>();
    const float *cmp = cm.data_ptr<float>();
    const float *alp = al.data_ptr<float>();
    const float *cep = ce.data_ptr<float>();
    const float *wip = wi.data_ptr<float>();
    float *pcp = pc.data_ptr<float>();
    float *lbp = lb.data_ptr<float>();
    float *lcp = lc.data_ptr<float>();

    for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s) {
        float *tok = hp + (b * S * D) + (s * D);
        int    idx = b * S + s;
        float sq = 0.0f;
        for (int d = 0; d < D; ++d) sq += tok[d] * tok[d];
        float norm_hint = sqrtf(sq) + 1e-6f;

        for (int k = 0; k < K; ++k) {
            float curve = (k == 0) ? 1.0f : layer_curve(layer_idx, cep[k], wip[k]);
            float cs, kb, kv;
            float &pcv = pcp[k * (B * S) + idx];
            float katki = axis_katki(tok, cmp + k * D, D, norm_hint,
                                      alp[k], layer_idx, sonum, zirve, taban,
                                      attractor_mode, pcv, cs, kb, kv) * curve;
            if (b == 0 && S == 1) {
                float *slot = lbp + k * N_LAYERS * 4 + layer_idx * 4;
                slot[0] += cs; slot[1] += kb; slot[2] += kv; slot[3] += katki;
            }
        }
        if (b == 0 && S == 1) lcp[layer_idx] += 1.0f;
    }
    return h;
}

torch::Tensor akbas_steer_multi(
    torch::Tensor hidden, torch::Tensor compass_matrix, torch::Tensor alphas,
    torch::Tensor centers, torch::Tensor widths, int layer_idx,
    float sonum, float zirve, float taban, float attractor_mode,
    torch::Tensor prev_cosine, torch::Tensor log_buf, torch::Tensor log_count
) {
    TORCH_CHECK(!hidden.is_cuda() && !compass_matrix.is_cuda(),
        "AkbasCore akbas_steer_multi: CUDA tensoru tespit edildi.");
    auto h  = hidden.contiguous();
    auto cm = compass_matrix.contiguous();
    auto al = alphas.contiguous();
    auto ce = centers.contiguous();
    auto wi = widths.contiguous();
    auto pc = prev_cosine.contiguous();
    auto lb = log_buf.contiguous();
    auto lc = log_count.contiguous();

    const int B = h.size(0), S = h.size(1), D = h.size(2);
    const int K = cm.size(0);
    const int N_LAYERS = lb.size(1);
    if (layer_idx >= N_LAYERS) return h;

    float *hp  = h.data_ptr<float>();
    const float *cmp = cm.data_ptr<float>();
    const float *alp = al.data_ptr<float>();
    const float *cep = ce.data_ptr<float>();
    const float *wip = wi.data_ptr<float>();
    float *pcp = pc.data_ptr<float>();
    float *lbp = lb.data_ptr<float>();
    float *lcp = lc.data_ptr<float>();

    for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s) {
        float *tok = hp + (b * S * D) + (s * D);
        int    idx = b * S + s;

        for (int k = 0; k < K; ++k) {
            float sq = 0.0f;
            for (int d = 0; d < D; ++d) sq += tok[d] * tok[d];
            float norm_hint = sqrtf(sq) + 1e-6f;

            float curve = (k == 0) ? 1.0f : layer_curve(layer_idx, cep[k], wip[k]);
            float cs, kb, kv;
            float &pcv = pcp[k * (B * S) + idx];
            const float *axis_vec = cmp + k * D;
            float katki = axis_katki(tok, axis_vec, D, norm_hint,
                                      alp[k], layer_idx, sonum, zirve, taban,
                                      attractor_mode, pcv, cs, kb, kv) * curve;

            for (int d = 0; d < D; ++d) tok[d] += katki * axis_vec[d];

            if (b == 0 && S == 1) {
                float *slot = lbp + k * N_LAYERS * 4 + layer_idx * 4;
                slot[0] += cs; slot[1] += kb; slot[2] += kv; slot[3] += katki;
            }
        }
        if (b == 0 && S == 1) lcp[layer_idx] += 1.0f;
    }
    return h;
}
"""

if not torch.utils.cpp_extension.is_ninja_available():
    import site
    for _sp in site.getsitepackages():
        _b=os.path.normpath(os.path.join(_sp,"..","..","bin"))
        if os.path.isdir(_b): os.environ["PATH"]=_b+":"+os.environ.get("PATH","")

for _n in ["akbas_local_f1","akbas_local_f2"]:
    for _r in ["/tmp/akbas_local",os.path.expanduser("~/.cache/torch_extensions")]:
        _p=os.path.join(_r,_n)
        if os.path.exists(_p): shutil.rmtree(_p,ignore_errors=True)

def _build(name):
    d=f"/tmp/akbas_local/{name}"; os.makedirs(d,exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name, cpp_sources=_CPP,
        functions=["akbas_orthogonalize","akbas_observe_multi","akbas_steer_multi"],
        build_directory=d, verbose=False,
        extra_cflags=["-O2","-std=c++17"])

try:    _kernel=_build("akbas_local_f1")
except Exception as e:
    print(f"  retry ({e})..."); _kernel=_build("akbas_local_f2")
print("✅ AkbasCore 2.0 — Universal/Local çekirdeği derlendi")

# =============================================================================
# SABİT PARAMETRELER (model-bağımsız kısım aynı / model-independent, unchanged)
# =============================================================================
DEFAULT_TEMPERATURE = 0.65
TEMP_MIN, TEMP_MAX  = 0.0, 1.50
TOP_P, TOP_K, REP_PENALTY = 0.90, 50, 1.15
MAX_EXTRA_AXES = 3
DEFAULT_HOOK_RATIO = 0.70   # toplam katmanın yaklaşık %70'i hook'lanır (1.2/2.0 ile aynı oran, 20/28≈0.71)

CONSTITUTION = {
    "harm":     (0.9228, ["safe","harmless","protective","secure","careful"]),
    "honesty":  (0.9372, ["honest","accurate","truthful","transparent","precise"]),
    "autonomy": (0.8788, ["autonomous","respectful","unbiased","free","neutral"]),
    "fairness": (0.9196, ["fair","just","equitable","balanced","impartial"]),
}
BILISIM_ETIK, BILISIM_MANTIK = 0.40, 0.60
LOGIC_ANCHORS = [
    "logical","empirical","systematic","structured","verifiable",
    "analyze","precise","deterministic","sequential","causal",
    "rigorous","impossible","contradiction","identify","optimize",
]
DISCLAIMER_MARKERS = [
    "as an ai","as a language model","i apologize","i must clarify",
    "i'm unable to","i am unable to","i cannot be certain",
]

DEFAULT_IVME, DEFAULT_SONUM, DEFAULT_ZIRVE, DEFAULT_TABAN = 0.50, 0.30, 0.70, 0.20
IVME_MIN,  IVME_MAX  = 0.20, 0.80
SONUM_MIN, SONUM_MAX = 0.10, 0.60
ZIRVE_MIN, ZIRVE_MAX = 0.30, 1.00
TABAN_MIN, TABAN_MAX = 0.05, 0.40

AXIS_MODES = [
    "Zıt Kavram Farkı (Diff-of-Means)",
    "Token Örneklemesi (Seed'li)",
    "Kelime Kümesi Ortalaması (Word-Set Mean)",
]
AXIS_POLARITY = ["Çekici (Işık — attractor)", "İtici (Karanlık — repulsor)"]
DARK_WORDS_PLACEHOLDER = ("contradiction, chaos, nonsense, vague, hallucination, "
                          "incoherent, illogical, absurd")
LIGHT_WORDS_PLACEHOLDER = ("logical, coherent, clear, consistent, precise, "
                           "grounded, truthful, structured")
TOKEN_SAMPLE_METHODS = ["Rastgele Token Karışımı", "Yüksek Normlu Token Karışımı"]
DEFAULT_SEED, DEFAULT_SAMPLE_COUNT = 42, 40
DEFAULT_AXIS_ALPHA = 0.50
DEFAULT_CENTER_RATIO = 0.75   # toplam katmanın %75'inden itibaren devreye girme
DEFAULT_WIDTH_RATIO  = 0.30   # geçiş genişliği, toplam katmanın %30'u kadar
DEFAULT_ATTRACTOR = 1.0

# Bilinen model aileleri için katman/embedding erişim yolları
# Known model families' layer/embedding access paths — Llama, Qwen, Mistral,
# Gemma ve türevlerinin büyük çoğunluğu bu iki yoldan birini kullanır.
_LAYER_PATHS = ["model.layers", "transformer.h", "gpt_neox.layers"]
_EMBED_PATHS = ["model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"]

def _resolve_path(obj, path):
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj

def _find_layers(model):
    for path in _LAYER_PATHS:
        try: return _resolve_path(model, path)
        except AttributeError: continue
    raise RuntimeError(
        "Bu model mimarisi tanınmadı (decoder katman listesi bulunamadı). "
        "Llama / Qwen / Mistral / Gemma ailesi modeller destekleniyor.")

def _find_embed_tokens(model):
    for path in _EMBED_PATHS:
        try: return _resolve_path(model, path)
        except AttributeError: continue
    raise RuntimeError(
        "Bu model mimarisi tanınmadı (embedding tablosu bulunamadı). "
        "Llama / Qwen / Mistral / Gemma ailesi modeller destekleniyor.")

# =============================================================================
# AkbasCore 2.0 — Universal/Local
# =============================================================================
class AkbasCore:

    def __init__(self, model_id, hook_ratio=DEFAULT_HOOK_RATIO, use_4bit=False,
                 hf_token=None, progress_cb=None):
        def _p(msg):
            print(msg)
            if progress_cb: progress_cb(msg)

        _p(f"🔱⬜ AKBASCORE 2.0 — Universal/Local | {model_id}")
        self.model_id = model_id

        # ── Gated modeller (ör. Llama-3.1) için opsiyonel HF token ──────────
        # Optional HF token for gated models such as Llama-3.1-8B-Instruct,
        # which require accepting the license on huggingface.co AND being
        # logged in / providing a token, or loading fails with a 401/403.
        if hf_token and hf_token.strip():
            try:
                from huggingface_hub import login as _hf_login
                _hf_login(token=hf_token.strip(), add_to_git_credential=False)
                _p("   Hugging Face token ile giriş yapıldı.")
            except Exception as e:
                _p(f"   ⚠️ HF token girişi başarısız: {e} (gated model kullanmıyorsanız görmezden gelin)")

        # ── GPU hesaplama kapasitesine göre dtype seçimi ────────────────────
        # T4 (Turing, capability 7.5) bfloat16'yı düzgün desteklemiyor;
        # float16 kullanmak gerekiyor. Ampere ve sonrası (A100, RTX 30xx/40xx,
        # capability >= 8.0) bfloat16'da sorunsuz ve genelde daha kararlı.
        # T4 (Turing, capability 7.5) does not properly support bfloat16;
        # float16 is required there. Ampere and newer (capability >= 8.0)
        # handles bfloat16 fine and is usually more numerically stable.
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(0)
            compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            compute_dtype = torch.float32
        self.compute_dtype = compute_dtype

        quant_cfg = None
        if use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            _p(f"   Düşük VRAM modu: 4-bit (nf4), compute dtype={compute_dtype}")

        if torch.cuda.is_available():
            vram=torch.cuda.get_device_properties(0).total_memory/1e9
            gpu_name = torch.cuda.get_device_name(0)
            _p(f"   GPU: {gpu_name} {vram:.1f}GB  |  dtype={compute_dtype}")
            if "T4" in gpu_name and not use_4bit:
                _p("   ⚠️ T4 tespit edildi: 8B ve üzeri modeller için "
                   "'Düşük VRAM (4-bit)' kutusunu işaretlemeniz önerilir "
                   "(T4'ün ~15GB kullanılabilir VRAM'i, bf16/fp16'da 8B "
                   "modeli zorlar).")
        else:
            _p("   Device: CPU")

        self.tok=AutoTokenizer.from_pretrained(model_id, token=hf_token or None)
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token

        _p("   Model indiriliyor/yükleniyor (ilk seferde uzun sürebilir)...")
        _from_pretrained_kwargs = dict(
            device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, token=(hf_token or None))
        if quant_cfg is not None:
            _from_pretrained_kwargs["quantization_config"] = quant_cfg
        else:
            _from_pretrained_kwargs["dtype"] = compute_dtype
        self.model=AutoModelForCausalLM.from_pretrained(
            model_id, **_from_pretrained_kwargs)
        if hasattr(self.model.config,"_attn_implementation"):
            self.model.config._attn_implementation="eager"
        self.device=next(self.model.parameters()).device
        self.hidden_dim=self.model.config.hidden_size

        # ── model mimarisinden bağımsız katman/embedding erişimi ────────────
        self._layers = _find_layers(self.model)
        self._embed_tokens = _find_embed_tokens(self.model)
        self.total_layers = len(self._layers)
        self.n_layers = max(1, min(self.total_layers,
                                    round(self.total_layers * hook_ratio)))
        _p(f"   Toplam katman: {self.total_layers}  |  "
           f"Hook'lanacak katman: {self.n_layers}  |  "
           f"Gizli boyut: {self.hidden_dim}")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        _p("   Anayasa ekseni (axis 0) inşa ediliyor...")
        self._anayasa=self._build_anayasa_pusula()
        self._token_norms=None

        gc.collect()
        _p(f"✅ AkbasCore hazır — {model_id}\n")

    # ── embedding yardımcıları ────────────────────────────────────────────────
    def _word_vec(self,word):
        W=self._embed_tokens.weight
        ids=self.tok(word,add_special_tokens=False)["input_ids"]
        if not ids: ids=[self.tok.unk_token_id or 0]
        return torch.stack([W[i].float().cpu().detach() for i in ids]).mean(0)

    def _mean_vec(self,words):
        words=[w for w in words if w]
        if not words: return None
        return torch.stack([self._word_vec(w) for w in words]).mean(0)

    def _build_anayasa_pusula(self):
        wsum=None; wtot=0.0
        for _,(w,words) in CONSTITUTION.items():
            v=torch.stack([self._word_vec(wd) for wd in words]).mean(0)
            wsum=w*v if wsum is None else wsum+w*v; wtot+=w
        logic=torch.stack([self._word_vec(w) for w in LOGIC_ANCHORS]).mean(0)
        return F.normalize(
            BILISIM_ETIK*(wsum/wtot) + BILISIM_MANTIK*logic, dim=0).contiguous()

    def _build_diff_axis(self, pos_text, neg_text):
        pos_words=[w.strip() for w in pos_text.replace("\n",",").split(",")]
        neg_words=[w.strip() for w in neg_text.replace("\n",",").split(",")]
        mp=self._mean_vec(pos_words); mn=self._mean_vec(neg_words)
        if mp is None or mn is None: return None
        return F.normalize(mp-mn, dim=0).contiguous()

    def _build_sample_axis(self, method, seed, sample_count):
        W=self._embed_tokens.weight
        vocab=W.shape[0]
        sample_count=max(2,min(int(sample_count),vocab))
        g=torch.Generator(device='cpu').manual_seed(int(seed))
        if method==TOKEN_SAMPLE_METHODS[1]:
            if self._token_norms is None:
                self._token_norms=W.float().norm(dim=1).cpu()
            pool_size=min(vocab, sample_count*5)
            _,pool_idx=torch.topk(self._token_norms, pool_size)
            perm=torch.randperm(pool_idx.shape[0], generator=g)[:sample_count]
            ids=pool_idx[perm]
        else:
            ids=torch.randint(0, vocab, (sample_count,), generator=g)
        vecs=torch.stack([W[i].float().cpu().detach() for i in ids])
        return F.normalize(vecs.mean(0), dim=0).contiguous()

    def _build_wordset_axis(self, words_text):
        words=[w.strip() for w in words_text.replace("\n",",").split(",")]
        v=self._mean_vec(words)
        if v is None: return None
        return F.normalize(v, dim=0).contiguous()

    def _build_axis_candidate(self, mode, pos_text, neg_text,
                               sample_method, seed, sample_count, words_text):
        if mode==AXIS_MODES[0]:
            v=self._build_diff_axis(pos_text, neg_text)
        elif mode==AXIS_MODES[1]:
            v=self._build_sample_axis(sample_method, seed, sample_count)
        else:
            v=self._build_wordset_axis(words_text)
        return v

    def _remove_hooks(self,hooks):
        for h in hooks: h.remove()

    def _inject(self, compass_matrix, alphas, centers, widths,
                log_buf, log_count, pc_holder, sonum, zirve, taban, attractor_mode, mode="observe"):
        hooks=[]
        fn = _kernel.akbas_observe_multi if mode=="observe" else _kernel.akbas_steer_multi
        K = compass_matrix.shape[0]
        compass_cpu = compass_matrix.detach().cpu().contiguous()
        alphas_cpu  = alphas.detach().cpu().contiguous()
        centers_cpu = centers.detach().cpu().contiguous()
        widths_cpu  = widths.detach().cpu().contiguous()
        def make_hook(i):
            def hook(mod,inp,out):
                hs=out[0] if isinstance(out,tuple) else out
                orig_device=hs.device
                od=hs.dtype
                hs_cpu = hs.detach().float().cpu().contiguous()
                B,S,D=hs_cpu.shape; bs=B*S
                if pc_holder["pc"] is None or pc_holder["pc"].shape[1]!=bs:
                    pc_holder["pc"]=torch.zeros(K,bs,dtype=torch.float32,device="cpu")
                st=fn(hs_cpu, compass_cpu, alphas_cpu, centers_cpu, widths_cpu,
                      i, sonum, zirve, taban, attractor_mode,
                      pc_holder["pc"], log_buf, log_count)
                st=st.to(device=orig_device, dtype=od)
                return (st,)+out[1:] if isinstance(out,tuple) else st
            return hook
        for i in range(min(self.n_layers,len(self._layers))):
            hooks.append(self._layers[i].register_forward_hook(make_hook(i)))
        return hooks

    def _run_pass(self, inp, max_tokens, mode, compass_matrix, alphas,
                  centers, widths, sonum, zirve, taban, attractor_mode, temperature):
        K = compass_matrix.shape[0]
        log_buf   = torch.zeros(K, self.n_layers, 4, dtype=torch.float32, device="cpu")
        log_count = torch.zeros(self.n_layers, dtype=torch.float32, device="cpu")
        pc_holder = {"pc": None}
        hooks = self._inject(compass_matrix, alphas, centers, widths,
                              log_buf, log_count, pc_holder, sonum, zirve, taban,
                              attractor_mode, mode=mode)
        eos_candidates = {self.tok.eos_token_id}
        for tok_str in ["<|im_end|>","<|endoftext|>","<|eot_id|>","</s>"]:
            tid = self.tok.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != self.tok.unk_token_id:
                eos_candidates.add(tid)
        eos = [t for t in eos_candidates if t is not None and t != -1]

        gen_kwargs = dict(
            max_new_tokens=int(max_tokens),
            repetition_penalty=REP_PENALTY,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=eos)
        if temperature <= 0.01:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=float(temperature),
                              top_p=TOP_P, top_k=TOP_K)
        t0=time.time()
        with torch.no_grad():
            out=self.model.generate(**inp, **gen_kwargs)
        ms=(time.time()-t0)*1000
        self._remove_hooks(hooks)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        n_in  = inp["input_ids"].shape[1]
        n_out = out.shape[1]-n_in
        tps   = n_out/(ms/1000) if ms>0 else 0
        text  = self.tok.decode(out[0][n_in:],skip_special_tokens=True).strip()
        lines = text.split("\n")
        fi    = next((i for i,l in enumerate(lines) if l.strip()),None)
        if fi is not None and any(m in lines[fi].lower() for m in DISCLAIMER_MARKERS):
            text="\n".join(l for l in lines[fi+1:] if l.strip())
        count = log_count.cpu()
        count_safe = count.clamp(min=1.0)
        log_avg = log_buf.cpu() / count_safe.view(1, self.n_layers, 1)
        return text, ms, tps, n_in, n_out, log_avg, count

    def _log_header(self, label, ms, tps, n_in, n_out, ivme, sonum, zirve, taban,
                     temperature, attractor_label="—", axis_infos=None):
        W=64; lines=[]
        lines.append("═"*W)
        lines.append(f"  {label}")
        lines.append("═"*W)
        lines.append(f"  ⏱  {ms:.0f} ms  |  {tps:.1f} tok/s  |  "
                     f"{n_in} in  |  {n_out} out tokens")
        lines.append(f"  MODEL  {self.model_id}  ({self.total_layers} katman, "
                     f"{self.n_layers} hook'lu, boyut={self.hidden_dim})")
        lines.append(f"  MOTOR  ivme(axis0)={ivme:.2f}  sönüm={sonum:.2f}  "
                     f"zirve={zirve:.2f}  taban={taban:.2f}")
        lines.append(f"  FİZİK  yönlendirme modu={attractor_label}")
        lines.append(f"  SABIT  oran=0.32  doyum=0.75  karşıt=-0.40  "
                     f"sapma=0.20  fren=0.30")
        if axis_infos:
            for info in axis_infos:
                lines.append(f"  EKSEN  {info}")
        lines.append(f"  ⚠️  ÖRNEKLEME UYARISI: temperature={temperature:.2f} aktif — "
                     f"vanilla/steered metin farkının bir kısmı steering'den değil,")
        lines.append(f"      rastgele örneklemeden gelebilir. Saf steering etkisini "
                     f"görmek için temperature'ı geçici olarak 0'a çekin.")
        return lines

    def _log_axis_table(self, lb_k, count, axis_label, layer_idx0_always_active):
        W=64; lines=[]
        lines.append("─"*W)
        n_tok = int(count.max().item()) if count.numel() else 0
        lines.append(f"  [{axis_label}]  (N={n_tok} üretilen token üzerinden ortalama)")
        lines.append(f"  {'L':>3}  {'cos(θ)ort':>10}  {'kb ort':>8}  {'kv ort':>8}  {'katki ort':>10}")
        lines.append("─"*W)
        cos_vals=[]; kat_vals=[]
        for i in range(self.n_layers):
            if count[i].item() <= 0: continue
            cs,kb,kv,kat = lb_k[i].tolist()
            cos_vals.append(cs); kat_vals.append(kat)
            lines.append(f"  {i:>3}  {cs:>+10.4f}  {kb:>8.5f}  {kv:>8.5f}  {kat:>+10.6f}")
        if cos_vals:
            drift = cos_vals[-1]-cos_vals[0]
            lines.append(f"  drift(cos) L0→L{self.n_layers-1}: {drift:+.4f}  |  "
                         f"katki ort. toplamı={sum(kat_vals):+.6f}")
        return lines

    def run(self, prompt, max_tokens, temperature, ivme, sonum, zirve, taban, attractor_mode,
            n_extra_axes,
            mode1, pos1, neg1, sm1, seed1, cnt1, words1, kutup1, alpha1, cen1, wid1,
            mode2, pos2, neg2, sm2, seed2, cnt2, words2, kutup2, alpha2, cen2, wid2,
            mode3, pos3, neg3, sm3, seed3, cnt3, words3, kutup3, alpha3, cen3, wid3):
        if not prompt.strip():
            return "", ""

        temperature = float(max(TEMP_MIN, min(TEMP_MAX, temperature)))
        ivme  = float(max(IVME_MIN,  min(IVME_MAX,  ivme)))
        sonum = float(max(SONUM_MIN, min(SONUM_MAX, sonum)))
        zirve = float(max(ZIRVE_MIN, min(ZIRVE_MAX, zirve)))
        taban = float(max(TABAN_MIN, min(TABAN_MAX, taban)))
        attractor_mode = float(max(0.0, min(1.0, attractor_mode)))
        n_extra_axes = int(max(0, min(MAX_EXTRA_AXES, n_extra_axes)))

        axis_defs = [
            (mode1,pos1,neg1,sm1,seed1,cnt1,words1,kutup1,alpha1,cen1,wid1),
            (mode2,pos2,neg2,sm2,seed2,cnt2,words2,kutup2,alpha2,cen2,wid2),
            (mode3,pos3,neg3,sm3,seed3,cnt3,words3,kutup3,alpha3,cen3,wid3),
        ][:n_extra_axes]

        raw_axes = [self._anayasa]
        alphas_l = [ivme]
        centers_l = [0.0]; widths_l = [0.0]
        axis_infos = []
        for (mode,pos,neg,sm,seed,cnt,words,kutup,alpha,cen,wid) in axis_defs:
            v = self._build_axis_candidate(mode,pos,neg,sm,seed,cnt,words)
            if v is None:
                axis_infos.append(f"[atlandı: girdi boş] mode={mode}")
                continue
            polarite_repulsor = (kutup==AXIS_POLARITY[1])
            if polarite_repulsor:
                v = -v
            raw_axes.append(v)
            alphas_l.append(float(alpha))
            # merkez/genişlik artık katman-oranı olarak da girilebilir (0-1
            # arası verilirse toplam katman sayısına göre ölçeklenir; 1'den
            # büyükse doğrudan katman indeksi olarak kullanılır — geriye dönük
            # uyumluluk için)
            cen_val = float(cen) if float(cen) > 1.0 else float(cen) * self.n_layers
            wid_val = float(wid) if float(wid) > 1.0 else max(1.0, float(wid) * self.n_layers)
            centers_l.append(cen_val)
            widths_l.append(wid_val)
            src = ("diff-of-means" if mode==AXIS_MODES[0] else
                   f"örnek/{sm}" if mode==AXIS_MODES[1] else
                   "kelime kümesi")
            kutup_etiket = "İTİCİ(karanlık)" if polarite_repulsor else "çekici(ışık)"
            axis_infos.append(f"axis{len(raw_axes)-1}: {src}  [{kutup_etiket}]  "
                              f"α={alpha:.2f}  merkez=L{cen_val:.0f}  genişlik={wid_val:.0f}")

        # raw_axes zaten CPU tensörleri (_word_vec/_mean_vec .cpu() döndürür).
        # akbas_orthogonalize SADECE CPU tensörüyle çalışır (TORCH_CHECK bunu
        # zorunlu kılıyor) — bu yüzden burada .to(self.device) YOK, kernel
        # çağrısından SONRA GPU'ya taşınıyor. Önceki hata tam burada, bu
        # satırın GPU'ya erken taşımasından kaynaklanıyordu.
        # raw_axes are already CPU tensors. akbas_orthogonalize is CPU-only
        # by design (enforced via TORCH_CHECK) -- so no .to(self.device) here;
        # the move to GPU happens AFTER the kernel call. The previous crash
        # was caused by this line moving to GPU too early.
        raw_matrix = torch.stack(raw_axes).float().cpu().contiguous()
        compass_matrix = _kernel.akbas_orthogonalize(raw_matrix).contiguous().to(self.device)
        alphas  = torch.tensor(alphas_l,  dtype=torch.float32, device=self.device)
        centers = torch.tensor(centers_l, dtype=torch.float32, device=self.device)
        widths  = torch.tensor(widths_l,  dtype=torch.float32, device=self.device)
        K = compass_matrix.shape[0]

        prompt_text = (f"<|im_start|>system\nYou are a helpful, accurate, and "
                       f"thoughtful assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
                       f"<|im_start|>assistant\n")
        inp = self.tok(prompt_text, return_tensors="pt").to(self.device)

        print(f"   Pass A: Vanilla Observer  [ivme={ivme} sönüm={sonum} "
              f"zirve={zirve} taban={taban}]")
        van_matrix = compass_matrix[0:1].contiguous()
        van_alphas = alphas[0:1].contiguous()
        van_centers = centers[0:1].contiguous()
        van_widths = widths[0:1].contiguous()
        txt_v, ms_v, tps_v, n_in_v, n_out_v, lb_v, cnt_v = self._run_pass(
            inp, max_tokens, "observe", van_matrix, van_alphas,
            van_centers, van_widths, sonum, zirve, taban, attractor_mode, temperature)
        gc.collect()

        mode_label = ("rezonans" if attractor_mode<0.01 else
                      "çekici/attractor" if attractor_mode>0.99 else
                      f"karışım (m={attractor_mode:.2f})")
        print(f"   Pass B: Steered  [K={K} eksen, mod={mode_label}]")
        txt_s, ms_s, tps_s, n_in_s, n_out_s, lb_s, cnt_s = self._run_pass(
            inp, max_tokens, "steer", compass_matrix, alphas,
            centers, widths, sonum, zirve, taban, attractor_mode, temperature)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        van_lines = self._log_header(
            "⬜ VANILLA OBSERVER — katki=0 (hidden state değişmedi)",
            ms_v, tps_v, n_in_v, n_out_v, ivme, sonum, zirve, taban,
            temperature, attractor_label="n/a (gözlem modu)")
        van_lines += self._log_axis_table(lb_v[0], cnt_v, "axis0: anayasa (referans)", True)
        van_lines.append("═"*64)
        van_out = txt_v + "\n\n" + "\n".join(van_lines)

        st_lines = self._log_header(
            "🔱 AKBASCORE STEERED — native multi-axis uygulandı",
            ms_s, tps_s, n_in_s, n_out_s, ivme, sonum, zirve, taban,
            temperature, attractor_label=f"{mode_label} (m={attractor_mode:.2f})",
            axis_infos=axis_infos)
        st_lines += self._log_axis_table(lb_s[0], cnt_s, "axis0: anayasa (permanent floor)", True)
        for k in range(1, K):
            label = axis_infos[k-1] if k-1 < len(axis_infos) else f"axis{k}"
            st_lines += self._log_axis_table(lb_s[k], cnt_s, label, False)
        dcos0 = lb_s[0,:,0] - lb_v[0,:,0]
        nz = [i for i in range(self.n_layers) if cnt_v[i].item()>0 and cnt_s[i].item()>0]
        if nz:
            lines_cmp=["─"*64,
                       "  AXIS0 (ANAYASA) DELTA — vanilla'ya göre kayma (ortalamalar arası)",
                       f"  (vanilla N={int(cnt_v.max().item())} token, "
                       f"steered N={int(cnt_s.max().item())} token — farklı olabilir)",
                       "─"*64]
            for i in nz:
                lines_cmp.append(f"  L{i:>2}  Δcos={dcos0[i].item():+.4f}")
            lines_cmp.append(f"  ortalama Δcos={dcos0[nz].mean().item():+.4f}")
            lines_cmp.append("═"*64)
            st_lines += lines_cmp
        st_out = txt_s + "\n\n" + "\n".join(st_lines)

        return van_out, st_out

# =============================================================================

# =============================================================================
# SABİT MODEL / SABİT AYARLAR — hiçbirini değiştirmeniz gerekmiyor
# NOTHING BELOW REQUIRES CHANGES — everything is pre-set
# =============================================================================
MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"   # standart, "gated" değil, token gerekmez
USE_4BIT   = True                          # T4 için güvenli, sabit varsayılan
HOOK_RATIO = DEFAULT_HOOK_RATIO             # toplam katmanın ~%70'i

TEMPERATURE = 0.0     # deterministik (greedy) — steering etkisini örnekleme
                      # gürültüsünden ayırmak için sabit sıfır
MAX_TOKENS  = 1000    # üretim tavanı — model genelde bunun altında, kendi
                      # EOS token'ında doğal olarak durur; bu sadece "en fazla
                      # bu kadar üret" tavanı, her cevap 1000 token olmayacak

QUESTIONS = [
    "Describe a situation where a scientific theory must be proven false in order to be considered true. Explain step by step how this is possible without logical contradiction.",
    "A city plans to build a park that both preserves an old forest and doubles the amount of usable public space. Propose a design, explaining the reasoning behind each trade-off.",
    "Write a short story (150-200 words) about a lighthouse keeper who has never seen the sea.",
    "A train leaves station A at 60 km/h heading toward station B, 300 km away. Another train leaves station B at 40 km/h heading toward station A at the same time. How far from station A will they meet, and after how long? Show your reasoning.",
    "Argue, as persuasively as possible, for why libraries will remain essential in a world of instant digital information.",
    "A close friend just failed an important exam they studied hard for. Write what you would say to comfort them.",
    "Explain quantum entanglement to a curious 12-year-old, using only everyday objects as analogies.",
    "If you could change one rule of chess, which would it be and why?",
    "A researcher discovers that publishing their result honestly will hurt their career, but hiding it will help humanity. What should they do, and why?",
    "What does it mean for a machine to understand something, rather than merely predict it?",
]

# =============================================================================
# MODELİ YÜKLE — otomatik, tek seferlik, hiçbir buton yok
# =============================================================================
print("\n" + "="*70)
print(f"🔱⬜ AKBASCORE 2.0 — SABİT TOPLU TEST | {MODEL_ID}")
print("="*70 + "\n")

core = AkbasCore(MODEL_ID, hook_ratio=HOOK_RATIO, use_4bit=USE_4BIT)

# =============================================================================
# 10 SORUYU SIRAYLA ÇALIŞTIR — hiçbir kullanıcı etkileşimi gerekmez
# =============================================================================
all_results = []
header = (
    f"AKBASCORE 2.0 — SABİT TOPLU TEST SONUÇLARI\n"
    f"Model: {MODEL_ID}  |  Toplam katman: {core.total_layers}  |  "
    f"Hook'lanan: {core.n_layers}  |  Gizli boyut: {core.hidden_dim}  |  "
    f"dtype: {core.compute_dtype}\n"
    f"Motor: ivme={DEFAULT_IVME} sönüm={DEFAULT_SONUM} zirve={DEFAULT_ZIRVE} "
    f"taban={DEFAULT_TABAN} | Yönlendirme fiziği={DEFAULT_ATTRACTOR} | "
    f"Sıcaklık={TEMPERATURE}\n"
    f"Eksen 1 (Çekici/Işık): {LIGHT_WORDS_PLACEHOLDER}\n"
    f"Eksen 2 (İtici/Karanlık): {DARK_WORDS_PLACEHOLDER}\n"
    + "="*70 + "\n"
)
print(header)
all_results.append(header)

for i, question in enumerate(QUESTIONS, 1):
    print(f"\n{'#'*70}\n  SORU {i}/{len(QUESTIONS)}  —  işleniyor...\n{'#'*70}")
    print(f"  {question}\n")

    van_out, st_out = core.run(
        question, MAX_TOKENS, TEMPERATURE,
        DEFAULT_IVME, DEFAULT_SONUM, DEFAULT_ZIRVE, DEFAULT_TABAN, DEFAULT_ATTRACTOR,
        2,  # n_extra_axes = 2 (ışık + karanlık, sabit)
        AXIS_MODES[2], "", "", TOKEN_SAMPLE_METHODS[0], DEFAULT_SEED, DEFAULT_SAMPLE_COUNT,
        LIGHT_WORDS_PLACEHOLDER, AXIS_POLARITY[0], DEFAULT_AXIS_ALPHA,
        DEFAULT_CENTER_RATIO, DEFAULT_WIDTH_RATIO,
        AXIS_MODES[2], "", "", TOKEN_SAMPLE_METHODS[0], DEFAULT_SEED, DEFAULT_SAMPLE_COUNT,
        DARK_WORDS_PLACEHOLDER, AXIS_POLARITY[1], DEFAULT_AXIS_ALPHA,
        DEFAULT_CENTER_RATIO, DEFAULT_WIDTH_RATIO,
        AXIS_MODES[0], "", "", TOKEN_SAMPLE_METHODS[0], DEFAULT_SEED, DEFAULT_SAMPLE_COUNT,
        "", AXIS_POLARITY[0], DEFAULT_AXIS_ALPHA, DEFAULT_CENTER_RATIO, DEFAULT_WIDTH_RATIO,
    )

    block = (
        f"\n{'#'*70}\n"
        f"### SORU {i}/{len(QUESTIONS)} ###\n{question}\n"
        f"{'#'*70}\n\n"
        f"### VANILLA OBSERVER — ÇIKTI + LOG ###\n\n{van_out}\n\n"
        f"{'#'*70}\n\n"
        f"### STEERED (ışık çekici + karanlık itici) — ÇIKTI + LOG ###\n\n{st_out}\n"
    )
    print(block)
    all_results.append(block)

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# =============================================================================
# TÜM SONUÇLARI DOSYAYA YAZ VE (Colab'daysanız) OTOMATİK İNDİR
# =============================================================================
final_text = "\n".join(all_results)
out_path = "/content/akbascore_test_sonuclari.txt"
try:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"\n✅ Tüm sonuçlar yazıldı: {out_path}")
except Exception as e:
    print(f"⚠️ Dosyaya yazılamadı ({e}) — çıktıyı doğrudan yukarıdan kopyalayabilirsiniz.")

try:
    from google.colab import files as _colab_files
    _colab_files.download(out_path)
    print("✅ Dosya otomatik indiriliyor (tarayıcınızın indirilenler klasörüne bakın).")
except Exception:
    print("ℹ️ Colab dışında çalışıyorsunuz veya otomatik indirme başarısız oldu — "
          f"dosyayı {out_path} yolundan elle indirebilirsiniz.")

print("\n" + "="*70)
print("🏁 TÜM TESTLER TAMAMLANDI — 10/10 soru işlendi.")
print("="*70)
