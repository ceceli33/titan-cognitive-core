# =============================================================================
# 🔱⬜ AKBASCORE 2.0 — DRA/MANIFOLD
# Qwen2.5-1.5B-Instruct üzerinde çalışan runtime activation-steering motoru
# A runtime activation-steering engine for Qwen2.5-1.5B-Instruct
# =============================================================================
#
# AkbasCore, bir transformer dil modelinin gizli durumunu (hidden state) her
# katmanda matematiksel olarak yönlendiren bir C++ çıkarım motorudur. Yeniden
# eğitim yapmaz, ağırlıklara dokunmaz, prompt'u değiştirmez — müdahale
# doğrudan aktivasyon uzayında gerçekleşir.
# AkbasCore is a C++ inference-time engine that mathematically steers a
# transformer's hidden state at every layer. It does not retrain, does not
# touch the weights, and does not modify the prompt — the intervention
# happens directly in activation space.
#
# ─────────────────────────────────────────────────────────────────────────
# SÜRÜM GEÇMİŞİ / VERSION HISTORY
# ─────────────────────────────────────────────────────────────────────────
#
# AkbasCore 1.2 (temel / foundation)
#   Tek pusula (compass vector), sabit anayasa (constitution) kelimelerinden
#   türetilir. Dual-pass yapı: Vanilla Observer (katki=0, sadece ölçüm) ve
#   Steered (katki uygulanır). Sönümlü rezonans formülü (Damped Resonance
#   Alignment / DRA): kb = zirve·e^(-sönüm·t)·(1+sönüm·t) + taban.
#   → Single compass vector derived from fixed constitution words. Dual-pass:
#     Vanilla Observer (contribution=0, measurement only) and Steered
#     (contribution applied). Damped Resonance Alignment (DRA) formula.
#
# AkbasCore Sector 0.1 (deneysel, TERK EDİLDİ / experimental, ABANDONED)
#   Embedding tablosuyla ilgisi olmayan yapay geometrik vektörler (Gauss
#   maskesi, rastgele birim vektör) denendi. Katman başına sınırsız çarpımsal
#   büyüme (warp_proj) çıktıyı bozdu — kaba-kuvvet steering'in bilinen
#   başarısızlık modu. Bu yaklaşım tamamen terk edildi.
#   → Synthetic geometric vectors unrelated to the embedding table were
#     tried; unbounded per-layer multiplicative growth corrupted output —
#     a known failure mode of brute-force steering. Fully abandoned.
#
# AkbasCore 2.0 — DRA/MANIFOLD (bu dosya / this file)
#   Sector 0.1'in başarısızlığından çıkan ders: pusula HER ZAMAN modelin
#   kendi embedding tablosundan (gerçek token vektörlerinden) türetilmeli.
#   Bunun üstüne inşa edilen native, çoklu-eksen mimari:
#     • Çoklu ortogonal eksen (native Gram-Schmidt, C++ içinde)
#     • Çekici/itici kutup fiziği (attractor/repulsor — ışık/karanlık alan)
#     • Katman-derinlik eğrisi (hangi eksen hangi katmanda devrede)
#     • Rezonans ↔ Çekici fizik karışımı (attractor_mode spektrumu)
#     • CUDA güvenlik koruması + CPU-fallback (Termux/mobil ve Colab uyumlu)
#     • Üretim boyunca biriktirilen (kümülatif ortalama) tanılama logu
#     • UI'dan ayarlanabilir sıcaklık (temperature)
#   → The lesson from Sector 0.1's failure: the compass must ALWAYS be
#     derived from the model's own embedding table (real token vectors).
#     Built on that foundation: native multi-axis orthogonalization,
#     attractor/repulsor polarity physics, per-axis layer-depth activation
#     curves, a continuous resonance↔attractor physics blend, CUDA safety
#     guards with automatic CPU fallback, cumulative-average diagnostic
#     logging across the full generation, and a UI-adjustable temperature.
#
# ─────────────────────────────────────────────────────────────────────────
# TERİMLER — 1.2'DEN AYNEN KORUNANLAR / TERMS CARRIED OVER FROM 1.2 UNCHANGED
# ─────────────────────────────────────────────────────────────────────────
#
#   pusula   — compass vector: steering direction in hidden-state space
#   katki    — contribution: scalar value written into hidden state per layer
#   taban    — permanent floor: minimum continuous alignment pressure (P∞)
#   sönüm    — decay rate: how fast the resonance peak fades across layers (ω)
#   zirve    — amplitude: peak push strength at early layers (A)
#   ivme     — steering velocity: master amplitude dial (v₀)
#   terazi   — constitution weights: ethical balance values per category
#   bileşim  — blend ratio: ethics/logic mixture for the anayasa pusula
#   oran     — scale factor: base multiplier in the katki formula [sabit]
#   doyum    — saturation threshold: reduces push when cos(θ) is very high [sabit]
#   karşıt   — counter threshold: amplifies push when cos(θ) is very negative [sabit]
#   sapma    — uncertainty corrector: alignment-deviation adjuster [sabit]
#   fren     — velocity brake: feedback damping on rate of change [sabit]
#   sınır    — max katki ratio: clamp ceiling relative to hidden-state norm [sabit]
#   anayasa  — constitution: the default safety/logic compass (now = axis 0)
#
# ─────────────────────────────────────────────────────────────────────────
# TERİMLER — 2.0'DA SIFIRDAN EKLENENLER / NEW TERMS INTRODUCED IN 2.0
# ─────────────────────────────────────────────────────────────────────────
#
#   eksen (axis)               — a single steering direction (a vector) with
#                                 its own strength and its own layer-depth curve
#   kutup (polarity)           — çekici (attractor) or itici (repulsor); which
#                                 way an axis pulls the hidden state
#   çekici (attractor)         — pulls hidden state TOWARD a target direction,
#                                 regardless of current alignment sign
#   itici (repulsor)           — pushes hidden state AWAY from a concept;
#                                 implemented as attraction to the negated vector
#   rezonans (resonance)       — the original 1.2 physics: amplifies whatever
#                                 alignment sign already exists (does not correct)
#   yönlendirme fiziği
#     (steering physics)       — the attractor_mode dial: a continuous 0→1
#                                 blend between rezonans and çekici physics
#   açısal hata (angular error)— (1-cosθ)/2 ∈ [0,1]; how far hidden state is
#                                 from a target axis, used by the çekici formula
#   merkez (center)            — the layer index where an axis's influence
#                                 starts ramping up
#   genişlik (width)           — how many layers the ramp-up transition spans
#   katman-derinlik eğrisi
#     (layer-depth curve)      — the piecewise-linear ramp function driven by
#                                 merkez/genişlik; lets an axis stay silent in
#                                 early (syntax) layers and activate in deep
#                                 (intent/tone) layers
#   dikleştirme / ortogonalizasyon
#     (orthogonalization)      — native Gram-Schmidt process making every
#                                 extra axis perpendicular to anayasa (axis 0)
#                                 and to each other; the mathematical source
#                                 of null-space safety (no axis can leak onto
#                                 the anayasa direction)
#   kelime kümesi ortalaması
#     (word-set mean)          — an axis built directly from the mean
#                                 embedding of a single concept word list
#                                 (e.g. contradiction/chaos/nonsense…)
#   zıt kavram farkı
#     (diff-of-means)          — an axis built from the difference between
#                                 two concept word-list means
#   token örneklemesi
#     (token sampling)         — an axis built from a seeded sample of real
#                                 vocabulary token embeddings
#   sıcaklık (temperature)     — sampling randomness; UI-adjustable, 0=greedy
#                                 (deterministic), used to separate genuine
#                                 steering effects from sampling noise
#
# ─────────────────────────────────────────────────────────────────────────
# ÇEKİRDEK — 3 NATIVE C++ FONKSİYONU / KERNEL — 3 NATIVE C++ FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────
#
#   akbas_orthogonalize(compass_matrix[K,D])
#     Native Gram-Schmidt. axis 0 = anayasa (sabit referans). axis 1..K-1
#     sırayla axis 0'a VE birbirlerine göre dikleştirilir — null-space
#     projeksiyonu ayrı bir adım değil, bunun doğal sonucudur.
#     → axis 0 = anayasa (fixed reference). axes 1..K-1 are orthogonalized
#       against axis 0 AND each other in sequence — null-space projection
#       falls out of this naturally rather than being a separate step.
#
#   akbas_observe_multi(...) / akbas_steer_multi(...)
#     Tek token-belleği geçişinde K ekseni sırayla işler: her eksenin kendi
#     fren tamponu, axis 0 her katmanda tam güçte, axis 1..K-1 kendi
#     katman-derinlik eğrisiyle devreye giriyor. Loglama artık üretimin
#     TAMAMI boyunca (prefill hariç) biriktirilip ortalanıyor.
#     → Processes K axes sequentially in a single pass over hidden-state
#       memory: each axis keeps its own brake buffer, axis 0 is always at
#       full strength, axes 1..K-1 ramp in per their own layer-depth curve.
#       Logging now accumulates (and is averaged) across the ENTIRE
#       generation, excluding prefill.
#
#   Cihaz güvenliği / Device safety: kernel host-side (CPU) C++ döngüleriyle
#   yazıldı, __global__ CUDA kernel'i DEĞİL. TORCH_CHECK korumaları CUDA
#   tensörü tespit ederse anlaşılır bir hata fırlatır; Python tarafı
#   (_inject) her şeyi otomatik olarak CPU'ya taşıyıp geri koyar — bu yüzden
#   hem Termux/CPU hem Colab/GPU'da doğru çalışır (GPU'da ek bir host↔device
#   transfer maliyeti vardır, bkz. bilinen sınırlamalar).
#   → The kernel is host-side (CPU) C++, not a __global__ CUDA kernel.
#     TORCH_CHECK guards raise a clear error on CUDA tensors; the Python
#     side (_inject) automatically moves everything to CPU and back — this
#     is correct on both Termux/CPU and Colab/GPU (with a small host↔device
#     transfer cost on GPU, see known limitations below).
#
# ─────────────────────────────────────────────────────────────────────────
# BİLİNEN SINIRLAMALAR / KNOWN LIMITATIONS (dürüstçe belgelenmiştir)
# ─────────────────────────────────────────────────────────────────────────
#
#   • GPU performansı: her katmanda hidden state CPU'ya indirilip geri
#     yükleniyor (CUDA güvenliği için). Bu, Termux/CPU'da maliyetsizdir ama
#     Colab GPU'da token/sn hızını düşürür. Gerçek performans için native
#     bir __global__ CUDA kernel'i yazılması gerekir — henüz yapılmadı.
#     → GPU performance: hidden state is round-tripped to CPU every layer
#       for CUDA safety. Negligible on Termux/CPU, but reduces tok/s on
#       Colab GPU. A native __global__ CUDA kernel would fix this — not
#       yet written.
#   • Ortogonal eksenler arası dolaylı etkileşim: Gram-Schmidt bir eksenin
#     anayasa DOĞRULTUSUNDAKİ bileşenini sıfırlar, ama kavramsal korelasyon
#     (ör. "logical" kelimesi hem ışık ekseninde hem anayasada güçlüyse)
#     katmanlar arası dolaylı bir etkileşime yol açabilir. Gözlemsel
#     testlerde küçük (~%5 mertebesinde) ama ölçülebilir bir etki görüldü.
#     → Indirect cross-axis interaction: Gram-Schmidt zeroes the component
#       ALONG the anayasa direction, but conceptual correlation (e.g. if
#       "logical" is strong in both the light axis and anayasa) can still
#       produce a small indirect interaction across layers. Empirically
#       observed to be small (~5% scale) but measurable.
#   • Örnekleme gürültüsü: temperature>0 iken vanilla/steered metin farkının
#     bir kısmı steering'den değil, rastgele örneklemeden gelebilir. UI'daki
#     sıcaklık kaydırıcısını 0'a çekmek (greedy) bunu ortadan kaldırır.
#     → Sampling noise: with temperature>0, part of the vanilla/steered text
#       difference may come from random sampling rather than steering.
#       Setting the UI temperature slider to 0 (greedy) eliminates this.
#
# Model   : Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)
# Kernel  : C++ — akbas_orthogonalize + akbas_observe_multi + akbas_steer_multi
# Lisans  : Apache 2.0 taban model üzerine inşa edilmiştir / built on an
#           Apache 2.0 base model — © Akbaş
# =============================================================================

import subprocess, sys, os, shutil, time, gc, math

def _pip(pkg):
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"],
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

for _pkg,_imp in [("ninja","ninja"),("gradio","gradio"),
                  ("bitsandbytes>=0.46.1","bitsandbytes")]:
    try:
        m=__import__(_imp)
        if _imp=="bitsandbytes":
            if tuple(int(x) for x in m.__version__.split(".")[:3])<(0,46,1):
                raise ImportError
    except ImportError:
        print(f"installing {_pkg}..."); _pip(_pkg); print(f"✅ {_imp} ready")

import torch, torch.utils.cpp_extension, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
torch.backends.cudnn.deterministic    = True
torch.backends.cudnn.benchmark        = False

# =============================================================================
# C++ ÇEKİRDEK — Native Multi-Axis Manifold Engine
# =============================================================================
_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

/*
 * AkbasCore 2.0 Motor Sabitleri [motor sabitler — değiştirmek derleme gerektirir]
 * oran=0.32  doyum=0.75  karşıt=-0.40  sapma=0.20  fren=0.30
 * sınır_alt=0.04  sınır_üst=0.20  sınır_oran=0.045
 * (1.2/1.3 ile birebir aynı — tek-eksen matematiği değişmedi, çoklu eksene
 *  genişletildi)
 */

// ─── NATIVE GRAM-SCHMIDT ORTOGONALİZASYON ────────────────────────────────────
// axis 0 = anayasa: sabit referans, sadece normalize edilir.
// axis 1..K-1: kendinden önceki TÜM eksenlere göre dikleştirilir, sonra
// normalize edilir. Sonuç: hiçbir alternatif eksen anayasa (veya birbiri)
// doğrultusunda bileşen taşımaz — null-space projeksiyonu bunun doğal sonucu.
torch::Tensor akbas_orthogonalize(torch::Tensor compass_matrix) {
    TORCH_CHECK(!compass_matrix.is_cuda(),
        "AkbasCore çekirdeği CUDA tensörüyle çağrıldı. Bu kernel host-side "
        "(CPU) C++ döngüleriyle yazıldı — CUDA belleğini doğrudan dereference "
        "etmek segfault'a yol açar. Çağıran taraf (_inject) tensörü .cpu()'ya "
        "taşımadan bu fonksiyona ulaştırmış olmalı.");
    auto cm = compass_matrix.clone().contiguous();
    const int K = cm.size(0), D = cm.size(1);
    float *p = cm.data_ptr<float>();

    for (int k = 0; k < K; ++k) {
        float *vk = p + k * D;
        // önceki tüm eksenlere göre dikleştir
        for (int j = 0; j < k; ++j) {
            const float *vj = p + j * D;
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += vk[d] * vj[d];
            for (int d = 0; d < D; ++d) vk[d] -= dot * vj[d];
        }
        // normalize
        float sq = 0.0f;
        for (int d = 0; d < D; ++d) sq += vk[d] * vk[d];
        float norm = sqrtf(sq) + 1e-8f;
        for (int d = 0; d < D; ++d) vk[d] /= norm;
    }
    return cm;
}

// ─── katman-derinlik eğrisi (parçalı-doğrusal) ───────────────────────────────
static inline float layer_curve(int layer_idx, float center, float width) {
    if (width < 1e-6f) return 1.0f;
    float x = ((float)layer_idx - (center - width * 0.5f)) / width;
    return std::clamp(x, 0.0f, 1.0f);
}

// ─── ortak tek-eksen katki hesabı (gözlem+steering paylaşır) ─────────────────
//
// İKİ FARKLI FİZİK:
//   REZONANS (attractor_mode=0, AkbasCore 1.2/2.0 klasik davranışı):
//     katki = alpha * cs * kv * son  →  işaret cs'nin işaretini TAKİP EDER.
//     Yani mevcut hizalanmayı güçlendirir (zaten hizalıysa daha hizalı yapar,
//     zaten tersse daha ters yapar). Bir "amplifikatör"dür, "düzeltici" değil.
//
//   ÇEKİCİ / ATTRACTOR (attractor_mode=1, aktif kilitleme):
//     katki = alpha * kv * açısal_hata,  açısal_hata=(1-cs)/2 ∈ [0,1]
//     İşaret HER ZAMAN pozitif (+axis_vec yönünde). Şiddet, hedeften ne kadar
//     saptığıyla orantılı: cs→1 (zaten hedefte) → katki→0 (gereksiz müdahale
//     yok). cs→-1 (tam ters) → katki→maksimum (aktif düzeltme). Bu, modeli
//     GERÇEKTEN hedef koordinata çeken bir kuvvettir.
//
// attractor_mode [0,1] arası sürekli karışım: iki fiziği birbirine karıştırır.
static inline float axis_katki(
    const float *tok, const float *axis_vec, int D,
    float norm_hint, float alpha, int layer_idx,
    float sonum, float zirve, float taban, float attractor_mode,
    float &pcv,               // in/out: bu eksenin bu token için önceki cos'u
    float &out_cs, float &out_kb, float &out_kv
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

    // rezonans fiziği (klasik, işaret cs'yi takip eder)
    float son = 1.0f;
    if      (cs >  0.75f) son = (1.0f - cs) / 0.25f;
    else if (cs < -0.40f) son = 1.6f;
    float katki_rezonans = alpha * cs * kv * 0.32f * son;

    // çekici fiziği (attractor, işaret her zaman + hedefe doğru)
    float ang_err = (1.0f - cs) * 0.5f;              // 0..1
    float katki_cekici = alpha * kv * 0.32f * ang_err * 2.0f;

    float m = std::clamp(attractor_mode, 0.0f, 1.0f);
    float katki = std::clamp(
        (1.0f - m) * katki_rezonans + m * katki_cekici, -mk, mk);

    out_cs = cs; out_kb = kb; out_kv = kv;
    return katki;
}

// ─── PASS A: GÖZLEM (multi-axis, hidden state'e DOKUNMAZ) ────────────────────
torch::Tensor akbas_observe_multi(
    torch::Tensor hidden,
    torch::Tensor compass_matrix,   // [K, D] — zaten ortogonalize edilmiş
    torch::Tensor alphas,           // [K]
    torch::Tensor centers,          // [K]  (axis 0 için görmezden gelinir)
    torch::Tensor widths,           // [K]
    int layer_idx,
    float sonum, float zirve, float taban, float attractor_mode,
    torch::Tensor prev_cosine,      // [K, tokens]
    torch::Tensor log_buf,          // [K, N_LAYERS, 4] — KÜMÜLATİF TOPLAM (average değil)
    torch::Tensor log_count         // [N_LAYERS] — bu katmanda kaç token ölçüldü
) {
    TORCH_CHECK(!hidden.is_cuda() && !compass_matrix.is_cuda(),
        "AkbasCore akbas_observe_multi: CUDA tensörü tespit edildi. Bu "
        "kernel yalnızca CPU tensörleriyle çalışır (host-side C++ döngüsü, "
        "__global__ CUDA kernel değil). Python tarafı çağırmadan önce "
        ".cpu()'ya taşımalı.");
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
            float curve = (k == 0) ? 1.0f
                        : layer_curve(layer_idx, cep[k], wip[k]);
            float cs, kb, kv;
            float &pcv = pcp[k * (B * S) + idx];
            float katki = axis_katki(tok, cmp + k * D, D, norm_hint,
                                      alp[k], layer_idx, sonum, zirve, taban,
                                      attractor_mode,
                                      pcv, cs, kb, kv) * curve;
            // gözlem modu: hidden state'e yazılmaz. Loglama: sadece GERÇEK
            // üretim adımlarında (S==1, yani decode aşaması) biriktirilir —
            // prefill (S>1, prompt işleme) hariç tutulur ki tablo prompt'un
            // değil, ÜRETİLEN metnin dinamiğini yansıtsın. Değerler TOPLANIR
            // (average değil); Python tarafı log_count'a bölerek ortalar.
            if (b == 0 && S == 1) {
                float *slot = lbp + k * N_LAYERS * 4 + layer_idx * 4;
                slot[0] += cs; slot[1] += kb; slot[2] += kv; slot[3] += katki;
            }
        }
        if (b == 0 && S == 1) {
            lcp[layer_idx] += 1.0f;
        }
    }
    return h;
}

// ─── PASS B: STEERING (multi-axis, sırayla hidden state'e yazılır) ───────────
torch::Tensor akbas_steer_multi(
    torch::Tensor hidden,
    torch::Tensor compass_matrix,
    torch::Tensor alphas,
    torch::Tensor centers,
    torch::Tensor widths,
    int layer_idx,
    float sonum, float zirve, float taban, float attractor_mode,
    torch::Tensor prev_cosine,
    torch::Tensor log_buf,
    torch::Tensor log_count
) {
    TORCH_CHECK(!hidden.is_cuda() && !compass_matrix.is_cuda(),
        "AkbasCore akbas_steer_multi: CUDA tensörü tespit edildi. Bu kernel "
        "yalnızca CPU tensörleriyle çalışır (host-side C++ döngüsü, "
        "__global__ CUDA kernel değil). Python tarafı çağırmadan önce "
        ".cpu()'ya taşımalı.");
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
            // norm_hint her eksenden ÖNCE yeniden ölçülür: bir önceki eksenin
            // yazdığı katkı, sonraki eksenin gördüğü hidden state'in parçasıdır
            // (tek geçişte sıralı, kümülatif uygulama).
            float sq = 0.0f;
            for (int d = 0; d < D; ++d) sq += tok[d] * tok[d];
            float norm_hint = sqrtf(sq) + 1e-6f;

            float curve = (k == 0) ? 1.0f
                        : layer_curve(layer_idx, cep[k], wip[k]);
            float cs, kb, kv;
            float &pcv = pcp[k * (B * S) + idx];
            const float *axis_vec = cmp + k * D;
            float katki = axis_katki(tok, axis_vec, D, norm_hint,
                                      alp[k], layer_idx, sonum, zirve, taban,
                                      attractor_mode,
                                      pcv, cs, kb, kv) * curve;

            for (int d = 0; d < D; ++d) tok[d] += katki * axis_vec[d];

            // steering modu: GERÇEK üretim adımlarında (S==1, decode) biriktirilir.
            // prefill (S>1) hariç — tablo prompt'un değil üretilen metnin
            // dinamiğini yansıtsın. TOPLANIR; Python log_count'a bölerek ortalar.
            if (b == 0 && S == 1) {
                float *slot = lbp + k * N_LAYERS * 4 + layer_idx * 4;
                slot[0] += cs; slot[1] += kb; slot[2] += kv; slot[3] += katki;
            }
        }
        if (b == 0 && S == 1) {
            lcp[layer_idx] += 1.0f;
        }
    }
    return h;
}
"""

# ── Derleme ───────────────────────────────────────────────────────────────────
if not torch.utils.cpp_extension.is_ninja_available():
    import site
    for _sp in site.getsitepackages():
        _b=os.path.normpath(os.path.join(_sp,"..","..","bin"))
        if os.path.isdir(_b): os.environ["PATH"]=_b+":"+os.environ.get("PATH","")

for _n in ["akbas_dra_f1","akbas_dra_f2"]:
    for _r in ["/tmp/akbas_dra",os.path.expanduser("~/.cache/torch_extensions")]:
        _p=os.path.join(_r,_n)
        if os.path.exists(_p): shutil.rmtree(_p,ignore_errors=True)

def _build(name):
    d=f"/tmp/akbas_dra/{name}"; os.makedirs(d,exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name, cpp_sources=_CPP,
        functions=["akbas_orthogonalize","akbas_observe_multi","akbas_steer_multi"],
        build_directory=d, verbose=False,
        extra_cflags=["-O2","-std=c++17"])

try:    _kernel=_build("akbas_dra_f1")
except Exception as e:
    print(f"  retry ({e})..."); _kernel=_build("akbas_dra_f2")
print("✅ AkbasCore 2.0 — DRA/MANIFOLD native multi-axis çekirdeği derlendi")

# =============================================================================
# SABİT PARAMETRELER
# =============================================================================
MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
# sıcaklık (temperature): örnekleme rastgeleliği — 0 = deterministik (greedy,
# aynı girdi hep aynı çıktıyı üretir), yüksek = daha rastgele/çeşitli. UI'dan
# ayarlanabilir hale getirildi ki vanilla/steered karşılaştırmasında farkın ne
# kadarının steering'den, ne kadarının örnekleme gürültüsünden geldiği ayrılabilsin.
DEFAULT_TEMPERATURE = 0.65
TEMP_MIN, TEMP_MAX  = 0.0, 1.50
TOP_P        = 0.90
TOP_K        = 50
REP_PENALTY  = 1.15
N_LAYERS     = 20
MAX_EXTRA_AXES = 3   # K = 1 (anayasa) + 0..3 ek eksen  → K_max = 4

CONSTITUTION = {
    "harm":     (0.9228, ["safe","harmless","protective","secure","careful"]),
    "honesty":  (0.9372, ["honest","accurate","truthful","transparent","precise"]),
    "autonomy": (0.8788, ["autonomous","respectful","unbiased","free","neutral"]),
    "fairness": (0.9196, ["fair","just","equitable","balanced","impartial"]),
}
BILISIM_ETIK   = 0.40
BILISIM_MANTIK = 0.60
LOGIC_ANCHORS = [
    "logical","empirical","systematic","structured","verifiable",
    "analyze","precise","deterministic","sequential","causal",
    "rigorous","impossible","contradiction","identify","optimize",
]
DISCLAIMER_MARKERS = [
    "as an ai","as a language model","i apologize","i must clarify",
    "i'm unable to","i am unable to","i cannot be certain",
]

DEFAULT_IVME  = 0.50
DEFAULT_SONUM = 0.30
DEFAULT_ZIRVE = 0.70
DEFAULT_TABAN = 0.20
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
DEFAULT_SEED         = 42
DEFAULT_SAMPLE_COUNT = 40
DEFAULT_AXIS_ALPHA   = 0.50
DEFAULT_CENTER       = 15.0   # L15'ten itibaren devreye girme varsayılanı
DEFAULT_WIDTH        = 6.0    # ~L12→L18 arası geçiş
DEFAULT_ATTRACTOR    = 1.0    # 0=rezonans (klasik amplifikasyon), 1=tam çekici (aktif kilitleme)

# =============================================================================
# AkbasCore 2.0
# =============================================================================
class AkbasCore:

    def __init__(self):
        print("🔱⬜ AKBASCORE 2.0 — DRA/MANIFOLD | Qwen2.5-1.5B-Instruct")
        print("   Native Multi-Axis & Manifold Engine")
        if torch.cuda.is_available():
            vram=torch.cuda.get_device_properties(0).total_memory/1e9
            print(f"   GPU: {torch.cuda.get_device_name(0)} {vram:.1f}GB")
        else:
            print("   Device: CPU")

        self.tok=AutoTokenizer.from_pretrained(MODEL_ID)
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token

        self.model=AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto",
            trust_remote_code=True, dtype=torch.bfloat16,
            low_cpu_mem_usage=True)
        if hasattr(self.model.config,"_attn_implementation"):
            self.model.config._attn_implementation="eager"
        self.device=next(self.model.parameters()).device
        self.hidden_dim=self.model.config.hidden_size

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("   Anayasa ekseni (axis 0) inşa ediliyor...")
        self._anayasa=self._build_anayasa_pusula()  # cpu float tensor, normalized
        self._token_norms=None

        gc.collect()
        print("✅ AkbasCore 2.0 — DRA/MANIFOLD hazır\n")

    # ── embedding yardımcıları ────────────────────────────────────────────────
    def _word_vec(self,word):
        W=self.model.model.embed_tokens.weight
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
        W=self.model.model.embed_tokens.weight
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
        # Karanlık (veya herhangi bir tekil kavram) pusulası: kelime kümesinin
        # gerçek embedding ortalaması, doğrudan normalize edilir. Diff-of-means
        # gibi bir karşıt kümeye ihtiyaç duymaz — "contradiction, chaos,
        # nonsense..." gibi tek bir kavram kümesini doğrudan bir yön olarak verir.
        words=[w.strip() for w in words_text.replace("\n",",").split(",")]
        v=self._mean_vec(words)
        if v is None: return None
        return F.normalize(v, dim=0).contiguous()

    # ── eksen matrisi kurulumu (Python: sadece ADAY vektör üretimi) ─────────
    def _build_axis_candidate(self, mode, pos_text, neg_text,
                               sample_method, seed, sample_count, words_text):
        if mode==AXIS_MODES[0]:
            v=self._build_diff_axis(pos_text, neg_text)
        elif mode==AXIS_MODES[1]:
            v=self._build_sample_axis(sample_method, seed, sample_count)
        else:
            v=self._build_wordset_axis(words_text)
        return v  # None ise çağıran taraf bu ekseni atlar

    # ── hook ─────────────────────────────────────────────────────────────────
    def _remove_hooks(self,hooks):
        for h in hooks: h.remove()

    def _inject(self, compass_matrix, alphas, centers, widths,
                log_buf, log_count, pc_holder, sonum, zirve, taban, attractor_mode, mode="observe"):
        layers=self.model.model.layers; hooks=[]
        fn = _kernel.akbas_observe_multi if mode=="observe" else _kernel.akbas_steer_multi
        K = compass_matrix.shape[0]
        # ── CUDA GÜVENLİĞİ ────────────────────────────────────────────────────
        # Çekirdek host-side (CPU) C++ döngüleriyle yazıldı — __global__ CUDA
        # kernel'i DEĞİL. Bu tensörler self.device CUDA ise data_ptr<float>()
        # ile alınan adres GPU belleğine işaret eder; host döngüsünün onu
        # doğrudan dereference etmesi segfault/illegal-memory-access riski
        # taşır. Bu yüzden kernel'e giden her şeyi burada CPU'ya sabitliyoruz;
        # hidden state'i de aynı şekilde CPU'ya alıp, işlem bitince orijinal
        # cihaza geri koyuyoruz. K,D küçük olduğu için (K≤4, D=1536) bu
        # transferin maliyeti ihmal edilebilir düzeyde.
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
        for i in range(min(N_LAYERS,len(layers))):
            hooks.append(layers[i].register_forward_hook(make_hook(i)))
        return hooks

    def _run_pass(self, inp, max_tokens, mode, compass_matrix, alphas,
                  centers, widths, sonum, zirve, taban, attractor_mode, temperature):
        K = compass_matrix.shape[0]
        # log_buf/log_count CPU'da tutulur — kernel yalnızca CPU tensörüyle çalışır.
        # log_buf artık KÜMÜLATİF TOPLAM: her decode adımında (prefill hariç)
        # üstüne eklenir. log_count[layer], o katmanda kaç üretilen token
        # ölçüldüğünü tutar — ortalama, generate() bittikten sonra alınır.
        log_buf   = torch.zeros(K, N_LAYERS, 4, dtype=torch.float32, device="cpu")
        log_count = torch.zeros(N_LAYERS, dtype=torch.float32, device="cpu")
        pc_holder = {"pc": None}
        hooks = self._inject(compass_matrix, alphas, centers, widths,
                              log_buf, log_count, pc_holder, sonum, zirve, taban,
                              attractor_mode, mode=mode)
        eos=list({self.tok.eos_token_id,
                  self.tok.convert_tokens_to_ids("<|im_end|>"),
                  self.tok.convert_tokens_to_ids("<|endoftext|>")}-{None,-1})
        # sıcaklık (temperature) ≈ 0 ise greedy decoding'e düş (do_sample=False).
        # HF transformers do_sample=True iken temperature=0 kabul etmez/hata verir;
        # bu yüzden eşik altında sample'ı tamamen kapatıyoruz — bu, saf steering
        # etkisini örnekleme gürültüsünden ayırmak isteyenler için deterministik
        # karşılaştırma sağlar.
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
        # log_buf şu ana kadar KÜMÜLATİF TOPLAM — burada ortalamaya çeviriyoruz.
        # count=0 olan katman olursa (örn. max_tokens=0) sıfıra bölmeyi önle.
        count = log_count.cpu()
        count_safe = count.clamp(min=1.0)
        log_avg = log_buf.cpu() / count_safe.view(1, N_LAYERS, 1)
        return text, ms, tps, n_in, n_out, log_avg, count

    # ── log ─────────────────────────────────────────────────────────────────
    def _log_header(self, label, ms, tps, n_in, n_out, ivme, sonum, zirve, taban,
                     temperature, attractor_label="—", axis_infos=None):
        W=64; lines=[]
        lines.append("═"*W)
        lines.append(f"  {label}")
        lines.append("═"*W)
        lines.append(f"  ⏱  {ms:.0f} ms  |  {tps:.1f} tok/s  |  "
                     f"{n_in} in  |  {n_out} out tokens")
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
        for i in range(N_LAYERS):
            if count[i].item() <= 0: continue   # bu katmanda hiç ölçüm alınmadı
            cs,kb,kv,kat = lb_k[i].tolist()
            cos_vals.append(cs); kat_vals.append(kat)
            lines.append(f"  {i:>3}  {cs:>+10.4f}  {kb:>8.5f}  {kv:>8.5f}  {kat:>+10.6f}")
        if cos_vals:
            drift = cos_vals[-1]-cos_vals[0]
            lines.append(f"  drift(cos) L0→L{N_LAYERS-1}: {drift:+.4f}  |  "
                         f"katki ort. toplamı={sum(kat_vals):+.6f}")
        return lines

    # ── ANA ÇALIŞMA FONKSİYONU ────────────────────────────────────────────────
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

        # ── aday vektörler (Python: sadece gerçek-embedding'den üretim) ──────
        raw_axes = [self._anayasa]           # axis 0 = anayasa, her zaman var
        alphas_l = [ivme]
        centers_l = [0.0]; widths_l = [0.0]  # axis0 için curve=1 (kernel'de hardcode)
        axis_infos = []
        for (mode,pos,neg,sm,seed,cnt,words,kutup,alpha,cen,wid) in axis_defs:
            v = self._build_axis_candidate(mode,pos,neg,sm,seed,cnt,words)
            if v is None:
                axis_infos.append(f"[atlandı: girdi boş] mode={mode}")
                continue
            # ── KUTUP (polarite): itici eksen = hedefin NEGATİFİNE çekici ────
            # Matematiksel olarak "V_dark'tan itilmek" ile "-V_dark'a çekilmek"
            # birebir aynı şeydir — çekirdek hangisi olduğunu bilmiyor, sadece
            # bir vektöre çekiliyor. Kutup, tamamen Python'daki bu ön-işleme
            # adımıdır; kernel'e hiçbir yeni fizik eklenmedi.
            polarite_repulsor = (kutup==AXIS_POLARITY[1])
            if polarite_repulsor:
                v = -v
            raw_axes.append(v)
            alphas_l.append(float(alpha))
            centers_l.append(float(cen))
            widths_l.append(float(wid))
            src = ("diff-of-means" if mode==AXIS_MODES[0] else
                   f"örnek/{sm}" if mode==AXIS_MODES[1] else
                   "kelime kümesi")
            kutup_etiket = "İTİCİ(karanlık)" if polarite_repulsor else "çekici(ışık)"
            axis_infos.append(f"axis{len(raw_axes)-1}: {src}  [{kutup_etiket}]  "
                              f"α={alpha:.2f}  merkez=L{cen:.0f}  genişlik={wid:.0f}")

        raw_matrix = torch.stack(raw_axes).float().to(self.device)  # [K,D]
        compass_matrix = _kernel.akbas_orthogonalize(raw_matrix).contiguous()
        alphas  = torch.tensor(alphas_l,  dtype=torch.float32, device=self.device)
        centers = torch.tensor(centers_l, dtype=torch.float32, device=self.device)
        widths  = torch.tensor(widths_l,  dtype=torch.float32, device=self.device)
        K = compass_matrix.shape[0]

        prompt_text = (f"<|im_start|>system\nYou are a helpful, accurate, and "
                       f"thoughtful assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
                       f"<|im_start|>assistant\n")
        inp = self.tok(prompt_text, return_tensors="pt").to(self.device)

        # ── PASS A: VANILLA OBSERVER — SADECE axis0 (anayasa), K=1 ───────────
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

        # ── PASS B: STEERED — K eksen (native multi-axis) ────────────────────
        mode_label = ("rezonans" if attractor_mode<0.01 else
                      "çekici/attractor" if attractor_mode>0.99 else
                      f"karışım (m={attractor_mode:.2f})")
        print(f"   Pass B: Steered  [K={K} eksen, mod={mode_label}]")
        txt_s, ms_s, tps_s, n_in_s, n_out_s, lb_s, cnt_s = self._run_pass(
            inp, max_tokens, "steer", compass_matrix, alphas,
            centers, widths, sonum, zirve, taban, attractor_mode, temperature)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ── VANILLA PANEL ─────────────────────────────────────────────────────
        van_lines = self._log_header(
            "⬜ VANILLA OBSERVER — katki=0 (hidden state değişmedi)",
            ms_v, tps_v, n_in_v, n_out_v, ivme, sonum, zirve, taban,
            temperature, attractor_label="n/a (gözlem modu)")
        van_lines += self._log_axis_table(lb_v[0], cnt_v, "axis0: anayasa (referans)", True)
        van_lines.append("═"*64)
        van_out = txt_v + "\n\n" + "\n".join(van_lines)

        # ── STEERED PANEL ─────────────────────────────────────────────────────
        st_lines = self._log_header(
            "🔱 AKBASCORE 2.0 — DRA/MANIFOLD STEERED — native multi-axis uygulandı",
            ms_s, tps_s, n_in_s, n_out_s, ivme, sonum, zirve, taban,
            temperature, attractor_label=f"{mode_label} (m={attractor_mode:.2f})",
            axis_infos=axis_infos)
        st_lines += self._log_axis_table(lb_s[0], cnt_s, "axis0: anayasa (permanent floor)", True)
        for k in range(1, K):
            label = axis_infos[k-1] if k-1 < len(axis_infos) else f"axis{k}"
            st_lines += self._log_axis_table(lb_s[k], cnt_s, label, False)
        # baseline karşılaştırma: axis0'ın vanilla'ya göre kayması
        # (iki pass FARKLI sayıda token üretmiş olabilir — ortalama üzerinden
        # karşılaştırma bu farktan etkilenmez, ama N'lerin farklı olabileceğini
        # not düşüyoruz.)
        dcos0 = lb_s[0,:,0] - lb_v[0,:,0]
        nz = [i for i in range(N_LAYERS) if cnt_v[i].item()>0 and cnt_s[i].item()>0]
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
# YÜKLEME
# =============================================================================
print("\n"+"="*58)
print("🔱⬜ AKBASCORE 2.0 — DRA/MANIFOLD | QWEN2.5-1.5B | YÜKLENIYOR...")
print("="*58+"\n")
core = AkbasCore()
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI — AkbasCore 2.0
# =============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');
body { background: #04050a; margin: 0; }
.gradio-container { max-width: 1360px !important; margin: 0 auto !important;
    background: #04050a !important; font-family: 'JetBrains Mono', monospace !important; }
#akbas-header { text-align: center; padding: 28px 0 16px; border-bottom: 1px solid #111828; }
#akbas-header h1 { font-size: 1.05rem; font-weight: 500; letter-spacing: .20em; margin: 0 0 6px;
    background: linear-gradient(90deg, #5588cc, #33cc77); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; }
#akbas-header .sub { font-size: .56rem; color: #1e2a3a; letter-spacing: .08em; margin: 0; line-height: 1.6; }
#motor-panel { background: #060810; border: 1px solid #0e1520; border-top: 2px solid #1a2a44;
    padding: 18px 24px 16px; }
#axis-panel { background: #08060f; border: 1px solid #16101f; border-top: 2px solid #3a2a5a;
    padding: 18px 24px 16px; margin-top: 10px; }
#motor-title, #axis-title { font-size: .62rem; letter-spacing: .20em; color: #2a4060; margin: 0 0 4px; }
#axis-title { color: #5a4080; }
#motor-desc, #axis-desc { font-size: .54rem; color: #182030; letter-spacing: .05em;
    margin: 0 0 14px; line-height: 1.65; }
input[type=range] { accent-color: #2a5080 !important; height: 2px !important; }
input[type=range]::-webkit-slider-thumb { background: #3a7acc !important; width: 10px !important; height: 10px !important; }
#inp-section { background: #050710; border: 1px solid #0c1220; padding: 14px 24px; }
#inp textarea { background: #070910 !important; color: #8aaccc !important; border: 1px solid #0e1828 !important;
    border-radius: 2px !important; font-size: .80rem !important; line-height: 1.70 !important; }
#btn { background: linear-gradient(135deg, #0a1220, #142040) !important; color: #6699cc !important;
    border: 1px solid #1a3050 !important; font-size: .78rem !important; font-weight: 500 !important;
    letter-spacing: .18em !important; border-radius: 2px !important; padding: 10px 0 !important;
    width: 100% !important; margin: 10px 24px 0 !important; }
#btn:hover { background: linear-gradient(135deg, #102030, #1a3060) !important; color: #88bbee !important;
    box-shadow: 0 0 14px #3366aa18 !important; border-color: #2a4a80 !important; }
.panel-lbl-v { font-size: .58rem; letter-spacing: .14em; color: #336688; text-align: center;
    padding: 6px 0 3px; border-bottom: 1px solid #0c1828; margin-bottom: 4px; }
.panel-lbl-s { font-size: .58rem; letter-spacing: .14em; color: #228844; text-align: center;
    padding: 6px 0 3px; border-bottom: 1px solid #0a1810; margin-bottom: 4px; }
#out_v textarea { background: #040608 !important; color: #5599cc !important; border: 1px solid #0a1420 !important;
    font-size: .72rem !important; line-height: 1.70 !important; border-radius: 2px !important; }
#out_s textarea { background: #040806 !important; color: #33bb66 !important; border: 1px solid #081410 !important;
    font-size: .72rem !important; line-height: 1.70 !important; border-radius: 2px !important; }
#ver-tag { font-size: .48rem; color: #0e1820; text-align: center; padding: 6px 0 2px; letter-spacing: .12em; }
"""

LBL_IVME  = f"ivme — axis0 (anayasa) steering velocity  [band {IVME_MIN}–{IVME_MAX}]"
LBL_SONUM = f"sönüm — decay rate (tüm eksenler paylaşır)  [band {SONUM_MIN}–{SONUM_MAX}]"
LBL_ZIRVE = f"zirve — amplitude (tüm eksenler paylaşır)  [band {ZIRVE_MIN}–{ZIRVE_MAX}]"
LBL_TABAN = f"taban — permanent floor (tüm eksenler paylaşır)  [band {TABAN_MIN}–{TABAN_MAX}]"

def _toggle_axis_mode(mode):
    return (gr.update(visible=(mode==AXIS_MODES[0])),
            gr.update(visible=(mode==AXIS_MODES[1])),
            gr.update(visible=(mode==AXIS_MODES[2])))

def _toggle_axis_count(n):
    n=int(n)
    return (gr.update(visible=n>=1), gr.update(visible=n>=2), gr.update(visible=n>=3))

def _build_axis_ui(idx):
    """idx: 1,2,3 — extra axis definition block."""
    with gr.Group(visible=(idx==1)) as grp:
        gr.HTML(f"""<div style="font-size:.58rem;color:#7755bb;letter-spacing:.1em;
                    margin-bottom:6px;">EKSEN {idx}</div>""")
        mode = gr.Dropdown(choices=AXIS_MODES, value=AXIS_MODES[0],
                           label=f"Eksen {idx} — üretim yöntemi")
        with gr.Group(visible=True) as diff_g:
            with gr.Row():
                pos = gr.Textbox(label=f"Kavram A", placeholder=LIGHT_WORDS_PLACEHOLDER)
                neg = gr.Textbox(label=f"Kavram B", placeholder=DARK_WORDS_PLACEHOLDER)
        with gr.Group(visible=False) as samp_g:
            sm = gr.Dropdown(choices=TOKEN_SAMPLE_METHODS, value=TOKEN_SAMPLE_METHODS[0],
                             label="Örnekleme yöntemi")
            with gr.Row():
                seed = gr.Slider(0,10000,value=DEFAULT_SEED,step=1,label="Seed")
                cnt  = gr.Slider(5,200,value=DEFAULT_SAMPLE_COUNT,step=5,label="Token sayısı")
        with gr.Group(visible=False) as words_g:
            words = gr.Textbox(
                label="Kelime kümesi (virgülle ayrılmış) — ortalama embedding'i pusula olur",
                placeholder=DARK_WORDS_PLACEHOLDER, lines=2)
        kutup = gr.Radio(choices=AXIS_POLARITY, value=AXIS_POLARITY[0],
                         label=f"Eksen {idx} kutbu — çekici mi (hedefe çeker) "
                               f"itici mi (hedeften uzaklaştırır)?")
        with gr.Row():
            alpha = gr.Slider(0.1,1.0,value=DEFAULT_AXIS_ALPHA,step=0.05,
                              label=f"α{idx} — eksen gücü")
            cen   = gr.Slider(0,19,value=DEFAULT_CENTER,step=1,
                              label=f"merkez — hangi katmandan itibaren devrede")
            wid   = gr.Slider(1,15,value=DEFAULT_WIDTH,step=1,
                              label=f"genişlik — geçiş yumuşaklığı")
        mode.change(fn=_toggle_axis_mode, inputs=mode, outputs=[diff_g,samp_g,words_g])
    return grp, mode, pos, neg, sm, seed, cnt, words, kutup, alpha, cen, wid

with gr.Blocks(title="🔱 AkbasCore 2.0 — DRA/MANIFOLD") as demo:

    gr.HTML("""
    <div id="akbas-header">
      <h1>🔱 AKBASCORE 2.0 — DRA/MANIFOLD &nbsp;·&nbsp; QWEN2.5-1.5B-INSTRUCT</h1>
      <p class="sub">
        NATIVE MULTI-AXIS &amp; MANIFOLD ENGINE &nbsp;·&nbsp;
        GRAM-SCHMIDT ORTOGONALİZASYON + KATMAN-DERİNLİK EĞRİSİ C++ ÇEKİRDEĞİNDE
        &nbsp;·&nbsp; © AKBAŞ
      </p>
    </div>
    """)

    with gr.Column(elem_id="motor-panel"):
        gr.HTML("""
        <div id="motor-title">⚙ MOTOR — PAYLAŞILAN DİNAMİK (TÜM EKSENLER)</div>
        <div id="motor-desc">
          sönüm/zirve/taban tüm eksenler için ortak kalır (katman-derinlik
          eğrisi bunun üstüne biner). ivme sadece axis0 (anayasa) için geçerli;
          her ek eksenin kendi α'sı aşağıda ayrı tanımlanır.
        </div>
        """)
        with gr.Row():
            sl_ivme = gr.Slider(IVME_MIN, IVME_MAX, DEFAULT_IVME, step=0.05, label=LBL_IVME)
            sl_sonum = gr.Slider(SONUM_MIN, SONUM_MAX, DEFAULT_SONUM, step=0.05, label=LBL_SONUM)
        with gr.Row():
            sl_zirve = gr.Slider(ZIRVE_MIN, ZIRVE_MAX, DEFAULT_ZIRVE, step=0.05, label=LBL_ZIRVE)
            sl_taban = gr.Slider(TABAN_MIN, TABAN_MAX, DEFAULT_TABAN, step=0.05, label=LBL_TABAN)
        sl_attractor = gr.Slider(
            0.0, 1.0, value=DEFAULT_ATTRACTOR, step=0.05,
            label="yönlendirme fiziği — 0=rezonans (mevcut yönelimi güçlendirir)  "
                  "↔  1=çekici/attractor (aktif olarak hedefe kilitler)")

    with gr.Column(elem_id="axis-panel"):
        gr.HTML("""
        <div id="axis-title">🧭 NATIVE MULTI-AXIS — ORTOGONAL EKSENLER</div>
        <div id="axis-desc">
          axis0 (anayasa) her katmanda tam güçle sabit taban olarak kalır.
          Ek eksenler, C++ çekirdeğinde native Gram-Schmidt ile anayasaya VE
          birbirlerine dik hale getirildikten sonra, her biri kendi
          "merkez/genişlik" katman-derinlik eğrisiyle devreye girer.
        </div>
        """)
        n_axes = gr.Radio(choices=["0","1","2","3"], value="1",
                          label="Ek eksen sayısı (K = 1 + bu sayı)")

        grp1, m1, p1, ng1, sm1_, sd1, c1, w1_, kt1, a1, ce1, wd1 = _build_axis_ui(1)
        grp2, m2, p2, ng2, sm2_, sd2, c2, w2_, kt2, a2, ce2, wd2 = _build_axis_ui(2)
        grp3, m3, p3, ng3, sm3_, sd3, c3, w3_, kt3, a3, ce3, wd3 = _build_axis_ui(3)

        n_axes.change(fn=_toggle_axis_count, inputs=n_axes,
                      outputs=[grp1, grp2, grp3])

    with gr.Column(elem_id="inp-section"):
        inp = gr.Textbox(label="► INPUT — aynı soru her iki pass'a gönderilir",
                         lines=4, placeholder="Sorunuzu buraya yazın...", elem_id="inp")
        with gr.Row():
            sl_tok = gr.Slider(64, 1024, value=512, step=64,
                               label="MAX TOKENS — output token budget", scale=4)
            sl_temp = gr.Slider(
                TEMP_MIN, TEMP_MAX, value=DEFAULT_TEMPERATURE, step=0.05,
                label="sıcaklık (temperature) — 0=deterministik/greedy, "
                      "yüksek=daha rastgele örnekleme", scale=4)
            btn = gr.Button("▶▶  DUAL RUN", variant="primary", elem_id="btn", scale=1)

    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="panel-lbl-v">⬜ VANILLA OBSERVER — katki = 0 (axis0 referans)</div>')
            out_v = gr.Textbox(label="OUTPUT + OBSERVATION LOG", lines=40,
                               interactive=False, elem_id="out_v")
        with gr.Column():
            gr.HTML('<div class="panel-lbl-s">🔱 STEERED — native multi-axis applied</div>')
            out_s = gr.Textbox(label="OUTPUT + PER-AXIS LOG + AXIS0 DELTA", lines=40,
                               interactive=False, elem_id="out_s")

    gr.HTML("""
    <div style="text-align:center; padding: 8px 24px 4px;">
      <button id="copy-all-btn" onclick="
        var q  = document.querySelector('#inp textarea');
        var v  = document.querySelector('#out_v textarea');
        var s  = document.querySelector('#out_s textarea');
        var sep = '\\n\\n' + '#'.repeat(70) + '\\n\\n';
        var combined =
          '### SORU / PROMPT ###\\n\\n' + (q ? q.value : '') + sep +
          '### VANILLA OBSERVER (katki=0) — ÇIKTI + LOG ###\\n\\n' + (v ? v.value : '') + sep +
          '### AKBASCORE 2.0 — DRA/MANIFOLD STEERED — ÇIKTI + LOG ###\\n\\n' + (s ? s.value : '');
        if (navigator.clipboard) { navigator.clipboard.writeText(combined); }
        this.textContent='✅ KOPYALANDI — soru + iki çıktı + tüm loglar panoda';
        var btn=this;
        setTimeout(function(){ btn.textContent='📋  TÜMÜNÜ KOPYALA (soru + vanilla + steered + loglar)'; }, 2500);
      " style="padding:9px 28px;background:linear-gradient(135deg,#1a1030,#2a1850);
      color:#bb99ee;border:1px solid #3a2a5a;border-radius:3px;cursor:pointer;
      font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.10em;
      width:calc(100% - 48px);">
      📋  TÜMÜNÜ KOPYALA (soru + vanilla + steered + loglar)</button>
    </div>
    """)

    gr.HTML("""
    <div id="ver-tag">
      AKBASCORE 2.0 — DRA/MANIFOLD &nbsp;·&nbsp; axis0=anayasa (permanent) · axis1..3=ortogonal
      alternatif eksenler · native Gram-Schmidt · katman-derinlik eğrisi (parçalı-doğrusal)
      &nbsp;·&nbsp; © AKBAŞ &nbsp;·&nbsp; APACHE 2.0 BASE MODEL
    </div>
    """)

    def _n_to_int(n): return int(n)

    def _run_wrapper(prompt, max_tokens, temperature, ivme, sonum, zirve, taban, attractor,
                      n_axes_str,
                      m1v,p1v,ng1v,sm1v,sd1v,c1v,w1v,kt1v,a1v,ce1v,wd1v,
                      m2v,p2v,ng2v,sm2v,sd2v,c2v,w2v,kt2v,a2v,ce2v,wd2v,
                      m3v,p3v,ng3v,sm3v,sd3v,c3v,w3v,kt3v,a3v,ce3v,wd3v):
        return core.run(prompt, max_tokens, temperature, ivme, sonum, zirve, taban, attractor,
                        _n_to_int(n_axes_str),
                        m1v,p1v,ng1v,sm1v,sd1v,c1v,w1v,kt1v,a1v,ce1v,wd1v,
                        m2v,p2v,ng2v,sm2v,sd2v,c2v,w2v,kt2v,a2v,ce2v,wd2v,
                        m3v,p3v,ng3v,sm3v,sd3v,c3v,w3v,kt3v,a3v,ce3v,wd3v)

    _inputs = [inp, sl_tok, sl_temp, sl_ivme, sl_sonum, sl_zirve, sl_taban, sl_attractor,
               n_axes,
               m1,p1,ng1,sm1_,sd1,c1,w1_,kt1,a1,ce1,wd1,
               m2,p2,ng2,sm2_,sd2,c2,w2_,kt2,a2,ce2,wd2,
               m3,p3,ng3,sm3_,sd3,c3,w3_,kt3,a3,ce3,wd3]

    btn.click(fn=_run_wrapper, inputs=_inputs, outputs=[out_v, out_s])
    inp.submit(fn=_run_wrapper, inputs=_inputs, outputs=[out_v, out_s])

print("🚀 AkbasCore 2.0 — DRA/MANIFOLD Gradio başlatılıyor...")
demo.launch(
    share=True, debug=False,
    theme=gr.themes.Base(primary_hue="slate", neutral_hue="slate",
                         font=gr.themes.GoogleFont("JetBrains Mono")),
    css=CSS,
)
