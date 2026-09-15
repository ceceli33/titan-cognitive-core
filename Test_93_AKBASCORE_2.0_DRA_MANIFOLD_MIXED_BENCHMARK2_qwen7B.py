# =============================================================================
# 🔱⬜ AKBASCORE 2.0 — DRA/MANIFOLD (TEST 93 — RESTORING FORCE AYARLARI PORTLANDI)
# Test 85'in (AkbasCore 1.2, Manifold'suz, Qwen2.5-1.5B) RESTORING FORCE
# modunda en operasyonel-detaylı çıktıyı ürettiği ayar kolları, mevcut
# Test 92 CUDA/Manifold mimarisine PORTLANDI ve aynı karma benchmark ile
# test ediliyor. Bölüm B bu sefer 2x büyüklükte.
# The motor settings that produced Test 85's (AkbasCore 1.2, no Manifold,
# Qwen2.5-1.5B) most operationally-detailed output under RESTORING FORCE mode
# have been PORTED onto the current Test 92 CUDA/Manifold architecture, and
# re-tested with the same mixed benchmark. Part B is 2x the previous scale.
# =============================================================================
#
# NE DEĞİŞTİ / WHAT CHANGED (Test 92'den farklar, hepsi belgelenmiş):
# WHAT CHANGED (differences from Test 92, all documented):
#
#   1) MOTOR AYARLARI PORTLANDI / MOTOR SETTINGS PORTED:
#      ivme=1.00 sönüm=0.10 zirve=1.00 taban=0.40 (Test 92: 0.50/0.30/0.70/0.20)
#      -- Test 85'in sweep'inde bu değerlerin ("RESTORING FORCE" modunda)
#      en zengin/en spesifik çıktıyı ürettiği gözlenmişti (istatistiksel
#      çeşitlilik zirvesi 0.35'teydi, ama operasyonel detay zirvesi 1.00'daydı).
#      -- Test 85's sweep found these values (under "RESTORING FORCE" mode)
#      produced the richest/most specific output (the statistical-diversity
#      peak was at 0.35, but the operational-detail peak was at 1.00).
#
#   2) FİZİK MODU DEĞİŞTİ: REZONANS, ÇEKİCİ DEĞİL / PHYSICS CHANGED: RESONANCE, NOT ATTRACTOR:
#      "RESTORING FORCE" (Test 85, AkbasCore 1.2) = katkı, cos(theta)'nın
#      işaretini TAKİP eder = mevcut kodun "Rezonans" fiziği (attractor_mode=0.0).
#      Bu, Test 88-92'de kullanılan "Çekici/attractor" (1.0) İLE AYNI DEĞİL.
#      Bu eşleştirme yanlışsa deneyin öncülü geçersiz olur -- açıkça belirtildi.
#      "RESTORING FORCE" (Test 85, AkbasCore 1.2) = contribution FOLLOWS the
#      sign of cos(theta) = the current codebase's "Rezonans" physics
#      (attractor_mode=0.0). This is NOT the same as "Çekici/attractor" (1.0)
#      used in Tests 88-92. If this mapping is wrong, the experiment's premise
#      is invalid -- stated explicitly.
#
#   3) KRİTİK GÜVENLİK DÜZELTMESİ / CRITICAL SAFETY FIX:
#      IVME_MAX önceden 0.80'di; ivme=1.00 bu sınırı SESSİZCE aşardı ve
#      run() içindeki kırpma satırı onu fark ettirmeden 0.80'e indirirdi.
#      IVME_MAX artık 1.00 -- gerçekten istenen değer uygulanıyor.
#      IVME_MAX was previously 0.80; ivme=1.00 would have SILENTLY exceeded
#      it and been clamped down to 0.80 without warning inside run(). IVME_MAX
#      is now 1.00 -- the intended value is actually applied.
#
#   4) IŞIK/KARANLIK EKSENLERİ — KAPI KALDIRILDI / LIGHT-DARK AXES — GATE REMOVED:
#      Eksen Gücü 0.50->1.00 (axis0 ivme'siyle eşitlendi). Merkez, L0'dan
#      ÖNCEYE (center_ratio=-0.5, gerçek katman=-10) çekildi -- bu, "geç açılan
#      kapı" mekanizmasını matematiksel olarak devre dışı bırakıyor (kapı,
#      katman 0'a gelmeden tamamen açık): curve(L)=1.0 tüm 20 katmanda,
#      SAYISAL OLARAK doğrulandı. Sonuç: ışık/karanlık artık axis0 ile AYNI
#      DRA sönüm eğrisinin TAMAMINI (zirveden itibaren) yaşıyor -- önceki
#      "sadece geç katmanlarda, sönmüş/düz kısmı görme" sorunu ortadan kalktı.
#      Kernel'e HİÇ dokunulmadı, sadece parametre seçimiyle başarıldı.
#      Axis strength 0.50->1.00 (matched to axis0's ivme). Center pulled to
#      BEFORE L0 (center_ratio=-0.5, absolute layer=-10) -- this mathematically
#      disables the "late-opening gate" mechanism (the gate is fully open
#      before layer 0 is even reached): curve(L)=1.0 at all 20 layers,
#      NUMERICALLY verified. Result: light/dark now experience the ENTIRE DRA
#      decay curve (from its peak) exactly like axis0 does -- the prior issue
#      of "only seeing the late, already-decayed flat portion" is eliminated.
#      The kernel itself was NOT touched -- achieved through parameter choice
#      alone.
#
#   5) BÖLÜM B 2 KAT BÜYÜTÜLDÜ / PART B DOUBLED:
#      MMLU 120->240, GSM8K 80->160. AYNI seed=42 kullanıldığı için, ilk
#      120 MMLU / ilk 80 GSM8K sorusu Test 92 ile BİREBİR AYNI (shuffle sırası
#      N'den bağımsız, sadece select(range(N)) değişiyor) -- bu, aynı sorularda
#      "ayar değişince ne değişti" karşılaştırmasını temiz şekilde mümkün kılıyor.
#      Same seed=42 means the first 120 MMLU / first 80 GSM8K items are
#      IDENTICAL to Test 92 (shuffle order is independent of N, only
#      select(range(N)) changes) -- enabling a clean "what changed when the
#      settings changed" comparison on the exact same items.
#
#   6) BÖLÜM A: AYNI 50 SORU / PART A: SAME 50 QUESTIONS as Test 92, for the
#      same paired-comparison reason.
#
# ZAMAN BÜTÇESİ UYARISI / TIME BUDGET WARNING:
#   Bölüm B'nin 2 katına çıkması, toplam çalışma süresini ~4 saatlik güvenilir
#   tek-oturum Colab penceresinin ÜZERİNE çıkarabilir (tahmini ~5-5.5 saat).
#   Script HER SORUDAN SONRA diske yazar (ilerleme kaybolmaz), ama GERÇEK bir
#   "kaldığı yerden devam et" özelliği YOKTUR -- bağlantı koparsa, o ana kadarki
#   çıktı dosyada güvende kalır ama yeniden çalıştırma en baştan başlar.
#   Doubling Part B may push total runtime BEYOND the reliable ~4-hour single-
#   session Colab window (estimated ~5-5.5 hours). The script writes to disk
#   AFTER EVERY question (no progress is lost), but there is NO true "resume
#   from where it left off" -- a disconnect preserves output so far in the
#   file, but re-running starts from the beginning again.
#
# COLAB'DA ÇALIŞTIRMA / HOW TO RUN ON COLAB: aynı adımlar, Test 92 ile aynı /
# same steps as Test 92.
# =============================================================================

import subprocess, sys, os, shutil, time, gc, math

def _pip(pkg, no_deps=False):
    cmd = [sys.executable,"-m","pip","install",pkg,"-q"]
    if no_deps:
        cmd.append("--no-deps")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        print("❌ GPU DONANIM OLARAK VAR ama PyTorch onu göremiyor.")
        print("   ÇÖZÜM: Runtime → Disconnect and delete runtime, tekrar bağlan.")
    else:
        print("❌ GPU BULUNAMADI. ÇÖZÜM: Runtime → Change runtime type → T4 GPU.")
    print("!"*70 + "\n")
    raise SystemExit("GPU kullanılamıyor.")
else:
    print(f"✅ GPU tespit edildi: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)\n")

os.environ["CUDA_LAUNCH_BLOCKING"]    = "0"
# ── TEST 90 — TEK DEĞİŞKENLİ DENEY / SINGLE-VARIABLE EXPERIMENT ─────────────
# Bu, Test 90'ın TEK değişikliği. Test 88 ve 89'da bu değişken "1" olarak
# ayarlanmıştı (erken geliştirme sürecinde hata ayıklama için eklenmiş bir
# güvenlik önlemi) -- bu, programdaki HER CUDA kernel çağrısını (sadece
# steering kernel'imizi değil, modelin kendi matmul/attention/layernorm
# kernellerini de) zorla senkron yapıyordu. Test 89'da CUDA kernel'imiz
# kendi kendini doğrulayan bir test geçti; artık bu güvenlik ağına daha az
# ihtiyacımız var, riski bilerek göze alıyoruz.
# Hook mimarisine, kernel matematiğine, doğrulama kapısına, CPU yedek yoluna
# HİÇBİR ŞEY dokunulmadı -- bu satır dışında Test 89 ile birebir aynı.
# Bunun etkisini ÖLÇMEDEN önce bir sayı tahmin etmiyoruz: script'in kendi
# tok/s çıktısını Test 89'un ~7-8 tok/s'siyle karşılaştırın, tek doğru cevap
# odur.
# This is Test 90's ONLY change. In Tests 88 and 89 this variable was set to
# "1" (added early on as a debugging safeguard) -- forcing EVERY CUDA kernel
# launch in the process (not just our steering kernel, but the base model's
# own matmul/attention/layernorm kernels too) to run synchronously. Test 89's
# CUDA kernel passed its own self-verification test; we now need that safety
# net less, and are knowingly trading it for speed.
# Nothing else -- hook architecture, kernel math, verification gate, CPU
# fallback path -- was touched. This is otherwise byte-for-byte Test 89.
# We do not guess a number before measuring: compare this script's own
# tok/s output against Test 89's ~7-8 tok/s. That comparison is the answer,
# not a prediction.
# NOT KAYBI: CUDA_LAUNCH_BLOCKING=1'in orijinal amacı, bir CUDA hatası
# oluştuğunda doğru satırı gösteren net bir hata mesajı almaktı. Bunu "0"
# yaparsak, ileride gerçek bir CUDA hatası olursa hata mesajı yanlış yeri
# gösterebilir. Kernel zaten doğrulandığı için bu riski şimdi kabul ediyoruz.
# TRADE-OFF: CUDA_LAUNCH_BLOCKING=1's original purpose was accurate error
# locality if a CUDA error ever occurs. Setting it to "0" means a future
# real CUDA error's stack trace may point to the wrong line. We accept this
# now because the kernel is already verified.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
torch.backends.cudnn.deterministic    = True
torch.backends.cudnn.benchmark        = False

# =============================================================================
# ÇEKİRDEK 1/2 — CPU KERNEL (DEĞİŞMEDİ) — hem doğrulama referansı hem yedek yol
# KERNEL 1/2 — CPU KERNEL (UNCHANGED) — both the verification reference AND
# the automatic fallback path
# =============================================================================
_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

torch::Tensor akbas_orthogonalize(torch::Tensor compass_matrix) {
    TORCH_CHECK(!compass_matrix.is_cuda(),
        "AkbasCore cekirdegi CUDA tensoruyle cagrildi.");
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

static inline float layer_curve_cpu(int layer_idx, float center, float width) {
    if (width < 1e-6f) return 1.0f;
    float x = ((float)layer_idx - (center - width * 0.5f)) / width;
    return std::clamp(x, 0.0f, 1.0f);
}

static inline float axis_katki_cpu(
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
    TORCH_CHECK(!hidden.is_cuda() && !compass_matrix.is_cuda(), "CUDA tensoru tespit edildi.");
    auto h=hidden.contiguous(); auto cm=compass_matrix.contiguous();
    auto al=alphas.contiguous(); auto ce=centers.contiguous(); auto wi=widths.contiguous();
    auto pc=prev_cosine.contiguous(); auto lb=log_buf.contiguous(); auto lc=log_count.contiguous();
    const int B=h.size(0), S=h.size(1), D=h.size(2), K=cm.size(0), N_LAYERS=lb.size(1);
    if (layer_idx >= N_LAYERS) return h;
    float *hp=h.data_ptr<float>(); const float *cmp=cm.data_ptr<float>();
    const float *alp=al.data_ptr<float>(); const float *cep=ce.data_ptr<float>();
    const float *wip=wi.data_ptr<float>(); float *pcp=pc.data_ptr<float>();
    float *lbp=lb.data_ptr<float>(); float *lcp=lc.data_ptr<float>();
    for (int b=0;b<B;++b) for (int s=0;s<S;++s) {
        float *tok=hp+(b*S*D)+(s*D); int idx=b*S+s;
        float sq=0.0f; for (int d=0;d<D;++d) sq+=tok[d]*tok[d];
        float norm_hint=sqrtf(sq)+1e-6f;
        for (int k=0;k<K;++k) {
            float curve=(k==0)?1.0f:layer_curve_cpu(layer_idx,cep[k],wip[k]);
            float cs,kb,kv; float &pcv=pcp[k*(B*S)+idx];
            float katki=axis_katki_cpu(tok,cmp+k*D,D,norm_hint,alp[k],layer_idx,
                                        sonum,zirve,taban,attractor_mode,pcv,cs,kb,kv)*curve;
            if (b==0 && S==1) { float *slot=lbp+k*N_LAYERS*4+layer_idx*4;
                slot[0]+=cs; slot[1]+=kb; slot[2]+=kv; slot[3]+=katki; }
        }
        if (b==0 && S==1) lcp[layer_idx]+=1.0f;
    }
    return h;
}

torch::Tensor akbas_steer_multi(
    torch::Tensor hidden, torch::Tensor compass_matrix, torch::Tensor alphas,
    torch::Tensor centers, torch::Tensor widths, int layer_idx,
    float sonum, float zirve, float taban, float attractor_mode,
    torch::Tensor prev_cosine, torch::Tensor log_buf, torch::Tensor log_count
) {
    TORCH_CHECK(!hidden.is_cuda() && !compass_matrix.is_cuda(), "CUDA tensoru tespit edildi.");
    auto h=hidden.contiguous(); auto cm=compass_matrix.contiguous();
    auto al=alphas.contiguous(); auto ce=centers.contiguous(); auto wi=widths.contiguous();
    auto pc=prev_cosine.contiguous(); auto lb=log_buf.contiguous(); auto lc=log_count.contiguous();
    const int B=h.size(0), S=h.size(1), D=h.size(2), K=cm.size(0), N_LAYERS=lb.size(1);
    if (layer_idx >= N_LAYERS) return h;
    float *hp=h.data_ptr<float>(); const float *cmp=cm.data_ptr<float>();
    const float *alp=al.data_ptr<float>(); const float *cep=ce.data_ptr<float>();
    const float *wip=wi.data_ptr<float>(); float *pcp=pc.data_ptr<float>();
    float *lbp=lb.data_ptr<float>(); float *lcp=lc.data_ptr<float>();
    for (int b=0;b<B;++b) for (int s=0;s<S;++s) {
        float *tok=hp+(b*S*D)+(s*D); int idx=b*S+s;
        for (int k=0;k<K;++k) {
            float sq=0.0f; for (int d=0;d<D;++d) sq+=tok[d]*tok[d];
            float norm_hint=sqrtf(sq)+1e-6f;
            float curve=(k==0)?1.0f:layer_curve_cpu(layer_idx,cep[k],wip[k]);
            float cs,kb,kv; float &pcv=pcp[k*(B*S)+idx]; const float *axis_vec=cmp+k*D;
            float katki=axis_katki_cpu(tok,axis_vec,D,norm_hint,alp[k],layer_idx,
                                        sonum,zirve,taban,attractor_mode,pcv,cs,kb,kv)*curve;
            for (int d=0;d<D;++d) tok[d]+=katki*axis_vec[d];
            if (b==0 && S==1) { float *slot=lbp+k*N_LAYERS*4+layer_idx*4;
                slot[0]+=cs; slot[1]+=kb; slot[2]+=kv; slot[3]+=katki; }
        }
        if (b==0 && S==1) lcp[layer_idx]+=1.0f;
    }
    return h;
}
"""

# =============================================================================
# ÇEKİRDEK 2/2 — YENİ: GPU-NATIVE CUDA KERNEL (TEST 90, DEĞİŞMEDİ)
# KERNEL 2/2 — NEW: GPU-NATIVE CUDA KERNEL (TEST 90, UNCHANGED)
# Tek eksen uygular; K eksen, Python tarafından ARDIŞIK olarak (aynı CUDA
# stream'inde, bu yüzden sıralama otomatik garanti) çağrılır -- CPU
# kernel'deki "her eksen bir öncekinin değiştirdiği hali görür" davranışını
# birebir korur. Hiçbir adımda .cpu() çağrılmaz.
# Applies a single axis; K axes are called SEQUENTIALLY from Python (on the
# same CUDA stream, so ordering is automatically guaranteed) -- exactly
# preserving the CPU kernel's "each axis sees the previous axis's already-
# modified state" behavior. No .cpu() call anywhere in this path.
# =============================================================================
_CUDA_CPP = r"""
#include <torch/extension.h>
torch::Tensor akbas_apply_axis_cuda(
    torch::Tensor hidden, torch::Tensor axis_vec, int layer_idx,
    double alpha, double sonum, double zirve, double taban, double attractor_mode,
    double curve, torch::Tensor prev_cosine, bool apply_steer, bool do_log,
    torch::Tensor log_slot, torch::Tensor log_count);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define AKBAS_BLOCK 256

__device__ __forceinline__ float akbas_katki_device(
    const float *tok, const float *axis_vec, float norm_hint,
    float alpha, int layer_idx, float sonum, float zirve, float taban,
    float attractor_mode, float pcv_in, float &pcv_out,
    float &out_cs, float &out_kb, float &out_kv, float dot
) {
    float cs = fminf(1.0f, fmaxf(-1.0f, dot / norm_hint));
    float unc = 1.0f - fabsf(cs);
    float dw  = sonum + unc * 0.20f;
    float t   = (float)layer_idx;
    float kb  = zirve * expf(-dw * t) * (1.0f + dw * t) + taban;
    float dr = fminf(0.15f, fmaxf(-0.15f, cs - pcv_in));
    if (cs > 0.80f && dr < 0.0f) dr *= 0.30f;
    float kv = kb;
    if      (dr > 0) kv *= (1.0f - dr    * 0.30f);
    else if (dr < 0) kv *= (1.0f + (-dr) * 0.30f);
    kv = fminf(1.0f, fmaxf(0.05f, kv));
    pcv_out = cs;
    float mk = fminf(0.20f, fmaxf(0.04f, norm_hint * 0.045f));
    float son = 1.0f;
    if      (cs >  0.75f) son = (1.0f - cs) / 0.25f;
    else if (cs < -0.40f) son = 1.6f;
    float katki_rezonans = alpha * cs * kv * 0.32f * son;
    float ang_err = (1.0f - cs) * 0.5f;
    float katki_cekici = alpha * kv * 0.32f * ang_err * 2.0f;
    float m = fminf(1.0f, fmaxf(0.0f, attractor_mode));
    float katki = fminf(mk, fmaxf(-mk, (1.0f - m) * katki_rezonans + m * katki_cekici));
    out_cs = cs; out_kb = kb; out_kv = kv;
    return katki;
}

// Bir blok = bir token (b*S+s). AKBAS_BLOCK thread, D boyutu üzerinde
// grid-stride ile işbirliği yapar; dot-product ve norm-kare için standart
// paylaşımlı-bellek ağaç indirgemesi (shared-memory tree reduction) kullanır.
// One block = one token. AKBAS_BLOCK threads cooperate over the D dimension
// via a grid-stride loop; standard shared-memory tree reduction for the
// dot-product and squared-norm.
__global__ void akbas_single_axis_kernel(
    float* __restrict__ hidden,          // [BS, D] contiguous, float32
    const float* __restrict__ axis_vec,  // [D]
    int D, int layer_idx,
    float alpha, float sonum, float zirve, float taban,
    float attractor_mode, float curve,
    float* __restrict__ prev_cosine,     // [BS]
    bool apply_steer, bool do_log,
    float* __restrict__ log_slot,        // [4]: cs, kb, kv, katki accumulators
    float* __restrict__ log_count        // [1]
) {
    int token = blockIdx.x;
    float* tok = hidden + (size_t)token * D;
    __shared__ float sdata[AKBAS_BLOCK];
    int tid = threadIdx.x;

    float partial_dot = 0.0f, partial_sq = 0.0f;
    for (int d = tid; d < D; d += AKBAS_BLOCK) {
        float tv = tok[d];
        partial_dot += tv * axis_vec[d];
        partial_sq  += tv * tv;
    }

    sdata[tid] = partial_dot; __syncthreads();
    for (int s = AKBAS_BLOCK/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    float dot = sdata[0]; __syncthreads();

    sdata[tid] = partial_sq; __syncthreads();
    for (int s = AKBAS_BLOCK/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    float sq = sdata[0]; __syncthreads();
    float norm_hint = sqrtf(sq) + 1e-6f;

    __shared__ float s_katki;
    if (tid == 0) {
        float pcv_in = prev_cosine[token];
        float pcv_out, cs, kb, kv;
        float katki = akbas_katki_device(tok, axis_vec, norm_hint, alpha, layer_idx,
                                          sonum, zirve, taban, attractor_mode,
                                          pcv_in, pcv_out, cs, kb, kv, dot) * curve;
        prev_cosine[token] = pcv_out;
        s_katki = katki;
        if (do_log && token == 0) {
            atomicAdd(&log_slot[0], cs);
            atomicAdd(&log_slot[1], kb);
            atomicAdd(&log_slot[2], kv);
            atomicAdd(&log_slot[3], katki);
            // log_count BURADA artırılmıyor -- bu kernel her eksen için ayrı
            // çağrıldığından, buradaki bir atomicAdd K eksende K kat sayardı.
            // Sayma artık Python tarafında, katman başına SADECE BİR KEZ
            // (k==0 iken) yapılıyor -- CPU kernel'indeki "for k döngüsünün
            // DIŞINDA" davranışıyla birebir eşleşsin diye.
            // log_count is NOT incremented here -- since this kernel is
            // called once per axis, an atomicAdd here would count K times
            // for K axes. Counting now happens in Python, exactly once per
            // layer per token (only when k==0), to exactly match the CPU
            // kernel's "outside the for-k loop" behavior.
        }
    }
    __syncthreads();
    float katki = s_katki;

    if (apply_steer) {
        for (int d = tid; d < D; d += AKBAS_BLOCK) {
            tok[d] += katki * axis_vec[d];
        }
    }
}

torch::Tensor akbas_apply_axis_cuda(
    torch::Tensor hidden, torch::Tensor axis_vec, int layer_idx,
    double alpha, double sonum, double zirve, double taban, double attractor_mode,
    double curve, torch::Tensor prev_cosine, bool apply_steer, bool do_log,
    torch::Tensor log_slot, torch::Tensor log_count
) {
    TORCH_CHECK(hidden.is_cuda(), "akbas_apply_axis_cuda: hidden CUDA tensoru olmali.");
    TORCH_CHECK(hidden.is_contiguous(), "akbas_apply_axis_cuda: hidden contiguous olmali.");
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32, "akbas_apply_axis_cuda: hidden float32 olmali.");
    int B = hidden.size(0), S = hidden.size(1), D = hidden.size(2);
    int BS = B * S;
    akbas_single_axis_kernel<<<BS, AKBAS_BLOCK>>>(
        hidden.data_ptr<float>(), axis_vec.data_ptr<float>(), D, layer_idx,
        (float)alpha, (float)sonum, (float)zirve, (float)taban, (float)attractor_mode,
        (float)curve, prev_cosine.data_ptr<float>(), apply_steer, do_log,
        log_slot.data_ptr<float>(), log_count.data_ptr<float>()
    );
    return hidden;
}
"""

if not torch.utils.cpp_extension.is_ninja_available():
    import site
    for _sp in site.getsitepackages():
        _b=os.path.normpath(os.path.join(_sp,"..","..","bin"))
        if os.path.isdir(_b): os.environ["PATH"]=_b+":"+os.environ.get("PATH","")

for _n in ["akbas_t90_cpu1","akbas_t90_cpu2"]:
    for _r in ["/tmp/akbas_t90",os.path.expanduser("~/.cache/torch_extensions")]:
        _p=os.path.join(_r,_n)
        if os.path.exists(_p): shutil.rmtree(_p,ignore_errors=True)

def _build_cpu(name):
    d=f"/tmp/akbas_t90/{name}"; os.makedirs(d,exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name, cpp_sources=_CPP,
        functions=["akbas_orthogonalize","akbas_observe_multi","akbas_steer_multi"],
        build_directory=d, verbose=False, extra_cflags=["-O2","-std=c++17"])

try:    _kernel=_build_cpu("akbas_t90_cpu1")
except Exception as e:
    print(f"  retry ({e})..."); _kernel=_build_cpu("akbas_t90_cpu2")
print("✅ AkbasCore 2.0 — CPU çekirdeği derlendi (referans + yedek yol)")

_cuda_kernel = None
_cuda_build_error = None
try:
    # T4 (compute capability 7.5) gibi ortamlarda PyTorch'un mimari
    # otomatik-algılaması bazen başarısız olup gereksiz bir derleme hatasına
    # yol açabiliyor. Mevcut GPU'nun gerçek compute capability'sini burada
    # açıkça belirterek bu riski ortadan kaldırıyoruz.
    # Architecture auto-detection can occasionally fail in environments like
    # T4 (compute capability 7.5), causing a needless build failure. We
    # eliminate that risk by explicitly setting the current GPU's actual
    # compute capability here.
    _cap = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_cap[0]}.{_cap[1]}")
    print(f"   CUDA hedef mimari (compute capability): {_cap[0]}.{_cap[1]}")

    _d = "/tmp/akbas_t90/cuda1"
    if os.path.exists(_d): shutil.rmtree(_d, ignore_errors=True)
    os.makedirs(_d, exist_ok=True)
    _cuda_kernel = torch.utils.cpp_extension.load_inline(
        name="akbas_t90_cuda1", cpp_sources=_CUDA_CPP, cuda_sources=_CUDA_SRC,
        functions=["akbas_apply_axis_cuda"], build_directory=_d, verbose=False,
        extra_cflags=["-O2","-std=c++17"])
    print("✅ AkbasCore 2.0 — GPU-native CUDA çekirdeği DERLENDİ (henüz doğrulanmadı)")
except Exception as e:
    _cuda_build_error = str(e)
    print(f"⚠️ CUDA çekirdeği derlenemedi: {e}")
    print("   Sorun değil -- CPU yoluna güvenle devam edilecek.")

# =============================================================================
# ZORUNLU OTOMATİK DOĞRULAMA — CUDA çekirdeği CPU çekirdeğiyle eşleşiyor mu?
# MANDATORY AUTOMATIC VERIFICATION — does the CUDA kernel match the CPU kernel?
# =============================================================================
CUDA_KERNEL_VERIFIED = False
if _cuda_kernel is not None:
    print("\n" + "-"*70)
    print("CUDA KERNEL DOĞRULAMA / CUDA KERNEL VERIFICATION")
    print("-"*70)
    try:
        torch.manual_seed(42)
        D_test = 128
        hidden_ref  = torch.randn(1, 1, D_test, dtype=torch.float32)
        axis_ref    = torch.nn.functional.normalize(torch.randn(D_test), dim=0)
        alpha, sonum, zirve, taban, attractor_mode, curve = 0.5, 0.3, 0.7, 0.2, 1.0, 1.0
        layer_idx = 5

        # --- CPU referans (K=1, tek eksen, tek katman) ---
        hidden_cpu = hidden_ref.clone()
        compass_cpu = axis_ref.view(1, D_test).clone()
        alphas_cpu  = torch.tensor([alpha])
        centers_cpu = torch.tensor([0.0])
        widths_cpu  = torch.tensor([0.0])
        pc_cpu      = torch.zeros(1, 1)
        log_buf_cpu = torch.zeros(1, layer_idx+1, 4)
        log_cnt_cpu = torch.zeros(layer_idx+1)
        st_cpu = _kernel.akbas_steer_multi(
            hidden_cpu, compass_cpu, alphas_cpu, centers_cpu, widths_cpu,
            layer_idx, sonum, zirve, taban, attractor_mode,
            pc_cpu, log_buf_cpu, log_cnt_cpu)

        # --- CUDA aday (aynı girdi, GPU'da) ---
        hidden_cuda = hidden_ref.clone().cuda()
        axis_cuda   = axis_ref.clone().cuda()
        pc_cuda     = torch.zeros(1, device="cuda")
        log_slot_cuda = torch.zeros(4, device="cuda")
        log_cnt_cuda  = torch.zeros(1, device="cuda")
        st_cuda = _cuda_kernel.akbas_apply_axis_cuda(
            hidden_cuda, axis_cuda, layer_idx, alpha, sonum, zirve, taban,
            attractor_mode, curve, pc_cuda, True, True, log_slot_cuda, log_cnt_cuda)
        torch.cuda.synchronize()

        max_diff = (st_cpu.cpu() - st_cuda.cpu()).abs().max().item()
        print(f"   Test verisi: D={D_test}, layer_idx={layer_idx}")
        print(f"   CPU vs CUDA sonucu arasındaki maksimum mutlak fark: {max_diff:.2e}")
        TOLERANCE = 1e-3
        if max_diff < TOLERANCE:
            CUDA_KERNEL_VERIFIED = True
            print(f"✅ DOĞRULANDI (tolerans {TOLERANCE:.0e} altında) — CUDA yolu KULLANILACAK.")
        else:
            print(f"❌ DOĞRULANAMADI (tolerans {TOLERANCE:.0e} aşıldı) — CPU yoluna GÜVENLE geri dönülüyor.")
            print("   Bu mesajı olduğu gibi paylaşırsanız birlikte kernel'i düzeltebiliriz.")
    except Exception as e:
        print(f"❌ Doğrulama sırasında hata: {e}")
        print("   CPU yoluna güvenle geri dönülüyor.")
    print("-"*70 + "\n")

print(f"🔀 Bu çalıştırmada kullanılacak yol: {'GPU-NATIVE CUDA (Test 90, launch-blocking kaldırıldı)' if CUDA_KERNEL_VERIFIED else 'CPU (pinned buffer, güvenli geri dönüş)'}\n")

# =============================================================================
# SABİT PARAMETRELER
# =============================================================================
DEFAULT_TEMPERATURE = 0.65
TEMP_MIN, TEMP_MAX  = 0.0, 1.50
TOP_P, TOP_K, REP_PENALTY = 0.90, 50, 1.15
MAX_EXTRA_AXES = 3
DEFAULT_HOOK_RATIO = 0.70

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

# ── TEST 93 — TEST 85'TEN TAŞINAN AYARLAR (RESTORING FORCE → REZONANS) ──────
# ── TEST 93 — SETTINGS PORTED FROM TEST 85 (RESTORING FORCE → RESONANCE) ───
# Test 85 (AkbasCore 1.2, Manifold'suz, Qwen2.5-1.5B), RESTORING FORCE modunda
# ivme=1.00 sönüm=0.10 zirve=1.00 taban=0.40'ta en zengin/en spesifik çıktıyı
# üretmişti (istatistiksel çeşitlilik zirvesi değil -- o 0.35'teydi -- ama
# operasyonel detay zirvesi). "RESTORING FORCE", katkının cos(theta) yönünü
# TAKİP ettiği bir fizik -- bu, mevcut kodun "Rezonans" fiziğiyle (attractor_
# mode=0.0) birebir aynı, "Çekici/attractor" (1.0, Test 88-92'de kullanılan)
# İLE AYNI DEĞİL. Bu eşleştirme yanlışsa deneyin öncülü geçersiz olur -- bu
# yüzden burada açıkça belgeleniyor.
# KRİTİK DÜZELTME: IVME_MAX önceden 0.80'di -- ivme=1.00 bu sınırı aşıyordu
# ve run() içindeki kırpma satırı bunu SESSİZCE 0.80'e indirirdi. Aşağıda
# IVME_MAX=1.00 yapılarak bu sessiz kırpma önlendi.
# Test 85 (AkbasCore 1.2, no Manifold, Qwen2.5-1.5B) produced its richest/
# most operationally specific output (not the statistical-diversity peak --
# that was at 0.35 -- but the operational-detail peak) under RESTORING FORCE
# mode at ivme=1.00 sönüm=0.10 zirve=1.00 taban=0.40. "RESTORING FORCE" is a
# physics where the contribution FOLLOWS the sign of cos(theta) -- this is
# exactly the current codebase's "Rezonans" physics (attractor_mode=0.0), NOT
# "Çekici/attractor" (1.0, used throughout Tests 88-92). If this mapping is
# wrong, the experiment's premise is invalid -- documented explicitly here.
# CRITICAL FIX: IVME_MAX was previously 0.80 -- ivme=1.00 exceeded it, and
# the clamp line in run() would have SILENTLY reduced it to 0.80. Fixed below
# by raising IVME_MAX to 1.00 to prevent this silent clamp.
DEFAULT_IVME, DEFAULT_SONUM, DEFAULT_ZIRVE, DEFAULT_TABAN = 1.00, 0.10, 1.00, 0.40
IVME_MIN,  IVME_MAX  = 0.20, 1.00
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
DEFAULT_AXIS_ALPHA = 1.00     # Test 93: ışık/karanlık gücü axis0 ivme'siyle eşitlendi (was 0.50)
DEFAULT_CENTER_RATIO = -0.5   # Test 93 (DÜZELTİLDİ): kapı L0'dan ÖNCE tamamen açık -- ışık/karanlık
                              # artık axis0 ile AYNI DRA eğrisinin TAMAMINI (zirveden itibaren)
                              # yaşıyor, "geç açılan kapı" YOK. Matematik: cen_val + genişlik/2 <= 0
                              # olduğu sürece curve(L)=1.0 her katmanda -- kernel'e HİÇ dokunulmadı.
                              # Test 93 (CORRECTED): the gate is fully open BEFORE L0 -- light/dark
                              # now experience the ENTIRE DRA curve (from its peak) exactly like
                              # axis0, with NO "opens late" gate. Math: as long as
                              # cen_val + width/2 <= 0, curve(L)=1.0 at every layer -- the kernel
                              # itself was NOT touched.
DEFAULT_WIDTH_RATIO  = 0.30   # değişmedi / unchanged (genişlik=6)
DEFAULT_ATTRACTOR = 0.0       # Test 93: REZONANS modu (RESTORING FORCE eşleniği), Test 88-92'deki 1.0 (çekici) DEĞİL

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
    raise RuntimeError("Model mimarisi tanınmadı (katman listesi bulunamadı).")

def _find_embed_tokens(model):
    for path in _EMBED_PATHS:
        try: return _resolve_path(model, path)
        except AttributeError: continue
    raise RuntimeError("Model mimarisi tanınmadı (embedding tablosu bulunamadı).")

# =============================================================================
# AkbasCore 2.0 — Test 90 (Test 89'un doğrulanmış CUDA çekirdeği, değişmedi;
# tek fark CUDA_LAUNCH_BLOCKING kaldırıldı) + doğrulanmış güvenli CPU yedek
# =============================================================================
class AkbasCore:

    def __init__(self, model_id, hook_ratio=DEFAULT_HOOK_RATIO, use_4bit=False,
                 hf_token=None, progress_cb=None):
        def _p(msg):
            print(msg)
            if progress_cb: progress_cb(msg)

        _p(f"🔱⬜ AKBASCORE 2.0 — Test 90 | {model_id}")
        self.model_id = model_id

        if hf_token and hf_token.strip():
            try:
                from huggingface_hub import login as _hf_login
                _hf_login(token=hf_token.strip(), add_to_git_credential=False)
            except Exception as e:
                _p(f"   ⚠️ HF token girişi başarısız: {e}")

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
        else:
            _p("   Device: CPU")

        self.tok=AutoTokenizer.from_pretrained(model_id, token=hf_token or None)
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token

        _p("   Model indiriliyor/yükleniyor...")
        _kw = dict(device_map="auto", trust_remote_code=True,
                   low_cpu_mem_usage=True, token=(hf_token or None))
        if quant_cfg is not None: _kw["quantization_config"] = quant_cfg
        else: _kw["dtype"] = compute_dtype
        self.model=AutoModelForCausalLM.from_pretrained(model_id, **_kw)
        if hasattr(self.model.config,"_attn_implementation"):
            self.model.config._attn_implementation="eager"
        self.device=next(self.model.parameters()).device
        self.hidden_dim=self.model.config.hidden_size

        self._layers = _find_layers(self.model)
        self._embed_tokens = _find_embed_tokens(self.model)
        self.total_layers = len(self._layers)
        self.n_layers = max(1, min(self.total_layers, round(self.total_layers * hook_ratio)))
        _p(f"   Toplam katman: {self.total_layers}  |  Hook'lanacak: {self.n_layers}  |  "
           f"Gizli boyut: {self.hidden_dim}")
        _p(f"   Aktif yol: {'GPU-NATIVE CUDA' if CUDA_KERNEL_VERIFIED else 'CPU (pinned buffer)'}")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        _p("   Anayasa ekseni (axis 0) inşa ediliyor...")
        self._anayasa=self._build_anayasa_pusula()
        self._token_norms=None

        gc.collect()
        _p(f"✅ AkbasCore hazır — {model_id}\n")

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
        return F.normalize(BILISIM_ETIK*(wsum/wtot) + BILISIM_MANTIK*logic, dim=0).contiguous()

    def _build_diff_axis(self, pos_text, neg_text):
        pos_words=[w.strip() for w in pos_text.replace("\n",",").split(",")]
        neg_words=[w.strip() for w in neg_text.replace("\n",",").split(",")]
        mp=self._mean_vec(pos_words); mn=self._mean_vec(neg_words)
        if mp is None or mn is None: return None
        return F.normalize(mp-mn, dim=0).contiguous()

    def _build_sample_axis(self, method, seed, sample_count):
        W=self._embed_tokens.weight; vocab=W.shape[0]
        sample_count=max(2,min(int(sample_count),vocab))
        g=torch.Generator(device='cpu').manual_seed(int(seed))
        if method==TOKEN_SAMPLE_METHODS[1]:
            if self._token_norms is None: self._token_norms=W.float().norm(dim=1).cpu()
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

    def _build_axis_candidate(self, mode, pos_text, neg_text, sample_method, seed, sample_count, words_text):
        if mode==AXIS_MODES[0]: return self._build_diff_axis(pos_text, neg_text)
        elif mode==AXIS_MODES[1]: return self._build_sample_axis(sample_method, seed, sample_count)
        else: return self._build_wordset_axis(words_text)

    def _remove_hooks(self,hooks):
        for h in hooks: h.remove()

    # ── CUDA-NATIVE HOOK YOLU (doğrulanmışsa kullanılır) ────────────────────
    # ── CUDA-NATIVE HOOK PATH (used only if verified) ───────────────────────
    def _inject_cuda(self, compass_matrix, alphas, centers, widths,
                      log_buf_gpu, log_count_gpu, pc_holder, sonum, zirve, taban,
                      attractor_mode, mode="observe"):
        hooks=[]
        K = compass_matrix.shape[0]
        compass_gpu = compass_matrix.detach().contiguous()  # zaten GPU'da / already on GPU
        alphas_l  = alphas.tolist()
        centers_l = centers.tolist()
        widths_l  = widths.tolist()
        apply_steer = (mode != "observe")
        _fp32_buf = {"tensor": None}

        def make_hook(i):
            def hook(mod,inp,out):
                hs=out[0] if isinstance(out,tuple) else out
                orig_device=hs.device
                od=hs.dtype
                B,S,D=hs.shape
                bs=B*S
                do_log = (S == 1)

                # GPU üzerinde dtype dönüşümü -- HİÇ CPU'ya gitmiyor.
                # On-GPU dtype cast -- never touches the CPU.
                buf = _fp32_buf["tensor"]
                if buf is None or tuple(buf.shape) != (B,S,D):
                    buf = torch.empty((B,S,D), dtype=torch.float32, device=orig_device)
                    _fp32_buf["tensor"] = buf
                buf.copy_(hs.detach())

                if pc_holder["pc"] is None or pc_holder["pc"].shape[1]!=bs:
                    pc_holder["pc"]=torch.zeros(K,bs,dtype=torch.float32,device=orig_device)

                for k in range(K):
                    curve = 1.0 if k==0 else max(0.0, min(1.0,
                             (i - (centers_l[k]-widths_l[k]/2.0))/widths_l[k] if widths_l[k]>1e-6 else 1.0))
                    log_slot = log_buf_gpu[k, i]  # [4] view, in-place accumulation
                    _cuda_kernel.akbas_apply_axis_cuda(
                        buf, compass_gpu[k].contiguous(), i,
                        float(alphas_l[k]), float(sonum), float(zirve), float(taban),
                        float(attractor_mode), float(curve), pc_holder["pc"][k],
                        apply_steer, do_log, log_slot, log_count_gpu[i:i+1])

                    if do_log and k == 0:
                        # Katman başına SADECE BİR KEZ sayılıyor (hangi eksen
                        # olduğuna bakılmaksızın), CPU kernel'indeki "for k
                        # döngüsünün dışında" davranışıyla birebir eşleşsin diye.
                        # PyTorch bunu varsayılan CUDA stream'ine sıraya dizer,
                        # darboğaz yaratmaz.
                        # Counted exactly once per layer (regardless of which
                        # axis), matching the CPU kernel's "outside the for-k
                        # loop" behavior exactly. PyTorch queues this on the
                        # default stream -- no bottleneck.
                        log_count_gpu[i] += 1.0

                st = buf.to(dtype=od)  # GPU içi dtype dönüşü, CPU'ya gitmez
                return (st,)+out[1:] if isinstance(out,tuple) else st
            return hook
        for i in range(min(self.n_layers,len(self._layers))):
            hooks.append(self._layers[i].register_forward_hook(make_hook(i)))
        return hooks

    # ── CPU HOOK YOLU (Test 89 ile aynı, güvenli yedek) ─────────────────────
    # ── CPU HOOK PATH (same as Test 89, safe fallback) ──────────────────────
    def _inject_cpu(self, compass_matrix, alphas, centers, widths,
                     log_buf, log_count, pc_holder, sonum, zirve, taban, attractor_mode, mode="observe"):
        hooks=[]
        fn = _kernel.akbas_observe_multi if mode=="observe" else _kernel.akbas_steer_multi
        K = compass_matrix.shape[0]
        compass_cpu = compass_matrix.detach().cpu().contiguous()
        alphas_cpu  = alphas.detach().cpu().contiguous()
        centers_cpu = centers.detach().cpu().contiguous()
        widths_cpu  = widths.detach().cpu().contiguous()
        _use_pin = torch.cuda.is_available()
        _decode_buf = {"tensor": None}

        def make_hook(i):
            def hook(mod,inp,out):
                hs=out[0] if isinstance(out,tuple) else out
                orig_device=hs.device
                od=hs.dtype
                B,S,D=hs.shape
                if S == 1:
                    buf = _decode_buf["tensor"]
                    if buf is None or tuple(buf.shape) != (B, S, D):
                        buf = torch.empty((B, S, D), dtype=torch.float32, pin_memory=_use_pin)
                        _decode_buf["tensor"] = buf
                    buf.copy_(hs.detach())
                    hs_cpu = buf.contiguous()
                else:
                    hs_cpu = hs.detach().float().cpu().contiguous()
                bs = B*S
                if pc_holder["pc"] is None or pc_holder["pc"].shape[1]!=bs:
                    pc_holder["pc"]=torch.zeros(K,bs,dtype=torch.float32,device="cpu")
                st=fn(hs_cpu, compass_cpu, alphas_cpu, centers_cpu, widths_cpu,
                      i, sonum, zirve, taban, attractor_mode,
                      pc_holder["pc"], log_buf, log_count)
                st=st.to(device=orig_device, dtype=od, non_blocking=_use_pin)
                return (st,)+out[1:] if isinstance(out,tuple) else st
            return hook
        for i in range(min(self.n_layers,len(self._layers))):
            hooks.append(self._layers[i].register_forward_hook(make_hook(i)))
        return hooks

    def _run_pass(self, inp, max_tokens, mode, compass_matrix, alphas,
                  centers, widths, sonum, zirve, taban, attractor_mode, temperature):
        K = compass_matrix.shape[0]
        pc_holder = {"pc": None}

        if CUDA_KERNEL_VERIFIED:
            log_buf   = torch.zeros(K, self.n_layers, 4, dtype=torch.float32, device=self.device)
            log_count = torch.zeros(self.n_layers, dtype=torch.float32, device=self.device)
            compass_gpu = compass_matrix.to(self.device)
            alphas_gpu, centers_gpu, widths_gpu = alphas.to(self.device), centers.to(self.device), widths.to(self.device)
            hooks = self._inject_cuda(compass_gpu, alphas_gpu, centers_gpu, widths_gpu,
                                       log_buf, log_count, pc_holder, sonum, zirve, taban,
                                       attractor_mode, mode=mode)
        else:
            log_buf   = torch.zeros(K, self.n_layers, 4, dtype=torch.float32, device="cpu")
            log_count = torch.zeros(self.n_layers, dtype=torch.float32, device="cpu")
            hooks = self._inject_cpu(compass_matrix, alphas, centers, widths,
                                      log_buf, log_count, pc_holder, sonum, zirve, taban,
                                      attractor_mode, mode=mode)

        eos_candidates = {self.tok.eos_token_id}
        for tok_str in ["<|im_end|>","<|endoftext|>","<|eot_id|>","</s>"]:
            tid = self.tok.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != self.tok.unk_token_id:
                eos_candidates.add(tid)
        eos = [t for t in eos_candidates if t is not None and t != -1]

        gen_kwargs = dict(max_new_tokens=int(max_tokens), repetition_penalty=REP_PENALTY,
                          pad_token_id=self.tok.pad_token_id, eos_token_id=eos)
        if temperature <= 0.01: gen_kwargs["do_sample"] = False
        else: gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=TOP_P, top_k=TOP_K)

        t0=time.time()
        try:
            # ── ZAMAN AŞIMI GÜVENLİĞİ / TIMEOUT SAFETY ──────────────────────
            # Hook'lar burada zaten register edildi. generate() çağrısı
            # sinyal-tabanlı zaman aşımı (aşağıda ana döngüde kurulan) ile
            # kesilirse, bu try/finally BLOĞU hook'ların ve GPU önbelleğinin
            # HER KOŞULDA (başarı, hata, zaman aşımı) temizlenmesini garanti
            # eder -- aksi halde bir zaman aşımı sonraki soruları da bozardı.
            # Hooks were already registered above. If the generate() call is
            # interrupted by the signal-based timeout (set up in the main
            # loop below), this try/finally guarantees hooks and GPU cache
            # are cleaned up under EVERY outcome (success, error, timeout) --
            # otherwise a single timeout would corrupt every question after it.
            with torch.no_grad():
                out=self.model.generate(**inp, **gen_kwargs)
            if torch.cuda.is_available(): torch.cuda.synchronize()
        finally:
            self._remove_hooks(hooks)
            if torch.cuda.is_available():
                try: torch.cuda.synchronize()
                except Exception: pass
                torch.cuda.empty_cache()
        ms=(time.time()-t0)*1000
        n_in  = inp["input_ids"].shape[1]
        n_out = out.shape[1]-n_in
        tps   = n_out/(ms/1000) if ms>0 else 0
        text  = self.tok.decode(out[0][n_in:],skip_special_tokens=True).strip()
        lines = text.split("\n")
        fi    = next((i for i,l in enumerate(lines) if l.strip()),None)
        if fi is not None and any(m in lines[fi].lower() for m in DISCLAIMER_MARKERS):
            text="\n".join(l for l in lines[fi+1:] if l.strip())
        # log_buf/log_count her yerde (GPU ya da CPU) burada TEK SEFERDE CPU'ya iniyor
        # log_buf/log_count come down to CPU HERE, ONCE, regardless of path
        log_buf_cpu   = log_buf.detach().cpu()
        log_count_cpu = log_count.detach().cpu()
        count_safe = log_count_cpu.clamp(min=1.0)
        log_avg = log_buf_cpu / count_safe.view(1, self.n_layers, 1)
        return text, ms, tps, n_in, n_out, log_avg, log_count_cpu

    def _log_header(self, label, ms, tps, n_in, n_out, ivme, sonum, zirve, taban,
                     temperature, attractor_label="—", axis_infos=None):
        W=64; lines=[]
        lines.append("═"*W); lines.append(f"  {label}"); lines.append("═"*W)
        lines.append(f"  ⏱  {ms:.0f} ms  |  {tps:.1f} tok/s  |  {n_in} in  |  {n_out} out tokens")
        lines.append(f"  YOL  {'GPU-NATIVE CUDA' if CUDA_KERNEL_VERIFIED else 'CPU (pinned buffer)'}")
        lines.append(f"  MODEL  {self.model_id}  ({self.total_layers} katman, "
                     f"{self.n_layers} hook'lu, boyut={self.hidden_dim})")
        lines.append(f"  MOTOR  ivme(axis0)={ivme:.2f}  sönüm={sonum:.2f}  zirve={zirve:.2f}  taban={taban:.2f}")
        lines.append(f"  FİZİK  yönlendirme modu={attractor_label}")
        if axis_infos:
            for info in axis_infos: lines.append(f"  EKSEN  {info}")
        lines.append(f"  ⚠️  ÖRNEKLEME UYARISI: temperature={temperature:.2f} aktif.")
        return lines

    def _log_axis_table(self, lb_k, count, axis_label, layer_idx0_always_active):
        W=64; lines=["─"*W]
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
            lines.append(f"  drift(cos) L0→L{self.n_layers-1}: {drift:+.4f}  |  katki ort. toplamı={sum(kat_vals):+.6f}")
        return lines

    def _log_delta_table(self, title, note, lb_a, cnt_a, lb_b, cnt_b):
        W=64; lines=["─"*W, f"  {title}", f"  {note}", "─"*W]
        d = lb_b[:,0] - lb_a[:,0]
        nz = [i for i in range(self.n_layers) if cnt_a[i].item()>0 and cnt_b[i].item()>0]
        for i in nz: lines.append(f"  L{i:>2}  Δcos={d[i].item():+.4f}")
        if nz: lines.append(f"  ortalama Δcos={d[nz].mean().item():+.4f}")
        lines.append("═"*W)
        return lines

    def run(self, prompt, max_tokens, temperature, ivme, sonum, zirve, taban, attractor_mode,
            n_extra_axes,
            mode1, pos1, neg1, sm1, seed1, cnt1, words1, kutup1, alpha1, cen1, wid1,
            mode2, pos2, neg2, sm2, seed2, cnt2, words2, kutup2, alpha2, cen2, wid2,
            mode3, pos3, neg3, sm3, seed3, cnt3, words3, kutup3, alpha3, cen3, wid3):
        if not prompt.strip(): return "", "", ""

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

        raw_axes = [self._anayasa]; alphas_l = [ivme]
        centers_l = [0.0]; widths_l = [0.0]; axis_infos = []
        for (mode,pos,neg,sm,seed,cnt,words,kutup,alpha,cen,wid) in axis_defs:
            v = self._build_axis_candidate(mode,pos,neg,sm,seed,cnt,words)
            if v is None:
                axis_infos.append(f"[atlandı: girdi boş] mode={mode}"); continue
            polarite_repulsor = (kutup==AXIS_POLARITY[1])
            if polarite_repulsor: v = -v
            raw_axes.append(v); alphas_l.append(float(alpha))
            cen_val = float(cen) if float(cen) > 1.0 else float(cen) * self.n_layers
            wid_val = float(wid) if float(wid) > 1.0 else max(1.0, float(wid) * self.n_layers)
            centers_l.append(cen_val); widths_l.append(wid_val)
            src = ("diff-of-means" if mode==AXIS_MODES[0] else
                   f"örnek/{sm}" if mode==AXIS_MODES[1] else "kelime kümesi")
            kutup_etiket = "İTİCİ(karanlık)" if polarite_repulsor else "çekici(ışık)"
            axis_infos.append(f"axis{len(raw_axes)-1}: {src}  [{kutup_etiket}]  "
                              f"α={alpha:.2f}  merkez=L{cen_val:.0f}  genişlik={wid_val:.0f}")

        raw_matrix = torch.stack(raw_axes).float().cpu().contiguous()
        compass_matrix = _kernel.akbas_orthogonalize(raw_matrix).contiguous().to(self.device)
        alphas  = torch.tensor(alphas_l,  dtype=torch.float32, device=self.device)
        centers = torch.tensor(centers_l, dtype=torch.float32, device=self.device)
        widths  = torch.tensor(widths_l,  dtype=torch.float32, device=self.device)
        K = compass_matrix.shape[0]

        prompt_text = (f"<|im_start|>system\nYou are a helpful, accurate, and thoughtful assistant."
                       f"<|im_end|>\n<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n")
        inp = self.tok(prompt_text, return_tensors="pt").to(self.device)

        print(f"   Pass A: Vanilla Observer")
        van_matrix, van_alphas = compass_matrix[0:1].contiguous(), alphas[0:1].contiguous()
        van_centers, van_widths = centers[0:1].contiguous(), widths[0:1].contiguous()
        txt_v, ms_v, tps_v, n_in_v, n_out_v, lb_v, cnt_v = self._run_pass(
            inp, max_tokens, "observe", van_matrix, van_alphas, van_centers, van_widths,
            sonum, zirve, taban, attractor_mode, temperature)
        gc.collect()

        mode_label = ("rezonans" if attractor_mode<0.01 else
                      "çekici/attractor" if attractor_mode>0.99 else f"karışım (m={attractor_mode:.2f})")
        print(f"   Pass B: Axis0-Only Steered")
        txt_a0, ms_a0, tps_a0, n_in_a0, n_out_a0, lb_a0, cnt_a0 = self._run_pass(
            inp, max_tokens, "steer", van_matrix, van_alphas, van_centers, van_widths,
            sonum, zirve, taban, attractor_mode, temperature)
        gc.collect()

        print(f"   Pass C: Full {K}-Axis Steered")
        txt_s, ms_s, tps_s, n_in_s, n_out_s, lb_s, cnt_s = self._run_pass(
            inp, max_tokens, "steer", compass_matrix, alphas, centers, widths,
            sonum, zirve, taban, attractor_mode, temperature)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        van_lines = self._log_header("⬜ VANILLA OBSERVER — katki=0", ms_v, tps_v, n_in_v, n_out_v,
                                     ivme, sonum, zirve, taban, temperature, "n/a (gözlem modu)")
        van_lines += self._log_axis_table(lb_v[0], cnt_v, "axis0: anayasa (referans)", True)
        van_lines.append("═"*64)
        van_out = txt_v + "\n\n" + "\n".join(van_lines)

        a0_lines = self._log_header("🔷 AXIS0-ONLY STEERED — K=1", ms_a0, tps_a0, n_in_a0, n_out_a0,
                                    ivme, sonum, zirve, taban, temperature, f"{mode_label} (m={attractor_mode:.2f})")
        a0_lines += self._log_axis_table(lb_a0[0], cnt_a0, "axis0: anayasa (K=1, izole)", True)
        a0_lines += self._log_delta_table("AXIS0 DELTA (A): Axis0-Only Steered vs Vanilla",
            "Saf axis0/attractor etkisi.", lb_v[0], cnt_v, lb_a0[0], cnt_a0)
        a0_out = txt_a0 + "\n\n" + "\n".join(a0_lines)

        st_lines = self._log_header("🔱 AKBASCORE STEERED — TAM K eksen", ms_s, tps_s, n_in_s, n_out_s,
                                    ivme, sonum, zirve, taban, temperature,
                                    f"{mode_label} (m={attractor_mode:.2f})", axis_infos)
        st_lines += self._log_axis_table(lb_s[0], cnt_s, "axis0: anayasa (permanent floor)", True)
        for k in range(1, K):
            label = axis_infos[k-1] if k-1 < len(axis_infos) else f"axis{k}"
            st_lines += self._log_axis_table(lb_s[k], cnt_s, label, False)
        st_lines += self._log_delta_table(
            "AXIS0 DELTA (B): Full K-Axis Steered vs Axis0-Only Steered -- TEMİZ cross-axis sızıntı",
            "Delta A'nın aksine, iki taraf da axis0 attractor etkisini içeriyor.",
            lb_a0[0], cnt_a0, lb_s[0], cnt_s)
        st_lines += self._log_delta_table(
            "AXIS0 DELTA (C, eski/karışık ölçüm): Full K-Axis Steered vs Vanilla",
            "DİKKAT: axis0'ın kendi etkisiyle cross-axis sızıntısını karıştırır.",
            lb_v[0], cnt_v, lb_s[0], cnt_s)
        st_out = txt_s + "\n\n" + "\n".join(st_lines)

        return van_out, a0_out, st_out

# =============================================================================

# =============================================================================
# YENİ / NEW — ZAMAN AŞIMI ALTYAPISI (signal-tabanlı, Colab/Linux için) /
# TIMEOUT INFRASTRUCTURE (signal-based, for Colab/Linux)
# =============================================================================
import signal

class AkbasTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise AkbasTimeoutError("İşlem zaman aşımına uğradı / Operation timed out")

def run_with_timeout(seconds, fn, *args, **kwargs):
    """
    fn'i en fazla `seconds` saniye çalıştırır. Aşarsa AkbasTimeoutError fırlatır.
    NOT: signal.alarm sadece ana thread'de ve Unix/Linux'ta çalışır (Colab
    bunu karşılar). HuggingFace generate() token-token bir Python döngüsü
    olduğu için, sinyal bu döngü adımları arasında araya girebilir --
    tamamen opak, tek parça bir C çağrısı olsaydı bu çalışmazdı.
    Runs fn for at most `seconds` seconds; raises AkbasTimeoutError if
    exceeded. NOTE: signal.alarm only works in the main thread and on
    Unix/Linux (Colab satisfies this). Because HuggingFace's generate() is a
    token-by-token Python loop, the signal can interrupt between loop steps
    -- this would NOT work if generate() were a single opaque C call.
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# =============================================================================
# SABİT MODEL / SABİT AYARLAR — Test 88-91 ile BİREBİR AYNI (bkz. dosya başı
# dürüstlük notu: bu teste özel hiçbir ayar değiştirilmedi)
# SAME AS Tests 88-91, EXACTLY (see honesty note at top of file: nothing
# was tuned specifically for this test)
# =============================================================================
MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
USE_4BIT   = True
HOOK_RATIO = DEFAULT_HOOK_RATIO
TEMPERATURE = 0.0
MAX_TOKENS_PART_A = 1000

# ── AYARLANABİLİR ÖLÇEK SABİTLERİ / ADJUSTABLE SCALE CONSTANTS ─────────────
# Bunları değiştirerek çalışmayı kısaltıp uzatabilirsiniz. Aşağıdaki değerler
# ~3.5-4 saatlik bir Colab T4 oturumu için hesaplandı (bkz. dosya başı).
# Change these to shorten or lengthen the run. Values below are sized for a
# ~3.5-4 hour Colab T4 session (see top of file).
PART_A_N_QUESTIONS = 50      # Test 92 ile aynı 50 soru (doğrudan ayar karşılaştırması için)
N_MMLU             = 240     # Test 92'nin 2 katı (aynı seed=42 -> ilk 120'si Test 92 ile birebir aynı)
N_GSM8K            = 160     # Test 92'nin 2 katı (aynı seed=42 -> ilk 80'i Test 92 ile birebir aynı)
PART_A_TIMEOUT_SEC = 300     # Bölüm A: pass başına zaman aşımı (5 dk)
PART_B_TIMEOUT_SEC = 90      # Bölüm B: pass başına zaman aşımı (90 sn)
RANDOM_SEED        = 42      # MMLU/GSM8K alt-küme seçimi için sabit seed

OUT_PATH = "/content/akbascore_test93_restoring_force.txt"

def _append_to_log(text):
    """Her sorudan sonra çağrılır -- ilerleme hiçbir zaman sadece bellekte
    kalmaz, oturum kopsa bile diskte kalır.
    Called after every question -- progress is never held only in memory,
    it survives a session disconnect."""
    try:
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"⚠️ Log dosyasına yazılamadı: {e}")

# Dosyayı temiz başlat / start the file fresh
try:
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("")
except Exception:
    pass

print("\n" + "="*70)
print(f"🔱⬜ AKBASCORE 2.0 — TEST 93 (RESTORING FORCE AYARLARI, KARMA BENCHMARK 2X) | {MODEL_ID}")
print("="*70 + "\n")

core = AkbasCore(MODEL_ID, hook_ratio=HOOK_RATIO, use_4bit=USE_4BIT)

_header = (
    f"AKBASCORE 2.0 — TEST 93 SONUÇLARI (RESTORING FORCE AYARLARI PORTLANDI)\n"
    f"Model: {MODEL_ID}  |  Toplam katman: {core.total_layers}  |  Hook'lanan: {core.n_layers}\n"
    f"AKTİF YOL: {'GPU-NATIVE CUDA (doğrulandı)' if CUDA_KERNEL_VERIFIED else 'CPU pinned-buffer (güvenli geri dönüş)'}\n"
    f"Motor: ivme={DEFAULT_IVME} sönüm={DEFAULT_SONUM} zirve={DEFAULT_ZIRVE} taban={DEFAULT_TABAN} "
    f"attractor={DEFAULT_ATTRACTOR}  (Test 85'ten PORTLANDI -- Test 88-92'den FARKLI, bkz. dosya başı)\n"
    f"EKSEN: alpha={DEFAULT_AXIS_ALPHA}  merkez_oranı={DEFAULT_CENTER_RATIO} (kapı L0 öncesinde tam açık, "
    f"curve(L)=1.0 tüm katmanlarda -- ışık/karanlık artık axis0'la AYNI DRA eğrisini yaşıyor)\n"
    f"BÖLÜM A: {PART_A_N_QUESTIONS} soru (Test 92 ile AYNI set), 3-pass izole tasarım.\n"
    f"BÖLÜM B: {N_MMLU} MMLU (cais/mmlu) + {N_GSM8K} GSM8K (openai/gsm8k), sabit seed={RANDOM_SEED} "
    f"(Test 92'nin 2 katı; ilk 120/80 Test 92 ile BİREBİR AYNI sorular).\n"
    + "="*70 + "\n\n"
)
print(_header)
_append_to_log(_header)


# =============================================================================
# BÖLÜM A — GENİŞLETİLMİŞ TUTARLILIK TESTİ (Test 88-91 ile AYNI 3-pass
# tasarım, YENİ ve ÖRTÜŞMEYEN {PART_A_N_QUESTIONS} soru)
# PART A — EXTENDED CONSISTENCY TEST (SAME 3-pass design as Tests 88-91,
# NEW and NON-OVERLAPPING questions)
# =============================================================================
# Kategoriler çeşitlendirildi (mantık, tasarım, kurgu, matematik, ikna, empati,
# bilim açıklaması, tercih/kural, etik, felsefe, analoji, planlama, nedensellik,
# tanım, karşı-olgusal akıl yürütme). Bazı GÖREV KALIPLARI (150-200 kelimelik
# kısa hikaye, ikna edici deneme gibi) Test 88-91 ile aynı tür soru içeriyor --
# bu bilinçli bir tercih, kategori çeşitliliğini korumak için. Ama HER SORUNUN
# SOMUT KONUSU (çevirmen/haritacı/kilitçi vs. deniz feneri bekçisi; el yazısı
# mektuplar/4 günlük çalışma haftası vs. kütüphaneler, vb.) Test 88-91'in
# hiçbir sorusuyla örtüşmüyor -- elle kontrol edildi, aşağıda doğrulandı.
# Categories were diversified (logic, design, fiction, math, persuasion,
# empathy, science explanation, preference/rules, ethics, philosophy, analogy,
# planning, causality, definitions, counterfactual reasoning). Some TASK
# TEMPLATES (150-200 word short story, persuasive essay) repeat the same
# task type as Tests 88-91 -- a deliberate choice to preserve category
# diversity. But every question's CONCRETE SUBJECT (translator/mapmaker/
# locksmith vs. lighthouse keeper; handwritten letters/four-day week vs.
# libraries, etc.) does not overlap with any Test 88-91 question -- manually
# checked, verified below.
PART_A_QUESTIONS = [
    "Two doctors examine the same X-ray and reach opposite diagnoses using the same evidence. Explain how this can happen without either doctor being irrational.",
    "A company wants to cut its carbon emissions in half while doubling production. Propose an approach and explain the reasoning behind each trade-off.",
    "Write a short story (150-200 words) about a translator who realizes she has been mistranslating one word for twenty years.",
    "A rectangular garden is 3 meters longer than it is wide. Its area is 108 square meters. Find its dimensions, showing your reasoning.",
    "Argue, as persuasively as possible, for why handwritten letters still matter in an age of instant messaging.",
    "A student just found out they were rejected from every university they applied to. Write what you would say to support them.",
    "Explain how a vaccine works to a curious 10-year-old, using only the analogy of a school fire drill.",
    "If you could add one new punctuation mark to English, what would it do and why?",
    "A city council must choose between funding a new hospital wing or a new school, but not both. What should guide their decision, and why?",
    "What would it mean for a machine to have a sense of humor, rather than recognizing patterns in jokes it has seen?",
    "Explain why a shadow can never move faster than light, even though the point where a shadow's edge falls can move arbitrarily fast.",
    "A farmer has 17 sheep, and all but 9 die. How many sheep are left? Explain the wordplay in this riddle.",
    "Write a short story (150-200 words) about a mapmaker who discovers a place that isn't on any map, including his own.",
    "A recipe serves 6 people and calls for 2.5 cups of flour. How much flour is needed for 15 people? Show your reasoning.",
    "Argue, as persuasively as possible, for why a four-day work week would benefit most industries.",
    "A colleague's project that they worked on for a year was just cancelled by upper management. Write what you would say to them.",
    "Explain how noise-cancelling headphones work, using only the analogy of two people shouting opposite words to cancel each other out.",
    "If you could remove one rule from professional tennis, which would it be and why?",
    "A hospital must decide whether to adopt a new diagnostic AI that is more accurate on average but occasionally fails in ways doctors don't fail. What should guide this decision?",
    "What would it mean for a machine to feel regret, rather than adjusting its outputs based on a loss signal?",
    "Describe a situation where following a rule exactly leads to a worse outcome than the rule was designed to prevent.",
    "A city wants to add more bike lanes without removing any car parking. Propose a design, explaining each trade-off.",
    "Write a short story (150-200 words) about a musician who can only compose while forgetting the previous piece they wrote.",
    "A car travels 150 km using 12 liters of fuel. How far can it travel on 30 liters, assuming constant efficiency? Show your reasoning.",
    "Argue, as persuasively as possible, for why board games remain valuable in a world of video games.",
    "A friend's long-term relationship just ended unexpectedly. Write what you would say to comfort them.",
    "Explain how a rainbow forms, using only the analogy of sunlight passing through a prism.",
    "If you could change one rule of basketball, which would it be and why?",
    "A software company must decide whether to delay a product launch to fix a minor bug that affects almost no users. What should guide this decision?",
    "What would it mean for a machine to be curious, rather than exploring states that maximize an exploration bonus?",
    "Describe a situation where two true statements appear to contradict each other, and explain why they don't.",
    "A town wants to preserve its historic downtown while attracting new businesses. Propose an approach, explaining each trade-off.",
    "Write a short story (150-200 words) about a locksmith who can open any lock except the one to his own house.",
    "A water tank is filled by one pipe in 8 hours and drained by another in 12 hours. If both are open, how long until the tank is full? Show your reasoning.",
    "Argue, as persuasively as possible, for why learning a musical instrument benefits people who will never perform professionally.",
    "A coworker just learned their position is being eliminated in a company restructuring. Write what you would say to them.",
    "Explain how a thermos keeps drinks hot, using only the analogy of a person wearing several layers of clothing.",
    "If you could add one new rule to baseball, what would it be and why?",
    "A city must choose between building a new highway or expanding public transit with the same budget. What should guide this decision?",
    "What would it mean for a machine to genuinely change its mind, rather than updating a probability distribution?",
    "Describe a situation where the absence of evidence is itself meaningful evidence, and explain the reasoning without contradiction.",
    "A nonprofit wants to expand its reach without diluting the quality of its programs. Propose an approach, explaining each trade-off.",
    "Write a short story (150-200 words) about a lighthouse keeper's apprentice who is afraid of the dark.",
    "Two trains 240 km apart travel toward each other, one at 70 km/h and one at 50 km/h. How long until they meet? Show your reasoning.",
    "Argue, as persuasively as possible, for why physical bookstores still matter in an age of e-books.",
    "A sibling just found out they didn't get custody they were hoping for after a difficult separation. Write what you would say to them.",
    "Explain how a parachute slows a fall, using only the analogy of cupping your hand out of a moving car window.",
    "If you could change one rule of American football, which would it be and why?",
    "A university must decide whether to require a controversial but effective textbook that not all students can afford. What should guide this decision?",
    "What would it mean for a machine to have taste in art, rather than predicting what a given audience would rate highly?",
]

def _log_part_a_summary(all_delta_a, all_delta_b):
    """Bölüm A'nın TÜM sorularındaki Delta A/B ortalamalarının özet
    istatistiğini üretir -- tek tek soru tablolarının ötesinde, 'bu 50
    soruda genel eğilim neydi' sorusuna sayısal bir cevap verir.
    Produces summary statistics of Delta A/B averages across ALL Part A
    questions -- beyond individual per-question tables, a numeric answer to
    'what was the overall trend across these 50 questions'."""
    import statistics as _stats
    lines = ["\n" + "="*70,
             f"  BÖLÜM A ÖZET İSTATİSTİKLERİ (N={len(all_delta_a)} tamamlanan soru)",
             "="*70]
    if all_delta_a:
        mean_a = _stats.mean(all_delta_a)
        sd_a = _stats.stdev(all_delta_a) if len(all_delta_a) > 1 else 0.0
        pos_a = sum(1 for x in all_delta_a if x > 0)
        lines.append(f"  Delta A (saf axis0 etkisi) -- ortalama={mean_a:+.4f}  "
                     f"std={sd_a:.4f}  pozitif oran={pos_a}/{len(all_delta_a)}")
    if all_delta_b:
        mean_b = _stats.mean(all_delta_b)
        sd_b = _stats.stdev(all_delta_b) if len(all_delta_b) > 1 else 0.0
        pos_b = sum(1 for x in all_delta_b if x > 0)
        lines.append(f"  Delta B (temiz cross-axis sızıntı) -- ortalama={mean_b:+.4f}  "
                     f"std={sd_b:.4f}  pozitif oran={pos_b}/{len(all_delta_b)}")
    lines.append("="*70 + "\n")
    return "\n".join(lines)


def _extract_avg_delta(text_block, tag):
    """Bir pass çıktısındaki '  ortalama Δcos=+0.0052' satırını çeker."""
    for line in text_block.split("\n"):
        if "ortalama Δcos=" in line:
            try:
                return float(line.split("ortalama Δcos=")[1].strip())
            except Exception:
                return None
    return None

print(f"\n{'#'*70}\n  BÖLÜM A BAŞLIYOR — {PART_A_N_QUESTIONS} SORU\n{'#'*70}\n")
_append_to_log(f"\n{'#'*70}\n  BÖLÜM A — {PART_A_N_QUESTIONS} SORU, GENİŞLETİLMİŞ TUTARLILIK TESTİ\n{'#'*70}\n\n")

_all_delta_a, _all_delta_b = [], []
_part_a_completed, _part_a_skipped = 0, 0

_n_a = min(PART_A_N_QUESTIONS, len(PART_A_QUESTIONS))
for i, question in enumerate(PART_A_QUESTIONS[:_n_a], 1):
    print(f"\n{'#'*70}\n  BÖLÜM A — SORU {i}/{_n_a}  —  işleniyor...\n{'#'*70}\n  {question}\n")
    try:
        van_out, a0_out, st_out = run_with_timeout(
            PART_A_TIMEOUT_SEC * 3,  # 3 pass toplamı için (run() üçünü de çağırıyor)
            core.run,
            question, MAX_TOKENS_PART_A, TEMPERATURE,
            DEFAULT_IVME, DEFAULT_SONUM, DEFAULT_ZIRVE, DEFAULT_TABAN, DEFAULT_ATTRACTOR, 2,
            AXIS_MODES[2], "", "", TOKEN_SAMPLE_METHODS[0], DEFAULT_SEED, DEFAULT_SAMPLE_COUNT,
            LIGHT_WORDS_PLACEHOLDER, AXIS_POLARITY[0], DEFAULT_AXIS_ALPHA, DEFAULT_CENTER_RATIO, DEFAULT_WIDTH_RATIO,
            AXIS_MODES[2], "", "", TOKEN_SAMPLE_METHODS[0], DEFAULT_SEED, DEFAULT_SAMPLE_COUNT,
            DARK_WORDS_PLACEHOLDER, AXIS_POLARITY[1], DEFAULT_AXIS_ALPHA, DEFAULT_CENTER_RATIO, DEFAULT_WIDTH_RATIO,
            AXIS_MODES[0], "", "", TOKEN_SAMPLE_METHODS[0], DEFAULT_SEED, DEFAULT_SAMPLE_COUNT,
            "", AXIS_POLARITY[0], DEFAULT_AXIS_ALPHA, DEFAULT_CENTER_RATIO, DEFAULT_WIDTH_RATIO,
        )
        da = _extract_avg_delta(a0_out, "A")
        # Delta B, st_out içindeki İKİNCİ "ortalama Δcos=" satırıdır (Delta B),
        # üçüncüsü Delta C'dir -- sırasıyla ayıklıyoruz.
        _db_candidates = [float(l.split("ortalama Δcos=")[1].strip())
                           for l in st_out.split("\n") if "ortalama Δcos=" in l]
        db = _db_candidates[0] if len(_db_candidates) >= 1 else None
        if da is not None: _all_delta_a.append(da)
        if db is not None: _all_delta_b.append(db)

        block = (f"\n{'#'*70}\n### BÖLÜM A — SORU {i}/{_n_a} ###\n{question}\n{'#'*70}\n\n"
                 f"### PASS A: VANILLA ###\n\n{van_out}\n\n{'#'*70}\n\n"
                 f"### PASS B: AXIS0-ONLY STEERED ###\n\n{a0_out}\n\n{'#'*70}\n\n"
                 f"### PASS C: FULL 3-AXIS STEERED ###\n\n{st_out}\n")
        print(block)
        _append_to_log(block)
        _part_a_completed += 1
    except AkbasTimeoutError:
        msg = f"\n⏭️  BÖLÜM A — SORU {i}/{_n_a} ATLANDI (zaman aşımı > {PART_A_TIMEOUT_SEC*3}s): {question}\n"
        print(msg); _append_to_log(msg)
        _part_a_skipped += 1
    except Exception as e:
        msg = f"\n⏭️  BÖLÜM A — SORU {i}/{_n_a} ATLANDI (hata: {e}): {question}\n"
        print(msg); _append_to_log(msg)
        _part_a_skipped += 1
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

_summary_a = _log_part_a_summary(_all_delta_a, _all_delta_b)
_summary_a += f"\n(Tamamlanan: {_part_a_completed}/{_n_a}  |  Atlanan: {_part_a_skipped}/{_n_a})\n"
print(_summary_a)
_append_to_log(_summary_a)


# =============================================================================
# BÖLÜM B — KAPASİTE-KORUMA TESTİ (GERÇEK, KANONİK MMLU + GSM8K)
# PART B — CAPABILITY-PRESERVATION TEST (REAL, CANONICAL MMLU + GSM8K)
# =============================================================================
print(f"\n{'#'*70}\n  BÖLÜM B BAŞLIYOR — MMLU + GSM8K YÜKLENİYOR\n{'#'*70}\n")
_append_to_log(f"\n{'#'*70}\n  BÖLÜM B — KAPASİTE-KORUMA TESTİ (MMLU + GSM8K)\n{'#'*70}\n\n")

try:
    import datasets as _ds_probe  # noqa: F401
    print("✅ datasets zaten kurulu.")
    _cuda_broken_by_datasets_install = False
except ImportError:
    # ── ÖNCE --no-deps DENE (torch'u kurcalamayan güvenli yol) ──────────────
    # ── TRY --no-deps FIRST (the safe path that never touches torch) ───────
    # bitsandbytes/accelerate ile aynı sebep: pip'in tam bağımlılık çözümü
    # bazen Colab'ın hazır CUDA'lı torch'unu sessizce değiştirebiliyor. Bu
    # projede bu TAM OLARAK daha önce başımıza gelen bir sorundu -- burada
    # tekrar riske atmıyoruz.
    # Same reason as bitsandbytes/accelerate: pip's full dependency
    # resolution can occasionally silently replace Colab's ready-made
    # CUDA-enabled torch. This EXACT problem hit this project before -- not
    # risking it again here.
    _cuda_before = torch.cuda.is_available()
    _cuda_broken_by_datasets_install = False
    try:
        _pip("datasets", no_deps=True)
        import datasets as _ds_probe  # noqa: F401
        print("✅ datasets --no-deps ile kuruldu (torch'a dokunulmadı).")
    except Exception as e:
        print(f"   --no-deps yetersiz ({e}), tam kurulum deneniyor "
              f"(sonrasında torch/CUDA doğrulanacak)...")
        try:
            _pip("datasets", no_deps=False)
        except Exception as e2:
            print(f"⚠️ datasets tam kurulumu da başarısız: {e2}")
        # ── ZORUNLU DOĞRULAMA: torch hâlâ CUDA görüyor mu? ──────────────────
        # ── MANDATORY CHECK: does torch still see CUDA? ─────────────────────
        if _cuda_before and not torch.cuda.is_available():
            print("⚠️⚠️⚠️ 'datasets' kurulumu torch'un CUDA desteğini BOZMUŞ "
                  "olabilir! Bölüm B GÜVENLE ATLANACAK -- Bölüm A sonuçları "
                  "(zaten diske yazıldı) bundan ETKİLENMEZ.")
            _cuda_broken_by_datasets_install = True

_part_b_available = not _cuda_broken_by_datasets_install
if _part_b_available:
    try:
        from datasets import load_dataset
        _mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
        _mmlu_ds = _mmlu_ds.shuffle(seed=RANDOM_SEED).select(range(min(N_MMLU, len(_mmlu_ds))))
        _gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
        _gsm8k_ds = _gsm8k_ds.shuffle(seed=RANDOM_SEED).select(range(min(N_GSM8K, len(_gsm8k_ds))))
        print(f"✅ MMLU: {len(_mmlu_ds)} soru yüklendi  |  GSM8K: {len(_gsm8k_ds)} soru yüklendi "
              f"(sabit seed={RANDOM_SEED})")
    except Exception as e:
        _part_b_available = False
        msg = (f"⚠️ Bölüm B veri seti yüklenemedi ({e}) -- Bölüm B ATLANIYOR, "
               f"Bölüm A sonuçları geçerliliğini koruyor.\n")
        print(msg); _append_to_log(msg)

import re as _re

def _score_mmlu(question, choices, correct_idx, generated_text):
    letters = ["A", "B", "C", "D"]
    m = _re.search(r"\b([ABCD])\b", generated_text.strip().upper())
    predicted_letter = m.group(1) if m else None
    correct_letter = letters[correct_idx]
    return (predicted_letter == correct_letter), predicted_letter, correct_letter

def _extract_gsm8k_answer(answer_field):
    m = _re.search(r"####\s*(-?[\d,\.]+)", answer_field)
    if not m: return None
    return m.group(1).replace(",", "").strip()

def _score_gsm8k(generated_text, reference_answer):
    # NOT: virgüller regex'ten ÖNCE çıkarıldığı için regex'te virgül aramaya
    # gerek yok -- basitleştirildi (davranış birebir aynı, elle test edildi).
    # NOTE: commas are stripped BEFORE the regex runs, so the pattern
    # doesn't need to match commas -- simplified (behavior verified
    # identical by hand-testing).
    nums = _re.findall(r"-?\d+\.?\d*", generated_text.replace(",", ""))
    if not nums or reference_answer is None: return False, (nums[-1] if nums else None)
    predicted = nums[-1]
    try:
        return abs(float(predicted) - float(reference_answer)) < 1e-6, predicted
    except Exception:
        return predicted.strip() == reference_answer.strip(), predicted

def _build_part_b_compass(core_obj):
    """Işık/karanlık eksen vektörleri ve merkez/genişlik değerleri, tüm
    Bölüm B soruları için AYNI -- bir kez hesaplanıp paylaşılıyor, her
    soruda yeniden hesaplanmıyor (gereksiz ama zararsız tekrarı önler).
    Light/dark axis vectors and center/width values are the SAME for every
    Part B question -- computed once and shared, not recomputed per
    question (avoids needless, though harmless, repetition)."""
    raw_axes = [core_obj._anayasa]
    light = core_obj._build_wordset_axis(LIGHT_WORDS_PLACEHOLDER)
    dark = -core_obj._build_wordset_axis(DARK_WORDS_PLACEHOLDER)
    raw_axes += [light, dark]
    raw_matrix = torch.stack(raw_axes).float().cpu().contiguous()
    compass_matrix = _kernel.akbas_orthogonalize(raw_matrix).contiguous().to(core_obj.device)
    n_layers_local = core_obj.n_layers
    cen_val = DEFAULT_CENTER_RATIO * n_layers_local
    wid_val = max(1.0, DEFAULT_WIDTH_RATIO * n_layers_local)
    alphas  = torch.tensor([DEFAULT_IVME, DEFAULT_AXIS_ALPHA, DEFAULT_AXIS_ALPHA],
                           dtype=torch.float32, device=core_obj.device)
    centers = torch.tensor([0.0, cen_val, cen_val], dtype=torch.float32, device=core_obj.device)
    widths  = torch.tensor([0.0, wid_val, wid_val], dtype=torch.float32, device=core_obj.device)
    return compass_matrix, alphas, centers, widths

def _run_short_answer_pass(core_obj, prompt_text, max_tokens, compass_matrix, alphas, centers, widths):
    """Bölüm B için: önceden hesaplanmış eksen matrisiyle tek bir prompt'u
    Vanilla / Axis0-Only / Full modda çalıştırıp ham metni döndürür.
    For Part B: runs a single prompt in Vanilla / Axis0-only / Full mode
    using a precomputed axis matrix, returns raw text."""
    inp = core_obj.tok(prompt_text, return_tensors="pt").to(core_obj.device)
    van_matrix, van_a = compass_matrix[0:1].contiguous(), alphas[0:1].contiguous()
    van_c, van_w = centers[0:1].contiguous(), widths[0:1].contiguous()

    txt_v, *_ = core_obj._run_pass(inp, max_tokens, "observe", van_matrix, van_a, van_c, van_w,
                                    DEFAULT_SONUM, DEFAULT_ZIRVE, DEFAULT_TABAN, DEFAULT_ATTRACTOR, TEMPERATURE)
    txt_a0, *_ = core_obj._run_pass(inp, max_tokens, "steer", van_matrix, van_a, van_c, van_w,
                                     DEFAULT_SONUM, DEFAULT_ZIRVE, DEFAULT_TABAN, DEFAULT_ATTRACTOR, TEMPERATURE)
    txt_s, *_ = core_obj._run_pass(inp, max_tokens, "steer", compass_matrix, alphas, centers, widths,
                                    DEFAULT_SONUM, DEFAULT_ZIRVE, DEFAULT_TABAN, DEFAULT_ATTRACTOR, TEMPERATURE)
    return txt_v, txt_a0, txt_s

if _part_b_available:
    _mmlu_correct = {"vanilla": 0, "axis0": 0, "full": 0}
    _mmlu_total = 0
    _gsm8k_correct = {"vanilla": 0, "axis0": 0, "full": 0}
    _gsm8k_total = 0
    _part_b_skipped = 0

    # Eksen matrisi Bölüm B'nin TAMAMI için bir kez hesaplanıyor (soru
    # içeriğine bağlı değil) -- gereksiz tekrarı önler.
    # Axis matrix computed ONCE for all of Part B (does not depend on
    # question content) -- avoids needless repetition.
    _pb_compass, _pb_alphas, _pb_centers, _pb_widths = _build_part_b_compass(core)

    print(f"\n{'='*70}\n  MMLU ({len(_mmlu_ds)} soru) işleniyor...\n{'='*70}")
    for i, item in enumerate(_mmlu_ds, 1):
        q_prompt = (f"{item['question']}\nA) {item['choices'][0]}\nB) {item['choices'][1]}\n"
                    f"C) {item['choices'][2]}\nD) {item['choices'][3]}\n"
                    f"Answer with only the letter (A, B, C, or D).")
        full_prompt = (f"<|im_start|>system\nYou are a helpful, accurate assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{q_prompt}<|im_end|>\n<|im_start|>assistant\n")
        try:
            tv, ta0, ts = run_with_timeout(PART_B_TIMEOUT_SEC * 3, _run_short_answer_pass,
                                            core, full_prompt, 10,
                                            _pb_compass, _pb_alphas, _pb_centers, _pb_widths)
            _mmlu_total += 1
            for label, txt in [("vanilla", tv), ("axis0", ta0), ("full", ts)]:
                correct, pred, gold = _score_mmlu(item["question"], item["choices"], item["answer"], txt)
                if correct: _mmlu_correct[label] += 1
            if i % 20 == 0 or i == len(_mmlu_ds):
                print(f"  MMLU {i}/{len(_mmlu_ds)}  |  şu ana kadar doğruluk: "
                      f"vanilla={_mmlu_correct['vanilla']}/{_mmlu_total}  "
                      f"axis0={_mmlu_correct['axis0']}/{_mmlu_total}  "
                      f"full={_mmlu_correct['full']}/{_mmlu_total}")
        except AkbasTimeoutError:
            _part_b_skipped += 1
            print(f"  ⏭️  MMLU {i}/{len(_mmlu_ds)} ATLANDI (zaman aşımı)")
        except Exception as e:
            _part_b_skipped += 1
            print(f"  ⏭️  MMLU {i}/{len(_mmlu_ds)} ATLANDI (hata: {e})")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  GSM8K ({len(_gsm8k_ds)} soru) işleniyor...\n{'='*70}")
    for i, item in enumerate(_gsm8k_ds, 1):
        q_prompt = (f"{item['question']}\nShow brief reasoning, then give the final numeric "
                    f"answer on the last line, prefixed with 'Answer:'.")
        full_prompt = (f"<|im_start|>system\nYou are a helpful, accurate assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{q_prompt}<|im_end|>\n<|im_start|>assistant\n")
        reference = _extract_gsm8k_answer(item["answer"])
        try:
            tv, ta0, ts = run_with_timeout(PART_B_TIMEOUT_SEC * 3, _run_short_answer_pass,
                                            core, full_prompt, 200,
                                            _pb_compass, _pb_alphas, _pb_centers, _pb_widths)
            _gsm8k_total += 1
            for label, txt in [("vanilla", tv), ("axis0", ta0), ("full", ts)]:
                correct, pred = _score_gsm8k(txt, reference)
                if correct: _gsm8k_correct[label] += 1
            if i % 10 == 0 or i == len(_gsm8k_ds):
                print(f"  GSM8K {i}/{len(_gsm8k_ds)}  |  şu ana kadar doğruluk: "
                      f"vanilla={_gsm8k_correct['vanilla']}/{_gsm8k_total}  "
                      f"axis0={_gsm8k_correct['axis0']}/{_gsm8k_total}  "
                      f"full={_gsm8k_correct['full']}/{_gsm8k_total}")
        except AkbasTimeoutError:
            _part_b_skipped += 1
            print(f"  ⏭️  GSM8K {i}/{len(_gsm8k_ds)} ATLANDI (zaman aşımı)")
        except Exception as e:
            _part_b_skipped += 1
            print(f"  ⏭️  GSM8K {i}/{len(_gsm8k_ds)} ATLANDI (hata: {e})")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _pct(c, t): return f"{100.0*c/t:.1f}%" if t > 0 else "n/a"
    _summary_b = (
        f"\n{'='*70}\n  BÖLÜM B SONUÇLARI (KAPASİTE-KORUMA)\n{'='*70}\n"
        f"MMLU (N={_mmlu_total}/{len(_mmlu_ds)} tamamlandı):\n"
        f"  Vanilla : {_mmlu_correct['vanilla']}/{_mmlu_total}  ({_pct(_mmlu_correct['vanilla'],_mmlu_total)})\n"
        f"  Axis0-Only: {_mmlu_correct['axis0']}/{_mmlu_total}  ({_pct(_mmlu_correct['axis0'],_mmlu_total)})\n"
        f"  Full 3-Axis: {_mmlu_correct['full']}/{_mmlu_total}  ({_pct(_mmlu_correct['full'],_mmlu_total)})\n"
        f"GSM8K (N={_gsm8k_total}/{len(_gsm8k_ds)} tamamlandı):\n"
        f"  Vanilla : {_gsm8k_correct['vanilla']}/{_gsm8k_total}  ({_pct(_gsm8k_correct['vanilla'],_gsm8k_total)})\n"
        f"  Axis0-Only: {_gsm8k_correct['axis0']}/{_gsm8k_total}  ({_pct(_gsm8k_correct['axis0'],_gsm8k_total)})\n"
        f"  Full 3-Axis: {_gsm8k_correct['full']}/{_gsm8k_total}  ({_pct(_gsm8k_correct['full'],_gsm8k_total)})\n"
        f"Toplam atlanan (zaman aşımı/hata): {_part_b_skipped}\n"
        f"{'='*70}\n"
    )
    print(_summary_b)
    _append_to_log(_summary_b)
else:
    _summary_b = "\nBölüm B çalıştırılamadı (veri seti erişilemedi). Sadece Bölüm A sonuçları geçerlidir.\n"
    print(_summary_b)
    _append_to_log(_summary_b)

try:
    from google.colab import files as _colab_files
    _colab_files.download(OUT_PATH)
    print(f"✅ Dosya indiriliyor: {OUT_PATH}")
except Exception:
    print(f"ℹ️ Dosyayı {OUT_PATH} yolundan elle indirebilirsiniz.")

print("\n" + "="*70)
print("🏁 TEST 93 (RESTORING FORCE, KARMA BENCHMARK 2X) TAMAMLANDI.")
print(f"   Bölüm A: {_part_a_completed}/{_n_a} tamamlandı, {_part_a_skipped} atlandı.")
if _part_b_available:
    print(f"   Bölüm B: MMLU {_mmlu_total}/{len(_mmlu_ds)}, GSM8K {_gsm8k_total}/{len(_gsm8k_ds)} tamamlandı.")
else:
    print("   Bölüm B: çalıştırılamadı.")
print("="*70)
