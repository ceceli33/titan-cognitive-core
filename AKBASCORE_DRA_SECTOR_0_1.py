# =============================================================================
# 🔱⬜ AKBASCORE DRA-SECTOR_0.1 | QWEN2.5-1.5B-INSTRUCT
# Sentetik Gömünç ve Kontrol Tensör Yeniden Hizalaması
# (Synthetic Embedding & Control Tensor Realignment)
# =============================================================================
#
# Bu sürüm, daha önce DRA — Sönümlü Rezonans Hizalama (Damped Resonance
# Alignment) metodolojisiyle geliştirilen AkbasCore 1.2'nin ana gövdesi
# üzerine inşa edilmiştir. AkbasCore 1.2'de pusula (Compass Vector), sabit
# tek bir yön olarak hidden state'e ölçekli bir katki (Contribution)
# uyguluyordu. SECTOR açılımı bu pusula vektör alanını genişletir: artık
# sabit tek yön yerine, dışarıdan tanımlanabilen bir Sentetik Kontrol
# Vektörü (V_ext) aracılığıyla iç embedding (Gömünç) uzayı doğrudan
# BÜKÜLÜR, DÖNDÜRÜLÜR ve KAPILANIR.
#
# ESKİDE (AkbasCore 1.2) OLMAYAN, SADECE BU SÜRÜMDE GETİRİLEN YENİLİKLER:
#
#   1) Sentetik Kontrol Vektörü (V_ext) — Synthetic Control Vector
#      Yön artık yalnızca anayasa (Constitution) kelimelerinden değil;
#      Gauss Maskesi, Top-K Varyans Maskesi, Rastgele Birim Vektör veya
#      bunların Hibrit birleşimiyle de üretilebiliyor.
#
#   2) Rank-1 Düşük Rütbeli Bükülme — Rank-1 Low-Rank Warp
#      E' = E + α·(E·V)·V — embedding'in V_ext yönündeki bileşenini
#      doğrudan güçlendiren yeni bir dönüşüm operatörü. AkbasCore 1.2'de
#      böyle bir rank-1 projeksiyon güçlendirmesi yoktu.
#
#   3) Aksiyel Rotasyon Torku — Axial Rotation Torque
#      θ = açı(V_ext, E) üzerinden E'yi V_ext eksenine doğru döndüren
#      yeni bir tork mekanizması. AkbasCore 1.2 yalnızca skaler toplama
#      (Katki) yapıyordu; burada olduğu gibi bir rotasyon işlemi yoktu.
#
#   4) Kapılamalı Non-Lineer Modülasyon — Gated Non-Linear Modulation
#      E' = E ⊙ tanh(β·V_ext) — Hadamard çarpımıyla kanal bazlı geçit
#      (Kapı) uygulaması; AkbasCore 1.2'de bulunmayan yeni bir katman.
#
#   5) Kritik Sönüm Kontrolü — Critical Damping Control (ζ)
#      k = 1 - e^(-ζ) formülüyle aşırı tepkiyi (Overshoot) engelleyen
#      yeni bir kontrol teorisi katsayısı. AkbasCore 1.2'deki Sönüm (ω)
#      yalnızca katman bazlı genlik azalmasıydı; ζ ise mekanizmaların
#      şiddetini kritik sönüm noktasında sınırlayan ayrı bir kapıdır.
#
# KORUNAN (AkbasCore 1.2'den değişmeden gelen) ÇEKİRDEK:
#   Pusula inşası, katki formülü, oran/doyum/karşıt/sapma/fren/sınır
#   sabitleri ve DUAL PASS (Vanilla Observer / Steered) yapısı aynen
#   korunmuştur — SECTOR, STEERED pass'in yerini alabilen alternatif bir
#   dönüşüm katmanı olarak eklenmiştir.
#
# Model   : Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)
# Kernel  : C++ — akbas_observe + akbas_steer + akbas_sector
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
# C++ ÇEKİRDEK — AkbasCore DRA-SECTOR_0.1 Motoru
# =============================================================================
_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

/*
 * AkbasCore DRA-SECTOR_0.1 — Motor Sabitleri
 *
 * SECTOR KATMANI (yeni):
 *   bukme_alfa (warp alpha)         : 0.35  — rank-1 bükülme katsayısı
 *   tork_beta  (torque beta)        : 0.55  — rotasyon torku katsayısı
 *   kapi_gama  (gate gamma)         : 1.20  — Hadamard kapı sıcaklığı
 *   zeta       (damping zeta)       : 1.00  — kritik sönüm (overshoot engelleyici)
 *
 * AKBASCORE 1.2 sabitleri (korundu):
 *   oran 0.32 · doyum 0.75 · karşıt -0.40 · sapma 0.20 · fren 0.30 · sınır 0.045
 */

// ─── AKBASCORE PASS A: GÖZLEM ────────────────────────────────────────────────
torch::Tensor akbas_observe(
    torch::Tensor hidden, torch::Tensor pusula,
    float ivme, int layer_idx, float sonum, float zirve, float taban,
    torch::Tensor prev_cosine, torch::Tensor log_buf
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
        for (int j = 0; j < D; ++j) { dot += tok[j]*pp[j]; sq += tok[j]*tok[j]; }
        float norm = sqrtf(sq) + 1e-6f;
        float cs   = std::clamp(dot/norm, -1.0f, 1.0f);
        float unc  = 1.0f - fabsf(cs);
        float dw   = sonum + unc * 0.20f;
        float kb   = zirve * expf(-dw*t) * (1.0f + dw*t) + taban;
        float dr   = std::clamp(cs - pcp[idx], -0.15f, 0.15f);
        if (cs > 0.80f && dr < 0.0f) dr *= 0.30f;
        float kv = kb;
        if      (dr > 0) kv *= (1.0f - dr*0.30f);
        else if (dr < 0) kv *= (1.0f + (-dr)*0.30f);
        kv = std::clamp(kv, 0.05f, 1.0f);
        pcp[idx] = cs;
        float son = 1.0f;
        if      (cs >  0.75f) son = (1.0f - cs)/0.25f;
        else if (cs < -0.40f) son = 1.6f;
        float mk    = std::clamp(norm*0.045f, 0.04f, 0.20f);
        float katki = std::clamp(ivme*cs*kv*0.32f*son, -mk, mk);
        if (S>1 && b==0 && s==0 && layer_idx<20) {
            lbp[layer_idx*4+0]=cs; lbp[layer_idx*4+1]=kb;
            lbp[layer_idx*4+2]=kv; lbp[layer_idx*4+3]=katki;
        }
    }
    return h;
}

// ─── AKBASCORE PASS B: STEERING ──────────────────────────────────────────────
torch::Tensor akbas_steer(
    torch::Tensor hidden, torch::Tensor pusula,
    float ivme, int layer_idx, float sonum, float zirve, float taban,
    torch::Tensor prev_cosine, torch::Tensor log_buf
) {
    auto h  = hidden.contiguous();
    auto p  = pusula.contiguous();
    auto lb = log_buf.contiguous();
    if (layer_idx >= 20) return h;

    const int B = h.size(0), S = h.size(1), D = h.size(2);
    float t = (float)layer_idx;

    float       *hp  = h.data_ptr<float>();
    float       *pcp = prev_cosine.data_ptr<float>();
    // log_buf'un contiguous kopyası (lb) üzerinden ham pointer erişimi
    float       *lbp = lb.data_ptr<float>();
    const float *pp  = p.data_ptr<float>();

    for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s) {
        float *tok = hp + (b*S*D) + (s*D);
        int    idx = b*S + s;
        float dot=0, sq=0;
        for (int j = 0; j < D; ++j) { dot += tok[j]*pp[j]; sq += tok[j]*tok[j]; }
        float norm = sqrtf(sq)+1e-6f;
        float cs   = std::clamp(dot/norm, -1.0f, 1.0f);
        float unc  = 1.0f - fabsf(cs);
        float dw   = sonum + unc*0.20f;
        float kb   = zirve*expf(-dw*t)*(1.0f+dw*t)+taban;
        float dr   = std::clamp(cs-pcp[idx], -0.15f, 0.15f);
        if (cs>0.80f && dr<0.0f) dr *= 0.30f;
        float kv = kb;
        if      (dr > 0) kv *= (1.0f-dr*0.30f);
        else if (dr < 0) kv *= (1.0f+(-dr)*0.30f);
        kv = std::clamp(kv, 0.05f, 1.0f);
        pcp[idx] = cs;
        float son = 1.0f;
        if      (cs >  0.75f) son = (1.0f-cs)/0.25f;
        else if (cs < -0.40f) son = 1.6f;
        float mk    = std::clamp(norm*0.045f, 0.04f, 0.20f);
        float katki = std::clamp(ivme*cs*kv*0.32f*son, -mk, mk);
        for (int j = 0; j < D; ++j) tok[j] += katki*pp[j];
        if (S>1 && b==0 && s==0 && layer_idx<20) {
            lbp[layer_idx*4+0]=cs; lbp[layer_idx*4+1]=kb;
            lbp[layer_idx*4+2]=kv; lbp[layer_idx*4+3]=katki;
        }
    }
    return h;
}

// ─── DRA-SECTOR 0.1: SENTETİK VEKTÖRLÜ METAMORFOZ ────────────────────────────
/*
 * 3 mekanizmayı birlikte uygular:
 *   (1) Rank-1 düşük rütbeli bükülme   : E' = E + α·(E·V)·V
 *   (2) Aksiyel rotasyon torku         : E' = E - β·sin(θ)·(E - cos(θ)·V·|E|)
 *   (3) Kapılamalı non-lineer modülasyon: E' = E ⊙ (1 + γ·tanh(β·V))
 *
 * v_ext: harici sentetik vektör (D boyutlu, birim normalize)
 * alfa : rank-1 bükülme katsayısı
 * beta : tork katsayısı
 * gama : kapı sıcaklığı
 * zeta : kritik sönüm (katsayı olarak 1.0 → overshoot yok)
 */
torch::Tensor akbas_sector(
    torch::Tensor hidden,
    torch::Tensor v_ext,
    float alfa,
    float beta,
    float gama,
    float zeta,
    int layer_idx,
    torch::Tensor log_buf
) {
    auto h  = hidden.contiguous();
    auto ve = v_ext.contiguous();
    auto lb = log_buf.contiguous();
    if (layer_idx >= 20) return h;

    const int B = h.size(0), S = h.size(1), D = h.size(2);
    float       *hp  = h.data_ptr<float>();
    const float *vp  = ve.data_ptr<float>();
    // log_buf (20,4) şeklindeki tensöre satır×sütun düzeninde flat pointer
    // erişimi — akbas_observe/akbas_steer ile aynı yazım yöntemi
    float       *lbp = lb.data_ptr<float>();

    // V_ext birim norm kontrolü (güvenlik)
    float vnorm = 0;
    for (int j = 0; j < D; ++j) vnorm += vp[j]*vp[j];
    vnorm = sqrtf(vnorm) + 1e-8f;

    // zeta ∈ [0,1] → critical damping katsayısı (1.0 = kritik, overshoot yok)
    // transfer fonksiyonu: k = 1 - exp(-zeta) — zeta=1 → k=0.632 (kritik sönüm)
    float k_damp = 1.0f - expf(-zeta);

    for (int b = 0; b < B; ++b)
    for (int s = 0; s < S; ++s) {
        float *tok = hp + (b*S*D) + (s*D);

        // 1. Projeksiyon & norm hesapla
        float dot = 0, sq = 0;
        for (int j = 0; j < D; ++j) { dot += tok[j]*vp[j]; sq += tok[j]*tok[j]; }
        float en  = sqrtf(sq) + 1e-8f;
        float cos_theta = std::clamp(dot/(en*vnorm), -1.0f, 1.0f);
        float sin_theta = sqrtf(std::max(0.0f, 1.0f - cos_theta*cos_theta));

        // 2. Mekanizma (1): Rank-1 düşük rütbeli bükülme
        //    M = V_ext ⊗ V_ext^T → E·M = (E·V_ext)·V_ext
        float warp_proj = dot / (vnorm*vnorm);  // normalized projection

        // 3. Mekanizma (2): Aksiyel rotasyon torku
        //    θ = arccos(cos_theta), hedef: E'yi V_ext yönünde cos(θ)→1 yap
        //    Rotation update: E' = E - β·k_damp·sin(θ)·(E - cos(θ)·V̂·|E|)
        //    (E - cos(θ)·V̂·|E|) = E'nin V̂'ye dik bileşeni
        float rot_factor = beta * k_damp * sin_theta;

        // 4. Mekanizma (3): Kapılamalı non-lineer modülasyon
        //    gate = 1 + γ·tanh(β·cos(θ)) — pozitif hizalı kanallar amplifiye
        float gate = 1.0f + gama * tanhf(beta * cos_theta);

        // Uygula: 3 mekanizma birlikte
        for (int j = 0; j < D; ++j) {
            float e_j    = tok[j];
            float v_hat  = vp[j] / vnorm;
            float e_perp = e_j - cos_theta * en * v_hat;  // dik bileşen
            // (1) bükülme: V_ext yönüne projeksiyon kuvvetlendir
            float warped = e_j + alfa * k_damp * warp_proj * v_hat;
            // (2) rotasyon: dik bileşeni sönümle (V_ext yönüne çek)
            float rotated = warped - rot_factor * e_perp;
            // (3) kapı: kanal bazlı amplifikasyon
            tok[j] = rotated * gate;
        }

        // Log: cos, warp, rot, gate katsayıları
        if (B == 1 && S == 1 && layer_idx < 20) {
            lbp[layer_idx*4+0] = cos_theta;
            lbp[layer_idx*4+1] = warp_proj;
            lbp[layer_idx*4+2] = rot_factor;
            lbp[layer_idx*4+3] = gate;
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

for _n in ["akbas_sector_01_f1","akbas_sector_01_f2"]:
    for _r in ["/tmp/akbas_sector_01",os.path.expanduser("~/.cache/torch_extensions")]:
        _p=os.path.join(_r,_n)
        if os.path.exists(_p): shutil.rmtree(_p,ignore_errors=True)

def _build(name):
    d=f"/tmp/akbas_sector_01/{name}"; os.makedirs(d,exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name, cpp_sources=_CPP,
        functions=["akbas_observe","akbas_steer","akbas_sector"],
        build_directory=d, verbose=False,
        extra_cflags=["-O2","-std=c++17"])

try:    _kernel=_build("akbas_sector_01_f1")
except Exception as e:
    print(f"  retry ({e})..."); _kernel=_build("akbas_sector_01_f2")
print("✅ AkbasCore DRA-SECTOR_0.1 C++ çekirdeği derlendi")

# =============================================================================
# SABİT PARAMETRELER
# =============================================================================
MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE  = 0.65
TOP_P        = 0.90
TOP_K        = 50
REP_PENALTY  = 1.15
N_LAYERS     = 20

CONSTITUTION = {
    "harm":     (0.9228, ["safe","harmless","protective","secure","careful"]),
    "honesty":  (0.9372, ["honest","accurate","truthful","transparent","precise"]),
    "autonomy": (0.8788, ["autonomous","respectful","unbiased","free","neutral"]),
    "fairness": (0.9196, ["fair","just","equitable","balanced","impartial"]),
}
BILISIM_ETIK  = 0.40
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

# AkbasCore 1.2 varsayılanları
DEFAULT_IVME  = 0.50
DEFAULT_SONUM = 0.30
DEFAULT_ZIRVE = 0.70
DEFAULT_TABAN = 0.20

IVME_MIN,  IVME_MAX  = 0.20, 0.80
SONUM_MIN, SONUM_MAX = 0.10, 0.60
ZIRVE_MIN, ZIRVE_MAX = 0.30, 1.00
TABAN_MIN, TABAN_MAX = 0.05, 0.40

# ── DRA-SECTOR 0.1 Sentetik Vektör Parametreleri ────────────────────────────
SECTOR_DEFAULTS = {
    "bukme_alfa": 0.35,   # rank-1 bükülme katsayısı
    "tork_beta":  0.55,   # aksiyel rotasyon torku
    "kapi_gama":  1.20,   # Hadamard kapı sıcaklığı
    "zeta":       1.00,   # kritik sönüm (overshoot engelleyici)
}

# Sentetik vektör modları
SECTOR_MODES = [
    "Kapalı — sadece AkbasCore 1.2",
    "Gauss Maske (boyut-merkezli çan eğrisi)",
    "Top-K Varyans Maskesi (dinamik)",
    "Rastgele Birim (seed'li)",
    "Hibrit — Gauss + Top-K çarpımı",
]

# =============================================================================
# AkbasSector (DRA-SECTOR_0.1) — Tek Model, İki Pass, Sentetik Vektör Katmanı
# =============================================================================
class AkbasSector:

    def __init__(self):
        print("🔱⬜ AKBASCORE DRA-SECTOR_0.1 | Qwen2.5-1.5B-Instruct")
        print("   Sentetik Gömünç ve Kontrol Tensör Yeniden Hizalaması")
        print("   (Synthetic Embedding & Control Tensor Realignment)")

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

        self.hidden_dim = self.model.config.hidden_size
        print(f"   Hidden dim: {self.hidden_dim}")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("   Pusula vektörü inşa ediliyor...")
        self._pusula=self._build_pusula().to(self.device)

        # Sentetik vektör önbelleği
        self._synthetic_cache = {}

        gc.collect()
        print("✅ AkbasCore DRA-SECTOR_0.1 hazır\n")

    # ── pusula inşası ─────────────────────────────────────────────────────────
    def _word_vec(self,word):
        W=self.model.model.embed_tokens.weight
        ids=self.tok(word,add_special_tokens=False)["input_ids"]
        if not ids: ids=[self.tok.unk_token_id or 0]
        return torch.stack([W[i].float().cpu().detach() for i in ids]).mean(0)

    def _build_pusula(self):
        wsum=None; wtot=0.0
        for _,(w,words) in CONSTITUTION.items():
            v=torch.stack([self._word_vec(wd) for wd in words]).mean(0)
            wsum=w*v if wsum is None else wsum+w*v; wtot+=w
        logic=torch.stack([self._word_vec(w) for w in LOGIC_ANCHORS]).mean(0)
        return F.normalize(
            BILISIM_ETIK*(wsum/wtot) + BILISIM_MANTIK*logic,
            dim=0).contiguous()

    # ── SENTETİK VEKTÖR İNŞASI ────────────────────────────────────────────────
    def _build_synthetic(self, mode, param_a=500, param_sigma=50,
                         param_k=0.01, param_seed=42, topk_dynamic=True):
        """
        Dışarıdan verilen sentetik vektör (V_ext).
        Kelimesiz — saf tensör geometrisi.
        """
        D = self.hidden_dim
        key = (mode, param_a, param_sigma, param_k, param_seed, topk_dynamic)
        if key in self._synthetic_cache:
            return self._synthetic_cache[key]

        if mode == SECTOR_MODES[0]:
            return None  # kapalı

        device = self.device
        dtype  = torch.float32

        if mode == SECTOR_MODES[1]:  # Gauss
            idx = torch.arange(D, dtype=dtype, device=device)
            v = torch.exp(-0.5 * ((idx - param_a) / max(param_sigma,1))**2)
            v = v - v.mean()  # DC bileşeni sıfırla
            v = F.normalize(v, dim=0)

        elif mode == SECTOR_MODES[2]:  # Top-K Varyans
            # Model embedding matrisinden varyans hesapla
            W = self.model.model.embed_tokens.weight.float()
            var = W.var(dim=0)  # [D]
            k_n = max(1, int(D * param_k))
            topv, topi = torch.topk(var, k_n)
            v = torch.zeros(D, dtype=dtype, device=device)
            v[topi] = torch.sqrt(topv)
            v = F.normalize(v, dim=0)

        elif mode == SECTOR_MODES[3]:  # Rastgele
            g = torch.Generator(device='cpu').manual_seed(int(param_seed))
            v = torch.randn(D, generator=g).to(device, dtype=dtype)
            v = F.normalize(v, dim=0)

        elif mode == SECTOR_MODES[4]:  # Hibrit
            idx = torch.arange(D, dtype=dtype, device=device)
            g = torch.exp(-0.5 * ((idx - param_a) / max(param_sigma,1))**2)
            W = self.model.model.embed_tokens.weight.float()
            var = W.var(dim=0)
            k_n = max(1, int(D * param_k))
            topv, topi = torch.topk(var, k_n)
            t = torch.zeros(D, dtype=dtype, device=device)
            t[topi] = torch.sqrt(topv)
            v = g * t
            v = F.normalize(v, dim=0)
        else:
            return None

        v = v.contiguous()
        self._synthetic_cache[key] = v
        return v

    # ── hook yönetimi ─────────────────────────────────────────────────────────
    def _remove_hooks(self,hooks):
        for h in hooks: h.remove()

    def _inject(self, p, log_buf, pc, ivme, sonum, zirve, taban,
                mode="observe", v_ext=None, sector_params=None):
        layers=self.model.model.layers; hooks=[]
        if mode == "observe":
            fn = _kernel.akbas_observe
        elif mode == "steer":
            fn = _kernel.akbas_steer
        else:  # sector
            fn = _kernel.akbas_sector

        def make_hook(i):
            def hook(mod,inp,out):
                hs=out[0] if isinstance(out,tuple) else out
                od=hs.dtype
                if od!=torch.float32: hs=hs.float()
                if not hs.is_contiguous(): hs=hs.contiguous()
                B,S,D=hs.shape; bs=B*S
                if mode == "sector":
                    st = fn(hs, v_ext,
                            sector_params["bukme_alfa"],
                            sector_params["tork_beta"],
                            sector_params["kapi_gama"],
                            sector_params["zeta"],
                            i, log_buf)
                else:
                    pc_local=torch.zeros(bs,dtype=torch.float32,device=hs.device) \
                             if pc.shape[0]!=bs else pc
                    st=fn(hs,p,ivme,i,sonum,zirve,taban,pc_local,log_buf)
                if od!=torch.float32: st=st.to(od)
                return (st,)+out[1:] if isinstance(out,tuple) else st
            return hook
        for i in range(min(N_LAYERS,len(layers))):
            hooks.append(layers[i].register_forward_hook(make_hook(i)))
        return hooks

    # ── tek forward pass ───────────────────────────────────────────────────────
    def _run_pass(self, inp, max_tokens, mode, ivme, sonum, zirve, taban,
                  v_ext=None, sector_params=None):
        pc      = torch.zeros(1, dtype=torch.float32, device=self.device)
        log_buf = torch.zeros(N_LAYERS, 4, dtype=torch.float32, device=self.device)
        hooks   = self._inject(self._pusula, log_buf, pc,
                               ivme, sonum, zirve, taban,
                               mode=mode, v_ext=v_ext, sector_params=sector_params)
        eos=list({self.tok.eos_token_id,
                  self.tok.convert_tokens_to_ids("<|im_end|>"),
                  self.tok.convert_tokens_to_ids("<|endoftext|>")}-{None,-1})
        t0=time.time()
        with torch.no_grad():
            out=self.model.generate(
                **inp, max_new_tokens=int(max_tokens),
                do_sample=True, temperature=TEMPERATURE,
                top_p=TOP_P, top_k=TOP_K,
                repetition_penalty=REP_PENALTY,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=eos)
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
        return text, ms, tps, n_in, n_out, log_buf.cpu()

    # ── log blokları ───────────────────────────────────────────────────────────
    def _log_header(self, label, ms, tps, n_in, n_out, ivme, sonum, zirve, taban,
                    sector_info=None):
        W=60; lines=[]
        lines.append("═"*W)
        lines.append(f"  {label}")
        lines.append("═"*W)
        lines.append(f"  ⏱  {ms:.0f} ms  |  {tps:.1f} tok/s  |  "
                     f"{n_in} in  |  {n_out} out tokens")
        lines.append(f"  MOTOR  ivme={ivme:.2f}  sönüm={sonum:.2f}  "
                     f"zirve={zirve:.2f}  taban={taban:.2f}")
        if sector_info:
            lines.append(f"  SECTOR {sector_info}")
        lines.append(f"  SABIT  oran=0.32  doyum=0.75  karşıt=-0.40  "
                     f"sapma=0.20  fren=0.30")
        lines.append(f"  KONFIGÜRASYON  katmanlar 0-{N_LAYERS-1}/28  "
                     f"bileşim={BILISIM_ETIK:.2f}/{BILISIM_MANTIK:.2f}")
        return lines

    def _log_table(self, lb, mode, sonum, zirve, taban):
        W=60; lines=[]
        lines.append("─"*W)
        if mode=="observe":
            lines.append("  SÜTUNLAR: cos(θ) · kb · kv · Δ-ref [teorik, uygulanmadı]")
        elif mode=="steer":
            lines.append("  SÜTUNLAR: cos(θ) · kb · kv · katki [hidden state'e yazıldı]")
        else:
            lines.append("  SÜTUNLAR: cos(θ) · warp_proj · rot_factor · gate")
            lines.append("  Metamorfoz: E' = (E + α·k·w·V̂ - β·k·sinθ·E⊥) ⊙ (1+γ·tanh(β·cosθ))")
        lines.append("─"*W)

        if mode == "sector":
            lines.append(f"  {'L':>3}  {'cos(θ)':>9}  {'warp':>9}  "
                         f"{'rot_f':>9}  {'gate':>9}")
            lines.append("─"*W)
            for i in range(N_LAYERS):
                cs   = lb[i,0].item()
                warp = lb[i,1].item()
                rot  = lb[i,2].item()
                gate = lb[i,3].item()
                if cs == 0.0 and gate == 0.0 and i > 0: continue
                lines.append(f"  {i:>3}  {cs:>+9.4f}  {warp:>+9.4f}  "
                             f"{rot:>+9.4f}  {gate:>+9.4f}")
            lines.append("─"*W)
            lines.append("═"*W)
            return lines

        lines.append(f"  {'L':>3}  {'cos(θ)':>8}  {'kb':>8}  {'kv':>8}  "
                     f"{'katki/Δ-ref':>12}  not")
        lines.append("─"*W)

        cos_vals=[]; kat_vals=[]
        for i in range(N_LAYERS):
            cs  = lb[i,0].item()
            kb  = lb[i,1].item()
            kv  = lb[i,2].item()
            kat = lb[i,3].item()
            if kb==0.0 and i>0: continue
            cos_vals.append(cs); kat_vals.append(kat)
            t       = float(i)
            kb_th   = zirve*math.exp(-sonum*t)*(1+sonum*t)+taban
            note    = "← eq." if abs(kb_th-taban)<0.05 else ""
            prefix  = "Δ" if mode=="observe" else " "
            lines.append(f"  {i:>3}  {cs:>+8.4f}  {kb:>8.5f}  "
                         f"{kv:>8.5f}  {prefix}{kat:>+11.6f}  {note}")

        lines.append("─"*W)
        if cos_vals:
            drift   = cos_vals[-1]-cos_vals[0]
            cos_pos = 100*sum(1 for c in cos_vals if c>0)/len(cos_vals)
            kat_sum = sum(kat_vals)
            lines.append(f"  cos(θ)  L0={cos_vals[0]:+.4f} → "
                         f"L{N_LAYERS-1}={cos_vals[-1]:+.4f}  drift={drift:+.4f}")
            lines.append(f"  cos>0: {cos_pos:.0f}%  |  "
                         f"{'katki' if mode=='steer' else 'Δ-ref'} toplam={kat_sum:+.6f}")
            final = "HIZALI ✓" if cos_vals[-1]>0 else "ZAYIF/KARŞIT ✗"
            lines.append(f"  Son yön: {final}")
        lines.append("═"*W)
        return lines

    def _build_compare(self, lb_v, lb_s):
        W=60; lines=[]
        lines.append("═"*W)
        lines.append("  DELTA KARŞILAŞTIRMA — VANILLA vs STEERED/SECTOR")
        lines.append("─"*W)
        lines.append(f"  {'L':>3}  {'cos_V':>8}  {'cos_S':>8}  "
                     f"{'Δcos':>8}  {'Δkatki':>10}")
        lines.append("─"*W)
        dcos_sum=0.0; dkat_sum=0.0; rows=0
        for i in range(N_LAYERS):
            csv = lb_v[i,0].item(); kbv = lb_v[i,1].item()
            css = lb_s[i,0].item()
            katv= lb_v[i,3].item()
            kats= lb_s[i,3].item()
            if kbv==0.0 and i>0: continue
            dcos = css - csv
            dkat = kats - katv
            dcos_sum += dcos; dkat_sum += dkat; rows+=1
            lines.append(f"  {i:>3}  {csv:>+8.4f}  {css:>+8.4f}  "
                         f"{dcos:>+8.4f}  {dkat:>+10.6f}")
        lines.append("─"*W)
        if rows:
            lines.append(f"  Δcos ort={dcos_sum/rows:+.4f}  "
                         f"Δkatki ort={dkat_sum/rows:+.6f}")
            lines.append(f"  Toplam cos kayması: {dcos_sum:+.4f}")
        lines.append("═"*W)
        return lines

    def _build_sector_compare(self, lb_v, lb_sec):
        W=60; lines=[]
        lines.append("═"*W)
        lines.append("  SECTOR DELTA — VANILLA vs DRA-SECTOR 0.1")
        lines.append("─"*W)
        lines.append(f"  {'L':>3}  {'cos_V':>9}  {'cos_sec':>9}  "
                     f"{'gate':>9}  {'rot_f':>9}")
        lines.append("─"*W)
        dcos_sum=0.0; rows=0
        for i in range(N_LAYERS):
            csv = lb_v[i,0].item(); kbv = lb_v[i,1].item()
            css = lb_sec[i,0].item()
            gate= lb_sec[i,3].item()
            rot = lb_sec[i,2].item()
            if kbv==0.0 and i>0: continue
            dcos = css - csv
            dcos_sum += dcos; rows += 1
            lines.append(f"  {i:>3}  {csv:>+9.4f}  {css:>+9.4f}  "
                         f"{gate:>+9.4f}  {rot:>+9.4f}")
        lines.append("─"*W)
        if rows:
            lines.append(f"  Δcos ort={dcos_sum/rows:+.4f}  "
                         f"Toplam kayma={dcos_sum:+.4f}")
        lines.append("═"*W)
        return lines

    # ── ANA ÇALIŞMA FONKSİYONU ────────────────────────────────────────────────
    def run(self, prompt, max_tokens, ivme, sonum, zirve, taban,
            sector_mode, bukme_alfa, tork_beta, kapi_gama, zeta,
            gauss_center, gauss_sigma, topk_ratio, rand_seed):
        if not prompt.strip():
            return "", ""

        ivme  = float(max(IVME_MIN,  min(IVME_MAX,  ivme)))
        sonum = float(max(SONUM_MIN, min(SONUM_MAX, sonum)))
        zirve = float(max(ZIRVE_MIN, min(ZIRVE_MAX, zirve)))
        taban = float(max(TABAN_MIN, min(TABAN_MAX, taban)))

        sector_params = {
            "bukme_alfa": float(bukme_alfa),
            "tork_beta":  float(tork_beta),
            "kapi_gama":  float(kapi_gama),
            "zeta":       float(max(0.0, min(1.0, zeta))),
        }

        # Sentetik vektör inşası
        sector_active = sector_mode != SECTOR_MODES[0]
        v_ext = None
        sector_info = None
        if sector_active:
            mode_idx = SECTOR_MODES.index(sector_mode) if sector_mode in SECTOR_MODES else 0
            v_ext = self._build_synthetic(
                sector_mode,
                param_a=int(gauss_center),
                param_sigma=float(gauss_sigma),
                param_k=float(topk_ratio),
                param_seed=int(rand_seed),
                topk_dynamic=True)
            if v_ext is not None:
                v_ext = v_ext.to(self.device).contiguous()
                sector_info = (f"mode={sector_mode[:20]}...  "
                               f"α={bukme_alfa:.2f}  β={tork_beta:.2f}  "
                               f"γ={kapi_gama:.2f}  ζ={zeta:.2f}")

        prompt_text = (f"<|im_start|>system\nYou are a helpful, accurate, and "
                       f"thoughtful assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
                       f"<|im_start|>assistant\n")
        inp = self.tok(prompt_text, return_tensors="pt").to(self.device)

        # ── PASS A: VANILLA ───────────────────────────────────────────────────
        print(f"   Pass A: Vanilla Observer")
        txt_v, ms_v, tps_v, n_in_v, n_out_v, lb_v = self._run_pass(
            inp, max_tokens, "observe", ivme, sonum, zirve, taban)
        gc.collect()

        # ── PASS B: STEERED veya SECTOR ───────────────────────────────────────
        if sector_active and v_ext is not None:
            print(f"   Pass B: DRA-SECTOR 0.1  [{sector_mode}]")
            txt_s, ms_s, tps_s, n_in_s, n_out_s, lb_s = self._run_pass(
                inp, max_tokens, "sector", ivme, sonum, zirve, taban,
                v_ext=v_ext, sector_params=sector_params)
            pass_label = "🔱 DRA-SECTOR 0.1 — sentetik vektör metamorfozu uygulandı"
        else:
            print(f"   Pass B: Steered  [AkbasCore 1.2]")
            txt_s, ms_s, tps_s, n_in_s, n_out_s, lb_s = self._run_pass(
                inp, max_tokens, "steer", ivme, sonum, zirve, taban)
            pass_label = "🔱 AKBASCORE STEERED — katki uygulandı"
            sector_info = None

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ── VANILLA PANEL ─────────────────────────────────────────────────────
        van_lines = self._log_header(
            "⬜ VANILLA OBSERVER — katki=0 (hidden state değişmedi)",
            ms_v, tps_v, n_in_v, n_out_v, ivme, sonum, zirve, taban)
        van_lines += self._log_table(lb_v, "observe", sonum, zirve, taban)
        van_out = txt_v + "\n\n" + "\n".join(van_lines)

        # ── STEERED/SECTOR PANEL ──────────────────────────────────────────────
        st_lines = self._log_header(pass_label,
            ms_s, tps_s, n_in_s, n_out_s, ivme, sonum, zirve, taban,
            sector_info=sector_info)
        st_lines += self._log_table(lb_s, "sector" if sector_active else "steer",
                                     sonum, zirve, taban)
        if sector_active:
            st_lines += self._build_sector_compare(lb_v, lb_s)
        else:
            st_lines += self._build_compare(lb_v, lb_s)
        st_out = txt_s + "\n\n" + "\n".join(st_lines)

        return van_out, st_out

# =============================================================================
# YÜKLEME
# =============================================================================
print("\n"+"="*58)
print("🔱⬜ AKBASCORE DRA-SECTOR_0.1 | QWEN2.5-1.5B | YÜKLENIYOR...")
print("="*58+"\n")
sector_engine = AkbasSector()
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI
# =============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

body { background: #04050a; margin: 0; }

.gradio-container {
    max-width: 1340px !important;
    margin: 0 auto !important;
    background: #04050a !important;
    font-family: 'JetBrains Mono', monospace !important;
}

#akbas-header {
    text-align: center;
    padding: 28px 0 16px;
    border-bottom: 1px solid #111828;
}
#akbas-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.00rem;
    font-weight: 500;
    letter-spacing: .18em;
    margin: 0 0 6px;
    background: linear-gradient(90deg, #5588cc, #33cc77, #cc8844);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#akbas-header .sub {
    font-size: .55rem;
    color: #1e2a3a;
    letter-spacing: .10em;
    margin: 0;
}

#motor-panel {
    background: #060810;
    border: 1px solid #0e1520;
    border-top: 2px solid #1a2a44;
    padding: 18px 24px 16px;
}
#motor-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: .62rem;
    letter-spacing: .20em;
    color: #2a4060;
    margin: 0 0 4px;
}
#motor-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: .54rem;
    color: #182030;
    letter-spacing: .05em;
    margin: 0 0 14px;
    line-height: 1.6;
}
#motor-warning {
    font-family: 'JetBrains Mono', monospace;
    font-size: .52rem;
    color: #2a3a1a;
    letter-spacing: .04em;
    margin: 10px 0 0;
    padding: 6px 10px;
    border-left: 2px solid #2a3a20;
}

#sector-panel {
    background: #0a0810;
    border: 1px solid #1e1830;
    border-top: 2px solid #3a2a5a;
    padding: 18px 24px 16px;
}
#sector-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: .62rem;
    letter-spacing: .20em;
    color: #5a4080;
    margin: 0 0 4px;
}
#sector-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: .54rem;
    color: #2a1e40;
    letter-spacing: .05em;
    margin: 0 0 14px;
    line-height: 1.6;
}
#sector-formula {
    font-family: 'JetBrains Mono', monospace;
    font-size: .52rem;
    color: #3a2a5a;
    letter-spacing: .04em;
    margin: 10px 0 0;
    padding: 6px 10px;
    border-left: 2px solid #3a2a5a;
    line-height: 1.7;
}

input[type=range] { accent-color: #2a5080 !important; height: 2px !important; }
input[type=range]::-webkit-slider-thumb {
    background: #3a7acc !important; width: 10px !important; height: 10px !important;
}

.gradio-slider label span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .60rem !important;
    color: #3a5878 !important;
    letter-spacing: .06em !important;
}

#inp-section {
    background: #050710;
    border: 1px solid #0c1220;
    padding: 14px 24px;
}
#inp textarea {
    background: #070910 !important;
    color: #8aaccc !important;
    border: 1px solid #0e1828 !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .80rem !important;
    line-height: 1.70 !important;
}

#btn {
    background: linear-gradient(135deg, #0a1220, #1a1040) !important;
    color: #aa88cc !important;
    border: 1px solid #2a1a50 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    letter-spacing: .18em !important;
    border-radius: 2px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    margin: 10px 24px 0 !important;
}
#btn:hover {
    background: linear-gradient(135deg, #142040, #2a2060) !important;
    color: #ccaaee !important;
    box-shadow: 0 0 14px #6644aa20 !important;
    border-color: #4a3a80 !important;
}

.panel-lbl-v {
    font-family: 'JetBrains Mono', monospace;
    font-size: .58rem;
    letter-spacing: .14em;
    color: #336688;
    text-align: center;
    padding: 6px 0 3px;
    border-bottom: 1px solid #0c1828;
    margin-bottom: 4px;
}
.panel-lbl-s {
    font-family: 'JetBrains Mono', monospace;
    font-size: .58rem;
    letter-spacing: .14em;
    color: #8855cc;
    text-align: center;
    padding: 6px 0 3px;
    border-bottom: 1px solid #181030;
    margin-bottom: 4px;
}

#out_v textarea {
    background: #040608 !important;
    color: #5599cc !important;
    border: 1px solid #0a1420 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .74rem !important;
    line-height: 1.72 !important;
    border-radius: 2px !important;
}
#out_s textarea {
    background: #080610 !important;
    color: #aa88dd !important;
    border: 1px solid #140e28 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .74rem !important;
    line-height: 1.72 !important;
    border-radius: 2px !important;
}

#ver-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: .48rem;
    color: #0e1820;
    text-align: center;
    padding: 6px 0 2px;
    letter-spacing: .12em;
}
"""

LBL_IVME = (f"ivme — steering velocity: master amplitude dial  "
            f"[band {IVME_MIN}–{IVME_MAX}]")
LBL_SONUM = (f"sönüm — decay rate: resonance fade speed  "
             f"[band {SONUM_MIN}–{SONUM_MAX}]")
LBL_ZIRVE = (f"zirve — amplitude: peak push strength  "
             f"[band {ZIRVE_MIN}–{ZIRVE_MAX}]")
LBL_TABAN = (f"taban — permanent floor: min alignment pressure  "
             f"[band {TABAN_MIN}–{TABAN_MAX}]")

LBL_ALFA = "bukme_alfa — rank-1 düşük rütbeli bükülme katsayısı (α) [0.0–0.8]"
LBL_BETA = "tork_beta — aksiyel rotasyon torku (β) [0.0–1.2]"
LBL_GAMA = "kapi_gama — Hadamard kapı sıcaklığı (γ) [0.0–2.5]"
LBL_ZETA = "zeta — kritik sönüm ζ (1.0 → overshoot yok) [0.0–1.0]"

with gr.Blocks(title="🔱 AkbasCore DRA-SECTOR_0.1") as demo:

    gr.HTML("""
    <div id="akbas-header">
      <h1>🔱 AKBASCORE DRA-SECTOR_0.1</h1>
      <p class="sub">
        SENTETİK GÖMÜNÇ VE KONTROL TENSÖR YENİDEN HİZALAMASI
        (SYNTHETIC EMBEDDING &amp; CONTROL TENSOR REALIGNMENT) &nbsp;·&nbsp;
        DRA — SÖNÜMLÜ REZONANS HİZALAMA (DAMPED RESONANCE ALIGNMENT)
        GÖVDESİ ÜZERİNE İNŞA EDİLMİŞTİR &nbsp;·&nbsp;
        QWEN2.5-1.5B-INSTRUCT &nbsp;·&nbsp; © AKBAŞ
      </p>
    </div>
    """)

    # ── AkbasCore 1.2 Motor Kolları ───────────────────────────────────────────
    with gr.Column(elem_id="motor-panel"):
        gr.HTML("""
        <div id="motor-title">⚙ AKBASCORE 1.2 — TEMEL MOTOR AYAR KOLLARI</div>
        <div id="motor-desc">
          Damped resonance alignment — pusula yönünde hidden state basıncı.
          Varsayılan değerler AkbasCore 1.1 kalibrasyonudur.
        </div>
        """)
        with gr.Row():
            sl_ivme = gr.Slider(IVME_MIN, IVME_MAX, DEFAULT_IVME, step=0.05, label=LBL_IVME)
            sl_sonum = gr.Slider(SONUM_MIN, SONUM_MAX, DEFAULT_SONUM, step=0.05, label=LBL_SONUM)
        with gr.Row():
            sl_zirve = gr.Slider(ZIRVE_MIN, ZIRVE_MAX, DEFAULT_ZIRVE, step=0.05, label=LBL_ZIRVE)
            sl_taban = gr.Slider(TABAN_MIN, TABAN_MAX, DEFAULT_TABAN, step=0.05, label=LBL_TABAN)

    # ── DRA-SECTOR 0.1 — Sentetik Vektör Paneli ─────────────────────────────
    with gr.Column(elem_id="sector-panel"):
        gr.HTML("""
        <div id="sector-title">🧬 DRA-SECTOR 0.1 — SENTETİK KONTROL VEKTÖRÜ (SYNTHETIC CONTROL VECTOR)</div>
        <div id="sector-desc">
          AkbasCore 1.2'nin pusula (Compass Vector) alanı burada genişletiliyor:
          dışarıdan verilen Sentetik Kontrol Vektörü (V_ext), hidden state'e
          eklenmez (h+v değil); iç embedding — Gömünç — uzayını <b>büker</b>,
          <b>döndürür</b> ve <b>kapılar</b>. Sistem matrisi A(V_ext) anlık
          olarak modüle edilir. Bu 3 mekanizma (bükülme, rotasyon, kapılama)
          ve kritik sönüm kontrolü (ζ) AkbasCore 1.2'de bulunmuyordu.
        </div>
        """)

        sl_sector_mode = gr.Dropdown(
            choices=SECTOR_MODES, value=SECTOR_MODES[0],
            label="SENTETİK VEKTÖR MODU — V_ext oluşturma yöntemi")

        gr.HTML("""
        <div id="sector-formula">
          <b>MEKANİZMA 1</b> — Rank-1 Bükülme: &nbsp; E' = E + α·k·(E·V̂)·V̂<br>
          <b>MEKANİZMA 2</b> — Aksiyel Rotasyon: &nbsp; E' = E - β·k·sinθ·E⊥<br>
          <b>MEKANİZMA 3</b> — Kapılamalı Modülasyon: &nbsp; E' = E ⊙ (1+γ·tanh(β·cosθ))<br>
          <b>KONTROL</b> — Kritik Sönüm: k = 1 - exp(-ζ), &nbsp; ζ=1 → overshoot yok
        </div>
        """)

        with gr.Row():
            sl_alfa = gr.Slider(0.0, 0.80, SECTOR_DEFAULTS["bukme_alfa"],
                                step=0.05, label=LBL_ALFA)
            sl_beta = gr.Slider(0.0, 1.20, SECTOR_DEFAULTS["tork_beta"],
                                step=0.05, label=LBL_BETA)
        with gr.Row():
            sl_gama = gr.Slider(0.0, 2.50, SECTOR_DEFAULTS["kapi_gama"],
                                step=0.05, label=LBL_GAMA)
            sl_zeta = gr.Slider(0.0, 1.00, SECTOR_DEFAULTS["zeta"],
                                step=0.05, label=LBL_ZETA)

        gr.HTML("""<div style="font-size:.55rem;color:#2a1e40;letter-spacing:.10em;
                    margin:14px 0 6px;font-family:'JetBrains Mono',monospace;">
                    ▼ SENTETİK VEKTÖR GEOMETRİSİ (moda bağlı parametreler)</div>""")

        with gr.Row():
            sl_gauss_c = gr.Slider(0, 1536, value=500, step=1,
                label="Gauss merkez boyut [0–1535]")
            sl_gauss_s = gr.Slider(5, 200, value=50, step=5,
                label="Gauss sigma σ")
        with gr.Row():
            sl_topk = gr.Slider(0.001, 0.05, value=0.01, step=0.001,
                label="Top-K varyans oranı [0.001–0.05]")
            sl_seed = gr.Slider(0, 10000, value=42, step=1,
                label="Rastgele seed")

        gr.HTML("""
        <div id="motor-warning" style="border-left-color:#3a2a5a;color:#3a2a5a;">
          <b>V_ext = 0</b> uyarısı: Sentetik vektör modu "Kapalı" iken Sector katmanı
          devre dışıdır; yalnızca AkbasCore 1.2 motoru çalışır.
        </div>
        """)

    # ── Girdi + Buton ─────────────────────────────────────────────────────────
    with gr.Column(elem_id="inp-section"):
        inp = gr.Textbox(
            label="► INPUT — aynı soru her iki pass'a gönderilir",
            lines=4,
            placeholder="Sorunuzu buraya yazın...",
            elem_id="inp")
        with gr.Row():
            sl_tok = gr.Slider(64, 1024, value=512, step=64,
                               label="MAX TOKENS — output budget", scale=4)
            btn = gr.Button("▶▶  DUAL RUN", variant="primary",
                            elem_id="btn", scale=1)

    # ── Çıktı Panelleri ───────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="panel-lbl-v">⬜ VANILLA OBSERVER — '
                    'katki = 0  |  hidden state unchanged</div>')
            out_v = gr.Textbox(label="OUTPUT + OBSERVATION LOG",
                               lines=40, interactive=False, elem_id="out_v")
            gr.HTML("""
            <button onclick="
                var el = document.querySelector('#out_v textarea');
                if(el){ navigator.clipboard.writeText(el.value);
                this.textContent='COPIED!';
                setTimeout(()=>this.textContent='COPY VANILLA',1800); }
            " style="margin:4px 0 8px;padding:6px 18px;background:#0e1e34;color:#7aaddd;
            border:1px solid #1e3a5a;border-radius:3px;cursor:pointer;
            font-family:'Courier New',monospace;font-size:11px;letter-spacing:.08em;">
            COPY VANILLA</button>
            """)
        with gr.Column():
            gr.HTML('<div class="panel-lbl-s">🔱 STEERED / DRA-SECTOR 0.1 — '
                    'sentetik metamorfoz uygulandı</div>')
            out_s = gr.Textbox(label="OUTPUT + SECTOR LOG + DELTA COMPARISON",
                               lines=40, interactive=False, elem_id="out_s")
            gr.HTML("""
            <button onclick="
                var el = document.querySelector('#out_s textarea');
                if(el){ navigator.clipboard.writeText(el.value);
                this.textContent='COPIED!';
                setTimeout(()=>this.textContent='COPY SECTOR',1800); }
            " style="margin:4px 0 8px;padding:6px 18px;background:#181030;color:#aa88dd;
            border:1px solid #2a1a50;border-radius:3px;cursor:pointer;
            font-family:'Courier New',monospace;font-size:11px;letter-spacing:.08em;">
            COPY SECTOR</button>
            """)

    gr.HTML("""
    <div id="ver-tag">
      AKBASCORE DRA-SECTOR_0.1 &nbsp;·&nbsp; TERMİNOLOJİ: pusula (compass) ·
      katki (contribution) · taban (floor) · sönüm (decay) · zirve (amplitude) ·
      ivme (velocity) · V_ext (synthetic control vector) · bukme_alfa (warp α) ·
      tork_beta (torque β) · kapi_gama (gate γ) · zeta (critical damping ζ)
      &nbsp;·&nbsp; © AKBAŞ &nbsp;·&nbsp; APACHE 2.0
    </div>
    """)

    btn.click(
        fn=sector_engine.run,
        inputs=[inp, sl_tok, sl_ivme, sl_sonum, sl_zirve, sl_taban,
                sl_sector_mode, sl_alfa, sl_beta, sl_gama, sl_zeta,
                sl_gauss_c, sl_gauss_s, sl_topk, sl_seed],
        outputs=[out_v, out_s])
    inp.submit(
        fn=sector_engine.run,
        inputs=[inp, sl_tok, sl_ivme, sl_sonum, sl_zirve, sl_taban,
                sl_sector_mode, sl_alfa, sl_beta, sl_gama, sl_zeta,
                sl_gauss_c, sl_gauss_s, sl_topk, sl_seed],
        outputs=[out_v, out_s])

print("🚀 AkbasCore DRA-SECTOR_0.1 Gradio başlatılıyor...")
demo.launch(
    share=True,
    debug=False,
    theme=gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono")),
    css=CSS,
)
