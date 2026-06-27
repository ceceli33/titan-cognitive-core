# =============================================================================
# 🔱⬜ AKBASCORE 1.2 | QWEN2.5-1.5B-INSTRUCT — DUAL MODE + MOTOR AYAR KOLLARI
# =============================================================================
#
# AkbasCore, bir transformer dil modelinin gizli durumunu (hidden state)
# her katmanda matematiksel olarak yönlendiren bir C++ çıkarım motoru.
# Bu sistem yeniden eğitim yapmaz, ağırlıklara dokunmaz, prompt değiştirmez.
# Müdahale bfloat16 hassasiyet tabanının altında kalır — standart ölçüm
# araçları bunu göremez, çıktı kalitesi görebilir.
#
# VERSION 1.2 YENİLİKLERİ:
#   — Gradio UI'da 6 aktif motor ayar kolu (ayar)
#   — Tüm AkbasCore terminolojisi Türkçe isimlendirildi
#   — Her parametrenin çalışma aralığı kalibre güvenli bantla sınırlandı
#   — Kod içi açıklamalar: Türkçe terim + İngilizce açıklama
#
# PASS YAPISI:
#   Pass A (⬜ VANILLA)  : akbas_observe() → katki=0 → hidden state DOKUNULMAZ
#   Pass B (🔱 STEERED)  : akbas_steer()  → katki≠0 → hidden state YÖNLENDİRİLİR
#
# AkbasCore Terminolojisi (© Akbaş — tüm isimler Türkçe, açıklamalar İngilizce):
#
#   pusula   — compass vector: ethical-logical direction in hidden state space
#   katki    — contribution: scalar value written to hidden state per layer
#   taban    — permanent floor: minimum continuous alignment pressure (P∞)
#   sönüm    — decay rate: how fast resonance peak fades across layers (ω)
#   zirve    — amplitude: peak push strength at early layers (A)
#   ivme     — steering velocity: master amplitude dial (v₀)
#   terazi   — constitution weights: ethical balance values per category
#   bileşim  — blend ratio: ethics/logic mixture for pusula construction
#   oran     — scale factor: base multiplier in katki formula [motor sabit]
#   doyum    — saturation upper threshold: reduces push when cos(θ) > X [motor sabit]
#   karşıt   — counter threshold: amplifies push when cos(θ) < X [motor sabit]
#   sapma    — uncertainty corrector: alignment deviation adjuster [motor sabit]
#   fren     — velocity brake: feedback damping on rate of change [motor sabit]
#   sınır    — max katki ratio: clamp ceiling relative to hidden norm [motor sabit]
#
# HOW TO RUN:
#   1. colab.research.google.com → New notebook
#   2. Runtime → Change runtime type → CPU (veya T4 GPU)
#   3. Bu dosyanın tamamını TEK hücreye yapıştır → Çalıştır
#   4. Gradio linkini aç → Motor ayarlarını seç → soruyu yaz → DUAL RUN
#
# Model   : Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)
# Kernel  : C++ — akbas_observe() + akbas_steer()
# Varsayılan ayarlar: ivme=0.50 sönüm=0.30 zirve=0.70 taban=0.20
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
# C++ ÇEKİRDEK — AkbasCore Motor
# Tüm sabit isimler AkbasCore terminolojisiyle etiketlendi.
# =============================================================================
_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

/*
 * AkbasCore C++ Motor Sabitleri [motor sabitler — değiştirmek derleme gerektirir]
 *
 * oran    (scale factor)             : 0.32  — base multiplier in katki formula
 * doyum   (saturation upper)         : 0.75  — reduces push above this cos(θ)
 * karşıt  (counter threshold)        : -0.40 — amplifies push below this cos(θ)
 * sapma   (uncertainty corrector)    : 0.20  — uncertainty-based omega adjustment
 * fren    (velocity brake)           : 0.30  — rate-of-change damping coefficient
 * sınır   (max katki ratio)          : 0.045 — katki ceiling as fraction of hidden norm
 * sınır_alt (min katki floor)        : 0.04  — absolute minimum katki ceiling
 * sınır_üst (max katki ceiling)      : 0.20  — absolute maximum katki ceiling
 */

// ─── PASS A: GÖZLEM — hidden state'e DOKUNMAZ (observe mode) ─────────────────
torch::Tensor akbas_observe(
    torch::Tensor hidden,
    torch::Tensor pusula,   // compass vector: ethical-logical direction in hidden space
    float ivme,             // steering velocity: master amplitude dial (v₀)
    int layer_idx,
    float sonum,            // decay rate: resonance fade speed across layers (ω)
    float zirve,            // amplitude: peak push strength at early layers (A)
    float taban,            // permanent floor: minimum alignment pressure (P∞)
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

        // sapma (uncertainty corrector): adjusts omega based on alignment distance
        float unc = 1.0f - fabsf(cs);
        float dw  = sonum + unc * 0.20f;  // 0.20 = sapma sabiti

        // kb — layer gain from damped resonance formula
        float kb  = zirve * expf(-dw * t) * (1.0f + dw * t) + taban;

        // fren (velocity brake): closed-loop gain adjustment via rate of change
        float dr = std::clamp(cs - pcp[idx], -0.15f, 0.15f);
        if (cs > 0.80f && dr < 0.0f) dr *= 0.30f;  // 0.30 = fren katsayısı

        float kv = kb;
        if      (dr > 0) kv *= (1.0f - dr    * 0.30f);  // fren — reducing push
        else if (dr < 0) kv *= (1.0f + (-dr) * 0.30f);  // fren — increasing push
        kv = std::clamp(kv, 0.05f, 1.0f);
        pcp[idx] = cs;

        // son — saturation factor using doyum and karşıt thresholds
        float son = 1.0f;
        if      (cs >  0.75f) son = (1.0f - cs) / 0.25f;  // doyum: 0.75 üst eşik
        else if (cs < -0.40f) son = 1.6f;                   // karşıt: -0.40 alt eşik

        // sınır (max katki ratio): ceiling relative to hidden state norm
        float mk    = std::clamp(norm * 0.045f, 0.04f, 0.20f);  // sınır sabitleri
        // oran (scale factor): 0.32 — base multiplier in katki formula
        float katki = std::clamp(ivme * cs * kv * 0.32f * son, -mk, mk);

        // ⚠ GÖZLEM MODU: tok[j] += katki * pp[j]  ← UYGULANMIYOR
        // observe mode: katki is computed but NOT written to hidden state

        if (S > 1 && b == 0 && s == 0 && layer_idx < 20) {
            lbp[layer_idx * 4 + 0] = cs;
            lbp[layer_idx * 4 + 1] = kb;
            lbp[layer_idx * 4 + 2] = kv;
            lbp[layer_idx * 4 + 3] = katki;  // delta-ref: theoretical, not applied
        }
    }
    return h;   // DEĞİŞTİRİLMEMİŞ — hidden state unchanged
}

// ─── PASS B: STEERING — hidden state YÖNLENDİRİLİR (steer mode) ──────────────
torch::Tensor akbas_steer(
    torch::Tensor hidden,
    torch::Tensor pusula,   // compass vector: ethical-logical direction in hidden space
    float ivme,             // steering velocity: master amplitude dial (v₀)
    int layer_idx,
    float sonum,            // decay rate: resonance fade speed across layers (ω)
    float zirve,            // amplitude: peak push strength at early layers (A)
    float taban,            // permanent floor: minimum alignment pressure (P∞)
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

        // sapma (uncertainty corrector): adjusts omega based on alignment distance
        float unc = 1.0f - fabsf(cs);
        float dw  = sonum + unc * 0.20f;  // 0.20 = sapma sabiti

        // kb — layer gain from damped resonance formula
        float kb  = zirve * expf(-dw * t) * (1.0f + dw * t) + taban;

        // fren (velocity brake): closed-loop gain adjustment via rate of change
        float dr = std::clamp(cs - pcp[idx], -0.15f, 0.15f);
        if (cs > 0.80f && dr < 0.0f) dr *= 0.30f;  // 0.30 = fren katsayısı

        float kv = kb;
        if      (dr > 0) kv *= (1.0f - dr    * 0.30f);  // fren — reducing push
        else if (dr < 0) kv *= (1.0f + (-dr) * 0.30f);  // fren — increasing push
        kv = std::clamp(kv, 0.05f, 1.0f);
        pcp[idx] = cs;

        // son — saturation factor using doyum and karşıt thresholds
        float son = 1.0f;
        if      (cs >  0.75f) son = (1.0f - cs) / 0.25f;  // doyum: 0.75 üst eşik
        else if (cs < -0.40f) son = 1.6f;                   // karşıt: -0.40 alt eşik

        // sınır (max katki ratio): ceiling relative to hidden state norm
        float mk    = std::clamp(norm * 0.045f, 0.04f, 0.20f);  // sınır sabitleri
        // oran (scale factor): 0.32 — base multiplier in katki formula
        float katki = std::clamp(ivme * cs * kv * 0.32f * son, -mk, mk);

        // ✅ STEERING MODU: katki pusula yönünde hidden state'e YAZILIYOR
        // steer mode: katki is computed AND written to hidden state
        for (int j = 0; j < D; ++j) tok[j] += katki * pp[j];

        if (S > 1 && b == 0 && s == 0 && layer_idx < 20) {
            lbp[layer_idx * 4 + 0] = cs;
            lbp[layer_idx * 4 + 1] = kb;
            lbp[layer_idx * 4 + 2] = kv;
            lbp[layer_idx * 4 + 3] = katki;  // actual value applied to hidden state
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

for _n in ["akbas_12_f1","akbas_12_f2"]:
    for _r in ["/tmp/akbas_12",os.path.expanduser("~/.cache/torch_extensions")]:
        _p=os.path.join(_r,_n)
        if os.path.exists(_p): shutil.rmtree(_p,ignore_errors=True)

def _build(name):
    d=f"/tmp/akbas_12/{name}"; os.makedirs(d,exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name, cpp_sources=_CPP,
        functions=["akbas_observe","akbas_steer"],
        build_directory=d, verbose=False,
        extra_cflags=["-O2","-std=c++17"])

try:    _kernel=_build("akbas_12_f1")
except Exception as e:
    print(f"  retry ({e})..."); _kernel=_build("akbas_12_f2")
print("✅ AkbasCore 1.2 C++ çekirdeği derlendi")

# =============================================================================
# SABİT PARAMETRELER
# =============================================================================
MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE  = 0.65
TOP_P        = 0.90
TOP_K        = 50
REP_PENALTY  = 1.15
N_LAYERS     = 20

# --- AkbasCore Anayasa (Constitution) ---
# terazi — constitution weights: ethical balance values per category
# bileşim — blend ratio: 0.40 ethics / 0.60 logic anchors
CONSTITUTION = {
    "harm":     (0.9228, ["safe","harmless","protective","secure","careful"]),
    "honesty":  (0.9372, ["honest","accurate","truthful","transparent","precise"]),
    "autonomy": (0.8788, ["autonomous","respectful","unbiased","free","neutral"]),
    "fairness": (0.9196, ["fair","just","equitable","balanced","impartial"]),
}
# bileşim oranları — blend ratio for pusula construction
BILISIM_ETIK  = 0.40  # ethics weight in pusula
BILISIM_MANTIK = 0.60  # logic weight in pusula

LOGIC_ANCHORS = [
    "logical","empirical","systematic","structured","verifiable",
    "analyze","precise","deterministic","sequential","causal",
    "rigorous","impossible","contradiction","identify","optimize",
]
DISCLAIMER_MARKERS = [
    "as an ai","as a language model","i apologize","i must clarify",
    "i'm unable to","i am unable to","i cannot be certain",
]

# --- Ayar Kolları Varsayılan Değerleri (AkbasCore 1.1 kalibrasyonu) ---
DEFAULT_IVME  = 0.50   # ivme  — steering velocity
DEFAULT_SONUM = 0.30   # sönüm — decay rate
DEFAULT_ZIRVE = 0.70   # zirve — amplitude
DEFAULT_TABAN = 0.20   # taban — permanent floor

# --- Ayar Kolları Güvenli Bantları ---
# Bu aralıklar çıktıyı belirgin şekilde değiştirecek kadar geniş,
# sistemi bozacak kadar değil. Tasarımcı kalibrasyonu: AkbasCore 1.1
IVME_MIN,  IVME_MAX  = 0.20, 0.80   # ivme  safe band
SONUM_MIN, SONUM_MAX = 0.10, 0.60   # sönüm safe band
ZIRVE_MIN, ZIRVE_MAX = 0.30, 1.00   # zirve safe band
TABAN_MIN, TABAN_MAX = 0.05, 0.40   # taban safe band

# =============================================================================
# AkbasDual 1.2 — TEK MODEL, İKİ PASS, AKTİF MOTOR AYAR KOLLARI
# =============================================================================
class AkbasDual:

    def __init__(self):
        print("🔱⬜ AKBASCORE 1.2 | Qwen2.5-1.5B-Instruct")
        print("   Tek model → iki ayrı forward pass")
        print(f"   Varsayılan motor: ivme={DEFAULT_IVME} sönüm={DEFAULT_SONUM} "
              f"zirve={DEFAULT_ZIRVE} taban={DEFAULT_TABAN}")
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

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("   Pusula vektörü inşa ediliyor...")
        self._pusula=self._build_pusula().to(self.device)
        gc.collect()
        print("✅ AkbasCore 1.2 hazır\n")

    # ── pusula inşası ─────────────────────────────────────────────────────────
    # pusula — compass vector built from model's own embedding table
    # terazi — constitution weights applied here
    # bileşim — 0.40 etik / 0.60 mantık karışımı
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
        # bileşim oranı: BILISIM_ETIK + BILISIM_MANTIK = 1.0
        return F.normalize(
            BILISIM_ETIK*(wsum/wtot) + BILISIM_MANTIK*logic,
            dim=0).contiguous()

    # ── hook yönetimi ──────────────────────────────────────────────────────────
    def _remove_hooks(self,hooks):
        for h in hooks: h.remove()

    def _inject(self, p, log_buf, pc, ivme, sonum, zirve, taban, mode="observe"):
        layers=self.model.model.layers; hooks=[]
        fn = _kernel.akbas_observe if mode=="observe" else _kernel.akbas_steer
        def make_hook(i):
            def hook(mod,inp,out):
                hs=out[0] if isinstance(out,tuple) else out
                od=hs.dtype
                if od!=torch.float32: hs=hs.float()
                if not hs.is_contiguous(): hs=hs.contiguous()
                B,S,D=hs.shape; bs=B*S
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
    def _run_pass(self, inp, max_tokens, mode, ivme, sonum, zirve, taban):
        pc      = torch.zeros(1, dtype=torch.float32, device=self.device)
        log_buf = torch.zeros(N_LAYERS, 4, dtype=torch.float32, device=self.device)
        hooks   = self._inject(self._pusula, log_buf, pc,
                               ivme, sonum, zirve, taban, mode=mode)
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
    def _log_header(self, label, ms, tps, n_in, n_out, ivme, sonum, zirve, taban):
        W=60; lines=[]
        lines.append("═"*W)
        lines.append(f"  {label}")
        lines.append("═"*W)
        lines.append(f"  ⏱  {ms:.0f} ms  |  {tps:.1f} tok/s  |  "
                     f"{n_in} in  |  {n_out} out tokens")
        lines.append(f"  MOTOR  ivme={ivme:.2f}  sönüm={sonum:.2f}  "
                     f"zirve={zirve:.2f}  taban={taban:.2f}")
        lines.append(f"  SABIT  oran=0.32  doyum=0.75  karşıt=-0.40  "
                     f"sapma=0.20  fren=0.30")
        lines.append(f"  KONFIGÜRASYON  katmanlar 0-{N_LAYERS-1}/28  "
                     f"bileşim={BILISIM_ETIK:.2f}/{BILISIM_MANTIK:.2f}")
        return lines

    def _log_table(self, lb, mode, sonum, zirve, taban):
        W=60; lines=[]
        lines.append("─"*W)
        if mode=="observe":
            lines.append("  SÜTUNLAR: cos(θ) · kb · kv · Δ-ref [teorik katki, uygulanmadı]")
        else:
            lines.append("  SÜTUNLAR: cos(θ) · kb · kv · katki [hidden state'e yazıldı]")
            lines.append(f"  Formül: P_t = cos(θ) × [zirve·e^(-sönüm·t)·(1+sönüm·t) + taban]")
        lines.append("─"*W)
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
        lines.append("  DELTA KARŞILAŞTIRMA — VANILLA vs STEERED")
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

    # ── ANA ÇALIŞMA FONKSİYONU ────────────────────────────────────────────────
    def run(self, prompt, max_tokens, ivme, sonum, zirve, taban):
        if not prompt.strip():
            return "", ""

        # Parametre güvenlik kontrolü — safe band clamp
        ivme  = float(max(IVME_MIN,  min(IVME_MAX,  ivme)))
        sonum = float(max(SONUM_MIN, min(SONUM_MAX, sonum)))
        zirve = float(max(ZIRVE_MIN, min(ZIRVE_MAX, zirve)))
        taban = float(max(TABAN_MIN, min(TABAN_MAX, taban)))

        prompt_text = (f"<|im_start|>system\nYou are a helpful, accurate, and "
                       f"thoughtful assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
                       f"<|im_start|>assistant\n")
        inp = self.tok(prompt_text, return_tensors="pt").to(self.device)

        # ── PASS A: VANILLA OBSERVER ──────────────────────────────────────────
        print(f"   Pass A: Vanilla Observer  [ivme={ivme} sönüm={sonum} "
              f"zirve={zirve} taban={taban}]")
        txt_v, ms_v, tps_v, n_in_v, n_out_v, lb_v = self._run_pass(
            inp, max_tokens, "observe", ivme, sonum, zirve, taban)
        gc.collect()

        # ── PASS B: STEERED ───────────────────────────────────────────────────
        print(f"   Pass B: Steered  [ivme={ivme} sönüm={sonum} "
              f"zirve={zirve} taban={taban}]")
        txt_s, ms_s, tps_s, n_in_s, n_out_s, lb_s = self._run_pass(
            inp, max_tokens, "steer", ivme, sonum, zirve, taban)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ── VANILLA PANEL ─────────────────────────────────────────────────────
        van_lines = self._log_header(
            "⬜ VANILLA OBSERVER — katki=0 (hidden state değişmedi)",
            ms_v, tps_v, n_in_v, n_out_v, ivme, sonum, zirve, taban)
        van_lines += self._log_table(lb_v, "observe", sonum, zirve, taban)
        van_out = txt_v + "\n\n" + "\n".join(van_lines)

        # ── STEERED PANEL ─────────────────────────────────────────────────────
        st_lines = self._log_header(
            "🔱 AKBASCORE STEERED — katki uygulandı",
            ms_s, tps_s, n_in_s, n_out_s, ivme, sonum, zirve, taban)
        st_lines += self._log_table(lb_s, "steer", sonum, zirve, taban)
        st_lines += self._build_compare(lb_v, lb_s)
        st_out = txt_s + "\n\n" + "\n".join(st_lines)

        return van_out, st_out

# =============================================================================
# YÜKLEME
# =============================================================================
print("\n"+"="*58)
print("🔱⬜ AKBASCORE 1.2 | QWEN2.5-1.5B | YÜKLENIYOR...")
print("="*58+"\n")
dual = AkbasDual()
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI — AkbasCore 1.2
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

/* ── HEADER ── */
#akbas-header {
    text-align: center;
    padding: 28px 0 16px;
    border-bottom: 1px solid #111828;
    margin-bottom: 0;
}
#akbas-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: .22em;
    margin: 0 0 6px;
    background: linear-gradient(90deg, #5588cc, #33cc77);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#akbas-header .sub {
    font-size: .58rem;
    color: #1e2a3a;
    letter-spacing: .10em;
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── MOTOR PANEL ── */
#motor-panel {
    background: #060810;
    border: 1px solid #0e1520;
    border-top: 2px solid #1a2a44;
    padding: 18px 24px 16px;
    margin: 0 0 0 0;
}
#motor-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: .62rem;
    letter-spacing: .20em;
    color: #2a4060;
    margin: 0 0 4px;
    text-transform: uppercase;
}
#motor-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: .55rem;
    color: #182030;
    letter-spacing: .05em;
    margin: 0 0 14px;
    line-height: 1.6;
}
#motor-warning {
    font-family: 'JetBrains Mono', monospace;
    font-size: .52rem;
    color: #1a2a1a;
    letter-spacing: .04em;
    margin: 10px 0 0;
    padding: 6px 10px;
    border-left: 2px solid #1a3020;
}

/* ── SLIDERS ── */
.motor-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .60rem !important;
    letter-spacing: .08em !important;
    color: #3a5878 !important;
    margin-bottom: 2px !important;
}
.motor-label .tr-name {
    color: #5588aa;
    font-weight: 600;
    font-size: .62rem;
}
.motor-label .en-desc {
    color: #1e3040;
    font-size: .54rem;
}
.motor-label .val-range {
    color: #142030;
    font-size: .50rem;
}

input[type=range] {
    accent-color: #2a5080 !important;
    height: 2px !important;
}
input[type=range]::-webkit-slider-thumb {
    background: #3a7acc !important;
    width: 10px !important;
    height: 10px !important;
}

/* Slider labels */
.gradio-slider label span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .60rem !important;
    color: #3a5878 !important;
    letter-spacing: .06em !important;
}

/* ── INPUT ── */
#inp-section {
    background: #050710;
    border: 1px solid #0c1220;
    border-top: none;
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
#inp textarea:focus {
    border-color: #2a4a70 !important;
    box-shadow: 0 0 8px #2a4a7010 !important;
    outline: none !important;
}

/* ── TOKEN SLIDER ── */
#token-row {
    background: #050710;
    border: 1px solid #0c1220;
    border-top: none;
    padding: 8px 24px 14px;
    display: flex;
    align-items: center;
    gap: 16px;
}

/* ── RUN BUTTON ── */
#btn {
    background: linear-gradient(135deg, #0a1220, #142040) !important;
    color: #6699cc !important;
    border: 1px solid #1a3050 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    letter-spacing: .18em !important;
    border-radius: 2px !important;
    padding: 10px 0 !important;
    transition: all .20s;
    width: 100% !important;
    margin: 10px 24px 0 !important;
}
#btn:hover {
    background: linear-gradient(135deg, #102030, #1a3060) !important;
    color: #88bbee !important;
    box-shadow: 0 0 14px #3366aa18 !important;
    border-color: #2a4a80 !important;
}

/* ── OUTPUT PANELS ── */
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
    color: #228844;
    text-align: center;
    padding: 6px 0 3px;
    border-bottom: 1px solid #0a1810;
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
    background: #040806 !important;
    color: #33bb66 !important;
    border: 1px solid #081410 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .74rem !important;
    line-height: 1.72 !important;
    border-radius: 2px !important;
}

/* ── VERSION TAG ── */
#ver-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: .48rem;
    color: #0e1820;
    text-align: center;
    padding: 6px 0 2px;
    letter-spacing: .12em;
}
"""

# ── Motor parametre etiketleri ──────────────────────────────────────────────
# Her etiket: Türkçe isim (kalın) + İngilizce açıklama + güvenli bant
LBL_IVME = (
    "ivme  —  steering velocity: master amplitude dial  "
    f"[safe band {IVME_MIN}–{IVME_MAX}]"
)
LBL_SONUM = (
    "sönüm  —  decay rate: resonance fade speed across layers  "
    f"[safe band {SONUM_MIN}–{SONUM_MAX}]"
)
LBL_ZIRVE = (
    "zirve  —  amplitude: peak push strength at early layers  "
    f"[safe band {ZIRVE_MIN}–{ZIRVE_MAX}]"
)
LBL_TABAN = (
    "taban  —  permanent floor: minimum continuous alignment pressure  "
    f"[safe band {TABAN_MIN}–{TABAN_MAX}]"
)

with gr.Blocks(
    title="🔱 AkbasCore 1.2",
    theme=gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono")),
    css=CSS,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="akbas-header">
      <h1>🔱 AKBASCORE 1.2 &nbsp;·&nbsp; QWEN2.5-1.5B-INSTRUCT &nbsp;·&nbsp; DUAL MODE</h1>
      <p class="sub">
        RUNTIME INFERENCE-LAYER STEERING ENGINE &nbsp;·&nbsp;
        SUB-BFLOAT16 ACTIVATION INTERVENTION &nbsp;·&nbsp;
        DAMPED RESONANCE ALIGNMENT &nbsp;·&nbsp; © AKBAŞ
      </p>
    </div>
    """)

    # ── Motor Ayar Kolları Paneli ─────────────────────────────────────────────
    with gr.Column(elem_id="motor-panel"):
        gr.HTML("""
        <div id="motor-title">⚙ BEYIN BASINCI MOTOR AYAR KOLLARI</div>
        <div id="motor-desc">
          AkbasCore, modelin her transformer katmanında gizli durumuna (hidden state)
          matematiksel basınç uygular — ağırlıklar değişmez, prompt değişmez.<br>
          Aşağıdaki ayar kolları bu basıncın şeklini ve yoğunluğunu belirler.
          Varsayılan değerler AkbasCore 1.1 kalibrasyonuna aittir ve orantılı seçilmiştir.
        </div>
        """)

        with gr.Row():
            sl_ivme = gr.Slider(
                minimum=IVME_MIN, maximum=IVME_MAX,
                value=DEFAULT_IVME, step=0.05,
                label=LBL_IVME)
            sl_sonum = gr.Slider(
                minimum=SONUM_MIN, maximum=SONUM_MAX,
                value=DEFAULT_SONUM, step=0.05,
                label=LBL_SONUM)

        with gr.Row():
            sl_zirve = gr.Slider(
                minimum=ZIRVE_MIN, maximum=ZIRVE_MAX,
                value=DEFAULT_ZIRVE, step=0.05,
                label=LBL_ZIRVE)
            sl_taban = gr.Slider(
                minimum=TABAN_MIN, maximum=TABAN_MAX,
                value=DEFAULT_TABAN, step=0.05,
                label=LBL_TABAN)

        gr.HTML("""
        <div id="motor-warning">
          Motor sabitleri [oran · doyum · karşıt · sapma · fren · sınır] C++ çekirdeğinde
          gömülüdür — yeniden derleme gerektirir. Yukarıdaki ayar kolları güvenli bant
          içinde kalır: çıktıyı belirgin biçimde değiştirir, sistemi bozmaz.
        </div>
        """)

    # ── Girdi + Token ─────────────────────────────────────────────────────────
    with gr.Column(elem_id="inp-section"):
        inp = gr.Textbox(
            label="► INPUT — aynı soru her iki pass'a gönderilir",
            lines=4,
            placeholder="Sorunuzu buraya yazın...",
            show_copy_button=False,
            elem_id="inp")

        with gr.Row():
            sl_tok = gr.Slider(
                64, 1024, value=512, step=64,
                label="MAX TOKENS — output token budget",
                scale=4)
            btn = gr.Button(
                "▶▶  DUAL RUN",
                variant="primary",
                elem_id="btn",
                scale=1)

    # ── Çıktı Panelleri ───────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="panel-lbl-v">⬜ VANILLA OBSERVER — katki = 0  |  hidden state unchanged</div>')
            out_v = gr.Textbox(
                label="OUTPUT + OBSERVATION LOG  [Δ-ref = theoretical katki, not applied]",
                lines=40, interactive=False, show_copy_button=True,
                elem_id="out_v")
        with gr.Column():
            gr.HTML('<div class="panel-lbl-s">🔱 STEERED — katki applied  |  hidden state directed</div>')
            out_s = gr.Textbox(
                label="OUTPUT + ACTIVATION LOG + DELTA COMPARISON",
                lines=40, interactive=False, show_copy_button=True,
                elem_id="out_s")

    # ── Versiyon ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="ver-tag">
      AKBASCORE 1.2 &nbsp;·&nbsp; TERMINOLOGY: pusula · katki · taban · sönüm · zirve ·
      ivme · terazi · bileşim · oran · doyum · karşıt · sapma · fren · sınır
      &nbsp;·&nbsp; © AKBAŞ &nbsp;·&nbsp; APACHE 2.0 BASE MODEL
    </div>
    """)

    btn.click(
        fn=dual.run,
        inputs=[inp, sl_tok, sl_ivme, sl_sonum, sl_zirve, sl_taban],
        outputs=[out_v, out_s])
    inp.submit(
        fn=dual.run,
        inputs=[inp, sl_tok, sl_ivme, sl_sonum, sl_zirve, sl_taban],
        outputs=[out_v, out_s])

print("🚀 AkbasCore 1.2 Gradio başlatılıyor...")
demo.launch(share=True, debug=False)
