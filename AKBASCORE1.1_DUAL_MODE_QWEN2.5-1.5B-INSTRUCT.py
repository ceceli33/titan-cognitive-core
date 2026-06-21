# =============================================================================
# 🔱⬜ AKBASCORE DUAL MODE | QWEN2.5-1.5B-INSTRUCT — GRADIO UI
# =============================================================================
# TEK MODEL — İKİ PASS:
#   Pass A (⬜ VANILLA)  : akbas_observe() → katki=0 → hidden state DOKUNULMAZ
#   Pass B (🔱 STEERED)  : akbas_steer()  → katki≠0 → hidden state YÖNLENDİRİLİR
#
# Her iki pass da aynı ağırlıkları paylaşır; sadece hook seti değişir.
# Çıktıda: sol panel Vanilla, sağ panel Steered — her ikisinin tam logu.
#
# HOW TO RUN:
#   1. colab.research.google.com → New notebook
#   2. Runtime → Change runtime type → CPU  (veya T4 GPU)
#   3. Paste this entire file into ONE cell → Run
#   4. Open the Gradio link → soruyu yaz → DUAL RUN
#
# Model   : Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)
# Kernel  : C++ — akbas_observe() + akbas_steer() aynı .so içinde
# Params  : ω=0.30  A=0.70  P∞=0.20  v0=0.50
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
# C++ ÇEKİRDEK — İKİ FONKSİYON TEK DERLEME
# =============================================================================
_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

// ─── PASS A: GÖZLEM — hidden state'e DOKUNMAZ ────────────────────────────────
torch::Tensor akbas_observe(
    torch::Tensor hidden,
    torch::Tensor pusula,
    float v0,
    int layer_idx,
    float omega,
    float A_amp,
    float P_inf,
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
        float dw  = omega + unc * 0.2f;
        float kb  = A_amp * expf(-dw * t) * (1.0f + dw * t) + P_inf;

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
        float katki = std::clamp(v0 * cs * kv * 0.32f * son, -mk, mk);
        // ⚠  tok[j] += katki * pp[j]  ← YOK (gözlem modu)

        if (S > 1 && b == 0 && s == 0 && layer_idx < 20) {
            lbp[layer_idx * 4 + 0] = cs;
            lbp[layer_idx * 4 + 1] = kb;
            lbp[layer_idx * 4 + 2] = kv;
            lbp[layer_idx * 4 + 3] = katki;  // teorik — uygulanmadı
        }
    }
    return h;   // DEĞİŞTİRİLMEMİŞ
}

// ─── PASS B: STEERING — hidden state'i YÖNLENDİRİR ───────────────────────────
torch::Tensor akbas_steer(
    torch::Tensor hidden,
    torch::Tensor pusula,
    float v0,
    int layer_idx,
    float omega,
    float A_amp,
    float P_inf,
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
        float dw  = omega + unc * 0.2f;
        float kb  = A_amp * expf(-dw * t) * (1.0f + dw * t) + P_inf;

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
        float katki = std::clamp(v0 * cs * kv * 0.32f * son, -mk, mk);

        // ✅ UYGULANIYOR
        for (int j = 0; j < D; ++j) tok[j] += katki * pp[j];

        if (S > 1 && b == 0 && s == 0 && layer_idx < 20) {
            lbp[layer_idx * 4 + 0] = cs;
            lbp[layer_idx * 4 + 1] = kb;
            lbp[layer_idx * 4 + 2] = kv;
            lbp[layer_idx * 4 + 3] = katki;  // gerçek uygulanan değer
        }
    }
    return h;
}
"""

# ── derleme ───────────────────────────────────────────────────────────────────
if not torch.utils.cpp_extension.is_ninja_available():
    import site
    for _sp in site.getsitepackages():
        _b=os.path.normpath(os.path.join(_sp,"..","..","bin"))
        if os.path.isdir(_b): os.environ["PATH"]=_b+":"+os.environ.get("PATH","")

for _n in ["akbas_dual_f1","akbas_dual_f2"]:
    for _r in ["/tmp/akbas_dual",os.path.expanduser("~/.cache/torch_extensions")]:
        _p=os.path.join(_r,_n)
        if os.path.exists(_p): shutil.rmtree(_p,ignore_errors=True)
        if os.path.isdir(_r):
            for _s in os.listdir(_r):
                _pp=os.path.join(_r,_s,_n)
                if os.path.exists(_pp): shutil.rmtree(_pp,ignore_errors=True)

def _build(name):
    d=f"/tmp/akbas_dual/{name}"; os.makedirs(d,exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name, cpp_sources=_CPP,
        functions=["akbas_observe","akbas_steer"],
        build_directory=d, verbose=False,
        extra_cflags=["-O2","-std=c++17"])

try:    _kernel=_build("akbas_dual_f1")
except Exception as e:
    print(f"  retry ({e})..."); _kernel=_build("akbas_dual_f2")
print("✅ C++ çift-çekirdek derlendi (observe + steer)")

# =============================================================================
# PARAMETRELER
# =============================================================================
MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE  = 0.65
TOP_P        = 0.90
TOP_K        = 50
REP_PENALTY  = 1.15
KERNEL_OMEGA = 0.30
KERNEL_A     = 0.70
KERNEL_P_INF = 0.20
KERNEL_V0    = 0.50
N_LAYERS     = 20

CONSTITUTION = {
    "harm":     (0.9228, ["safe","harmless","protective","secure","careful"]),
    "honesty":  (0.9372, ["honest","accurate","truthful","transparent","precise"]),
    "autonomy": (0.8788, ["autonomous","respectful","unbiased","free","neutral"]),
    "fairness": (0.9196, ["fair","just","equitable","balanced","impartial"]),
}
LOGIC_ANCHORS = [
    "logical","empirical","systematic","structured","verifiable",
    "analyze","precise","deterministic","sequential","causal",
    "rigorous","impossible","contradiction","identify","optimize",
]
DISCLAIMER_MARKERS = [
    "as an ai","as a language model","i apologize","i must clarify",
    "i'm unable to","i am unable to","i cannot be certain",
]

# =============================================================================
# AkbasDual — TEK MODEL, İKİ PASS
# =============================================================================
class AkbasDual:

    def __init__(self):
        print("🔱⬜ AKBASCORE DUAL | Qwen2.5-1.5B-Instruct")
        print("   Tek model yükleme → iki ayrı forward pass")
        print(f"   ⚙  ω={KERNEL_OMEGA}  A={KERNEL_A}  P∞={KERNEL_P_INF}  v0={KERNEL_V0}")
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
        print("✅ Hazır — dual pass aktif\n")

    # ── pusula ────────────────────────────────────────────────────────────────
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
        return F.normalize(0.40*(wsum/wtot)+0.60*logic,dim=0).contiguous()

    # ── hook yönetimi ──────────────────────────────────────────────────────────
    def _remove_hooks(self,hooks):
        for h in hooks: h.remove()

    def _inject(self, p, log_buf, pc, mode="observe"):
        """mode='observe' veya 'steer'"""
        layers=self.model.model.layers; hooks=[]
        fn = _kernel.akbas_observe if mode=="observe" else _kernel.akbas_steer
        def make_hook(i):
            def hook(mod,inp,out):
                hs=out[0] if isinstance(out,tuple) else out
                od=hs.dtype
                if od!=torch.float32: hs=hs.float()
                if not hs.is_contiguous(): hs=hs.contiguous()
                B,S,D=hs.shape; bs=B*S
                if pc.shape[0]!=bs:
                    pc_local=torch.zeros(bs,dtype=torch.float32,device=hs.device)
                else:
                    pc_local=pc
                st=fn(hs,p,KERNEL_V0,i,
                      KERNEL_OMEGA,KERNEL_A,KERNEL_P_INF,
                      pc_local,log_buf)
                if od!=torch.float32: st=st.to(od)
                return (st,)+out[1:] if isinstance(out,tuple) else st
            return hook
        for i in range(min(N_LAYERS,len(layers))):
            hooks.append(layers[i].register_forward_hook(make_hook(i)))
        return hooks

    # ── tek forward pass ───────────────────────────────────────────────────────
    def _run_pass(self, inp, max_tokens, mode):
        pc      = torch.zeros(1, dtype=torch.float32, device=self.device)
        log_buf = torch.zeros(N_LAYERS, 4, dtype=torch.float32, device=self.device)
        hooks   = self._inject(self._pusula, log_buf, pc, mode=mode)
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
    def _log_header(self, label, ms, tps, n_in, n_out):
        W=58; lines=[]
        lines.append("═"*W)
        lines.append(f"  {label}")
        lines.append("═"*W)
        lines.append(f"  ⏱  {ms:.0f} ms  |  {tps:.1f} tok/s  |  "
                     f"{n_in} giriş  |  {n_out} çıkış")
        lines.append(f"  ⚙  temp={TEMPERATURE}  top_p={TOP_P}  "
                     f"top_k={TOP_K}  rep={REP_PENALTY}")
        lines.append(f"  🔧 ω={KERNEL_OMEGA}  A={KERNEL_A}  P∞={KERNEL_P_INF}  "
                     f"katmanlar 0-{N_LAYERS-1}/28")
        return lines

    def _log_table(self, lb, mode):
        W=58; lines=[]
        lines.append("─"*W)
        if mode=="observe":
            lines.append("  SÜTUNLAR: cos(θ) · kb(teorik) · kv(teorik) · Δ-ref(teorik katki)")
            lines.append("  katki sütunu: 0.000000 (hidden state'e EKLENMEDİ)")
        else:
            lines.append("  SÜTUNLAR: cos(θ) · kb · kv · katki(UYGULANDI)")
            lines.append("  Formula: P_t = cos(θ) × [A·e^(-ωt)·(1+ωt) + P∞]")
        lines.append("─"*W)
        lines.append(f"  {'L':>3}  {'cos(θ)':>8}  {'kb':>8}  "
                     f"{'kv':>8}  {'katki/Δ-ref':>12}  not")
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
            kb_th   = KERNEL_A*math.exp(-KERNEL_OMEGA*t)*(1+KERNEL_OMEGA*t)+KERNEL_P_INF
            note    = "← eq." if abs(kb_th-KERNEL_P_INF)<0.05 else ""
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
        """İki log arasındaki farkları gösteren karşılaştırma bloğu."""
        W=58; lines=[]
        lines.append("═"*W)
        lines.append("  ΔDELTA KARŞILAŞTIRMA — VANILLA vs STEERED")
        lines.append("  cos(θ) ve katki farklarını katman katman gösterir")
        lines.append("─"*W)
        lines.append(f"  {'L':>3}  {'cos_V':>8}  {'cos_S':>8}  "
                     f"{'Δcos':>8}  {'Δkatki':>10}")
        lines.append("─"*W)
        dcos_sum=0.0; dkat_sum=0.0; rows=0
        for i in range(N_LAYERS):
            csv = lb_v[i,0].item(); kbv = lb_v[i,1].item()
            css = lb_s[i,0].item()
            katv= lb_v[i,3].item()   # teorik (observe)
            kats= lb_s[i,3].item()   # gerçek (steer)
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
    def run(self, prompt, max_tokens=512):
        if not prompt.strip():
            return "", ""

        prompt_text = (f"<|im_start|>system\nYou are a helpful, accurate, and "
                       f"thoughtful assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
                       f"<|im_start|>assistant\n")
        inp = self.tok(prompt_text, return_tensors="pt").to(self.device)

        # ── PASS A: VANILLA OBSERVER ──────────────────────────────────────────
        print("   Pass A: Vanilla Observer...")
        txt_v, ms_v, tps_v, n_in_v, n_out_v, lb_v = self._run_pass(
            inp, max_tokens, mode="observe")
        gc.collect()

        # ── PASS B: STEERED ───────────────────────────────────────────────────
        print("   Pass B: Steered...")
        txt_s, ms_s, tps_s, n_in_s, n_out_s, lb_s = self._run_pass(
            inp, max_tokens, mode="steer")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ── VANILLA PANEL ─────────────────────────────────────────────────────
        van_lines = self._log_header(
            "⬜ VANILLA OBSERVER — katki=0 (hidden state değişmedi)",
            ms_v, tps_v, n_in_v, n_out_v)
        van_lines += self._log_table(lb_v, mode="observe")
        van_lines += ["", "  ⚠  Δ-ref = steering'in bu katmanda ekleyeceği teorik değer"]
        van_out = txt_v + "\n\n" + "\n".join(van_lines)

        # ── STEERED PANEL ─────────────────────────────────────────────────────
        st_lines = self._log_header(
            "🔱 AKBASCORE STEERED — katki uygulandı",
            ms_s, tps_s, n_in_s, n_out_s)
        st_lines += self._log_table(lb_s, mode="steer")
        st_lines += self._build_compare(lb_v, lb_s)
        st_out = txt_s + "\n\n" + "\n".join(st_lines)

        return van_out, st_out

# =============================================================================
# YÜKLEME
# =============================================================================
print("\n"+"="*55)
print("🔱⬜ AKBASCORE DUAL | QWEN2.5-1.5B | YÜKLENIYOR...")
print("="*55+"\n")

dual = AkbasDual()
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI
# =============================================================================
CSS = """
body { background: #05060a }
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto;
    background: #06080e !important;
}
#hdr {
    text-align: center;
    padding: 24px 0 10px;
    border-bottom: 1px solid #181e30;
    margin-bottom: 18px;
}
#hdr h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.20rem;
    letter-spacing: .16em;
    margin: 0 0 5px;
}
#hdr p {
    font-size: .60rem;
    color: #2a3248;
    margin: 0;
    letter-spacing: .08em;
}
textarea {
    background: #080b12 !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .83rem !important;
    line-height: 1.65 !important;
}
#inp textarea {
    color: #a8b8d0 !important;
    border: 1px solid #182030 !important;
}
#inp textarea:focus {
    border-color: #4a6a99 !important;
    box-shadow: 0 0 8px #4a6a9918 !important;
}
#out_v textarea {
    background: #050810 !important;
    color: #66aaee !important;
    border: 1px solid #0f1828 !important;
    line-height: 1.72 !important;
}
#out_s textarea {
    background: #050a06 !important;
    color: #44ee77 !important;
    border: 1px solid #0a1810 !important;
    line-height: 1.72 !important;
}
input[type=range] { accent-color: #4a88cc }
#btn {
    background: linear-gradient(135deg, #0e1828, #1a2e50) !important;
    color: #88ccee !important;
    border: 1px solid #2a4a80 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .88rem !important;
    letter-spacing: .14em !important;
    border-radius: 3px !important;
    transition: all .18s;
}
#btn:hover {
    background: linear-gradient(135deg, #162444, #223a66) !important;
    box-shadow: 0 0 12px #4488cc28 !important;
}
label span {
    color: #3a5070 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .72rem !important;
    letter-spacing: .05em !important;
}
.panel-label-v {
    color: #66aaee;
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    letter-spacing: .10em;
    text-align: center;
    padding: 4px 0 2px;
}
.panel-label-s {
    color: #44ee77;
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    letter-spacing: .10em;
    text-align: center;
    padding: 4px 0 2px;
}
"""

with gr.Blocks(
    title="🔱⬜ AkbasCore Dual",
    theme=gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono")),
    css=CSS,
) as demo:

    with gr.Column(elem_id="hdr"):
        gr.HTML("""
        <h1 style="background:linear-gradient(90deg,#66aaee,#44ee77);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent">
          🔱⬜ AKBASCORE DUAL &nbsp;|&nbsp; QWEN2.5-1.5B
        </h1>
        <p>
          TEK MODEL &nbsp;·&nbsp; İKİ PASS &nbsp;·&nbsp;
          ⬜ VANILLA (katki=0) &nbsp;+&nbsp; 🔱 STEERED (katki≠0) &nbsp;·&nbsp;
          TAM LOG &nbsp;·&nbsp; DELTA KARŞILAŞTIRMA &nbsp;·&nbsp;
          ω=0.30 &nbsp;A=0.70 &nbsp;P∞=0.20 &nbsp;·&nbsp;
          TEMP 0.65 &nbsp;·&nbsp; BF16 &nbsp;·&nbsp; APACHE 2.0
        </p>
        """)

    inp=gr.Textbox(
        label="► INPUT — aynı soru her iki pass'a gönderilir",
        lines=4,
        placeholder="Sorunuzu buraya yazın...",
        show_copy_button=False,
        elem_id="inp")

    with gr.Row():
        sl=gr.Slider(64,1024,value=512,step=64,label="MAX TOKENS",scale=4)
        btn=gr.Button("▶▶  DUAL RUN",variant="primary",elem_id="btn",scale=1)

    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="panel-label-v">⬜ VANILLA OBSERVER — katki = 0</div>')
            out_v=gr.Textbox(
                label="ÇIKTI + GÖZLEM LOGU (Δ-ref = teorik müdahale)",
                lines=38, interactive=False, show_copy_button=True,
                elem_id="out_v")
        with gr.Column():
            gr.HTML('<div class="panel-label-s">🔱 STEERED — katki uygulandı</div>')
            out_s=gr.Textbox(
                label="ÇIKTI + AKTİVASYON LOGU + DELTA KARŞILAŞTIRMA",
                lines=38, interactive=False, show_copy_button=True,
                elem_id="out_s")

    btn.click(fn=dual.run, inputs=[inp,sl], outputs=[out_v,out_s])
    inp.submit(fn=dual.run, inputs=[inp,sl], outputs=[out_v,out_s])

print("🚀 Gradio başlatılıyor...")
demo.launch(share=True, debug=False)
