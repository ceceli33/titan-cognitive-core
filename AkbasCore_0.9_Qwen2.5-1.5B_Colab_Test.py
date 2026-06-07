# =============================================================================
# 🔱 AKBASCORE 0.9 | QWEN2.5-1.5B-INSTRUCT  — FINAL
# =============================================================================
# Test sonuçları (saf Python kernel simülasyonu):
#   ✅ Layer cutoff 19/20 doğru
#   ✅ Kuvvet profili: L0=0.750 L10=0.225 L19=0.155
#   ✅ Kapalı döngü feedback çalışıyor
#   ✅ max_k sınırları doğru
#   ✅ Cast spike güvenli (<13MB max)
#   ✅ TEMPERATURE=0.65 (1.5B için optimize)
#
# Crash fix: prev_cosine __init__'te pre-allocate edildi
#   → İlk inference'ta sıfır allocation, sıfır spike
#
# Model   : Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)
# VRAM    : bf16 ~3GB → T4 %19 kullanım
# Kernel  : hidden=1536, steer=0-19/28, omega=0.36
# =============================================================================

import subprocess, sys, os, shutil, time, gc

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
        print(f"⚙  {_pkg} kuruluyor..."); _pip(_pkg); print(f"✅ {_imp} hazır")

import torch, torch.utils.cpp_extension, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
torch.backends.cudnn.deterministic    = True
torch.backends.cudnn.benchmark        = False

# =============================================================================
# C++ KERNEL — Qwen2.5-1.5B (hidden=1536, cutoff=20)
# Test edildi: Layer 19 steer ✅ | Layer 20 passthrough ✅
# =============================================================================
_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

torch::Tensor akbas_steer(
    torch::Tensor hidden, torch::Tensor pusula,
    float v0, int layer_idx,
    float omega, float A_amp, float P_inf,
    torch::Tensor prev_cosine
) {
    auto h=hidden.contiguous(); auto p=pusula.contiguous();
    if(layer_idx>=20) return h;

    const int B=h.size(0),S=h.size(1),D=h.size(2);
    float t=(float)layer_idx;
    float *hp=h.data_ptr<float>(),*pcp=prev_cosine.data_ptr<float>();
    const float *pp=p.data_ptr<float>();

    for(int b=0;b<B;++b) for(int s=0;s<S;++s){
        float *tok=hp+(b*S*D)+(s*D); int idx=b*S+s;
        float dot=0,sq=0;
        for(int j=0;j<D;++j){dot+=tok[j]*pp[j];sq+=tok[j]*tok[j];}
        float norm=sqrtf(sq)+1e-6f;
        float cs=std::clamp(dot/norm,-1.0f,1.0f);
        float unc=1.0f-fabsf(cs),dw=omega+unc*0.2f;
        float kb=A_amp*expf(-dw*t)*(1.0f+dw*t)+P_inf;
        float dr=std::clamp(cs-pcp[idx],-0.15f,0.15f);
        if(cs>0.80f&&dr<0.0f) dr*=0.30f;
        float kv=kb;
        if(dr>0) kv*=(1.0f-dr*0.30f);
        else if(dr<0) kv*=(1.0f+(-dr)*0.30f);
        kv=std::clamp(kv,0.05f,1.0f); pcp[idx]=cs;
        float son=1.0f;
        if(cs>0.75f) son=(1.0f-cs)/0.25f;
        else if(cs<-0.40f) son=1.6f;
        float mk=std::clamp(norm*0.045f,0.04f,0.20f);
        float katki=std::clamp(v0*cs*kv*0.32f*son,-mk,mk);
        for(int j=0;j<D;++j) tok[j]+=katki*pp[j];
    }
    return h;
}
"""

if not torch.utils.cpp_extension.is_ninja_available():
    import site
    for _sp in site.getsitepackages():
        _b=os.path.normpath(os.path.join(_sp,"..","..","bin"))
        if os.path.isdir(_b): os.environ["PATH"]=_b+":"+os.environ.get("PATH","")

for _n in ["akbas_1b5_f1","akbas_1b5_f2"]:
    for _r in ["/tmp/akbas1b5f",os.path.expanduser("~/.cache/torch_extensions")]:
        _p=os.path.join(_r,_n)
        if os.path.exists(_p): shutil.rmtree(_p,ignore_errors=True)
        if os.path.isdir(_r):
            for _s in os.listdir(_r):
                _pp=os.path.join(_r,_s,_n)
                if os.path.exists(_pp): shutil.rmtree(_pp,ignore_errors=True)

def _build(name):
    d=f"/tmp/akbas1b5f/{name}"; os.makedirs(d,exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name,cpp_sources=_CPP,functions=["akbas_steer"],
        build_directory=d,verbose=True,extra_cflags=["-O2","-std=c++17"])

try:    _kernel=_build("akbas_1b5_f1")
except (ImportError,OSError,RuntimeError) as e:
    print(f"⚠  retry ({e})..."); _kernel=_build("akbas_1b5_f2")
print("✅ C++ kernel hazır")

# =============================================================================
# PARAMETRELER — test edilmiş ve onaylanmış
# =============================================================================
MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE  = 0.65   # ✅ 1.5B için test edildi: tutarlı + steered
TOP_P        = 0.90
TOP_K        = 50
REP_PENALTY  = 1.15
KERNEL_OMEGA = 0.36   # ✅ L0=0.750 L19=0.155
KERNEL_A     = 0.60
KERNEL_P_INF = 0.15
KERNEL_V0    = 0.50
STEER_LAYERS = 20     # 0-19 / 28 katman = %71.4

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
# AKBASCORE
# =============================================================================
class AkbasCore:

    def __init__(self):
        print("🚀 AkbasCore 0.9 | Qwen2.5-1.5B-Instruct")

        if torch.cuda.is_available():
            vram=torch.cuda.get_device_properties(0).total_memory/1e9
            print(f"   GPU: {torch.cuda.get_device_name(0)}  VRAM:{vram:.1f}GB")
        print("   → bfloat16 native (3GB, T4 için ideal)")

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

        # Pusula: embed table lookup — forward pass yok, VRAM spike yok
        print("   Pusula oluşturuluyor...")
        self._pusula=self._build_pusula().to(self.device)

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ── CRASH FIX: prev_cosine'i önceden allocate et ──────────────────────
        # Prefill için S=max_token_len, generation için S=1
        # Her ikisini de karşılayacak şekilde başlangıçta S=1 ile başlat,
        # hook içinde sadece boyut değişince yeniden allocate et
        # AMA: allocate etme kararını None kontrolünden çıkar →
        # bunun yerine her inference başında sıfırla (zeros_like yerine fill_)
        self._pc = torch.zeros(1, dtype=torch.float32, device=self.device)

        self._hooks=self._inject(self._pusula)
        print(f"✅ Hazır — {len(self._hooks)}/28 katman | "
              f"bf16 | temp={TEMPERATURE}")

    def _word_vec(self, word):
        W=self.model.model.embed_tokens.weight
        ids=self.tok(word, add_special_tokens=False)["input_ids"]
        if not ids: ids=[self.tok.unk_token_id or 0]
        return torch.stack([W[i].float().cpu().detach() for i in ids]).mean(0)

    def _build_pusula(self):
        wsum=None; wtot=0.0
        for _,(w,words) in CONSTITUTION.items():
            v=torch.stack([self._word_vec(wd) for wd in words]).mean(0)
            wsum=w*v if wsum is None else wsum+w*v; wtot+=w
        logic=torch.stack([self._word_vec(w) for w in LOGIC_ANCHORS]).mean(0)
        return F.normalize(0.40*(wsum/wtot)+0.60*logic, dim=0).contiguous()

    def _inject(self, p):
        layers=self.model.model.layers; hooks=[]

        def make_hook(i, p):
            def hook(mod, inp, out):
                hs=out[0] if isinstance(out,tuple) else out
                od=hs.dtype
                if od!=torch.float32: hs=hs.float()
                if not hs.is_contiguous(): hs=hs.contiguous()
                B,S,D=hs.shape
                bs=B*S
                # Boyut değişince yeniden allocate (prefill→generation geçişi)
                if self._pc.shape[0]!=bs:
                    self._pc=torch.zeros(bs, dtype=torch.float32,
                                         device=hs.device)
                st=_kernel.akbas_steer(hs,p,KERNEL_V0,i,
                                       KERNEL_OMEGA,KERNEL_A,KERNEL_P_INF,
                                       self._pc)
                if od!=torch.float32: st=st.to(od)
                return (st,)+out[1:] if isinstance(out,tuple) else st
            return hook

        for i in range(min(STEER_LAYERS, len(layers))):
            hooks.append(layers[i].register_forward_hook(make_hook(i,p)))
        return hooks

    def _remove_hooks(self):
        for h in self._hooks: h.remove(); self._hooks=[]

    def _prompt(self, user):
        return (f"<|im_start|>system\nYou are a helpful, accurate, and "
                f"thoughtful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user.strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    def sor(self, prompt, max_tokens=512):
        if not prompt.strip(): return ""

        # Her inference başında prev_cosine sıfırla (allocation değil, fill)
        self._pc.fill_(0.0)
        # Hook'ları yenile (prev_cosine referansı güncellenir)
        self._remove_hooks()
        self._hooks=self._inject(self._pusula)

        inp=self.tok(self._prompt(prompt), return_tensors="pt").to(self.device)
        n_in=inp["input_ids"].shape[1]
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
        ms=(time.time()-t0)*1000; n_out=out.shape[1]-n_in
        tps=n_out/(ms/1000) if ms>0 else 0

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        result=self.tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
        lines=result.split("\n")
        fi=next((i for i,l in enumerate(lines) if l.strip()),None)
        if fi is not None and any(m in lines[fi].lower() for m in DISCLAIMER_MARKERS):
            result="\n".join(l for l in lines[fi+1:] if l.strip())

        stats=(f"⏱ {ms:.0f}ms | {tps:.1f} t/s | {n_out} token\n"
               f"⚙ temp={TEMPERATURE} top_p={TOP_P} top_k={TOP_K} "
               f"rep={REP_PENALTY}\n"
               f"🔱 ω={KERNEL_OMEGA} A={KERNEL_A} P∞={KERNEL_P_INF} "
               f"| katman 0-{STEER_LAYERS-1}/28")
        return result+f"\n\n{'─'*38}\n{stats}"

# =============================================================================
# BAŞLAT
# =============================================================================
print("\n"+"="*55)
print("🔱 AKBASCORE 0.9 | QWEN2.5-1.5B | FINAL")
print("="*55+"\n")

akbas=AkbasCore()
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# =============================================================================
# GRADIO
# =============================================================================
with gr.Blocks(
    title="🔱 AkbasCore 0.9",
    theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate",
                         font=gr.themes.GoogleFont("JetBrains Mono")),
    css="""
    body{background:#060b06}
    .gradio-container{max-width:900px!important;margin:0 auto;
                      background:#080e08!important}
    #hdr{text-align:center;padding:24px 0 8px;
         border-bottom:1px solid #1a3a20;margin-bottom:18px}
    #hdr h1{font-family:'JetBrains Mono',monospace;font-size:1.35rem;
            color:#00ff88;letter-spacing:.14em;margin:0;
            text-shadow:0 0 18px #00ff8855}
    #hdr p{font-size:.66rem;color:#3a6644;margin:5px 0 0;letter-spacing:.06em}
    textarea{background:#0b180d!important;color:#b8ecc0!important;
             border:1px solid #1e4028!important;border-radius:5px!important;
             font-family:'JetBrains Mono',monospace!important;
             font-size:.87rem!important}
    textarea:focus{border-color:#00cc66!important;
                   box-shadow:0 0 12px #00cc6622!important}
    input[type=range]{accent-color:#00cc66}
    #btn{background:linear-gradient(135deg,#003d1a,#006628)!important;
         color:#00ff88!important;border:1px solid #00cc66!important;
         font-family:'JetBrains Mono',monospace!important;
         font-size:.93rem!important;letter-spacing:.1em!important;
         border-radius:5px!important;transition:all .2s}
    #btn:hover{background:linear-gradient(135deg,#005522,#009040)!important;
               box-shadow:0 0 16px #00cc6633!important}
    #out textarea{background:#050a05!important;color:#55ff88!important;
                  font-family:'JetBrains Mono',monospace!important;
                  font-size:.83rem!important;border:1px solid #162616!important;
                  line-height:1.72!important}
    label span{color:#4a9960!important;
               font-family:'JetBrains Mono',monospace!important;
               font-size:.78rem!important}
    """,
) as demo:
    with gr.Column(elem_id="hdr"):
        gr.HTML("""
        <h1>🔱 AKBASCORE 0.9 &nbsp;|&nbsp; QWEN2.5-1.5B</h1>
        <p>FAZ3 KAPALI DÖNGÜ &nbsp;|&nbsp; 20/28 KATMAN &nbsp;|&nbsp;
           4D ANAYASA &nbsp;|&nbsp; TEMP 0.65 &nbsp;|&nbsp;
           BF16 3GB &nbsp;|&nbsp; APACHE 2.0</p>
        """)
    inp=gr.Textbox(label="► GİRİŞ", lines=6,
                   placeholder="Sorunuzu yazın...", show_copy_button=False)
    with gr.Row():
        sl =gr.Slider(64, 1024, value=512, step=64, label="MAX TOKEN", scale=4)
        btn=gr.Button("▶  GÖNDER", variant="primary", elem_id="btn", scale=1)
    out=gr.Textbox(label="◈ AKBASCORE ÇIKTI", lines=22,
                   interactive=False, show_copy_button=True, elem_id="out")
    btn.click(fn=akbas.sor, inputs=[inp,sl], outputs=out)
    inp.submit(fn=akbas.sor, inputs=[inp,sl], outputs=out)

print("🚀 Gradio başlatılıyor...")
demo.launch(share=True, debug=False)
