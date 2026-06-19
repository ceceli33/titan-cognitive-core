# =============================================================================
# 🔱 AKBASCORE 1.0 | QWEN2.5-1.5B-INSTRUCT — GRADIO UI
# =============================================================================
# AkbasCore 1.0: C++ kernel writes real internal values to log_buf.
# Every cs, kb, kv, katki shown in the output came directly from
# inside akbas_steer() — not recalculated in Python.
#
# HOW TO RUN:
#   1. colab.research.google.com → New notebook
#   2. Runtime → Change runtime type → CPU
#   3. Paste this entire file into ONE cell → Run
#   4. Open the Gradio link → type your question → send
#
# Model  : Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)
# Kernel : hidden=1536, steer=0-19/28, omega=0.30, A=0.70, P_inf=0.20
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

_CPP = r"""
#include <torch/extension.h>
#include <cmath>
#include <algorithm>

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

    float *hp  = h.data_ptr<float>();
    float *pcp = prev_cosine.data_ptr<float>();
    float *lbp = lb.data_ptr<float>();
    const float *pp = p.data_ptr<float>();

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
        _b=os.path.normpath(os.path.join(_sp,"..","..","bin"))
        if os.path.isdir(_b): os.environ["PATH"]=_b+":"+os.environ.get("PATH","")

for _n in ["akbas_10g_f1","akbas_10g_f2"]:
    for _r in ["/tmp/akbas10g",os.path.expanduser("~/.cache/torch_extensions")]:
        _p=os.path.join(_r,_n)
        if os.path.exists(_p): shutil.rmtree(_p,ignore_errors=True)
        if os.path.isdir(_r):
            for _s in os.listdir(_r):
                _pp=os.path.join(_r,_s,_n)
                if os.path.exists(_pp): shutil.rmtree(_pp,ignore_errors=True)

def _build(name):
    d=f"/tmp/akbas10g/{name}"; os.makedirs(d,exist_ok=True)
    return torch.utils.cpp_extension.load_inline(
        name=name,cpp_sources=_CPP,functions=["akbas_steer"],
        build_directory=d,verbose=False,extra_cflags=["-O2","-std=c++17"])

try:    _kernel=_build("akbas_10g_f1")
except Exception as e:
    print(f"  retry ({e})..."); _kernel=_build("akbas_10g_f2")
print("✅ C++ kernel 1.0 compiled")

MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE  = 0.65
TOP_P        = 0.90
TOP_K        = 50
REP_PENALTY  = 1.15
KERNEL_OMEGA = 0.30
KERNEL_A     = 0.70
KERNEL_P_INF = 0.20
KERNEL_V0    = 0.50
STEER_LAYERS = 20

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

class AkbasCore:

    def __init__(self):
        print("🔱 AkbasCore 1.0 | Qwen2.5-1.5B-Instruct")
        print("   C++ kernel writes real internal values — 100% live.")
        print(f"   ⚙ PARAMS: ω={KERNEL_OMEGA} A={KERNEL_A} P∞={KERNEL_P_INF}")
        if torch.cuda.is_available():
            vram=torch.cuda.get_device_properties(0).total_memory/1e9
            print(f"   GPU: {torch.cuda.get_device_name(0)} {vram:.1f}GB")
        else:
            print("   Device: CPU")

        self.tok=AutoTokenizer.from_pretrained(MODEL_ID)
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token

        self.model=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,device_map="auto",
            trust_remote_code=True,dtype=torch.bfloat16,
            low_cpu_mem_usage=True)
        if hasattr(self.model.config,"_attn_implementation"):
            self.model.config._attn_implementation="eager"
        self.device=next(self.model.parameters()).device

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("   Building compass vector...")
        self._pusula=self._build_pusula().to(self.device)
        gc.collect()

        self._pc=torch.zeros(1,dtype=torch.float32,device=self.device)
        self._log_buf=torch.zeros(
            STEER_LAYERS,4,dtype=torch.float32,device=self.device)
        self._hooks=self._inject(self._pusula)
        print(f"✅ Ready — {len(self._hooks)}/28 layers hooked\n")

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

    def _inject(self,p):
        layers=self.model.model.layers; hooks=[]
        def make_hook(i,p):
            def hook(mod,inp,out):
                hs=out[0] if isinstance(out,tuple) else out
                od=hs.dtype
                if od!=torch.float32: hs=hs.float()
                if not hs.is_contiguous(): hs=hs.contiguous()
                B,S,D=hs.shape; bs=B*S
                if self._pc.shape[0]!=bs:
                    self._pc=torch.zeros(bs,dtype=torch.float32,device=hs.device)
                st=_kernel.akbas_steer(
                    hs,p,KERNEL_V0,i,
                    KERNEL_OMEGA,KERNEL_A,KERNEL_P_INF,
                    self._pc,self._log_buf)
                if od!=torch.float32: st=st.to(od)
                return (st,)+out[1:] if isinstance(out,tuple) else st
            return hook
        for i in range(min(STEER_LAYERS,len(layers))):
            hooks.append(layers[i].register_forward_hook(make_hook(i,p)))
        return hooks

    def _remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks=[]

    def _prompt(self,user):
        return (f"<|im_start|>system\nYou are a helpful, accurate, and "
                f"thoughtful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user.strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    def _build_log_block(self,ms,tps,n_out):
        lb=self._log_buf.cpu(); W=54; lines=[]
        lines.append("─"*W)
        lines.append(f"⏱ {ms:.0f}ms | {tps:.1f} t/s | {n_out} token")
        lines.append(f"⚙ temp={TEMPERATURE} top_p={TOP_P} "
                     f"top_k={TOP_K} rep={REP_PENALTY}")
        lines.append(f"🔱 ω={KERNEL_OMEGA} A={KERNEL_A} "
                     f"P∞={KERNEL_P_INF} | layers 0-{STEER_LAYERS-1}/28")
        lines.append("─"*W)
        lines.append("KERNEL ACTIVATION LOG")
        lines.append("Values written by C++ akbas_steer() — 100% live")
        lines.append("Formula: P_t = cos(θ) × [A·e^(-ωt)·(1+ωt) + P∞]")
        lines.append("─"*W)
        lines.append(f"{'L':>3} {'cos(θ)':>8} {'kb':>8} "
                     f"{'kv':>8} {'katki':>9}  note")
        lines.append("─"*W)
        cos_vals=[]
        for i in range(STEER_LAYERS):
            cs=lb[i,0].item(); kb=lb[i,1].item()
            kv=lb[i,2].item(); kat=lb[i,3].item()
            if kb==0.0 and i>0: continue
            cos_vals.append(cs)
            t=float(i)
            kb_t=KERNEL_A*math.exp(-KERNEL_OMEGA*t)*(1+KERNEL_OMEGA*t)+KERNEL_P_INF
            note="← eq." if abs(kb_t-KERNEL_P_INF)<0.05 else ""
            lines.append(f"{i:>3} {cs:>+8.4f} {kb:>8.5f} "
                         f"{kv:>8.5f} {kat:>+9.6f}  {note}")
        lines.append("─"*W)
        if cos_vals:
            drift=cos_vals[-1]-cos_vals[0]
            lines.append(f"cos(θ) L0={cos_vals[0]:+.4f} → "
                         f"L19={cos_vals[-1]:+.4f}  drift={drift:+.4f}")
            final="ALIGNED ✓" if cos_vals[-1]>0 else "MISALIGNED ✗"
            lines.append(f"Final: {final} | "
                         f"P∞={KERNEL_P_INF} ω={KERNEL_OMEGA} A={KERNEL_A}")
        lines.append("─"*W)
        return "\n".join(lines)

    def sor(self,prompt,max_tokens=512):
        if not prompt.strip(): return ""
        self._pc.fill_(0.0)
        self._log_buf.zero_()
        self._remove_hooks()
        self._hooks=self._inject(self._pusula)

        inp=self.tok(self._prompt(prompt),return_tensors="pt").to(self.device)
        n_in=inp["input_ids"].shape[1]
        eos=list({self.tok.eos_token_id,
                  self.tok.convert_tokens_to_ids("<|im_end|>"),
                  self.tok.convert_tokens_to_ids("<|endoftext|>")}-{None,-1})

        t0=time.time()
        with torch.no_grad():
            out=self.model.generate(
                **inp,max_new_tokens=int(max_tokens),
                do_sample=True,temperature=TEMPERATURE,
                top_p=TOP_P,top_k=TOP_K,
                repetition_penalty=REP_PENALTY,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=eos)
        ms=(time.time()-t0)*1000
        n_out=out.shape[1]-n_in
        tps=n_out/(ms/1000) if ms>0 else 0

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        result=self.tok.decode(out[0][n_in:],skip_special_tokens=True).strip()
        lines=result.split("\n")
        fi=next((i for i,l in enumerate(lines) if l.strip()),None)
        if fi is not None and any(
                m in lines[fi].lower() for m in DISCLAIMER_MARKERS):
            result="\n".join(l for l in lines[fi+1:] if l.strip())

        return result+"\n\n"+self._build_log_block(ms,tps,n_out)

# =============================================================================
# LOAD
# =============================================================================
print("\n"+"="*55)
print("🔱 AKBASCORE 1.0 | QWEN2.5-1.5B | LOADING...")
print("="*55+"\n")

akbas=AkbasCore()
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI
# =============================================================================
with gr.Blocks(
    title="🔱 AkbasCore 1.0",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono")),
    css="""
    body { background: #040904 }
    .gradio-container {
        max-width: 920px !important;
        margin: 0 auto;
        background: #060b06 !important;
    }
    #hdr {
        text-align: center;
        padding: 28px 0 10px;
        border-bottom: 1px solid #1a3a20;
        margin-bottom: 20px;
    }
    #hdr h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.30rem;
        color: #00ff88;
        letter-spacing: .18em;
        margin: 0 0 6px;
        text-shadow: 0 0 22px #00ff8844;
    }
    #hdr p {
        font-size: .63rem;
        color: #2a5a38;
        margin: 0;
        letter-spacing: .08em;
    }
    textarea {
        background: #080f08 !important;
        color: #a8e0b0 !important;
        border: 1px solid #1a3520 !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .86rem !important;
        line-height: 1.65 !important;
    }
    textarea:focus {
        border-color: #00cc66 !important;
        box-shadow: 0 0 10px #00cc6618 !important;
    }
    #out textarea {
        background: #030703 !important;
        color: #44ee77 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .82rem !important;
        border: 1px solid #102010 !important;
        line-height: 1.75 !important;
    }
    input[type=range] { accent-color: #00cc66 }
    #btn {
        background: linear-gradient(135deg, #002e12, #005522) !important;
        color: #00ff88 !important;
        border: 1px solid #007733 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .90rem !important;
        letter-spacing: .12em !important;
        border-radius: 4px !important;
        transition: all .18s;
    }
    #btn:hover {
        background: linear-gradient(135deg, #004020, #008040) !important;
        box-shadow: 0 0 14px #00cc6628 !important;
    }
    label span {
        color: #3a7a50 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .75rem !important;
        letter-spacing: .05em !important;
    }
    """,
) as demo:

    with gr.Column(elem_id="hdr"):
        gr.HTML("""
        <h1>🔱 AKBASCORE 1.0 &nbsp;|&nbsp; QWEN2.5-1.5B</h1>
        <p>
          CLOSED-LOOP C++ KERNEL &nbsp;|&nbsp;
          20/28 LAYERS &nbsp;|&nbsp;
          4D CONSTITUTION &nbsp;|&nbsp;
          LIVE LOG &nbsp;|&nbsp;
          TEMP 0.65 &nbsp;|&nbsp;
          BF16 &nbsp;|&nbsp;
          APACHE 2.0
        </p>
        """)

    inp=gr.Textbox(
        label="► INPUT",
        lines=5,
        placeholder="Ask your question here...",
        show_copy_button=False)

    with gr.Row():
        sl=gr.Slider(64,1024,value=512,step=64,label="MAX TOKENS",scale=4)
        btn=gr.Button("▶  SEND",variant="primary",elem_id="btn",scale=1)

    out=gr.Textbox(
        label="◈ OUTPUT  +  KERNEL ACTIVATION LOG",
        lines=30,
        interactive=False,
        show_copy_button=True,
        elem_id="out")

    btn.click(fn=akbas.sor,inputs=[inp,sl],outputs=out)
    inp.submit(fn=akbas.sor,inputs=[inp,sl],outputs=out)

print("🚀 Launching Gradio...")
demo.launch(share=True,debug=False)
