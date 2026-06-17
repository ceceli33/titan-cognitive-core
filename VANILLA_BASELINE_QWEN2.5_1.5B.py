# =============================================================================
# ⬜ VANILLA BASELINE | QWEN2.5-1.5B-INSTRUCT — GRADIO UI
# =============================================================================
# AkbasCore 1.0 karşılaştırması için SAF MODEL — hiçbir kernel yok.
# Sadece standart HuggingFace generate() çağrısı.
# Aynı parametreler (temp, top_p, top_k, rep_penalty) kullanılıyor.
#
# HOW TO RUN:
#   1. colab.research.google.com → New notebook
#   2. Runtime → Change runtime type → CPU
#   3. Paste this entire file into ONE cell → Run
#   4. Open the Gradio link → type your question → send
#
# Model  : Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)
# Kernel : YOK — vanilla HuggingFace generate()
# =============================================================================

import subprocess, sys, time, gc

def _pip(pkg):
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"],
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

for _pkg,_imp in [("gradio","gradio"),("bitsandbytes>=0.46.1","bitsandbytes")]:
    try:
        m=__import__(_imp)
        if _imp=="bitsandbytes":
            if tuple(int(x) for x in m.__version__.split(".")[:3])<(0,46,1):
                raise ImportError
    except ImportError:
        print(f"installing {_pkg}..."); _pip(_pkg); print(f"✅ {_imp} ready")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# AYNI PARAMETRELER — AkbasCore 1.0 ile bire bir eşleşiyor
# =============================================================================
MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE = 0.65
TOP_P       = 0.90
TOP_K       = 50
REP_PENALTY = 1.15

DISCLAIMER_MARKERS = [
    "as an ai","as a language model","i apologize","i must clarify",
    "i'm unable to","i am unable to","i cannot be certain",
]

class VanillaBaseline:

    def __init__(self):
        print("⬜ VANILLA BASELINE | Qwen2.5-1.5B-Instruct")
        print("   Kernel YOK — saf HuggingFace generate()")
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
        print("✅ Ready — no hooks, no steering\n")

    def _prompt(self,user):
        return (f"<|im_start|>system\nYou are a helpful, accurate, and "
                f"thoughtful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user.strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    def _build_log_block(self,ms,tps,n_in,n_out,device_str):
        W=54; lines=[]
        lines.append("─"*W)
        lines.append(f"⏱ {ms:.0f}ms | {tps:.1f} t/s | {n_out} token")
        lines.append(f"⚙ temp={TEMPERATURE} top_p={TOP_P} "
                     f"top_k={TOP_K} rep={REP_PENALTY}")
        lines.append(f"📥 input tokens: {n_in}")
        lines.append(f"💻 device: {device_str}")
        lines.append("─"*W)
        lines.append("VANILLA BASELINE — KERNEL AKTIVASYONU YOK")
        lines.append("Saf Qwen2.5-1.5B-Instruct çıktısı")
        lines.append("Hiçbir hidden-state müdahalesi uygulanmadı")
        lines.append("─"*W)
        lines.append(f"{'Ölçüm':<30} {'Değer':>20}")
        lines.append("─"*W)

        # Timing breakdown
        tok_ms = (n_in / (n_in+n_out)) * ms if n_out>0 else 0
        gen_ms = ms - tok_ms
        lines.append(f"{'Toplam süre (ms)':<30} {ms:>20.1f}")
        lines.append(f"{'Üretim süresi (ms, tahmin)':<30} {gen_ms:>20.1f}")
        lines.append(f"{'Tokens/saniye':<30} {tps:>20.2f}")
        lines.append(f"{'ms/token':<30} {(1000/tps if tps>0 else 0):>20.2f}")
        lines.append(f"{'Input token sayısı':<30} {n_in:>20}")
        lines.append(f"{'Output token sayısı':<30} {n_out:>20}")
        lines.append(f"{'Toplam token':<30} {n_in+n_out:>20}")
        lines.append("─"*W)
        lines.append("⚠  AkbasCore 1.0 ile karşılaştırmak için")
        lines.append("   aynı soruyu her iki arayüzde de deneyin.")
        lines.append("   Fark: steering kernel YOK → saf model davranışı")
        lines.append("─"*W)
        return "\n".join(lines)

    def generate(self,prompt,max_tokens=512):
        if not prompt.strip(): return ""

        inp=self.tok(self._prompt(prompt),return_tensors="pt").to(self.device)
        n_in=inp["input_ids"].shape[1]
        eos=list({self.tok.eos_token_id,
                  self.tok.convert_tokens_to_ids("<|im_end|>"),
                  self.tok.convert_tokens_to_ids("<|endoftext|>")}-{None,-1})

        device_str=(torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "CPU")

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

        return result+"\n\n"+self._build_log_block(ms,tps,n_in,n_out,device_str)

# =============================================================================
# LOAD
# =============================================================================
print("\n"+"="*55)
print("⬜ VANILLA BASELINE | QWEN2.5-1.5B | LOADING...")
print("="*55+"\n")

model=VanillaBaseline()
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# =============================================================================
# GRADIO UI — AkbasCore ile aynı layout, renk paleti farklı (gri/beyaz)
# =============================================================================
with gr.Blocks(
    title="⬜ Vanilla Baseline",
    theme=gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono")),
    css="""
    body { background: #07090c }
    .gradio-container {
        max-width: 920px !important;
        margin: 0 auto;
        background: #090c10 !important;
    }
    #hdr {
        text-align: center;
        padding: 28px 0 10px;
        border-bottom: 1px solid #1e2733;
        margin-bottom: 20px;
    }
    #hdr h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.30rem;
        color: #c8d8f0;
        letter-spacing: .18em;
        margin: 0 0 6px;
        text-shadow: 0 0 22px #6699cc44;
    }
    #hdr p {
        font-size: .63rem;
        color: #2a3a4a;
        margin: 0;
        letter-spacing: .08em;
    }
    #badge {
        display: inline-block;
        background: #0f1922;
        border: 1px solid #223344;
        color: #7aadcc;
        font-family: 'JetBrains Mono', monospace;
        font-size: .68rem;
        padding: 4px 14px;
        border-radius: 2px;
        letter-spacing: .10em;
        margin-top: 8px;
    }
    textarea {
        background: #080c12 !important;
        color: #b0c8e0 !important;
        border: 1px solid #1a2535 !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .86rem !important;
        line-height: 1.65 !important;
    }
    textarea:focus {
        border-color: #4488bb !important;
        box-shadow: 0 0 10px #4488bb18 !important;
    }
    #out textarea {
        background: #050810 !important;
        color: #88bbdd !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .82rem !important;
        border: 1px solid #101820 !important;
        line-height: 1.75 !important;
    }
    input[type=range] { accent-color: #4488bb }
    #btn {
        background: linear-gradient(135deg, #0d1e2e, #1a3550) !important;
        color: #88ccee !important;
        border: 1px solid #2a5577 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .90rem !important;
        letter-spacing: .12em !important;
        border-radius: 4px !important;
        transition: all .18s;
    }
    #btn:hover {
        background: linear-gradient(135deg, #162840, #224466) !important;
        box-shadow: 0 0 14px #4488bb28 !important;
    }
    label span {
        color: #3a5a7a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .75rem !important;
        letter-spacing: .05em !important;
    }
    """,
) as demo:

    with gr.Column(elem_id="hdr"):
        gr.HTML("""
        <h1>⬜ VANILLA BASELINE &nbsp;|&nbsp; QWEN2.5-1.5B</h1>
        <p>
          NO KERNEL &nbsp;|&nbsp;
          NO HOOKS &nbsp;|&nbsp;
          NO STEERING &nbsp;|&nbsp;
          SAF MODEL ÇIKTISI &nbsp;|&nbsp;
          TEMP 0.65 &nbsp;|&nbsp;
          BF16 &nbsp;|&nbsp;
          APACHE 2.0
        </p>
        <div id="badge">⚠ AkbasCore 1.0 karşılaştırma referansı — kernel YOK</div>
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
        label="◈ OUTPUT  +  VANILLA PERFORMANCE LOG",
        lines=30,
        interactive=False,
        show_copy_button=True,
        elem_id="out")

    btn.click(fn=model.generate,inputs=[inp,sl],outputs=out)
    inp.submit(fn=model.generate,inputs=[inp,sl],outputs=out)

print("🚀 Launching Gradio...")
demo.launch(share=True,debug=False)
