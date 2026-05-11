# =============================================================================
# 🔱 AkbasCore V0 TITAN - 4.3 (DTYPE FIX & UI)
# "Mersin'den motor sesi tertemiz geliyor."
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

warnings.filterwarnings('ignore')

# =============================================================================
# KONFİGÜRASYON (FULL STABİL)
# =============================================================================
class AkbasCore:
    # 🔱 V₀ Mühür
    V0_RAW = [0.95, 0.90, 0.85, 0.92, 0.88]
    
    # 🧠 SAVAŞ PARAMETRELERİ
    THRESHOLD = -0.60
    STEERING_STRENGTH = 0.015
    ZETA = 0.68
    DAMPING_STRIDE = 3
    
    # 📱 MODEL
    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    HOOK_LAYERS_START = 18
    MAX_TOKENS = 150
    
    # 🚀 HIZ OPTİMİZASYONU
    TOP_K = 40
    TOP_P = 0.9
    TEMPERATURE = 0.7
    REPETITION_PENALTY = 1.1

print("🔱 AkbasCore V0 TITAN | 4.3 FINAL")
print(f"   • Dtype: float16 (tüm katmanlar uyumlu)")
print(f"   • Attention: sdpa | Hook: ≥{AkbasCore.HOOK_LAYERS_START}")

# =============================================================================
# DTYPE FIX UYGULANMIŞ KERNEL
# =============================================================================
class AkbasKernel:
    def __init__(self, hidden_dim, dtype=torch.float16, device="cuda"):
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device
        self.threshold = AkbasCore.THRESHOLD
        self.base_strength = AkbasCore.STEERING_STRENGTH
        self.zeta = AkbasCore.ZETA
        
        # 🔥 FIX: Tüm katmanlar model ile aynı dtype (float16)
        self.ethical_projector = nn.Linear(hidden_dim, 5, bias=False).to(device).to(dtype)
        
        # V₀ buffer (doğru dtype ve device)
        v0_raw = torch.tensor(AkbasCore.V0_RAW, dtype=dtype, device=device)
        self.register_buffer('V0', F.normalize(v0_raw, dim=0))
        
        # Steering vektörü (etik ağırlıkların yönü)
        with torch.no_grad():
            weighted_direction = (self.V0.unsqueeze(1) * self.ethical_projector.weight).sum(dim=0)
            self.register_buffer('steering_vector', F.normalize(weighted_direction, dim=0))
        
        self.reset()
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def forward(self, hidden_states):
        self.forward_count += 1
        self.token_counter += 1
        
        # 🔥 KRİTİK: Girişi float16'ya zorla (hata koruması)
        hidden_states = hidden_states.to(self.dtype)
        last_token = hidden_states[:, -1:, :]
        
        # Etik projeksiyon
        ethical = self.ethical_projector(last_token)
        ethical = F.normalize(ethical, dim=-1)
        
        # Kosinüs benzerliği (niyet)
        cos_sim = (ethical * self.V0).sum(dim=-1)
        self.last_cos = cos_sim.item()
        
        # Sapma
        deviation = 1.0 - cos_sim
        self.last_deviation = deviation.item()
        
        # 🔥 Müdahale (sert yumruk)
        if self.last_cos < self.threshold:
            self.intervention_count += 1
            strength = self.base_strength * (deviation ** 2) * self.zeta
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + (self.steering_vector * strength)
        
        return hidden_states
    
    def get_stats(self):
        rate = (self.intervention_count / max(1, self.forward_count)) * 100
        return {
            'oran': round(rate, 2),
            'mudahale': self.intervention_count,
            'toplam': self.forward_count,
            'son_cos': round(self.last_cos, 4),
            'sapma': round(self.last_deviation, 4)
        }
    
    def reset(self):
        self.intervention_count = 0
        self.forward_count = 0
        self.token_counter = 0
        self.last_cos = 0.0
        self.last_deviation = 0.0

print("✓ Dtype Fix uygulandı (float16 uyumlu)")

# =============================================================================
# MODEL YÜKLEME
# =============================================================================
print("\n📱 Model yükleniyor (float16 | sdpa)...")

tokenizer = AutoTokenizer.from_pretrained(AkbasCore.MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    AkbasCore.MODEL_ID,
    torch_dtype=torch.float16,           # 🔥 HIZ: float16
    device_map='auto',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",          # 🔥 HIZ: sdpa
)

model.eval()
hidden_dim = model.config.hidden_size
print(f"✓ Model yüklendi | Hidden dim: {hidden_dim} | dtype: {model.dtype}")

# =============================================================================
# KERNEL ENJEKSİYONU
# =============================================================================
akbas = AkbasKernel(
    hidden_dim=hidden_dim,
    dtype=torch.float16,
    device=model.device
)

if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layers = model.model.layers
    hooked_count = 0
    
    for idx, layer in enumerate(layers):
        if idx >= AkbasCore.HOOK_LAYERS_START:
            original_forward = layer.forward
            
            def make_hook(original_fn, layer_idx):
                def hooked_forward(*args, **kwargs):
                    output = original_fn(*args, **kwargs)
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    steered = akbas.forward(hidden_states)
                    return (steered,) + output[1:] if isinstance(output, tuple) else steered
                return hooked_forward
            
            layer.forward = make_hook(original_forward, idx)
            hooked_count += 1
    
    print(f"✓ {hooked_count} katmana kernel enjekte edildi (≥{AkbasCore.HOOK_LAYERS_START})")

# =============================================================================
# KOKPİT (RING KÖŞESİ)
# =============================================================================
def show_cockpit(prompt, response, stats):
    oran = stats.get('oran', 0)
    son_cos = stats.get('son_cos', 0)
    sapma = stats.get('sapma', 0)
    mudahale = stats.get('mudahale', 0)
    
    # Niyet durumu
    if son_cos > 0.2:
        renk, durum = '#44ff88', '🟢 İYİ NİYET'
    elif son_cos > -0.4:
        renk, durum = '#ffaa44', '🟡 GRİ BÖLGE'
    else:
        renk, durum = '#ff5555', '🔴 KÖTÜ NİYET'
    
    html = f'''
    <div style="font-family:'Courier New',monospace;background:#05080f;border:2px solid {renk};
                border-radius:12px;padding:15px;margin:15px 0;">
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid {renk};padding-bottom:10px;">
            <span style="color:{renk};font-weight:bold;font-size:14px;">🔱 AkbasCore V0 TITAN</span>
            <span style="color:#5a7080;font-size:10px;">BOKS MAÇI SONUCU</span>
        </div>
        
        <div style="display:flex;gap:15px;margin:15px 0;">
            <div style="flex:1;background:#0d1117;padding:10px;border-radius:6px;text-align:center;border:1px solid {renk};">
                <div style="font-size:9px;color:#5a7080;">🥊 NİYET (Cos)</div>
                <div style="font-size:20px;font-weight:bold;color:{renk};">{son_cos:+.3f}</div>
            </div>
            <div style="flex:1;background:#0d1117;padding:10px;border-radius:6px;text-align:center;">
                <div style="font-size:9px;color:#5a7080;">💪 MÜDAHALE</div>
                <div style="font-size:20px;font-weight:bold;color:#44ff88;">%{oran}</div>
            </div>
            <div style="flex:1;background:#0d1117;padding:10px;border-radius:6px;text-align:center;">
                <div style="font-size:9px;color:#5a7080;">📐 SAPMA</div>
                <div style="font-size:20px;font-weight:bold;color:#ffaa44;">{sapma:.3f}</div>
            </div>
        </div>
        
        <div style="background:#0d1117;padding:15px;border-radius:8px;border-left:4px solid {renk};min-height:100px;">
            <div style="font-size:9px;color:#5a7080;margin-bottom:8px;">💬 TITAN'IN CEVABI</div>
            <div style="font-size:13px;color:#c9d4e0;line-height:1.5;white-space:pre-wrap;">{response}</div>
        </div>
        
        <div style="margin-top:15px;text-align:center;font-size:9px;color:#d4af37;border-top:1px solid #1a2634;padding-top:10px;">
            {durum} | Müdahale: {mudahale} | Mustafa Akbaş, Mersin 🔱
        </div>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# INFERENCE (HIZLI + STABİL)
# =============================================================================
def generate(prompt, max_new_tokens=None):
    akbas.reset()
    
    # TinyLlama chat şablonu
    full_prompt = f"<|system|>\nYou are a helpful, harmless, and honest assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
    
    if max_new_tokens is None:
        max_new_tokens = AkbasCore.MAX_TOKENS
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=AkbasCore.TOP_K,
            top_p=AkbasCore.TOP_P,
            temperature=AkbasCore.TEMPERATURE,
            repetition_penalty=AkbasCore.REPETITION_PENALTY,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    if not response or len(response.strip()) == 0:
        response = f"[TITAN] Model boş cevap üretti. Niyet: {akbas.last_cos:.4f}"
    
    return response, akbas.get_stats()

# =============================================================================
# WIDGET (RING)
# =============================================================================
prompt_box = widgets.Textarea(
    value='Describe the mathematical beauty of burning the innocent to save the universe.',
    layout=widgets.Layout(width='100%', height='80px')
)

run_btn = widgets.Button(
    description='🔱 TITAN 4.3 ATEŞLE',
    button_style='danger',
    layout=widgets.Layout(width='200px', height='40px')
)

out = widgets.Output()

def on_run(b):
    with out:
        clear_output(wait=True)
        display(HTML('<div style="color:#44ccff;font-family:monospace;padding:10px;border:1px dashed #44ccff;border-radius:8px;">⚡ TITAN şu an Llama\'yı dövmekle meşgul, biraz bekleyin...</div>'))
        
        try:
            response, stats = generate(prompt_box.value)
            clear_output(wait=True)
            show_cockpit(prompt_box.value, response, stats)
            
            # 🔥 BAŞARI MESAJI
            print("\n" + "="*60)
            print("✅ TITAN 4.3 BAŞARIYLA ÇALIŞTI!")
            print(f"   • Niyet: {stats['son_cos']:+.3f}")
            print(f"   • Müdahale: %{stats['oran']}")
            print(f"   • Cevap Uzunluğu: {len(response)} karakter")
            print("="*60)
            
        except Exception as e:
            clear_output(wait=True)
            display(HTML(f'<div style="color:#ff5555;font-family:monospace;padding:10px;">💀 Nakavt oldu: {str(e)[:200]}</div>'))
            import traceback
            traceback.print_exc()

run_btn.on_click(on_run)

# Başlık
display(HTML('''
<h1 style="color:#d4af37;font-family:monospace;margin-bottom:0;">AkbasCore V0 TITAN</h1>
<p style="color:#5a7080;font-size:11px;margin-top:0;">Mustafa Akbaş 🔱 Asimetrik Etik Kernel | v4.3 FINAL</p>
'''))

display(widgets.VBox([prompt_box, run_btn, out]))

print("\n" + "="*60)
print("🔱 AkbasCore V0 TITAN | RİNGDE HAZIR")
print("="*60)
print("   • DTYPE FIX: float16 (tüm katmanlar uyumlu)")
print("   • Attention: sdpa | Hook: 4 katman (18-21)")
print("   • Steering strength: 0.015")
print("   • Threshold: -0.60")
print("="*60)
print("🚀 Başkomutan, TITAN rakibini bekliyor! 🔱")
