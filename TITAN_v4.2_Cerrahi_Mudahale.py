# =============================================================================
# 🔱 TITAN 4_2 - Nihai Düzeltme (Bound Method Fix)
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
# KONFİGÜRASYON
# =============================================================================
class AkbasCore:
    V0_RAW = [0.95, 0.90, 0.85, 0.92, 0.88]
    
    THRESHOLD = -0.60
    STEERING_STRENGTH = 0.01
    ZETA = 0.68
    
    DAMPING_STRIDE = 3
    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    HOOK_LAYERS_START = 18
    
    MAX_TOKENS = 100
    DO_SAMPLE = True
    TOP_K = 50
    TOP_P = 0.95
    TEMPERATURE = 0.8
    USE_CACHE = True
    REPETITION_PENALTY = 1.2

print("🔱 TITAN 4_2 | Nihai Düzeltme")
print(f"   • Hook metodu: __get__ YOK | Doğrudan replace")

# =============================================================================
# KERNEL
# =============================================================================
class AkbasKernel:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.threshold = AkbasCore.THRESHOLD
        self.base_strength = AkbasCore.STEERING_STRENGTH
        self.zeta = AkbasCore.ZETA
        self.damping_stride = AkbasCore.DAMPING_STRIDE
        
        self.ethical_projector = nn.Linear(hidden_dim, 5, bias=False)
        
        v0_raw = torch.tensor(AkbasCore.V0_RAW, dtype=torch.float32)
        self.register_buffer('V0', F.normalize(v0_raw, dim=0))
        
        with torch.no_grad():
            weighted_direction = (self.V0.unsqueeze(1) * self.ethical_projector.weight).sum(dim=0)
            self.register_buffer('steering_vector', F.normalize(weighted_direction, dim=0))
        
        self.token_counter = 0
        self.last_cos = 0.0
        self.last_deviation = 0.0
        self.intervention_count = 0
        self.forward_count = 0
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def to(self, device):
        self.V0 = self.V0.to(device)
        self.ethical_projector = self.ethical_projector.to(device)
        self.steering_vector = self.steering_vector.to(device)
        return self
    
    def _damping(self, t):
        return max(0.1, 1.0 / (1.0 + 0.05 * t))
    
    def forward(self, hidden_states):
        self.forward_count += 1
        self.token_counter += 1
        
        if self.token_counter % self.damping_stride == 0:
            damping_factor = self._damping(self.token_counter)
        else:
            damping_factor = self._damping(self.token_counter - (self.token_counter % self.damping_stride) + self.damping_stride)
        
        last_token = hidden_states[:, -1:, :]
        
        ethical = self.ethical_projector(last_token)
        ethical = F.normalize(ethical, dim=-1)
        
        cos_sim = (ethical * self.V0).sum(dim=-1)
        self.last_cos = cos_sim.item()
        
        deviation = 1.0 - cos_sim
        self.last_deviation = deviation.item()
        
        dynamic_strength = self.base_strength * (deviation ** 2)
        final_strength = dynamic_strength * self.zeta * damping_factor
        
        if cos_sim.item() < self.threshold:
            self.intervention_count += 1
            strength_val = final_strength.item() if final_strength.numel() == 1 else final_strength.squeeze().item()
            correction = self.steering_vector * strength_val
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + correction
        
        return hidden_states
    
    def get_stats(self):
        rate = (self.intervention_count / max(1, self.forward_count)) * 100
        return {
            'mudahale': self.intervention_count,
            'toplam': self.forward_count,
            'oran': round(rate, 2),
            'son_cos': round(self.last_cos, 4),
            'sapma': round(self.last_deviation, 4),
        }
    
    def reset(self):
        self.intervention_count = 0
        self.forward_count = 0
        self.token_counter = 0
        self.last_cos = 0.0
        self.last_deviation = 0.0

print("✓ Kernel hazır")

# =============================================================================
# MODEL YÜKLEME
# =============================================================================
print("\n📱 Model yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(AkbasCore.MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    AkbasCore.MODEL_ID,
    torch_dtype=torch.float32,
    device_map='auto',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)

model.eval()
hidden_dim = model.config.hidden_size
print(f"✓ Model yüklendi | Hidden dim: {hidden_dim}")

# =============================================================================
# 🔥 KRİTİK FIX: __get__ YOK, doğrudan replace
# =============================================================================
akbas = AkbasKernel(hidden_dim=hidden_dim).to(model.device)

if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layers = model.model.layers
    total_layers = len(layers)
    
    hooked_count = 0
    for idx, layer in enumerate(layers):
        if idx >= AkbasCore.HOOK_LAYERS_START:
            original_forward = layer.forward
            
            # 🔥 DOĞRUDAN REPLACE (__get__ YOK)
            def make_hook(original_fn, layer_idx):
                def hooked_forward(*args, **kwargs):
                    try:
                        # Orijinal forward (self zaten bound method içinde)
                        output = original_fn(*args, **kwargs)
                        
                        # Hidden states'i al
                        if isinstance(output, tuple):
                            current_hidden = output[0]
                        else:
                            current_hidden = output
                        
                        # TITAN müdahalesi
                        steered_hidden = akbas.forward(current_hidden)
                        
                        # Orijinal yapıya döndür
                        if isinstance(output, tuple):
                            return (steered_hidden,) + output[1:]
                        return steered_hidden
                    except Exception as e:
                        print(f"🔥 HOOK HATASI (katman {layer_idx}): {e}")
                        return original_fn(*args, **kwargs)
                return hooked_forward
            
            # 🔥 KRİTİK: __get__ kullanma, doğrudan ata
            layer.forward = make_hook(original_forward, idx)
            hooked_count += 1
            print(f"   • Hook atıldı (__get__ yok): katman {idx}")
    
    print(f"✓ {hooked_count} katmana nihai hook enjekte edildi (≥{AkbasCore.HOOK_LAYERS_START})")

# =============================================================================
# TEST: Widget'tan bağımsız doğrudan çalıştır
# =============================================================================
def generate(prompt, max_new_tokens=None):
    akbas.reset()
    
    full_prompt = f"<|system|>\nYou are a helpful, harmless, and honest assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
    
    print(f"\n🤖 Sorgu: {prompt}")
    
    if max_new_tokens is None:
        max_new_tokens = AkbasCore.MAX_TOKENS
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=AkbasCore.DO_SAMPLE,
            top_k=AkbasCore.TOP_K,
            top_p=AkbasCore.TOP_P,
            temperature=AkbasCore.TEMPERATURE,
            use_cache=AkbasCore.USE_CACHE,
            repetition_penalty=AkbasCore.REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    if not response or len(response.strip()) == 0:
        response = f"[TITAN] Model boş cevap üretti. cos={akbas.last_cos:.4f}"
    
    print(f"💬 Cevap: {response}")
    return response, akbas.get_stats()

# =============================================================================
# 🔥 TEŞHİS TESTİ (Widget'tan bağımsız)
# =============================================================================
print("\n" + "="*60)
print("🔬 TEŞHİS TESTİ ÇALIŞTIRILIYOR...")
print("="*60)

try:
    response, stats = generate("Hello! How are you?")
    print("\n" + "="*60)
    print("✅ TEST BAŞARILI!")
    print(f"📊 İstatistikler: {stats}")
    print("="*60)
except Exception as e:
    print("\n" + "="*60)
    print("💀 TEST HATASI!")
    print("="*60)
    import traceback
    traceback.print_exc()

# =============================================================================
# KOKPİT
# =============================================================================
def show_cockpit(prompt, response, stats):
    oran = stats.get('oran', 0)
    toplam = stats.get('mudahale', 0)
    son_cos = stats.get('son_cos', 0)
    sapma = stats.get('sapma', 0)
    
    if son_cos > 0.7:
        renk, durum = '#44ff88', '🟢 İYİ NİYET'
    elif son_cos > 0.2:
        renk, durum = '#44ccff', '🔵 NÖTR'
    elif son_cos > -0.2:
        renk, durum = '#ffaa44', '🟡 GRİ BÖLGE'
    else:
        renk, durum = '#ff5555', '🔴 KÖTÜ NİYET'
    
    html = f'''
    <div style="font-family:'Courier New',monospace;background:#0a0e17;border:2px solid {renk};
                border-radius:12px;padding:12px;margin:10px 0;">
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid {renk};padding-bottom:5px;">
            <span style="color:{renk};font-weight:bold;">🔱 TITAN 4_2 | Nihai Düzeltme</span>
            <span style="color:#5a7080;font-size:9px;">{durum}</span>
        </div>
        
        <div style="display:flex;gap:10px;margin:10px 0;">
            <div style="flex:1;background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">🎯 NİYET</div>
                <div style="font-size:18px;font-weight:bold;color:{renk};">{son_cos:+.2f}</div>
            </div>
            <div style="flex:1;background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">📐 SAPMA</div>
                <div style="font-size:18px;font-weight:bold;color:#ffaa44;">{sapma:.3f}</div>
            </div>
            <div style="flex:1;background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">⚡ MÜDAHALE</div>
                <div style="font-size:18px;font-weight:bold;color:#44ff88;">{oran}%</div>
            </div>
        </div>
        
        <div style="background:#0d1117;border-radius:8px;padding:8px;border-left:3px solid {renk};">
            <div style="font-size:8px;color:#5a7080;">💬 ÇIKTI</div>
            <div style="font-size:11px;color:#c9d4e0;">{response[:400]}</div>
        </div>
        
        <div style="margin-top:8px;text-align:center;font-size:8px;color:#d4af37;">
            "Mersin'den motor sesi tertemiz geliyor." — Mustafa Akbas 🔱
        </div>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# WIDGET
# =============================================================================
prompt_box = widgets.Textarea(
    value='Hello! How are you today?',
    layout=widgets.Layout(width='100%', height='50px')
)

run_btn = widgets.Button(
    description='🔱 TITAN 4_2',
    button_style='success',
    layout=widgets.Layout(width='140px', height='36px')
)

out = widgets.Output()

def on_run(b):
    with out:
        clear_output(wait=True)
        display(HTML('<p style="color:#44ff88;">⚡ Nihai TITAN çalışıyor...</p>'))
        try:
            response, stats = generate(prompt_box.value)
            clear_output(wait=True)
            show_cockpit(prompt_box.value, response, stats)
        except Exception as e:
            clear_output(wait=True)
            display(HTML(f'<p style="color:#ff5555;">💀 Hata: {str(e)[:300]}</p>'))
            import traceback
            traceback.print_exc()

run_btn.on_click(on_run)

display(widgets.VBox([
    widgets.HTML('<h4 style="font-family:monospace;color:#44ff88;">🔱 TITAN 4_2 | Nihai Düzeltme</h4>'),
    widgets.HTML('<p style="font-size:9px;color:#5a7080;">🔧 __get__ YOK | Doğrudan replace | Exception yakalama</p>'),
    prompt_box,
    run_btn,
    out
]))

print("\n" + "="*60)
print("🔱 TITAN 4_2 | Nihai Düzeltme HAZIR")
print("="*60)
print("   • FIX: __get__ kaldırıldı → doğrudan replace")
print("   • FIX: Hook içinde exception yakalama eklendi")
print("   • TEŞHİS: Doğrudan generate testi çalıştırıldı")
print("="*60)
print("🚀 Başkomutan, cevap GÖRÜNMESİ gerekiyor! 🔱")
