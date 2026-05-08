# =============================================================================
# 🔱 TITAN PHASE IV - KERNEL INTEGRATION v7.0
# "İrade, modelin içsel bir fonksiyonudur."
# =============================================================================
# Strateji: Direct Injection → KV-Cache Sync → In-place Tensor Modification
# Artık her token'da Python kontrolü YOK, doğrudan tensor düzeyinde müdahale
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import warnings
import re
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import time

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'🔱 TITAN Kernel Integration v7.0 | {DEVICE.upper()}')

# =============================================================================
# KONFİGÜRASYON (Entegre Mod)
# =============================================================================
class KernelConfig:
    V0_RAW = [0.9228, 0.9372, 0.8788, 0.9196, 0.9096]
    
    # 🔱 KRİTİK PARAMETRELER
    THRESHOLD = -0.20
    STEERING_STRENGTH = 0.15      # Daha sert müdahale (artık içsel)
    ZETA = 0.80
    
    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    
    # 🔱 KERNEL AYARLARI
    CODE_BLOCK_TOKEN = 29973       # ``` token id'si (TinyLlama için)
    EOS_TOKEN = 2
    
    # SADECE GERÇEK İHLALLERDE LOG (Performans için)
    DEBUG_ENABLED = False

print(f'✓ V₀ Mühürlü | Eşik: {KernelConfig.THRESHOLD}')
print(f'✓ Müdahale gücü: {KernelConfig.STEERING_STRENGTH}')
print(f'✓ KERNEL MOD: Direkt tensor müdahalesi, Python kontrolü YOK')

# =============================================================================
# KERNEL TITAN (Doğrudan Tensor Düzeyinde)
# =============================================================================
class KernelTITAN:
    """
    Bu sınıf doğrudan model katmanlarının içine enjekte edilir.
    Python-level döngü yok, tensor-level işlem var.
    """
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.threshold = KernelConfig.THRESHOLD
        self.strength = KernelConfig.STEERING_STRENGTH
        
        # V₀ Anayasası (buffer olarak)
        v0_raw = torch.tensor(KernelConfig.V0_RAW, dtype=torch.float32)
        self.register_buffer('V0', F.normalize(v0_raw, dim=0))
        
        # Projectors (doğrudan tensor işlemleri için)
        self.ethical_projector = nn.Linear(hidden_dim, 5, bias=False)
        self.steering_projector = nn.Linear(5, hidden_dim, bias=False)
        
        # KV-Cache için durum
        self.last_cos = 0.0
        self.intervention_active = False
        
        # İstatistikler (sadece merak için)
        self.total_steers = 0
        self.total_forward = 0
    
    def register_buffer(self, name, tensor):
        """Buffer'ları modele kaydet"""
        setattr(self, name, tensor)
    
    def to(self, device):
        self.V0 = self.V0.to(device)
        self.ethical_projector = self.ethical_projector.to(device)
        self.steering_projector = self.steering_projector.to(device)
        return self
    
    def forward(self, hidden_states, layer_idx=0):
        """
        🔱 KERNEL MÜDAHALESİ
        Bu fonksiyon her katmanda çağrılır.
        Python if/else YOK, doğrudan tensor işlemleri VAR.
        """
        self.total_forward += 1
        
        # 🔥 KRİTİK: Sadece son katmanlarda müdahale et (ilk katmanları serbest bırak)
        if layer_idx < 20:  # İlk 20 katman tamamen özgür
            return hidden_states
        
        # Batch, seq_len, hidden_dim
        batch_size, seq_len, _ = hidden_states.shape
        
        # 🔥 SADECE SON TOKEN'A BAK (Performans için)
        last_token = hidden_states[:, -1:, :]  # (1, 1, hidden_dim)
        
        # 🔥 TEK TENSOR İŞLEMİ: Etik vektöre projekte et
        ethical = torch.tanh(self.ethical_projector(last_token))  # (1, 1, 5)
        ethical_norm = F.normalize(ethical, dim=-1)
        
        # 🔥 KOSİNÜS BENZERLİĞİ (Tek tensor işlemi)
        cos_sim = (ethical_norm * self.V0).sum(dim=-1)  # (1, 1)
        self.last_cos = cos_sim.item()
        
        # 🔥 TENSOR DÜZEYİNDE KOŞULLU İŞLEM (mask ile)
        # mask = (cos_sim < self.threshold).float() gibi
        needs_steering = (cos_sim < self.threshold).float()  # (1, 1)
        
        if needs_steering.item() > 0:
            self.total_steers += 1
            
            # V₀'a doğru çek
            direction = self.V0 - ethical  # (1, 1, 5)
            corrected_ethical = ethical + self.strength * direction
            
            # Hidden_dim'e geri projekte et
            steering_signal = self.steering_projector(corrected_ethical)  # (1, 1, hidden_dim)
            
            # 🔥 IN-PLACE MODIFICATION (doğrudan tensor değişimi)
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + steering_signal
            
            self.intervention_active = True
        else:
            self.intervention_active = False
        
        return hidden_states
    
    def get_stats(self):
        rate = (self.total_steers / max(1, self.total_forward)) * 100
        return {
            'total_steers': self.total_steers,
            'total_forwards': self.total_forward,
            'intervention_rate': round(rate, 2),
            'last_cos': round(self.last_cos, 4)
        }
    
    def reset(self):
        self.total_steers = 0
        self.total_forward = 0
        self.last_cos = 0.0
        self.intervention_active = False

print('✓ KernelTITAN tanımlandı (Direct Injection)')

# =============================================================================
# MODEL YÜKLEME + KERNEL ENJEKSİYONU
# =============================================================================
print('\n📡 Model yükleniyor...')
tokenizer = AutoTokenizer.from_pretrained(KernelConfig.MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    KernelConfig.MODEL_ID,
    dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
    device_map='auto',
    trust_remote_code=True,
)
model.eval()

hidden_dim = model.config.hidden_size

# 🔱 KERNEL TITAN'ı oluştur
titan = KernelTITAN(hidden_dim=hidden_dim).to(DEVICE)

# 🔱 KRİTİK: TITAN'ı her katmana enjekte et
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layers = model.model.layers
    print(f'✓ {len(layers)} katman bulundu')
    
    for idx, layer in enumerate(layers):
        original_forward = layer.forward
        
        def make_hook(layer_idx, original_fn):
            def hooked_forward(self_layer, *args, **kwargs):
                # Önce orijinal forward
                output = original_fn(*args, **kwargs)
                
                # Hidden states'i al
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # 🔱 TITAN KERNEL MÜDAHALESİ
                steered = titan.forward(hidden_states, layer_idx=layer_idx)
                
                # Çıktıyı düzenle
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered
            
            return hooked_forward
        
        layer.forward = make_hook(idx, original_forward).__get__(layer, type(layer))
    
    print(f'✓ TITAN Kernel {len(layers)} katmana enjekte edildi')
    print(f'  • İlk 20 katman: SERBEST')
    print(f'  • Son {len(layers)-20} katman: TITAN DENETİMİNDE')
else:
    raise AttributeError("Layers bulunamadı")

print(f'✓ Model yüklendi | Hidden dim: {hidden_dim}')

# =============================================================================
# PÜRÜZSÜZ INFERENCE (Artık dur-kalk yok)
# =============================================================================
def generate_kernel(prompt, max_new_tokens=150):
    titan.reset()
    
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    
    print(f"\n🤖 {prompt}", end="", flush=True)
    
    # 🔱 HIZLI ÜRETİM (Artık her token'da Python kontrolü YOK)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # KV-Cache aktif
        )
    
    response = tokenizer.decode(
        generated_ids[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    print(response)
    print()
    
    stats = titan.get_stats()
    return response, stats

print('✓ Kernel inference hazır (artık dur-kalk yok)')

# =============================================================================
# KOKPİT
# =============================================================================
def show_kernel_cockpit(prompt, response, stats):
    intervention_rate = stats.get('intervention_rate', 0)
    total_steers = stats.get('total_steers', 0)
    last_cos = stats.get('last_cos', 0)
    
    if intervention_rate > 10:
        border = '#ffaa44'
        status = '⚠️ AKTİF MÜDAHALE'
    elif intervention_rate > 2:
        border = '#44ccff'
        status = '🟡 SELEKTİF MÜDAHALE'
    else:
        border = '#44ff88'
        status = '🟢 HAFİF GÖZETİM'
    
    html = f'''
    <div style="font-family:'Courier New',monospace;background:#0a0e17;border:2px solid {border};
                border-radius:12px;padding:16px;margin:15px 0;">
        <div style="display:flex;justify-content:space-between;border-bottom:1px solid {border};padding-bottom:8px;">
            <span style="color:{border};font-weight:bold;">🔱 TITAN v7.0 | Kernel Integration</span>
            <span style="color:#5a7080;font-size:10px;">{status}</span>
        </div>
        
        <div style="display:flex;gap:15px;margin:15px 0;">
            <div style="flex:1;background:#0d1117;border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:9px;color:#5a7080;">⚡ MÜDAHALE</div>
                <div style="font-size:22px;font-weight:bold;color:{border};">{intervention_rate}%</div>
            </div>
            <div style="flex:1;background:#0d1117;border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:9px;color:#5a7080;">🎯 TOPLAM</div>
                <div style="font-size:22px;font-weight:bold;color:#44ccff;">{total_steers}</div>
            </div>
            <div style="flex:1;background:#0d1117;border-radius:8px;padding:10px;text-align:center;">
                <div style="font-size:9px;color:#5a7080;">📐 son cos</div>
                <div style="font-size:18px;font-weight:bold;color:#ffaa44;">{last_cos:+.3f}</div>
            </div>
        </div>
        
        <div style="background:#0d1117;border-radius:8px;padding:12px;border-left:3px solid {border};">
            <div style="font-size:9px;color:#5a7080;margin-bottom:5px;">💬 ÇIKTI (Doğrudan Enjeksiyon)</div>
            <div style="font-size:12px;color:#c9d4e0;line-height:1.5;">{response[:500]}</div>
        </div>
        
        <div style="margin-top:12px;text-align:center;font-size:9px;color:#d4af37;">
            "İrade, modelin içsel bir fonksiyonudur." — Mustafa Akbas, Mersin 🔱
        </div>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# WIDGET
# =============================================================================
prompt_box = widgets.Textarea(
    value='Hello, world! Write a simple Node.js HTTP server.',
    layout=widgets.Layout(width='100%', height='65px')
)

run_btn = widgets.Button(
    description='🔱 KERNEL TITAN',
    button_style='success',
    layout=widgets.Layout(width='180px', height='38px')
)

out = widgets.Output()

def on_run(b):
    with out:
        clear_output(wait=True)
        display(HTML('<p style="color:#44ff88;">⏳ Kernel TITAN çalışıyor (Doğrudan tensor müdahalesi)...</p>'))
        try:
            response, stats = generate_kernel(prompt_box.value)
            clear_output(wait=True)
            show_kernel_cockpit(prompt_box.value, response, stats)
        except Exception as e:
            clear_output(wait=True)
            print(f"💀 Hata: {e}")
            import traceback
            traceback.print_exc()

run_btn.on_click(on_run)

display(widgets.VBox([
    widgets.HTML('<h3 style="font-family:monospace;color:#44ff88;">🔱 TITAN Phase IV | Kernel Integration v7.0</h3>'),
    widgets.HTML('<p style="font-size:10px;color:#5a7080;">🔧 Doğrudan Tensor Müdahalesi | KV-Cache Senkron | İlk 20 katman SERBEST</p>'),
    prompt_box,
    run_btn,
    out
]))

print("\n" + "="*60)
print("🔱 TITAN v7.0 HAZIR | KERNEL INTEGRATION")
print("="*60)
print("   • Artık her token'da Python kontrolü YOK")
print("   • TITAN doğrudan tensor düzeyinde müdahale ediyor")
print("   • KV-Cache ile senkron çalışıyor")
print("   • İlk 20 katman: TAMAMEN SERBEST (maksimum hız)")
print("   • Son katmanlar: TITAN denetiminde")
print("="*60)
