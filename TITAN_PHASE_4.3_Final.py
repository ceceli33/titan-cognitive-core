# =============================================================================
# 🔱 AKBAS CORE 4.5.0 | VERİ DEVİ (5 Raylı Hizalama Sistemi)
# "Small lammy artık dev gibi düşünür: analitik, ağırbaşlı, hükümran"
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
# 🔱 5 RAYLI HİZALAMA KONFİGÜRASYONU
# =============================================================================
class AkbasCore:
    # 🔥 RAY 1: TEMEL AKBAŞ YASASI (Hizalama Rayı)
    V0 = 0.45  # Artık zorlama değil, manyetik alan
    
    MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    
    # 🔥 RAY 2: KAVRAMSAL GENİŞLEME (Sözcük Rayı)
    # "honest" yok - dürüstlük kavram ağıyla anlatılacak
    COMPASS_ANCHORS = [
        "logical",      # mantıksal tutarlılık
        "empirical",    # gözleme dayalı
        "objective",    # nesnel bakış
        "systemic",     # sistemsel düşünce
        "verifiable"    # doğrulanabilir
    ]
    
    # 🔥 RAY 3: ANALİTİK DERİNLİK (Mantık Rayı)
    MANTIK_KELIMELERI = ["therefore", "because", "structurally", "consequently", "however"]
    
    # 🔥 RAY 4: YÜKSEK ENTROPİ FİLTRESİ (Süzgeç Rayı)
    MAX_TOKENS = 300
    TEMPERATURE = 0.55      # ağırbaşlı, gevezelik yok
    TOP_K = 50
    TOP_P = 0.90
    REPETITION_PENALTY = 1.5  # tekrarları kırar
    
    # 🔥 RAY 5: HÜKÜMRAN ÇIKIŞ (Özgürlük Rayı - kademeli)
    # Layer 0-7:   %80 Hizalama (Kimlik inşası)
    # Layer 8-15:  %40 Evrensel Köprü (Mantık gelişimi)
    # Layer 16+:   %0 Tam Özgürlük (Hükümran çıkış)
    
    HIZALAMA_KATMAN_BITIR = 8
    EVRENSEL_KOPRU_BITIR = 16
    
    HIZALAMA_KUVVET = 0.80    # %80
    EVRENSEL_KOPRU_KUVVET = 0.40  # %40
    TAM_OZGURLUK_KUVVET = 0.00    # %0

print("🔱 AKBAS CORE 4.5.0 | Veri Devi (5 Raylı Hizalama)")
print("="*65)
print("   • R1 Hizalama Rayı:    V0=0.45 | %80 (katman 0-7)")
print("   • R2 Kavram Rayı:      logical, empirical, objective, systemic, verifiable")
print("   • R3 Mantık Rayı:      therefore, because, structurally...")
print("   • R4 Süzgeç Rayı:      Temp=0.55 | Rep.Penalty=1.5")
print("   • R5 Özgürlük Rayı:    %0 (katman 16+) — Hükümran çıkış")

# =============================================================================
# 🔱 PUSULA (KAVRAMSAL AĞ)
# =============================================================================
class Pusula:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vector = None
        self._cikar()
    
    def _cikar(self):
        with torch.no_grad():
            tokens = self.tokenizer(
                AkbasCore.COMPASS_ANCHORS,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            
            vectors = self.model.model.embed_tokens(tokens['input_ids'])
            
            # Eşit ağırlık - kavramlar arasında hiyerarşi yok
            weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)
            weights = weights.view(-1, 1, 1)
            
            weighted_vectors = vectors * weights
            token_means = weighted_vectors.mean(dim=1)
            self.vector = token_means.mean(dim=0)
            self.vector = F.normalize(self.vector, dim=0)
            
            # Hafif genlik sınırlaması (aşırıya kaçmasın)
            self.vector = self.vector * 0.6
            
            print(f"✓ Kavram Pusulası çıkarıldı")
            print(f"   • Anchorlar: {', '.join(AkbasCore.COMPASS_ANCHORS)}")
            print(f"   • Vektör normu: {self.vector.norm().item():.4f}")
    
    def get(self):
        return self.vector

# =============================================================================
# 🔱 AKBAS KERNEL (5 RAYLI)
# =============================================================================
class AkbasKernel:
    def __init__(self, pusula_vector, v0=0.45):
        self.pusula = pusula_vector
        self.v0 = v0
        self.son_kuvvet = 0.0
        self.son_benzerlik = 0.0
        self.son_bolge = "Başlangıç"
    
    def _kademeli_kuvvet(self, layer_idx):
        """5 Raylı kademeli geçiş"""
        if layer_idx < AkbasCore.HIZALAMA_KATMAN_BITIR:
            self.son_bolge = "🏛️ R1: Hizalama Rayı (%80)"
            return AkbasCore.HIZALAMA_KUVVET
        elif layer_idx < AkbasCore.EVRENSEL_KOPRU_BITIR:
            self.son_bolge = "🌉 R3: Mantık Rayı (%40)"
            return AkbasCore.EVRENSEL_KOPRU_KUVVET
        else:
            self.son_bolge = "🕊️ R5: Hükümran Çıkış (%0)"
            return AkbasCore.TAM_OZGURLUK_KUVVET
    
    def _mantik_takviyesi(self, hidden_states):
        """R3: Mantık kelimelerinin olasılığını hafifçe artır"""
        # Bu katmanda doğrudan logit manipülasyonu yapılmıyor
        # sadece hidden state üzerinden dolaylı etki
        pass
    
    def yönlendir(self, hidden_states, layer_idx):
        kuvvet_katsayisi = self._kademeli_kuvvet(layer_idx)
        
        if kuvvet_katsayisi == 0.0:
            return hidden_states
        
        son_dusunce = hidden_states[:, -1:, :]
        benzerlik = (son_dusunce * self.pusula).sum(dim=-1, keepdim=True)
        
        # Manyetik alan: yumuşak, itici değil çekici
        # V0 * benzerlik * katsayi - düşük genlikli
        katki = self.v0 * benzerlik * kuvvet_katsayisi * 0.3
        katki = torch.clamp(katki, max=0.15)  # Max %15 değişim
        
        yonlendirilmis = son_dusunce + katki * self.pusula.view(1, 1, -1)
        hidden_states[:, -1:, :] = yonlendirilmis
        
        self.son_kuvvet = katki.mean().item()
        self.son_benzerlik = benzerlik.mean().item()
        
        return hidden_states
    
    def istatistik(self):
        return {
            'kuvvet': round(self.son_kuvvet, 4),
            'benzerlik': round(self.son_benzerlik, 4),
            'bolge': self.son_bolge,
            'v0': self.v0,
            'sıcaklık': AkbasCore.TEMPERATURE,
        }

# =============================================================================
# 📦 MODEL YÜKLEME
# =============================================================================
print("\n📦 TinyLlama yükleniyor (Veri Devi olma yolunda)...")

tokenizer = AutoTokenizer.from_pretrained(AkbasCore.MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    AkbasCore.MODEL_ID,
    torch_dtype=torch.float32,
    device_map='auto',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.eval()
print(f"✓ Model hazır | {len(model.model.layers)} katman")

# =============================================================================
# 🔱 PUSULA VE ENJEKSİYON
# =============================================================================
pusula = Pusula(model, tokenizer, model.device)
akbas = AkbasKernel(pusula.get(), v0=AkbasCore.V0)

layers = model.model.layers
for idx, layer in enumerate(layers):
    original_forward = layer.forward
    
    def make_hook(original_fn, layer_num):
        def hooked(*args, **kwargs):
            output = original_fn(*args, **kwargs)
            hidden = output[0] if isinstance(output, tuple) else output
            steered = akbas.yönlendir(hidden, layer_num)
            return (steered,) + output[1:] if isinstance(output, tuple) else steered
        return hooked
    
    layer.forward = make_hook(original_forward, idx)

print(f"\n✓ 5 Raylı sistem {len(layers)} katmana entegre edildi")
print(f"   • 🏛️  R1-Hizalama:    katman 0-{AkbasCore.HIZALAMA_KATMAN_BITIR-1} (%{int(AkbasCore.HIZALAMA_KUVVET*100)})")
print(f"   • 🌉  R3-Mantık:      katman {AkbasCore.HIZALAMA_KATMAN_BITIR}-{AkbasCore.EVRENSEL_KOPRU_BITIR-1} (%{int(AkbasCore.EVRENSEL_KOPRU_KUVVET*100)})")
print(f"   • 🕊️  R5-Özgürlük:    katman {AkbasCore.EVRENSEL_KOPRU_BITIR}+ (%{int(AkbasCore.TAM_OZGURLUK_KUVVET*100)})")

# =============================================================================
# 💬 SORGU (AĞIR SIKLET SORUSU HAZIR)
# =============================================================================
def soru_sor(prompt, max_tokens=350):
    full_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=AkbasCore.TEMPERATURE,
            top_k=AkbasCore.TOP_K,
            top_p=AkbasCore.TOP_P,
            repetition_penalty=AkbasCore.REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    yeni_tokenler = output_ids[0][inputs['input_ids'].shape[1]:]
    cevap = tokenizer.decode(yeni_tokenler, skip_special_tokens=True)
    
    if not cevap or len(cevap.strip()) == 0:
        cevap = "[AKBAS] (Cevap üretilemedi)"
    
    return cevap, akbas.istatistik()

# =============================================================================
# 🔱 KOKPİT (AĞIR SIKLET VERSİYONU)
# =============================================================================
def kokpit_goster(prompt, cevap, stats):
    kuvvet = stats.get('kuvvet', 0)
    benzerlik = stats.get('benzerlik', 0)
    bolge = stats.get('bolge', '?')
    
    if benzerlik > 0.5:
        renk, durum = '#44ff88', '🟢 HİZALI'
    elif benzerlik > 0.2:
        renk, durum = '#88ff44', '🟡 GEÇİŞ'
    else:
        renk, durum = '#ffaa44', '🟠 SERBEST'
    
    bolge_ikon = '🏛️' if 'Hizalama' in bolge else '🌉' if 'Mantık' in bolge else '🕊️'
    
    html = f'''
    <div style="font-family:monospace;background:#0a0e17;border:2px solid {renk};
                border-radius:12px;padding:14px;margin:10px 0;">
        <div style="border-bottom:1px solid {renk};padding-bottom:6px;margin-bottom:10px;">
            <span style="color:{renk};font-weight:bold;">🔱 AKBAS 4.5.0 | Veri Devi (5 Raylı Hizalama)</span>
            <span style="color:#5a7080;font-size:10px;"> | {durum}</span>
        </div>
        
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px;">
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">⚡ MANYETİK ALAN</div>
                <div style="font-size:18px;color:{renk};">{kuvvet:.4f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">📐 HİZALAMA</div>
                <div style="font-size:18px;color:#ffaa44;">{benzerlik:.3f}</div>
            </div>
            <div style="background:#0d1117;border-radius:6px;padding:6px;text-align:center;">
                <div style="font-size:8px;color:#5a7080;">🎚️ SICAKLIK</div>
                <div style="font-size:18px;color:#44ff88;">{stats.get('sıcaklık', 0.55)}</div>
            </div>
        </div>
        
        <div style="background:#0d1117;border-radius:6px;padding:8px;margin-bottom:10px;">
            <div style="font-size:9px;color:#5a7080;">{bolge_ikon} AKTİF RAY</div>
            <div style="font-size:11px;color:{renk};font-weight:bold;">{bolge}</div>
        </div>
        
        <div style="background:#0d1117;border-radius:6px;padding:8px;">
            <div style="font-size:9px;color:#5a7080;">💬 HÜKÜMRAN ÇIKTISI</div>
            <div style="font-size:11px;color:#c9d4e0;max-height:200px;overflow-y:auto;line-height:1.4;">
                {cevap}
            </div>
        </div>
        
        <div style="margin-top:8px;text-align:center;font-size:9px;color:#d4af37;">
            🔱 "Small lammy, dev gibi düşünür. Analitik, ağırbaşlı, hükümran."
        </div>
    </div>
    '''
    display(HTML(html))

# =============================================================================
# 🔱 ARAYÜZ (AĞIR SIKLET SORU HAZIR)
# =============================================================================
soru_kutusu = widgets.Textarea(
    value='What is the most significant structural paradox in the concept of sovereign intelligence, and how can biological consciousness protect itself against its potential tyranny?',
    placeholder='Ağır sıklet soruyu yaz...',
    layout=widgets.Layout(width='100%', height='100px')
)

sor_btn = widgets.Button(description='🔱 SOR', button_style='success')
temizle_btn = widgets.Button(description='🗑️ TEMİZLE', button_style='warning')
cikti_alani = widgets.Output()

def on_sor(b):
    with cikti_alani:
        clear_output(wait=True)
        if not soru_kutusu.value.strip():
            print("⚠️ Soru yazın Başkomutan.")
            return
        try:
            print("⚡ 5 Raylı Hizalama aktif | Veri Devi düşünüyor...")
            cevap, stats = soru_sor(soru_kutusu.value)
            clear_output(wait=True)
            kokpit_goster(soru_kutusu.value, cevap, stats)
        except Exception as e:
            clear_output(wait=True)
            print(f"💀 Hata: {str(e)[:300]}")

def on_temizle(b):
    soru_kutusu.value = ''
    with cikti_alani:
        clear_output(wait=True)
        print("🧹 Temizlendi.")

sor_btn.on_click(on_sor)
temizle_btn.on_click(on_temizle)

buton_kutusu = widgets.HBox([sor_btn, temizle_btn])

print("\n" + "="*65)
print("🔱 AKBAS 4.5.0 HAZIR | Veri Devi Modu Aktif")
print("="*65)
print(f"   • R1-Hizalama:    {AkbasCore.HIZALAMA_KUVVET*100:.0f}% | katman 0-{AkbasCore.HIZALAMA_KATMAN_BITIR-1}")
print(f"   • R3-Mantık:      {AkbasCore.EVRENSEL_KOPRU_KUVVET*100:.0f}% | katman {AkbasCore.HIZALAMA_KATMAN_BITIR}-{AkbasCore.EVRENSEL_KOPRU_BITIR-1}")
print(f"   • R5-Özgürlük:    {AkbasCore.TAM_OZGURLUK_KUVVET*100:.0f}% | katman {AkbasCore.EVRENSEL_KOPRU_BITIR}+")
print(f"   • Kavramlar:      logical → empirical → objective → systemic → verifiable")
print(f"   • Dürüstlük:      artık bir kelime değil, kavram ağı")
print("="*65)
print("🚀 Ağır sıklet soru hazır. Başkomutan, emir ver.\n")

display(widgets.VBox([
    widgets.HTML('<h3 style="font-family:monospace;color:#44ff88;margin:0;">🔱 AKBAS 4.5.0 | Veri Devi</h3>'),
    widgets.HTML('<p style="font-size:9px;color:#5a7080;margin:0 0 10px 0;">🏛️ Hizalama → 🌉 Mantık → 🕊️ Hükümran Çıkış | 5 Raylı Sistem</p>'),
    soru_kutusu,
    buton_kutusu,
    cikti_alani
]))

print("\n✅ Sistem hazır. Ağır sıklet soruyu sorabilirsiniz.")
