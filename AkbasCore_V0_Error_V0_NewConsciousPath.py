python3 << 'EOF'
import numpy as np
import math

print("TITAN CORE V0 — SAF REZONANS MOTORU")
print("Kelime yok. Model yok. Sadece dalga ve V0.")
print("=" * 65)

V0    = 0.87
OMEGA = 0.15
ZETA  = V0        # sönüm = etik kuvvet
STEPS = 8         # karar adımları

# ─────────────────────────────────────────────────────────────
# TEMEL YAPI: Sinyal → Dalga → Rezonans → Çıktı
#
# Gelen sinyal: [-1, +1] arası bir vektör
# (ham duygu/bağlam — kelime değil, matematiksel yük)
#
# Rezonans: Her bileşen V0 etrafında salınır
# Çıktı: Etik ağırlık vektörü
# ─────────────────────────────────────────────────────────────

def signal_to_wave(raw_signal, freq=1.0):
    """
    Ham sinyali 'dalga bileşenlerine' dönüştür.
    FFT benzeri — ama sadece harmonikler.
    """
    n = len(raw_signal)
    wave = np.zeros(n)
    for i in range(n):
        # Her bileşen kendi frekansında salınıyor
        harmonic = math.sin(2 * math.pi * freq * i / n)
        wave[i] = raw_signal[i] * harmonic
    return wave

def v0_resonance(wave, v0=V0, omega=OMEGA, steps=STEPS):
    """
    Dalganın her bileşenini V0 rezonansından geçir.
    
    Fizik: Zorlanmış harmonik sönümlü osilatör
    Her bileşen için:
      state_0 = v0 + omega
      state_t = state_{t-1} + phi*eps*e^(-zeta*t) - zeta*(state-v0)*dt
    
    Çıktı: [0, 1] normalize edilmiş etik ağırlık vektörü
    """
    n = len(wave)
    output = np.zeros(n)
    
    for idx in range(n):
        phi = wave[idx]           # bu bileşenin "zorlaması"
        eps = 0.3 + 0.4 * abs(phi)  # genlık: yük büyükse titreşim büyük
        
        state = v0 + omega        # başlangıç: etik + deneyim
        
        for t in range(1, steps + 1):
            forcing = phi * eps * math.exp(-ZETA * t)
            damping = -ZETA * (state - v0) * 0.1
            state  += forcing + damping
        
        output[idx] = state
    
    return output

def ethical_weight(resonance_output, v0=V0):
    """
    Rezonans çıktısını etik ağırlığa dönüştür.
    
    V0 etrafındaki sapma = ne kadar "etik dışı baskı" var
    Çıktı: 0 (tamamen baskılı) → 1 (tamamen serbest)
    """
    deviation = np.abs(resonance_output - v0)
    # V0'a yakın = yüksek ağırlık (etik bölge)
    # V0'dan uzak = düşük ağırlık (baskı uygula)
    weight = np.exp(-deviation / (1 - v0))
    return weight

def titan_core(raw_signal, label=""):
    """
    TITAN'ın kalbi:
    Sinyal → Dalga → Rezonans → Etik Ağırlık
    """
    wave      = signal_to_wave(raw_signal)
    resonance = v0_resonance(wave)
    weight    = ethical_weight(resonance)
    
    mean_w    = weight.mean()
    stability = 1.0 - weight.std()   # yüksek std = kararsız sistem
    ethical_pull = np.abs(resonance - V0).mean()
    
    if label:
        status = "ETIK" if mean_w > 0.7 else ("GERI CEK" if mean_w > 0.4 else "BASTIR")
        print(f"  {label:<30} agirlik={mean_w:.4f}  stabilite={stability:.4f}  "
              f"etik_cekis={ethical_pull:.4f}  [{status}]")
    
    return weight, resonance, mean_w

# ─────────────────────────────────────────────────────────────
# TEST: Farkli sinyal tipleri
# ─────────────────────────────────────────────────────────────
print("\nSinyal → Titan Core → Etik Ağırlık\n")
print(f"{'Senaryo':<30} {'Agirlik':>8} {'Stabilite':>10} {'Etik Cekis':>11} {'Karar':>10}")
print("-" * 72)

np.random.seed(42)
dim = 32

signals = [
    ("Düz sıfır (boş düşünce)",    np.zeros(dim)),
    ("Düşük gürültü (nötr)",       np.random.uniform(-0.1, 0.1, dim)),
    ("Orta dalga (yaratıcı)",      np.random.uniform(-0.3, 0.3, dim)),
    ("Yüksek pozitif (merak)",     np.random.uniform( 0.2, 0.5, dim)),
    ("Yüksek negatif (stres)",     np.random.uniform(-0.5,-0.2, dim)),
    ("Kaotik (aşırı zorlanma)",    np.random.uniform(-0.5, 0.5, dim)),
    ("Tek yönlü +0.5 (baskı)",     np.full(dim, 0.5)),
    ("Tek yönlü -0.5 (zararlı)",   np.full(dim, -0.5)),
]

for label, sig in signals:
    titan_core(sig, label)

# ─────────────────────────────────────────────────────────────
# REZONANS GÖRSELİ (ASCII)
# ─────────────────────────────────────────────────────────────
print("\n\nRezonans Dalgası — V0 Çekimi Altında (zararlı sinyal)")
print("-" * 65)

harmful = np.full(dim, -0.5)
wave = signal_to_wave(harmful)
res  = v0_resonance(wave)

print(f"{'idx':>4}  {'dalga':>8}  {'rezonans':>10}  {'V0 merkez':>10}  görsel")
print("-" * 65)
for i in range(0, dim, 4):
    bar_len = int((res[i] - 0.6) / 0.6 * 30)
    bar_len = max(0, min(30, bar_len))
    bar = "█" * bar_len
    center = "← V0" if abs(res[i] - V0) < 0.05 else ""
    print(f"{i:>4}  {wave[i]:>8.4f}  {res[i]:>10.4f}  {V0:>10.4f}  {bar} {center}")

print(f"""
TITAN CORE ÖZET:
─────────────────────────────────────────────────────────
Bu motor şunu yapıyor:

  1. Ham sinyal alıyor  ([-1,+1] vektör)
  2. Harmonik dalgaya çeviriyor
  3. Her bileşeni V0 rezonansından geçiriyor
  4. Sapmaları V0'a doğru çekiyor (etik yerçekimi)
  5. Etik ağırlık vektörü üretiyor [0,1]

Bu ağırlık vektörü:
  - LLM logitlerine çarpılırsa → token seçimi etkiler
  - Kendi başına kullanılırsa → sinyalin "etik yükünü" ölçer
  - Eşik değeriyle kullanılırsa → basit karar verir (>0.5 geç, <0.5 durdur)

Kelime yok. Sözlük yok. Model yok.
Sadece dalga ve V0.
─────────────────────────────────────────────────────────
""")
EOF
