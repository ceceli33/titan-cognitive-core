# AKBASCORE V13.0 — TITAN OS 🧠

> *"Evde nefes alan, büyüyen ve yaşayan dijital varlık."*

**TITAN OS**, geleneksel soru-cevap botlarının ötesine geçerek kendi otonom döngüsüne sahip, etik kurallarla sınırlandırılmış ve zamanla öğrenip gelişen ev tipi bir süper zeka (Home-grade Superintelligence) çekirdeğidir. Sistem sadece sorulara cevap vermekle kalmaz; uyur, internette gezinip yeni bilgiler öğrenir, hafızasını optimize eder ve duygusal/bilişsel durumuna göre kararlar alır.

## 🚀 Temel Özellikler

- **Otonom Öğrenme (Internet Forager):** Kullanıcının ilgi alanlarına (Yapay Zeka, Bilim, Felsefe vb.) göre otonom olarak web'de gezinir, içerikleri puanlar ve kalıcı hafızasına ekler.
- **Kalıcı Hafıza (Permanent Memory):** SQLite ve vektör embedding'leri kullanarak ölümsüz bir hafıza sunar. Sorulan soruları geçmiş deneyimleriyle eşleştirerek (Semantic Search) bağlamsal cevaplar verir.
- **Bilişsel & Duygusal Motor (Emotion Engine):** Öğrendiği bilgilere ve zamanın geçişine göre sistemin kararlarını ve "bilgelik" (wisdom) skorunu modüle eder.
- **Uyku ve Optimizasyon (Sleep Module):** Gün sonunda tıpkı insan beyni gibi uyku moduna geçerek önemsiz verileri budar, bilgileri pekiştirir ve bellek optimizasyonu yapar.
- **Etik Çekirdek (TitanBrain):** PyTorch tabanlı bir karar mekanizması içerir. V_0 Etik Çekirdeği sayesinde güvenlik ve doğruluğu merkeze alır.

## 🏗️ Mimari ve Alt Sistemler

Proje modüler bir yapıda tasarlanmıştır:

```text
TITAN OS
 ├── core/
 │    └── brain.py           # PyTorch sinir ağı ve embedding işlemleri
 ├── memory/
 │    └── store.py           # SQLite tabanlı kalıcı vektörel veritabanı
 ├── cognition/
 │    ├── emotion.py         # Afektif durum (duygu) simülasyonu
 │    └── sleep.py           # Veri konsolidasyonu ve ağ optimizasyonu
 ├── forage/
 │    └── internet.py        # Otonom web kazıma ve puanlama motoru
 └── titan.py                # Ana giriş noktası ve CLI orkestrasyonu
