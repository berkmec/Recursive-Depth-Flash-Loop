# Recursive Depth Flash Loop (`RDFL`)

**Ağırlık-Paylaşımlı Özyinelemeli Derinlik, FP32 Çalışan Bellek Yuvaları, Kopuk Spekülatif Başlıklar ve Doğrulayıcı-Kapılı Öz-Gelişim İçeren Bayt Düzeyi Sabit-Bellekli Hibrit Mimari**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](pyproject.toml)
[![Bağımsız Denetim](https://img.shields.io/badge/Denetim-KABUL_(1.073%2F1.073_hücre)-1D6B5E.svg)](results/audited_summary.json)

**[English](README.md)** · **[Türkçe](README.tr.md)** · **[Mimari Belirtim](docs/ARCHITECTURE.md)** · **[Denetlenmiş Özet JSON](results/audited_summary.json)**

---

## Özet

Standart otoregresif Transformer mimarilerinde anahtar–değer (KV) önbelleği dizi uzunluğuyla doğrusal $O(T)$ büyür; pencereli dikkat ise belleği sınırlamak uğruna pencere ufku $M$ ötesindeki durumu kaybeder. Saf doğrusal özyinelemeli modellerde ise seçici üzerine yazma (`MQAR`), negatif özdeğer gerektiren durum geçişleri (`Parity`, permütasyon grupları) ve çıpasız derinlik tekrarları kararsızlaşır.

**Recursive Depth Flash Loop (`RDFL`)**, tokenizer kullanmayan bayt düzeyi ($x_t \in \{0,\dots,255\}$) bir hibrit dizi mimarisidir; nedensel bir giriş gövdesini sabit boyutlu $O(1)$ durumlar üzerinde çalışan ağırlık-paylaşımlı bir özyinelemeli iyileştirme çekirdeğiyle birleştirir:

1. **Giriş-Çıpalı Özyinelemeli Derinlik:** Paylaşımlı özyinelemeli blok, her adımda normalize giriş çıpası ve adım gömmesiyle ($h_t^{(r-1)} + \alpha_r e_t + e_r$) $R$ kez yinelenir; çıpasız tekrarlarda görülen optimizasyon bozulmasını ortadan kaldırır.
2. **Householder Yazma Kapılı Parça-Paralel `Gated DeltaNet`:** Nedensel kısa konvolüsyon ($k=4$), çok-ölçekli yavaş sönümleme ($\alpha_0 \in [0{,}953, 0{,}999]$) ve $\beta_t = 2\sigma(\cdot) \in (0, 2)$ yazma kapısını birleştirir; dizi uzunluğundan bağımsız $S_t^{(r)} \in \mathbb{R}^{H \times d_h \times d_h}$ matris durumu taşır (`FP64`'te parça-paralel `forward == step` farkı `4,44e-16`).
3. **`BDH` Seyrek Etkileşim Kapısı ve Parça-Sınırı `FP32 SlotCore`:** İkili-seyrek ReLU etkileşimini parça sınırlarında kesin `float32` çalışan $S=4$ yuva belleğiyle eşleştirerek karma-hassasiyet taşmalarını tamamen giderir (`0` NaN).
4. **Sıfır-Başlangıçlı Kalıntı Uyarlama ve Yardımcı Başlıklar:** $W_{\uparrow}|_0 = 0$ çıkışı mevcut önceden eğitilmiş gövdelere adım-0'da `0,0` farkla bağlanır; kopuk erken çıkış (`K3`), çok-baytlı ileri bakış (`MTP-4`), sıfır-parametreli kalibre belirsizlik (`MSP + T*`), deterministik araç yürütme (`ToolRunner`) ve doğrulayıcı-kapılı sınırlı öz-gelişim (`AZR` RSI) ile bütünleşir.

---

## 1. Mimari

![RDFL Akademik Mimari Diyagramı](docs/diagrams/rdfl_architecture.svg)

*Şekil 1: `RDFL` hesaplama grafiği (aşağıdan yukarıya). Mavi yol nedensel giriş gövdesini ve dar çıpa projeksiyonunu; kehribar yol ağırlık-paylaşımlı özyinelemeli derinlik döngüsünü ve parça-sınırı `FP32 SlotCore` geri beslemesini; camgöbeği bloklar sabit $O(1)$ `Gated DeltaNet` matris durumlarını ve kopuk yardımcı tahmin başlıklarını (`K3`, `MTP-4`) gösterir. Tüm denklemler [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) içindedir.*

---

## 2. Denetlenmiş `1,20M` Kıyaslamalar ve Öz-Yetenekler

![1.20M Kıyaslamalar ve Bellek Ölçeklenmesi](docs/diagrams/benchmarks_1_2m.svg)

![Denetlenmiş Öz-Yetenekler ve 46x Ölçek Geçişi](docs/diagrams/self_capabilities_v2.svg)

- **Eş-Parametre `1,20M` (`204` koşu, `411/411` hücre doğrulandı):** `RDFL`, `Parity-64`, `S5-Perm32` ve `MQAR Overwrite` görevlerinin üçünde de **`%100,00`** (`RLT-0/1/2`: `%20,9–22,4`), `Multi-Hop ICL`'de **`%22,72`** (`2,57M` eş-FLOP `GDN`'nin `5,7×` üstünde) ve bayt dil modellemede **`2,2896 bpb`** (`RLT-0`'a göre `-0,1123 bpb`) elde eder. `T = 32.768` çıkarım durumu `576 MiB` tam KV'ye karşı sabit **`1,71 MiB`** (`337×` küçük).
- **Öz-Yetenekler V2 (`662/662` hücre doğrulandı, ayrık tohumlar `42/137/2026`, `0` NaN):** `FP32 SlotCore` 3-atlamalı graf doğruluğunu `%68,09 → %96,89` (`37,2×` düşük std) taşır; kopuk `K3` sıfır bayt farkla `1,48×–1,54×` FLOP tasarrufu verir; `MSP + T*` `AUROC 0,84–0,90` ile `%40` çekinmede `+20` puan doğruluk kazandırır; istemden öz-düzeltme (`ÖY-5`) simetrik testte `%49` bozma nedeniyle reddedilmiştir; hakemsiz kör RSI `7,52 bpb`'ye çökerken doğrulayıcı-kapılı `AZR` RSI `0` geri almayla en iyi `LM = 2,9301 bpb` ve en yüksek `IPT = %80,45` değerine ulaşmıştır.
- **`55M` Ölçek Geçişi (`Diogenes-RDFL-55M`, canlı devam eden koşu):** `1,20M`'de `+0,0323 bpb` vergi ödeyen `MTP-4`, açık `55M` ağırlıklarına yayıldığında kontrollü A/B deneyinde vergiyi **`32×`** eriterek `+0,0010 bpb` LM (`-0,0005 bpb` SFT) düzeyine düşürmüş, `7.000.` adımda doğal metinde **`%51,6` `t+2`** isabeti ve `T = 2048` uzun bağlamda **`-0,0142 bpb`** iyileşme sağlamıştır.

---

## 3. Yöntemsel Kısıtlar ve Tekrarlanabilirlik

- Mevcut sonuçlar saf PyTorch parça-paralel WY + SDPA ile üretilmiştir; sıfırdan büyük ölçekli (`200M+`) eğitim için `GatedDeltaNet` WY yinelemesinin birleşik Triton / Flash Linear Attention çekirdeğine taşınması gerekir.
- `1,20M` RSI deneylerinde uzak dağılım-dışı (`OOD-Uzak`, `L = 7, 8`) genelleme tüm kollarda `%0,0` kalmıştır; `55M` ölçeğindeki davranış devam eden koşuda ölçülmektedir.

```bash
git clone https://github.com/berkmec/Recursive-Depth-Flash-Loop.git
cd Recursive-Depth-Flash-Loop
pip install -e ".[dev]"
python -m unittest discover -s tests -v
python scripts/build_diagrams.py
```
