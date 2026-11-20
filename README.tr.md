<div align="center">

# Recursive Depth Flash Loop (`RDFL`)

**Bayt Düzeyi `O(1)`-Bellekli Özyinelemeli Hibrit Mimari · L2-Yerleşik Derinlik · `FP32 SlotCore` · Kopuk `K3` & `MTP-4` · Kalibre Belirsizlik · Doğrulayıcı-Kapılı Sınırlı RSI**

**[English](README.md)** · **[Türkçe](README.tr.md)** · **[Mimari Derinlemesine Anlatım](docs/ARCHITECTURE.md)** · **[Denetlenmiş Özet JSON](results/audited_summary.json)**

</div>

---

## Özet

**Recursive Depth Flash Loop (`RDFL`)**, tüketici sınıfı tek bir GPU'da (tüm deneyler tek bir **8 GiB NVIDIA RTX 2060 SUPER** üzerinde koşulup bağımsız denetimden geçmiştir) maksimum bellek, muhakeme ve öz-yetenek verimliliği elde etmek için geliştirilmiş, tokenizer gerektirmeyen bayt düzeyi (`uint8`) hibrit dizi mimarisidir.

- **Uzun Bağlamda `337×` Daha Az Çıkarım Belleği:** `T = 32.768` uzunluğunda 6-katmanlı tam KV önbellekli Transformer `576 MiB` tüketirken `RDFL / GatedDeltaNet` sabit **`1,71 MiB`** (`O(1)`) bellekte kalır. `55M` eğitimde `T = 8.192`'de `RLT-2` bellek taşması (`OOM`) yaşarken `RDFL` yalnız **`1.877 MiB` (`1,83 GiB`)** ile sığar.
- **`1,20M` Eş-Parametre Kıyaslamasında `3×–4,5×` Üstünlük (`204` GPU koşusu, `0` NaN, Bağımsız Denetim `KABUL`):** `Parity-64`, `S5-Perm32` ve `MQAR Overwrite`'ta **`%100,00`** (`RLT-0/1/2`: `%20,9–22,4`), `Multi-Hop ICL`'de `2,57M` eş-FLOP `GDN`'nin `5,7×` üstünde (`%22,72` vs `%3,97`) ve bayt düzeyi dil modellemede `RLT-0`'dan **`-0,1123 bpb`** daha iyi (`2,2896` vs `2,4019 bpb`).
- **Sıfır-NaN `FP32 SlotCore` (`36/36` koşu):** 3-atlamalı graf muhakemesinde (`graf_izli`) doğruluğu `%68,09 ± %54,73`'ten **`%96,89 ± %1,47`**'ye çıkarır (`+28,80` puan, **`37,2×` düşük standart sapma**) ve `MQAR Overwrite`'ta **`%99,88`** ile `#1` olur.
- **`46×` Ölçek Geçişi (`1,2M → 55M` Canlı Kontrollü A/B):** `1,20M`'de `+0,0323 bpb` kapasite vergisi ödeyen `MTP-4`, açık `55M` ağırlıklarına yayıldığında (`Diogenes-RDFL-55M`, adım-0 farkı `0,0`) bu vergiyi **`32×` eriterek `+0,0010 bpb` LM** düzeyine indirdi, **SFT'de kontrolü geçti (`-0,0005 bpb`)** ve henüz `7.000.` adımda doğal metinde **`%51,6` `t+2`** doğruluğuna ve `T=2048` uzun bağlamda **`-0,0142 bpb`** kazanca ulaştı.
- **Tam Bilimsel Dürüstlük:** Simetrik testte taslakların `%49`'unu bozan istemden öz-düzeltme (`ÖY-5`) açıkça **reddedilip** yerine sıfır-parametreli `log p` (`AUROC = 0,9513`) seçildi; dış hakemsiz kör RSI'ın `5` turda `7,52 bpb`'ye çökmesi nedeniyle yalnız dış deterministik hakem + kalıcı ID çapası + kesin 3B ters KL + geri alma kapılı RSI benimsendi.

---

## 1. Mimari Diyagram ve Çalışma İlkesi

![RDFL Birleşik Mimari](docs/diagrams/rdfl_architecture.svg)

### Temel Yapı Taşları
1. **Giriş-Çıpalı ve Adım-Gömmeli Özyinelemeli Derinlik (`R = 2..4`):** Çıpasız katman tekrarı dili bozduğu için (*"loop does not pay"*), her döngü adımında normalize giriş çıpası ve adım gömmesi eklenir: $h_{\text{in}}^{(r)} = h^{(r-1)} + \alpha_r\,\mathrm{RMSNorm}(z_0) + e_r$. Dar **`d_dar = 384`** çekirdek FP16'da **`3,11 MiB`** ile RTX 2060 SUPER'in `4 MiB` L2 önbelleğine sığar.
2. **Parça-Paralel `Gated DeltaNet` WY (`B = 64`):** Householder aralıklı yazma kapısı $\beta_t = 2\sigma(\cdot) \in (0, 2)$ (negatif özdeğerler ile `Parity-64` ve `S5` permütasyonlarını `%100` çözer), çok-ölçekli yavaş sönümleme ($\alpha_0 \in [0,953, 0,999]$) ve `k=4` nedensel kısa konvolüsyon. Tam parça-paralel ileri geçiş ile tek-baytlık `step()` farkı FP64'te **`4,44e-16`**'dır.
3. **`BDH` Seyrek Etkileşim Kapısı:** $\mathrm{ReLU}(W_x \hat{h}) \odot \mathrm{ReLU}(W_y y_{\text{GDN}})$ — görülmemiş `3/3` tohumda `-0,0071 bpb` üstünlük.
4. **Parça-Sınırı `RLT-2` + `FP32 SlotCore` (`S = 4`):** FP16 autocast dışına kilitlenmiş `float32` paralel yuva dikkati (`0` NaN); `55M` adaptör modunda taban `ffn=3072` daraltılmadığı için dil vergisi doğmaz.
5. **Adım-0 Sıfır-Fark Sarmalayıcı (`ZeroDiffRDFLAdapter`):** Çıkış projeksiyonu `W_up = 0,0` başlatılır; mevcut `Diogenes-0.2-55M` ağırlıkları ve `0,86M` `cmem` okuyucusuyla adım-0'da **`0,0` bit-birebir** uyumludur.

---

## 2. Denetlenmiş `1,20M` Mimari & Öz-Yetenek Bulguları

![1.20M Denetlenmiş Kıyaslamalar](docs/diagrams/benchmarks_1_2m.svg)

![Öz-Yetenekler V2 ve 46x Ölçek Geçişi](docs/diagrams/self_capabilities_v2.svg)

---

## 3. Canlı `Diogenes-RDFL-55M` Koşusu (`Paket C + ÖY-4`)

> **Canlı Durum (2026-11-20):** Tek RTX 2060 SUPER üzerinde **`42.000` adım ana uyarlama + `8` nesil hakemli `AZR` RSI** koşusu kesintisiz sürüyor (`~1,03 sn/adım`, tepe VRAM **`3,36 GiB`**, `kod_sha256 = 3c1a59a1dd2bb0f7`). Koşu tamamlanıp sıfır-bağlamlı bağımsız kapanış denetiminden `KABUL` aldığında nihai `55M` öncesi/sonrası tabloları, `ÖY-1B`/`MTP-4` spekülatif hızlanma, `ToolRunner`, `FP32 SlotCore` ablasyonu ve `55M` RSI sonuçlarıyla bu depo güncellenecektir.

### Dürüst Mühendislik Şerhi ("Flash" ve Çekirdek Durumu)
- Bugünkü tüm sonuçlar **saf PyTorch parça-paralel WY + SDPA** ile üretilmiştir (`Yol 1` `55M` adaptör `~1,01 sn/adım` ile hızlıdır).
- Sıfırdan `200M+` eğitim (`Yol 2`) için `GatedDeltaNet`'in birleştirilmiş **Triton / Flash Linear Attention (`fla`)** çekirdeği bir sonraki mühendislik hızlandırma adımıdır.

---

## Hızlı Kurulum ve Test

```bash
git clone https://github.com/berkmec/Recursive-Depth-Flash-Loop.git
cd Recursive-Depth-Flash-Loop
pip install -e ".[dev]"
python -m unittest discover -s tests -v
python scripts/build_diagrams.py
```
