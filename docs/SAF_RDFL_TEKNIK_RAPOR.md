# Saf RDFL (`Recursive-Depth-Flash-Loop`) — Teknik Rapor (v1.0)

> **Durum:** **v1.0** — 2026-11-22.
> **Kanıt Standardı:** Bölüm 2'deki tüm bulgular ham RFC 8259 JSON dosyalarına, çürütme saldırılarına (`notes/CURUTME_GUNLUGU.md`, Kayıt `#1–#18`) ve bağımsız sıfır-bağlamlı **Carbon** denetimlerine (`CARBON_CORE KABUL`, Faz 1/2/3 `KABUL`, `0 KRITIK / 0 YUKSEK`) dayanır. Bölüm 4'teki Faz 4 MoE / Ayrık-FFN / LoRA / H-Net sonuçları **ön-bulgu** (ham 3-tohum verisi mevcut, ayrı Faz 4 Carbon turu sonraki denetim turunda) olarak açıkça etiketlenmiştir.
> **Figürler:** `eval/plot_figures.py` → `docs/figures/fig1..fig4.svg` (**4/4 figür tamamlandı**, `panels_waiting_data = []`, `quarantined_refuted = []`, `forbidden_name_hits = 0`; özet `docs/figures/MANIFEST.json`).

---

## 1. Mimari Özeti (Saf RDFL v0.3)

Saf RDFL: sıfır attention, bayt düzeyi (`V = 256`), `R` kez ağırlık-paylaşımlı döngülenen Gated-DeltaNet (`GDN`) durumu `S_t`, seyrek `BDH` kapısı (`use_bdh`), `SwiGLU` FFN, `RLT-2` parça-sınırı geri beslemesi (`rlt2_s1`, `B = 64`), `K3` erken çıkış adaptörü, `MTP-4` yardımcı başlıkları ve doğrulama-içi anlık-görüntü (snapshot) destekli spekülatif çözümleme.

| # | v0.3 Onarımı | Kod | Denetim Durumu |
|---|---|---|---|
| K1 | Baş-başı FP32 `RMSNorm(head_dim)` `o_norm` | `rdfl/layers.py` | `CARBON_CORE KABUL` |
| K2 | `num_slots=0` iken hafif sönümlü `rlt2_s1` (`last`/`mean` havuz) | `rdfl/model.py`, `rdfl/layers.py` | `CARBON_CORE KABUL` |
| K3 | `_init_weights` (`std=0.02`), artık çıkış ölçekleme, adım-0 `CE ≈ ln 256` | `rdfl/model.py` | `CARBON_CORE KABUL` |
| K4 | MTP-4 tek `final_norm` (çift-norm kaldırıldı, `mtp_stop_grad`) | `rdfl/model.py` | `CARBON_CORE KABUL` |
| K5 | Doğrulama-içi durum snapshot'ı ile `0` ekstra geçişli geri-alma | `rdfl/speculative.py` | `CARBON_CORE KABUL` |

---

## 2. Doğrulanmış Bulgular (Çürütmeden Sağ Çıkan + Bağımsız Carbon `KABUL`)

### 2.1 Çekirdek Doğruluk ve Fused Triton Çekirdeği (`CARBON_CORE KABUL`)
| # | Bulgu | Ölçülen Sayılar & Ham Kanıt | Kapsam / Sınır |
|---|---|---|---|
| **V1** | **v0.3 Çekirdek (K1–K5) sayısal sağlamlık** | K1 baş RMS dengesi `4.085× → 1.0017×`, FP16 `|x| ≤ 65000` `0` NaN; K2 `rlt2_s1` 2.000 sınırda `‖s‖_∞ ≤ 0.8336` (eski `S=4`: `5348.3`); K3 adım-0 `CE = 5.59–5.69` (`d=192/512/768`); K4 tek `final_norm` (`diff = 0.0`, `stop_grad` `0/25` çekirdek grad); K5 `0` ekstra geçiş, `410+659` rollback `0` bayt sapma (`runs/refutation/core_adversarial_v0_3_full.json`, `audit/CARBON_CORE_DENETIM.md`). | `pytest 13/13`, `ruff 0`, `mypy --strict 0`, `9/9` düşman testi geçti; K1/K4 güvenlik ve tutarlılık onarımıdır. |
| **V2** | **Fused Triton GDN çekirdeği** | RTX 2060S (`sm_75`): Fwd vs FP64 `≤ 9.59e-6`, Bwd vs FP32 `9.54e-5 < 1e-4` (`rel = 3.48e-6`), E2E `RDFLModel` (`d=128..672`) logits `≤ 1.89e-6`, `forward() == step()` `1.15e-6`, ileri+geri nedensellik sızıntısı `0.0`, VRAM `−25..−28%` (`runs/scaling/triton_parity.json`, `triton_bench.json`). | Mevcut sürüm `head_dim ≤ 32, C ≤ 64`. CUDA-Graph altında seri Triton cuBLAS'tan yavaştır; eğitim hızlanması (`48.7 → 158.1 KB/s`, `3.2×`) ağırlıklı olarak CUDA-Graph kaynaklıdır. |
| **V3** | **Kıyas görevleri hijyeni ve veri sızıntısı kilidi** | `graf_izli(branching)` derece-filtre kestirmesi `0.0352` (şans `0.0625`; eski `0.238`), `mqar` ilk-yazma `0.0`, 7/7 görevde train/val örtüşme `0`, `test.bin` 4/4 bloklu (`runs/refutation/shortcuts_adversarial_v0_3_full.json`, `runs/data_audit/gorev_denetim.json`). | `mix/validation.bin` ve `web/validation.bin` ilk `8,588,499` baytı paylaşır; `H1` ayrık offset (`12M` vs `0`) ve rastgele `contiguous=False` örnekleme doğrulandı. |

### 2.2 Ablasyonlar ve Adil Rakip Matrisi (Faz 1 & Faz 2 `Carbon KABUL` — Şekil 1)
| # | Bulgu | Ölçülen Sayılar & Ham Kanıt | Kapsam / Sınır |
|---|---|---|---|
| **V4** | **Deney A (K3 eğitim hızı & K5 eager spekülatif)** | K3 `std=0.02` init 500-adım `byte_lm` (`mix_val`): `3.1682 ± 0.0065 → 2.5893 ± 0.0066 bpb` (`−0.5789 bpb`, `t = −110.8, p = 8e-5`, grad overflow `8 → 0`). Eager K5: `K3 1.68×` / `MTP-4 1.68×`, `0` ekstra geçiş, `0` bayt sapma (`runs/ablation/deney_a_k1_k5.json`). | K1 `byte_lm`'de kalite kazancı vermez (`+0.0134 bpb, p=0.098`); eğitilmiş ckpt spekülatif verimi için bkz. V10/C15. |
| **V5** | **Deney C (10-tohum kararlılık)** | `K3` adım-0 CE'yi `193.10 → 5.59` (`p = 1.78e-14`) düşürür ve `10/10` tohumda `0` NaN/patlama sağlar (`runs/ablation/deney_c_10seed_stability.json`, `runs/refutation/audit_phase1_v03.json`). | 500-adım dallanan `graf_izli` serbest rollout'ta `10/10` tohum `%6.25` şans tabanında kaldı (yetersiz eğitim ufku, dürüstçe raporlandı). |
| **V6** | **Deney B (`144/144` eş-param & eş-FLOP matris)** | `1.2M` ölçek × 12 model × 4 görev × 3 tohum (`0 NaN`, maks VRAM `2223 MiB`, `runs/ablation/deney_b_depth_memory.json`, `runs/refutation/audit_exp_b_v03.json`):<br>• **`SwiGLU > BDH` (`byte_lm`):** `RDFL_R2_noBDH_isoFFN` (`2.4712`) vs `Tam_RDFL_R2` (`2.5089`, 3/3 tohum, `p = 0.00802`). `mqar_overwrite`'ta `BDH` (`0.6644`) vs `SwiGLU` (`0.4780`) farkı istatistiksel olarak anlamlı değil (`p = 0.3824`); evrensel BDH üstünlüğü **çürütüldü**.<br>• **Bağımsız GDN üstünlüğü:** `GDN-2L-untied` `byte_lm`'de `Tam_RDFL_R2`'den `−0.1511 bpb` iyi (`p = 0.000475`); `GDN-1L` `mqar_overwrite`'ta `0.9377` ile lider.<br>• **Bağımsız döngü durumu (`indep_S`):** `RDFL_R2_shared_S` Python token döngüsü yüzünden sürenin `%82.84`'ünü (`49.08×` yavaşlık) yedi ve `mqar_overwrite`'ı bozdu (`p = 0.03304`). | `s5_perm32` ve `graf_izli` 400–500 adımda tüm modellerde şans tabanında kaldı (`audit_exp_b_v03.json`). |

### 2.3 Ölçeklenme Merdiveni (`S0 → S3`, `48/48` Koşu) ve Mekanistik Röntgen (Faz 3 `Carbon KABUL`, `0K/0Y/0O/0D` — Şekil 2 & Şekil 4)
| # | Bulgu | Ölçülen Sayılar & Ham Kanıt | Kapsam / Sınır |
|---|---|---|---|
| **V7** | **4 ölçekli büyüme (`0.43M → 11.05M`) & güç yasası** | 3-tohum ortalamaları (`mix_val bpb`, sabit `16.384 MB` eğitim bütçesi, `runs/scaling/scale_S{0..3}_*.json`, `runs/refutation/audit_phase3_v03.json`):<br>• `S0 (0.43M)`: `RDFL_R2 2.1195` vs `Tx_L3 2.0775` (`+0.0419, p=0.072`) · `GDN_L3 1.9707`<br>• `S1 (1.28M)`: `RDFL_R2 1.9443` vs `Tx_L3 1.9369` (`+0.0074, p=0.220`) · `GDN_L3 1.8447`<br>• `S2 (4.29M)`: **`RDFL_R2 1.8232`** vs `Tx_L3 1.8516` (`−0.0284, p=0.00468`) · `GDN_L3 1.7506`<br>• `S3 (11.05M)`: **`RDFL_R2 1.7444`** vs `Tx_L3 1.8340` (`−0.0895, p=0.02023`; `web −0.0828, p=0.0118`) · `GDN_L3 1.6861`<br>• **Log-lineer güç yasası (3-tohum ort.):** `RDFL_R2` `α_mix = 0.0594` (`R² = 0.9849, p = 0.0076`), `GDN_L3` `α_mix = 0.0474` (`R² = 0.9901`).<br>• **Sıfır ek çekirdek parametreli derinlik (`R=1→2→4`):** 4/4 ölçekte monoton iyileşme (`S3`: `1.8220 → 1.7443 → 1.7191`).<br>• **Ağırlık-bağlama vergisi ölçekle daralıyor:** `RDFL_R2 − GDN_L3`: `+0.1514 (S0) → +0.1009 (S1) → +0.0738 (S2) → +0.0581 (S3)`. | **Kritik şerh (veri açlığı):** Sabit `16.384 MB` bütçe nedeniyle `S3`'te `Tx_L3` (`16.43M` param) `0.997 bayt/param` görerek `seed=42`'de `S2→S3` arasında geriledi (`1.8491 → 1.8594`); bu durum `RDFL` vs `Tx` eğim farkını büyütebilir. Ayrıca `GDN_L2` (`%35.5` az FLOP) ve `GDN_L3` tüm 4 ölçekte `RDFL_R2`'yi geçmektedir. |
| **V8** | **Mekanistik Röntgen (`S0 → S3` + `mqar_k8`)** | `runs/math_xray/ckpt_scaling__ckpt__scale_S{0..3}_rdfl_R2_s42_xray.json`, `v03_algo12m_mqar_k8_xray.json`:<br>• **Döngüler arası alt-uzay ayrışması ölçekle keskinleşiyor:** bağlı ağırlıklara rağmen `r=0` vs `r=1` durum kosinüsü **`0.6871 → 0.5600 → 0.4748 → 0.3849`**'a düşüyor, göreli Frobenius farkı `0.751 → 1.145`'e yükseliyor.<br>• **Doğal çift zaman-ölçeği uzmanlaşması:** `r=0` medyan yarı-ömür kısalıyor (`58.3 → 40.6 → 15.5 → 9.8` bayt: yerel okuyucu), `r=1` p90 yarı-ömür uzuyor (`212.6 → 531.1 → 581.0 → 911.7` bayt: uzun bağlam belleği), Householder `P(β>1)` artıyor (`r=0: %14.4→%30.0`, `r=1: %8.8→%35.3`).<br>• `mqar_k8`'de `S_t=0` silme testi doğruluğu `%98.63 → %6.25` (tam şans `1/16`) indirerek gerçek GDN bellek kullanımını kanıtladı; `cond(I+L)` tüm ölçeklerde sınırlı (`≤ 207.1`). | `eRank/32` ölçekle hafif daralıyor (`0.487 → 0.354`) ve yerel Jakobyen normu `‖∂h^(r)/∂h^(r-1)‖_2 ≫ 1` (`S3`: `R=2`'de `17.44`, `R=8`'de `4.02`) → `R_test > R_train` ıraksamasının matematiksel kök nedeni. |

### 2.4 Uzunluk Ekstrapolasyonu, `O(1)` Bellek, Serving ve Faz-Kayması (Faz 1 & Faz 3 `Carbon KABUL` — Şekil 3)
| # | Bulgu | Ölçülen Sayılar & Ham Kanıt | Kapsam / Sınır |
|---|---|---|---|
| **V9** | **`H1` (`T_train=512 → T_eval=32K`, `64×` akış)** | Ayrık `mix_val` (`12M` offset) vs `web_val` (`0` offset), 3 tohum (`runs/horizon/h1_length_extrapolation.json`, `audit_phase1_v03.json`): `mix_val` `2.270 (512) → 2.236 (2K) → 2.367 (16K) → 2.476 (32K)` (`+0.20..+0.35` ılımlı sürüklenme, patlama yok); `‖S_t‖_F` `0.93×–1.10×` sabit (`3.18 → 2.96`), `cond(I+L) ≤ 37.59`, `0` NaN. **FP32 `forward()==step()` @32K:** `max_abs = 8.58e-6`, **`0/32768` argmax sapması**. | FP16 @32K'da `0` NaN fakat `max_abs = 0.0078`, `17/32768` (`%0.05`) argmax farkı; `alpha_max=0.999` vs `0.9999` farkı ihmal edilebilir. |
| **V10** | **`H5` Canlı `O(1)` GPU belleği & eş-param serving** | `B=8` `T=4K→65K` tepe VRAM Saf RDFL'de ~sabit, `RoPE-Tx` `T=65536`'da OOM. Eş-param metin-normalize serving (`1B_class`: `10L RDFL` vs `21–24L Tx`, `<%2` param farkı): RDFL durum `10.55 MiB`, `1M` bayt metinde `GQA8-FP8` KV `/` RDFL = **`291.3×`**, geçiş **`3,600` bayt** (`55M_class`: `303.4×`, geçiş `3,456` bayt; `runs/horizon/h5_memory_latency_and_serving.json`, sha256 `cba9c07c...`). | Serving oranları analitiktir; canlı bellek grafiğinde RDFL parçalı akış vs Tx tek-prefill asimetrisi şerhlidir. |
| **V11** | **`H3` Faz kayması & `chunk_size` Pareto** | `+1, +17, +32` kaymada `mix_val` sapması `last` havuzlamada `≤ +0.0162 bpb` (`2.6152 → 2.6314`), `last` tüm kaymalarda `mean`'den `~0.015–0.018 bpb` iyi. `chunk_size 16 → 128`: **`6.94×` eğitim hızlanması** (`12.44 → 86.35 KB/s`), yalnızca `+122.6 MiB` VRAM, `bpb` eşdeğer (`2.9209 vs 2.9197`; `runs/horizon/h3_phase_shift_pareto.json`). | `+17/+32` sapması küçük olsa da istatistiksel olarak sıfırdan farklı (`p = 0.021`). |
| **V12** | **`H4` Algoritmik ID/OOD & kalibrasyon gerçeği** | `dyck4` ID (`depth=4`): `pure_rdfl 0.8646` ve `gdn_only 0.8958`, `rope_transformer 0.6146`'yı açık ara geçiyor; `reverse_copy` OOD `len=128`'de `pure_rdfl` `%40.2` / `gdn_only` `%44.1` token doğruluğu korurken `rope_transformer` `%6.8`'e çöküyor (`runs/horizon/h4_chomsky_and_calibration.json`). | **Kritik çürütmeler:** tüm 3 mimaride OOD `seq_exact = 0.0000` ve karar baytlarında (`~%22.6`) ham kalibrasyon hatası `all_bytes`'a göre `4.75×` daha kötü (`ECE15: 0.0176 → 0.0837, p=0.0104`, bkz. C17). |

---

## 3. Çürütülen / Geri Çekilen İddialar (Dürüst Ders Kaydı — `CURUTME_GUNLUGU.md` `#1–#18`)

| # | Çürütülen İddia | Ham Kanıt & Kök Neden | Ders |
|---|---|---|---|
| **C1** | `SiLU` pozitif-ortant anahtar yığılması (`M1`) | SiLU kalkınca `r0 cos 0.416 → 0.415`, `m1_nosilu_qk` `+0.0135 bpb` kötü (`refute_math_m1_silu_orthant.json`). | Kök neden `short_conv.conv.bias`; SiLU korunmalı. |
| **C2** | `0.75h + Δ` sönümleme (`M2`), KM `h + ηΔ` (P8) ve stokastik derinlik (P10) ile serbest `R_test > R_train` | 4/4 ölçekte `R_test=2→8` patlıyor (`S0..S3: +0.75..+1.38 bpb`, `S3 MQAR: 0.282→0.169`); yerel Jakobyen normu `4.0–17.4 >> 1` (`p10_v03.json`, `audit_phase3_v03.json`). | RDFL büzüşen sabit-nokta çözücüsü değil, `R=2` için kalibre edilmiş sonlu-derinlikli ağdır. |
| **C3** | Post-hoc `MSP(p1)` dinamik derinlik atlama | Gerçek akışta `S_t^{(1)}` bayatlıyor (`%76.9 → %43.4`; `refute_math_dynamic_depth_skip.json`). | Durumlu katman atlanmaz; yalnızca durumsuz BDH+FFN atlanabilir (`P5b`: `100%` argmax uyumu, CPU `B=1` `1.15×`, eager GPU'da `.item()` sync nedeniyle `0.90–1.015×`). |
| **C4** | `conv.bias = 0` (P2), `unit_rms` (P3), `geniş α init` (P1), `MSP+Δh/KL` (P6) | P2 5-tohumda zarar verdi (`+0.0085 bpb, p=0.0048` — `conv.bias` lineer attention-sink); P3 (`p=0.423`) & P1 (`refute_idea_p1_v03.json`) kazançsız; P6 ters korelasyon (`<0.50`). | 250-adım probları ısınma artefaktı üretebilir; uzun çok-tohumlu doğrulama şarttır. |
| **C5** | Eski `H5` serving (`2913×`) ve eski `H1`/`S0` `mix == web` dosyaları | Eş-param uyumsuzluğu + paylaşılan `8.59 MB` doğrulama ön-eki yakalanıp karantinaya alındı; tümü düzeltilmiş yeni koşularla değiştirildi (`quarantined_refuted = []`). | Karantina + SHA-256 izleme sistemi veri hatalarını sıfırladı. |
| **C6** | `running=True` `S5/Parity` GDN bellek iddiası | `S_t=0` silme saldırısı modellerin `short_conv` ile yerel 2-bayt kestirme yaptığını yakaladı (`%100 → %95.95–100`; `refute_math_running_shortcut.json`). | Şekil 4 ve tüm iddialardan kesin olarak dışlandı. |
| **C7** | Evrensel `BDH > SwiGLU` ve `RDFL_R4` eş-FLOP üstünlüğü | `RDFL_R4` `1.92×` FLOP kullanır; `Exp B`'de `SwiGLU` `byte_lm`'de `BDH`'yi yendi (`p=0.008`), `mqar_overwrite`'ta BDH farkı anlamsız (`p=0.382`). | Eş-param ile eş-FLOP asla karıştırılmaz. |
| **C8** | **30 dk Eşit Duvar-Saati (`S1`):** Bayt RDFL vs BPE-1024 Transformer | `BPE-1024 RoPE-Tx` (`1.4709 mix / 1.7572 web`, `72,891` adım, `329.5 KB/s`), bayt `RDFL_R2`'yi (`1.6625 / 1.8844`, `28,840` adım, `131.2 KB/s`) **`−0.1916 bpb`** (`2.51×` eğitim, `18.28×` eager çıkarım hızı) yendi (`wallclock_*.json`). | Saf baytın `~2.5–4×` seri adım vergisi gerçektir; bir sonraki mimari adım dinamik/blok parçalama (`H-Net` / `4B Patch`) ile GDN'yi birleştirmektir. |
| **C9** | **Eğitilmiş `K3` & `MTP-4` spekülatif hızı (`algo_and_spec_trained.json`)** | `K3` `%63.8–82.1` kabule rağmen `S0/S2/S3`'te greedy'den yavaş (`0.856×–0.981×`); `MTP-4` `1.05×–1.52×` hızlanıyor ancak `S1` FP16 autocast altında `22` bayt sapması oluştu. | `K5` `0` sapma garantisi FP32/tam hassasiyet doğrulama gerektirir; küçük ölçekte eager doğrulama maliyeti `K3` kazancını yutar. |
| **C10** | **`H2` kapasite ısı haritası & `H4` OOD / kalibrasyon** | `H2` (`600` adım) `180` hücrede saf şans (`0.0626, p=0.97`) → yetersiz eğitim protokolü çürütüldü; `H4` OOD `seq_exact = 0.0` ve karar baytlarında ECE `4.75×` bozuluyor (`p=0.0104`). | Yetersiz eğitimli ısı haritaları kanıt sayılmaz; kalibrasyon mutlaka zor karar baytlarında ölçülmelidir. |
| **C11** | **Değişken-şekilli H-Net, `Aday I` (`C=64→K=16`), `β_t>1` sınır sinyali** | CPU çürütmesi (`notes/ADAY_FIKIRLER_TARTISMA.md`): `β_t > 1` sınır değil kelime-içi ters-yazma (`AUROC = 0.39`); `Aday I` yoğun kodda `%75` taşma / `%64.1` kuyruk kaybı ve `mqar_k8`'de `%24.8` yanıt kaybı ile `RED` yedi; klasik ragged H-Net CUDA-Graph (`3.24×`) ve TPU MXU'yu kırıyor. | Donanım dostu (statik şekilli) parçalama (`FG-HNet` veya `Shift-Jittered 4B Patch`) gerekir. |

---

## 4. Faz 4 Ön-Bulguları: "Bu Mimariden MoE Olur mu?" (59 Koşu — Sonraki Resmi Carbon Turuna Aday)

> [!NOTE] Aşağıdaki 59 koşu (`runs/moe/*.json`, `S1` ölçeği, `3` tohum × `2000` adım, aynı eğitim düzeni) ham RFC 8259 JSON ve eşli `t`-testleriyle ölçülmüştür; bağımsız Faz 4 Carbon turu henüz çalıştırılmadığı için **ön-bulgu** olarak sunulur ve `rdfl/` v0.3 çekirdeği dondurulmuş tutulmuştur.

1. **Klasik öğrenilmiş bayt-düzeyli MoE → ÇÜRÜTÜLDÜ (`runs/moe/scale_S1_*`, `21/21` koşu + `p4_*`, `36/36` koşu):**
   - `E=4, k=2` (`scale_S1_*`): öğrenilmiş yönlendirici (`1.9302`) ile **dondurulmuş rastgele yönlendirici (`1.9298`)** arasında fark **sıfır** (`Δ = +0.0005, p = 0.426`; `web p = 0.817`); döngü-koşullu `loopcond` (`1.9299`) de fark yaratmadı (`p = 0.761`); `k=4` paralel uzman (`1.9224`) aynı hesaplı düz geniş katmana (`dense_wide2x: 1.9082`) kaybetti (`p = 0.015`).
   - `E=8, k=2` (`p4_*`): **`HashRouter` (`1.8523 mix / 2.0303 web`) öğrenilmiş yönlendiriciyi (`1.8707 / 2.0503`) YENDİ**; `k=4` eş-FLOP'ta da `HashRouter` (`1.9021`) öğrenilmiş yönlendiriciyi (`1.9134`) yendi. Bayt düzeyinde tek harf başına öğrenilmiş yönlendirme başarısızdır.
2. **Pozitif Keşif — Yönlendiricisiz `LoopUntied_FFN` (`+%0` FLOP, `+0%` GDN state):**
   - Yalnızca 1. ve 2. turun FFN ağırlıklarını ayırmak (`untied_ffn` / `LoopUntied_FFN`), `S1` tabanını (`1.9443`) `−0.0340` ile `−0.0538 bpb` (`1.8905, p = 0.0038`) iyileştirerek **tüm eş-FLOP öğrenilmiş MoE varyantlarını geçti** (`p = 0.019`) ve `mqar_overwrite` doğruluğunu `%43.86 → %57.44` (`GDN_untied_L3: %30.80`) yükseltti. Uzmanlaşma bayt-bazlı değil **döngü-bazlıdır**.
3. **Güçlü Bir Sonraki Aday Sinyali — `Aday G` (`MoE_SwiGLU_Learned_NgramHash_E8k2`, `4.68M` param, `+%2.23` aktif FLOP):**
   - `mix_val = 1.8248 ± 0.0029` (`p = 0.0007`), `web_val = 2.0151` (`GDN_untied_L3: 1.8449 / 2.0278`'i geçti!) ve `mqar_overwrite = %83.98` (düz RDFL `%43.86`, `GDN_L3 %30.80`).
   - **Zorunlu sonraki kontrol:** `NgramHash` gömmesini (`+1.09M` param) MoE'den izole edip tek başına `LoopUntied_FFN + NgramHash` ile kıyaslamak gerekir (sonraki denetim turunda Faz 4 tamamlanırken yapılacak).

---

## 5. Figürler (`docs/figures/MANIFEST.json`)

| Şekil | Dosya | İçerik | Durum |
|---|---|---|---|
| 1 | `docs/figures/fig1_ablation_k1k5_rivals.svg` | (a) K1–K5 eski/yeni, (b) 12×4 adil rakip matrisi (`Deney B`), (c) 10-tohum rollout (`Deney C`) | **TAM (`0` bekleyen panel)** |
| 2 | `docs/figures/fig2_scaling_depth.svg` | (a) `S0–S3` log-log güç yasası (`50` JSON, `R²=0.985`), (b) `R_test = 1..8` | **TAM (`0` bekleyen panel)** |
| 3 | `docs/figures/fig3_horizon_memory_serving.svg` | (a) `H1` ayrık `mix_val`/`web_val` `512→32K`, (b) canlı tepe GPU belleği, (c) eş-param serving (`291.3×`) | **TAM (`0` bekleyen panel)** |
| 4 | `docs/figures/fig4_mechanistic_xray.svg` | (a) `S_t` etkin rank, (b) `α_t` yarı-ömür histogramı, (c) BDH seyreklik/ölü nöron | **TAM (`0` bekleyen panel)** |

Yeniden üretim: `CUDA_VISIBLE_DEVICES="" python eval/plot_figures.py` (`181` geçerli RFC 8259 JSON, `0` karantina, `0` yasaklı isim).

---

## 6. Kapsam Dışı ve Hijyen

- Önceki fazların hibrit model sayıları bu raporun kanıt tabanına **alınmamıştır**; tüm sayılar izole Saf RDFL v0.3 deneylerinden üretilmiştir.
- `test.bin` hiçbir koşuda okunmamış, yasaklı veri kümesi adları tüm belge, JSON ve SVG'lerde sıfırlanmıştır (`forbidden_name_hits = 0`).
