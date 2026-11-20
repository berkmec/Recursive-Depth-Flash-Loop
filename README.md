# Recursive Depth Flash Loop (`RDFL`)

**A Byte-Level Constant-Memory Hybrid Sequence Architecture with Weight-Tied Recursive Depth, FP32 Chunk-Boundary Memory, Decoupled Speculative Heads, and Verifier-Gated Self-Improvement**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](pyproject.toml)
[![Raw JSON Verification](https://img.shields.io/badge/Verified_Cells-320%2F320_(0_mismatches)-1D6B5E.svg)](benchmarks/verify_audited_results.py)

**[English](README.md)** · **[Türkçe](README.tr.md)** · **[Architectural Specification](docs/ARCHITECTURE.md)** · **[Audited Summary JSON](results/audited_summary.json)** · **[Raw Audited JSONs](results/raw/)**

---

## Abstract

Standard autoregressive Transformers maintain a key–value (KV) cache whose memory footprint grows linearly with sequence length $O(T)$, while windowed-attention backbones bound memory at the cost of losing state beyond the attention horizon $M$. Conversely, pure linear-recurrent models struggle with exact associative overwrite and negative-eigenvalue state transitions (e.g., parity and non-abelian permutations).

We introduce **Recursive Depth Flash Loop (`RDFL`)**, a tokenizer-free, byte-level ($x_t \in \{0,\dots,255\}$) hybrid sequence architecture that couples a causal input trunk with a weight-tied recursive refinement core operating over constant-size $O(1)$ recurrent states:

1. **Weight-Tied Recursive Depth (with Optional Input Anchoring):** Iterates a shared recurrent block $R$ times ($h_t^{(r-1)} + \alpha_r e_t + e_r$). Controlled `1.20M` ablations (Section 2) show that `RDFL`'s primary capability gain stems from the synergy of `GatedDeltaNet + BDH + RLT-2` rather than input anchoring alone (unanchored `A6_R4_cipasiz` matches anchored `A6_R4_cipali` on `Multi-Hop ICL` and slightly improves `Byte LM` at marginally higher variance).
2. **Chunk-Parallel Gated DeltaNet with Householder Write Gates:** Combines causal short convolution ($k=4$), multi-scale slow decay ($\alpha_0 \in [0.953, 0.999]$), and Householder-range write gates ($\beta_t = 2\sigma(\cdot) \in (0,2)$), maintaining an exact $O(1)$ matrix state $S_t^{(r)} \in \mathbb{R}^{H \times d_h \times d_h}$ with chunk-parallel WY and single-step equivalence (`4.44e-16` max error in FP64).
3. **Sparse Associative Gating (`BDH`) & Chunk-Boundary `FP32 SlotCore`:** Pairs Gated DeltaNet with a doubly-sparse ReLU interaction gate and a chunk-boundary `float32`-locked memory bank ($S=4$), eliminating mixed-precision NaN crashes (`0/36` NaN across unseen seeds; see Section 3.1 for both its iso-param `Byte LM` trade-off and our mechanistic symmetry-collapse finding).
4. **Zero-Initialization Residual Adaptation & Lossless Auxiliary Heads:** A zero-initialized output projection ($W_{\uparrow}|_0 = 0$) attaches `RDFL` onto pretrained backbones with exact `0.0` step-0 deviation, alongside lossless speculative decoding (`K3` and `MTP-4` in [`rdfl/speculative.py`](rdfl/speculative.py)), zero-parameter calibrated uncertainty (`MSP + T*`), zero-retention ring streaming, and verifier-gated bounded self-improvement (`AZR` RSI).

---

## 1. Architecture

![RDFL Architecture](docs/diagrams/rdfl_architecture.svg)

*Figure 1: Computational graph of `RDFL` (bottom-to-top). Blue denotes the causal input trunk and narrow projection; amber denotes the weight-tied recursive depth loop and chunk-boundary `FP32 SlotCore` feedback; teal denotes the constant-size $O(1)$ Gated DeltaNet matrix states and decoupled auxiliary prediction heads (`K3`, `MTP-4`). Full equations and exact module distinctions (`RDFLModel` vs. `ZeroDiffRDFLAdapter`) are provided in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*

---

## 2. Iso-Parameter `1.20M` Benchmarks (`204` Controlled Runs)

All `1.20M` architectural comparisons were executed across 3 independent random seeds and verified against raw JSON artifacts in [`results/raw/`](results/raw/) (frozen code hash `ab9e7db6220038a6`; `0` NaN across all `186` core and architectural ablation runs; the `2` FP16 NaN runs in unguarded FP16 `A6_SLOT4_MESAJLI` on `S5-Perm32` directly motivated `FP32 SlotCore` in Section 3).

![RDFL 1.20M Benchmarks and Decode State Scaling](docs/diagrams/benchmarks_1_2m.svg)

All 8 controlled arms (`mean ± sample std` across 3 seeds, verified programmatically from [`results/raw/ablasyon_sonuclari.json`](results/raw/ablasyon_sonuclari.json); algorithmic columns report the harder tail / overwrite / 3-hop diagnostic metrics `kuyruk_48_63_acc`, `kuyruk_24_31_acc`, `uzerine_yazilan_acc`, `3hop_acc`):

| Arm ID | Model Family | Params | `Parity-64` tail (%) | `S5-Perm32` tail (%) | `MQAR Overwrite` (%) | `Multi-Hop ICL` 3-hop (%) | Byte LM (`bpb` ↓) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `A0` | `RLT-0` (Feedback-Off) | `1.20M` | `50.18 ± 0.37` | `19.45 ± 0.65` | `22.40 ± 1.28` | `7.13 ± 0.88` | `2.4019 ± 0.0065` |
| `A1` | `RLT-1` (Token-Step Feedback) | `1.20M` | `79.78 ± 26.25` | `99.93 ± 0.12` | `21.96 ± 1.32` | `7.32 ± 0.49` | `2.3648 ± 0.0174` |
| `A2` | `RLT-2` (Chunk Boundary) | `1.20M` | `50.15 ± 0.43` | `20.27 ± 1.86` | `20.92 ± 1.31` | `7.03 ± 0.29` | `2.4207 ± 0.0217` |
| `A3` | `Flash Hebbian` (Raw) | `1.20M` | `64.68 ± 23.05` | `92.75 ± 3.73` | `3.21 ± 0.48` | `5.63 ± 2.48` | `2.3504 ± 0.0028` |
| `A4` | `GatedDeltaNet` (Iso-Param) | `1.20M` | `100.00 ± 0.00` | `73.13 ± 3.54` | `19.27 ± 7.73` | `6.12 ± 3.03` | `2.2786 ± 0.0050` |
| `A4B` | `GatedDeltaNet` (Iso-FLOP) | `2.57M` | `100.00 ± 0.00` | `95.17 ± 3.93` | `23.63 ± 1.31` | `3.97 ± 0.50` | **`2.1893 ± 0.0064`** |
| `A5` | `BDH-CQ Loop` | `1.20M` | `49.88 ± 0.40` | `72.09 ± 5.39` | `3.36 ± 0.23` | `3.29 ± 0.39` | `2.4676 ± 0.0041` |
| **`A6`** | **`RDFL` (`Delta+BDH+RLT2`)** | **`1.20M`** | **`100.00 ± 0.00`** | **`100.00 ± 0.00`** | **`100.00 ± 0.00`** | **`22.72 ± 1.30`** | `2.2896 ± 0.0060` |

- **State Memory Complexity (`L=6, d=768, H=8`, `d_h=96`):** At measured context lengths $T \in \{512,\, 4{,}096,\, 32{,}768\}$, FP16 Full-KV Transformer decode cache scales linearly as `9.0 / 72.0 / 576.0 MiB`, whereas `GatedDeltaNet` maintains a constant **`1.7139 MiB`** (`1.71 MiB` measured in [`results/raw/gpu_profil_55m.json`](results/raw/gpu_profil_55m.json)) — an exact **`336.1×`** reduction at $T = 32{,}768$ (reproducible on CPU via `python benchmarks/profile_decode_memory.py`). At `55M` training context $T = 8{,}192$, `RLT-2` OOMs above the `3,072 MiB` ceiling while `RDFL (R=2)` trains in `1,877.4 MiB`.
- **Structural & Input-Anchoring Ablations (`dongu_cipa_rlt2_ablasyonu`):**
  - *Chunk-Boundary Feedback (`RLT-2`):* Removing boundary feedback (`A6_R4_rlt2siz`) drops `Multi-Hop ICL` 3-hop accuracy from `22.72 ± 1.30%` to `10.32 ± 11.20%` (`-12.40 pp`, `8.6×` higher std), while slightly lowering `Byte LM` (`2.2790` vs `2.2896 bpb`).
  - *Loop Depth (`R`):* `A6_R2_cipali` matches `A6_R4_cipali` (`23.28 ± 0.92%`, `2.2903 ± 0.0019 bpb`) at `1.8×` higher throughput (`34.43` vs `19.00 KB/s`), whereas `A6_R1_cipali` degrades to `18.00 ± 8.27%` (`2.3044 bpb`).
  - *Input Anchoring (`cipali` vs. `cipasiz` — Honest Null Finding):* Unanchored `A6_R4_cipasiz` achieves equivalent `Multi-Hop ICL` (`22.69 ± 1.76%` vs `22.72 ± 1.30%`) and slightly better `Byte LM` (`2.2843 ± 0.0065` vs `2.2896 ± 0.0060 bpb`) than anchored `A6_R4_cipali`; similarly, `A5_R4_cipasiz` outperforms `A5_R4_cipali` on both (`3.71 ± 0.74%` vs `3.29 ± 0.39%`; `2.4379 ± 0.0145` vs `2.4676 ± 0.0041 bpb`). `RDFL`'s core advantage therefore derives from the **`GatedDeltaNet + BDH + RLT-2`** interaction rather than input anchoring alone.

---

## 3. Audited Self-Capabilities (`ÖY-1 .. ÖY-10` Full Scope Transparency)

`ÖY-1 .. ÖY-8` were evaluated under a pre-registered protocol on unseen seeds `(42, 137, 2026)` strictly disjoint from calibration seed `401` (frozen code hash `c8c2744cd90baca1`; `0` FP16 NaN; peak `1,705.7 MiB` VRAM; raw JSONs in [`results/raw/`](results/raw/)). `ÖY-9` and `ÖY-10` statuses are reported transparently below so all 10 program items are accounted for without overclaiming.

![Audited Self-Capabilities and 46x Scale Crossover](docs/diagrams/self_capabilities_v2.svg)

| ID & Mechanism | Status / Verdict | Audited Empirical Findings (Verified from `results/raw/*.json`) |
| :--- | :---: | :--- |
| **`ÖY-1/1B`: `K3` Decoupled Early-Exit & Speculative Decode** | **Adopt** | Detached `R=1` projection improves `R=1` LM by `-0.3899 bpb` (`2.8886 -> 2.4987`) at `+0.0041 bpb` `R=4` tax. Parallel verification yields **`0` byte mismatch** across `288/288` prompts and `1.48×–1.54×` rollback-adjusted FLOP speedup (`1.09×` `K3/K2`, `0.991×` `K3/K4`, `1.28×` `K1/K4` eager wall-clock). |
| **`ÖY-2`: Stateful Adaptive Depth** | **Conditional** | Lossy early exit saves `25.69%` FLOPs at `+0.0444 bpb` / `-0.94 pp` on `byte_lm` (`chunk B=32`; byte-step: `30.57%` FLOPs at `+0.1234 bpb`) and `36.38%` FLOPs at `+0.0662 bpb` / `+0.23 pp` on `graf_izli`. Superseded in deployment by lossless `K3` speculative decode (`0` quality loss). |
| **`ÖY-3`: Calibrated `MSP + T*` Uncertainty** | **Adopt** | Zero extra parameters: `1.2M` `AUROC = 0.8424`, `ECE15 = 0.0116`, `+20.13 pp` accuracy at `40%` abstention (`52.68% -> 72.81%`), `8.32×` lower graph ECE; at `55M`, **`AUROC = 0.8994 / 0.9041`** with `74.65% -> 95.35%` SFT byte accuracy at `40%` abstention. |
| **`ÖY-4`: `MTP-4` Multi-Byte Lookahead** | **Adopt (`55M+`)** | Sharp **`46×` scale crossover**: `+0.0323 bpb` LM capacity tax (`+0.0272 bpb` on `rsi_zincir_L34`) at `1.20M` diminishes **`32×`** to `+0.0010 bpb` LM (`-0.0005 bpb` SFT) under full `55M` weight propagation (Section 4). Before MTP training (step 0), untrained heads slow speculative wall-clock (`0.501×`). |
| **`ÖY-5`: Prompt Self-Correction & MLP Verifier** | **Reject** | Under symmetric `T = 0.0` evaluation, prompt revision corrupted equal-step `D0` drafts (`87.37% -> 48.44% / 49.22%`). Zero-parameter sequence **`log p` (`AUROC = 0.9513`)** decisively outperformed a 25K-parameter MLP verifier (`0.5867`). |
| **`ÖY-6`: `FP32 SlotCore` Chunk-Boundary Memory** | **Adopt (with Caveats 3.1)** | Locking updates to `float32` yields **`0/36` NaN**, lifts 3-hop `graf_izli` from `68.09 ± 54.73%` to **`96.89 ± 1.47%`** (`37.3×` lower std), and reaches `99.88 ± 0.09%` on `MQAR Overwrite`. **Caveats:** `+0.0173 bpb` iso-param `Byte LM` tax (`2.3819` vs `2.3646`, `ffn 512->128`) and live slot symmetry collapse (Section 3.1). `BDH` beats `BDH-less` on `3/3` seeds (`-0.0071 bpb`). |
| **`ÖY-7`: Verifier-Gated Bounded `AZR` RSI** | **Adopt (Bounded)** | Over `75` live generations, unverified blind RSI collapsed (`LM: 7.5162 bpb`, `ID: 63.54%`). External deterministic verifier + permanent `Tur 0` ID anchor + `30%` LM replay + exact 3D per-token reverse KL + rollback yielded **best `LM = 2.9301 bpb`**, `0` rollbacks, `ID = 98.44%`, and highest `IPT = 80.45%` (vs Oracle `71.26%`, Control `66.59%`). |
| **`ÖY-8`: Zero-Retention Ring Stream** | **Adopt** | Receipt-locked streaming ring buffer enforces `0.0 MiB` residual disk (`32.0 MiB` stream, `4.0 MiB` peak disk, `0.0 MiB` VRAM). |
| **`ÖY-9`: Online Weight Self-Editing (`SEAL` / `E-SELF`)** | **Deferred (Out of Scope)** | Evaluated via literature review only; deferred from the `1.20M` / `55M` V2 experimental matrix (no standalone experiment claimed). |
| **`ÖY-10`: `ToolRunner` (`<calc>` / `<check>` / `<note>`)** | **Pre-Training Baseline Only** | Harness code resides in the external training repository (not bundled in this package). The only raw measurement in `results/raw/taban_55m_degerlendirme.json` is the pre-training `55M` baseline (`n=24` prompts: `0.0%` tool calls, `0.0%` accuracy). Post-training `55M` evaluation is pending completion of the ongoing `42K` run. |

### 3.1 Seed-Level Breakdown (`graf_izli`), Iso-Param `Byte LM` Trade-Off & Mechanistic Honesty Finding

Per-seed `graf_izli` accuracy and `Byte LM` (`bpb` ↓) across unseen seeds `(42, 137, 2026)` from [`results/raw/oy5_6_sonuclari.json`](results/raw/oy5_6_sonuclari.json):

| Arm (`ÖY-6`, `1.20M`) | `graf_izli` Seeds `(42 / 137 / 2026)` | `graf_izli` Mean ± Std | `Byte LM` (`bpb` ↓) |
| :--- | :---: | :---: | :---: |
| `OY6_A6_TABAN` (`ffn=512`, Single-Vector) | `99.60% / 99.77% / 4.89%` | `68.09 ± 54.73%` | `2.3646 ± 0.0017` |
| `OY6_A6_BDHSIZ` (`ffn=896`, BDH-Less) | `5.97% / 99.34% / 99.92%` | `68.41 ± 54.08%` | `2.3717 ± 0.0030` |
| `OY6_A6_SASKINLIK_GDN` (`Titans-A`) | `99.61% / 59.72% / 85.06%` | `81.46 ± 20.19%` | **`2.3641 ± 0.0015`** |
| **`OY6_A6_SLOT4_FP32` (`ffn=128`, `FP32 SlotCore`)** | **`95.30% / 98.19% / 97.17%`** | **`96.89 ± 1.47%`** | `2.3819 ± 0.0055` (`+0.0173`) |

- **Iso-Parameter `Byte LM` Trade-Off (`+0.0173 bpb`):** Keeping `OY6_A6_SLOT4_FP32` iso-parameter at `1.20M` required shrinking its FFN from `512` to `128`, which caused a `+0.0173 bpb` regression on `Byte LM` (`2.3819` vs `2.3646`, worst of the 4 `ÖY-6` arms) despite dramatically stabilizing 3-hop graph tracking (`37.3×` lower std). At `55M`, `FP32 SlotCore` is attached externally onto the unshrunk `ffn=3072` backbone to avoid cannibalizing FFN width.
- **Mechanistic Honesty Finding — Symmetric Collapse of `FP32 SlotCore` Slots ([`results/raw/slotcore_55m_cosine_probe.json`](results/raw/slotcore_55m_cosine_probe.json)):** Direct CPU probing of the live trained `55M` checkpoint (step `21,000`, `16` real text chunks / `1,024` bytes from `mix/validation.bin`) shows that while `init_slots` start at pairwise cosine **`0.5017–0.7512`** (`mean = 0.6446`, `norm = 0.6941`), after `16` chunks all $S=4$ slots collapse to pairwise cosine **`0.999982`** (`min = 0.999959`, `max = 0.999999`) with mean L2 norm **`664.5`** (growing further from `~577` observed at step `17,500`; reproducible on CPU via `python benchmarks/run_1_2m_suite.py --smoke --measure-slot-cosine`, which reaches `cos ≈ 0.988` after 8 chunks). Because `commit_boundary` broadcasts the identical `chunk_end_state` through shared weights without slot-axis competition or norm damping, **`FP32 SlotCore`'s empirical stability gain arises from an accumulative FP32 chunk-boundary residual state — not from "4 specialized independent slots."** For **`v0.3`**, we plan competitive slot-axis `SlotAttention` routing (softmax normalized across slots) paired with per-slot `RMSNorm` and learned contraction/damping.

---

## 4. `55M` Warm Adaptation & Scale Crossover (`Diogenes-RDFL-55M`)

We attach `RDFL` (`d = 384`, `R = 2`, `2.81M`), `FP32 SlotCore` (`S = 4`, `1.33M`), `K3` (`0.74M`), and `MTP-4` (`2.36M`) (`7.24M` total adapter parameters) onto a pretrained `55.08M` byte-level windowed-attention backbone (`M = 512`, `d_base = 1536`, `ffn = 3072` preserved unshrunk; `62.32M` grand total) paired with an episodic content-memory reader (`cmem`).

- **Gate-1 Verification ([`results/raw/dogrulama_kapisi_55m.json`](results/raw/dogrulama_kapisi_55m.json), `kod_sha256 = 3c1a59a1dd2bb0f7`):** Exact step-0 deviation `0.0` across logits, normalized hidden state, and `cmem` outputs; strict causality error `0.0`; `forward() == step()` parity `1.335e-05` (FP32); `0/12` smoke NaNs (`3,438.7 MiB` peak VRAM).
- **Controlled `1,500`-Step `MTP-4` A/B Gate ([`results/raw/saat1_mtp_kapisi.json`](results/raw/saat1_mtp_kapisi.json)) & Preliminary Live Trajectory ([`results/raw/diogenes_55m_live_val_snapshot.json`](results/raw/diogenes_55m_live_val_snapshot.json)):**

| Model State (`fast-val` protocol*) | `LM (T=512)` ↓ | `K3 R=1 (T=512)` ↓ | `LM (T=2048)` ↓ | `SFT Val` ↓ | `MTP t+2 / t+3 / t+4` ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Pretrained `55M` Base (Step 0) | `1.1268` | `1.1268` | `1.0419` | `1.2416` | `10.1% / 9.2% / 10.7%` |
| Control `G0` ($\lambda_{\text{MTP}} = 0$, 1.5K; `RDFL` on) | `1.1232` | `1.1233` | `1.0344` | `1.2415` | `10.2% / 9.3% / 10.7%` |
| Full `55M` Prop `G1` ($\lambda_{\text{MTP}} = 0.15$, 1.5K; `RDFL` on) | **`1.1242`** (`+0.0010`) | **`1.1237`** | **`1.0362`** | **`1.2410`** (`-0.0005`) | **`34.1% / 24.6% / 21.4%`** |
| *Preliminary* Live `G1` (Step 7,000 / 42,000) | `1.1207` | `1.1198` | `1.0277` (`-0.0142`) | `1.2442` | `51.6% / 39.6% / 32.4%` |
| *Preliminary* Live `G1` (Step 17,500 / 42,000) | `1.1188` | `1.1187` | `1.0278` (`-0.0141`) | `1.2492` | `54.2% / 42.5% / 34.5%` |
| *Preliminary* Live `G1` (Step 21,000 / 42,000) | `1.1197` | `1.1201` | `1.0254` (`-0.0165`) | `1.2485` | `54.9% / 42.9% / 35.4%` |

> *\*Evaluation Protocol Footnote:* The table above uses **`fast-val`** (`4` validation batches, `200` SFT lines; stored in [`results/raw/diogenes_55m_live_val_snapshot.json`](results/raw/diogenes_55m_live_val_snapshot.json)), yielding Step-0 `1.1268 / 1.0419 / 1.2416`. By contrast, the pre-training full battery in [`results/raw/taban_55m_degerlendirme.json`](results/raw/taban_55m_degerlendirme.json) uses **`full-val`** (`8` validation batches, `400` SFT lines), yielding Step-0 `1.1549 / 1.0718 / 1.2567` (while its internal `oy6_slotcore_ablasyon` sub-check uses `fast-val` `4` batches -> `1.0419`).
>
> **Control Scope Note:** The `1,500`-step `G0` vs. `G1` gate specifically **isolates `MTP-4`** ($\lambda_{\text{MTP}} = 0$ vs. $0.15$) with `RDFL` attached in both arms. To isolate `RDFL`'s incremental contribution over plain continued pretraining on the identical data mixture, a dedicated **RDFL-off continued-pretraining control (`G_BASE_ONLY`, `1,500` steps)** will run immediately after the ongoing `42,000`-step adaptation finishes, followed by the full post-training `55M` evaluation battery (including `ToolRunner`, `cmem`, real-document QA, speculative wall-clock speedups, and `8`-generation `AZR` RSI).

---

## 5. Methodological Limitations

1. **Reference Kernel Implementation:** All current experiments execute exact pure-PyTorch chunk-parallel WY and PyTorch SDPA. While Path-1 warm adaptation trains efficiently, large-scale from-scratch pretraining (`200M+`) requires fusing the `GatedDeltaNet` WY recurrence into a Triton / Flash Linear Attention kernel.
2. **Slot Symmetry Collapse & Iso-Param LM Tax (`v0.2`):** As detailed in Section 3.1, `Fp32SlotCoreBoundary` collapses its $S=4$ slots to pairwise cosine `0.99998` with growing norm (`664.5` at step `21K`) over long streams, and incurs a `+0.0173 bpb` LM tax when constrained to iso-param `ffn=128` at `1.20M`; competitive slot-axis routing with per-slot `RMSNorm` and damping is scheduled for `v0.3`.
3. **Input Anchoring Null Effect at `1.20M`:** Controlled ablations (Section 2) show that input anchoring does not drive `RDFL`'s core gains (`A6_R4_cipasiz` matches `A6_R4_cipali` on `Multi-Hop ICL` and slightly beats it on `Byte LM`).
4. **Far-OOD Extrapolation & Pending `55M` Post-Training Evaluation:** At `1.20M`, `OOD-Far` (`L = 7, 8`) remained `0.0%` across all RSI arms including Oracle supervision; `55M` post-training metrics (including `ToolRunner` `ÖY-10` and `G_BASE_ONLY` control) await completion of the ongoing `42K` run.
5. **Hardware & Verification Protocol:** All experiments ran on a single consumer 8 GiB Turing GPU with strict train/validation/test separation; **`320/320`** cells in `results/audited_summary.json` are programmatically verified against `results/raw/*.json` in CI.

---

## 6. Reference Implementation & One-Command Reproducibility (CPU)

```bash
git clone https://github.com/berkmec/Recursive-Depth-Flash-Loop.git
cd Recursive-Depth-Flash-Loop
pip install -e ".[dev]"

# 1. Mathematical invariant unit tests (chunk==step FP64, causality, zero-diff, K3/MTP-4 lossless speculative, FP32 SlotCore lock, AUROC ties)
python -m unittest discover -s tests -v

# 2. Programmatically verify all 320 audited cells in results/audited_summary.json against results/raw/*.json
python benchmarks/verify_audited_results.py

# 3. Measure O(1) decode state vs. FP16 full-KV cache directly from CPU PyTorch tensors
python benchmarks/profile_decode_memory.py

# 4. Run 1.20M algorithmic suite (4 tasks x 4 arms) + SlotCore pairwise cosine diagnostic (<10s on CPU)
python benchmarks/run_1_2m_suite.py --smoke

# 5. Regenerate & XML-validate all vector figures directly from audited JSONs
python scripts/build_diagrams.py
```

```python
import torch
from rdfl import (
    RDFLConfig,
    RDFLModel,
    ZeroDiffRDFLAdapter,
    greedy_decode,
    msp_uncertainty,
    speculative_decode_k3,
    speculative_decode_mtp4,
)

cfg = RDFLConfig(d_model=192, num_heads=6, ffn_dim=512, num_loops=2, num_slots=4)
model = RDFLModel(cfg).double().eval()
prompt = torch.randint(0, 256, (1, 32))

# Lossless (0-byte-mismatch) speculative decoding with K3 and MTP-4
ref = greedy_decode(model, prompt, max_new_tokens=16)
res_k3 = speculative_decode_k3(model, prompt, max_new_tokens=16, draft_k=2)
res_mtp = speculative_decode_mtp4(model, prompt, max_new_tokens=16)
assert (res_k3.tokens == ref).all() and (res_mtp.tokens == ref).all()

# Zero-parameter MSP uncertainty & zero-diff pretrained backbone adapter
logits, _, _, _ = model.forward_stateful(prompt)
conf, unc = msp_uncertainty(logits)
adapter = ZeroDiffRDFLAdapter(d_base=1536, d_narrow=384, num_heads=6, num_loops=2)
z_base = torch.randn(1, 32, 1536)
assert (adapter(z_base) - z_base).abs().max().item() == 0.0
```

---

## References

1. **Gated DeltaNet:** Songlin Yang, Jan Kautz, and Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. [`arXiv:2412.06464`](https://arxiv.org/abs/2412.06464)
2. **RLT:** Yifan Zhang et al. *Recurrent Looped Transformer.* 2026. [`arXiv:2610.07591`](https://arxiv.org/abs/2610.07591)
3. **BDH & BDH-CQ:** Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, and Michał Bartoszkiewicz. *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* 2025. [`arXiv:2509.26507`](https://arxiv.org/abs/2509.26507) · Björn Engdahl, Adrian Kosowski, Jan Chorowski, Zuzanna Stamirowska, Przemysław Uznański, Junlin Jiang, Rohan Phadke, Remigiusz Kinas, and Richard Zhong. *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* 2026. [`arXiv:2608.09888`](https://arxiv.org/abs/2608.09888)
4. **Slot Attention:** Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. *Object-Centric Learning with Slot Attention.* NeurIPS 2020. [`arXiv:2006.15055`](https://arxiv.org/abs/2006.15055)
5. **Medusa:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. [`arXiv:2401.10774`](https://arxiv.org/abs/2401.10774)
6. **MTP:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction.* ICML 2024. [`arXiv:2404.19737`](https://arxiv.org/abs/2404.19737)
7. **AZR:** Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. *Absolute Zero: Reinforced Self-play Reasoning with Zero Data.* 2025. [`arXiv:2505.03335`](https://arxiv.org/abs/2505.03335)
8. **Dr. GRPO:** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. [`arXiv:2503.20783`](https://arxiv.org/abs/2503.20783)
9. **Titans:** Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. *Titans: Learning to Memorize at Test Time.* 2025. [`arXiv:2501.00663`](https://arxiv.org/abs/2501.00663)
10. **SEAL:** Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, and Pulkit Agrawal. *Self-Adapting Language Models.* 2025. [`arXiv:2506.10943`](https://arxiv.org/abs/2506.10943)
11. **MSP:** Dan Hendrycks and Kevin Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017. [`arXiv:1610.02136`](https://arxiv.org/abs/1610.02136)
12. **Temperature Scaling:** Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. [`arXiv:1706.04599`](https://arxiv.org/abs/1706.04599)

---

## Citation

```bibtex
@software{berkmec2026rdfl,
  author  = {berkmec},
  title   = {Recursive Depth Flash Loop (RDFL): A Byte-Level Constant-Memory Hybrid Sequence Architecture},
  year    = {2026},
  url     = {https://github.com/berkmec/Recursive-Depth-Flash-Loop}
}
```
