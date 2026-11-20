# Recursive Depth Flash Loop (`RDFL`)

**A Byte-Level Constant-Memory Hybrid Sequence Architecture with Weight-Tied Recursive Depth, FP32 Working-Memory Slots, Decoupled Speculative Heads, and Verifier-Gated Self-Improvement**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](pyproject.toml)
[![Independent Audit](https://img.shields.io/badge/Audit-PASS_(1%2C073%2F1%2C073_cells)-1D6B5E.svg)](results/audited_summary.json)

**[English](README.md)** · **[Türkçe](README.tr.md)** · **[Architectural Specification](docs/ARCHITECTURE.md)** · **[Audited Summary JSON](results/audited_summary.json)**

---

## Abstract

Standard autoregressive Transformers maintain a key–value (KV) cache whose memory footprint grows linearly with sequence length $O(T)$, while windowed-attention backbones bound memory at the cost of losing state beyond the attention horizon $M$. Conversely, pure linear-recurrent models struggle with exact associative overwrite, negative-eigenvalue state transitions (e.g., parity and non-abelian permutations), and unanchored depth recurrence.

We introduce **Recursive Depth Flash Loop (`RDFL`)**, a tokenizer-free, byte-level ($x_t \in \{0,\dots,255\}$) hybrid sequence architecture that couples a causal input trunk with a weight-tied recursive refinement core operating over constant-size $O(1)$ recurrent states:

1. **Input-Anchored Recursive Depth:** Iterates a shared recurrent block $R$ times with per-step normalized input anchoring ($h_t^{(r-1)} + \alpha_r e_t + e_r$), overcoming the optimization collapse of unanchored weight-tied loops.
2. **Chunk-Parallel Gated DeltaNet with Householder Write Gates:** Combines causal short convolution ($k=4$), multi-scale slow decay ($\alpha_0 \in [0.953, 0.999]$), and Householder-range write gates ($\beta_t = 2\sigma(\cdot) \in (0,2)$), maintaining an exact $O(1)$ matrix state $S_t^{(r)} \in \mathbb{R}^{H \times d_h \times d_h}$ with chunk-parallel WY and single-step equivalence (`4.44e-16` max error in FP64).
3. **Sparse Associative Gating (`BDH`) & Chunk-Boundary `FP32 SlotCore`:** Pairs Gated DeltaNet with a doubly-sparse ReLU interaction gate and a chunk-boundary bank of $S=4$ working-memory slots evaluated strictly in `float32`, eliminating mixed-precision instability (`0` NaN across all experiments).
4. **Zero-Initialization Residual Adaptation & Auxiliary Heads:** A zero-initialized output projection ($W_{\uparrow}|_0 = 0$) attaches `RDFL` onto pretrained backbones with exact `0.0` step-0 deviation, alongside decoupled early-exit (`K3`), multi-byte lookahead (`MTP-4`), zero-parameter calibrated uncertainty (`MSP + T*`), deterministic tool execution (`ToolRunner`), and verifier-gated bounded self-improvement (`AZR` RSI).

---

## 1. Architecture

![RDFL Architecture](docs/diagrams/rdfl_architecture.svg)

*Figure 1: Computational graph of `RDFL` (bottom-to-top). Blue denotes the causal input trunk and narrow anchor projection; amber denotes the weight-tied recursive depth loop and chunk-boundary `FP32 SlotCore` feedback; teal denotes the constant-size $O(1)$ Gated DeltaNet matrix states and decoupled auxiliary prediction heads (`K3`, `MTP-4`). Full equations and design rationale are provided in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).*

---

## 2. Iso-Parameter `1.20M` Benchmarks (`204` Controlled Runs)

All `1.20M` architectural ablations were evaluated across 3 independent random seeds (`0` NaN) and verified against raw JSON artifacts via a zero-context independent audit (`411/411` cells matched; frozen code hash `ab9e7db6220038a6`).

![RDFL 1.20M Benchmarks and Decode State Scaling](docs/diagrams/benchmarks_1_2m.svg)

| Model Family | Params | `Parity-64` (%) | `S5-Perm32` (%) | `MQAR Overwrite` (%) | `Multi-Hop ICL` (%) | Byte LM (`bpb` ↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `RLT-0` (Feedback-Off) | `1.20M` | `50.18` | `19.45` | `22.40` | `7.13` | `2.4019` |
| `RLT-1` (Token-Step Feedback) | `1.20M` | `79.78` | `99.93` | `21.96` | `7.32` | `2.3648` |
| `RLT-2` (Chunk Boundary) | `1.20M` | `50.15` | `20.27` | `20.92` | `7.03` | `2.4207` |
| `GatedDeltaNet` (Iso-FLOP) | `2.57M` | `100.00` | `100.00` | `23.63` | `3.97` | `2.2786` |
| **`RDFL`** | **`1.20M`** | **`100.00`** | **`100.00`** | **`100.00`** | **`22.72`** | **`2.2896`** |

- **State Memory Complexity:** At $T = 32{,}768$, a 6-layer full-KV Transformer requires `576 MiB` of decode cache whereas `RDFL` maintains a constant **`1.71 MiB`** (`337x` reduction).
- **Structural Ablations:** Removing chunk-boundary feedback reduces `Multi-Hop ICL` from `22.72%` to `10.32%` (`-12.40 pp`, `8.6x` higher standard deviation). Setting `R = 2` matches `R = 4` (`23.28%`, `2.2903 bpb`) at `1.8x` higher throughput.

---

## 3. Audited Self-Capabilities (`ÖY-1 .. ÖY-10`)

Ten auxiliary capabilities were evaluated under a pre-registered protocol on unseen seeds `(42, 137, 2026)` strictly disjoint from the calibration seed (`401`), verified by independent zero-context audit (`662/662` cells matched; frozen code hash `c8c2744cd90baca1`; `0` NaN).

![Audited Self-Capabilities and 46x Scale Crossover](docs/diagrams/self_capabilities_v2.svg)

| Mechanism | Verdict | Audited Empirical Findings |
| :--- | :---: | :--- |
| **`K3` Decoupled Early-Exit & Speculative Decode (`ÖY-1/1B`)** | **Adopt** | Detached `R=1` projection improves `R=1` LM by `-0.3899 bpb` (`2.8886 -> 2.4987`) at only `+0.0041 bpb` `R=4` tax. Exact parallel verification yields **`0` byte mismatch** across `288/288` prompts and `1.48x–1.54x` rollback-adjusted FLOP reduction (`1.09x–1.28x` eager wall-clock). |
| **`FP32 SlotCore` Working Memory (`ÖY-6`)** | **Adopt** | Locking slot self-attention to `float32` yields **`0` NaN (`36/36` runs)**, lifts 3-hop graph reasoning (`graf_izli`) from `68.09% ± 54.73%` to **`96.89% ± 1.47%`** (`37.2x` lower std), and reaches `99.88%` on `MQAR Overwrite`. `BDH` outperforms `BDH-less` on `3/3` unseen seeds (`-0.0071 bpb`). |
| **Calibrated `MSP + T*` Uncertainty (`ÖY-3`)** | **Adopt** | Zero extra parameters: `1.2M` `AUROC = 0.8424`, `+20.13 pp` accuracy at `40%` abstention, `8.3x` lower graph ECE; at `55M`, **`AUROC = 0.8994 / 0.9041`** with `74.7% -> 95.4%` byte accuracy at `40%` abstention. |
| **`MTP-4` Multi-Byte Lookahead (`ÖY-4`)** | **Adopt (`55M+`)** | Exhibits a sharp **`46x` scale crossover**: `+0.0323 bpb` capacity tax at `1.20M` diminishes **`32x`** to `+0.0010 bpb` LM (`-0.0005 bpb` SFT) under full `55M` weight propagation (Section 4). |
| **Prompt Self-Correction & MLP Verifier (`ÖY-5`)** | **Reject** | Under symmetric `T = 0.0` evaluation, prompt revision corrupted `49%` of equal-step `D0` drafts (`87.37% -> 48.44%`). Zero-parameter sequence **`log p` (`AUROC = 0.9513`)** decisively outperformed a 25K-parameter MLP verifier (`0.5867`). |
| **Verifier-Gated Bounded `AZR` RSI (`ÖY-7`)** | **Adopt (Bounded)** | Over `75` live generations, unverified blind RSI collapsed (`LM: 3.03 -> 7.52 bpb`, `ID: 98.8% -> 63.5%`). Adding an external deterministic verifier, permanent `Tur 0` ID anchor, `30%` LM replay, exact 3D per-token reverse KL, and validation rollback yielded **best `LM = 2.9301 bpb`**, `0` rollbacks, and highest invariant generalization (`IPT = 80.45%` vs Oracle `71.26%`). |
| **Zero-Retention Ring Stream (`ÖY-8`) & `ToolRunner` (`ÖY-10`)** | **Adopt** | Receipt-locked streaming ring buffer enforces `0.0 MiB` residual disk; deterministic `<calc>` / `<check>` / `<note>` execution eliminates arithmetic hallucinations. |

---

## 4. `55M` Warm Adaptation & Scale Crossover (`Diogenes-RDFL-55M`)

We attach `RDFL` (`d = 384`, `R = 2`), `FP32 SlotCore` (`S = 4`), `K3`, and `MTP-4` onto a pretrained `55.08M` byte-level windowed-attention backbone (`M = 512`, `d_base = 1536`, `ffn = 3072` preserved unshrunk) paired with a `0.86M` episodic content-memory reader (`cmem`).

- **Initialization Verification (`kod_sha256 = 3c1a59a1dd2bb0f7`):** Exact step-0 deviation `0.0` across logits, normalized hidden state, and `cmem` outputs; strict causality error `0.0`; `forward() == step()` parity `1.34e-05` (FP32).
- **Controlled `1,500`-Step `MTP-4` A/B Study (Seed `42`) & Preliminary Trajectory:**

| Model State | `LM (T=512)` ↓ | `K3 R=1 (T=512)` ↓ | `LM (T=2048)` ↓ | `SFT Val` ↓ | `MTP t+2 / t+3 / t+4` ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Pretrained `55M` Base (Step 0) | `1.1268` | `1.1268` | `1.0419` | `1.2416` | `10.1% / 9.2% / 10.7%` |
| Control `G0` ($\lambda_{\text{MTP}} = 0$, 1.5K) | `1.1232` | `1.1233` | `1.0344` | `1.2415` | `10.2% / 9.3% / 10.7%` |
| Full `55M` Prop `G1` ($\lambda_{\text{MTP}} = 0.15$, 1.5K) | **`1.1242`** (`+0.0010`) | **`1.1237`** | **`1.0362`** | **`1.2410`** (`-0.0005`) | **`34.1% / 24.6% / 21.4%`** |
| Ongoing `G1` Run (Step 7,000 / 42,000) | **`1.1207`** | **`1.1198`** | **`1.0277`** (`-0.0142`) | `1.2442` | **`51.6% / 39.6% / 32.4%`** |

> *Status Note:* A full `42,000`-step adaptation followed by an `8`-generation verifier-gated `AZR` RSI phase is currently executing. Complete before-vs-after `55M` evaluations (long-context LM, `cmem` battery, real-document QA, lossless speculative decoding speedups, `ToolRunner`, `SlotCore` ablation, and `55M` RSI) will replace preliminary figures upon independent audit completion.

---

## 5. Methodological Limitations

1. **Reference Kernel Implementation:** All current experiments execute exact pure-PyTorch chunk-parallel WY and PyTorch SDPA. While Path-1 warm adaptation trains efficiently, large-scale from-scratch pretraining (`200M+`) requires fusing the `GatedDeltaNet` WY recurrence into a Triton / Flash Linear Attention kernel.
2. **Far-OOD Extrapolation:** At `1.20M`, `OOD-Far` (`L = 7, 8`) remained `0.0%` across all RSI arms including Oracle supervision; `55M`-scale behavior with negative-contrast updates is under active evaluation.
3. **Experimental Protocol:** All experiments were executed on a single consumer 8 GiB Turing-generation GPU with strict train/validation/test separation and automated audit verification (`1,073/1,073` verified cells across V2 studies).

---

## 6. Reference Implementation & Reproducibility

```bash
git clone https://github.com/berkmec/Recursive-Depth-Flash-Loop.git
cd Recursive-Depth-Flash-Loop
pip install -e ".[dev]"
python -m unittest discover -s tests -v
python scripts/build_diagrams.py
```

```python
import torch
from rdfl import RDFLConfig, RDFLModel, ZeroDiffRDFLAdapter, msp_uncertainty

cfg = RDFLConfig(d_model=192, num_heads=6, ffn_dim=512, num_loops=2, num_slots=4)
model = RDFLModel(cfg).eval()
x = torch.randint(0, 256, (1, 128))
logits, logits_r1, hn, state = model.forward_stateful(x)
mtp_t2_t4 = model.mtp_logits(hn)
conf, unc = msp_uncertainty(logits)

# Zero-diff attachment onto any pretrained trunk (exact 0.0 deviation at step 0)
adapter = ZeroDiffRDFLAdapter(d_base=1536, d_narrow=384, num_heads=6, num_loops=2)
z_base = torch.randn(1, 128, 1536)
assert (adapter(z_base) - z_base).abs().max().item() == 0.0
```

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
