# Architectural Specification: Recursive Depth Flash Loop (`RDFL`)

![RDFL v0.5.0 Pure-Recurrent Architecture](diagrams/rdfl_architecture.svg)

---

## 1. `RDFL v0.5.0` — Pure-Recurrent Unified Architecture (`rdfl/v05.py`)

`RDFL v0.5.0` (`RDFLv05Model`, `565,216` parameters in the reference CPU-verified configuration) completes the transition from the `v0.3` windowed-attention hybrid to a **100% pure-recurrent, tokenizer-free byte-level architecture**:

- **Zero Attention / Zero KV-Cache:** No `Softmax(QK^T)`, no RoPE, no PyTorch SDPA anywhere in `rdfl/v05.py`.
- **Zero `SlotCore`:** Symmetric 4-slot `FP32 SlotCore` (which collapsed to `cos = 0.999982` in `v0.3`) is completely absent; `v0.5.0` uses damped 1-slot `Fp32RLT2Boundary` (`rlt2_s1`).
- **Zero External Tokenizer:** Operates directly on raw byte streams $x_{1:t} \in \{0, \dots, 255\}^t$.
- **Strict $O(1)$ Constant State:** Exactly **`25,136` bytes** (`B=1`, `FP32`) across $T \in \{16..320\}$, $R \in \{1..1024\}$, and tasks, with `FP64 forward() == step()` parity $\le 8.10 \times 10^{-15}$ and future-byte leakage $= 0.0$.

### 1.1 `v0.3` Hybrid vs. `v0.5.0` Pure-Recurrent Comparison

| Dimension | Legacy `v0.3` (`rdfl/model.py`, `rdfl/layers.py`) | Unified `v0.5.0` (`rdfl/v05.py`) |
| :--- | :--- | :--- |
| **Attention / KV-Cache** | Optional causal windowed-attention trunk (`ZeroDiffRDFLAdapter`) + SDPA | **Zero Attention · Zero KV-Cache · Zero SDPA** |
| **Byte Stem & State Tracking** | Single-step `GatedDeltaNetLayer` (`n_h = 1`, `conv.bias != 0`) | `MSNgramHashStem` (`2/3/4/6`-gram) + **`MicroDeltaProductGDNLayer`** (`n_h = 4`, `BiasFreeCausalShortConv`, `n_assoc=2` + `n_hh=2`, `SoftRMSNorm`) |
| **Sequence Resolution** | Pure $P=1$ byte-level macro loop | **2-scale $P=1 \rightarrow P=4$** causal patch packer (`256 → 48 → 64`, `4×` fewer macro steps) |
| **Recursive Reasoning & Halting** | Fixed $R$-step loop (`R_test > R_train` diverges when Jacobian $> 1$) | **`step_halting` (default):** norm-bounded residual + learned halting gate $g_{\text{step}} \in (0, 1)$ (`g_step=0` bit-exact no-op, bounded at $R=1024$) + `reason_from_state` $W_r$ scratchpad; optional **`adfp`** contractive mode |
| **Chunk-Boundary State** | `rlt2_s1` default (`slotcore_s4` legacy ablation) | `Fp32RLT2Boundary` (`rlt2_s1`), committed **only** from $r=0$ output $h_1$ (`S_macro(R=1) == S_macro(R=16)`) |
| **Continual Memory** | Offline `AZR` RSI (`v0.3`) | **Stage 2–3:** Causal surprise gate (`logits_fast`, `0` leak) + **`OrthogonalSlowGDN`** (`M2 OWM-GDN`) + **`ConsolidationManager`** (`OGP + EWC + KL`, `0` replay) |
| **Decoder & Auxiliary Heads** | `K3` + `MTP-4` (`t+2..t+4`) | **Single-pass `dec_ffn` (`calls == 1`)** + untied `head` + `K3` (`0.0` core grad) + `MTP-3` (`t+2..t+4`) + `ÖY-3` (`MSP + T*`) |

### 1.2 Exact Mathematical Formulation (4-Stage Pipeline)

#### Stage 0 — Byte Stem & Dual-Regime `Micro-DeltaProduct` ($P=1$, $d_{\text{byte}}=64$)
Given raw byte input $x_{1:t} \in \{0, \dots, 255\}^t$, `RDFLv05Model` sums the byte embedding $E_{\text{byte}}(x_t)$ with a causal multi-scale Count-Sketch $\pm 1$ `2/3/4/6`-gram hash residual equipped with a byte-conditioned digit-protecting gate $\gamma(x_t) \in (0, 1)^4$ (`MSNgramHashStem`):

$$h_{b0, t} = E_{\text{byte}}(x_t) + \alpha_{\text{ng}}\,\mathrm{RMSNorm}\!\left(W_{\text{ng}}\,\big[\gamma_k(x_t)\, s_k \odot E_k(\mathrm{hash}_k(x_{\le t}))\big]_{k \in \{2,3,4,6\}}\right)$$

Normalized activations $\tilde{x}_t = \mathrm{RMSNorm}(h_{b0, t})$ feed `MicroDeltaProductGDNLayer` ($H=4$ heads: $n_{\text{assoc}}=2$ associative DeltaNet heads over `short_conv` output $x_{\text{sc}}$, and $n_{\text{hh}}=2$ Householder group-tracking heads over gated residual $x_{\text{hh}} = x + g_{\text{conv}} \odot x_{\text{sc}}$, where `BiasFreeCausalShortConv` uses kernel size $k=4$ and `bias=False`). Each token executes $n_h = 4$ sequential Householder-range delta reflections via a chunk-parallel Mod-1 WY triangular solver (`C_tok = chunk_size // n_h = 16`):

$$S_{t, 0}^{(h)} = \alpha_t^{(h)} S_{t-1}^{(h)}, \qquad S_{t, m}^{(h)} = S_{t, m-1}^{(h)} + \beta_{t, m}^{(h)}\!\left(v_{t, m}^{(h)} - S_{t, m-1}^{(h)} k_{t, m}^{(h)}\right)\!(k_{t, m}^{(h)})^\top, \quad m = 1, \dots, n_h$$

with $\beta_{t,m}^{(h)} = 2\sigma(\cdot) \in (0, 2)$ and energy-preserving `SoftRMSNorm` output gating:

$$c_t = h_{b0, t} + W_o\!\left(\mathrm{SoftRMSNorm}(S_{t, n_h} q_t) \odot \mathrm{SiLU}(W_g \tilde{x}_t)\right) \in \mathbb{R}^{d_{\text{byte}}}$$

#### Stage 1 — $P=4$ Macro Patch & Non-Collapsing `StepReasoner` ($d_{\text{macro}}=64$)
Every $P=4$ bytes are packed causally (`RMSNorm` $\rightarrow 4 d_{\text{byte}} \rightarrow 48 \rightarrow d_{\text{macro}}$) into macro tokens $u_m$. Inside `MacroRDFLv05Core`, `Fp32RLT2Boundary` (`rlt2_s1`) injects damped boundary feedback from prior chunk state $s_{\text{slot}} \in \mathbb{R}^{1 \times d_{\text{macro}}}$:

- **Loop $r = 0$ (Single Write):** `GDNReadOnlyLayer.write_and_read()` executes one chunk-parallel Gated DeltaNet write (`K_mem = 1`, `SoftSpectralLinear` projections, `bounded_q_norm`) followed by `BoundedBdhGate` (`ReLU(tanh(·)) * ReLU(tanh(·))`) and `BoundedSwiGLU` to produce $h_1$. Boundary state $s_{\text{slot}}$ commits **strictly from $h_1$** at chunk boundaries (`slot_st = commit_boundary(slot_st, h1_seg[:, -1])`), guaranteeing bit-exact $S_{\text{macro}}(R=1) == S_{\text{macro}}(R=16)$.
- **Loops $r \ge 1$ (`reasoning_mode="step_halting"`, default):** `GDNReadOnlyLayer.read_only()` queries the frozen $r=0$ WY cache without modifying $S_{\text{macro}}$. Each step combines phase-conditioned state $u_r = \mathrm{rms\_norm\_tau}(h_r + W_{\text{phase}}\phi(r) + e_{\text{ref}}, \tau=1.0)$ with `BoundedBdhGate` and `BoundedSwiGLU` into a norm-bounded candidate $h_{\text{cand}} = \mathrm{rms\_norm\_tau}(h_r + \delta_r, \tau=1.0)$, gated by a **phase-independent** learned halting gate $g_{\text{step}}(h_r, o_r) \in (0, 1)$:

$$h_{r+1} = h_r + g_{\text{step}}(h_r, o_r) \cdot \left(h_{\text{cand}} - h_r\right)$$

  Because $g_{\text{step}}$ does not depend on $\phi(r)$, once halted ($g_{\text{step}} \rightarrow 0$), extra iterations remain fixed-point stable ($h_{r+1} \equiv h_r$, verified `0.0` diff when `gate_override=0.0`), while avoiding contractive collapse up to $R=1024$ (`0` NaN/Inf, `mean_step_jvp_top_sv = 1.5621`). An optional backward-compatible `"adfp"` mode (`(1-\lambda)h + \lambda f_{\text{anchor}}`, `mean_step_jvp_top_sv = 0.7703`) is also preserved.
- **Unified Silent Reasoning (`reason_from_state` / `forward_step_reasoning`):** On single-token multi-hop tasks, `encode_for_reasoning()` extracts $(h_0, S_{\text{fast}}[:n_{\text{assoc}}], c_{\text{last}})$ and `reason_from_state()` iterates $R$ steps over read-only $S_{\text{fast}}$ plus a loop-internal $O(1)$ Delta-Householder scratchpad $W_r \in \mathbb{R}^{H \times d_h \times d_h}$, also gated by $g_{\text{step}}$ (`W_{r+1} = W_r + g_step * (W_cand - W_r)`).
- **Emergent Global Workspace (`J-Space` Jacobian Lens):** Strict out-of-sample Jacobian SVD (`K=6, R=8`, `clean_acc = 0.9896`) reveals a shared low-rank broadcast subspace across silent steps (`r_mid = 4, k = 14`): `5/5` monotone intermediate linear probes, **`2.6335×`** functional broadcast over $J_\perp$ (`p = 5.27e-15`), **`-80.21 pp`** selective ablation drop (`0.9896 → 0.1875` vs `0.9688` full $J_\perp$), **`100.0%`** same-table patching, and **`92.71%`** cross-table compositional patching ($P_B^4(v_2(A))$ vs `10.42%` in $J_\perp$; vs `adfp` null control `0.0%` / `5.21%`).

#### Stage 2–3 — Zero-Loop Causal Surprise Gate, `M2 OWM-GDN`, and `ConsolidationManager`
Executed **outside** the Stage 1 loop (`0.0` future leakage):

1. **Stage 2 Causal Surprise Gate:** Computes patch-averaged surprise solely from causally shifted `logits_fast` (never `S_slow`) and Stage-1 loop delta:
   $$u_{\text{tot}, m} = 0.72\,\bar{u}_{\text{NLL}, m} + 0.18\,\bar{u}_{\text{MSP}, m} + 0.10\,\tanh\!\left(\frac{\delta_{\text{conv}, m}}{\tau_\delta}\right), \qquad g_{\text{surp}, m} = \sigma\!\big(\kappa_{\text{surp}}(u_{\text{tot}, m} - \tau_{\text{surp}})\big) \cdot \mathrm{ramp}(u_{\text{tot}, m})$$
2. **Stage 3 `OrthogonalSlowGDN` (`M2 OWM-GDN`):** Scales effective decay and write rates by $g_{\text{surp}}$ (`log_alpha_eff = g_surp * log_alpha_base`, `beta_eff = g_surp * beta_base`) and projects normalized keys onto the orthogonal complement of committed slow-memory covariance $C_{\text{commit}}$:
   $$P_\perp = \left(I + \mu\,\tilde{C}_{\text{commit}}\right)^{-1}, \qquad p_m = P_\perp k_m, \qquad \eta_m = \frac{\beta_{\text{eff}, m}}{k_m^\top p_m + \epsilon_{\text{reg}}}$$
   $$S_{\text{slow}, m} = \alpha_{\text{eff}, m} S_{\text{slow}, m-1} + \eta_m \left(v_m - \alpha_{\text{eff}, m} S_{\text{slow}, m-1} k_m\right) p_m^\top$$
   producing $\delta_{\text{slow}} = \tanh(\gamma_{\text{slow}}) \odot y_{\text{slow}}$ (`0` drift on low-surprise distractors).
3. **Time-Scale 3 `ConsolidationManager` (Zero Replay Buffer):** After each task boundary, consolidates model parameters across tasks via (i) post-AdamW **Orthogonal Gradient Projection (`OGP`)** $\Delta W \leftarrow \Delta W (I + \mu_{\text{ogp}}\tilde{C})^{-1}$ applied to all `nn.Linear` / `SoftSpectralLinear` layers excluding `k3_adapter` and `mtp_heads`, (ii) scale-normalized **Online `EWC`** diagonal Fisher penalty over all `requires_grad` model parameters, and (iii) **reverse `KL`** anchor $D_{\text{KL}}(\pi_\theta \| \pi_0)$.

#### Stage 4 — Single-Pass Byte Decoder & Decoupled Ecosystem Heads
Completed macro context $h_{\text{fast}}$ and slow residual $\delta_{\text{slow}}$ are expanded causally back to byte resolution using $h_{\text{bos}}$ / $0$ before the first 4-byte patch completes ($t < 3$) and the last completed patch index $m_{\text{causal}}(t) = \lfloor (t + 1) / 4 \rfloor - 1$ for $t \ge 3$ ($0.0$ future-byte leakage):

$$h_{\text{ctx}, t} = \begin{cases} h_{\text{bos}}, & t < 3 \\ h_{\text{fast},\, m_{\text{causal}}(t)}, & t \ge 3 \end{cases}, \qquad \delta_{\text{ctx}, t} = \begin{cases} 0, & t < 3 \\ \delta_{\text{slow},\, m_{\text{causal}}(t)}, & t \ge 3 \end{cases}$$

$$y_{\text{base}, t} = c_t + W_{\text{ctx}}\,\mathrm{RMSNorm}(h_{\text{ctx}, t}) + e_{\text{slot}}[t \bmod 4], \qquad h_{\text{fast\_byte}, t} = y_{\text{base}, t} + \mathrm{SwiGLU}_{\text{dec}}\!\left(\mathrm{RMSNorm}(y_{\text{base}, t})\right)$$

Followed by slow residual addition $h_{\text{dec}, t} = h_{\text{fast\_byte}, t} + W_{\text{ctx\_slow}}\,\delta_{\text{ctx}, t}$ and untied readout $\mathrm{logits}_t = W_{\text{out}}\mathrm{RMSNorm}(h_{\text{dec}, t})$ (`W_out != W_emb`). `dec_ffn` executes **strictly once** per byte (`dec_ffn_calls == 1`, `slowOFF` and `slowON`). Attached auxiliary capabilities: **`K3`** (`k3_adapter` on detached $h_{r1}$, `0.0` core/head gradient), **`MTP-3`** (`t+2, t+3, t+4` bottleneck adapters), and **`ÖY-3`** (`u_t = 1 - max softmax(logits / T*)`).

### 1.3 Exact `O(1)` Constant Inference State Breakdown (`RDFLv05State`)

Measured directly from `state_nbytes(model._init_state(B=1, cpu, float32))` on the `565,216`-parameter reference config (`results/v05_audited_summary.json`):

| Field in `RDFLv05State` | Shape (`B=1`, `H=4`, `d_h=16`, `d=64`, `P=4`) | Dtype | Exact Bytes | Role |
| :--- | :---: | :---: | :---: | :--- |
| `s_fast` | `(1, 4, 16, 16)` | `float32` | `4,096 B` | Stage-0 Micro-DeltaProduct matrix state |
| `conv_state` | `(1, 64, 3)` | `float32` | `768 B` | Stage-0 causal short-conv (`k=4`) buffer |
| `s_macro` | `(1, 4, 16, 16)` | `float32` | `4,096 B` | Stage-1 Macro GDN matrix state |
| `conv_macro` | `(1, 64, 3)` | `float32` | `768 B` | Stage-1 macro short-conv (`k=4`) buffer |
| `slot_state` | `(1, 1, 64)` | `float32` | `256 B` | `Fp32RLT2Boundary` (`rlt2_s1`) damped slot |
| `s_slow` | `(1, 4, 16, 16)` | `float32` | `4,096 B` | Stage-3 `M2 OWM-GDN` slow matrix state |
| `c_slow` (`c_commit`, `c_pending`) | `2 × (1, 4, 16, 16)` | `float32` | `8,192 B` | Stage-3 orthogonal key-covariance accumulators |
| `prev_fast_logits` | `(1, 256)` | `float32` | `1,024 B` | Causally shifted logits for `u_NLL` surprise |
| `stem_state` (`6` tensors) | `(1,5)` `i64`; `(1,4,64)`, `3×(1,64)`, `(1,2)` `f32` | mixed | `1,840 B` | N-gram prefix (`40 B`), $P=4$ patch ring (`1,024 B`), cached `h_fast`/`delta_slow`/`h_r1` (`768 B`), `surp_acc` (`8 B`) |
| **Total (`state_nbytes`)** | **Constant across $T \in [16, 320]$, $R \in [1, 1024]$** | — | **`25,136 B`** | **Strict $O(1)$ Inference State** |

### 1.4 Kernel & Triton Implementation Note

The reference PyTorch implementation in `rdfl/v05.py` uses **zero SDPA and zero Attention** (`torch.linalg.solve_triangular` for chunk-parallel `WY` and `torch.linalg.solve` for `M2` covariance projection in exact FP32/FP64). By contrast, legacy `rdfl/model.py` (`ZeroDiffRDFLAdapter`) was designed to wrap PyTorch SDPA windowed-attention backbones. Fusing `MicroDeltaProductGDNLayer`, `GDNReadOnlyLayer`, and `OrthogonalSlowGDN` into Triton / Flash Linear Attention (`fla`) kernels preserves the exact mathematical specification while removing Python chunk-loop overhead for large-scale GPU pretraining.

---

## 2. Legacy `v0.3` Reference Architecture (Windowed-Attention Hybrid + `FP32 SlotCore`)

> **Legacy Reference Scope:** The sections below document the **`v0.3` reference modules** (`rdfl/model.py`, `rdfl/layers.py`, `ZeroDiffRDFLAdapter`) used for the `1.20M` iso-parameter ablations and `55M` hybrid side-branch (`Diogenes-RDFL-55M`, `782/782` verified cells). For the current pure-recurrent `v0.5.0` architecture (`0` Attention, `0` SlotCore), see **Section 1** above and [`docs/V05_TECHNICAL_REPORT.md`](V05_TECHNICAL_REPORT.md).

### 2.1 Problem Statement and Design Principles (`v0.3` Hybrid Context)

Autoregressive sequence models face a structural trade-off between **context horizon**, **memory complexity**, and **algorithmic state tracking**:

- **Full Self-Attention** provides exact random access over $x_{1:t}$ at the cost of $O(T)$ decode memory: for a 6-layer backbone with $d = 768,\, H = 8\, (d_h = 96)$, FP16 KV-cache scales as `9.0 / 72.0 / 576.0 MiB` across $T \in \{512,\, 4{,}096,\, 32{,}768\}$.
- **Windowed Attention** bounds cache size to $O(M)$ but drops all state beyond window length $M$.
- **Pure Linear Recurrence** achieves $O(1)$ memory but requires Householder-range write gates ($\beta_t \in (0, 2)$) for negative-eigenvalue state transitions (`Parity-64`, `S5-Perm32`) and structured nonlinear / boundary feedback for multi-hop associative reasoning.

Legacy `v0.3` `RDFL` combined four mechanisms: (i) weight-tied recursive depth (with optional input anchoring; see Section 2.2.3), (ii) chunk-parallel Gated DeltaNet with Householder-range write gates (`1.7139 MiB` exact $O(1)$ decode state for $L=6,\, d=768,\, H=8$, **`336.1×`** smaller than Full-KV at $T = 32{,}768$), (iii) doubly-sparse associative gating (`BDH`), and (iv) chunk-boundary `FP32 SlotCore` memory (later superseded by `rlt2_s1`).

### 2.2 Exact Mathematical Formulation (`v0.3` Legacy Modules)

#### 2.2.1 Causal Input Trunk and Narrow Projection
Given a raw byte prefix $x_{1:t} \in \{0, \dots, 255\}^t$, a causal input trunk $E_\theta$ (either a pretrained windowed-attention backbone of width $d_{\text{base}}$ or a standalone embedding stem) produces $z_t^{\text{base}} = E_\theta(x_{1:t})$. A narrow bottleneck projection maps this stream into the recursive latent dimension $d$:

$$z_t^0 = W_{\downarrow}\,\mathrm{RMSNorm}(z_t^{\text{base}}) \in \mathbb{R}^d, \qquad e_t = \mathrm{RMSNorm}(z_t^0)$$

#### 2.2.2 Chunk-Boundary `FP32 SlotCore` Feedback and Symmetry-Collapse Diagnosis
The sequence is partitioned into causal chunks of length $B$ (e.g., $B = 64$). A bank of $S$ working-memory slots $M_{c-1} \in \mathbb{R}^{S \times d}$ (stored strictly in `float32` / `float64`) summarizes prior chunks $1, \dots, c-1$. Before chunk $c$ enters the recursive loop, its mean-pooled summary injects gated boundary feedback:

$$r_{c-1} = \mathrm{RMSNorm}\!\left(\frac{1}{S}\sum_{s=1}^S M_{c-1,s}\right), \qquad g_t = \sigma\!\left(W_g[z_t^0;\, r_{c-1}] + b_g\right), \qquad h_t^{(0)} = z_t^0 + \alpha\, g_t \odot W_m r_{c-1}$$

Strictly at chunk boundaries $t \equiv 0 \pmod B$, the `v0.2` slot bank executes both the residual `tanh` update (`slot_update`) and off-diagonal slot-to-slot self-attention **strictly in `float32` / `float64`** outside mixed-precision autocast (`rdfl/layers.py:Fp32SlotCoreBoundary.commit_boundary`), eliminating the `2/18` FP16 NaN failures observed in unguarded FP16 `A6_SLOT4_MESAJLI` (`0/36` NaN across unseen seeds):

$$\tilde{M}_c = M_{c-1} + 0.1\,\tanh\!\big(W_u[M_{c-1};\, \mathbf{1}_S (h_{\text{end}}^{(R)})^\top]\big), \qquad M_c = \tilde{M}_c + 0.15\,\mathrm{SlotAttn}_{\text{FP32}}(\tilde{M}_c)$$

- **Iso-Param `Byte LM` Trade-Off:** Under the `1.20M` iso-parameter constraint (`ffn` shrunk from `512` to `128`), `OY6_A6_SLOT4_FP32` trades `+0.0173 bpb` on `Byte LM` (`2.3819 ± 0.0055` vs `2.3646 ± 0.0017`) for a `37.3×` std reduction on 3-hop `graf_izli` (`68.09 ± 54.73% -> 96.89 ± 1.47%`). At `55M`, `FP32 SlotCore` sits externally on the narrow `d=384` adapter so the base `ffn=3072` stays unshrunk.
- **Symmetry Collapse & Zero `55M` State-Tracking Gain (`results/raw/slotcore_55m_cosine_probe.json`, `ana_55m_degerlendirme.json`, `rsi_55m_degerlendirme.json`):** Direct CPU probing of the trained `55M` checkpoint (step `21,000`, `16` real chunks / `1,024` bytes from `mix/validation.bin`) reveals that although `init_slots` begin with pairwise cosine similarity `0.5017–0.7512` (`mean = 0.6446`, `norm = 0.6941`), after `16` chunks all $S=4$ slots converge to pairwise cosine **`0.999982`** (`min = 0.999959`, `max = 0.999999`) with mean L2 norm **`664.5`**. Furthermore, full `55M` `SlotCore` ON/OFF ablation yields **`delta_durum = 0.0`** (`59.38%` vs `59.38%`) and `delta_bpb_2048 = -0.0005` (Ana) / `-0.0003` (RSI). Symmetric `FP32 SlotCore` is therefore **deferred / replaced** by `Fp32RLT2Boundary` (`rlt2_s1`) in `v0.3` and completely omitted in `v0.5.0`.

#### 2.2.3 Weight-Tied Recursive Core ($r = 1, \dots, R$) & Input-Anchoring Ablation
At each iteration $r \in \{1, \dots, R\}$, the shared core receives the previous hidden state combined with an optional input anchor $e_t$ and learned step embedding $e_r \in \mathbb{R}^d$:

$$\tilde{h}_t^{(r)} = \mathrm{RMSNorm}\!\left(h_t^{(r-1)} + \alpha_r\, e_t + e_r\right)$$

> *Empirical Ablation & Degree-Ceiling Notes (`results/raw/ablasyon_sonuclari.json`):* Controlled `1.20M` ablations show that `A6_R4_cipasiz` (unanchored) matches `A6_R4_cipali` on `Multi-Hop ICL` (`22.69 ± 1.76%` vs `22.72 ± 1.30%`) and slightly improves `Byte LM` (`2.2843` vs `2.2896 bpb`). Input anchoring is therefore retained only as a mild variance-regularizing option (`±1.30` vs `±1.76`). In addition, `Multi-Hop ICL` `22.72%` lies within the `1/4 = 25.0%` (`1/8 = 12.5%`, combined `18.75%`) structural degree-filtering ceiling (`KPN-Y01`).

1. **Chunk-Parallel Gated DeltaNet (`WY` Solver):** After a depthwise causal short convolution ($k=4$), L2-normalized queries/keys ($q_t, k_t$), multi-scale slow decay gates ($\alpha_t = \sigma(W_\alpha x_c + b_\alpha)$ with biases initialized across $[3.0, 6.9]$), and Householder-range write gates ($\beta_t = 2\sigma(W_\beta x_c) \in (0, 2)$) update the constant-size associative matrix state $S_t^{(r)} \in \mathbb{R}^{H \times d_h \times d_h}$:

$$S_t^{(r)} = \alpha_t S_{t-1}^{(r)} + \beta_t \left(v_t - \alpha_t S_{t-1}^{(r)} k_t\right) k_t^\top, \qquad y_t^{(r)} = W_o\!\left(\mathrm{RMSNorm}(S_t^{(r)} q_t) \odot \mathrm{SiLU}(W_g \tilde{h}_t^{(r)})\right)$$

2. **BDH Sparse Associative Interaction Gate & Standalone `SwiGLU` Sub-Block:**

$$u_t^{(r)} = h_t^{(r-1)} + y_t^{(r)} + W_{\text{dec}}\!\left(\mathrm{ReLU}(W_x \tilde{h}_t^{(r)}) \odot \mathrm{ReLU}(W_y y_t^{(r)})\right)$$

- In standalone **`RDFLModel`** (`rdfl/model.py`), each loop step appends a residual `SwiGLU` FFN sub-block: $h_t^{(r)} = u_t^{(r)} + \mathrm{SwiGLU}(\mathrm{RMSNorm}(u_t^{(r)}))$.
- In **`ZeroDiffRDFLAdapter`** (`rdfl/model.py`), the internal `SwiGLU` sub-block is omitted ($h_t^{(r)} = u_t^{(r)}$) because the pretrained backbone already supplies unshrunk FFN layers (`ffn = 3072`).

#### 2.2.4 Zero-Initialization Residual Fusion, Lossless Speculative Heads, and Bounded RSI
After the final iteration $R$, a zero-initialized projection $W_{\uparrow}$ ($W_{\uparrow}|_0 = 0$) merges the recursive refinement back into the trunk stream:

$$\Delta z_t = W_{\uparrow}\,\mathrm{RMSNorm}(h_t^{(R)}), \qquad z_t^{\text{fused}} = z_t^{\text{base}} + \Delta z_t$$

`ZeroDiffRDFLAdapter.forward(z_base)` returns $z_t^{\text{fused}} = z_t^{\text{base}} + \Delta z_t$ directly (`0.0` exact step-0 deviation). Attached capabilities include decoupled `K3` (`rdfl/speculative.py`, `0`-byte greedy mismatch across `288/288` prompts), `MTP-4` (`t+2..t+4` heads), calibrated `MSP + T*` (`rdfl/calibration.py`), and bounded `AZR` RSI regularized by exact per-token 3D reverse KL:

$$D_{\text{KL}}(\pi_g \,\|\, \pi_0) = \frac{1}{BT}\sum_{b=1}^B\sum_{t=1}^T \sum_{v=0}^{255} \pi_g(v \mid x_{b,<t}) \log \frac{\pi_g(v \mid x_{b,<t})}{\pi_0(v \mid x_{b,<t})}$$

### 2.3 Legacy `RDFL v0.3` Core Hardening (`K1–K5`) & Mechanistic Design Laws

The `v0.3` reference package (`rdfl/model.py`, `rdfl/layers.py`, `tests/test_rdfl.py`, `13/13` unit tests) resolves five structural bugs (`K1–K5`, [`results/raw/deney_a_k1_k5.json`](../results/raw/deney_a_k1_k5.json)) and incorporates 5-seed mechanistic findings ([`math_xray_master_synthesis.json`](../results/raw/math_xray_master_synthesis.json), [`p3_long_v03.json`](../results/raw/p3_long_v03.json), [`p2_long_v03.json`](../results/raw/p2_long_v03.json), [`h5_memory_latency_and_serving.json`](../results/raw/h5_memory_latency_and_serving.json)):

1. **`K1` (Per-Head FP32 `o_norm` & `alpha_max`):** Per-head `float32`-stabilized output normalization (`0` FP16 NaN; `+0.0134 bpb`, `p=0.098` within noise).
2. **`K2` (`rlt2_s1` Default Boundary & Mean/Last Pooling):** Replaces symmetric 4-slot `SlotCore` (`cos=0.999982`) with damped 1-slot `Fp32RLT2Boundary` (`boundary_mode="rlt2_s1"`).
3. **`K3` (`std=0.02` Initialization):** Eliminates step-0 logit explosion (`d=192: CE 193.82 -> 5.54`, `d=768: 752.45 -> 5.78`), cuts FP16 overflows (`8 -> 0`), and improves 500-step `Byte LM` (`mix_val`) by **`-0.5789 bpb`** (`p = 8e-5`).
4. **`K4` (Single `final_norm` & `mtp_stop_grad`):** Removes redundant double `final_norm` on `MTP-4` heads (`2.2023 -> 2.0638` at 150 steps).
5. **`K5` (`0`-Extra-Pass Snapshot Rollback):** `RDFLModel.forward_with_snapshots()` yields **`1.68×`** eager-mode wall-clock speedup (`0` extra passes, `0` byte mismatch).
6. **Mechanistic Laws & Falsified Hypotheses:**
   - **Genuine Memory & Householder Regime:** `mqar_k8` collapses `98.63% -> 6.25%` when $S_t=0$; `byte_lm` shows `BDH L0 = 90.5%`, `1.69×` post-whitespace decision-byte concentration, `P(beta > 1.0) = 32.2% / 33.4%`, and `0` rank collapse up to `T = 2048`.
   - **`BDH` Sparsity Tax (`P3_long`) & `conv.bias` Attention Sink (`P2_long`):** `P3 unit_rms` was falsified (`p=0.423`); iso-param `use_bdh=False` (`SwiGLU`) beats `BDH ReLU*ReLU` on `5/5` seeds (`-0.0121` / `-0.0150 bpb`, `p=0.0018`). `P2 conv.bias=0` was falsified (`+0.0085 bpb`, `p=0.0048`), confirming `short_conv.conv.bias` acts as a linear attention-sink channel (replaced by `BiasFreeCausalShortConv` in `v0.5.0`).
   - **`H5` Serving Wall:** Live `B=8` GPU `RoPE-Transformer` hits OOM at `T=65,536` while Pure `RDFL` stays `O(1)` at `153.77 MiB`; at `1M` text bytes (`4B/tok` `GQA-8 FP8`), `1B` Pure `RDFL` (`10.55 MiB`) is **`291.3×`** smaller (`3,072 MiB` KV).

---

## 3. References

1. **Gated DeltaNet:** Songlin Yang, Jan Kautz, and Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. [`arXiv:2412.06464`](https://arxiv.org/abs/2412.06464)
2. **DeltaProduct:** Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products.* 2025.
3. **RLT:** Yifan Zhang et al. *Recurrent Looped Transformer.* 2026. [`arXiv:2610.07591`](https://arxiv.org/abs/2610.07591)
4. **BDH & BDH-CQ:** Adrian Kosowski, Przemysław Uznański, Jan Chorowski, Zuzanna Stamirowska, and Michał Bartoszkiewicz. *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* 2025. [`arXiv:2509.26507`](https://arxiv.org/abs/2509.26507) · Björn Engdahl, Adrian Kosowski, Jan Chorowski, Zuzanna Stamirowska, Przemysław Uznański, Junlin Jiang, Rohan Phadke, Remigiusz Kinas, and Richard Zhong. *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* 2026. [`arXiv:2608.09888`](https://arxiv.org/abs/2608.09888)
5. **Global Workspace Theory (GWT):** Bernard J. Baars. *A Cognitive Theory of Consciousness.* Cambridge University Press, 1988.
6. **Orthogonal Weight Modification (OWM):** Guanxiong Zeng, Yang Chen, Bo Cui, and Shan Yu. *Continual learning of context-dependent processing in neural networks.* Nature Machine Intelligence, 2019.
7. **Elastic Weight Consolidation (EWC):** James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017.
8. **Slot Attention:** Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. *Object-Centric Learning with Slot Attention.* NeurIPS 2020. [`arXiv:2006.15055`](https://arxiv.org/abs/2006.15055)
9. **Medusa:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024. [`arXiv:2401.10774`](https://arxiv.org/abs/2401.10774)
10. **MTP:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction.* ICML 2024. [`arXiv:2404.19737`](https://arxiv.org/abs/2404.19737)
11. **AZR:** Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. *Absolute Zero: Reinforced Self-play Reasoning with Zero Data.* 2025. [`arXiv:2505.03335`](https://arxiv.org/abs/2505.03335)
12. **Dr. GRPO:** Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. *Understanding R1-Zero-Like Training: A Critical Perspective.* 2025. [`arXiv:2503.20783`](https://arxiv.org/abs/2503.20783)
13. **Titans:** Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. *Titans: Learning to Memorize at Test Time.* 2025. [`arXiv:2501.00663`](https://arxiv.org/abs/2501.00663)
14. **SEAL:** Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, and Pulkit Agrawal. *Self-Adapting Language Models.* 2025. [`arXiv:2506.10943`](https://arxiv.org/abs/2506.10943)
15. **MSP:** Dan Hendrycks and Kevin Gimpel. *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017. [`arXiv:1610.02136`](https://arxiv.org/abs/1610.02136)
16. **Temperature Scaling:** Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. [`arXiv:1706.04599`](https://arxiv.org/abs/1706.04599)
17. **Sleep Consolidation:** Ali Behrouz et al. *Language Models Need Sleep.* 2026. [`arXiv:2606.03979`](https://arxiv.org/abs/2606.03979)
