# Architectural Specification: Recursive Depth Flash Loop (`RDFL`)

![RDFL Architecture](diagrams/rdfl_architecture.svg)

---

## 1. Problem Statement and Design Principles

Autoregressive sequence models face a structural trade-off between **context horizon**, **memory complexity**, and **algorithmic state tracking**:

- **Full Self-Attention** provides exact random access over $x_{1:t}$ at the cost of $O(T)$ decode memory: for a 6-layer backbone with $d = 768,\, H = 8\, (d_h = 96)$, FP16 KV-cache scales as `9.0 / 72.0 / 576.0 MiB` across $T \in \{512,\, 4{,}096,\, 32{,}768\}$.
- **Windowed Attention** bounds cache size to $O(M)$ but drops all state beyond window length $M$.
- **Pure Linear Recurrence** achieves $O(1)$ memory but requires Householder-range write gates ($\beta_t \in (0, 2)$) for negative-eigenvalue state transitions (`Parity-64`, `S5-Perm32`) and structured nonlinear / boundary feedback for multi-hop associative reasoning.

**`RDFL`** combines four mechanisms: (i) weight-tied recursive depth (with optional input anchoring; see Section 2.3 ablation note), (ii) chunk-parallel Gated DeltaNet with Householder-range write gates (`1.7139 MiB` exact $O(1)$ decode state for $L=6,\, d=768,\, H=8$, **`336.1×`** smaller than Full-KV at $T = 32{,}768$), (iii) doubly-sparse associative gating (`BDH`), and (iv) chunk-boundary `FP32 SlotCore` memory.

---

## 2. Exact Mathematical Formulation

### 2.1 Causal Input Trunk and Narrow Projection
Given a raw byte prefix $x_{1:t} \in \{0, \dots, 255\}^t$, a causal input trunk $E_\theta$ (either a pretrained windowed-attention backbone of width $d_{\text{base}}$ or a standalone embedding stem) produces $z_t^{\text{base}} = E_\theta(x_{1:t})$. A narrow bottleneck projection maps this stream into the recursive latent dimension $d$:

$$z_t^0 = W_{\downarrow}\,\mathrm{RMSNorm}(z_t^{\text{base}}) \in \mathbb{R}^d, \qquad e_t = \mathrm{RMSNorm}(z_t^0)$$

### 2.2 Chunk-Boundary `FP32 SlotCore` Feedback and Symmetry-Collapse Diagnosis
The sequence is partitioned into causal chunks of length $B$ (e.g., $B = 64$). A bank of $S$ working-memory slots $M_{c-1} \in \mathbb{R}^{S \times d}$ (stored strictly in `float32` / `float64`) summarizes prior chunks $1, \dots, c-1$. Before chunk $c$ enters the recursive loop, its mean-pooled summary injects gated boundary feedback:

$$r_{c-1} = \mathrm{RMSNorm}\!\left(\frac{1}{S}\sum_{s=1}^S M_{c-1,s}\right), \qquad g_t = \sigma\!\left(W_g[z_t^0;\, r_{c-1}] + b_g\right), \qquad h_t^{(0)} = z_t^0 + \alpha\, g_t \odot W_m r_{c-1}$$

Strictly at chunk boundaries $t \equiv 0 \pmod B$, the `v0.2` slot bank executes both the residual `tanh` update (`slot_update`) and off-diagonal slot-to-slot self-attention **strictly in `float32` / `float64`** outside mixed-precision autocast (`rdfl/layers.py:Fp32SlotCoreBoundary.commit_boundary`), eliminating the `2/18` FP16 NaN failures observed in unguarded FP16 `A6_SLOT4_MESAJLI` (`0/36` NaN across unseen seeds):

$$\tilde{M}_c = M_{c-1} + 0.1\,\tanh\!\big(W_u[M_{c-1};\, \mathbf{1}_S (h_{\text{end}}^{(R)})^\top]\big), \qquad M_c = \tilde{M}_c + 0.15\,\mathrm{SlotAttn}_{\text{FP32}}(\tilde{M}_c)$$

#### Mechanistic Diagnosis (`v0.2` Symmetry Collapse), Iso-Param Tax & Planned `v0.3` Fix
- **Iso-Param `Byte LM` Trade-Off:** Under the `1.20M` iso-parameter constraint (`ffn` shrunk from `512` to `128`), `OY6_A6_SLOT4_FP32` trades `+0.0173 bpb` on `Byte LM` (`2.3819 ± 0.0055` vs `2.3646 ± 0.0017`) for a `37.3×` std reduction on 3-hop `graf_izli` (`68.09 ± 54.73% -> 96.89 ± 1.47%`). At `55M`, `FP32 SlotCore` sits externally on the narrow `d=384` adapter so the base `ffn=3072` stays unshrunk.
- **Symmetry Collapse & Zero `55M` State-Tracking Gain (`results/raw/slotcore_55m_cosine_probe.json`, `ana_55m_degerlendirme.json`, `rsi_55m_degerlendirme.json`):** Direct CPU probing of the trained `55M` checkpoint (step `21,000`, `16` real chunks / `1,024` bytes from `mix/validation.bin`) reveals that although `init_slots` begin with pairwise cosine similarity `0.5017–0.7512` (`mean = 0.6446`, `norm = 0.6941`), after `16` chunks all $S=4$ slots converge to pairwise cosine **`0.999982`** (`min = 0.999959`, `max = 0.999999`) with mean L2 norm **`664.5`**. Furthermore, full `55M` `SlotCore` ON/OFF ablation yields **`delta_durum = 0.0`** (`59.38%` vs `59.38%`) and `delta_bpb_2048 = -0.0005` (Ana) / `-0.0003` (RSI). Symmetric `FP32 SlotCore` is therefore **deferred / replaced in `v0.3`** rather than retained as an active component.
- **Root Cause:** Both the uniform broadcast $\mathbf{1}_S (h_{\text{end}}^{(R)})^\top$ into shared $W_u$ and the undamped additive recurrence $M_c = M_{c-1} + \Delta_c$ lack slot-axis competition and contraction. Consequently, `v0.2`'s empirical gains arise from an **accumulative FP32 chunk-boundary residual state** rather than `4` specialized independent slots.
- **Future (`v0.4+`) Multi-Slot `SlotAttention` Extension:** Replace uniform broadcast with competitive **slot-axis `SlotAttention`** (Locatello et al., 2020: attention weights softmax-normalized across the slot dimension $S$ over chunk keys/values) coupled with per-slot **`RMSNorm`** and learned contraction $\gamma \in (0, 1)$ (while `v0.3` ships `Fp32RLT2Boundary` / `rlt2_s1` as the default 1-slot boundary):

$$A_c = \mathrm{softmax}_{\text{slot}}\!\left(\frac{Q(M_{c-1})\, K(H_c)^\top}{\sqrt{d}}\right), \qquad M_c = \mathrm{RMSNorm}\!\Big((1 - \gamma) \odot M_{c-1} + \gamma \odot \mathrm{GRU/MLP}\big(M_{c-1},\, \bar{A}_c V(H_c)\big)\Big)$$

### 2.3 Weight-Tied Recursive Core ($r = 1, \dots, R$) & Input-Anchoring Ablation
At each iteration $r \in \{1, \dots, R\}$, the shared core receives the previous hidden state combined with an optional input anchor $e_t$ and learned step embedding $e_r \in \mathbb{R}^d$:

$$\tilde{h}_t^{(r)} = \mathrm{RMSNorm}\!\left(h_t^{(r-1)} + \alpha_r\, e_t + e_r\right)$$

> *Empirical Ablation & Degree-Ceiling Notes (`results/raw/ablasyon_sonuclari.json`):* Controlled `1.20M` ablations show that `A6_R4_cipasiz` (unanchored) matches `A6_R4_cipali` on `Multi-Hop ICL` (`22.69 ± 1.76%` vs `22.72 ± 1.30%`) and slightly improves `Byte LM` (`2.2843` vs `2.2896 bpb`), while `A5_R4_cipasiz` beats `A5_R4_cipali` on both (`3.71%` vs `3.29%`; `2.4379` vs `2.4676 bpb`). Input anchoring is therefore retained only as a mild variance-regularizing option (`±1.30` vs `±1.76`). In addition, `Multi-Hop ICL` `22.72%` lies within the `1/4 = 25.0%` (`1/8 = 12.5%`, combined `18.75%`) structural degree-filtering ceiling (`KPN-Y01`), reflecting in-context structural node-role tracking rather than unbounded multi-hop graph traversal.

1. **Chunk-Parallel Gated DeltaNet (`WY` Solver):** After a depthwise causal short convolution ($k=4$), L2-normalized queries/keys ($q_t, k_t$), multi-scale slow decay gates ($\alpha_t = \sigma(W_\alpha x_c + b_\alpha)$ with biases initialized across $[3.0, 6.9]$, corresponding to $\alpha_0 \in [0.953, 0.999]$), and Householder-range write gates ($\beta_t = 2\sigma(W_\beta x_c) \in (0, 2)$) update the constant-size associative matrix state $S_t^{(r)} \in \mathbb{R}^{H \times d_h \times d_h}$:

$$S_t^{(r)} = \alpha_t S_{t-1}^{(r)} + \beta_t \left(v_t - \alpha_t S_{t-1}^{(r)} k_t\right) k_t^\top, \qquad y_t^{(r)} = W_o\!\left(\mathrm{RMSNorm}(S_t^{(r)} q_t) \odot \mathrm{SiLU}(W_g \tilde{h}_t^{(r)})\right)$$

Allowing $\beta_t \in (0, 2)$ permits negative eigenvalues in $(I - \beta_t k_t k_t^\top)$, which is algebraically necessary to execute bit-flips (`Parity-64`) and odd permutations (`S5-Perm32`). Within each chunk of length $B$, the recurrence is computed in parallel via unitriangular lower-triangular solve (`WY` representation), achieving exact equivalence with single-byte `step()` execution (`4.44e-16` max difference in FP64).

2. **BDH Sparse Associative Interaction Gate & Standalone `SwiGLU` Sub-Block:**

$$u_t^{(r)} = h_t^{(r-1)} + y_t^{(r)} + W_{\text{dec}}\!\left(\mathrm{ReLU}(W_x \tilde{h}_t^{(r)}) \odot \mathrm{ReLU}(W_y y_t^{(r)})\right)$$

- In standalone **`RDFLModel`** (`rdfl/model.py`), each loop step appends a residual `SwiGLU` FFN sub-block: $h_t^{(r)} = u_t^{(r)} + \mathrm{SwiGLU}(\mathrm{RMSNorm}(u_t^{(r)}))$.
- In **`ZeroDiffRDFLAdapter`** (`rdfl/model.py`, illustrated in Figure 1 for Path-1 warm adaptation), the internal `SwiGLU` sub-block is omitted ($h_t^{(r)} = u_t^{(r)}$) because the pretrained backbone already supplies unshrunk FFN layers (`ffn = 3072`).

### 2.4 Zero-Initialization Residual Fusion, Lossless Speculative Heads, and Bounded RSI
After the final iteration $R$, a zero-initialized projection $W_{\uparrow}$ ($W_{\uparrow}|_0 = 0$) merges the recursive refinement back into the trunk stream:

$$\Delta z_t = W_{\uparrow}\,\mathrm{RMSNorm}(h_t^{(R)}), \qquad z_t^{\text{fused}} = z_t^{\text{base}} + \Delta z_t$$

`ZeroDiffRDFLAdapter.forward(z_base)` returns $z_t^{\text{fused}} = z_t^{\text{base}} + \Delta z_t$ directly; the outer pretrained backbone's final `RMSNorm` and readout projection then produce $\hat{z}_t = \mathrm{RMSNorm}(z_t^{\text{fused}})$ and $p_\Theta(x_{t+1} \mid x_{1:t}) = \mathrm{softmax}(W_{\text{out}}\hat{z}_t)$. Because $\Delta z_t \equiv 0$ at step 0, warm adaptation preserves `100%` of the pretrained backbone and external episodic memory (`cmem`) behavior on initialization (`0.0` exact deviation).

- **Decoupled Early-Exit (`K3`) & Lossless Speculative Decode:** Applies a lightweight bottleneck adapter to $\mathrm{sg}(h_t^{(1)})$ with frozen readout weights (`0.000` gradient interference on the main $R$-loop). During decoding ([`rdfl/speculative.py`](../rdfl/speculative.py)), $k$ draft bytes are proposed using only $r=1$, then verified in one chunk-parallel $R$-loop call with exact state rollback on mismatch, guaranteeing **`0`-byte deviation** from greedy $R$-loop decoding (`288/288` prompts audited).
- **Multi-Byte Lookahead (`MTP-4`) & `55M` Causal Controls (`results/raw/g_base_only_kontrol.json`):** Attaches $3$ zero-initialized residual bottlenecks to $\hat{z}_t$ predicting $x_{t+2}, x_{t+3}, x_{t+4}$ with zero draft forward passes during speculative verification ([`speculative_decode_mtp4`](../rdfl/speculative.py)). At `55M`, the `1,500`-step `G0` vs. `G1` gate isolates `MTP-4` (`+0.0010 bpb` LM, `-0.0005 bpb` SFT), while the completed RDFL-off `G_BASE_ONLY` (`1,500`-step) control achieves composite `1.1157` (`3.3×` faster than `G0: 1.1158` / `G1: 1.1167`) and `100%` `<calc>` `ToolRunner` using only the `55M` core + reader (`maks|delta_z| == 0.0`). On the best `21K` checkpoint (`en_iyi.pt`), zeroing `delta_z` yields negligible pure-LM `bpb` deltas (`-0.0010 / -0.0002 / 0.0000`) and `0.0` state-tracking delta (`59.38%`), while sharply dropping co-adapted generation (`completion 0.78 -> 0.34`, `repetition 0.40 -> 0.78`, `OY-10 1.00 -> 0.5417`, `real_qa 13.28% -> 10.94%`).
- **Calibrated Uncertainty (`MSP + T*`):** Uses scalar temperature scaling $T^*$ fitted on a held-out split to produce selective abstention uncertainty $u_t = 1 - \max_v p_\Theta(v \mid x_{1:t}; T^*)$ with zero additional parameters (evaluated via mid-rank tie-exact `auroc()` in `rdfl/calibration.py`).
- **Bounded Verifier-Gated RSI (`ÖY-7`):** Optimizes policy updates strictly over externally verified traces regularized by natural-language replay, a permanent `Tur 0` in-distribution anchor, a validation rollback gate, and exact per-token 3D reverse KL (`55M` 8-gen RSI in `oy7_rsi_55m.json` showed `0` rollbacks but revealed a hard `L=4 <-> L=5` wall and SFT completion regression `0.78 -> 0.58` without SFT/`EOT` replay, motivating SFT-guarded consolidation akin to Behrouz et al., 2026):

$$D_{\text{KL}}(\pi_g \,\|\, \pi_0) = \frac{1}{BT}\sum_{b=1}^B\sum_{t=1}^T \sum_{v=0}^{255} \pi_g(v \mid x_{b,<t}) \log \frac{\pi_g(v \mid x_{b,<t})}{\pi_0(v \mid x_{b,<t})}$$

---

## 3. `RDFL v0.3` Core Hardening (`K1–K5`) & Mechanistic Design Laws

The `v0.3` reference package (`rdfl/`, `tests/test_rdfl.py`, `13/13` unit tests) resolves five structural bugs (`K1–K5`, [`results/raw/deney_a_k1_k5.json`](../results/raw/deney_a_k1_k5.json)) and incorporates 5-seed mechanistic findings ([`math_xray_master_synthesis.json`](../results/raw/math_xray_master_synthesis.json), [`p3_long_v03.json`](../results/raw/p3_long_v03.json), [`p2_long_v03.json`](../results/raw/p2_long_v03.json), [`h5_memory_latency_and_serving.json`](../results/raw/h5_memory_latency_and_serving.json)):

1. **`K1` (Per-Head FP32 `o_norm` & `alpha_max`):** Replaces global `o_norm` with per-head `float32`-stabilized output normalization (`self.o_norm = RMSNorm(self.head_dim)` computed in `float32`/`float64` in `rdfl/layers.py`) to prevent slow/fast head norm imbalance (`0` FP16 NaN; `+0.0134 bpb`, `p=0.098` within noise).
2. **`K2` (`rlt2_s1` Default Boundary & Mean/Last Pooling):** Replaces symmetric 4-slot `SlotCore` (`cos=0.999982` collapse) with damped 1-slot `Fp32RLT2Boundary` (`boundary_mode="rlt2_s1"`) as the default boundary module, preserving `slotcore_s4` as an explicit ablation mode.
3. **`K3` (`std=0.02` Initialization):** Initializes tied embeddings, linear weights, and `step_emb` at `std=0.02`, eliminating step-0 logit explosion (`d=192: CE 193.82 -> 5.54`, `d=768: 752.45 -> 5.78`), reducing FP16 overflows (`8 -> 0`), and improving 500-step `Byte LM` (`mix_val`) by **`-0.5789 bpb`** (`p = 8e-5`).
4. **`K4` (Single `final_norm` & `mtp_stop_grad`):** Removes redundant double `final_norm` on `MTP-4` heads and adds `mtp_stop_grad=True` to eliminate main-head CE interference (`2.2023 -> 2.0638` at 150 steps).
5. **`K5` (`0`-Extra-Pass Snapshot Rollback):** `RDFLModel.forward_with_snapshots()` records per-token streaming states during verification, yielding **`1.68×`** eager-mode wall-clock speedup (`0` extra passes, `0` byte mismatch).
6. **Mechanistic Laws & Falsified Hypotheses:**
   - **Genuine Memory & Householder Regime:** `mqar_k8` collapses `98.63% -> 6.25%` when $S_t=0$; `byte_lm` shows `BDH L0 = 90.5%`, `1.69×` post-whitespace decision-byte concentration, `P(beta > 1.0) = 32.2% / 33.4%`, and `0` rank collapse up to `T = 2048`.
   - **`BDH` Sparsity Tax (`P3_long`) & `conv.bias` Attention Sink (`P2_long`):** `P3 unit_rms` was falsified (`p=0.423`); iso-param `use_bdh=False` (`SwiGLU`) beats `BDH ReLU*ReLU` on `5/5` seeds (`-0.0121` / `-0.0150 bpb`, `p=0.0018`). `P2 conv.bias=0` was falsified (`+0.0085 bpb` regression, `p=0.0048`), confirming `short_conv.conv.bias` acts as a linear attention-sink channel.
   - **`H5` Serving Wall:** Live `B=8` GPU `RoPE-Transformer` hits OOM at `T=65,536` while Pure `RDFL` stays `O(1)` at `153.77 MiB`; at `1M` text bytes (`4B/tok` `GQA-8 FP8`), `1B` Pure `RDFL` (`10.55 MiB`) is **`291.3×`** smaller (`3,072 MiB` KV).

---

## 4. Kernel Implementation Note

The reference PyTorch implementation executes chunk-parallel `WY` updates in exact FP32/FP64 via `torch.linalg.solve_triangular` alongside PyTorch SDPA. Fusing the chunk-parallel `GatedDeltaNet` recurrence into a Triton / Flash Linear Attention (`fla`) kernel leaves the mathematical specification unchanged while substantially accelerating large-scale from-scratch training.

---

## 5. References

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
13. **Sleep Consolidation:** Ali Behrouz et al. *Language Models Need Sleep.* 2026. [`arXiv:2606.03979`](https://arxiv.org/abs/2606.03979)
