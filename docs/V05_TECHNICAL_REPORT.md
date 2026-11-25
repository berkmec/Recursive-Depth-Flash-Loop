# RDFL v0.5.0 Technical Report & Audited CPU Verification

**Unified `O(1)` Pure-Recurrent Byte Architecture (`RDFLv05Model`) — Zero Attention, Zero KV-Cache, Zero External BPE, Zero CUDA**

> **Transparency Scope:** All `v0.5.0` empirical metrics below were measured deterministically (`seed=42`, single-threaded CPU FP32/FP64) via [`benchmarks/verify_v05.py`](../benchmarks/verify_v05.py) and audited against [`results/v05_audited_summary.json`](../results/v05_audited_summary.json). Existing `v0.3` 55M GPU checkpoints (`150M` tokens) and `782/782` raw audited cells ([`results/audited_summary.json`](../results/audited_summary.json)) remain 100% preserved and separated from CPU-scale `v0.5.0` verification. A `v0.5-Reason 10M` GPU pretraining run is planned as the next stage.

---

## 1. Unified Architecture (`rdfl/v05.py`)

[`RDFLv05Model`](../rdfl/v05.py) (`565,216` parameters in default CPU config) unifies four architectural legs plus the `v0.5.0` non-collapsing reasoning upgrade inside one constant-memory (`25,136` bytes per `B=1` FP32 sequence across `T=16..320` and `R=1..1024`) architecture:

1. **Stage 0 — Byte Stem (`P2 + P3`):**
   - `256`-byte vocabulary embedding (`d_byte=64`) + causal `MSNgramHashStem` (`n ∈ {2,3,4,6}` hash residuals).
   - Dual-regime `MicroDeltaProductGDNLayer` (`n_h=4`, `4` micro-heads: `n_assoc=2` associative DeltaNet heads + `n_hh=2` orthogonalized Householder group-tracking heads with `BiasFreeCausalShortConv`).
2. **Stage 1 — `P=4` Macro Core + `StepReasoner` (`P1 + v0.5 Halting`):**
   - Fixed-shape `P=4` patch packing (`d_macro=64`) into `MacroRDFLv05Core` (`GDNReadOnlyLayer` writes loop `r=0` and performs zero-solver WY read-only queries for `r ≥ 1`, ensuring `S_macro(R=1) == S_macro(R=16)` bit-exact boundary invariance).
   - **v0.5 Reasoning & Halting Upgrade (`reasoning_mode="step_halting"`, backward-compatible `"adfp"` preserved):**
     - Replaces the contractive ADFP contraction bowl (`JVP ≤ 0.85`) with norm-bounded residual updates (`h_cand = rms_norm_tau(h + step_gate * delta_r, tau=1.0)`), loop-internal `O(1)` Delta-Householder scratchpad `W_r`, and a **learned halting gate** `g_step(h_q, o_halt) ∈ (0, 1)` (`h_{r+1} = h_r + g_step * (h_cand - h_r)`).
     - **Exact No-Op Guarantee:** When `gate_override == 0.0`, `h_{r+1} == h_r` (`0.0` diff) and `W_{r+1} == W_r` (`0.0` diff).
     - Unified single-token silent reasoning API directly on `RDFLv05Model`: `encode_for_reasoning` (uses causal 2-byte window over associative heads `[:n_assoc]`), `reason_from_state(..., eps_halt=...)`, `logits_from_reasoning_h`, and `forward_step_reasoning`.
3. **Stage 2–3 — Continual Learning (`P4`):**
   - Outside the P1 loop: causal token-shifted perplexity+MSP+loop-delta surprise gate feeding `OrthogonalSlowGDN` (`M2 OWM-GDN` with EMA key-covariance projector) + `ConsolidationManager` (`OGP + EWC + KL`, zero replay).
   - **`v0.4.1 F1` Fix:** `forward_stateful(..., return_aux=True)` returns strictly symmetric keys and shapes for both `pos == 0` and `pos > 0` (`max_diff = 4.72e-15`): patch-level `g_surp`, `delta_conv`, `r_used`, `g_step_last` with shape `[B, n_completed_patches]`, token-level `u_msp`, `nll_t` with shape `[B, T]`, and `logits_fast` with shape `[B, T, 256]`.
4. **Stage 4 — Single-Pass Byte Decoder + K3 + MTP-3:**
   - Single `dec_ffn` execution (`dec_ffn_calls == 1` even when `update_slow=True`), untied `head`, detached `K3` early-exit adapter (`0.0` core/head gradient tax), and `MTP-3` (`t+2..t+4`) lookahead heads.

---

## 2. Empirical Results (`results/v05_audited_summary.json`)

### 2.1 Section 1 — Mathematical Invariants & Parity

| Check | Measured Value | Target / Contract |
|---|---|---|
| `FP64 forward == step` (`R=2,4`, `slowOFF/ON`) | `5.27e-15 .. 5.72e-15` | `≤ 1e-13` |
| `FP32 argmax` mismatch (`R=2,4`, `slowOFF/ON`) | `0` across all `48` steps | `0` |
| Partial-patch resume (`37 + 11` FP64 diff) | `3.75e-15` (`s_slow`: `9.37e-17`) | `≤ 1e-13` |
| `gate_mode="random_matched"` FP64 diff | `5.38e-15` | `≤ 1e-13` |
| `eps_halt > 0` FP64 `forward == step` diff | `8.10e-15` | `≤ 1e-13` |
| `gate_override == 0.0` exact no-op (stream / reason) | `0.0` / `0.0` | `0.0` exact |
| `return_aux` `pos==0` vs `pos>0` symmetry | `keys_match=true`, `4.72e-15` | Symmetric |
| Causality (`update_slow ∈ {False, True}`) | `0.0` | `0.0` exact |
| Recurrent state size (`T=16..320`, `R=1..1024`) | `25,136` bytes (constant) | Strict `O(1)` |
| `R = 1024` finite (`0` NaN/Inf) | `true` (`h_norm_max=9.2451`, `w_norm_max=7.2491`) | Finite |
| `K3` core/head max grad (`adapter grad = 32.32`) | `0.0` | `0.0` exact |
| Single-pass `dec_ffn` calls (`slowOFF` / `slowON`) | `1` / `1` | `1` |

### 2.2 Section 2 — Silent Multi-Step Pointer Reasoning & Halting vs ADFP

Evaluated on variable-depth permutation pointer-chasing (`K ∈ {2,4,6,8}` ID, `K ∈ {10,12}` OOD, zero intermediate supervision) under an **equal `1,000`-step training budget** (`step_halting` `1,000` steps reaching `stage=4` vs `adfp` `1,000` steps remaining at `stage=1`):

| Reasoning Depth `K` (`R = K+2`) | `RDFLv05Model` (`step_halting`, `1000` steps) | `RDFLv05Model` (`adfp`, `1000` steps) |
|---|---|---|
| `K = 2` (ID, `R=4`) | **`1.0000`** | `0.3203` |
| `K = 4` (ID, `R=6`) | **`0.9922`** | `0.1562` |
| `K = 6` (ID, `R=8`) | **`1.0000`** | `0.1875` |
| `K = 8` (ID, `R=10`) | **`0.8516`** | `0.2344` |
| `K = 10` (OOD, `R=12`) | **`0.3906`** | `0.3125` |
| `K = 12` (OOD, `R=14`) | `0.3047` | `0.1328` |

- **Halting Gate & Stability Behavior (`K=4, 6` over `R=K+2, 16, 32, 64`):**
  - `step_halting` preserves bounded states across `R=1024` (`0` NaN/Inf, `mean_step_jvp_top_sv = 1.5621` vs contractive ADFP `0.7703`), and `g_step` decays monotonically during extra loops (`g_first=0.9978 → g_last=0.2881` at `K=4, R=64`; `g_first=0.9976 → g_last=0.3040` at `K=6, R=64`), triggering dynamic early exit (`r_used_mean = 12.08..22.66` at `eps_halt=0.40`).
  - **Empirical Caveat:** Because learned `g_step` decays gradually (`~0.30`) rather than snapping `< 0.05` immediately at step `K+3` in this 1,000-step CPU budget, running extra un-halted iterations past `K+2` degrades accuracy (`K=4: 0.9922 → 0.2969/0.2422/0.1641` full, `0.2188/0.1875/0.2031` with `eps_halt=0.40`; `K=6: 0.9922 → 0.3047/0.3281/0.2656`). Sharper threshold halting under longer multi-curriculum schedules remains an active engineering target.

### 2.3 Section 3 — Anthropic Global Workspace (`J-Space` / Jacobian Lens, Strict OOS `x_cal → x_te`)

All subspaces fit via Frobenius-normalized end-to-end Jacobian SVD on `x_cal` (`n=96`), evaluated on unseen `x_te` (`n=96`) at `K=6, R=8` (`step_halting` `clean_acc = 0.9896` vs equal-budget `adfp` null control `clean_acc = 0.1979`, `J/J_perp = 0.9984×`, `p = 0.380`, `flip_to_A = 0.0`, compositional `0.0521`):

- **Sequential Interruption Readout & Monotone Probes:** Intermediate hop values peak sequentially across silent steps: `v_1` at `r=3` (`1.0000`), `v_2` at `r=4` (`1.0000`), `v_3` at `r=5` (`1.0000`), `v_4` at `r=6` (`0.9896`), `v_5` at `r=7` (`1.0000`), `v_6` at `r=8` (`0.9896`). Held-out Ridge linear probes achieve **`5/5` monotone adjacent-step progression** (`peak_r = [1, 4, 5, 6, 7, 8]`, accuracies `0.9688..1.0000`).
- **OOS Variance Concentration Caveat (`var_J` vs High-Dim `<10%` Hypothesis):** In `d=64`, `J-space` (`k90 = 11..37`) captures `42.15% → 93.87%` of `h_r` variance — **unlike** a high-dimensional `<10%` regime — exceeding random subspace expectation (`23.44% → 57.81%`) and `OOS J_perp_topk` for `r ≥ 1`, while causal ablation/patching still cleanly isolate `J` from `J_perp`.
- **OOS Functional Global Broadcast (`r_mid=4, k=14`):** `J-space` mean `= 0.6226` vs `J_perp` `= 0.2364` (**`2.6335×`**, `p = 5.27e-15`) and random-`k` `= 0.3374` (**`1.8453×`**, `p = 7.80e-15`).
- **OOS Selective Causal Ablation (`r_mid=4, k=14`):** Ablating `J-space` collapses accuracy from `0.9896 → 0.1875` (`-0.8021`), whereas ablating full orthogonal `J_perp` (`50` dims) keeps `0.9688` and random-`k` keeps `0.9708`.
- **OOS Concept & Compositional Patching (`93/96` clean pairs):**
  - Same-table patching in `J-space`: **`100.0%` flip-to-A** (`0.0%` keep-B); in full `J_perp`: **`0.0%` flip-to-A** (`97.85%` keep-B).
  - **Cross-table compositional patching** ($P_B^4(v_2(A))$): `J-space` achieves **`0.9271`** vs `J_perp` **`0.1042`**.

### 2.4 Section 4 — 4-Problem Regression Lock (Same `RDFLv05Model` Class)

- **P1 (`dyck_stack(k=4, depth=4)`, chance `0.25`):** `R1=0.2604 → R16=0.2639`; strict `O(1)` macro boundary invariance `t80_s_macro_r1_vs_r16_diff = 0.0` (60 short smoke steps).
- **P2 (`s5_perm32`, `running=False`, strict OOD `L ≤ 8` → `L ∈ {2,4,8,16,32}`, chance `0.00833`):** `val_acc_l2 = 0.8438` (`101.3×`), `val_acc_l4 = 0.6641` (`79.7×`), `val_acc_l8 = 0.4844` (`58.1×`), OOD `val_acc_l16_ood = 0.1172` (`14.1×`), `val_acc_l32_ood = 0.0234` (`2.8×`); zeroing `s_fast` collapses back to `0.0078` (`l2` and `l8`).
- **P3 (Byte Efficiency, Synthetic UTF-8 Turkish+Math Corpus):** `dec_ffn_calls = 1`, hook FLOP/B `250,816` (`slowOFF`) / `302,272` (`slowON`), shift robustness `bpb_shift_0 = 0.6933` (`Δ+1 = +0.0047`, `Δ+17 = +0.0036`).
- **P4 (Continual Learning, `0` Replay, chance `0.125`):**
  - `ACC`: `rdfl_v05_cl3_slow_m2 = 0.2517` vs `frozen = 0.1719` vs `naive_adamw = 0.0851`; `BWT = -0.0625` (vs `-0.4167` naive), `PLAST_new = 0.1797`.
  - Streaming surprise gate: `g_surp(familiar) = 0.3498` vs `g_surp(novel) = 0.7522`, `g_surp(256B distractor) = 0.1335`; post-distractor `S_slow` cosine preservation **`cos_surp = 0.5268 > cos_rand = 0.4380 > cos_one = 0.3085`**.
  - Adaptive Space/Boundary Gate (`M_fast(hh)` spectral radius `|λ|`): `copy_l16` (`digit=0.9959, space=0.9949, P=4/byte=0.9440`) vs `shuf16_bag` (`digit=0.9953, space=0.9911, P=4/byte=0.8643`).

---

## 3. Reproducibility & Verification Commands

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. pytest -q
ruff check rdfl benchmarks tests scripts
mypy --strict rdfl/v05.py benchmarks/verify_v05.py rdfl/__init__.py tests/test_rdfl_v05.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/verify_v05.py --out results/v05_audited_summary.json
```
