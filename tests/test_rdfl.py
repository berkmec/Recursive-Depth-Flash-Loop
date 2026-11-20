"""Unit tests verifying RDFL mathematical invariants on CPU."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from rdfl import (
    BottleneckAdapter,
    GatedDeltaNetLayer,
    RDFLConfig,
    RDFLModel,
    RMSNorm,
    ZeroDiffRDFLAdapter,
    auroc,
    ece_15,
    fit_temperature,
    greedy_decode,
    reverse_kl_per_token,
    speculative_decode_k3,
    speculative_decode_mtp4,
)
from rdfl.layers import Fp32RLT2Boundary


class TestRDFL(unittest.TestCase):
    def test_01_chunk_parallel_equals_step_fp64(self) -> None:
        torch.manual_seed(7)
        cfg = RDFLConfig(d_model=48, num_heads=4, ffn_dim=96, num_loops=2, chunk_size=8)
        model = RDFLModel(cfg).double().eval()
        x = torch.randint(0, 256, (1, 20))
        with torch.no_grad():
            full_logits = model(x)
            state = None
            steps = []
            for t in range(x.shape[1]):
                lt, state = model.step(x[:, t : t + 1], state)
                steps.append(lt)
            step_logits = torch.cat(steps, dim=1)
        self.assertLess(float((full_logits - step_logits).abs().max().item()), 1e-12)

    def test_02_strict_causality(self) -> None:
        torch.manual_seed(11)
        cfg = RDFLConfig(d_model=48, num_heads=4, ffn_dim=96, num_loops=2, chunk_size=8)
        model = RDFLModel(cfg).eval()
        x = torch.randint(0, 256, (1, 24))
        x_mod = x.clone()
        x_mod[0, 13] = (x_mod[0, 13] + 17) % 256
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x_mod)
        self.assertEqual(float((y1[:, :13] - y2[:, :13]).abs().max().item()), 0.0)

    def test_03_zero_diff_adapter_identity(self) -> None:
        adapter = ZeroDiffRDFLAdapter(d_base=64, d_narrow=32, num_heads=4, chunk_size=8)
        z = torch.randn(2, 20, 64)
        with torch.no_grad():
            out = adapter(z)
        self.assertEqual(float((out - z).abs().max().item()), 0.0)

    def test_04_k3_decoupled_zero_core_gradient(self) -> None:
        cfg = RDFLConfig(d_model=48, num_heads=4, ffn_dim=96, num_loops=2, chunk_size=8)
        model = RDFLModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        _, logits_r1, _, _ = model.forward_stateful(x)
        F.cross_entropy(logits_r1.reshape(-1, 256), y.reshape(-1)).backward()  # type: ignore[no-untyped-call]
        self.assertIsNone(model.gdn.qkv.weight.grad)
        self.assertIsNone(model.emb.weight.grad)
        k3_grad = model.k3_adapter.up.weight.grad
        assert k3_grad is not None
        self.assertGreater(float(k3_grad.abs().sum().item()), 0.0)

    def test_05_mtp4_propagation_vs_stop_grad(self) -> None:
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        m_full = RDFLModel(RDFLConfig(d_model=48, num_heads=4, ffn_dim=96, mtp_stop_grad=False))
        _, _, h_fin, _ = m_full.forward_stateful(x)
        mtp_lg = m_full.mtp_logits(h_fin)
        F.cross_entropy(mtp_lg[0][:, :-1].reshape(-1, 256), y[:, 1:].reshape(-1)).backward()  # type: ignore[no-untyped-call]
        qkv_grad = m_full.gdn.qkv.weight.grad
        assert qkv_grad is not None
        self.assertGreater(float(qkv_grad.abs().sum().item()), 0.0)

        m_stop = RDFLModel(RDFLConfig(d_model=48, num_heads=4, ffn_dim=96, mtp_stop_grad=True))
        _, _, h_fin2, _ = m_stop.forward_stateful(x)
        mtp_lg2 = m_stop.mtp_logits(h_fin2)
        F.cross_entropy(mtp_lg2[0][:, :-1].reshape(-1, 256), y[:, 1:].reshape(-1)).backward()  # type: ignore[no-untyped-call]
        self.assertIsNone(m_stop.gdn.qkv.weight.grad)
        head0 = m_stop.mtp_heads[0]
        assert isinstance(head0, BottleneckAdapter)
        head_grad = head0.up.weight.grad
        assert head_grad is not None
        self.assertGreater(float(head_grad.abs().sum().item()), 0.0)

    def test_06_calibration_and_exact_3d_reverse_kl(self) -> None:
        gen = torch.Generator().manual_seed(42)
        lg = torch.randn(4, 32, 256, generator=gen)
        ref = torch.randn(4, 32, 256, generator=gen)
        expected = (
            F.softmax(lg, dim=-1) * (F.log_softmax(lg, dim=-1) - F.log_softmax(ref, dim=-1))
        ).sum(dim=-1).mean()
        self.assertAlmostEqual(float(reverse_kl_per_token(lg, ref)), float(expected), places=6)
        self.assertAlmostEqual(float(reverse_kl_per_token(lg, lg)), 0.0, places=6)
        t_star = fit_temperature(lg, torch.randint(0, 256, (4, 32), generator=gen))
        self.assertTrue(0.2 <= t_star <= 8.0)
        self.assertEqual(auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])), 1.0)
        self.assertEqual(auroc(np.array([0.5, 0.5]), np.array([1, 0])), 0.5)
        self.assertEqual(auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0])), 0.5)
        self.assertAlmostEqual(
            auroc(np.array([0.9, 0.5, 0.5, 0.1]), np.array([1, 1, 0, 0])), 0.875, places=6
        )
        self.assertLess(ece_15(np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 1, 0, 0])), 0.15)

    def test_07_lossless_speculative_k3_and_mtp4(self) -> None:
        torch.manual_seed(19)
        cfg = RDFLConfig(
            d_model=48, num_heads=4, ffn_dim=96, num_loops=2, chunk_size=8, num_slots=4
        )
        model = RDFLModel(cfg).double().eval()
        prompt = torch.randint(0, 256, (1, 6))
        ref = greedy_decode(model, prompt, max_new_tokens=18)

        for draft_k in (1, 2, 4):
            res_k3 = speculative_decode_k3(model, prompt, max_new_tokens=18, draft_k=draft_k)
            self.assertEqual(int((res_k3.tokens != ref).sum().item()), 0)
            self.assertGreater(res_k3.verify_calls, 0)
            self.assertGreaterEqual(res_k3.tokens_per_verify, 1.0)
            self.assertEqual(res_k3.extra_passes, 0)

        res_mtp = speculative_decode_mtp4(model, prompt, max_new_tokens=18)
        self.assertEqual(int((res_mtp.tokens != ref).sum().item()), 0)
        self.assertGreater(res_mtp.verify_calls, 0)
        self.assertGreater(res_mtp.proposed_drafts, 0)
        self.assertEqual(res_mtp.extra_passes, 0)

        cfg_r1 = RDFLConfig(
            d_model=48, num_heads=4, ffn_dim=96, num_loops=1, chunk_size=8, num_slots=0
        )
        m_r1 = RDFLModel(cfg_r1).double().eval()
        ref_r1 = greedy_decode(m_r1, prompt, max_new_tokens=12)
        res_r1 = speculative_decode_k3(m_r1, prompt, max_new_tokens=12, draft_k=3)
        self.assertEqual(int((res_r1.tokens != ref_r1).sum().item()), 0)
        self.assertEqual(res_r1.rollbacks, 0)
        self.assertEqual(res_r1.extra_passes, 0)
        self.assertEqual(res_r1.acceptance_rate, 1.0)

    def test_08_fp32_slotcore_lock_under_fp16_and_autocast(self) -> None:
        torch.manual_seed(23)
        cfg = RDFLConfig(
            d_model=48, num_heads=4, ffn_dim=96, num_loops=2, chunk_size=8, num_slots=4
        )
        x = torch.randint(0, 256, (1, 32))
        m_fp16 = RDFLModel(cfg).half().eval()
        with torch.no_grad():
            logits_16, _, _, st_16 = m_fp16.forward_stateful(x)
        self.assertEqual(st_16.slot_state.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(st_16.slot_state).all().item()))
        self.assertTrue(bool(torch.isfinite(logits_16).all().item()))

        m_bf16 = RDFLModel(cfg).eval()
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits_bf, _, _, st_bf = m_bf16.forward_stateful(x)
        self.assertEqual(st_bf.slot_state.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(st_bf.slot_state).all().item()))
        self.assertTrue(bool(torch.isfinite(logits_bf).all().item()))

    def test_09_k1_per_head_fp32_o_norm_and_alpha_max(self) -> None:
        # RMSNorm FP32 upcast prevents FP16 overflow for |x| > 255.9
        norm = RMSNorm(16).half()
        big_fp16 = torch.full((2, 4, 16), 400.0, dtype=torch.float16)
        with torch.no_grad():
            norm_out = norm(big_fp16)
        self.assertEqual(norm_out.dtype, torch.float16)
        self.assertTrue(bool(torch.isfinite(norm_out).all().item()))
        self.assertAlmostEqual(float(norm_out.float().mean().item()), 1.0, places=2)

        # Per-head o_norm shape and alpha_max support
        gdn_def = GatedDeltaNetLayer(48, 4, chunk_size=8)
        self.assertEqual(tuple(gdn_def.o_norm.weight.shape), (12,))
        alpha_def = torch.sigmoid(gdn_def.gate_proj.bias[:4])
        self.assertAlmostEqual(float(alpha_def[-1].item()), float(torch.sigmoid(torch.tensor(6.9))), places=5)

        gdn_ext = GatedDeltaNetLayer(48, 4, chunk_size=8, alpha_max=0.9999)
        alpha_ext = torch.sigmoid(gdn_ext.gate_proj.bias[:4])
        self.assertAlmostEqual(float(alpha_ext[-1].item()), 0.9999, places=5)

        # Verify per-head normalization runs in FP32 even when FP16 input has large head variance
        gdn_fp16 = GatedDeltaNetLayer(32, 4, chunk_size=8, alpha_max=0.9999).half().eval()
        with torch.no_grad():
            gdn_fp16.out_proj.weight.copy_(torch.eye(32, dtype=torch.float16))
            gdn_fp16.g_proj.weight.zero_()
            x = torch.randn(1, 16, 32, dtype=torch.float16)
            y, _ = gdn_fp16(x)
        self.assertTrue(bool(torch.isfinite(y).all().item()))

    def test_10_k2_rlt2_s1_boundary_and_pooling(self) -> None:
        torch.manual_seed(31)
        # num_slots=0 no longer deletes RLT-2 boundary under default boundary_mode="rlt2_s1"
        cfg_s1 = RDFLConfig(
            d_model=48, num_heads=4, ffn_dim=96, num_loops=2, chunk_size=8, num_slots=0
        )
        m_s1 = RDFLModel(cfg_s1).double().eval()
        self.assertIsInstance(m_s1.boundary, Fp32RLT2Boundary)

        cfg_none = RDFLConfig(
            d_model=48, num_heads=4, ffn_dim=96, num_loops=2, chunk_size=8, boundary_mode="none"
        )
        self.assertIsNone(RDFLModel(cfg_none).boundary)

        # Damped RLT-2 S=1 state stays bounded across 100 boundary commits
        assert isinstance(m_s1.boundary, Fp32RLT2Boundary)
        st = m_s1.boundary.initial_state(2, dtype=torch.float32)
        for _ in range(100):
            st = m_s1.boundary.commit_boundary(st, torch.full((2, 48), 50.0))
        self.assertLess(float(st.abs().max().item()), 1.05)

        # Both boundary_pool="last" and "mean" preserve exact forward()==step() and causality
        x = torch.randint(0, 256, (1, 20))
        for pool in ("last", "mean"):
            cfg_p = RDFLConfig(
                d_model=48,
                num_heads=4,
                ffn_dim=96,
                num_loops=2,
                chunk_size=8,
                boundary_mode="rlt2_s1",
                boundary_pool=pool,
            )
            mp = RDFLModel(cfg_p).double().eval()
            with torch.no_grad():
                full = mp(x)
                s_run = None
                step_outs = []
                for t in range(x.shape[1]):
                    lt, s_run = mp.step(x[:, t : t + 1], s_run)
                    step_outs.append(lt)
                self.assertLess(float((full - torch.cat(step_outs, dim=1)).abs().max().item()), 1e-12)
                x_mod = x.clone()
                x_mod[0, 11] = (x_mod[0, 11] + 9) % 256
                self.assertEqual(float((mp(x)[:, :11] - mp(x_mod)[:, :11]).abs().max().item()), 0.0)

    def test_11_k3_init_step0_ce_r_test_scaling_and_ablation_flags(self) -> None:
        torch.manual_seed(41)
        x = torch.randint(0, 256, (4, 32))
        y = torch.randint(0, 256, (4, 32))
        for d_model, n_heads in ((48, 4), (192, 6), (512, 8), (768, 12)):
            cfg = RDFLConfig(d_model=d_model, num_heads=n_heads, ffn_dim=d_model * 2)
            model = RDFLModel(cfg).eval()
            x_d = x[:, :16]
            y_d = y[:, :16]
            with torch.no_grad():
                logits = model(x_d)
                ce = float(F.cross_entropy(logits.reshape(-1, 256), y_d.reshape(-1)).item())
                lg_std = float(logits.std().item())
            self.assertLess(lg_std, 2.0)
            self.assertGreaterEqual(ce, 5.0)
            self.assertLessEqual(ce, 6.2)
            self.assertAlmostEqual(ce, math.log(256.0), delta=0.65)

        # R_test > R_train for clamped and sinusoidal, ValueError for learned
        x_small = torch.randint(0, 256, (1, 18))
        for mode in ("clamped", "sinusoidal"):
            cfg_r = RDFLConfig(
                d_model=48,
                num_heads=4,
                ffn_dim=96,
                num_loops=2,
                chunk_size=8,
                step_emb_mode=mode,
            )
            mr = RDFLModel(cfg_r).double().eval()
            with torch.no_grad():
                lg_r2 = mr(x_small, num_loops=2)
                lg_r4 = mr(x_small, num_loops=4)
                lg_r8 = mr(x_small, num_loops=8)
                self.assertGreater(float((lg_r4 - lg_r2).abs().max().item()), 1e-4)
                self.assertGreater(float((lg_r8 - lg_r4).abs().max().item()), 1e-4)
                s_run = None
                step_r4 = []
                for t in range(x_small.shape[1]):
                    lt, s_run = mr.step(x_small[:, t : t + 1], s_run, num_loops=4)
                    step_r4.append(lt)
                self.assertLess(float((lg_r4 - torch.cat(step_r4, dim=1)).abs().max().item()), 1e-12)

        mr_learned = RDFLModel(
            RDFLConfig(d_model=48, num_heads=4, ffn_dim=96, num_loops=2, step_emb_mode="learned")
        )
        with self.assertRaises(ValueError):
            mr_learned(x_small, num_loops=4)

        # Ablation flags: shared_gdn_state, short_conv_mode="first_only", use_bdh=False
        for shared_s, conv_m, bdh_flag in (
            (True, "shared", True),
            (False, "first_only", False),
            (True, "first_only", False),
        ):
            cfg_ab = RDFLConfig(
                d_model=48,
                num_heads=4,
                ffn_dim=96,
                num_loops=2,
                chunk_size=8,
                shared_gdn_state=shared_s,
                short_conv_mode=conv_m,
                use_bdh=bdh_flag,
            )
            mab = RDFLModel(cfg_ab).double().eval()
            if not bdh_flag:
                self.assertIsNone(mab.bdh)
            with torch.no_grad():
                full_ab, _, _, st_ab = mab.forward_stateful(x_small)
                if shared_s:
                    assert st_ab.gdn_states[0] is not None and st_ab.gdn_states[1] is not None
                    self.assertIs(st_ab.gdn_states[0][0], st_ab.gdn_states[1][0])
                s_run = None
                step_ab = []
                for t in range(x_small.shape[1]):
                    lt, s_run = mab.step(x_small[:, t : t + 1], s_run)
                    step_ab.append(lt)
                self.assertLess(float((full_ab - torch.cat(step_ab, dim=1)).abs().max().item()), 1e-12)
                x_mod = x_small.clone()
                x_mod[0, 9] = (x_mod[0, 9] + 5) % 256
                self.assertEqual(float((mab(x_small)[:, :9] - mab(x_mod)[:, :9]).abs().max().item()), 0.0)

    def test_12_k4_mtp4_single_final_norm(self) -> None:
        torch.manual_seed(53)
        x = torch.randint(0, 256, (2, 16))
        for stop_grad in (False, True):
            cfg = RDFLConfig(
                d_model=48, num_heads=4, ffn_dim=96, num_loops=2, mtp_stop_grad=stop_grad
            )
            model = RDFLModel(cfg).double().eval()
            with torch.no_grad():
                model.final_norm.weight.copy_(torch.linspace(0.5, 3.0, 48, dtype=torch.float64))
                logits, _, h_fin, _ = model.forward_stateful(x)
                logits_fwd, h_fin_fwd = model(x, return_pre_norm=True)
                self.assertEqual(float((logits - logits_fwd).abs().max().item()), 0.0)
                self.assertEqual(float((h_fin - h_fin_fwd).abs().max().item()), 0.0)
                # At init (up.weight == 0), mtp_logits apply final_norm ONCE and match logits
                for mtp_lg in model.mtp_logits(h_fin):
                    self.assertLess(float((mtp_lg - logits).abs().max().item()), 1e-12)

    def test_13_k5_snapshot_speculative_and_calibration_mask(self) -> None:
        torch.manual_seed(67)
        for pool in ("last", "mean"):
            for shared_s in (False, True):
                cfg = RDFLConfig(
                    d_model=48,
                    num_heads=4,
                    ffn_dim=96,
                    num_loops=2,
                    chunk_size=8,
                    boundary_mode="rlt2_s1",
                    boundary_pool=pool,
                    shared_gdn_state=shared_s,
                )
                model = RDFLModel(cfg).double().eval()
                # Prompt length 6 crosses chunk_size=8 during first verify_in!
                prompt = torch.randint(0, 256, (1, 6))
                ref = greedy_decode(model, prompt, max_new_tokens=16)

                # Verify snapshots match prefix forward_stateful exactly across chunk boundary
                with torch.no_grad():
                    _, _, _, st0 = model.forward_stateful(prompt)
                    seg = torch.randint(0, 256, (1, 5))
                    _, _, _, snaps = model.forward_with_snapshots(seg, state=st0)
                    for p in range(seg.shape[1]):
                        _, _, _, st_ref = model.forward_stateful(seg[:, : p + 1], state=st0)
                        self.assertEqual(snaps[p].pos, st_ref.pos)
                        self.assertLess(
                            float((snaps[p].slot_state - st_ref.slot_state).abs().max().item()),
                            1e-12,
                        )
                        for r in range(cfg.num_loops):
                            sr = snaps[p].gdn_states[r]
                            rr = st_ref.gdn_states[r]
                            assert sr is not None and rr is not None
                            self.assertLess(float((sr[0] - rr[0]).abs().max().item()), 1e-12)

                # Perturb k3_adapter & mtp_heads so drafts miss and force rollbacks through snapshot path
                with torch.no_grad():
                    model.k3_adapter.up.weight.normal_(mean=0.0, std=5.0)
                    for head in model.mtp_heads:
                        assert isinstance(head, BottleneckAdapter)
                        head.up.weight.normal_(mean=0.0, std=5.0)

                # Count actual _forward_core calls during speculative_decode_k3 & mtp4
                with mock.patch.object(
                    model, "_forward_core", wraps=model._forward_core
                ) as spy:
                    res_k3 = speculative_decode_k3(model, prompt, max_new_tokens=16, draft_k=3)
                    self.assertEqual(spy.call_count, 1 + res_k3.verify_calls)
                    self.assertEqual(int((res_k3.tokens != ref).sum().item()), 0)
                    self.assertGreater(res_k3.rollbacks, 0)
                    self.assertEqual(res_k3.extra_passes, 0)

                    spy.reset_mock()
                    res_mtp = speculative_decode_mtp4(model, prompt, max_new_tokens=16)
                    self.assertEqual(spy.call_count, 1 + res_mtp.verify_calls)
                    self.assertEqual(int((res_mtp.tokens != ref).sum().item()), 0)
                    self.assertGreater(res_mtp.rollbacks, 0)
                    self.assertEqual(res_mtp.extra_passes, 0)

        # FP32 and FP16 snapshot speculative check (0 extra passes, finite states)
        cfg_fp = RDFLConfig(
            d_model=48, num_heads=4, ffn_dim=96, num_loops=2, chunk_size=8, boundary_mode="rlt2_s1"
        )
        m_fp32 = RDFLModel(cfg_fp).eval()
        with torch.no_grad():
            m_fp32.k3_adapter.up.weight.normal_(mean=0.0, std=5.0)
        prompt_fp = torch.randint(0, 256, (1, 6))
        ref_fp32 = greedy_decode(m_fp32, prompt_fp, max_new_tokens=16)
        res_fp32 = speculative_decode_k3(m_fp32, prompt_fp, max_new_tokens=16, draft_k=3)
        self.assertEqual(int((res_fp32.tokens != ref_fp32).sum().item()), 0)
        self.assertEqual(res_fp32.extra_passes, 0)

        m_fp16 = RDFLModel(cfg_fp).half().eval()
        with torch.no_grad():
            m_fp16.k3_adapter.up.weight.normal_(mean=0.0, std=5.0)
        res_fp16 = speculative_decode_k3(m_fp16, prompt_fp, max_new_tokens=16, draft_k=3)
        self.assertEqual(res_fp16.extra_passes, 0)
        self.assertEqual(tuple(res_fp16.tokens.shape), (1, 16))

        # Calibration wide range [0.2, 8.0] and decision-byte mask
        gen = torch.Generator().manual_seed(99)
        base_lg = torch.randn(4, 32, 16, generator=gen)
        targets = base_lg.argmax(dim=-1)
        # Overconfident logits (scaled up) -> optimal T* > 2.0 when noisy
        noisy_targets = targets.clone()
        noisy_targets[:, ::2] = (noisy_targets[:, ::2] + 1) % 16
        t_over = fit_temperature(base_lg * 5.0, noisy_targets)
        self.assertGreater(t_over, 2.0)
        self.assertLessEqual(t_over, 8.0)

        # Underconfident logits (scaled down on exact targets) -> optimal T* < 0.4
        t_under = fit_temperature(base_lg * 0.25, targets)
        self.assertLess(t_under, 0.4)
        self.assertGreaterEqual(t_under, 0.2)

        # Decision-byte mask selects only masked tokens
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, 1::2] = True  # only exact targets!
        t_masked = fit_temperature(base_lg * 0.25, noisy_targets, mask=mask)
        t_direct = fit_temperature((base_lg * 0.25)[mask], noisy_targets[mask])
        self.assertAlmostEqual(t_masked, t_direct, places=6)
        self.assertLess(t_masked, 0.4)


if __name__ == "__main__":
    unittest.main()
