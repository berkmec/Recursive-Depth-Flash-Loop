"""Unit tests verifying RDFL mathematical invariants on CPU."""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from rdfl import (
    RDFLConfig,
    RDFLModel,
    ZeroDiffRDFLAdapter,
    auroc,
    ece_15,
    fit_temperature,
    greedy_decode,
    reverse_kl_per_token,
    speculative_decode_k3,
    speculative_decode_mtp4,
)


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
        F.cross_entropy(logits_r1.reshape(-1, 256), y.reshape(-1)).backward()
        self.assertIsNone(model.gdn.qkv.weight.grad)
        self.assertIsNone(model.emb.weight.grad)
        self.assertGreater(float(model.k3_adapter.up.weight.grad.abs().sum().item()), 0.0)

    def test_05_mtp4_propagation_vs_stop_grad(self) -> None:
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        m_full = RDFLModel(RDFLConfig(d_model=48, num_heads=4, ffn_dim=96, mtp_stop_grad=False))
        _, _, hn, _ = m_full.forward_stateful(x)
        mtp_lg = m_full.mtp_logits(hn)
        F.cross_entropy(mtp_lg[0][:, :-1].reshape(-1, 256), y[:, 1:].reshape(-1)).backward()
        self.assertGreater(float(m_full.gdn.qkv.weight.grad.abs().sum().item()), 0.0)

        m_stop = RDFLModel(RDFLConfig(d_model=48, num_heads=4, ffn_dim=96, mtp_stop_grad=True))
        _, _, hn2, _ = m_stop.forward_stateful(x)
        mtp_lg2 = m_stop.mtp_logits(hn2)
        F.cross_entropy(mtp_lg2[0][:, :-1].reshape(-1, 256), y[:, 1:].reshape(-1)).backward()
        self.assertIsNone(m_stop.gdn.qkv.weight.grad)
        self.assertGreater(float(m_stop.mtp_heads[0].up.weight.grad.abs().sum().item()), 0.0)

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
        self.assertTrue(0.4 <= t_star <= 2.0)
        self.assertEqual(auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])), 1.0)
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

        res_mtp = speculative_decode_mtp4(model, prompt, max_new_tokens=18)
        self.assertEqual(int((res_mtp.tokens != ref).sum().item()), 0)
        self.assertGreater(res_mtp.verify_calls, 0)
        self.assertGreater(res_mtp.proposed_drafts, 0)

        # R=1 model with zero-init K3 adapter must achieve 100% draft acceptance and 0 rollbacks.
        cfg_r1 = RDFLConfig(
            d_model=48, num_heads=4, ffn_dim=96, num_loops=1, chunk_size=8, num_slots=0
        )
        m_r1 = RDFLModel(cfg_r1).double().eval()
        ref_r1 = greedy_decode(m_r1, prompt, max_new_tokens=12)
        res_r1 = speculative_decode_k3(m_r1, prompt, max_new_tokens=12, draft_k=3)
        self.assertEqual(int((res_r1.tokens != ref_r1).sum().item()), 0)
        self.assertEqual(res_r1.rollbacks, 0)
        self.assertEqual(res_r1.acceptance_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
