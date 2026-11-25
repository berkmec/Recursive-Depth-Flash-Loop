"""Comprehensive mathematical-invariant unit tests for RDFL v0.5."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from rdfl import (
    ConsolidationManager,
    RDFLv05Config,
    RDFLv05Model,
)
from rdfl.v05 import GateMode, ReasoningMode, state_nbytes


def _tiny_cfg(
    *,
    reasoning_mode: ReasoningMode = "step_halting",
    mtp_stop_grad: bool = False,
) -> RDFLv05Config:
    return RDFLv05Config(
        d_byte=32,
        d_macro=32,
        micro_heads=4,
        macro_heads=4,
        n_h=4,
        patch_size=4,
        patch_bottleneck=24,
        ffn_macro=64,
        dec_ffn=32,
        num_loops=2,
        chunk_size=16,
        ngram_buckets=(64, 64, 64, 64),
        ngram_dim=8,
        reasoning_mode=reasoning_mode,
        mtp_stop_grad=mtp_stop_grad,
    )


class TestRDFLv05(unittest.TestCase):
    def test_01_purity_ast_and_module_audit(self) -> None:
        v05_path = Path(__file__).resolve().parent.parent / "rdfl" / "v05.py"
        src = v05_path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden_names = {
            "MultiheadAttention",
            "scaled_dot_product_attention",
            "AutoTokenizer",
            "tiktoken",
            "sentencepiece",
        }
        found_forbidden: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                found_forbidden.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in forbidden_names:
                found_forbidden.append(node.attr)
        self.assertEqual(found_forbidden, [])
        self.assertNotIn("cuda", src.lower())

        model = RDFLv05Model(_tiny_cfg())
        self.assertEqual(model.cfg.vocab_size, 256)
        for mod in model.modules():
            if isinstance(mod, nn.Conv1d):
                self.assertIsNone(mod.bias)

    def test_02_fp64_parity_slow_off_on_partial_patch_and_random_matched(self) -> None:
        torch.manual_seed(42)
        rmodes: tuple[ReasoningMode, ...] = ("step_halting", "adfp")
        gmodes: tuple[GateMode, ...] = ("surprise", "random_matched")
        for rmode in rmodes:
            cfg = _tiny_cfg(reasoning_mode=rmode)
            m64 = RDFLv05Model(cfg).double().eval()
            m32 = RDFLv05Model(cfg).float().eval()
            m32.load_state_dict({k: v.float() for k, v in m64.state_dict().items()})

            x = torch.randint(0, 256, (1, 48), dtype=torch.long)
            for upd_slow in (False, True):
                for gmode in gmodes:
                    if (not upd_slow) and gmode == "random_matched":
                        continue
                    with torch.no_grad():
                        f64 = m64(
                            x,
                            num_loops=4,
                            update_slow=upd_slow,
                            gate_mode=gmode,
                        )
                        st64 = None
                        s64_list = []
                        for t in range(x.shape[1]):
                            lt, st64 = m64.step(
                                x[:, t : t + 1],
                                state=st64,
                                num_loops=4,
                                update_slow=upd_slow,
                                gate_mode=gmode,
                                random_gate_seed=123,
                                random_gate_prob=0.5,
                            )
                            s64_list.append(lt)
                        if gmode == "random_matched":
                            f64, _, _, _ = m64.forward_stateful(
                                x,
                                num_loops=4,
                                update_slow=upd_slow,
                                gate_mode="random_matched",
                                random_gate_seed=123,
                                random_gate_prob=0.5,
                            )
                        c64 = torch.cat(s64_list, dim=1)
                    self.assertLess(float((f64 - c64).abs().max().item()), 1e-13)

                    with torch.no_grad():
                        f32, _, _, _ = m32.forward_stateful(
                            x,
                            num_loops=4,
                            update_slow=upd_slow,
                            gate_mode=gmode,
                            random_gate_seed=123,
                            random_gate_prob=0.5,
                        )
                        st32 = None
                        s32_list = []
                        for t in range(x.shape[1]):
                            lt32, st32 = m32.step(
                                x[:, t : t + 1],
                                state=st32,
                                num_loops=4,
                                update_slow=upd_slow,
                                gate_mode=gmode,
                                random_gate_seed=123,
                                random_gate_prob=0.5,
                            )
                            s32_list.append(lt32)
                        c32 = torch.cat(s32_list, dim=1)
                    self.assertEqual(int((f32.argmax(-1) != c32.argmax(-1)).sum().item()), 0)

            # Partial-patch resume parity: 37 + 11 = 48
            with torch.no_grad():
                full, _, _, st_full = m64.forward_stateful(x, num_loops=4, update_slow=True)
                p1, _, _, st1 = m64.forward_stateful(x[:, :37], num_loops=4, update_slow=True)
                p2, _, _, st2 = m64.forward_stateful(
                    x[:, 37:], state=st1, num_loops=4, update_slow=True
                )
                cat = torch.cat([p1, p2], dim=1)
                self.assertLess(float((full - cat).abs().max().item()), 1e-13)
                assert st_full.s_slow is not None and st2.s_slow is not None
                self.assertLess(float((st_full.s_slow - st2.s_slow).abs().max().item()), 1e-13)

    def test_03_causality_and_o1_state_and_r1024_finiteness(self) -> None:
        torch.manual_seed(7)
        model = RDFLv05Model(_tiny_cfg()).double().eval()
        x = torch.randint(0, 256, (1, 70), dtype=torch.long)
        for upd_slow in (False, True):
            for cut in (3, 16, 37, 64):
                x2 = x.clone()
                x2[:, cut:] = (x2[:, cut:] + 73) % 256
                with torch.no_grad():
                    y1 = model(x, num_loops=4, eps_halt=0.05, update_slow=upd_slow)
                    y2 = model(x2, num_loops=4, eps_halt=0.05, update_slow=upd_slow)
                self.assertEqual(float((y1[:, :cut] - y2[:, :cut]).abs().max().item()), 0.0)

        # O(1) constant state size across T=16..320 and R=1..1024
        m32 = RDFLv05Model(_tiny_cfg()).float().eval()
        sizes: list[int] = []
        with torch.no_grad():
            for t_len, r_val in ((16, 1), (64, 2), (256, 8), (320, 64), (32, 1024)):
                xt = torch.randint(0, 256, (1, t_len), dtype=torch.long)
                _, _, _, st = m32.forward_stateful(xt, num_loops=r_val, update_slow=True)
                sizes.append(state_nbytes(st))
        self.assertEqual(len(set(sizes)), 1)
        self.assertGreater(sizes[0], 0)

        # R=1024 finiteness and boundedness in both streaming and step_reasoning modes
        with torch.no_grad():
            xr = torch.randint(0, 256, (1, 20), dtype=torch.long)
            lg_str, _, h_str, _ = m32.forward_stateful(xr, num_loops=1024, update_slow=True)
            self.assertTrue(bool(torch.isfinite(lg_str).all().item()))
            self.assertTrue(bool(torch.isfinite(h_str).all().item()))
            res_rs = m32.forward_step_reasoning(xr, num_loops=1024, collect_trace=False)
            self.assertTrue(bool(torch.isfinite(res_rs["logits"]).all().item()))
            self.assertTrue(bool(torch.isfinite(res_rs["h_final"]).all().item()))
            self.assertTrue(bool(torch.isfinite(res_rs["w_final"]).all().item()))

    def test_04_return_aux_pos0_vs_posgt0_symmetry(self) -> None:
        torch.manual_seed(13)
        model = RDFLv05Model(_tiny_cfg()).double().eval()
        x = torch.randint(0, 256, (2, 24), dtype=torch.long)
        with torch.no_grad():
            _, _, _, _, aux_pos0 = model.forward_stateful(
                x, num_loops=3, update_slow=True, return_aux=True
            )
            _, _, _, st_mid, aux_a = model.forward_stateful(
                x[:, :8], num_loops=3, update_slow=True, return_aux=True
            )
            _, _, _, _, aux_b = model.forward_stateful(
                x[:, 8:], state=st_mid, num_loops=3, update_slow=True, return_aux=True
            )
        expected_keys = {
            "g_surp",
            "u_msp",
            "nll_t",
            "delta_conv",
            "r_used",
            "g_step_last",
            "logits_fast",
        }
        self.assertEqual(set(aux_pos0.keys()), expected_keys)
        self.assertEqual(set(aux_a.keys()), expected_keys)
        self.assertEqual(set(aux_b.keys()), expected_keys)
        for k in expected_keys:
            cat_k = torch.cat([aux_a[k], aux_b[k]], dim=1)
            self.assertEqual(tuple(aux_pos0[k].shape), tuple(cat_k.shape))
            if aux_pos0[k].is_floating_point():
                self.assertLess(float((aux_pos0[k] - cat_k).abs().max().item()), 1e-12)
            else:
                self.assertTrue(bool((aux_pos0[k] == cat_k).all().item()))

    def test_05_k3_decoupling_mtp3_and_single_dec_ffn(self) -> None:
        torch.manual_seed(17)
        model = RDFLv05Model(_tiny_cfg(mtp_stop_grad=False))
        x = torch.randint(0, 256, (2, 16), dtype=torch.long)
        y = torch.randint(0, 256, (2, 16), dtype=torch.long)

        # K3 zero gradient tax on core/head
        model.zero_grad(set_to_none=True)
        _, lg_r1, _, _ = model.forward_stateful(x, num_loops=2)
        torch.autograd.backward(F.cross_entropy(lg_r1.reshape(-1, 256), y.reshape(-1)))
        for name, param in model.named_parameters():
            if "k3_adapter" in name:
                self.assertIsNotNone(param.grad)
            else:
                self.assertIsNone(param.grad)

        # MTP-3 shapes & propagation
        model.zero_grad(set_to_none=True)
        _, _, h_dec, _ = model.forward_stateful(x, num_loops=2)
        mtp_outs = model.mtp_logits(h_dec)
        self.assertEqual(len(mtp_outs), 3)
        for out in mtp_outs:
            self.assertEqual(tuple(out.shape), (2, 16, 256))
        torch.autograd.backward(
            F.cross_entropy(mtp_outs[0][:, :-1].reshape(-1, 256), y[:, 1:].reshape(-1))
        )
        self.assertIsNotNone(model.byte_emb.weight.grad)

        # Single-pass dec_ffn call count (1 call even with update_slow=True)
        dec_calls = 0

        def _hook(_mod: nn.Module, _inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            nonlocal dec_calls
            dec_calls += 1

        handle = model.dec_ffn.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _ = model(x, num_loops=3, update_slow=True)
        finally:
            handle.remove()
        self.assertEqual(dec_calls, 1)

    def test_06_g_step_zero_exact_noop_and_halting_parity(self) -> None:
        torch.manual_seed(29)
        model = RDFLv05Model(_tiny_cfg()).double().eval()
        x = torch.randint(0, 256, (2, 24), dtype=torch.long)
        with torch.no_grad():
            # Exact no-op when gate_override == 0.0: R=1 == R=16 == R=64 bit-exact!
            lg_r1 = model(x, num_loops=1, gate_override=0.0)
            lg_r16 = model(x, num_loops=16, gate_override=0.0)
            lg_r64 = model(x, num_loops=64, gate_override=0.0)
            self.assertEqual(float((lg_r1 - lg_r16).abs().max().item()), 0.0)
            self.assertEqual(float((lg_r1 - lg_r64).abs().max().item()), 0.0)

            # In step reasoning mode: gate_override=0.0 gives h_final == h_0 bit-exact!
            h_0, s_fast, c_last, slot_last = model.encode_for_reasoning(x)
            res_r16 = model.reason_from_state(
                h_0, s_fast, c_last, num_loops=16, slot_idx=slot_last, gate_override=0.0
            )
            self.assertEqual(float((res_r16["h_final"] - h_0).abs().max().item()), 0.0)
            self.assertEqual(float(res_r16["w_final"].abs().max().item()), 0.0)

            # eps_halt early-exit preserves exact FP64 forward() == step() parity
            f64 = model(x[:1], num_loops=8, eps_halt=0.95, update_slow=True)
            st = None
            steps = []
            for t in range(x.shape[1]):
                lt, st = model.step(
                    x[:1, t : t + 1], state=st, num_loops=8, eps_halt=0.95, update_slow=True
                )
                steps.append(lt)
            c64 = torch.cat(steps, dim=1)
            self.assertLess(float((f64 - c64).abs().max().item()), 1e-13)

            # ConsolidationManager smoke
            cons = ConsolidationManager(model.float())
            self.assertEqual(float(cons.penalty_loss().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
