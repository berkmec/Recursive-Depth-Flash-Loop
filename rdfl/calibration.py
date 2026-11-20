"""Zero-parameter uncertainty calibration (MSP + T*) and exact 3D per-token reverse KL."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor


def msp_uncertainty(logits: Tensor, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    """Returns (confidence = max_p, uncertainty = 1 - max_p) with zero extra parameters."""
    probs = F.softmax(logits.float() / temperature, dim=-1)
    conf = probs.max(dim=-1).values
    return conf, 1.0 - conf


def fit_temperature(logits: Tensor, targets: Tensor) -> float:
    """Grid-searches NLL-optimal scalar temperature T* on a held-out calibration split."""
    lg = logits.float().reshape(-1, logits.shape[-1])
    y = targets.long().reshape(-1)
    best_t, best_nll = 1.0, float("inf")
    for t_cand in np.linspace(0.40, 2.00, 65):
        nll = float(F.cross_entropy(lg / float(t_cand), y))
        if nll < best_nll:
            best_nll, best_t = nll, float(t_cand)
    return best_t


def ece_15(confidence: NDArray[Any], correct: NDArray[Any]) -> float:
    """15-bin Expected Calibration Error."""
    edges = np.linspace(0.0, 1.0, 16)
    n = max(1, len(confidence))
    err = 0.0
    for i in range(15):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence <= hi if i == 14 else confidence < hi)
        if mask.any():
            err += (mask.sum() / n) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(err)


def auroc(scores: NDArray[Any], labels: NDArray[Any]) -> float:
    """Exact rank-based Area Under the ROC Curve with mid-rank tie handling."""
    pos = np.asarray(labels) > 0.5
    n_pos = int(pos.sum())
    n_neg = len(pos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    sorted_s = arr[order]
    ranks = np.empty(len(arr), dtype=np.float64)
    i, n = 0, len(arr)
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def aurc(uncertainty: NDArray[Any], errors: NDArray[Any]) -> float:
    """Area Under the Risk-Coverage curve (lower is better)."""
    order = np.argsort(uncertainty)
    err_sorted = errors[order]
    cum_risk = np.cumsum(err_sorted) / np.arange(1, len(err_sorted) + 1)
    return float(cum_risk.mean())


def reverse_kl_per_token(logits_policy: Tensor, logits_ref: Tensor) -> Tensor:
    """Exact 3D per-token reverse KL: D_KL(pi_policy || pi_ref).

    Avoids the PyTorch 3D `F.kl_div(reduction='batchmean')` sequence-length inflation bug.
    """
    log_p = F.log_softmax(logits_policy.float(), dim=-1)
    log_q = F.log_softmax(logits_ref.float().detach(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()
