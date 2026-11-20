"""Recursive Depth Flash Loop (RDFL) reference PyTorch implementation."""

from rdfl.calibration import (
    aurc,
    auroc,
    ece_15,
    fit_temperature,
    msp_uncertainty,
    reverse_kl_per_token,
)
from rdfl.layers import (
    BdhSparseGate,
    BottleneckAdapter,
    CausalShortConv,
    Fp32SlotCoreBoundary,
    GatedDeltaNetLayer,
    RMSNorm,
)
from rdfl.model import RDFLConfig, RDFLModel, RDFLState, ZeroDiffRDFLAdapter
from rdfl.speculative import (
    SpeculativeDecodeResult,
    greedy_decode,
    speculative_decode_k3,
    speculative_decode_mtp4,
)

__all__ = [
    "BdhSparseGate",
    "BottleneckAdapter",
    "CausalShortConv",
    "Fp32SlotCoreBoundary",
    "GatedDeltaNetLayer",
    "RDFLConfig",
    "RDFLModel",
    "RDFLState",
    "RMSNorm",
    "SpeculativeDecodeResult",
    "ZeroDiffRDFLAdapter",
    "aurc",
    "auroc",
    "ece_15",
    "fit_temperature",
    "greedy_decode",
    "msp_uncertainty",
    "reverse_kl_per_token",
    "speculative_decode_k3",
    "speculative_decode_mtp4",
]
