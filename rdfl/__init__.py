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
from rdfl.model import RDFLConfig, RDFLModel, ZeroDiffRDFLAdapter

__all__ = [
    "BdhSparseGate",
    "BottleneckAdapter",
    "CausalShortConv",
    "Fp32SlotCoreBoundary",
    "GatedDeltaNetLayer",
    "RDFLConfig",
    "RDFLModel",
    "RMSNorm",
    "ZeroDiffRDFLAdapter",
    "aurc",
    "auroc",
    "ece_15",
    "fit_temperature",
    "msp_uncertainty",
    "reverse_kl_per_token",
]
