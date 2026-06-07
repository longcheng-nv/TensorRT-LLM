"""Twin registry — operator-specific dual twins plug in here.

Adding a new operator = drop a module that calls register_twin(name, TwinClass) and
implements the Twin protocol. The harness core stays operator-agnostic. This is the seam
that turns twin maintenance into "regenerate-and-revalidate" (see dual-twin-authoring skill).
"""

from __future__ import annotations

from typing import Dict, Protocol

import numpy as np


class Twin(Protocol):
    name: str

    def generate_inputs(self, rng, shape, distribution): ...
    def reference(self, A, B, shape, ref_dtype) -> np.ndarray: ...
    def real_and_dual(self, A, B, policy, shape, with_cross: bool):
        """Return (Y_real, dual_channels: dict[tag->ndarray], extra: dict)."""
        ...


_REGISTRY: Dict[str, Twin] = {}


def register_twin(name: str, twin: Twin) -> None:
    _REGISTRY[name] = twin


def get_twin(name: str) -> Twin:
    if name not in _REGISTRY:
        raise KeyError(f"twin '{name}' not registered; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_twins():
    return sorted(_REGISTRY)


# register built-ins
from . import moe_gemm  # noqa: E402,F401
