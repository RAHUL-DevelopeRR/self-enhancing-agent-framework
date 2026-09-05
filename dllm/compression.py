"""Compression boundaries for DLLM.

Semantic compaction belongs in ContextCompiler. Weight/KV quantization belongs
in the inference runtime. TurboQuant belongs at the numerical vector/KV layer.
V0 exposes and reports that boundary without pretending llama.cpp currently
uses an unverified TurboQuant implementation.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionStatus:
    semantic_compaction: bool
    vector_backend: str
    turboquant_requested: bool
    turboquant_available: bool
    turboquant_active: bool
    note: str


def compression_status(semantic_compaction: bool, vector_backend: str) -> CompressionStatus:
    requested = vector_backend.lower() == "turboquant"
    available = importlib.util.find_spec("turboquant") is not None
    active = requested and available
    if active:
        note = "TurboQuant package detected for the optional vector-index backend; validate its implementation before production use."
    elif requested:
        note = "TurboQuant was requested but no compatible package is installed; lexical retrieval remains active."
    else:
        note = "TurboQuant is not active. Use llama.cpp KV quantization now; benchmark TurboQuant separately."
    return CompressionStatus(
        semantic_compaction=semantic_compaction,
        vector_backend=vector_backend,
        turboquant_requested=requested,
        turboquant_available=available,
        turboquant_active=active,
        note=note,
    )
