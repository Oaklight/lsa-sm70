"""Latent Sparse Attention (LSA) for NVIDIA V100 (SM70).

LSA clusters keys in a learned latent space, has each query score bucket
centroids instead of every key, and reads only the selected keys plus a local
window and sinks. This package ports the flat D-LSA design to SM70 and serves
it as an SGLang attention backend:

- the selection core is plain PyTorch and runs on CPU or CUDA;
- two Triton programs score and expand on CUDA;
- ``LsaSm70Backend`` registers ``--attention-backend lsa_sm70`` with the
  pinned engine and reads through its Triton decode kernel.

The reference is `Prefrontal-AI/latent-sparse-attention` (Apache-2.0); see
``PROVENANCE.md`` for what is vendored and what is written here.

Installing the package registers the SGLang plugin entry point. This module
and ``register()`` stay import-light: torch, triton and sglang load on first
use.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def register() -> None:
    """Component entry point: register the LSA attention backend."""
    from .backend_registry import install

    install()
