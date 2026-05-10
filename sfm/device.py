"""Central GPU/device management for the SfM pipeline.

All modules import `get_device()` and `has_gpu()` from here instead of
duplicating CUDA-detection logic.  The device is resolved once and cached.
"""

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device():
    """
    Return the best available torch.device (CUDA > CPU), cached after first call.
    Returns None when PyTorch is not installed.
    """
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(
                f"GPU: {props.name}  "
                f"({props.total_memory // 1024 ** 2} MB VRAM, "
                f"compute {props.major}.{props.minor})"
            )
            return torch.device("cuda")
        logger.info("CUDA not available — using CPU torch device.")
        return torch.device("cpu")
    except ImportError:
        logger.debug("PyTorch not installed — GPU acceleration disabled.")
        return None


def has_gpu() -> bool:
    """Return True if a CUDA GPU is accessible via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def torch_available() -> bool:
    """Return True if PyTorch is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
