"""Initialization of the backend package for the Baseline pipeline."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def _limit_torch_threads() -> None:
    """Configures PyTorch intra-op threads to ensure computational stability."""
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

_limit_torch_threads()
