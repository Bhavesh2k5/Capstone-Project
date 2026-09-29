import numpy as np
import torch


def l2_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def l2_normalize_np(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def as_float32(x) -> np.ndarray:
    arr = np.asarray(x)
    return np.ascontiguousarray(arr.astype(np.float32))
