import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import decord

    _HAS_DECORD = True
except Exception:
    decord = None
    _HAS_DECORD = False

BACKENDS = ("auto", "decord", "opencv")
_MAX_SCAN_FRAMES = 4096


class VideoDecodeError(RuntimeError):
    pass


def _uniform_indices(total: int, num_frames: int) -> np.ndarray:
    if total <= 0:
        raise VideoDecodeError("video has no frames")
    if total >= num_frames:
        return np.linspace(0, total - 1, num_frames).astype(np.int64)
    tail = np.full(num_frames - total, total - 1, dtype=np.int64)
    return np.concatenate([np.arange(total, dtype=np.int64), tail])


def _read_with_decord(path: str, indices: np.ndarray) -> np.ndarray:
    vr = decord.VideoReader(path, num_threads=2)
    total = len(vr)
    if total == 0:
        raise VideoDecodeError(f"empty video: {path}")
    safe = np.clip(indices, 0, total - 1)
    return vr.get_batch(safe.tolist()).asnumpy()


def _grab_all_opencv(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoDecodeError(f"cannot open video: {path}")
    frames = []
    while len(frames) < _MAX_SCAN_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise VideoDecodeError(f"no frames decoded: {path}")
    return np.stack(frames)


def _read_with_opencv(path: str, num_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoDecodeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        frames = _grab_all_opencv(path)
        indices = _uniform_indices(len(frames), num_frames)
        return frames[indices]
    indices = _uniform_indices(total, num_frames)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoDecodeError(f"cannot open video: {path}")
    wanted = set(int(i) for i in indices)
    found = {}
    position = 0
    max_index = max(wanted)
    while position <= max_index:
        ok = cap.grab()
        if not ok:
            break
        if position in wanted:
            ok, frame = cap.retrieve()
            if ok:
                found[position] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        position += 1
    cap.release()
    if not found:
        raise VideoDecodeError(f"no frames decoded: {path}")
    last = found[max(found)]
    return np.stack([found.get(i, last) for i in sorted(wanted)])


def sample_frames(path, num_frames: int, backend: str = "auto") -> np.ndarray:
    path = str(path)
    num_frames = int(num_frames)
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}'; expected one of {BACKENDS}")
    if backend == "decord" and not _HAS_DECORD:
        raise VideoDecodeError("decord backend requested but decord is not installed")
    try:
        if backend == "decord" or (backend == "auto" and _HAS_DECORD):
            return _read_with_decord(path, _uniform_indices(_probe_total_decord(path), num_frames))
        return _read_with_opencv(path, num_frames)
    except VideoDecodeError:
        raise
    except Exception as exc:
        raise VideoDecodeError(f"decode failed for {path}: {exc}") from exc


def _probe_total_decord(path: str) -> int:
    vr = decord.VideoReader(path, num_threads=2)
    return len(vr)


def frames_to_tensor(frames: np.ndarray, size: int) -> torch.Tensor:
    size = int(size)
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        if (h, w) != (size, size):
            interp = cv2.INTER_AREA if size < h else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (size, size), interpolation=interp)
        resized.append(frame)
    arr = np.stack(resized).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def normalize_frames(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def collate_fn(batch):
    return {
        "video": torch.stack([b["video"] for b in batch]),
        "label": [b["label"] for b in batch],
        "path": [b["path"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
        "metadata": [b["metadata"] for b in batch],
    }
