import logging
from abc import abstractmethod

import torch
from torch.utils.data import Dataset

from data.data_utils import VideoDecodeError, frames_to_tensor, sample_frames

logger = logging.getLogger(__name__)


class BaseVideoDataset(Dataset):
    def __init__(self, config: dict, name: str):
        self.config = config
        self.name = name
        vcfg = config.get("video", {})
        self.num_frames = int(vcfg.get("num_frames", 8))
        self.frame_size = int(vcfg.get("frame_size", 224))
        self.backend = vcfg.get("decode_backend", "auto")
        self.samples = []

    @abstractmethod
    def _build_index(self):
        ...

    def finalize(self):
        self.samples = self._build_index()
        if not self.samples:
            logger.warning("%s: no videos indexed - check dataset path or extraction state", self.name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if not self.samples:
            raise RuntimeError(f"{self.name}: dataset is empty")
        n = len(self.samples)
        last_error = None
        for attempt in range(min(5, n)):
            sample = self.samples[(index + attempt) % n]
            try:
                frames = sample_frames(sample["path"], self.num_frames, backend=self.backend)
                size = int(sample.get("frame_size", self.frame_size))
                video = frames_to_tensor(frames, size)
                return {
                    "video": video,
                    "label": sample["label"],
                    "path": sample["path"],
                    "dataset": self.name,
                    "metadata": sample.get("metadata", {}),
                }
            except VideoDecodeError as exc:
                last_error = exc
                logger.warning("%s: skipping unreadable video %s (%s)", self.name, sample["path"], exc)
        raise RuntimeError(f"{self.name}: no decodable video near index {index}: {last_error}")

    def labels(self):
        return [s["label"] for s in self.samples]
