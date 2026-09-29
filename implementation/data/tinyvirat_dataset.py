import json
import logging
from pathlib import Path

from data.base_dataset import BaseVideoDataset
from utils.common import dataset_root

logger = logging.getLogger(__name__)

DEFAULT_SPLITS = {
    "train": "tiny_train_v2.json",
    "val": "tiny_val_v2.json",
    "test": "tiny_test_v2_public.json",
}

TinyVIRAT_CLASSES = [
    "Opening", "Interacts", "Pull", "activity_carrying", "Entering", "vehicle_moving",
    "Exiting", "Loading", "Talking", "activity_running", "vehicle_turning_left",
    "vehicle_stopping", "Riding", "Closing", "activity_walking", "Push",
    "specialized_using_tool", "vehicle_starting", "specialized_miscellaneous",
    "activity_standing", "Transport_HeavyCarry", "activity_gesturing",
    "vehicle_turning_right", "specialized_talking_phone", "specialized_texting_phone", "Misc",
]


class TinyVIRATDataset(BaseVideoDataset):
    def __init__(self, config: dict, split: str = "test"):
        super().__init__(config, "tinyvirat")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: train, val, test")
        self.split = split
        self.class_map = self._load_class_map()
        self.finalize()

    def _load_class_map(self):
        cfg = self.config["data"][self.name]
        path = dataset_root(self.config, self.name) / cfg.get("class_map", "class_map.json")
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("tinyvirat: failed to read class_map.json (%s)", exc)
        return {c: i for i, c in enumerate(TinyVIRAT_CLASSES)}

    def _split_file(self) -> Path:
        cfg = self.config["data"][self.name]
        splits = cfg.get("splits", DEFAULT_SPLITS)
        return dataset_root(self.config, self.name) / splits.get(self.split, DEFAULT_SPLITS[self.split])

    def _resolve_clip(self, rel: Path):
        root = dataset_root(self.config, self.name)
        candidates = [
            root / "videos" / self.split / rel,
            root / "videos" / self.split / rel.name,
            root / "videos" / rel,
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def _build_index(self):
        split_file = self._split_file()
        if not split_file.is_file():
            logger.warning("tinyvirat: split file '%s' not found", split_file)
            return []
        entries = json.loads(split_file.read_text(encoding="utf-8"))
        cfg = self.config["data"][self.name]
        max_clips = cfg.get("max_clips")
        samples = []
        for entry in entries:
            rel = Path(str(entry.get("path", "")).replace("\\", "/"))
            if not rel.name:
                continue
            path = self._resolve_clip(rel)
            if path is None:
                continue
            dim = entry.get("dim") or []
            size = self.frame_size
            if len(dim) >= 3:
                native = max(int(dim[1]), int(dim[2]))
            elif len(dim) == 2:
                native = max(int(dim[0]), int(dim[1]))
            else:
                native = self.frame_size
            if native < size:
                size = native
            samples.append({
                "path": str(path),
                "label": list(entry.get("label", [])),
                "metadata": {"video_id": entry.get("video_id"), "dim": dim, "split": self.split},
                "frame_size": size,
            })
            if max_clips and len(samples) >= int(max_clips):
                break
        return samples
