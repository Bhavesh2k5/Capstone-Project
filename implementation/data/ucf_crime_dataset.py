import logging
from pathlib import Path

from data.base_dataset import BaseVideoDataset
from utils.common import config_file, dataset_root

logger = logging.getLogger(__name__)

ANOMALY_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]
VIDEO_EXTS = (".mp4", ".avi")


class UCFCrimeDataset(BaseVideoDataset):
    def __init__(self, config: dict, mode: str = "all"):
        super().__init__(config, "ucf_crime")
        if mode not in ("train", "test", "all"):
            raise ValueError("mode must be one of: train, test, all")
        self.mode = mode
        self.finalize()

    def _video_files(self, folder: Path):
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)

    def _build_index(self):
        root = dataset_root(self.config, self.name)
        if self.mode == "train":
            return self._index_from_train_split(root)
        if self.mode == "test":
            return self._index_from_test_annotations(root)
        return self._index_from_folders(root)

    def _index_from_folders(self, root: Path):
        cfg = self.config["data"][self.name]
        samples = []
        for part in cfg.get("extracted_parts", []):
            folder = root / part
            if not folder.is_dir():
                logger.info("ucf_crime: extracted part '%s' not found - skipping (still zipped?)", part)
                continue
            if "Normal" in part:
                for p in self._video_files(folder):
                    samples.append({"path": str(p), "label": "Normal", "metadata": {"part": part}})
                continue
            for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
                label = class_dir.name
                for p in self._video_files(class_dir):
                    samples.append({"path": str(p), "label": label, "metadata": {"part": part}})
        return samples

    def _index_from_train_split(self, root: Path):
        split = config_file(self.config, self.name, "train_split")
        if not split.is_file():
            logger.warning("ucf_crime: train split '%s' not found - falling back to folder scan", split)
            return self._index_from_folders(root)
        samples = []
        for line in split.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            rel = Path(tokens[0].replace("\\", "/"))
            label = tokens[1] if len(tokens) > 1 else rel.parent.name
            candidate = root / rel
            if not candidate.is_file():
                matches = [p for p in root.rglob(rel.name) if p.is_file()]
                candidate = matches[0] if matches else None
            if candidate is not None and candidate.is_file():
                samples.append({"path": str(candidate), "label": label, "metadata": {"split": "train"}})
        return samples

    def _index_from_test_annotations(self, root: Path):
        ann = config_file(self.config, self.name, "test_annotations")
        if not ann.is_file():
            logger.warning("ucf_crime: test annotations '%s' not found", ann)
            return []
        name_map = {}
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                name_map.setdefault(p.name, p)
        samples = []
        for line in ann.read_text(encoding="utf-8", errors="ignore").splitlines():
            tokens = line.split()
            if len(tokens) < 3:
                continue
            video_name, label = tokens[0], tokens[1]
            path = name_map.get(video_name)
            if path is None:
                continue
            samples.append({
                "path": str(path),
                "label": label,
                "metadata": {"split": "test", "temporal": tokens[2:]},
            })
        return samples
