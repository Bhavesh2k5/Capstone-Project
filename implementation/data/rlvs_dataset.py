import logging
from pathlib import Path

from data.base_dataset import BaseVideoDataset
from utils.common import dataset_root

logger = logging.getLogger(__name__)


class RLVSDataset(BaseVideoDataset):
    LABEL_MAP = {"Violence": "violence", "NonViolence": "non_violence"}

    def __init__(self, config: dict):
        super().__init__(config, "rlvs")
        self.finalize()

    def _build_index(self):
        root = dataset_root(self.config, self.name)
        cfg = self.config["data"][self.name]
        patterns = list(cfg.get("extensions", ["*.mp4", "*.avi"]))
        if "*.mp4" not in patterns or "*.avi" not in patterns:
            logger.warning(
                "rlvs: extensions config must include both *.mp4 and *.avi - "
                "49 NonViolence clips are AVI and would be silently dropped"
            )
        samples = []
        for cls in cfg.get("classes", ["Violence", "NonViolence"]):
            class_dir = root / cls
            if not class_dir.is_dir():
                logger.warning("rlvs: class folder '%s' not found under %s", cls, root)
                continue
            files = {}
            for pattern in patterns:
                for f in class_dir.glob(pattern):
                    if f.is_file():
                        files[str(f.resolve())] = f
            label = self.LABEL_MAP.get(cls, cls.lower())
            for f in sorted(files.values()):
                samples.append({"path": str(f), "label": label, "metadata": {"class_dir": cls}})
        return samples
