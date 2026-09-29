import logging
from pathlib import Path

from data.base_dataset import BaseVideoDataset
from utils.common import dataset_root

logger = logging.getLogger(__name__)


class UCFArgDataset(BaseVideoDataset):
    def __init__(self, config: dict, viewpoint_filter=None):
        super().__init__(config, "ucf_arg")
        cfg = self.config["data"][self.name]
        self.viewpoints = list(cfg.get("viewpoints", []))
        if viewpoint_filter:
            if isinstance(viewpoint_filter, str):
                viewpoint_filter = [viewpoint_filter]
            self.viewpoints = [v for v in self.viewpoints if v in viewpoint_filter]
        self.finalize()

    def _build_index(self):
        root = dataset_root(self.config, self.name)
        samples = []
        for vp in self.viewpoints:
            outer = root / vp
            if (outer / vp).is_dir():
                bases = [outer / vp]
            elif outer.is_dir():
                bases = [outer]
            else:
                logger.info("ucf_arg: viewpoint folder '%s' not found - skipping", outer)
                continue
            for base in bases:
                for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                    for f in sorted(class_dir.glob("*.avi")):
                        meta = self._parse_name(f.stem, vp, class_dir.name)
                        samples.append({"path": str(f), "label": meta["action"], "metadata": meta})
        return samples

    @staticmethod
    def _parse_name(stem: str, default_viewpoint: str, default_action: str) -> dict:
        parts = stem.split("_")
        meta = {
            "actor": parts[0] if parts else "",
            "run": parts[1] if len(parts) > 1 else "",
            "viewpoint": default_viewpoint,
            "action": default_action,
        }
        if len(parts) >= 4:
            meta["viewpoint"] = parts[2]
            meta["action"] = "_".join(parts[3:])
        elif len(parts) == 3:
            meta["action"] = parts[2]
        return meta
