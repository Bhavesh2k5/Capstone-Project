import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_LOGGERS = {}


def get_logger(name: str) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str = "auto") -> str:
    if preference and str(preference).lower() != "auto":
        return str(preference)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _rebase(value, base: Path) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str(base / p)


def load_config(path=None) -> dict:
    config_path = Path(path or os.environ.get("CAPSTONE_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_root = os.environ.get("CAPSTONE_DATA_ROOT")
    if env_root:
        cfg.setdefault("data", {})["root"] = env_root
    out_root = os.environ.get("CAPSTONE_OUTPUT_ROOT")
    base = Path(out_root) if out_root else PROJECT_ROOT
    for section, key in (
        ("retrieval", "embedding_save_dir"),
        ("retrieval", "index_save_dir"),
        ("experiment", "output_dir"),
    ):
        if section in cfg and key in cfg[section]:
            cfg[section][key] = _rebase(cfg[section][key], base)
    device = os.environ.get("CAPSTONE_DEVICE")
    if device:
        for name in ("vljepa", "clip"):
            if name in cfg.get("models", {}):
                cfg["models"][name]["device"] = device
        if "llm" in cfg:
            cfg["llm"]["device"] = device
    cfg["_config_path"] = str(config_path)
    cfg["_output_root"] = str(base)
    return cfg


def dataset_root(cfg: dict, name: str) -> Path:
    entry = cfg["data"][name]
    p = Path(str(entry["path"]))
    if p.is_absolute():
        return p
    return Path(str(cfg["data"]["root"])) / p


def config_file(cfg: dict, name: str, key: str) -> Path:
    p = Path(str(cfg["data"][name][key]))
    if p.is_absolute():
        return p
    return dataset_root(cfg, name) / p


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clear_gpu_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
