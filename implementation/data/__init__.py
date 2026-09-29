from data.base_dataset import BaseVideoDataset
from data.rlvs_dataset import RLVSDataset
from data.tinyvirat_dataset import TinyVIRATDataset
from data.ucf_arg_dataset import UCFArgDataset
from data.ucf_crime_dataset import UCFCrimeDataset

DATASETS = ("ucf_crime", "ucf_arg", "rlvs", "tinyvirat")


def build_dataset(name: str, config: dict, split=None, viewpoint_filter=None) -> BaseVideoDataset:
    name = str(name).lower()
    if name == "ucf_crime":
        return UCFCrimeDataset(config, mode=split or "all")
    if name == "ucf_arg":
        return UCFArgDataset(config, viewpoint_filter=viewpoint_filter)
    if name == "rlvs":
        return RLVSDataset(config)
    if name == "tinyvirat":
        return TinyVIRATDataset(config, split=split or "test")
    raise ValueError(f"unknown dataset '{name}'; expected one of {DATASETS}")


__all__ = [
    "DATASETS",
    "BaseVideoDataset",
    "RLVSDataset",
    "TinyVIRATDataset",
    "UCFArgDataset",
    "UCFCrimeDataset",
    "build_dataset",
]
