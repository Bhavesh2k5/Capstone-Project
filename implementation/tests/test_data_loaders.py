import json

import pytest

from data import build_dataset
from data.rlvs_dataset import RLVSDataset
from data.ucf_arg_dataset import UCFArgDataset
from data.ucf_crime_dataset import UCFCrimeDataset
from data.tinyvirat_dataset import TinyVIRATDataset


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_rlvs_loads_mp4_and_avi(sample_config, tmp_path):
    root = tmp_path / "rlvs"
    for name in ("V_1.mp4", "V_2.mp4"):
        _touch(root / "Violence" / name)
    _touch(root / "NonViolence" / "NV_1.mp4")
    _touch(root / "NonViolence" / "NV_602.avi")

    dataset = RLVSDataset(sample_config)
    labels = [s["label"] for s in dataset.samples]
    paths = [s["path"] for s in dataset.samples]
    assert labels.count("violence") == 2
    assert labels.count("non_violence") == 2
    assert any(p.endswith(".avi") for p in paths)


def test_ucf_arg_parses_filenames_and_viewpoints(sample_config, tmp_path):
    root = tmp_path / "cross camera data"
    for vp in ("aerial_clips", "ground_clips"):
        _touch(root / vp / vp / "boxing" / f"person01_01_{vp[:-6]}_boxing.avi")
    _touch(root / "aerial_clips" / "aerial_clips" / "running" / "person02_03_aerial_running.avi")

    dataset = UCFArgDataset(sample_config)
    assert len(dataset) == 3
    meta = dataset.samples[0]["metadata"]
    assert meta["viewpoint"] in ("aerial", "ground")
    assert meta["action"] == "boxing"

    aerial_only = UCFArgDataset(sample_config, viewpoint_filter="aerial_clips")
    assert all(s["metadata"]["viewpoint"] == "aerial" for s in aerial_only.samples)


def test_ucf_crime_skips_missing_parts(sample_config, tmp_path):
    root = tmp_path / "Anomaly-Detection-Dataset"
    _touch(root / "Anomaly-Videos-Part-1" / "Abuse" / "Abuse001_x264.mp4")
    _touch(root / "Anomaly-Videos-Part-1" / "Arrest" / "Arrest001_x264.mp4")
    _touch(root / "Normal_Videos_for_Event_Recognition" / "Normal_001.mp4")

    sample_config["data"]["ucf_crime"]["extracted_parts"] = [
        "Anomaly-Videos-Part-1",
        "Anomaly-Videos-Part-2",
    ]
    dataset = UCFCrimeDataset(sample_config, mode="all")
    labels = sorted(s["label"] for s in dataset.samples)
    assert labels == ["Abuse", "Arrest", "Normal"]


def test_ucf_crime_train_split(sample_config, tmp_path):
    root = tmp_path / "Anomaly-Detection-Dataset"
    _touch(root / "Anomaly-Videos-Part-1" / "Abuse" / "Abuse001_x264.mp4")
    split = root / "Anomaly_Train.txt"
    split.write_text("Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4 Abuse\n", encoding="utf-8")
    dataset = UCFCrimeDataset(sample_config, mode="train")
    assert len(dataset) == 1
    assert dataset.samples[0]["label"] == "Abuse"


def test_tinyvirat_json_index_and_native_size(sample_config, tmp_path):
    root = tmp_path / "tinyvirat"
    (root / "class_map.json").write_text(json.dumps({"activity_walking": 0, "activity_running": 1}))
    (root / "tiny_test.json").write_text(
        json.dumps(
            [
                {"id": "0", "video_id": "V1", "path": "V1/000000.mp4", "dim": [82, 72, 72],
                 "label": ["activity_walking", "activity_running"]},
                {"id": "1", "video_id": "V1", "path": "V1/missing.mp4", "dim": [80, 72, 72],
                 "label": ["activity_walking"]},
            ]
        )
    )
    _touch(root / "videos" / "test" / "V1" / "000000.mp4")

    dataset = TinyVIRATDataset(sample_config, split="test")
    assert len(dataset) == 1
    sample = dataset.samples[0]
    assert sample["label"] == ["activity_walking", "activity_running"]
    assert sample["frame_size"] == 72


def test_build_dataset_rejects_unknown(sample_config):
    with pytest.raises(ValueError):
        build_dataset("nope", sample_config)
