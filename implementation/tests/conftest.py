import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_config(tmp_path):
    return {
        "data": {
            "root": str(tmp_path),
            "ucf_crime": {
                "path": "Anomaly-Detection-Dataset",
                "train_split": "Anomaly_Train.txt",
                "test_annotations": "Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
                "extensions": [".mp4", ".avi"],
                "extracted_parts": ["Anomaly-Videos-Part-1"],
            },
            "ucf_arg": {
                "path": "cross camera data",
                "viewpoints": ["aerial_clips", "ground_clips"],
                "classes": ["boxing", "running"],
            },
            "rlvs": {
                "path": "rlvs",
                "classes": ["Violence", "NonViolence"],
                "extensions": ["*.mp4", "*.avi"],
            },
            "tinyvirat": {
                "path": "tinyvirat",
                "class_map": "class_map.json",
                "splits": {"test": "tiny_test.json"},
                "max_clips": None,
            },
        },
        "video": {"num_frames": 4, "frame_size": 64, "fps": 30, "decode_backend": "auto"},
        "models": {
            "default": "clip",
            "vljepa": {
                "visual_backbone": "facebook/vjepa2-vits16-224",
                "text_encoder": "openai/clip-vit-base-patch32",
                "align": "truncate",
                "num_frames": 4,
                "input_size": 64,
                "device": "cpu",
                "batch_size": 2,
            },
            "clip": {
                "primary": "openai/clip-vit-base-patch32",
                "secondary": "openai/clip-vit-large-patch14",
                "default_variant": "primary",
                "input_size": 64,
                "device": "cpu",
                "batch_size": 2,
                "video_strategy": "mean_frame_pooling",
            },
        },
        "retrieval": {
            "index_type": "FlatIP",
            "auto_ivf_threshold": 20000,
            "top_k": 5,
            "embedding_save_dir": str(tmp_path / "results" / "embeddings"),
            "index_save_dir": str(tmp_path / "results" / "indices"),
        },
        "llm": {
            "model": "google/flan-t5-small",
            "quantization": "none",
            "max_new_tokens": 32,
            "temperature": 0.0,
            "max_input_tokens": 256,
            "device": "cpu",
        },
        "evaluation": {
            "recall_k_values": [1, 5],
            "latency_runs": 2,
            "throughput_batches": 1,
            "faithfulness_metric": "grounding_score",
            "hallucination_threshold": 0.5,
            "max_eval_clips": 8,
            "generalization_max_per_class": 4,
        },
        "experiment": {"name": "test", "seed": 42, "output_dir": str(tmp_path / "results")},
    }
