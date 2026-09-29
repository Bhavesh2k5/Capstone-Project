import statistics
import time

import torch

from utils.common import clear_gpu_cache


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_encoding_latency(encoder, sample_videos, n_runs: int = 50) -> dict:
    if not sample_videos:
        raise ValueError("need at least one sample video")
    latencies = []
    for i in range(max(1, n_runs)):
        video = sample_videos[i % len(sample_videos)].unsqueeze(0)
        _sync()
        start = time.perf_counter()
        encoder.encode_video(video)
        _sync()
        latencies.append((time.perf_counter() - start) * 1000.0)
    clear_gpu_cache()
    return {
        "mean_ms": statistics.mean(latencies),
        "std_ms": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        "n_runs": len(latencies),
    }


def measure_memory_usage(encoder, sample_video) -> dict:
    video = sample_video.unsqueeze(0)
    if torch.cuda.is_available():
        clear_gpu_cache()
        torch.cuda.reset_peak_memory_stats()
    encoder.encode_video(video)
    gpu_peak = None
    if torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    cpu_rss = None
    try:
        import psutil

        cpu_rss = psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        try:
            import resource

            cpu_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass
    return {"peak_gpu_mb": gpu_peak, "cpu_rss_mb": cpu_rss}


def measure_throughput(encoder, sample_videos, batch_size: int = 4, n_batches: int = 4) -> dict:
    if not sample_videos:
        raise ValueError("need at least one sample video")
    start = time.perf_counter()
    count = 0
    for b in range(max(1, n_batches)):
        batch = torch.stack(
            [sample_videos[(b * batch_size + i) % len(sample_videos)] for i in range(batch_size)]
        )
        encoder.encode_video(batch)
        count += batch_size
    _sync()
    elapsed = time.perf_counter() - start
    return {
        "clips_per_second": count / max(elapsed, 1e-9),
        "batch_size": batch_size,
        "n_batches": n_batches,
    }
