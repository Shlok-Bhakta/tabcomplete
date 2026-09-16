"""Latency harness: prefill/decode throughput, TTFT, RAM/VRAM peaks.

CPU-safe: peak VRAM is None when CUDA is absent. benchmark() drives a raw
model with token ids (no tokenizer needed) using the cache the way an
editor session would: one prefill forward, then cached single-token decodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

__all__ = ["GenerationStats", "peak_ram_mb", "peak_vram_mb", "benchmark"]


@dataclass
class GenerationStats:
    prefill_tokens: int = 0
    decoded_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    time_to_first_token: float = 0.0
    wall_time: float = 0.0
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None

    @property
    def prefill_tokens_per_second(self) -> float:
        return self.prefill_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    @property
    def decode_tokens_per_second(self) -> float:
        return self.decoded_tokens / self.decode_seconds if self.decode_seconds else 0.0


def peak_ram_mb() -> float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return None


def peak_vram_mb() -> float | None:
    try:
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20
    except Exception:
        return None


@torch.no_grad()
def benchmark(model, input_ids: torch.Tensor, max_new_tokens: int = 8) -> GenerationStats:
    """Prefill input_ids, then greedily decode max_new_tokens with cache."""
    model.eval()
    stats = GenerationStats(prefill_tokens=int(input_ids.numel()))
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    past = None
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats.prefill_seconds = time.perf_counter() - t0
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(dim=-1)
    stats.time_to_first_token = time.perf_counter() - start
    t1 = time.perf_counter()
    for _ in range(max_new_tokens):
        out = model(next_id, past_key_values=past, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(dim=-1)
        stats.decoded_tokens += 1
    stats.decode_seconds = time.perf_counter() - t1
    stats.wall_time = time.perf_counter() - start
    stats.peak_ram_mb = peak_ram_mb()
    stats.peak_vram_mb = peak_vram_mb()
    return stats
