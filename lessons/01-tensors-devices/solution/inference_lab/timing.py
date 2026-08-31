from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import torch


@dataclass(frozen=True, slots=True)
class TimingStats:
    """Raw latency samples and a few descriptive statistics."""

    device: str
    warmup_iterations: int
    samples_ms: tuple[float, ...]

    @property
    def iterations(self) -> int:
        return len(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return float(median(self.samples_ms))

    @property
    def minimum_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def maximum_ms(self) -> float:
        return max(self.samples_ms)


def synchronize(device: torch.device) -> None:
    """Wait until work queued on the supported device has completed."""
    if device.type == "cpu":
        return
    if device.type == "mps":
        torch.mps.synchronize()
        return
    raise ValueError("Lesson 1 synchronization supports only cpu and mps devices")


def benchmark_operation(
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 3,
    iterations: int = 10,
) -> TimingStats:
    """Warm up and collect synchronized wall-clock latency samples."""
    if warmup < 0:
        raise ValueError("warmup must be at least zero")
    if iterations < 1:
        raise ValueError("iterations must be at least one")

    for _ in range(warmup):
        operation()
        synchronize(device)

    samples: list[float] = []
    for _ in range(iterations):
        synchronize(device)
        started = perf_counter()
        operation()
        synchronize(device)
        samples.append((perf_counter() - started) * 1_000)
    return TimingStats(device.type, warmup, tuple(samples))
