from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter  # noqa: F401 - available for the learner implementation

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
    del device
    raise NotImplementedError("Synchronize CPU or MPS work")


def benchmark_operation(
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmup: int = 3,
    iterations: int = 10,
) -> TimingStats:
    """Warm up and collect synchronized wall-clock latency samples."""
    del operation, device, warmup, iterations
    raise NotImplementedError("Measure synchronized operation latency")
