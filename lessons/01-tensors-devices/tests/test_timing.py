import inference_lab.timing as timing
import pytest
import torch

from inference_lab import TimingStats, benchmark_operation, synchronize


def test_synchronize_accepts_cpu() -> None:
    synchronize(torch.device("cpu"))


def test_synchronize_rejects_unsupported_device() -> None:
    with pytest.raises(ValueError, match="cpu and mps"):
        synchronize(torch.device("meta"))


def test_benchmark_runs_warmup_and_synchronizes_each_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clock_values = iter((10.0, 10.001, 20.0, 20.003))

    def operation() -> None:
        events.append("operation")

    monkeypatch.setattr(timing, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(timing, "perf_counter", lambda: next(clock_values))

    stats = benchmark_operation(
        operation,
        device=torch.device("cpu"),
        warmup=1,
        iterations=2,
    )

    assert events == [
        "operation",
        "sync",
        "sync",
        "operation",
        "sync",
        "sync",
        "operation",
        "sync",
    ]
    assert stats.samples_ms == pytest.approx((1.0, 3.0))


def test_benchmark_rejects_invalid_iteration_counts() -> None:
    with pytest.raises(ValueError, match="warmup must be at least zero"):
        benchmark_operation(lambda: None, device=torch.device("cpu"), warmup=-1)
    with pytest.raises(ValueError, match="iterations must be at least one"):
        benchmark_operation(lambda: None, device=torch.device("cpu"), iterations=0)


def test_timing_stats_report_literal_summary() -> None:
    stats = TimingStats(device="cpu", warmup_iterations=2, samples_ms=(1.0, 3.0, 2.0))
    assert stats.iterations == 3
    assert stats.median_ms == 2.0
    assert stats.minimum_ms == 1.0
    assert stats.maximum_ms == 3.0
