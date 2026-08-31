import random

import pytest
import torch

from inference_lab import DevicePolicy, DeviceUnavailableError, seed_everything


def test_cpu_policy_reports_requested_and_selected_device() -> None:
    policy = DevicePolicy.resolve("cpu")
    assert policy.requested == "cpu"
    assert policy.selected == torch.device("cpu")
    assert policy.dtype is torch.float32
    report = policy.report()
    assert report["requested_device"] == "cpu"
    assert report["selected_device"] == "cpu"
    assert report["dtype"] == "torch.float32"
    assert report["reason"]


def test_auto_policy_reports_actual_device() -> None:
    policy = DevicePolicy.resolve("auto")
    expected = "mps" if torch.backends.mps.is_available() else "cpu"
    assert policy.requested == "auto"
    assert policy.selected.type == expected
    assert policy.report()["selected_device"] == expected


def test_explicit_mps_never_silently_falls_back() -> None:
    if torch.backends.mps.is_available():
        assert DevicePolicy.resolve("mps").selected.type == "mps"
    else:
        with pytest.raises(DeviceUnavailableError, match="MPS"):
            DevicePolicy.resolve("mps")


def test_unknown_device_preference_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto, cpu, mps"):
        DevicePolicy.resolve("cuda")


def test_seed_everything_repeats_python_and_torch_sequences() -> None:
    seed_everything(17)
    python_first = [random.random() for _ in range(3)]
    torch_first = torch.rand(3)

    seed_everything(18)
    python_different = [random.random() for _ in range(3)]
    torch_different = torch.rand(3)

    seed_everything(17)
    assert [random.random() for _ in range(3)] == python_first
    torch.testing.assert_close(torch.rand(3), torch_first)
    assert python_different != python_first
    assert not torch.equal(torch_different, torch_first)
