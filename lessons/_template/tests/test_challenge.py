from inference_lab import implementation_status


def test_implementation_returns_ready() -> None:
    assert implementation_status() == "ready"
