import inference_lab


def test_package_has_semantic_version() -> None:
    assert inference_lab.__version__ == "0.1.0"
