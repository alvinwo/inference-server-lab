import pytest
import torch

from inference_lab import affine


def test_affine_matches_hand_calculated_values() -> None:
    inputs = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    weight = torch.tensor([[2.0, 0.0], [1.0, -1.0], [0.5, 0.5]])
    bias = torch.tensor([1.0, 2.0, -1.0])
    expected = torch.tensor([[3.0, 1.0, 0.5], [-1.0, -2.0, 0.0]])
    torch.testing.assert_close(affine(inputs, weight, bias), expected)


def test_affine_supports_batch_and_token_dimensions() -> None:
    inputs = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 1.0], [0.0, 2.0]],
        ]
    )
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.tensor([10.0, -10.0])
    expected = torch.tensor(
        [
            [[11.0, -8.0], [13.0, -6.0]],
            [[9.0, -9.0], [10.0, -8.0]],
        ]
    )
    output = affine(inputs, weight, bias)
    assert output.shape == (2, 2, 2)
    torch.testing.assert_close(output, expected)


def test_affine_preserves_dtype_and_device() -> None:
    inputs = torch.ones((2, 3), dtype=torch.float32, device="cpu")
    weight = torch.ones((4, 3), dtype=torch.float32, device="cpu")
    bias = torch.zeros(4, dtype=torch.float32, device="cpu")
    output = affine(inputs, weight, bias)
    assert output.dtype is torch.float32
    assert output.device == torch.device("cpu")


def test_affine_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        affine(torch.tensor(1.0), torch.ones((2, 1)), torch.ones(2))
    with pytest.raises(ValueError, match="weight must have shape"):
        affine(torch.ones((2, 3)), torch.ones(3), torch.ones(3))
    with pytest.raises(ValueError, match="input feature dimension"):
        affine(torch.ones((2, 3)), torch.ones((4, 2)), torch.ones(4))
    with pytest.raises(ValueError, match="bias must have shape"):
        affine(torch.ones((2, 3)), torch.ones((4, 3)), torch.ones(3))
    with pytest.raises(ValueError, match="same dtype"):
        affine(
            torch.ones((2, 3), dtype=torch.float32),
            torch.ones((4, 3), dtype=torch.float64),
            torch.ones(4, dtype=torch.float32),
        )
