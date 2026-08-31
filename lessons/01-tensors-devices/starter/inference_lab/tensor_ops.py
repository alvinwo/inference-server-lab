import torch


def affine(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply ``inputs @ weight.T + bias`` over any leading batch dimensions."""
    del inputs, weight, bias
    raise NotImplementedError("Implement batched affine projection")
