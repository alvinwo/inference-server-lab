import torch


def affine(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply ``inputs @ weight.T + bias`` over any leading batch dimensions."""
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [output_features, input_features]")
    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError("input feature dimension must match weight input features")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias must have shape [output_features]")
    if not (inputs.dtype == weight.dtype == bias.dtype):
        raise ValueError("inputs, weight, and bias must have the same dtype")
    if not (inputs.device == weight.device == bias.device):
        raise ValueError("inputs, weight, and bias must be on the same device")
    return torch.matmul(inputs, weight.transpose(0, 1)) + bias
