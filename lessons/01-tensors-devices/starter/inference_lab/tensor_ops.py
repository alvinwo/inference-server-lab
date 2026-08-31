import torch


def affine(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply ``inputs @ weight.T + bias`` over any leading batch dimensions.

    Chapter Step 4 implements the calculation shown in guided experiment 3.
    Step 5 then adds the six shape, dtype, and device contract checks before
    the calculation. Work through those checkpoints in order.
    """
    del inputs, weight, bias
    raise NotImplementedError("Implement batched affine projection")
