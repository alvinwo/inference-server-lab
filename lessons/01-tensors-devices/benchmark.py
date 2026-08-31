import json
import os
import platform

import torch

from inference_lab import DevicePolicy, affine, benchmark_operation, seed_everything


def main() -> None:
    requested_device = os.environ.get("INFERENCE_LAB_DEVICE", "cpu")
    policy = DevicePolicy.resolve(requested_device, dtype=torch.float32)
    seed_everything(7)

    shape = (2, 16, 32)
    output_features = 64
    inputs = torch.randn(shape, device=policy.selected, dtype=policy.dtype)
    weight = torch.randn((output_features, shape[-1]), device=policy.selected, dtype=policy.dtype)
    bias = torch.randn(output_features, device=policy.selected, dtype=policy.dtype)

    def operation() -> torch.Tensor:
        return affine(inputs, weight, bias)

    with torch.inference_mode():
        stats = benchmark_operation(
            operation,
            device=policy.selected,
            warmup=3,
            iterations=10,
        )

    print(
        json.dumps(
            {
                "lesson_id": "01-tensors-devices",
                **policy.report(),
                "shape": list(shape),
                "output_features": output_features,
                "warmup_iterations": stats.warmup_iterations,
                "iterations": stats.iterations,
                "samples_ms": stats.samples_ms,
                "median_ms": stats.median_ms,
                "minimum_ms": stats.minimum_ms,
                "maximum_ms": stats.maximum_ms,
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "platform": platform.platform(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
