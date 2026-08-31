import time

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    print("Lesson 1 concept lab: tensors, devices, and timing")

    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    print(f"1. shape: {tuple(tokens.shape)} = [batch, tokens, features]")
    assert tokens[0, 0].shape == (4,)

    bias = torch.tensor([10.0, 20.0, 30.0, 40.0])
    shifted = tokens + bias
    print(f"2. broadcasting: {tuple(tokens.shape)} + {tuple(bias.shape)}")
    assert shifted[1, 2, 3].item() == 63.0

    inputs = torch.ones((2, 3, 4))
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    projected = torch.matmul(inputs, weight.transpose(0, 1))
    print(f"3. matmul: {tuple(inputs.shape)} @ {tuple(weight.T.shape)}")
    assert projected.shape == (2, 3, 5)

    float32 = torch.ones(8, dtype=torch.float32)
    float16 = float32.to(torch.float16)
    print(
        "4. dtype bytes per element: "
        f"float32={float32.element_size()}, float16={float16.element_size()}"
    )

    selected = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    on_device = projected.to(selected)
    print(f"5. selected device: {selected.type}; tensor reports: {on_device.device.type}")
    assert on_device.device.type == selected.type

    torch.manual_seed(7)
    first = torch.rand(3)
    torch.manual_seed(7)
    second = torch.rand(3)
    print(f"6. seeded values repeat: {torch.equal(first, second)}")
    assert torch.equal(first, second)

    source = torch.ones(3, requires_grad=True)
    with torch.inference_mode():
        inference_output = source * 2
    print(f"7. inference output tracks gradients: {inference_output.requires_grad}")
    assert not inference_output.requires_grad

    matrix = torch.randn((128, 128), device=selected)
    for _ in range(3):
        torch.matmul(matrix, matrix)
        synchronize(selected)
    synchronize(selected)
    started = time.perf_counter()
    torch.matmul(matrix, matrix)
    synchronize(selected)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    print(f"8. synchronized sample: {elapsed_ms:.3f} ms (observation, not a target)")
    print("Concept lab complete. Continue with the challenge tests.")


if __name__ == "__main__":
    main()
