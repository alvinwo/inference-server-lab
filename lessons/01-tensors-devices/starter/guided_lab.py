"""Lesson 1 guided lab: read one section, predict, run, and experiment.

This file uses ``# %%`` cell markers. VS Code and PyCharm can run each cell
individually, but the whole file also runs as an ordinary Python program:

    uv run python work/01-tensors-devices/guided_lab.py

Nothing in this file is unfinished. It is the worked example you study before
implementing the smaller production-shaped functions in ``inference_lab/``.
"""

import random
import time

import torch


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def synchronize(device: torch.device) -> None:
    """A tiny local helper; you will later build and test the reusable version."""
    if device.type == "mps":
        torch.mps.synchronize()


# %% [markdown]
# ## 1. A tensor is data plus metadata
#
# An inference server does not pass Python lists through a Transformer. It passes
# tensors. A tensor carries numerical values plus metadata that determines how
# PyTorch interprets and executes them: shape, dtype, and device.
#
# Before running this cell, predict the shape, dtype, and device.


# %%
def experiment_1_tensor_metadata() -> None:
    section("1/7 Tensors carry data and metadata")
    token_ids = torch.tensor([[10, 42, 7], [5, 5, 9]], dtype=torch.int64)

    print(f"values:\n{token_ids}")
    print(f"shape:  {tuple(token_ids.shape)}")
    print(f"dtype:  {token_ids.dtype}")
    print(f"device: {token_ids.device}")
    print("meaning: two requests, each containing three token IDs")

    assert token_ids.shape == (2, 3)
    assert token_ids.dtype is torch.int64


# %% [markdown]
# ## 2. Shape names are part of the design
#
# A shape such as ``[2, 3, 4]`` is much easier to reason about when we name it
# ``[batch, tokens, hidden]``. The final dimension holds the vector describing
# one token. Adding a ``[hidden]`` bias broadcasts it over every request and token.
#
# Change ``batch`` or ``tokens`` after the first run. The code should still work.


# %%
def experiment_2_shapes_and_broadcasting() -> None:
    section("2/7 Shapes give dimensions meaning")
    batch, tokens, hidden = 2, 3, 4
    hidden_states = torch.zeros((batch, tokens, hidden))
    bias = torch.tensor([10.0, 20.0, 30.0, 40.0])
    shifted = hidden_states + bias

    print(f"hidden states: {tuple(hidden_states.shape)} [batch, tokens, hidden]")
    print(f"bias:          {tuple(bias.shape)} [hidden]")
    print(f"result:        {tuple(shifted.shape)}")
    print(f"one token after broadcasting: {shifted[0, 0].tolist()}")

    assert shifted.shape == (batch, tokens, hidden)
    torch.testing.assert_close(shifted[1, 2], bias)


# %% [markdown]
# ## 3. Affine projection changes the feature dimension
#
# Transformer attention and MLP blocks repeatedly apply ``x @ W.T + b``.
# ``x`` contains token vectors. Each row of ``W`` describes one output feature.
# PyTorch preserves all leading dimensions and changes only the final feature
# dimension: ``[batch, tokens, input] -> [batch, tokens, output]``.
#
# On paper, determine the output shape before running this cell.


# %%
def experiment_3_affine_projection() -> None:
    section("3/7 Affine projection changes features")
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [3, 2]
    bias = torch.tensor([10.0, 20.0, 30.0])  # [3]

    output = torch.matmul(inputs, weight.transpose(0, 1)) + bias

    print(f"inputs:  {tuple(inputs.shape)} [batch, tokens, input_features]")
    print(f"weight:  {tuple(weight.shape)} [output_features, input_features]")
    print(f"output:  {tuple(output.shape)} [batch, tokens, output_features]")
    print(f"first projected token: {output[0, 0].tolist()}")

    assert output.shape == (1, 2, 3)
    torch.testing.assert_close(output[0, 0], torch.tensor([11.0, 22.0, 33.0]))


# %% [markdown]
# ## 4. A device policy makes execution visible
#
# ``auto`` is a request, not a device. On an Apple Silicon Mac it can select MPS;
# elsewhere it selects CPU. Production logs and benchmark output must record the
# actual selection. Otherwise an apparent "MPS benchmark" may secretly be CPU.


# %%
def experiment_4_devices() -> None:
    section("4/7 Devices decide where work happens")
    requested = "auto"
    selected = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    values = torch.arange(4, dtype=torch.float32).to(selected)

    print(f"requested policy: {requested}")
    print(f"selected device:  {selected.type}")
    print(f"tensor reports:   {values.device.type}")
    print("rule: explicit 'mps' should raise if unavailable; only 'auto' may fall back")

    assert values.device.type == selected.type


# %% [markdown]
# ## 5. Seeds make one experiment repeatable
#
# A fixed seed is useful for debugging and comparisons. Python and PyTorch have
# separate random-number generators, so seed both. A seed does not promise the
# same values across different hardware, libraries, or nondeterministic kernels.


# %%
def experiment_5_random_seeds() -> None:
    section("5/7 Seeds make an experiment repeatable")
    random.seed(7)
    torch.manual_seed(7)
    python_first = random.random()
    torch_first = torch.rand(3)

    random.seed(7)
    torch.manual_seed(7)
    python_second = random.random()
    torch_second = torch.rand(3)

    print(f"Python repeats:  {python_first == python_second}")
    print(f"PyTorch repeats: {torch.equal(torch_first, torch_second)}")
    print(f"sample tensor:   {torch_first.tolist()}")

    assert python_first == python_second
    torch.testing.assert_close(torch_first, torch_second)


# %% [markdown]
# ## 6. Inference is not training
#
# Training records operations so gradients can later be computed. A serving
# engine only performs a forward pass, so it uses ``torch.inference_mode()`` to
# avoid that bookkeeping. This is separate from selecting CPU or MPS.


# %%
def experiment_6_inference_mode() -> None:
    section("6/7 Inference mode removes training bookkeeping")
    training_input = torch.ones(3, requires_grad=True)
    training_output = training_input * 2
    with torch.inference_mode():
        inference_output = training_input * 2

    print(f"training result tracks gradients:  {training_output.requires_grad}")
    print(f"inference result tracks gradients: {inference_output.requires_grad}")

    assert training_output.requires_grad
    assert not inference_output.requires_grad


# %% [markdown]
# ## 7. Measure completed work
#
# Accelerator operations can be asynchronous: Python may return while the device
# is still working. Warmup prevents one-time setup from dominating the sample.
# Synchronization places the timer around completed work rather than dispatch.
# The number printed here is an observation, never a pass/fail target.


# %%
def experiment_7_timing() -> None:
    section("7/7 Honest timing waits for completed work")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    matrix = torch.randn((128, 128), device=device)

    with torch.inference_mode():
        for _ in range(3):
            torch.matmul(matrix, matrix)
            synchronize(device)

        synchronize(device)
        started = time.perf_counter()
        torch.matmul(matrix, matrix)
        synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1_000

    print(f"device: {device.type}")
    print("warmup iterations: 3 (not included in the sample)")
    print(f"one synchronized sample: {elapsed_ms:.3f} ms")
    assert elapsed_ms >= 0


def main() -> None:
    print("Lesson 1 guided lab")
    print("Read one section, predict its result, run it, then change one value.")
    experiment_1_tensor_metadata()
    experiment_2_shapes_and_broadcasting()
    experiment_3_affine_projection()
    experiment_4_devices()
    experiment_5_random_seeds()
    experiment_6_inference_mode()
    experiment_7_timing()
    print("\nGuided lab complete. You now have the pieces used by the challenge.")


if __name__ == "__main__":
    main()
