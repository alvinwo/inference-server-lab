# Lesson 1: Tensors, devices, and trustworthy timing

Estimated time: 20–40 minutes for the concept lab, then 1–3 hours for the challenge.

## Scenario

You are about to build model layers and eventually run them inside an inference server. A tensor
operation that is correct for one two-dimensional CPU example can still fail when requests add
batch dimensions, when tensors use different dtypes, or when work runs asynchronously on an
accelerator.

Your first production-shaped task is to create a small, explicit tensor runtime. It must say which
device it actually selected, produce repeatable random inputs within one environment, apply a
batched affine projection, and measure completed work rather than merely the time needed to enqueue
it.

## Learning goals

By the end of this lesson, you will be able to:

- read tensor shapes as named dimensions such as batch, token, and feature;
- explain how matrix multiplication handles the last two dimensions and broadcasts leading ones;
- keep tensor dtype and device placement consistent across an operation;
- distinguish an explicit device request from an automatic policy;
- seed Python and PyTorch for repeatable experiments in one environment; and
- benchmark with warmup and device synchronization.

## Concept lab

Start the lesson from the repository root, then run the lab copied into your workspace:

```bash
uv run python -m course start 01
uv run python work/01-tensors-devices/concept_lab.py
```

The lab is executable documentation. Read it from top to bottom and change a few shapes while you
work. It covers:

1. a `[batch, tokens, features]` tensor;
2. adding a one-dimensional bias through broadcasting;
3. multiplying by a transposed `[output_features, input_features]` weight;
4. the storage difference between float32 and float16;
5. moving a tensor to CPU or MPS and asking the tensor where it lives;
6. repeating a random sequence with a fixed seed;
7. disabling autograd bookkeeping with `torch.inference_mode`; and
8. warming up and synchronizing before reporting one timing observation.

New terms:

- **dtype** is the numeric representation of each tensor element, such as `torch.float32`.
- **device** is where PyTorch stores and executes a tensor, such as `cpu` or `mps`.
- **broadcasting** expands compatible size-one or missing dimensions without manually copying data.
- **warmup** runs an operation before measurement so one-time initialization is not mistaken for
  steady-state cost.
- **synchronization** waits until queued accelerator work has completed.

## Baseline

Before editing anything, run:

```bash
uv run python -m course test 01
```

The expected baseline is `13 failed, 1 passed`. Those failures are the challenge—not an installation
problem. The passing test confirms that the provided `TimingStats` scaffold already summarizes raw
samples.

If you see a collection error or a different failure count, stop and inspect the first error before
writing code.

## Challenge

Implement the `NotImplementedError` sites inside `work/01-tensors-devices/inference_lab/`.

### 1. Resolve a visible device policy

Implement `DevicePolicy.resolve()` in `devices.py`:

- `cpu` always selects CPU.
- `mps` selects MPS only when `torch.backends.mps.is_available()` is true.
- An unavailable explicit `mps` request raises `DeviceUnavailableError`; it must not silently run on
  CPU.
- `auto` visibly selects MPS when available and CPU otherwise.
- Any other preference raises `ValueError` listing the supported choices.

The returned object records both the request and the actual `torch.device`. Its provided `report()`
method is the metadata later server startup logs will use.

Implement `seed_everything()` so the Python `random` module and PyTorch repeat their sequences when
given the same seed. Reproducibility means repeatability within the same software and hardware
environment; it does not promise identical values across PyTorch versions or CPU and MPS.

### 2. Implement a batched affine projection

Implement `affine()` in `tensor_ops.py` for the equation:

```text
output = inputs @ weight.T + bias
```

`inputs` may have shape `[input_features]`, `[tokens, input_features]`, or
`[batch, tokens, input_features]`. `weight` always has shape
`[output_features, input_features]`, and `bias` has shape `[output_features]`. Validate ranks,
feature dimensions, dtype, and device before computing the result. Let PyTorch broadcast the bias
over every leading dimension.

### 3. Measure completed work

Implement `synchronize()` and `benchmark_operation()` in `timing.py`:

- CPU synchronization is a no-op.
- MPS synchronization calls `torch.mps.synchronize()`.
- Other device types are rejected in this CPU/MPS lesson.
- Negative warmup and zero iterations are rejected.
- Warmup calls are not included in the samples.
- Synchronize before starting each timer and after the operation, then store milliseconds.

Keep the public interfaces and file names unchanged; later lessons build on them.

## Tests

Run the whole checkpoint after every small change:

```bash
uv run python -m course test 01
```

The visible tests check:

- explicit CPU, automatic selection, explicit MPS failure, and invalid preferences;
- repeatability for both Python and PyTorch random sequences;
- literal affine values, three-dimensional inputs, dtype/device preservation, and bad shapes;
- safe CPU synchronization and rejection of unsupported devices;
- warmup count, synchronization order, literal timing samples, and invalid iteration counts.

Read a failing assertion as a description of the next behavior to implement. Do not change the
tests to make the checkpoint green.

## Measurement

After all tests pass, run the controlled CPU benchmark:

```bash
uv run python -m course benchmark 01
```

It prints one JSON object containing the workload shape, dtype, requested and selected device,
warmup count, all latency samples, summary statistics, and software/platform metadata. There is no
speed threshold: the evidence is whether the method is honest and reproducible.

On an Apple Silicon Mac, compare the same workload with visible automatic device selection:

```bash
INFERENCE_LAB_DEVICE=auto uv run python -m course benchmark 01
```

Do not conclude that one device is universally faster from this tiny workload. Launch overhead,
shape, dtype, thermal state, and PyTorch version all matter.

## Tiered hints

### Conceptual hint

For affine projection, only the final input dimension participates in the matrix product. Every
dimension before it describes independent rows of feature vectors. The bias aligns with the new
final dimension.

For timing, an accelerator call can return before its work finishes. A wall-clock timer is useful
only when the boundaries surround completed work.

### Interface hint

- In `DevicePolicy.resolve`, decide the selected `torch.device` first, then construct the frozen
  dataclass with a human-readable reason.
- In `affine`, compare `inputs.shape[-1]` with `weight.shape[1]`, then transpose only dimensions 0
  and 1 of the two-dimensional weight.
- In `benchmark_operation`, warm up in one loop and collect samples in a separate loop.

### Near-solution hint

The affine computation needs one `torch.matmul` and one addition after validation. The measured loop
has this order: synchronize, read the clock, call the operation, synchronize, read the clock, convert
seconds to milliseconds. The warmup loop calls the operation and synchronizes but never reads the
clock.

## Reference solution

Finish the challenge and write your engineering journal before opening `solution/`. Then compare
behavior and reasoning, not just line-by-line syntax:

```bash
diff -ru work/01-tensors-devices/inference_lab \
  lessons/01-tensors-devices/solution/inference_lab
```

The reference solution is deliberately portable and readable. It is not proof that no other correct
implementation exists.

## Engineering journal

Create `work/01-tensors-devices/JOURNAL.md` and answer:

1. Which shape mistake took the longest to understand, and what dimension names clarified it?
2. What device did `auto` select, and how did the program make that visible?
3. Why is silent CPU fallback dangerous when interpreting a benchmark?
4. What changed between the first timing call and warmed-up samples?
5. Which guarantees does a fixed seed provide, and which does it not provide?
6. What limitation would you address before reusing this timing helper in a production benchmark?

## Industrial connection

This lesson is smaller than an inference engine, but its invariants appear everywhere in one:

- A model runner must keep weights, activations, positions, and cache tensors on compatible devices
  and dtypes.
- Batched token representations repeatedly pass through affine projections for attention and MLP
  layers.
- Startup and benchmark reports must identify the actual device; otherwise comparisons are not
  auditable.
- Accelerator benchmarks must account for asynchronous execution.

After completing the challenge, skim—not memorize—vLLM's
[`DeviceConfig`](https://github.com/vllm-project/vllm/blob/main/vllm/config/device.py) and SGLang's
[`ModelRunner`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py).
Notice how quickly real systems accumulate device, dtype, memory, and execution concerns around
model code.

Primary PyTorch references:

- [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [`torch.mps.synchronize`](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html)
- [`torch.matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [`torch.inference_mode`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
- [Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## Stretch challenge

Optional: add a benchmark mode that compares float32 and float16 on the selected device while still
reporting every metadata field. First write down what you expect, then measure. Do not make a later
lesson depend on this extension, and do not introduce a performance pass/fail threshold.
