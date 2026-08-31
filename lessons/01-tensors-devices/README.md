# Lesson 1: Tensors, devices, and trustworthy timing

This is a learn-by-doing chapter. You are not expected to know PyTorch or GPU programming before
starting. Read a small idea, run real code, change it, observe the result, and only then implement
the corresponding piece of the inference runtime.

Estimated time: 60–90 minutes for the guided chapter and 1–3 hours for the implementation.

## What are we building?

An inference server turns text requests into generated text:

```text
"Explain KV cache"
        │
        ▼
tokenizer → token IDs → Transformer → next-token scores → sampler → "A"
                       ▲
                       │
                 repeated many times
```

The Transformer does not operate on strings. It operates on **tensors**: multidimensional arrays
with a shape, numeric type, and execution device. Before we can build attention, KV cache, batching,
or a scheduler, we need a small reliable layer underneath them:

```text
DevicePolicy ── chooses and reports CPU or MPS
      │
      ▼
tensor inputs ── affine projection: x @ W.T + b ── tensor outputs
      │                                                   │
      └──────── trustworthy synchronized timing ──────────┘
```

That is Lesson 1. You are not building an HTTP server yet. You are building three foundations that
every later server component will rely on.

## The four files and what they mean

Your workspace is `work/01-tensors-devices/`.

| File | Role in this chapter | Later role in the server |
|---|---|---|
| `guided_lab.py` | Complete experiments to read, run, and modify | Builds your PyTorch mental model |
| `inference_lab/devices.py` | Select a device and seed randomness | Server startup configuration |
| `inference_lab/tensor_ops.py` | Apply one model-like projection | The core operation inside model layers |
| `inference_lab/timing.py` | Measure completed work | Honest benchmarks and latency metrics |

The guided lab is complete. The three `inference_lab` files are the code you implement.

## How to study this chapter

For every experiment, use this loop:

1. Read the explanation and the corresponding section of `guided_lab.py`.
2. Predict the shape, value, or behavior before running it.
3. Run the code and compare the output with your prediction.
4. Change one value or shape and run it again.
5. Explain the result aloud or in one sentence.
6. Implement the small runtime step that uses the same idea.

Do not begin with the full test suite. It intentionally contains all unfinished behaviors and is
useful only at the end.

## Setup and first run

From the repository root:

```bash
uv sync --frozen --python 3.12
uv run python work/01-tensors-devices/guided_lab.py
uv run python -m course steps 01
```

`guided_lab.py` uses `# %%` cell markers. You may run it as one normal Python program, or open it in
VS Code or PyCharm and run one cell at a time. No Jupyter setup is required.

---

## Part 1 — Tensors carry data and metadata

A Python integer such as `42` has one value. A tensor can contain millions or billions of values.
It also carries three pieces of metadata that matter throughout an inference engine:

- **shape**: how values are arranged and what dimensions mean;
- **dtype**: how each value is represented, such as `torch.float32`; and
- **device**: where values are stored and operations execute, such as `cpu` or `mps`.

In the first guided experiment, token IDs have shape `[2, 3]`:

```text
[[10, 42, 7],    ← request 0 has three tokens
 [ 5,  5, 9]]    ← request 1 has three tokens
```

The dimension names are `[batch, tokens]`. `batch=2` means two requests are processed together.

Try this in `guided_lab.py`:

1. Change the tensor to contain three rows. What is the new batch dimension?
2. Remove one token from only the first row. Why can PyTorch no longer create one rectangular
   tensor from that nested list?
3. Change `dtype=torch.int64` to `torch.float32`. What changes in the output, and why are integer
   IDs more appropriate for tokenizer output?

Inference connection: later, continuous batching decides which requests share a batch. Tensor
shapes are how that scheduling decision reaches the model.

## Part 2 — Shapes give dimensions meaning

A Transformer usually represents each token with a vector:

```text
[batch, tokens, hidden]
   2       3       4
```

This shape means two requests, three token positions per request, and four numbers describing each
token. Real models use much larger hidden dimensions, but the rules are identical.

The guided lab adds a bias with shape `[hidden]` to a tensor with shape
`[batch, tokens, hidden]`. PyTorch **broadcasts** the bias across every batch item and token. It acts
as if the bias had been copied, without requiring you to make those copies manually.

Try this:

1. Change `batch` from 2 to 5. Does the bias code change?
2. Change `hidden` from 4 to 5 without changing the bias. Read the error and identify the
   incompatible dimension.
3. Repair the bias so its length matches `hidden`.

Rule to remember: operations align dimensions from the right. A `[hidden]` bias naturally aligns
with the last dimension of `[batch, tokens, hidden]`.

## Part 3 — Affine projection is everywhere in a Transformer

The equation

```text
output = inputs @ weight.T + bias
```

is an affine projection. Attention uses projections to create queries, keys, and values. MLP layers
use larger projections to expand and contract token representations. The operation changes the
last dimension while preserving the leading dimensions:

```text
inputs:  [batch, tokens, input_features]
weight:  [output_features, input_features]
bias:    [output_features]
output:  [batch, tokens, output_features]
```

Why transpose the weight? Matrix multiplication needs the inner dimensions to match:

```text
[tokens, input_features] @ [input_features, output_features]
```

but model weights are conventionally stored as `[output_features, input_features]`.

In guided experiment 3, calculate the first output token by hand before running the code. Then:

1. Remove `weight.transpose(0, 1)` and read the shape error.
2. Change the bias to zeros. Which part of the output changes?
3. Add another batch item. Confirm that only the leading batch dimension changes.

## Part 4 — Device selection must be visible

On your Mac, PyTorch may use:

- `cpu`: always available; or
- `mps`: Apple's Metal acceleration backend, available on supported Apple Silicon/macOS setups.

This course distinguishes a **request** from a **selection**:

```text
requested="auto"  → selected="mps" if available, otherwise "cpu"
requested="cpu"   → selected="cpu"
requested="mps"   → selected="mps", or raise an error
```

An explicit MPS request must never silently fall back to CPU. Silent fallback produces misleading
benchmark reports and can hide deployment mistakes.

In guided experiment 4, compare the requested policy with the device reported by the tensor. Try
creating a CPU tensor explicitly and printing its `.device`.

## Part 5 — Repeatable experiments need seeds

Random weights and inputs make experiments useful, but they also make debugging confusing. A seed
sets the starting state of a random-number generator. Python and PyTorch use separate generators,
so seed both.

Run guided experiment 5, then:

1. Change only the second seed from 7 to 8. Confirm the values differ.
2. Seed Python but not PyTorch. Which sequence repeats?
3. Restart the program with the original seed. Confirm the sample repeats in your environment.

A seed improves repeatability; it does not guarantee identical values across every PyTorch version,
device, or nondeterministic operation.

## Part 6 — Inference mode is different from device selection

Training records operations so gradients can be calculated later. Inference only needs the forward
result. `torch.inference_mode()` disables that training bookkeeping.

This is independent of device selection:

```text
device policy     answers: where does this execute?
inference mode    answers: should PyTorch record training information?
```

Run guided experiment 6 and inspect `requires_grad` on both outputs.

## Part 7 — Accelerator timing needs synchronization

CPU operations normally finish before the Python call returns. Accelerator calls may enqueue work
and return while the accelerator is still busy. Timing only the Python call can therefore measure
dispatch time instead of completed computation.

A trustworthy basic sample looks like this:

```text
warm up several times (not measured)
synchronize
start clock
run operation
synchronize
stop clock
```

Warmup removes one-time initialization from the measured samples. Synchronization creates honest
timer boundaries. Run guided experiment 7 twice and observe that the values vary; a benchmark
should keep multiple raw samples rather than treating one number as truth.

## Concept summary

- A tensor combines values with shape, dtype, and device metadata.
- Dimension names such as `[batch, tokens, hidden]` turn shapes into a model of the data.
- Affine projection preserves leading dimensions and changes the final feature dimension.
- Requested device and selected device are different facts; report both.
- A fixed seed improves repeatability inside one environment but is not a universal guarantee.
- Inference mode disables training bookkeeping; it does not choose an execution device.
- Warmup and synchronization are necessary for a basic honest accelerator timing sample.

---

# Build the runtime, one checkpoint at a time

You now know what the three runtime modules mean. Each step below asks you to transfer one pattern
from `guided_lab.py` into reusable, tested code.

List the checkpoints at any time:

```bash
uv run python -m course steps 01
```

## Step 1 — Seed Python and PyTorch

Open `work/01-tensors-devices/inference_lab/devices.py` and find `seed_everything()`.

Why it exists: a server benchmark needs repeatable inputs when comparing two implementations.

Implement this recipe:

```text
give the seed to Python's random module
give the same seed to PyTorch
if MPS is available, give the seed to torch.mps too
```

Remove the placeholder `del seed` and `NotImplementedError`. The imports are already present.

Run only this checkpoint:

```bash
uv run python -m course test 01 --step seed
```

Done means `1 passed`. Do not work on the other functions yet.

## Step 2 — Represent an explicit CPU decision

Find `DevicePolicy.resolve()` in `devices.py`. The `DevicePolicy` dataclass is a record containing:

- `requested`: what the user asked for;
- `selected`: the actual `torch.device`;
- `dtype`: the numeric representation; and
- `reason`: a sentence suitable for logs.

Start with only the CPU case:

```text
if requested is "cpu":
    return a DevicePolicy containing torch.device("cpu"), the supplied dtype,
    and a non-empty explanation
otherwise:
    keep raising NotImplementedError for now
```

Run:

```bash
uv run python -m course test 01 --step cpu-policy
```

Done means `1 passed`.

## Step 3 — Complete device selection

Extend the same method in this order:

1. Reject anything outside `auto`, `cpu`, and `mps` with `ValueError`. Include the supported names
   in the message.
2. Keep your CPU branch.
3. For explicit `mps`, return MPS when `torch.backends.mps.is_available()` is true. Otherwise raise
   `DeviceUnavailableError` with `MPS` in the message.
4. For `auto`, select MPS when available and CPU otherwise. Record the reason.

Run:

```bash
uv run python -m course test 01 --step device-selection
```

Done means `3 passed`.

## Step 4 — Implement the affine calculation

Open `tensor_ops.py`. First implement the happy path you studied in guided experiment 3:

```text
transpose dimensions 0 and 1 of weight
matrix-multiply inputs by that transposed weight
add bias
return the result
```

Remove the placeholder `del` and `NotImplementedError` lines.

Run:

```bash
uv run python -m course test 01 --step affine-core
```

Done means `2 passed`.

## Step 5 — Protect the affine contract

Production code should reject invalid tensors near the boundary with a useful message. Before the
calculation, check:

1. `inputs` has at least one dimension;
2. `weight` has exactly two dimensions;
3. `inputs.shape[-1] == weight.shape[1]`;
4. `bias` has one dimension and `bias.shape[0] == weight.shape[0]`;
5. all three tensors have the same dtype; and
6. all three tensors are on the same device.

Raise `ValueError` for each violated rule. The test output tells you the meaningful phrase expected
in each message.

Run:

```bash
uv run python -m course test 01 --step affine-contract
```

Done means `2 passed`.

## Step 6 — Synchronize supported devices

Open `timing.py` and find `synchronize()`.

Implement three branches:

```text
cpu → return immediately; there is nothing extra to wait for
mps → call torch.mps.synchronize(), then return
anything else → raise ValueError explaining that this lesson supports cpu and mps
```

Run:

```bash
uv run python -m course test 01 --step synchronize
```

Done means `2 passed`.

## Step 7 — Benchmark completed operations

Find `benchmark_operation()` in `timing.py`. The provided `TimingStats` class stores your result.

Implement this sequence:

1. Reject `warmup < 0` and `iterations < 1` with `ValueError`.
2. Warmup loop: call the operation, then synchronize. Do not time or save these calls.
3. Measurement loop: synchronize, read `perf_counter()`, call the operation, synchronize, read the
   clock again, convert seconds to milliseconds, and append the sample.
4. Return `TimingStats(device.type, warmup, tuple(samples))`.

Run:

```bash
uv run python -m course test 01 --step benchmark
```

Done means `2 passed`.

## Integrate all the pieces

Now run the whole lesson:

```bash
uv run python -m course test 01
```

You are finished when it reports `14 passed`. Then collect a controlled CPU benchmark:

```bash
uv run python -m course benchmark 01
```

On a Mac with MPS available, compare automatic device selection:

```bash
INFERENCE_LAB_DEVICE=auto uv run python -m course benchmark 01
```

Do not use this tiny workload to declare one device universally faster. Shape, dtype, launch
overhead, software version, thermal state, and background load all affect the observation.

## Explain what you built

Create `work/01-tensors-devices/JOURNAL.md` and answer in your own words:

1. In `[batch, tokens, hidden]`, what does one value of each dimension represent?
2. Why does affine projection transpose the stored weight?
3. What is the difference between requested and selected device?
4. Why is silent fallback dangerous for a benchmark?
5. Why do we warm up and synchronize?
6. Where will each Lesson 1 component appear in a future inference server?

If you cannot explain one answer, return to the corresponding guided experiment and change one
thing. That loop—predict, run, observe, explain—is the point of the lesson.

## Reference solution

Only after all seven attempts, compare your implementation with
`lessons/01-tensors-devices/solution/inference_lab/`. Compare behavior and reasoning, not merely
syntax. The solution notes explain design choices and limitations.

## Primary references

- [PyTorch tensor introduction](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [`torch.matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [`torch.inference_mode`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
- [Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)

After finishing, skim vLLM's
[`DeviceConfig`](https://github.com/vllm-project/vllm/blob/main/vllm/config/device.py) and SGLang's
[`ModelRunner`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py).
You are not expected to understand them yet. Notice only that device, dtype, and execution concerns
surround the model code in real inference systems.
