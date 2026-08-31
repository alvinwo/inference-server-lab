# Build an Inference Server, One Challenge at a Time

Inference Server Lab is an open-source, educational journey from Python programs to a
production-shaped LLM inference server. You build the mechanisms yourself, measure them, and then
connect each idea to systems such as vLLM and SGLang.

## Who this is for

This course is for developers who are comfortable with Python but new to PyTorch, GPU programming,
and AI infrastructure. Each lesson isolates one mechanism, gives you tests and tiered hints, and
ends at a working checkpoint.

## What you will build

Version 1 grows from tensors and a tiny decoder-only Transformer into real-model serving,
streaming, per-request KV caches, continuous batching, block-based memory management, metrics,
an OpenAI-compatible API subset, and controlled load benchmarks. Version 2 can add prefix caching,
chunked prefill, structured output, speculative decoding, and quantization after the core engine is
stable.

## Hardware: CPU required, Apple Silicon/MPS optional, NVIDIA not required

Every required correctness path works on a normal laptop CPU. Apple Silicon learners may use MPS
where a lesson supports it, but the lesson must report the device actually selected. NVIDIA hardware
is not required. Performance numbers are meaningful only when compared on the same machine and
workload.

## Five-minute setup

Install [uv](https://docs.astral.sh/uv/), clone this repository, and run:

```bash
uv sync --python 3.12
uv run python -m course list
uv run python scripts/verify_checkpoints.py
```

The list shows each lesson as `published` or `planned`; a lesson can be started only after its
starter, tests, solution, notes, and benchmark are published. For a published lesson, use:

```bash
uv run python -m course start 01
uv run python -m course test 01
uv run python -m course benchmark 01
uv run python -m course reset 01 --yes
```

Reset is recoverable: it moves your workspace into `work/.trash/` instead of deleting it.

Lesson 1 is now published. Its learner guide is
[`lessons/01-tensors-devices/README.md`](lessons/01-tensors-devices/README.md).

## Version 1 lesson map

| Phase | Lessons | Result |
| --- | --- | --- |
| Foundations | 01–03 | Device-aware tensor work, a tiny Transformer, and generation |
| First real server | 04–07 | Model adapter, real weights, HTTP/streaming lifecycle, benchmarks |
| Inference engine | 08–11 | KV cache, padded and continuous batching, block memory manager |
| Production shape | 12–14 | Compatibility API, observability, overload safety, graduation load test |

See [the roadmap](docs/roadmap.md) for lesson titles and delivery milestones.

## How starter, tests, solution, and engineering notes work

`course start` copies only the lesson's `starter/` into your private, gitignored workspace. The
starter deliberately fails only the challenge tests named by that lesson. `course test` runs the
same acceptance tests with your workspace first on Python's import path. When you finish—or after
you have made a serious attempt—compare your design with `solution/` and its engineering notes.
The solution is a reference with tradeoffs, not the only acceptable design.

## Educational limitations

This project teaches industrial boundaries and invariants, but it is not a production replacement
for vLLM, SGLang, or a managed inference platform. Early implementations favor clarity over custom
kernels, distributed execution, broad model compatibility, or hardened multi-tenant security. Do
not expose the educational server to untrusted traffic.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing code or lessons. Contributions should keep
CPU correctness offline, preserve starter/solution isolation, and explain ideas for learners new to
AI infrastructure.

## License

Licensed under the [Apache License 2.0](LICENSE).
