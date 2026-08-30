# Inference Server Lab: Project and Curriculum Design

**Status:** Delivery candidate

**Date:** 2026-08-29

**Working title:** Inference Server Lab

**Repository name:** `inference-server-lab`

**Python package:** `inference_lab`

## 1. Vision

Inference Server Lab is an open-source, challenge-based journey for Python developers who want to become AI infrastructure engineers. Learners gradually build a production-shaped large-language-model inference server rather than treating inference as a call to `model.generate()`.

The project has two equally important outputs:

1. A portable inference engine that grows from a tiny decoder-only Transformer into a single-machine server with streaming, KV caching, continuous batching, block-based memory management, observability, and load testing.
2. A course in which every lesson provides a runnable starter, executable tests, a reference implementation, benchmark evidence, and an engineering journal explaining decisions and mistakes.

The project is educational, not a production replacement for vLLM or SGLang. Its purpose is to make the core mechanisms small enough to understand while preserving the component boundaries, invariants, measurements, and failure modes found in industrial systems.

## 2. Audience and prerequisites

The primary learner:

- Is comfortable writing Python.
- Is new to PyTorch, GPU programming, and inference infrastructure.
- Has access to any modern laptop; a MacBook with Apple Silicon is the primary development environment.
- Should not need an NVIDIA GPU or paid cloud account for the core course.

The course assumes basic functions, classes, type hints, virtual environments, and command-line usage. It teaches tensor operations, neural-network inference, asynchronous request handling, scheduling, memory ownership, profiling, and performance analysis in context.

## 3. Learning outcomes

After Version 1, a learner should be able to:

- Explain tokenization, logits, sampling, autoregressive decoding, prefill, and decode.
- Implement a small Llama-like decoder-only Transformer in PyTorch.
- Load configuration, tokenizer data, and safetensor weights for a supported model family.
- Explain why an inference server separates API handling, scheduling, model execution, and KV-cache management.
- Stream text correctly, including tokens that do not independently form valid decoded text.
- Implement and validate a KV cache.
- Explain static batching, continuous batching, iteration-level scheduling, and token budgets.
- Implement a block-based KV allocator with explicit ownership and capacity limits.
- Measure TTFT, TPOT, inter-token latency, end-to-end latency, throughput, queue delay, and cache utilization.
- Handle overload, cancellation, timeout, and internal failure without leaking memory or killing the engine.
- Read vLLM and SGLang architecture documentation with a working mental model.
- Identify which parts of the learning server are general and which are model-, device-, or kernel-specific.

## 4. Scope

### Version 1: required single-machine track

- CPU execution on all supported laptops.
- Optional PyTorch MPS acceleration on Apple Silicon.
- One model loaded at a time.
- Decoder-only, Llama-like text generation.
- Tiny deterministic model for fast tests.
- `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `83212e1e2b3cfd6958f3707877bb878945dea8ee` as the initial real-model target. It is small, Llama-compatible, uses safetensors, and is Apache-2.0 licensed.
- Native teaching API first, followed by an OpenAI-compatible subset.
- Token streaming and cancellation.
- KV caching.
- Padded/static batching as a stepping stone.
- Continuous batching with an explicit token budget.
- Block-based KV memory allocation, admission control, and backpressure.
- Correctness, integration, load, and invariant tests.
- Metrics, structured logs, profiler exercises, and reproducible benchmark reports.

### Version 2: advanced single-machine track

- Automatic prefix caching.
- Chunked prefill and prefill/decode fairness.
- Structured output through token masking.
- Speculative decoding with a small draft model.
- Quantization concepts and a portable reference implementation, with backend-specific acceleration clearly separated.

### Later optional tracks

- CUDA kernels and paged-attention kernels.
- Tensor, pipeline, and data parallelism.
- Multi-process and multi-node serving.
- Mixture-of-experts and multimodal architectures.
- LoRA serving, KV offload, and prefill/decode disaggregation.

### Explicit non-goals for Version 1

- Universal model support.
- Training or fine-tuning.
- Custom CUDA or Metal kernels.
- Multi-GPU execution.
- Authentication, billing, tenancy, or production security hardening.
- Exact performance parity with vLLM, SGLang, llama.cpp, or MLX.
- A promise that the educational server is safe for production workloads.

## 5. Design principles

### 5.1 Encounter the problem before the mechanism

Every optimization follows this sequence:

`baseline → observe bottleneck → form hypothesis → implement → verify correctness → benchmark → interpret`

KV caching is introduced only after learners measure repeated attention work. Continuous batching appears only after learners see static batches stall behind unequal output lengths.

### 5.2 Correct before fast

Every optimized path has a simple reference path. Learners compare logits, tokens, state transitions, and allocator state before interpreting speed measurements.

### 5.3 Portable core, optional acceleration

CPU is the correctness baseline. MPS is an optional execution device, not a separate course. Device choice, dtype, synchronization, and fallback behavior are explicit so a silent CPU fallback cannot invalidate a benchmark.

### 5.4 Industrial boundaries, educational implementations

The engine mirrors the separation used by modern inference systems—API, engine core, scheduler, model runner, sampler, and KV manager—but begins in one process. Interfaces leave room for future workers without introducing unused distributed abstractions.

### 5.5 Your journey is part of the product

The reference solution is not only finished code. Each lesson records what was attempted, what failed, what changed, measured results, remaining limitations, and how the mechanism appears in industrial engines.

### 5.6 No network requirement for correctness tests

Unit and integration tests use tiny local fixtures. Downloading a real model is an explicit lesson action and is cached. Network-dependent tests are marked separately and never block the core CI suite.

## 6. Repository design

```text
inference-server-lab/
├── README.md
├── LICENSE
├── pyproject.toml
├── course/                         # learner workspace/checkpoint CLI
├── src/inference_lab/              # polished Version 1 engine
├── lessons/
│   ├── 01-tensors-devices/
│   │   ├── README.md               # lesson and challenge brief
│   │   ├── starter/inference_lab/  # starting checkpoint
│   │   ├── tests/                  # visible acceptance tests
│   │   └── solution/
│   │       ├── inference_lab/      # reference implementation
│   │       └── NOTES.md            # engineering journal
│   └── ...
├── labs/                           # small isolated concept experiments
├── benchmarks/                     # workloads, runner, report schema
├── docs/
│   ├── concepts/
│   ├── glossary.md
│   ├── architecture.md
│   └── roadmap.md
├── scripts/                        # course/checkpoint verification
├── tests/                          # final-engine tests
└── .github/                        # CI and contribution templates
```

Each lesson snapshot is intentionally self-contained. Duplication is accepted because it makes any checkpoint browsable and runnable without reconstructing Git history. A verification script ensures that:

- A starter passes every prerequisite test from earlier lessons.
- The current challenge tests fail only in the expected places before implementation.
- The solution passes the complete lesson test set.
- The final engine matches the last Version 1 solution behavior.

Learner work is created under a gitignored `work/<lesson>/` directory so course files remain pristine. A small Python course CLI exposes consistent commands:

```text
python -m course list
python -m course start 08
python -m course test 08
python -m course benchmark 08
python -m course reset 08
```

The CLI copies the starter, prints the goal and next command, runs the correct tests, and never modifies or reveals the solution.

## 7. Repeating lesson contract

Every lesson uses the same structure:

1. **Scenario:** A concrete symptom, user need, or production-shaped failure.
2. **Learning goals:** Three to five observable abilities.
3. **Concept lab:** A 20–40 minute isolated experiment when the main challenge would otherwise introduce too many ideas at once.
4. **Baseline:** A runnable command and expected output before any edits.
5. **Challenge:** One primary mechanism with stable interfaces and constrained TODOs.
6. **Tests:** Named acceptance criteria, including edge cases and invariants.
7. **Measurement:** A profile, trace, counter, or before/after benchmark.
8. **Tiered hints:** A conceptual hint, an interface hint, and a near-solution hint.
9. **Reference solution:** Complete code plus a readable diff from the starter.
10. **Engineering journal:** What was tried, why the final design was chosen, surprising results, and remaining limitations.
11. **Industrial connection:** Specific concepts and source-reading pointers for vLLM and SGLang.
12. **Stretch challenge:** Optional exploration that is never required by the next lesson.

Core challenges should take roughly 1–3 hours after the concept lab. Every lesson ends in a working checkpoint; no lesson ends with a half-integrated subsystem required merely to begin the next one.

## 8. Version 1 curriculum

### Phase A: foundations

#### Lesson 1 — Tensors, devices, and trustworthy timing

- **Scenario:** The same operation behaves differently across shapes, dtypes, CPU, and MPS.
- **Challenge:** Implement device selection and a small tensor-operation toolkit used by later model layers.
- **Concepts:** Tensor shapes, broadcasting, matrix multiplication, dtype, inference mode, device transfer, seeding, warmup, and device synchronization.
- **Evidence:** Shape and numeric tests; a benchmark that correctly synchronizes asynchronous devices.
- **Artifact:** `DevicePolicy` that reports the actual device and never silently changes it.

#### Lesson 2 — A tiny decoder-only Transformer

- **Scenario:** To optimize generation, learners first need a model they can inspect end to end.
- **Labs:** Embeddings and linear layers; RMSNorm and MLP; masked self-attention; rotary position embeddings; grouped-query attention.
- **Challenge:** Assemble a small Llama-like decoder with a plain full-sequence forward pass.
- **Evidence:** Layer-level numerical tests, causal-mask tests, shape tests, and deterministic logits.
- **Artifact:** A tiny seeded model with a deliberately small vocabulary and context length.

#### Lesson 3 — Autoregressive generation and sampling

- **Scenario:** A forward pass produces logits, not text.
- **Challenge:** Implement greedy decoding, temperature, top-k sampling, EOS handling, maximum-token limits, and reproducible seeds.
- **Concepts:** Tokenization boundaries, logits, probability distributions, stop reasons, and the difference between model execution and generation policy.
- **Evidence:** Deterministic sequences, statistical sampling sanity checks, and termination tests.
- **Artifact:** A synchronous generation loop that intentionally has no KV cache.

### Phase B: first real server

#### Lesson 4 — Model adapters and real weights

- **Scenario:** A general engine cannot hardcode every model’s layer names and tensor layout.
- **Challenge:** Define a narrow `ModelAdapter` contract and load a pinned SmolLM2 checkpoint using Hugging Face configuration, tokenizer files, and safetensors.
- **Boundary:** Transformers may be used as an optional test oracle, but its generation loop is not used by the engine.
- **Evidence:** Weight-name mapping tests and logit comparisons with an oracle on short fixed inputs.
- **Artifact:** One production-realistic Llama-family adapter plus the tiny test adapter.

#### Lesson 5 — Prefill, decode, and the naive HTTP server

- **Scenario:** A useful model needs a request boundary, but one request at a time is visibly inefficient.
- **Challenge:** Expose a minimal native generation endpoint around the synchronous loop.
- **Concepts:** Prompt tokenization, prefill versus decode, request validation, finish reasons, and end-to-end latency.
- **Evidence:** HTTP integration tests and a concurrency demonstration showing head-of-line blocking.
- **Artifact:** A correct one-at-a-time server used as the reference baseline.

#### Lesson 6 — Request lifecycle, streaming, and cancellation

- **Scenario:** Users should see tokens before generation finishes and should be able to disconnect.
- **Challenge:** Add a request state machine, an asynchronous ingress queue, per-request output channels, Server-Sent Events, incremental detokenization, timeout, and cancellation.
- **Execution model:** The API runs on an asyncio loop; a dedicated engine thread owns scheduling and model execution. This prevents a forward pass from blocking connection handling while keeping Version 1 single-process.
- **Evidence:** State-transition tests, UTF-8/token-boundary streaming tests, disconnect tests, and resource-release assertions.
- **Artifact:** An asynchronous facade over the still-serial engine.

### Phase C: measure and build the engine

#### Lesson 7 — Serving benchmarks and profiler literacy

- **Scenario:** “It feels faster” is not evidence.
- **Challenge:** Build a repeatable benchmark harness with warmup and environment capture.
- **Metrics:** Time to first token (TTFT), time per output token (TPOT), inter-token latency (ITL), end-to-end latency, request throughput, input/output token throughput, queue time, and error rate.
- **Workloads:** Long-prefill/short-decode, short-prefill/long-decode, mixed lengths, burst traffic, and fixed concurrency.
- **Evidence:** A machine-readable JSON report and a short interpretation, not a universal pass/fail speed threshold.
- **Artifact:** The benchmark foundation used by every optimization lesson.

#### Lesson 8 — Per-request KV cache

- **Scenario:** Uncached decoding recomputes attention for the entire prefix at every output token.
- **Challenge:** Change attention and the model runner to append and reuse per-layer keys and values.
- **Concepts:** KV tensor shapes, positions, grouped-query attention, cache growth, ownership, and prefill/decode differences.
- **Evidence:** Cached and uncached logits/tokens must match within dtype tolerance; compute-shape and timing comparisons show removed work.
- **Artifact:** A simple contiguous cache per request.

#### Lesson 9 — Padded batching

- **Scenario:** Running requests independently underuses vectorized hardware.
- **Challenge:** Batch prefill and decode across sequences with different lengths using masks, padding, and position metadata.
- **Concepts:** Batch dimensions, padding waste, ragged logical state versus dense tensors, and batch-level sampling.
- **Evidence:** Batched results match independent execution; a workload exposes wasted padding and slow-request coupling.
- **Artifact:** Static batched execution that is correct but intentionally inflexible.

#### Lesson 10 — Continuous batching and token-budget scheduling

- **Scenario:** Static batches wait for the longest generation and cannot admit new work.
- **Challenge:** Implement iteration-level scheduling so requests join and leave between model steps.
- **Scheduler contract:** `schedule(engine_state, token_budget) -> SchedulePlan` returns admitted requests, scheduled token counts, and capacity decisions without executing the model.
- **Policy:** FCFS first, with decode progress protected from starvation. Prefill and decode are explicit in traces even if the scheduler later unifies them under one token budget.
- **Evidence:** Deterministic scheduler simulations, mixed-length integration tests, timeline traces, and throughput/latency comparison with static batching.
- **Artifact:** Waiting/running queues and a continuous engine step loop.

#### Lesson 11 — Block-based KV memory management

- **Scenario:** Contiguous per-request caches cause fragmentation, unpredictable growth, and weak admission control.
- **Challenge:** Preallocate KV blocks, map logical sequence positions to physical blocks, reserve capacity, append tokens, and release ownership on every terminal path.
- **Concepts:** Block tables, free lists, reference counts, internal fragmentation, capacity watermarks, and preemption by recomputation.
- **Evidence:** Allocator invariant and property tests, forced-capacity tests, cancellation/failure leak tests, and cache-utilization metrics.
- **Portability note:** The portable PyTorch attention path gathers through a block table. It teaches PagedAttention’s memory model but does not claim the performance of a fused paged-attention kernel.
- **Artifact:** `KVCacheManager` and allocator interfaces that Version 2 prefix caching can extend.

### Phase D: production shape

#### Lesson 12 — OpenAI-compatible API subset

- **Scenario:** Clients should not need a custom SDK to exercise the engine.
- **Challenge:** Implement `/v1/models`, `/v1/completions`, and `/v1/chat/completions`, including streaming chunks, chat templates, usage fields, finish reasons, and a consistent error envelope.
- **Evidence:** Schema tests, a standard OpenAI client smoke test, disconnect tests, and compatibility fixtures.
- **Artifact:** A documented compatibility subset; unsupported parameters fail clearly rather than being ignored.

#### Lesson 13 — Observability and operational safety

- **Scenario:** A server can be correct while queues, memory pressure, or latency silently deteriorate.
- **Challenge:** Add structured logs, request IDs, liveness/readiness, and Prometheus-style metrics.
- **Signals:** Queue depth and time, running requests, scheduled tokens, batch size, KV utilization, preemptions, cancellations, TTFT, TPOT, request duration, and failures by category.
- **Evidence:** Metric-value integration tests, log-correlation tests, readiness behavior, and bounded-label-cardinality review.
- **Artifact:** A small operational dashboard example and a failure-triage guide.

#### Lesson 14 — Load, resilience, and graduation benchmark

- **Scenario:** The final engine must remain correct under mixed concurrent traffic and constrained memory.
- **Challenge:** Add bounded queues, overload responses, graceful shutdown, startup warmup, deterministic fault injection, and a final load suite.
- **Error policy:** Invalid request → client error; full ingress queue → overload error; unavailable model → service unavailable; per-request execution failure → request failure plus cleanup; corrupted engine-wide state → fail readiness and shut down safely.
- **Evidence:** Closed-loop concurrency and open-loop arrival tests, burst tests, forced OOM/capacity tests, cancellation storms, graceful-shutdown tests, and comparison against the Lesson 5 baseline.
- **Artifact:** A reproducible graduation report explaining throughput/latency trade-offs and the remaining gap to an industrial engine.

## 9. Runtime architecture

### 9.1 Components

- **API:** Validates and tokenizes input, applies chat templates, creates requests, streams decoded output, and maps domain failures to HTTP responses.
- **Engine:** Owns the request registry and lifecycle, receives control messages, calls the scheduler, dispatches plans, and publishes outputs.
- **Scheduler:** Maintains waiting and running collections and makes pure scheduling decisions under token and KV-capacity budgets.
- **Model runner:** Converts a `SchedulePlan` into tensors, performs prefill or decode, updates KV state, and returns logits or sampled tokens.
- **Model adapters:** Define model construction, weight mapping, attention metadata, and supported capabilities for a model family.
- **Sampler:** Applies temperature and top-k policies with per-request random-generator state.
- **KV cache manager:** Owns physical cache storage, logical-to-physical block tables, allocation, reference counts, and cleanup.
- **Detokenization stream:** Converts incremental token IDs to valid text chunks without assuming one token equals one Unicode string.
- **Metrics and tracing:** Observe boundaries without controlling them.

### 9.2 Request lifecycle

```text
RECEIVED → VALIDATED → QUEUED → RUNNING → FINISHED
                                ├──────→ CANCELLED
                                └──────→ FAILED
```

Only the engine changes request state. Every terminal transition is idempotent and invokes resource cleanup exactly once. API disconnects send cancellation commands; they do not directly mutate engine data.

### 9.3 Data flow

```text
HTTP/SSE client
  → API validation, tokenization, chat template
  → thread-safe ingress/control queue
  → engine request registry
  → scheduler plan under token + memory budgets
  → model runner prefill/decode
  → sampler
  → per-request token event stream
  → incremental detokenizer
  → SSE response
```

### 9.4 Core interfaces

Interfaces are typed and narrow. Exact method names may change during implementation, but these ownership boundaries are requirements:

- `ModelAdapter`: configuration, model construction, weight loading, capability declaration.
- `Scheduler`: pure selection and admission decisions; no tensor execution.
- `ModelRunner`: tensor execution; no HTTP behavior.
- `KVCacheManager`: cache capacity and ownership; no request prioritization.
- `RequestStream`: token, completion, and failure events; no engine mutation.
- `MetricsSink`: observations only; business logic must work with a no-op sink.

## 10. Device and model strategy

### 10.1 Device behavior

- `cpu` always works and is the numeric reference.
- `mps` is selected only when explicitly requested or through a visible `auto` policy.
- Startup prints the chosen device, dtype, model memory estimate, and cache budget.
- Unsupported MPS operations fail with an actionable message by default. Global silent CPU fallback is not enabled for benchmarks.
- Benchmarks perform warmup and device synchronization around measured regions.
- Correctness tolerances are dtype-aware and established against CPU float32.

### 10.2 Model support

Version 1 supports a deliberately narrow Llama-like capability profile: causal decoder, RMSNorm, rotary embeddings, SwiGLU-style MLP, grouped-query attention, tied embeddings where configured, and text-only generation.

The adapter rejects unsupported configuration features during startup. The project does not claim that a matching `model_type` guarantees support. Adding a second architecture is a future exercise used to test whether the adapter boundary is genuine.

Model files are loaded from safetensors when available. Remote model code is not executed. Model revisions are pinned in course metadata for reproducibility.

## 11. Error handling and capacity policy

Domain errors are defined in the engine layer and mapped at the API boundary:

- Validation and unsupported-parameter errors.
- Model-not-ready or unsupported-model errors.
- Queue-full and KV-capacity errors.
- Deadline and cancellation events.
- Request-scoped execution failures.
- Fatal engine failures.

The engine uses bounded ingress and per-request output queues. Admission checks reserve enough capacity to make safe progress rather than accepting unlimited requests. If capacity becomes unavailable after admission, Version 1 may preempt a request by freeing its KV blocks and later recomputing; swapping is out of scope.

Cleanup is centralized and idempotent. Tests inspect allocator state after success, EOS, length limit, cancellation, timeout, sampling error, model error, API disconnect, and graceful shutdown.

## 12. Testing strategy

### 12.1 Test layers

- **Unit:** Tensor functions, model layers, sampling, state transitions, scheduler policies, block tables, and allocators.
- **Differential:** Optimized versus reference logits/tokens; custom model versus an optional Transformers oracle.
- **Invariant/property:** No double-free, no shared writable block without reference ownership, no scheduled request missing capacity, and conservation of total blocks.
- **Integration:** Native and compatible API, streaming, cancellation, timeout, metrics, and model-loading boundaries.
- **Load/resilience:** Mixed lengths, burst arrival, fixed concurrency, overload, fault injection, and shutdown.
- **Checkpoint:** Every lesson starter and solution remains in the expected state.

### 12.2 CI matrix

The required CI suite runs on Linux CPU with the tiny local model. A small macOS job validates installation and device selection; MPS acceleration tests are optional because runner availability and performance variance should not block contributions. Network/model-download and long benchmark jobs run separately.

Exact performance thresholds are not used across heterogeneous hardware. CI checks benchmark schema and correctness, while regression automation compares results only on controlled hardware.

## 13. Benchmark methodology

Every report records:

- Git revision, lesson/engine version, Python and PyTorch versions.
- OS, CPU, memory, device, dtype, and MPS/CUDA availability.
- Model revision, input/output token distributions, seed, cache size, token budget, concurrency, and request arrival pattern.
- Warmup policy and whether device synchronization was used.
- Success/error counts and finish reasons.

Core metrics are reported as distributions where meaningful, including median and p95/p99:

- TTFT.
- TPOT.
- ITL.
- End-to-end latency.
- Queue delay.
- Request throughput.
- Input, output, and total token throughput.
- KV-block utilization and preemption count.

The course distinguishes closed-loop concurrency from open-loop arrival-rate testing. Results compare two implementations on the same machine and workload. Reports explain trade-offs; they do not reduce engineering quality to one tokens-per-second number.

## 14. Version 2 curriculum

Version 2 begins only after the Version 1 engine and graduation suite are stable.

1. **Automatic prefix caching:** Hash full KV blocks, track references, reuse shared prefixes, and evict unreferenced blocks. Compare a block-hash approach with SGLang’s radix-tree mental model.
2. **Chunked prefill:** Allocate a token budget across long prefills and active decodes, measure TTFT/ITL trade-offs, and prevent decode starvation.
3. **Structured output:** Compile a small JSON-schema subset into token-level constraints and mask invalid logits. Separate grammar compilation from per-step mask application.
4. **Speculative decoding:** Add a small draft path, verification, acceptance accounting, and benchmarks that expose when speculation loses.
5. **Quantization:** Teach weight and KV-cache precision, calibration versus weight-only methods, memory accounting, accuracy checks, and backend-specific kernel limits. The portable path demonstrates semantics; accelerated kernels remain optional.

## 15. Open-source project experience

The public repository includes:

- Apache-2.0 license.
- A concise README with audience, course map, hardware expectations, and a five-minute tiny-model quickstart.
- `CONTRIBUTING.md`, code of conduct, security policy, issue templates, and pull-request checklist.
- Architecture decision records for major trade-offs.
- Documentation explaining that educational simplifications are intentional.
- A versioned curriculum and migration notes when lesson interfaces change.
- CI that verifies every reference solution and final engine.
- Good-first issues limited to documentation, tests, labs, or isolated adapters until the core boundaries stabilize.

The README leads with the journey and learner outcome, not benchmark claims. Public benchmark results always identify hardware and workload.

## 16. Three-perspective review and revisions

### Inference-systems expert review

**Findings:** The initial design had the correct major components but underemphasized prefill/decode semantics, incremental detokenization, device synchronization, token-budget scheduling, allocator invariants, and workload-sensitive serving metrics. It risked implying that block-based storage alone equals an optimized PagedAttention kernel.

**Revisions:** Prefill/decode now appears before optimization; streaming includes a real incremental decoder; the scheduler uses an explicit token budget; block tables and kernel optimization are distinguished; TTFT/TPOT/ITL and open/closed-loop workloads are required; resource ownership and cleanup are first-class tests.

### Lecture-teacher review

**Findings:** Building a Transformer in one challenge was too large for learners new to PyTorch. Optimizations needed a consistent scientific narrative, and reference solutions needed to explain reasoning rather than only reveal code.

**Revisions:** Lesson 2 contains focused labs; every lesson follows the same scenario-to-evidence contract; hints are tiered; each checkpoint is runnable; benchmarks are introduced formally before optimization; industrial source-reading is deferred until learners have the relevant mental model.

### Beginner-student review

**Findings:** Repository snapshots can be confusing to edit, model downloads can make early tests slow or fragile, MPS fallback can hide what actually ran, and a visible solution directory can tempt learners before they have a working loop.

**Revisions:** A course CLI creates a clean work directory and runs the right checks; early tests use tiny local fixtures; real-model tests are separately marked and cached; startup reports device/dtype; solutions are never imported by learner commands; every lesson supplies expected baseline output and three levels of hints.

## 17. Delivery phases

### Milestone 0 — Project skeleton

Repository metadata, package layout, course CLI, tiny fixtures, CI, documentation skeleton, and checkpoint verification.

### Milestone 1 — Foundations

Lessons 1–3 and a tested tiny-model generation path.

### Milestone 2 — First real server

Lessons 4–7, real-model adapter, naive/async server, streaming, and benchmark harness.

### Milestone 3 — Inference engine

Lessons 8–11, KV cache, batching, scheduler, and block allocator.

### Milestone 4 — Production shape

Lessons 12–14, compatibility API, observability, resilience, final engine, and graduation report.

### Milestone 5 — Advanced track

Version 2 lessons, added one at a time with their own design and benchmark evidence.

## 18. Acceptance criteria for Version 1

Version 1 is complete when:

- A new learner can finish Lessons 1–3 on CPU without downloading a model.
- The documented real model produces text on CPU and on a supported Apple Silicon/MPS environment.
- The final engine serves concurrent streaming requests through the documented OpenAI-compatible subset.
- Cached and batched paths match the reference path within documented tolerances.
- Continuous batching admits and retires requests between iterations.
- KV blocks are fully reclaimed after every tested terminal path.
- Overload is bounded and returns a stable error instead of exhausting memory.
- Benchmark reports contain the required environment, workload, latency, throughput, and cache fields.
- Every lesson starter and solution passes its expected checkpoint verification in CI.
- The final documentation clearly identifies simplifications and maps them to industrial equivalents.

## 19. Primary references informing the design

- vLLM architecture overview: <https://docs.vllm.ai/en/latest/design/arch_overview/>
- vLLM V1 guide and unified token-budget scheduler: <https://docs.vllm.ai/en/stable/usage/v1_guide/>
- vLLM automatic prefix caching: <https://docs.vllm.ai/en/latest/design/prefix_caching/>
- vLLM serving benchmark metrics: <https://docs.vllm.ai/en/latest/benchmarking/dashboard/>
- SGLang project and scheduler architecture: <https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/01-architecture-overview.md>
- SGLang KV-cache design: <https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/07-kv-cache.md>
- PyTorch MPS backend: <https://docs.pytorch.org/docs/stable/notes/mps.html>
- Hugging Face model loading and safetensors: <https://huggingface.co/docs/transformers/models>
- Hugging Face incremental token decoding: <https://huggingface.co/docs/tokenizers/api/decoders>
- SmolLM2-135M-Instruct model configuration and license: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/83212e1e2b3cfd6958f3707877bb878945dea8ee/config.json>
