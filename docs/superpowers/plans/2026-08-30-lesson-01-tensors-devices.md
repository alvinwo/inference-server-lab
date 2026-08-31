# Lesson 1: Tensors, Devices, and Trustworthy Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the first complete learner challenge, teaching portable PyTorch tensor work and honest device timing on CPU-first laptops with optional Apple Silicon/MPS execution.

**Architecture:** The lesson is a self-contained snapshot under `lessons/01-tensors-devices`. Its starter and solution expose identical `inference_lab` interfaces, shared offline tests define the checkpoint, and a device-aware benchmark imports only the active learner workspace. The course catalog changes Lesson 1 from `planned` to `published` only after starter failures and the solution pass are verified exactly.

**Tech Stack:** Python 3.12, PyTorch 2.13, pytest 9.1, standard-library dataclasses, statistics, time, json, platform, and pathlib.

**Spec:** `docs/superpowers/specs/2026-08-29-inference-server-learning-project-design.md`

## Global Constraints

- CPU correctness must run offline on every supported laptop.
- MPS is optional, visibly selected, and never silently falls back during an explicit MPS request.
- Starter and solution expose the same public API and never import one another.
- The challenge uses only `torch`, the Python standard library, and existing course tooling.
- The benchmark reports metadata and samples but enforces no performance threshold.
- Timing performs warmup and synchronizes the selected device before and after measured work.
- Correctness uses CPU float32 literal expectations; no model download is required.

---

### Task 1: Define the published lesson checkpoint

**Files:**
- Create: `tests/course/test_lesson_01.py`
- Create: `lessons/01-tensors-devices/tests/test_devices.py`
- Create: `lessons/01-tensors-devices/tests/test_tensor_ops.py`
- Create: `lessons/01-tensors-devices/tests/test_timing.py`
- Create: `lessons/01-tensors-devices/checkpoint.toml`
- Modify: `course/lessons.toml`

**Interfaces:**
- Consumes: `verify_checkpoint(project_root, lesson)` and `find_lesson(project_root, "01")`.
- Produces: a published `Lesson` whose starter failure set is exact and whose solution must pass.

- [ ] Write repository integration tests asserting Lesson 1 is published, verifies through `verify_checkpoint`, and its benchmark emits the required metadata.
- [ ] Write shared behavior tests for `DevicePolicy.resolve`, `seed_everything`, `affine`, `synchronize`, and `benchmark_operation` using literal numeric expectations and real CPU tensors.
- [ ] Add one timing orchestration test that replaces only the hardware synchronization boundary and clock, then asserts warmup and synchronization order plus literal millisecond samples.
- [ ] Run `uv run pytest tests/course/test_lesson_01.py -v` and confirm failure because the lesson snapshot is missing.
- [ ] Add `checkpoint.toml` with the exact starter node IDs and mark only Lesson 1 `published` in `course/lessons.toml`.

### Task 2: Implement identical starter and solution interfaces

**Files:**
- Create: `lessons/01-tensors-devices/starter/inference_lab/__init__.py`
- Create: `lessons/01-tensors-devices/starter/inference_lab/devices.py`
- Create: `lessons/01-tensors-devices/starter/inference_lab/tensor_ops.py`
- Create: `lessons/01-tensors-devices/starter/inference_lab/timing.py`
- Create: `lessons/01-tensors-devices/solution/inference_lab/__init__.py`
- Create: `lessons/01-tensors-devices/solution/inference_lab/devices.py`
- Create: `lessons/01-tensors-devices/solution/inference_lab/tensor_ops.py`
- Create: `lessons/01-tensors-devices/solution/inference_lab/timing.py`

**Interfaces:**
- `DevicePolicy.resolve(requested: str = "auto", dtype: torch.dtype = torch.float32) -> DevicePolicy`
- `DevicePolicy.report() -> dict[str, str]`
- `seed_everything(seed: int) -> None`
- `affine(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor`
- `synchronize(device: torch.device) -> None`
- `benchmark_operation(operation: Callable[[], object], *, device: torch.device, warmup: int = 3, iterations: int = 10) -> TimingStats`

- [ ] Implement shared dataclasses, docstrings, validation, and `NotImplementedError` challenge bodies in the starter without solution hints.
- [ ] Run the lesson tests against the starter and confirm only the node IDs declared in `checkpoint.toml` fail.
- [ ] Implement the solution using visible auto selection, explicit MPS errors, Python and PyTorch seeding, batched matrix multiplication plus bias broadcasting, CPU/MPS synchronization, warmup, and per-iteration synchronization.
- [ ] Run the lesson tests against the solution and confirm every test passes.

### Task 3: Add the concept lab and evidence-producing benchmark

**Files:**
- Create: `lessons/01-tensors-devices/starter/concept_lab.py`
- Create: `lessons/01-tensors-devices/solution/concept_lab.py`
- Create: `lessons/01-tensors-devices/benchmark.py`

**Interfaces:**
- Consumes: the public learner package from the active implementation directory.
- Produces: one JSON object containing `lesson_id`, requested and selected device, dtype, tensor shape, warmup count, iteration count, latency samples and summary, Python version, PyTorch version, and platform.

- [ ] Write the concept lab as an executable 20–40 minute tour of shape, broadcasting, matrix multiplication, dtype, device transfer, seeding, `torch.inference_mode`, warmup, and synchronization.
- [ ] Write the benchmark to allocate inputs before measurement, execute `affine` inside inference mode, and print one JSON object without thresholds.
- [ ] Run the benchmark against the solution and validate its JSON through the repository integration test.

### Task 4: Write the learner journey and engineering explanation

**Files:**
- Create: `lessons/01-tensors-devices/README.md`
- Create: `lessons/01-tensors-devices/starter/README.md`
- Create: `lessons/01-tensors-devices/solution/NOTES.md`
- Modify: `README.md`
- Modify: `docs/roadmap.md`

**Interfaces:**
- Consumes: the tested commands and APIs from Tasks 1–3.
- Produces: a beginner-ready 1–3 hour challenge after a 20–40 minute concept lab.

- [ ] Fill every repeating lesson-contract section: scenario, goals, concept lab, baseline, challenge, tests, measurement, three hint levels, reference solution, journal prompts, industrial connections, and stretch work.
- [ ] Explain tensor shapes and device timing with concrete examples, define all new terms before using them, and distinguish reproducibility within one environment from cross-device equality.
- [ ] Link primary PyTorch documentation for MPS, synchronization, matmul, and reproducibility; give source-reading pointers for vLLM and SGLang without claiming API compatibility.
- [ ] Add a short starter README containing the next three commands and update repository onboarding and Milestone 1 progress without claiming the milestone complete.

### Task 5: Verify, publish, and prepare the learner workspace

**Files:**
- Modify: `course/lessons.toml`
- Create locally, gitignored: `work/01-tensors-devices/`

**Interfaces:**
- Consumes: all Lesson 1 artifacts and course commands.
- Produces: a GitHub-published lesson plus a clean local starter workspace for the learner.

- [ ] Run `uv run python -m course verify 01` and confirm the starter has the exact intended failures while the solution passes.
- [ ] Run `uv run python scripts/verify_checkpoints.py` and confirm both `_template` and Lesson 1 verify.
- [ ] Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `uv run pytest`.
- [ ] Run the concept lab and the benchmark against the solution on the current laptop; confirm the reported device and valid JSON without judging speed.
- [ ] Commit, fast-forward to `main`, rerun the complete gate, and push `main`.
- [ ] Run `uv run python -m course start 01` from the main checkout so the learner receives only the starter under `work/01-tensors-devices/`.
