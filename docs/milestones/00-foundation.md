# Milestone 0 acceptance record

Milestone 0 establishes executable course infrastructure. It does not claim that an inference
engine or curriculum lesson is already implemented.

## Reproducible environment

- Tested implementation commit: `3ca476a3151a31560c76c3a8f7633e4553481e6d`
- Date: 2026-08-30
- Operating system: macOS 26.2 (25C56), arm64
- Python: CPython 3.12.11
- uv: 0.8.0
- Required correctness device: CPU

## Acceptance results

| Gate | Command or evidence | Result |
| --- | --- | --- |
| Frozen environment | `uv sync --frozen --python 3.12` | exit 0 |
| Lint | `uv run ruff check .` | exit 0 |
| Formatting | `uv run ruff format --check .` | exit 0; 38 files formatted |
| Static types | `uv run mypy` | exit 0; no issues in 10 source files |
| Tests | `uv run pytest` | exit 0; 37 passed |
| Checkpoints | `uv run python scripts/verify_checkpoints.py` | exit 0; template verified |
| Course catalog | `uv run python -m course list` | exit 0; 14 ordered planned lessons |
| Learner flow | `tests/course/test_template.py` | start, intended failure, solution pass, JSON benchmark, archived note |

The automated learner-flow test uses a temporary published copy of the author template. It proves
that the starter fails only its named checkpoint, the solution passes, the benchmark emits valid
JSON, and reset preserves a learner-created file under the archive path.

## Three-perspective review

### Inference-systems expert

The foundation keeps course tooling separate from the future serving runtime, executes checkpoints
in isolated processes, puts the learner implementation first on `PYTHONPATH`, and verifies exact
starter failures rather than accepting any failure. Pytest roots are pinned per lesson so node IDs
remain stable in local checkouts and CI. The milestone intentionally contains no model download,
network-dependent correctness test, serving-performance claim, or false production guarantee.

### Lecture teacher

The fourteen-lesson catalog follows the dependency order from tensors through production-shaped
serving. The executable template requires the same scenario-to-evidence sections for every lesson,
three hint levels, measurement, engineering reasoning, an industrial connection, and an optional
stretch challenge. Starter and solution states are continuously checked, making each published
lesson a runnable checkpoint rather than a prose chapter.

### Beginner student

Onboarding states the assumed Python background and explains CPU, Apple Silicon/MPS, planned lesson
status, and the exact command loop. Learner work is copied to a gitignored location, a second start
cannot overwrite it, and reset requires `--yes` and archives files with collision-safe names. Errors
are short and actionable rather than tracebacks. The README explicitly says that this foundation is
educational and not a production replacement.

## Entry criteria for Milestone 1

Milestone 1 starts by publishing Lesson 1 with offline CPU tests, an optional MPS path that reports
the device actually used, an executable benchmark with correct synchronization, and a complete
reference solution. It must not alter workspace or checkpoint contracts without a reviewed
architecture decision record.
