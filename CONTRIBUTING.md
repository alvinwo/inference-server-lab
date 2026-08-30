# Contributing

Thank you for helping people learn inference systems. Small, reviewable changes with clear teaching
value are easiest to maintain.

## Development setup

Python 3.12 is recommended; Python 3.11–3.13 is supported. Install uv, fork the repository, and run:

```bash
uv sync --frozen --python 3.12
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/verify_checkpoints.py
```

Run this sequence before opening a pull request. Tests must not require a model download or network
access unless they are explicitly separated from the correctness gate.

## Lesson contributions

Copy `lessons/_template` and follow every section in its README. A lesson must have one primary
challenge, stable interfaces, tiered hints, an exact list of intended starter failures, a complete
solution with engineering notes, and a benchmark that reports its environment and workload. Never
import solution code from a starter or learner command.

Good first contributions are documentation improvements, tests, focused concept labs, and isolated
model adapters. Changes to course contracts or engine boundaries should begin with an architecture
decision record.

## Pull requests

Explain the learner problem, the design choice, verification performed, and any compatibility or
performance impact. Performance claims must include hardware, software versions, workload, warmup,
sample count, and synchronization method. By participating, you agree to follow the code of conduct
and license your contribution under Apache-2.0.
