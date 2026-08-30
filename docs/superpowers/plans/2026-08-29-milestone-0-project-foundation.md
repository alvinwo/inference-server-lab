# Milestone 0 Project Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a tested Python project, learner workspace CLI, lesson/checkpoint contract, documentation skeleton, and CI foundation on which the fourteen Version 1 lessons can be built safely.

**Architecture:** Keep the polished engine package (`src/inference_lab`) separate from the course tooling (`course`) and lesson snapshots (`lessons`). The course CLI reads one TOML catalog, copies published starters into a gitignored workspace, runs tests with the learner implementation first on `PYTHONPATH`, archives resets instead of deleting work, and verifies starter/solution checkpoints through structured pytest reports.

**Tech Stack:** Python 3.12, uv, PyTorch 2.13, pytest 9.1, Hypothesis 6.165, Ruff 0.16, mypy 2.3, standard-library `argparse`, `dataclasses`, `pathlib`, `shutil`, `subprocess`, and `tomllib`.

**Spec:** `docs/superpowers/specs/2026-08-29-inference-server-learning-project-design.md`

## Global Constraints

- CPU execution must work on all supported laptops; Apple Silicon/MPS is optional.
- Python developers new to PyTorch and GPU programming are the primary audience.
- Unit and integration correctness tests must not require network access or a model download.
- Lesson snapshots are self-contained; the starter and solution never import one another.
- Learner commands never import from `solution/`.
- Learner edits live under gitignored `work/<lesson-id>/` paths.
- Reset is recoverable: archive work under `work/.trash/`; never recursively delete an unresolved path.
- Python support is `>=3.11,<3.14`; `.python-version` recommends `3.12`.
- Runtime dependency for this milestone is `torch>=2.13,<2.14`.
- Development dependencies are `pytest>=9.1,<10`, `hypothesis>=6.165,<7`, `ruff>=0.16.5,<0.17`, and `mypy>=2.3,<3`.
- The project license is Apache-2.0.
- The initial real-model target remains `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `83212e1e2b3cfd6958f3707877bb878945dea8ee`; Milestone 0 does not download it.

---

## File map

```text
.python-version                         Recommended interpreter
pyproject.toml                          Package metadata, dependencies, tool configuration
uv.lock                                 Reproducible dependency resolution
README.md                               Minimal package readme, expanded in Task 8
src/inference_lab/__init__.py           Final-engine package identity
course/__init__.py                      Course-tooling package identity
course/__main__.py                      CLI parser and command dispatch
course/errors.py                        Stable user-facing course errors
course/models.py                        Lesson and command result dataclasses
course/catalog.py                       TOML catalog loading and lesson lookup
course/lessons.toml                     Fourteen-lesson publication catalog
course/workspace.py                     Starter copy and recoverable reset
course/runner.py                        Isolated test and benchmark subprocesses
course/pytest_reporter.py               Structured node-id/outcome report plugin
course/checkpoints.py                   Starter/solution contract verification
lessons/README.md                       Snapshot rules for lesson authors
lessons/_template/                      Executable lesson-author template
scripts/verify_checkpoints.py           Repository-wide checkpoint entry point
tests/test_package.py                   Installation/import smoke tests
tests/conftest.py                       Shared repository-root fixture
tests/course/test_catalog.py            Catalog tests
tests/course/test_workspace.py          Copy/archive safety tests
tests/course/test_cli.py                CLI behavior tests
tests/course/test_runner.py             PYTHONPATH and result tests
tests/course/test_checkpoints.py        Expected-failure verifier tests
README.md                               Learner-facing project entry point
CONTRIBUTING.md                         Contribution workflow
CODE_OF_CONDUCT.md                      Contributor Covenant
SECURITY.md                             Educational-project security policy
LICENSE                                 Apache License 2.0 text
docs/architecture.md                    Runtime and curriculum boundaries
docs/glossary.md                        Beginner terminology
docs/roadmap.md                         Version 1/2 milestones
.github/workflows/ci.yml                Linux quality gate and macOS install smoke test
.github/ISSUE_TEMPLATE/bug.yml          Structured bug report
.github/ISSUE_TEMPLATE/lesson.yml       Lesson feedback report
.github/pull_request_template.md         Test and teaching-quality checklist
```

### Task 1: Establish the Python package and quality gates

**Files:**
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `README.md`
- Create: `src/inference_lab/__init__.py`
- Create: `course/__init__.py`
- Create: `tests/test_package.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: No project code.
- Produces: Importable `inference_lab`, `inference_lab.__version__ == "0.1.0"`, and standard `uv run` quality commands.

- [ ] **Step 1: Write the failing package smoke test**

```python
# tests/test_package.py
import inference_lab


def test_package_has_semantic_version() -> None:
    assert inference_lab.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and confirm the project is not packaged**

Run: `uv run --python 3.12 pytest tests/test_package.py -v`

Expected: FAIL because `pyproject.toml` and `inference_lab` do not exist.

- [ ] **Step 3: Add package metadata and tool configuration**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[project]
name = "inference-server-lab"
version = "0.1.0"
description = "Build a production-shaped LLM inference server through progressive challenges."
readme = "README.md"
requires-python = ">=3.11,<3.14"
license = "Apache-2.0"
dependencies = [
  "torch>=2.13,<2.14",
]

[dependency-groups]
dev = [
  "hypothesis>=6.165,<7",
  "mypy>=2.3,<3",
  "pytest>=9.1,<10",
  "ruff>=0.16.5,<0.17",
]

[tool.hatch.build.targets.wheel]
packages = ["src/inference_lab", "course"]

[tool.pytest.ini_options]
addopts = "-ra --strict-config --strict-markers"
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["inference_lab", "course"]
```

```text
# .python-version
3.12
```

```python
# src/inference_lab/__init__.py
"""A learning-focused LLM inference server."""

__version__ = "0.1.0"
```

```python
# course/__init__.py
"""Tools for navigating and verifying the challenge curriculum."""
```

Create `README.md` with the heading `# Inference Server Lab` and the sentence `An educational project for building a production-shaped LLM inference server through progressive challenges.` Task 8 expands this file after the CLI exists.

Append `.mypy_cache/` and `.python-version` must remain tracked; do not add `.python-version` to `.gitignore`.

- [ ] **Step 4: Resolve and lock dependencies**

Run: `uv lock --python 3.12`

Expected: `uv.lock` is created and resolves Python 3.12 with the declared dependency ranges.

- [ ] **Step 5: Run the package and quality checks**

Run: `uv run --python 3.12 pytest tests/test_package.py -v`

Expected: PASS.

Run: `uv run ruff check src tests`

Expected: PASS.

Run: `uv run mypy`

Expected: PASS.

- [ ] **Step 6: Commit the package foundation**

```bash
git add .python-version pyproject.toml uv.lock README.md src/inference_lab course/__init__.py tests/test_package.py .gitignore
git commit -m "build: establish Python project foundation"
```

### Task 2: Define and validate the lesson catalog

**Files:**
- Create: `course/errors.py`
- Create: `course/models.py`
- Create: `course/catalog.py`
- Create: `course/lessons.toml`
- Create: `tests/conftest.py`
- Create: `tests/course/test_catalog.py`

**Interfaces:**
- Consumes: Python 3.11 `tomllib` and repository root `Path`.
- Produces: `Lesson`, `LessonStatus`, `load_catalog(project_root)`, and `find_lesson(project_root, lesson_id)`.

- [ ] **Step 1: Write failing catalog tests**

```python
# tests/course/test_catalog.py
from pathlib import Path

import pytest

from course.catalog import find_lesson, load_catalog
from course.errors import CourseError
from course.models import LessonStatus


def test_catalog_contains_all_version_one_lessons(project_root: Path) -> None:
    lessons = load_catalog(project_root)
    assert len(lessons) == 14
    assert lessons[0].lesson_id == "01-tensors-devices"
    assert lessons[-1].lesson_id == "14-load-resilience"
    assert all(lesson.status is LessonStatus.PLANNED for lesson in lessons)


def test_find_lesson_accepts_number_or_full_id(project_root: Path) -> None:
    by_number = find_lesson(project_root, "08")
    by_id = find_lesson(project_root, "08-kv-cache")
    assert by_number == by_id


def test_find_lesson_rejects_unknown_id(project_root: Path) -> None:
    with pytest.raises(CourseError, match="Unknown lesson '99'"):
        find_lesson(project_root, "99")
```

Add a session fixture in `tests/conftest.py` returning `Path(__file__).parents[1]` as `project_root`.

- [ ] **Step 2: Run the catalog tests and verify the missing-module failure**

Run: `uv run pytest tests/course/test_catalog.py -v`

Expected: FAIL because the `course` package is not implemented.

- [ ] **Step 3: Implement the catalog types and validation**

```python
# course/models.py
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LessonStatus(StrEnum):
    PLANNED = "planned"
    PUBLISHED = "published"


@dataclass(frozen=True, slots=True)
class Lesson:
    number: str
    slug: str
    title: str
    status: LessonStatus
    path: Path

    @property
    def lesson_id(self) -> str:
        return f"{self.number}-{self.slug}"
```

```python
# course/errors.py
class CourseError(Exception):
    """A user-actionable course tooling error."""
```

`load_catalog()` must set each lesson path to `<project-root>/lessons/<lesson-id>` and reject duplicate numbers, duplicate full IDs, non-two-digit numbers, unknown status strings, and an empty catalog with `CourseError`. `find_lesson()` must match either the two-digit number or full ID and include valid IDs in its error message.

- [ ] **Step 4: Add the complete Version 1 catalog**

```toml
# course/lessons.toml
[[lessons]]
number = "01"
slug = "tensors-devices"
title = "Tensors, devices, and trustworthy timing"
status = "planned"

[[lessons]]
number = "02"
slug = "tiny-transformer"
title = "A tiny decoder-only Transformer"
status = "planned"

[[lessons]]
number = "03"
slug = "generation-sampling"
title = "Autoregressive generation and sampling"
status = "planned"

[[lessons]]
number = "04"
slug = "model-adapters"
title = "Model adapters and real weights"
status = "planned"

[[lessons]]
number = "05"
slug = "naive-server"
title = "Prefill, decode, and the naive HTTP server"
status = "planned"

[[lessons]]
number = "06"
slug = "streaming-lifecycle"
title = "Request lifecycle, streaming, and cancellation"
status = "planned"

[[lessons]]
number = "07"
slug = "serving-benchmarks"
title = "Serving benchmarks and profiler literacy"
status = "planned"

[[lessons]]
number = "08"
slug = "kv-cache"
title = "Per-request KV cache"
status = "planned"

[[lessons]]
number = "09"
slug = "padded-batching"
title = "Padded batching"
status = "planned"

[[lessons]]
number = "10"
slug = "continuous-batching"
title = "Continuous batching and token-budget scheduling"
status = "planned"

[[lessons]]
number = "11"
slug = "block-kv-memory"
title = "Block-based KV memory management"
status = "planned"

[[lessons]]
number = "12"
slug = "openai-api"
title = "OpenAI-compatible API subset"
status = "planned"

[[lessons]]
number = "13"
slug = "observability"
title = "Observability and operational safety"
status = "planned"

[[lessons]]
number = "14"
slug = "load-resilience"
title = "Load, resilience, and graduation benchmark"
status = "planned"
```

- [ ] **Step 5: Run the catalog tests**

Run: `uv run pytest tests/course/test_catalog.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the lesson catalog**

```bash
git add course tests/conftest.py tests/course/test_catalog.py
git commit -m "feat: define the version one lesson catalog"
```

### Task 3: Implement `list` and safe starter workspaces

**Files:**
- Create: `course/workspace.py`
- Create: `course/__main__.py`
- Create: `tests/course/test_workspace.py`
- Create: `tests/course/test_cli.py`

**Interfaces:**
- Consumes: `Lesson`, `LessonStatus`, `find_lesson()`.
- Produces: `WorkspaceManager.start(lesson) -> Path`, `course.main(argv) -> int`, `python -m course list`, and `python -m course start LESSON`.

- [ ] **Step 1: Write failing workspace-copy tests**

```python
# tests/course/test_workspace.py
from pathlib import Path

import pytest

from course.errors import CourseError
from course.models import Lesson, LessonStatus
from course.workspace import WorkspaceManager


def published_lesson(root: Path) -> Lesson:
    lesson_path = root / "lessons" / "01-example"
    starter = lesson_path / "starter"
    starter.mkdir(parents=True)
    (starter / "answer.py").write_text("VALUE = 1\n")
    return Lesson("01", "example", "Example", LessonStatus.PUBLISHED, lesson_path)


def test_start_copies_only_the_starter(tmp_path: Path) -> None:
    lesson = published_lesson(tmp_path)
    destination = WorkspaceManager(tmp_path).start(lesson)
    assert (destination / "answer.py").read_text() == "VALUE = 1\n"
    assert not (destination / "solution").exists()


def test_start_refuses_to_overwrite_work(tmp_path: Path) -> None:
    lesson = published_lesson(tmp_path)
    manager = WorkspaceManager(tmp_path)
    manager.start(lesson)
    with pytest.raises(CourseError, match="already exists"):
        manager.start(lesson)


def test_start_rejects_planned_lesson(tmp_path: Path) -> None:
    lesson = Lesson("01", "example", "Example", LessonStatus.PLANNED, tmp_path / "missing")
    with pytest.raises(CourseError, match="not published"):
        WorkspaceManager(tmp_path).start(lesson)
```

- [ ] **Step 2: Run the tests and confirm the missing-workspace failure**

Run: `uv run pytest tests/course/test_workspace.py -v`

Expected: FAIL because `WorkspaceManager` does not exist.

- [ ] **Step 3: Implement starter copying with resolved-path guards**

Implement the guarded copy directly:

```python
from pathlib import Path
import shutil

from course.errors import CourseError
from course.models import Lesson, LessonStatus


class WorkspaceManager:
    def __init__(self, project_root: Path, work_root: Path | None = None) -> None:
        self.project_root = project_root.resolve()
        selected_work_root = work_root if work_root is not None else self.project_root / "work"
        self.work_root = selected_work_root.resolve()

    def _destination(self, lesson: Lesson) -> Path:
        destination = (self.work_root / lesson.lesson_id).resolve()
        if destination.parent != self.work_root:
            raise CourseError(f"Unsafe workspace path for lesson '{lesson.lesson_id}'")
        return destination

    def start(self, lesson: Lesson) -> Path:
        if lesson.status is not LessonStatus.PUBLISHED:
            raise CourseError(f"Lesson '{lesson.lesson_id}' is not published")
        starter = lesson.path.resolve() / "starter"
        if not starter.is_dir():
            raise CourseError(f"Lesson '{lesson.lesson_id}' has no starter")
        destination = self._destination(lesson)
        if destination.exists():
            raise CourseError(f"Workspace '{destination}' already exists")
        self.work_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(starter, destination)
        return destination
```

- [ ] **Step 4: Write failing CLI tests**

```python
# tests/course/test_cli.py
from course.__main__ import main


def test_list_prints_status_and_title(capsys) -> None:
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "01  planned    Tensors, devices, and trustworthy timing" in output
    assert "14  planned    Load, resilience, and graduation benchmark" in output


def test_unknown_lesson_is_a_clean_cli_error(capsys) -> None:
    assert main(["start", "99"]) == 2
    assert "Unknown lesson '99'" in capsys.readouterr().err
```

- [ ] **Step 5: Implement CLI parsing and error mapping**

Use standard-library `argparse`. `main(argv: Sequence[str] | None = None) -> int` returns `0` for success and `2` for `CourseError`, prints errors to stderr without a traceback, and derives the project root from `Path(__file__).resolve().parents[1]`.

- [ ] **Step 6: Run workspace and CLI tests**

Run: `uv run pytest tests/course/test_workspace.py tests/course/test_cli.py -v`

Expected: PASS.

- [ ] **Step 7: Commit starter workspace support**

```bash
git add course/workspace.py course/__main__.py tests/course/test_workspace.py tests/course/test_cli.py
git commit -m "feat: add lesson listing and starter workspaces"
```

### Task 4: Make lesson reset recoverable

**Files:**
- Modify: `course/workspace.py`
- Modify: `course/__main__.py`
- Modify: `tests/course/test_workspace.py`
- Modify: `tests/course/test_cli.py`

**Interfaces:**
- Consumes: Existing `WorkspaceManager` path guards.
- Produces: `WorkspaceManager.archive(lesson) -> Path` and `python -m course reset LESSON --yes`.

- [ ] **Step 1: Write failing archive tests**

```python
def test_archive_moves_work_to_recoverable_trash(tmp_path: Path) -> None:
    lesson = published_lesson(tmp_path)
    manager = WorkspaceManager(tmp_path)
    active = manager.start(lesson)
    (active / "notes.txt").write_text("keep me")

    archived = manager.archive(lesson, timestamp="20260829T120000Z")

    assert not active.exists()
    assert archived == tmp_path / "work" / ".trash" / "01-example-20260829T120000Z"
    assert (archived / "notes.txt").read_text() == "keep me"


def test_archive_refuses_paths_outside_work_root(tmp_path: Path) -> None:
    lesson = published_lesson(tmp_path)
    manager = WorkspaceManager(tmp_path, work_root=tmp_path / "work")
    with pytest.raises(CourseError, match="No active workspace"):
        manager.archive(lesson, timestamp="20260829T120000Z")
```

- [ ] **Step 2: Run the archive tests and verify failure**

Run: `uv run pytest tests/course/test_workspace.py -k archive -v`

Expected: FAIL because `archive()` does not exist.

- [ ] **Step 3: Implement archive instead of delete**

`archive()` must resolve and validate the exact active directory, create `work/.trash`, and use `shutil.move`. The default timestamp is UTC formatted as `%Y%m%dT%H%M%SZ`. If the generated archive already exists, append `-2`, `-3`, and so on. It must never call `rmtree`.

- [ ] **Step 4: Add reset confirmation behavior**

`python -m course reset 01` must return `2` and print `Refusing reset without --yes; your workspace was not changed.` `python -m course reset 01 --yes` archives the workspace and prints the recoverable path.

- [ ] **Step 5: Run the reset test suite**

Run: `uv run pytest tests/course/test_workspace.py tests/course/test_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Commit recoverable reset**

```bash
git add course/workspace.py course/__main__.py tests/course/test_workspace.py tests/course/test_cli.py
git commit -m "feat: archive learner work during reset"
```

### Task 5: Run lesson tests and benchmarks in isolation

**Files:**
- Create: `course/pytest_reporter.py`
- Create: `course/runner.py`
- Create: `tests/course/test_runner.py`
- Modify: `course/__main__.py`

**Interfaces:**
- Consumes: Published `Lesson` and an implementation directory.
- Produces: `TestRun`, `run_tests()`, `run_benchmark()`, `python -m course test LESSON`, and `python -m course benchmark LESSON`.

- [ ] **Step 1: Write a failing isolation test**

```python
# tests/course/test_runner.py
from pathlib import Path

from course.models import Lesson, LessonStatus
from course.runner import run_tests


def test_run_tests_imports_workspace_before_installed_package(tmp_path: Path) -> None:
    lesson_path = tmp_path / "lessons" / "01-example"
    tests = lesson_path / "tests"
    implementation = tmp_path / "work" / "01-example"
    package = implementation / "inference_lab"
    tests.mkdir(parents=True)
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('ORIGIN = "workspace"\n')
    (tests / "test_origin.py").write_text(
        "import inference_lab\n\n"
        "def test_origin():\n"
        '    assert inference_lab.ORIGIN == "workspace"\n'
    )
    lesson = Lesson("01", "example", "Example", LessonStatus.PUBLISHED, lesson_path)

    result = run_tests(tmp_path, lesson, implementation)

    assert result.returncode == 0
    assert result.failed_node_ids == ()
    assert "1 passed" in result.stdout
```

- [ ] **Step 2: Run the runner test and verify failure**

Run: `uv run pytest tests/course/test_runner.py -v`

Expected: FAIL because `course.runner` does not exist.

- [ ] **Step 3: Define structured subprocess results**

Add to `course/models.py`:

```python
@dataclass(frozen=True, slots=True)
class TestRun:
    returncode: int
    stdout: str
    stderr: str
    failed_node_ids: tuple[str, ...]
```

`course.pytest_reporter` must record one `{nodeid, outcome}` object for each call-stage report and write JSON to the path in `INFERENCE_LAB_PYTEST_REPORT` during `pytest_sessionfinish`.

- [ ] **Step 4: Implement isolated pytest execution**

`run_tests(project_root, lesson, implementation_root) -> TestRun` must invoke:

```text
<python> -m pytest <lesson-tests> -q -p course.pytest_reporter
```

Set subprocess `cwd` to `lesson.path`, pass `tests` as the pytest target so node IDs are stable, set `PYTHONPATH` to `<implementation-root><pathsep><project-root>` ahead of any inherited value, set the structured-report environment variable to a temporary file, capture text output, and return failed node IDs from the JSON report. Reject missing test or implementation directories with `CourseError` before spawning.

- [ ] **Step 5: Add benchmark runner tests and implementation**

`run_benchmark(project_root, lesson, implementation_root) -> CompletedProcess[str]` runs `<python> <lesson>/benchmark.py` with the same `PYTHONPATH` ordering. A missing benchmark raises `CourseError("Lesson '<id>' has no benchmark yet")`; a nonzero exit is returned unchanged so the CLI can show stdout/stderr.

- [ ] **Step 6: Wire `test` and `benchmark` into the CLI**

Both commands require a published lesson and an active `work/<lesson-id>` workspace. They stream captured output to the appropriate terminal stream and return the child exit code.

- [ ] **Step 7: Run runner and CLI tests**

Run: `uv run pytest tests/course/test_runner.py tests/course/test_cli.py -v`

Expected: PASS.

- [ ] **Step 8: Commit isolated lesson execution**

```bash
git add course/models.py course/pytest_reporter.py course/runner.py course/__main__.py tests/course/test_runner.py tests/course/test_cli.py
git commit -m "feat: run lesson checks in isolated workspaces"
```

### Task 6: Verify starter and solution checkpoint contracts

**Files:**
- Create: `course/checkpoints.py`
- Create: `tests/course/test_checkpoints.py`
- Modify: `course/__main__.py`

**Interfaces:**
- Consumes: `run_tests()` structured outcomes and `checkpoint.toml`.
- Produces: `CheckpointReport`, `verify_checkpoint(project_root, lesson)`, and `python -m course verify LESSON`.

- [ ] **Step 1: Write failing checkpoint tests**

```python
# tests/course/test_checkpoints.py
from pathlib import Path

import pytest

from course.checkpoints import verify_checkpoint
from course.errors import CourseError
from course.models import Lesson, LessonStatus


def make_checkpoint(root: Path, expected_failure: str) -> Lesson:
    lesson_path = root / "lessons" / "01-example"
    for implementation, value in (("starter", "not-ready"), ("solution", "ready")):
        package = lesson_path / implementation / "inference_lab"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f'VALUE = "{value}"\n')
    tests = lesson_path / "tests"
    tests.mkdir()
    (tests / "test_challenge.py").write_text(
        "from inference_lab import VALUE\n\n"
        "def test_value_is_ready():\n"
        '    assert VALUE == "ready"\n'
    )
    (lesson_path / "README.md").write_text("# Example\n")
    (lesson_path / "solution" / "NOTES.md").write_text("# Notes\n")
    (lesson_path / "checkpoint.toml").write_text(
        'expected_starter_failures = ["' + expected_failure + '"]\n'
    )
    return Lesson("01", "example", "Example", LessonStatus.PUBLISHED, lesson_path)


def test_verify_checkpoint_accepts_exact_expected_failure(tmp_path: Path) -> None:
    lesson = make_checkpoint(tmp_path, "tests/test_challenge.py::test_value_is_ready")
    report = verify_checkpoint(tmp_path, lesson)
    assert report.starter_failures == ("tests/test_challenge.py::test_value_is_ready",)
    assert report.solution_passed is True


def test_verify_checkpoint_rejects_unexpected_failure(tmp_path: Path) -> None:
    lesson = make_checkpoint(tmp_path, "tests/test_challenge.py::test_different_name")
    with pytest.raises(CourseError, match="Starter failure mismatch"):
        verify_checkpoint(tmp_path, lesson)
```

- [ ] **Step 2: Run checkpoint tests and verify failure**

Run: `uv run pytest tests/course/test_checkpoints.py -v`

Expected: FAIL because `verify_checkpoint` does not exist.

- [ ] **Step 3: Implement checkpoint parsing and required-file validation**

```python
@dataclass(frozen=True, slots=True)
class CheckpointReport:
    lesson_id: str
    starter_failures: tuple[str, ...]
    solution_passed: bool
```

`verify_checkpoint()` must require these paths before running tests:

```text
README.md
checkpoint.toml
starter/inference_lab/__init__.py
solution/inference_lab/__init__.py
solution/NOTES.md
tests/
```

Parse `expected_starter_failures` as a unique list of strings. Run the starter and require the exact failed-node-ID set to match. Run the solution and require exit code zero with no failed node IDs. Collection errors or missing reports are verifier errors, not accepted starter failures.

- [ ] **Step 4: Add the `verify` CLI command**

`python -m course verify 01` prints `01-example: starter state expected; solution passes` and returns zero. On mismatch it prints the expected and actual sorted node IDs and returns `2`.

- [ ] **Step 5: Run checkpoint and CLI tests**

Run: `uv run pytest tests/course/test_checkpoints.py tests/course/test_cli.py -v`

Expected: PASS.

- [ ] **Step 6: Commit checkpoint verification**

```bash
git add course/checkpoints.py course/__main__.py tests/course/test_checkpoints.py tests/course/test_cli.py
git commit -m "feat: verify lesson starter and solution states"
```

### Task 7: Add an executable lesson-author template and repository verifier

**Files:**
- Create: `lessons/README.md`
- Create: `lessons/_template/README.md`
- Create: `lessons/_template/checkpoint.toml`
- Create: `lessons/_template/starter/inference_lab/__init__.py`
- Create: `lessons/_template/solution/inference_lab/__init__.py`
- Create: `lessons/_template/solution/NOTES.md`
- Create: `lessons/_template/tests/test_challenge.py`
- Create: `lessons/_template/benchmark.py`
- Create: `scripts/verify_checkpoints.py`
- Create: `tests/course/test_template.py`

**Interfaces:**
- Consumes: `verify_checkpoint()` and `load_catalog()`.
- Produces: A copyable author template and repository-wide verification command.

- [ ] **Step 1: Write the failing template verification test**

```python
# tests/course/test_template.py
from pathlib import Path

from course.checkpoints import verify_checkpoint
from course.models import Lesson, LessonStatus


def test_author_template_has_expected_starter_and_passing_solution(project_root: Path) -> None:
    template = project_root / "lessons" / "_template"
    lesson = Lesson("00", "template", "Lesson template", LessonStatus.PUBLISHED, template)
    report = verify_checkpoint(project_root, lesson)
    assert report.starter_failures == (
        "tests/test_challenge.py::test_implementation_returns_ready",
    )
    assert report.solution_passed is True
```

Add a second test that copies `_template` into `tmp_path/lessons/00-template`, constructs a published `Lesson`, starts a workspace with `WorkspaceManager`, confirms `run_tests()` reports the one expected starter failure, replaces the workspace package with the template solution, confirms tests pass, confirms `run_benchmark()` returns JSON with `implementation_status == "ready"`, writes a learner note, archives the workspace, and confirms the note exists in the archive. This is the automated start → test → solve → benchmark → reset smoke flow used by final acceptance.

- [ ] **Step 2: Run the template test and verify missing files**

Run: `uv run pytest tests/course/test_template.py -v`

Expected: FAIL because `lessons/_template` does not exist.

- [ ] **Step 3: Create the executable template**

The starter exposes:

```python
def implementation_status() -> str:
    raise NotImplementedError("Implement this lesson challenge")
```

The solution returns `"ready"`. The challenge test asserts that result. `checkpoint.toml` contains exactly:

```toml
expected_starter_failures = [
  "tests/test_challenge.py::test_implementation_returns_ready",
]
```

The benchmark prints one JSON object with `lesson_id`, `implementation_status`, and `duration_seconds`. The template README contains every section in the repeating lesson contract; the fields contain author instructions in HTML comments so rendered learner pages do not display unfinished guidance.

- [ ] **Step 4: Implement repository-wide verification**

`scripts/verify_checkpoints.py` verifies the author template first, then every catalog lesson whose status is `published`. It exits nonzero on the first failure and prints a one-line success for each verified checkpoint. It exits successfully when no curriculum lessons are published because the executable template still proves the verifier path.

- [ ] **Step 5: Run template and repository verification**

Run: `uv run pytest tests/course/test_template.py -v`

Expected: PASS.

Run: `uv run python scripts/verify_checkpoints.py`

Expected: `_template: starter state expected; solution passes`.

- [ ] **Step 6: Commit the author template**

```bash
git add lessons scripts/verify_checkpoints.py tests/course/test_template.py
git commit -m "feat: add executable lesson author template"
```

### Task 8: Publish the open-source documentation foundation

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `docs/architecture.md`
- Create: `docs/glossary.md`
- Create: `docs/roadmap.md`
- Create: `tests/test_documentation.py`

**Interfaces:**
- Consumes: The approved spec and CLI commands.
- Produces: Accurate learner onboarding and contributor expectations.

- [ ] **Step 1: Write failing documentation contract tests**

```python
# tests/test_documentation.py
from pathlib import Path


def test_readme_contains_required_expectations(project_root: Path) -> None:
    readme = (project_root / "README.md").read_text()
    assert "educational" in readme.lower()
    assert "CPU" in readme
    assert "Apple Silicon" in readme
    assert "python -m course list" in readme
    assert "not a production replacement" in readme.lower()


def test_open_source_policy_files_exist(project_root: Path) -> None:
    required = {
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "docs/architecture.md",
        "docs/glossary.md",
        "docs/roadmap.md",
    }
    assert required <= {str(path.relative_to(project_root)) for path in project_root.rglob("*")}


def test_license_is_apache_2(project_root: Path) -> None:
    license_text = (project_root / "LICENSE").read_text()
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
```

- [ ] **Step 2: Run the documentation tests and verify failure**

Run: `uv run pytest tests/test_documentation.py -v`

Expected: FAIL because the public documentation is absent.

- [ ] **Step 3: Write the learner-facing README**

Use this exact top-level order:

```text
1. Build an Inference Server, One Challenge at a Time
2. Who this is for
3. What you will build
4. Hardware: CPU required, Apple Silicon/MPS optional, NVIDIA not required
5. Five-minute setup
6. Version 1 lesson map
7. How starter, tests, solution, and engineering notes work
8. Educational limitations
9. Contributing
10. License
```

The setup uses `uv sync --python 3.12`, `uv run python -m course list`, and explains that lessons remain `planned` until published. Do not claim performance results before controlled benchmarks exist.

- [ ] **Step 4: Add policy and concept documentation**

Use the unmodified Apache License 2.0 canonical text from <https://www.apache.org/licenses/LICENSE-2.0.txt>. Use Contributor Covenant 2.1 for `CODE_OF_CONDUCT.md`. `SECURITY.md` must state that private vulnerability reports belong in GitHub Security Advisories and that this educational server must not be exposed as a production service.

`docs/architecture.md` summarizes the API → engine → scheduler → model runner → KV manager flow and links to the full design spec. `docs/glossary.md` defines token, logit, sampling, prefill, decode, KV cache, batch, scheduler, TTFT, TPOT, ITL, and throughput in beginner language. `docs/roadmap.md` lists Milestones 0–5 from the spec without dates or promises.

- [ ] **Step 5: Run documentation and link-text checks**

Run: `uv run pytest tests/test_documentation.py -v`

Expected: PASS.

Run: `uv run ruff check .`

Expected: PASS; Markdown is not linted by Ruff, and Python snippets in docs are not executed in this task.

- [ ] **Step 6: Commit the open-source foundation**

```bash
git add README.md LICENSE CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md docs tests/test_documentation.py
git commit -m "docs: add learner and contributor foundation"
```

### Task 9: Add contribution templates and CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/ISSUE_TEMPLATE/bug.yml`
- Create: `.github/ISSUE_TEMPLATE/lesson.yml`
- Create: `.github/pull_request_template.md`
- Modify: `CONTRIBUTING.md`

**Interfaces:**
- Consumes: Locked dependencies and all repository verification commands.
- Produces: Required Linux quality gate and non-blocking macOS installation smoke job.

- [ ] **Step 1: Add a local CI-equivalent command to contributor docs**

Document this exact command sequence:

```bash
uv sync --frozen --python 3.12
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python scripts/verify_checkpoints.py
```

- [ ] **Step 2: Create the Linux CI job**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0
        with:
          enable-cache: true
      - run: uv sync --frozen --python 3.12
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy
      - run: uv run pytest
      - run: uv run python scripts/verify_checkpoints.py
```

- [ ] **Step 3: Add a macOS package smoke job**

Add `macos-smoke` on `macos-14` that checks out, installs uv, runs `uv sync --frozen --python 3.12`, and runs `uv run pytest tests/test_package.py tests/course/test_catalog.py`. Do not assert that MPS is present and do not run performance thresholds.

- [ ] **Step 4: Add structured contribution templates**

The bug form requires OS, architecture, Python/PyTorch versions, device selected, lesson ID, command, expected result, actual result, and minimal logs without secrets. The lesson-feedback form requires lesson ID, confusing section, time spent, failing command, hint level used, and suggested improvement. The pull-request template requires tests, checkpoint verification, beginner-language review, benchmark metadata when performance changes, and confirmation that no solution code leaks into starters.

- [ ] **Step 5: Run the complete local gate**

Run: `uv sync --frozen --python 3.12`

Run: `uv run ruff check .`

Run: `uv run ruff format --check .`

Run: `uv run mypy`

Run: `uv run pytest`

Run: `uv run python scripts/verify_checkpoints.py`

Expected: Every command exits zero; the test output includes the package, course CLI, workspace safety, runner isolation, checkpoint, template, and documentation tests.

- [ ] **Step 6: Inspect Git state for generated or learner files**

Run: `git status --short`

Expected: Only the intended `.github` and `CONTRIBUTING.md` changes appear; `.venv/`, `.superpowers/`, `work/`, and model/benchmark caches do not appear.

- [ ] **Step 7: Commit CI and contribution workflows**

```bash
git add .github CONTRIBUTING.md
git commit -m "ci: verify package and lesson checkpoints"
```

### Task 10: Final Milestone 0 acceptance and handoff

**Files:**
- Modify: `docs/roadmap.md`
- Create: `docs/milestones/00-foundation.md`

**Interfaces:**
- Consumes: All Milestone 0 commands and test results.
- Produces: A reproducible completion record and clear entry criteria for Lesson 1 work.

- [ ] **Step 1: Write the milestone acceptance record**

Record the exact tested commit, Python version, uv version, OS, and the exit status of:

```text
ruff check
ruff format --check
mypy
pytest
checkpoint verification
course list smoke test
template start/test/benchmark/reset smoke flow
```

The template smoke flow uses a temporary copy whose catalog marks the template published; it must prove the starter fails only the named challenge test, the solution passes, the benchmark emits valid JSON, and reset archives the learner file.

- [ ] **Step 2: Run the final CLI smoke sequence**

Run the repository test that automates the template smoke flow rather than editing `course/lessons.toml` manually:

Run: `uv run pytest tests/course/test_template.py -v`

Expected: PASS.

Run: `uv run python -m course list`

Expected: Fourteen ordered lessons, all marked `planned`.

- [ ] **Step 3: Mark Milestone 0 complete in the roadmap**

Change only the Milestone 0 status from `planned` to `complete`. State that Milestone 1 starts by publishing Lesson 1 and must not alter the established workspace/checkpoint contracts without an architecture decision record.

- [ ] **Step 4: Run final verification from a clean checkout state**

Run: `git diff --check`

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest && uv run python scripts/verify_checkpoints.py`

Expected: All commands exit zero.

- [ ] **Step 5: Commit the milestone record**

```bash
git add docs/roadmap.md docs/milestones/00-foundation.md
git commit -m "docs: record milestone zero acceptance"
```
