import tomllib
from pathlib import Path
from typing import Any

from course.errors import CourseError
from course.models import CheckpointReport, Lesson
from course.runner import run_tests

_REQUIRED_PATHS = (
    "README.md",
    "checkpoint.toml",
    "starter/inference_lab/__init__.py",
    "solution/inference_lab/__init__.py",
    "solution/NOTES.md",
    "tests",
)


def _expected_failures(lesson: Lesson) -> tuple[str, ...]:
    checkpoint = lesson.path / "checkpoint.toml"
    try:
        data: dict[str, Any] = tomllib.loads(checkpoint.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CourseError(f"Cannot read {lesson.lesson_id}/checkpoint.toml: {error}") from error
    raw = data.get("expected_starter_failures")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise CourseError("expected_starter_failures must be a list of test node IDs")
    if len(raw) != len(set(raw)):
        raise CourseError("expected_starter_failures must not contain duplicates")
    return tuple(sorted(raw))


def verify_checkpoint(project_root: Path, lesson: Lesson) -> CheckpointReport:
    for relative in _REQUIRED_PATHS:
        if not (lesson.path / relative).exists():
            raise CourseError(f"Lesson '{lesson.lesson_id}' is missing {relative}")

    expected = _expected_failures(lesson)
    starter = run_tests(project_root, lesson, lesson.path / "starter")
    if starter.returncode not in ({0} if not expected else {1}):
        raise CourseError(f"Starter tests could not be verified (pytest exit {starter.returncode})")
    if starter.failed_node_ids != expected:
        raise CourseError(
            "Starter failure mismatch: "
            f"expected {list(expected)}, actual {list(starter.failed_node_ids)}"
        )

    solution = run_tests(project_root, lesson, lesson.path / "solution")
    if solution.returncode != 0 or solution.failed_node_ids:
        raise CourseError(
            "Solution tests failed: "
            f"exit {solution.returncode}, failures {list(solution.failed_node_ids)}"
        )
    return CheckpointReport(lesson.lesson_id, starter.failed_node_ids, True)
