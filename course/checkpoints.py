import tomllib
from pathlib import Path
from typing import Any

from course.errors import CourseError
from course.models import CheckpointReport, Lesson, LessonStep
from course.runner import run_guided_lab, run_tests

_REQUIRED_PATHS = (
    "README.md",
    "checkpoint.toml",
    "starter/guided_lab.py",
    "starter/inference_lab/__init__.py",
    "solution/inference_lab/__init__.py",
    "solution/NOTES.md",
    "tests",
)


def _checkpoint_data(lesson: Lesson) -> dict[str, Any]:
    checkpoint = lesson.path / "checkpoint.toml"
    try:
        return tomllib.loads(checkpoint.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CourseError(f"Cannot read {lesson.lesson_id}/checkpoint.toml: {error}") from error


def load_steps(lesson: Lesson) -> tuple[LessonStep, ...]:
    """Load the ordered learner checkpoints for a lesson."""
    data = _checkpoint_data(lesson)
    expected = set(_expected_failures_from_data(data))
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        raise CourseError("steps must be a list of guided checkpoint tables")

    steps: list[LessonStep] = []
    seen_ids: set[str] = set()
    seen_tests: set[str] = set()
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise CourseError("each guided checkpoint must be a table")
        step_id = raw_step.get("id")
        title = raw_step.get("title")
        raw_tests = raw_step.get("test_node_ids")
        if not isinstance(step_id, str) or not step_id:
            raise CourseError("each guided checkpoint needs a non-empty id")
        if step_id in seen_ids:
            raise CourseError(f"duplicate guided checkpoint id '{step_id}'")
        if not isinstance(title, str) or not title:
            raise CourseError(f"guided checkpoint '{step_id}' needs a non-empty title")
        if (
            not isinstance(raw_tests, list)
            or not raw_tests
            or not all(isinstance(item, str) for item in raw_tests)
        ):
            raise CourseError(f"guided checkpoint '{step_id}' needs test node IDs")
        for node_id in raw_tests:
            if node_id not in expected:
                raise CourseError(
                    f"guided checkpoint '{step_id}' contains unknown test node ID '{node_id}'"
                )
            if node_id in seen_tests:
                raise CourseError(f"test node ID '{node_id}' appears in more than one step")
            seen_tests.add(node_id)
        seen_ids.add(step_id)
        steps.append(LessonStep(step_id, title, tuple(raw_tests)))
    return tuple(steps)


def find_step(lesson: Lesson, step_id: str) -> LessonStep:
    for step in load_steps(lesson):
        if step.step_id == step_id:
            return step
    available = ", ".join(step.step_id for step in load_steps(lesson)) or "none"
    raise CourseError(
        f"Unknown step '{step_id}' for lesson {lesson.number}; choose from: {available}"
    )


def _expected_failures_from_data(data: dict[str, Any]) -> tuple[str, ...]:
    raw = data.get("expected_starter_failures")
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise CourseError("expected_starter_failures must be a list of test node IDs")
    if len(raw) != len(set(raw)):
        raise CourseError("expected_starter_failures must not contain duplicates")
    return tuple(sorted(raw))


def _expected_failures(lesson: Lesson) -> tuple[str, ...]:
    return _expected_failures_from_data(_checkpoint_data(lesson))


def verify_checkpoint(project_root: Path, lesson: Lesson) -> CheckpointReport:
    for relative in _REQUIRED_PATHS:
        if not (lesson.path / relative).exists():
            raise CourseError(f"Lesson '{lesson.lesson_id}' is missing {relative}")

    guided_lab = run_guided_lab(project_root, lesson, lesson.path / "starter")
    if guided_lab.returncode != 0:
        raise CourseError(
            f"Guided lab failed (exit {guided_lab.returncode}): {guided_lab.stderr.strip()}"
        )

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
