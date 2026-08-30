import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from course.errors import CourseError
from course.models import Lesson, LessonStatus, TestRun


def _environment(project_root: Path, implementation_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [str(implementation_root.resolve()), str(project_root.resolve())]
    inherited = environment.get("PYTHONPATH")
    if inherited:
        python_paths.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def _validate_lesson(lesson: Lesson, implementation_root: Path) -> None:
    if lesson.status is not LessonStatus.PUBLISHED:
        raise CourseError(f"Lesson '{lesson.lesson_id}' is not published")
    if not implementation_root.is_dir():
        raise CourseError(f"Missing implementation directory '{implementation_root}'")


def _failed_node_ids(report_path: Path) -> tuple[str, ...]:
    if not report_path.is_file():
        return ()
    try:
        raw: Any = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(raw, list):
        return ()
    failures: list[str] = []
    for item in raw:
        if (
            isinstance(item, dict)
            and item.get("outcome") == "failed"
            and isinstance(item.get("nodeid"), str)
        ):
            failures.append(item["nodeid"])
    return tuple(sorted(failures))


def run_tests(project_root: Path, lesson: Lesson, implementation_root: Path) -> TestRun:
    _validate_lesson(lesson, implementation_root)
    tests = lesson.path / "tests"
    if not tests.is_dir():
        raise CourseError(f"Lesson '{lesson.lesson_id}' has no tests")
    with tempfile.TemporaryDirectory(prefix="inference-lab-pytest-") as temporary:
        report_path = Path(temporary) / "report.json"
        environment = _environment(project_root, implementation_root)
        environment["INFERENCE_LAB_PYTEST_REPORT"] = str(report_path)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-p", "course.pytest_reporter"],
            cwd=lesson.path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        failures = _failed_node_ids(report_path)
    return TestRun(result.returncode, result.stdout, result.stderr, failures)


def run_benchmark(
    project_root: Path, lesson: Lesson, implementation_root: Path
) -> subprocess.CompletedProcess[str]:
    _validate_lesson(lesson, implementation_root)
    benchmark = lesson.path / "benchmark.py"
    if not benchmark.is_file():
        raise CourseError(f"Lesson '{lesson.lesson_id}' has no benchmark yet")
    return subprocess.run(
        [sys.executable, str(benchmark)],
        cwd=lesson.path,
        env=_environment(project_root, implementation_root),
        capture_output=True,
        text=True,
        check=False,
    )
