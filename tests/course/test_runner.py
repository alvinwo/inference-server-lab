import json
from pathlib import Path

import pytest

from course.errors import CourseError
from course.models import Lesson, LessonStatus
from course.runner import run_benchmark, run_tests


def example_lesson(tmp_path: Path) -> tuple[Lesson, Path]:
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
    return Lesson("01", "example", "Example", LessonStatus.PUBLISHED, lesson_path), implementation


def test_run_tests_imports_workspace_before_installed_package(tmp_path: Path) -> None:
    lesson, implementation = example_lesson(tmp_path)

    result = run_tests(tmp_path, lesson, implementation)

    assert result.returncode == 0
    assert result.failed_node_ids == ()
    assert "1 passed" in result.stdout


def test_run_tests_reports_failed_node_ids(tmp_path: Path) -> None:
    lesson, implementation = example_lesson(tmp_path)
    (lesson.path / "tests" / "test_origin.py").write_text(
        "def test_expected_failure():\n    assert False\n"
    )
    result = run_tests(tmp_path, lesson, implementation)
    assert result.returncode == 1
    assert result.failed_node_ids == ("tests/test_origin.py::test_expected_failure",)


def test_run_tests_rejects_missing_implementation(tmp_path: Path) -> None:
    lesson, implementation = example_lesson(tmp_path)
    implementation.rename(tmp_path / "gone")
    with pytest.raises(CourseError, match="implementation directory"):
        run_tests(tmp_path, lesson, implementation)


def test_run_benchmark_uses_workspace_and_returns_json(tmp_path: Path) -> None:
    lesson, implementation = example_lesson(tmp_path)
    (lesson.path / "benchmark.py").write_text(
        'import json\nfrom inference_lab import ORIGIN\nprint(json.dumps({"origin": ORIGIN}))\n'
    )
    result = run_benchmark(tmp_path, lesson, implementation)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"origin": "workspace"}


def test_run_benchmark_rejects_missing_script(tmp_path: Path) -> None:
    lesson, implementation = example_lesson(tmp_path)
    with pytest.raises(CourseError, match="has no benchmark yet"):
        run_benchmark(tmp_path, lesson, implementation)
