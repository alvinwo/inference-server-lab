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


def test_verify_checkpoint_rejects_a_broken_solution(tmp_path: Path) -> None:
    lesson = make_checkpoint(tmp_path, "tests/test_challenge.py::test_value_is_ready")
    (lesson.path / "solution" / "inference_lab" / "__init__.py").write_text(
        'VALUE = "still-broken"\n'
    )
    with pytest.raises(CourseError, match="Solution tests failed"):
        verify_checkpoint(tmp_path, lesson)


def test_verify_checkpoint_requires_contract_files(tmp_path: Path) -> None:
    lesson = make_checkpoint(tmp_path, "tests/test_challenge.py::test_value_is_ready")
    (lesson.path / "solution" / "NOTES.md").unlink()
    with pytest.raises(CourseError, match=r"solution/NOTES\.md"):
        verify_checkpoint(tmp_path, lesson)
