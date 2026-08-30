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


def test_archive_moves_work_to_recoverable_trash(tmp_path: Path) -> None:
    lesson = published_lesson(tmp_path)
    manager = WorkspaceManager(tmp_path)
    active = manager.start(lesson)
    (active / "notes.txt").write_text("keep me")

    archived = manager.archive(lesson, timestamp="20260829T120000Z")

    assert not active.exists()
    assert archived == tmp_path / "work" / ".trash" / "01-example-20260829T120000Z"
    assert (archived / "notes.txt").read_text() == "keep me"


def test_archive_uses_a_unique_name(tmp_path: Path) -> None:
    lesson = published_lesson(tmp_path)
    manager = WorkspaceManager(tmp_path)
    manager.start(lesson)
    first = manager.archive(lesson, timestamp="20260829T120000Z")
    manager.start(lesson)
    second = manager.archive(lesson, timestamp="20260829T120000Z")
    assert first.name == "01-example-20260829T120000Z"
    assert second.name == "01-example-20260829T120000Z-2"


def test_archive_refuses_when_no_workspace_exists(tmp_path: Path) -> None:
    lesson = published_lesson(tmp_path)
    manager = WorkspaceManager(tmp_path, work_root=tmp_path / "work")
    with pytest.raises(CourseError, match="No active workspace"):
        manager.archive(lesson, timestamp="20260829T120000Z")
