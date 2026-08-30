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


@pytest.mark.parametrize(
    "catalog_text, message",
    [
        ("", "empty"),
        (
            "[[lessons]]\nnumber='1'\nslug='bad'\ntitle='Bad'\nstatus='planned'\n",
            "two digits",
        ),
        (
            "[[lessons]]\nnumber='01'\nslug='bad'\ntitle='Bad'\nstatus='unknown'\n",
            "Unknown lesson status",
        ),
    ],
)
def test_catalog_rejects_invalid_entries(
    tmp_path: Path, catalog_text: str, message: str
) -> None:
    course_dir = tmp_path / "course"
    course_dir.mkdir()
    (course_dir / "lessons.toml").write_text(catalog_text)
    with pytest.raises(CourseError, match=message):
        load_catalog(tmp_path)
