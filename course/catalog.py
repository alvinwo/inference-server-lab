import tomllib
from pathlib import Path
from typing import Any

from course.errors import CourseError
from course.models import Lesson, LessonStatus


def _required_string(entry: dict[str, Any], field: str, index: int) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CourseError(f"Lesson {index} has an invalid '{field}'")
    return value


def load_catalog(project_root: Path) -> tuple[Lesson, ...]:
    catalog_path = project_root / "course" / "lessons.toml"
    try:
        data = tomllib.loads(catalog_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise CourseError(f"Cannot load lesson catalog: {error}") from error

    entries = data.get("lessons")
    if not isinstance(entries, list) or not entries:
        raise CourseError("Lesson catalog is empty")

    lessons: list[Lesson] = []
    numbers: set[str] = set()
    lesson_ids: set[str] = set()
    for index, raw_entry in enumerate(entries, start=1):
        if not isinstance(raw_entry, dict):
            raise CourseError(f"Lesson {index} must be a TOML table")
        entry: dict[str, Any] = raw_entry
        number = _required_string(entry, "number", index)
        slug = _required_string(entry, "slug", index)
        title = _required_string(entry, "title", index)
        status_value = _required_string(entry, "status", index)
        if len(number) != 2 or not number.isdigit():
            raise CourseError(f"Lesson number '{number}' must contain two digits")
        try:
            status = LessonStatus(status_value)
        except ValueError as error:
            raise CourseError(f"Unknown lesson status '{status_value}'") from error
        lesson_id = f"{number}-{slug}"
        if number in numbers:
            raise CourseError(f"Duplicate lesson number '{number}'")
        if lesson_id in lesson_ids:
            raise CourseError(f"Duplicate lesson ID '{lesson_id}'")
        numbers.add(number)
        lesson_ids.add(lesson_id)
        lessons.append(Lesson(number, slug, title, status, project_root / "lessons" / lesson_id))
    return tuple(lessons)


def find_lesson(project_root: Path, lesson_id: str) -> Lesson:
    lessons = load_catalog(project_root)
    for lesson in lessons:
        if lesson_id in {lesson.number, lesson.lesson_id}:
            return lesson
    valid_ids = ", ".join(lesson.lesson_id for lesson in lessons)
    raise CourseError(f"Unknown lesson '{lesson_id}'. Valid lessons: {valid_ids}")
