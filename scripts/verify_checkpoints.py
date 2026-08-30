from pathlib import Path

from course.catalog import load_catalog
from course.checkpoints import verify_checkpoint
from course.errors import CourseError
from course.models import Lesson, LessonStatus


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    template = Lesson(
        "00",
        "template",
        "Lesson author template",
        LessonStatus.PUBLISHED,
        project_root / "lessons" / "_template",
    )
    lessons = (
        template,
        *(
            lesson
            for lesson in load_catalog(project_root)
            if lesson.status is LessonStatus.PUBLISHED
        ),
    )
    try:
        for lesson in lessons:
            verify_checkpoint(project_root, lesson)
            label = "_template" if lesson.number == "00" else lesson.lesson_id
            print(f"{label}: starter state expected; solution passes")
    except CourseError as error:
        print(f"Checkpoint verification failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
