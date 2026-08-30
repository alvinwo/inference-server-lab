import shutil
from datetime import UTC, datetime
from pathlib import Path

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

    def archive(self, lesson: Lesson, timestamp: str | None = None) -> Path:
        active = self._destination(lesson)
        if not active.is_dir():
            raise CourseError(f"No active workspace for lesson '{lesson.lesson_id}'")

        trash = (self.work_root / ".trash").resolve()
        if trash.parent != self.work_root:
            raise CourseError("Unsafe workspace archive path")
        trash.mkdir(parents=True, exist_ok=True)
        archive_time = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base_name = f"{lesson.lesson_id}-{archive_time}"
        archived = trash / base_name
        suffix = 2
        while archived.exists():
            archived = trash / f"{base_name}-{suffix}"
            suffix += 1
        shutil.move(active, archived)
        return archived
