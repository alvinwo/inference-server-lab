from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LessonStatus(StrEnum):
    PLANNED = "planned"
    PUBLISHED = "published"


@dataclass(frozen=True, slots=True)
class Lesson:
    number: str
    slug: str
    title: str
    status: LessonStatus
    path: Path

    @property
    def lesson_id(self) -> str:
        return f"{self.number}-{self.slug}"
