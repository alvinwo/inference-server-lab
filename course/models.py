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


@dataclass(frozen=True, slots=True)
class TestRun:
    returncode: int
    stdout: str
    stderr: str
    failed_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LessonStep:
    step_id: str
    title: str
    test_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CheckpointReport:
    lesson_id: str
    starter_failures: tuple[str, ...]
    solution_passed: bool
