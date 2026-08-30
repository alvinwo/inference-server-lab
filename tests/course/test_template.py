import json
import shutil
from pathlib import Path

from course.checkpoints import verify_checkpoint
from course.models import Lesson, LessonStatus
from course.runner import run_benchmark, run_tests
from course.workspace import WorkspaceManager


def test_author_template_has_expected_starter_and_passing_solution(
    project_root: Path,
) -> None:
    template = project_root / "lessons" / "_template"
    lesson = Lesson("00", "template", "Lesson template", LessonStatus.PUBLISHED, template)
    report = verify_checkpoint(project_root, lesson)
    assert report.starter_failures == (
        "tests/test_challenge.py::test_implementation_returns_ready",
    )
    assert report.solution_passed is True


def test_template_supports_the_complete_learner_flow(
    project_root: Path, tmp_path: Path
) -> None:
    template = project_root / "lessons" / "_template"
    lesson_path = tmp_path / "lessons" / "00-template"
    shutil.copytree(template, lesson_path)
    lesson = Lesson(
        "00", "template", "Lesson template", LessonStatus.PUBLISHED, lesson_path
    )
    manager = WorkspaceManager(tmp_path)
    workspace = manager.start(lesson)

    starter = run_tests(tmp_path, lesson, workspace)
    assert starter.failed_node_ids == (
        "tests/test_challenge.py::test_implementation_returns_ready",
    )

    shutil.copytree(
        lesson.path / "solution" / "inference_lab",
        workspace / "inference_lab",
        dirs_exist_ok=True,
    )
    solution = run_tests(tmp_path, lesson, workspace)
    assert solution.returncode == 0

    benchmark = run_benchmark(tmp_path, lesson, workspace)
    payload = json.loads(benchmark.stdout)
    assert payload["implementation_status"] == "ready"
    assert payload["duration_seconds"] >= 0

    (workspace / "learner-note.txt").write_text("preserve this")
    archive = manager.archive(lesson, timestamp="20260829T120000Z")
    assert (archive / "learner-note.txt").read_text() == "preserve this"
