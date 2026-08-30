import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from course.catalog import find_lesson, load_catalog
from course.checkpoints import verify_checkpoint
from course.errors import CourseError
from course.runner import run_benchmark, run_tests
from course.workspace import WorkspaceManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m course",
        description="Navigate the Inference Server Lab challenges.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List every lesson and publication status.")

    start = commands.add_parser("start", help="Copy a lesson starter into work/.")
    start.add_argument("lesson")

    reset = commands.add_parser("reset", help="Archive an active lesson workspace.")
    reset.add_argument("lesson")
    reset.add_argument("--yes", action="store_true", help="Confirm the recoverable reset.")

    for name, help_text in (
        ("test", "Run a lesson's tests against your workspace."),
        ("benchmark", "Run a lesson's benchmark against your workspace."),
        ("verify", "Verify a lesson's starter and solution contract."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("lesson")
    return parser


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "list":
            for lesson in load_catalog(PROJECT_ROOT):
                print(f"{lesson.number}  {lesson.status.value:<9}  {lesson.title}")
            return 0

        if args.command == "reset" and not args.yes:
            raise CourseError("Refusing reset without --yes; your workspace was not changed.")

        lesson = find_lesson(PROJECT_ROOT, args.lesson)
        manager = WorkspaceManager(PROJECT_ROOT)
        if args.command == "start":
            destination = manager.start(lesson)
            print(f"Started {lesson.lesson_id} in {_display_path(destination)}")
            return 0
        if args.command == "reset":
            archived = manager.archive(lesson)
            print(f"Archived {lesson.lesson_id} to {_display_path(archived)}")
            return 0
        if args.command == "test":
            test_result = run_tests(PROJECT_ROOT, lesson, manager._destination(lesson))
            print(test_result.stdout, end="")
            print(test_result.stderr, end="", file=sys.stderr)
            return test_result.returncode
        if args.command == "benchmark":
            benchmark_result = run_benchmark(PROJECT_ROOT, lesson, manager._destination(lesson))
            print(benchmark_result.stdout, end="")
            print(benchmark_result.stderr, end="", file=sys.stderr)
            return benchmark_result.returncode
        if args.command == "verify":
            report = verify_checkpoint(PROJECT_ROOT, lesson)
            print(f"{report.lesson_id}: starter state expected; solution passes")
            return 0
        raise CourseError(f"Unsupported command '{args.command}'")
    except CourseError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
