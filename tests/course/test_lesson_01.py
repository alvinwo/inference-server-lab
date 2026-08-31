import json
import subprocess
import sys
from pathlib import Path

from course.catalog import find_lesson
from course.checkpoints import verify_checkpoint
from course.models import LessonStatus
from course.runner import run_benchmark

EXPECTED_STARTER_FAILURES = (
    "tests/test_devices.py::test_auto_policy_reports_actual_device",
    "tests/test_devices.py::test_cpu_policy_reports_requested_and_selected_device",
    "tests/test_devices.py::test_explicit_mps_never_silently_falls_back",
    "tests/test_devices.py::test_seed_everything_repeats_python_and_torch_sequences",
    "tests/test_devices.py::test_unknown_device_preference_is_rejected",
    "tests/test_tensor_ops.py::test_affine_matches_hand_calculated_values",
    "tests/test_tensor_ops.py::test_affine_preserves_dtype_and_device",
    "tests/test_tensor_ops.py::test_affine_rejects_incompatible_shapes",
    "tests/test_tensor_ops.py::test_affine_supports_batch_and_token_dimensions",
    "tests/test_timing.py::test_benchmark_rejects_invalid_iteration_counts",
    "tests/test_timing.py::test_benchmark_runs_warmup_and_synchronizes_each_sample",
    "tests/test_timing.py::test_synchronize_accepts_cpu",
    "tests/test_timing.py::test_synchronize_rejects_unsupported_device",
)


def test_lesson_01_is_published_with_a_verified_checkpoint(project_root: Path) -> None:
    lesson = find_lesson(project_root, "01")
    assert lesson.status is LessonStatus.PUBLISHED
    report = verify_checkpoint(project_root, lesson)
    assert report.starter_failures == EXPECTED_STARTER_FAILURES
    assert report.solution_passed is True


def test_lesson_01_solution_benchmark_emits_reproducible_metadata(
    project_root: Path,
) -> None:
    lesson = find_lesson(project_root, "01")
    result = run_benchmark(project_root, lesson, lesson.path / "solution")
    assert result.returncode == 0, result.stderr
    assert "Failed to initialize NumPy" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["lesson_id"] == "01-tensors-devices"
    assert payload["requested_device"] == "cpu"
    assert payload["selected_device"] == "cpu"
    assert payload["dtype"] == "torch.float32"
    assert payload["shape"] == [2, 16, 32]
    assert payload["warmup_iterations"] == 3
    assert payload["iterations"] == 10
    assert len(payload["samples_ms"]) == 10
    assert payload["median_ms"] >= 0
    assert payload["python_version"]
    assert payload["torch_version"]
    assert payload["platform"]


def test_lesson_01_concept_lab_runs_before_the_challenge(project_root: Path) -> None:
    lesson = find_lesson(project_root, "01")
    result = subprocess.run(
        [sys.executable, str(lesson.path / "starter" / "concept_lab.py")],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "Concept lab complete" in result.stdout
