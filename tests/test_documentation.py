from pathlib import Path


def test_readme_contains_required_expectations(project_root: Path) -> None:
    readme = (project_root / "README.md").read_text()
    assert "educational" in readme.lower()
    assert "CPU" in readme
    assert "Apple Silicon" in readme
    assert "python -m course list" in readme
    assert "not a production replacement" in readme.lower()


def test_open_source_policy_files_exist(project_root: Path) -> None:
    required = {
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "docs/architecture.md",
        "docs/glossary.md",
        "docs/roadmap.md",
    }
    paths = {str(path.relative_to(project_root)) for path in project_root.rglob("*")}
    assert required <= paths


def test_license_is_apache_2(project_root: Path) -> None:
    license_text = (project_root / "LICENSE").read_text()
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text


def test_readme_headings_follow_the_learning_path(project_root: Path) -> None:
    readme = (project_root / "README.md").read_text()
    headings = [line for line in readme.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Who this is for",
        "## What you will build",
        "## Hardware: CPU required, Apple Silicon/MPS optional, NVIDIA not required",
        "## Five-minute setup",
        "## Version 1 lesson map",
        "## How starter, tests, solution, and engineering notes work",
        "## Educational limitations",
        "## Contributing",
        "## License",
    ]


def test_ci_and_contribution_templates_exist(project_root: Path) -> None:
    required = {
        ".github/workflows/ci.yml",
        ".github/ISSUE_TEMPLATE/bug.yml",
        ".github/ISSUE_TEMPLATE/lesson.yml",
        ".github/pull_request_template.md",
    }
    paths = {str(path.relative_to(project_root)) for path in project_root.rglob("*")}
    assert required <= paths
    workflow = (project_root / ".github" / "workflows" / "ci.yml").read_text()
    for command in (
        "uv sync --frozen --python 3.12",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run mypy",
        "uv run pytest",
        "uv run python scripts/verify_checkpoints.py",
    ):
        assert command in workflow


def test_milestone_zero_has_a_completion_record(project_root: Path) -> None:
    roadmap = (project_root / "docs" / "roadmap.md").read_text()
    record = project_root / "docs" / "milestones" / "00-foundation.md"
    assert "Milestone 0 — Project skeleton (`complete`)" in roadmap
    assert record.is_file()
    record_text = record.read_text()
    assert "3ca476a3151a31560c76c3a8f7633e4553481e6d" in record_text
    assert "37 passed" in record_text
