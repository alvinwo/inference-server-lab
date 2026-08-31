from pathlib import Path

from course import __main__ as cli


def make_published_checkpoint(root: Path) -> None:
    course_dir = root / "course"
    lesson_path = root / "lessons" / "01-example"
    course_dir.mkdir()
    (course_dir / "lessons.toml").write_text(
        '[[lessons]]\nnumber = "01"\nslug = "example"\ntitle = "Example"\nstatus = "published"\n'
    )
    for state, value in (("starter", "not-ready"), ("solution", "ready")):
        package = lesson_path / state / "inference_lab"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f'STATUS = "{value}"\n')
    tests = lesson_path / "tests"
    tests.mkdir()
    (tests / "test_challenge.py").write_text(
        'from inference_lab import STATUS\n\ndef test_status():\n    assert STATUS == "ready"\n'
    )
    (lesson_path / "README.md").write_text("# Example\n")
    (lesson_path / "starter" / "guided_lab.py").write_text('print("complete guided example")\n')
    (lesson_path / "solution" / "NOTES.md").write_text("# Notes\n")
    (lesson_path / "checkpoint.toml").write_text(
        'expected_starter_failures = ["tests/test_challenge.py::test_status"]\n'
    )
    (lesson_path / "benchmark.py").write_text(
        'import json\nfrom inference_lab import STATUS\nprint(json.dumps({"status": STATUS}))\n'
    )


def test_list_prints_status_and_title(capsys) -> None:
    assert cli.main(["list"]) == 0
    output = capsys.readouterr().out
    assert "01  published  Tensors, devices, and trustworthy timing" in output
    assert "14  planned    Load, resilience, and graduation benchmark" in output


def test_unknown_lesson_is_a_clean_cli_error(capsys) -> None:
    assert cli.main(["start", "99"]) == 2
    assert "Unknown lesson '99'" in capsys.readouterr().err


def test_reset_requires_explicit_confirmation(capsys) -> None:
    assert cli.main(["reset", "01"]) == 2
    assert "Refusing reset without --yes" in capsys.readouterr().err


def test_start_and_reset_archive_workspace(tmp_path: Path, monkeypatch, capsys) -> None:
    course_dir = tmp_path / "course"
    starter = tmp_path / "lessons" / "01-example" / "starter"
    course_dir.mkdir()
    starter.mkdir(parents=True)
    (starter / "answer.py").write_text("VALUE = 1\n")
    (course_dir / "lessons.toml").write_text(
        '[[lessons]]\nnumber = "01"\nslug = "example"\ntitle = "Example"\nstatus = "published"\n'
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    assert cli.main(["start", "01"]) == 0
    assert (tmp_path / "work" / "01-example" / "answer.py").exists()
    capsys.readouterr()
    assert cli.main(["reset", "01", "--yes"]) == 0
    output = capsys.readouterr().out
    assert "work/.trash/01-example-" in output
    assert not (tmp_path / "work" / "01-example").exists()


def test_test_benchmark_and_verify_commands(tmp_path: Path, monkeypatch, capsys) -> None:
    make_published_checkpoint(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    assert cli.main(["verify", "01"]) == 0
    assert "starter state expected; solution passes" in capsys.readouterr().out

    assert cli.main(["start", "01"]) == 0
    capsys.readouterr()
    assert cli.main(["test", "01"]) == 1
    assert "1 failed" in capsys.readouterr().out

    workspace_package = tmp_path / "work" / "01-example" / "inference_lab" / "__init__.py"
    workspace_package.write_text('STATUS = "ready"\n')
    assert cli.main(["test", "01"]) == 0
    assert "1 passed" in capsys.readouterr().out

    assert cli.main(["benchmark", "01"]) == 0
    assert '"status": "ready"' in capsys.readouterr().out


def test_test_command_reports_a_missing_workspace_cleanly(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    make_published_checkpoint(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    assert cli.main(["test", "01"]) == 2
    assert "Missing implementation directory" in capsys.readouterr().err


def test_steps_and_step_test_commands_guide_one_checkpoint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    make_published_checkpoint(tmp_path)
    checkpoint = tmp_path / "lessons" / "01-example" / "checkpoint.toml"
    checkpoint.write_text(
        'expected_starter_failures = ["tests/test_challenge.py::test_status"]\n\n'
        "[[steps]]\n"
        'id = "status"\n'
        'title = "Return the ready status"\n'
        'test_node_ids = ["tests/test_challenge.py::test_status"]\n'
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)

    assert cli.main(["steps", "01"]) == 0
    steps_output = capsys.readouterr().out
    assert "status" in steps_output
    assert "Return the ready status" in steps_output

    assert cli.main(["start", "01"]) == 0
    capsys.readouterr()
    assert cli.main(["test", "01", "--step", "status"]) == 1
    assert "1 failed" in capsys.readouterr().out

    assert cli.main(["test", "01", "--step", "missing"]) == 2
    assert "Unknown step 'missing'" in capsys.readouterr().err
