from pathlib import Path

from course import __main__ as cli


def test_list_prints_status_and_title(capsys) -> None:
    assert cli.main(["list"]) == 0
    output = capsys.readouterr().out
    assert "01  planned    Tensors, devices, and trustworthy timing" in output
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
