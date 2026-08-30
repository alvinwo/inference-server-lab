import json
import os
from pathlib import Path

from _pytest.config import ExitCode
from _pytest.main import Session
from _pytest.reports import TestReport

_REPORTS: list[dict[str, str]] = []


def pytest_sessionstart(session: Session) -> None:
    del session
    _REPORTS.clear()


def pytest_runtest_logreport(report: TestReport) -> None:
    if report.when == "call":
        _REPORTS.append({"nodeid": report.nodeid, "outcome": report.outcome})


def pytest_sessionfinish(session: Session, exitstatus: ExitCode) -> None:
    del session, exitstatus
    destination = os.environ.get("INFERENCE_LAB_PYTEST_REPORT")
    if destination:
        Path(destination).write_text(json.dumps(_REPORTS))
