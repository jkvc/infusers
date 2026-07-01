"""End-to-end tests against deployed Modal infrastructure (opt-in: pytest -m modal)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODAL_APP = "infusers/modal_app/dummy_image.py"
DUMMY_PATH = "dummy.image"


def _run_modal(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "modal", *args]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _modal_available() -> bool:
    if os.environ.get("SKIP_MODAL_E2E") == "1":
        return False
    try:
        result = _run_modal(["profile", "current"], timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


pytestmark = pytest.mark.modal


@pytest.fixture(scope="module")
def require_modal() -> None:
    if not _modal_available():
        pytest.skip("Modal not configured (run `uv run modal setup`) or SKIP_MODAL_E2E=1")


@pytest.fixture(scope="module")
def deployed_app(require_modal: None) -> None:
    result = _run_modal(["deploy", MODAL_APP], timeout=900)
    if result.returncode != 0:
        pytest.fail(f"modal deploy failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def test_modal_describe(deployed_app: None) -> None:
    result = _run_modal(["run", f"{MODAL_APP}::smoke_describe"], timeout=300)
    assert result.returncode == 0, f"smoke_describe failed:\n{result.stderr}\n{result.stdout}"
    assert DUMMY_PATH in result.stdout


def test_modal_infer_smoke(deployed_app: None) -> None:
    result = _run_modal(
        [
            "run",
            f"{MODAL_APP}::smoke",
            "--prompt",
            "dummy e2e json",
            "--seed",
            "42",
        ],
        timeout=300,
    )
    assert result.returncode == 0, f"smoke infer failed:\n{result.stderr}\n{result.stdout}"
    assert "/tmp/dummy-smoke.webp" in result.stdout

    webp_path = Path("/tmp/dummy-smoke.webp")
    assert webp_path.is_file()
    assert webp_path.stat().st_size > 50


def test_modal_infer_stream(deployed_app: None) -> None:
    result = _run_modal(
        [
            "run",
            f"{MODAL_APP}::smoke_stream",
            "--prompt",
            "dummy e2e stream",
            "--seed",
            "7",
        ],
        timeout=300,
    )
    assert result.returncode == 0, f"smoke_stream failed:\n{result.stderr}\n{result.stdout}"
    assert "progress events:" in result.stdout
    assert "/tmp/dummy-smoke-stream.webp" in result.stdout

    webp_path = Path("/tmp/dummy-smoke-stream.webp")
    assert webp_path.is_file()
    assert webp_path.stat().st_size > 50
