import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture(autouse=True)
def _no_real_env_preflight(monkeypatch):
    """Controller tests use stub workers; the real offline build is tested separately."""
    import run_batch

    monkeypatch.setattr(
        run_batch,
        "run_env_preflight",
        lambda: {
            "python": "test",
            "executable": "test",
            "sets": {},
            "at": "t",
            "host": {"hostname": "test", "cpus": 1},
        },
    )
