import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "development", Path(__file__).parents[1] / "scripts/develop_rails.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_development_metadata_selected():
    entries = [
        {"id": str(i), "role": "development", "level": "simple"} for i in range(12)
    ]
    entries.append({"id": "unseen", "role": "heldout", "file": "/never/open"})
    assert len(module.development_entries(entries)) == 12
    with pytest.raises(ValueError):
        module.development_entries(entries, ["unseen"])
    assert [e["id"] for e in module.development_entries(entries, ["1"])] == ["1"]
    assert {v["model"] for v in module.CONFIG["roles"].values()} == {"openrouter/free"}
    assert set(module.CONFIG["roles"]) == {"navigator", "worker"}
    assert module.CONFIG["sandbox"]["allow_level_b"] is False


def test_quota_metadata_does_not_include_credentials(monkeypatch):
    import io

    monkeypatch.setenv("OPENROUTER_API_KEY", "distinctive-fake-secret")

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *a, **k: Reply(
            json.dumps(
                {
                    "data": {
                        "label": "private-label",
                        "free_model_daily_requests": {
                            "used": 0,
                            "limit": 1000,
                            "remaining": 1000,
                        },
                    }
                }
            ).encode()
        ),
    )
    result = module.quota()
    assert result["free_model_daily_requests"]["remaining"] == 1000
    assert "distinctive-fake-secret" not in json.dumps(result)
    assert "private-label" not in json.dumps(result)
