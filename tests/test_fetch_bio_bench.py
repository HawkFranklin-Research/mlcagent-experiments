import hashlib
import json

import fetch_bio_bench as fetch
import pytest

REVISION = "a" * 40


def release(files):
    lines = [
        f"{hashlib.sha256(body).hexdigest()}  {path}" for path, body in files.items()
    ]
    return {**files, "SHA256SUMS.txt": "\n".join(lines).encode()}


def fake(store, calls):
    def get(address, secret, destination):
        assert secret == "secret" and "secret" not in address
        path = address.split(f"/resolve/{REVISION}/", 1)[1]
        calls.append(path)
        destination.write_bytes(store[path])

    return get


def test_downloads_only_requested_files_and_verifies_hashes(tmp_path):
    store = release(
        {
            "MANIFEST.csv": b"path\n",
            "prepared/S1/anemia.csv": b"a,b\n1,2\n",
            "raw/source.csv": b"x\n",
        }
    )
    calls = []
    record = fetch.fetch_release(
        "org/repo", REVISION, tmp_path, "secret", fetch=fake(store, calls)
    )
    assert set(record["files"]) == {"MANIFEST.csv", "prepared/S1/anemia.csv"}
    assert "raw/source.csv" not in calls
    assert (tmp_path / "prepared/S1/anemia.csv").read_bytes() == b"a,b\n1,2\n"
    assert "secret" not in (tmp_path / "fetch.json").read_text()
    calls.clear()
    fetch.fetch_release(
        "org/repo", REVISION, tmp_path, "secret", fetch=fake(store, calls)
    )
    assert calls == ["SHA256SUMS.txt"]  # verified files are kept, not downloaded again
    assert fetch.verify(tmp_path)["revision"] == REVISION


def test_changed_file_is_refused_and_not_kept(tmp_path):
    store = release({"MANIFEST.csv": b"path\n", "prepared/S1/anemia.csv": b"good"})
    store["prepared/S1/anemia.csv"] = b"tampered"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch.fetch_release(
            "org/repo", REVISION, tmp_path, "secret", fetch=fake(store, [])
        )
    assert not (tmp_path / "prepared/S1/anemia.csv").exists()


def test_revision_must_be_pinned(tmp_path):
    with pytest.raises(ValueError, match="40-character"):
        fetch.fetch_release("org/repo", "main", tmp_path, "secret", fetch=fake({}, []))


def test_token_from_file_is_stripped(tmp_path):
    secret = tmp_path / "token.txt"
    secret.write_text(" value\n")
    assert fetch.token(secret) == "value"
    assert json.dumps({"ok": True})
