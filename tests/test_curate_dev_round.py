import hashlib

import pytest
from curate_dev_round import curate


def test_setup_allowlist_preserves_originals_and_omits_data(tmp_path):
    source = tmp_path / "table/setup-001/.mlca/llm/response.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"response":"synthetic"}')
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    private = tmp_path / "table/setup-001/.mlca/data/private.csv"
    private.parent.mkdir()
    private.write_text("DO_NOT_COPY")
    assert curate(tmp_path) == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    assert (
        tmp_path / "setup-evidence/table/setup-001/.mlca/llm/response.json"
    ).is_file()
    assert not (tmp_path / "setup-evidence/table/setup-001/.mlca/data").exists()
    assert "DO_NOT_COPY" not in (tmp_path / "setup-evidence/MANIFEST.csv").read_text()
    with pytest.raises(FileExistsError):
        curate(tmp_path)
