from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safeguard-tests"))

from run_flaws import write_matrix  # noqa: E402


def test_flaw_catalog_has_control_script_for_every_case():
    spec = yaml.safe_load((ROOT / "safeguard-tests/flaws.yaml").read_text())
    assert spec["reviewer"] == "R1-5"
    assert spec["cases"]
    for case in spec["cases"]:
        assert case["id"] and case["planted_script"] and case["control_script"]
        assert case["template"] and case["brief"] and case["setup"]


def test_matrix_writer_emits_stub_row(tmp_path):
    row = {
        "flaw": "F-stub/control",
        "table": "synthetic",
        "outcome": "control_passed",
        "check": "clean fixture",
        "message": "reached report",
        "run_path": "runs/stub/project",
    }
    write_matrix([row], tmp_path)
    csv_text = (tmp_path / "matrix.csv").read_text()
    markdown = (tmp_path / "MATRIX.md").read_text()
    assert "F-stub/control" in csv_text and "control_passed" in csv_text
    assert "F-stub/control" in markdown and "clean fixture" in markdown
