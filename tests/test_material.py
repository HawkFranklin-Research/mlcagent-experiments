import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mlca.config import Config
from mlca.evaluation.bench import TaskCard
from mlca.templates.registry import get
from prepare import prepare


def test_all_cards_validate():
    paths = list((ROOT / "bench/cards").glob("*.yaml")) + list(
        (ROOT / "bio-bench").glob("S*/task_card*.yaml")
    )
    assert len(paths) >= 40
    for path in paths:
        card = TaskCard.model_validate(yaml.safe_load(path.read_text()))
        Config.model_validate(card.config)
        assert all(
            item["event"] in get(card.template).events
            for item in card.custom_instructions
            if isinstance(item, dict)
        )


def test_preparation_counts_and_hashes():
    out = ROOT / "data_prep"
    expected = {
        "ER": {"1": 81, "0": 39},
        "PR": {"1": 68, "0": 47},
        "HER2": {"1": 15, "0": 107},
    }
    import json

    s2 = json.loads((out / "S2/manifest.json").read_text())
    for item in s2["outputs"]:
        assert (
            item["labels"] == expected[item["file"].split("_")[1].split(".")[0].upper()]
        )
        assert (
            hashlib.sha256((out / "S2" / item["file"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    s4 = json.loads((out / "S4/manifest.json").read_text())
    assert [item["rows"] for item in s4["outputs"]] == [415, 102]


def test_s1_checksum_only_is_repeatable(tmp_path):
    import pandas as pd

    source = tmp_path / "source/final results/hfr/Anemia"
    source.mkdir(parents=True)
    pd.DataFrame({"Result": [0, 1]}).to_csv(source / "anemia.csv", index=False)
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"SOURCE_DATA": str(tmp_path / "source")}))
    a = prepare("S1", config, tmp_path / "a")
    b = prepare("S1", config, tmp_path / "b")
    assert a == b
    assert not (tmp_path / "a/S1/anemia.csv").exists()
