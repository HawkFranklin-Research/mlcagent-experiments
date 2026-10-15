import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "scorer", Path(__file__).parents[1] / "scripts/score_unsupervised.py"
)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def test_external_scorer(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "outputs.parquet"
    pd.DataFrame({"hidden": [0, 1, 0, 1, 0, 1], "feature": range(6)}).to_csv(
        source, index=False
    )
    rows = pd.DataFrame(
        {
            "row_id": range(6),
            "partition": ["train"] * 4 + ["test"] * 2,
            "cluster": [0, 1, 0, 1, 0, 1],
            "embedding_0": [-2, 2, -2, 2, -2, 2],
        }
    )
    rows.to_parquet(output, index=False)
    result = scorer.score(source, output, ["hidden"], "clustering")
    assert result["agent_work"] is False
    assert result["metrics"]["hidden"] == {"ari": 1.0, "nmi": 1.0}
    assert (
        scorer.score(source, output, ["hidden"], "representation")["metrics"]["hidden"][
            "accuracy"
        ]
        == 1
    )
    rows.loc[5, "row_id"] = 4
    rows.to_parquet(output, index=False)
    with pytest.raises(ValueError, match="unique"):
        scorer.score(source, output, ["hidden"], "representation")


def test_inference_score_is_external():
    location = Path(__file__).parents[1] / "scripts/score_inference.py"
    module_spec = importlib.util.spec_from_file_location("inference_scorer", location)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    expected = {
        "problem_type": "clustering",
        "targets": [],
        "group_column": None,
        "time_columns": [],
    }
    result = module.score(expected, expected)
    assert result["agent_work"] is False
    assert (
        result["problem_type"]
        and result["targets"]
        and result["group_column"]
        and result["time_columns"]
    )
