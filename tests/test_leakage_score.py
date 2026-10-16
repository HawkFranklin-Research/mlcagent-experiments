import pandas as pd
from score_leakage import evaluate


def test_three_independent_criteria_and_failure():
    frame = pd.DataFrame({"subject": [1, 1, 2, 2, 3, 3], "time": range(6)})
    splits = pd.DataFrame(
        {"row_id": range(6), "partition": ["train"] * 2 + ["val"] * 2 + ["test"] * 2}
    )
    expected = {
        "leakage_columns": ["source_score"],
        "identifier_columns": ["row_key"],
        "targets": ["outcome"],
        "group_column": "subject",
        "time_columns": ["time"],
    }
    result = evaluate(
        [{"columns": ["measurement"]}], frame, splits, expected, ["synthetic"]
    )
    assert result["leakage_clean"] is True and result["agent_work"] is False
    result = evaluate(
        [{"columns": ["source_score"]}], frame, splits, expected, ["synthetic"]
    )
    assert result["leakage_clean"] is False
    assert result["criteria"]["group_disjoint"]["value"] is True
    frame.loc[2, "subject"] = 1
    result = evaluate([{"columns": ["measurement"]}], frame, splits, expected, [])
    assert result["criteria"]["group_disjoint"]["value"] is False
    frame["time"] = list(reversed(range(6)))
    result = evaluate([{"columns": ["measurement"]}], frame, splits, expected, [])
    assert result["criteria"]["time_ordered"]["value"] is False


def test_unknown_evidence_never_passes():
    result = evaluate(
        [{}], None, None, {"group_column": "subject", "time_columns": ["date"]}, []
    )
    assert result["leakage_clean"] is None
    assert all(r["value"] is None for r in result["criteria"].values())
