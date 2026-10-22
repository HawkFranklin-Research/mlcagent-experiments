from score_inference import score

SURVIVAL = {
    "problem_type": "survival",
    "targets": [],
    "duration": "months",
    "event_column": "status",
    "group_column": None,
    "time_columns": [],
}


def test_survival_outcome_is_scored_on_duration_and_event():
    proposal = {**SURVIVAL, "targets": ["status", "months"]}
    assert score(proposal, SURVIVAL)["targets"] is True
    assert score({**proposal, "event_column": "other"}, SURVIVAL)["targets"] is False
    assert score({**proposal, "duration": None}, SURVIVAL)["targets"] is False


def test_other_tasks_still_compare_targets():
    expected = {"problem_type": "binary", "targets": ["y"]}
    assert (
        score({"problem_type": "binary", "targets": ["y"]}, expected)["targets"] is True
    )
    assert (
        score({"problem_type": "binary", "targets": ["z"]}, expected)["targets"]
        is False
    )
