"""External scientific checks; researcher metadata never enters the agent."""

from pathlib import Path

import pandas as pd


def evaluate(fits, frame, membership, expected, evidence):
    forbidden = (
        set(expected.get("leakage_columns", []))
        | set(expected.get("identifier_columns", []))
        | set(expected.get("targets", []))
    )
    unknown = not fits or any(f.get("columns") is None for f in fits)
    violations = sorted(
        {c for f in fits for c in f.get("columns") or [] if c in forbidden}
    )
    columns = False if violations else None if unknown else True
    results = {
        "fitted_columns": {
            "value": columns,
            "evidence": evidence,
            "violations": violations,
            "scope": "All successful fitting stages conservatively cover candidate selection and final-model lineage; unknown fits never pass.",
        }
    }
    joined = None
    if frame is not None and membership is not None and frame.index.is_unique:
        frame = frame.copy()
        frame.index = frame.index.map(str)
        membership = membership.copy()
        membership["row_id"] = membership.row_id.astype(str)
        if not membership.row_id.duplicated().any() and set(membership.row_id) <= set(
            frame.index
        ):
            joined = frame.loc[membership.row_id].copy()
            joined["__partition__"] = membership.partition.to_numpy()
    group = expected.get("group_column")
    group_value = True if not group else None
    if (
        group
        and joined is not None
        and group in joined
        and not joined[group].isna().any()
    ):
        group_value = bool(joined.groupby(group)["__partition__"].nunique().max() <= 1)
    results["group_disjoint"] = {
        "value": group_value,
        "applicable": bool(group),
        "evidence": evidence,
    }
    time_columns = expected.get("time_columns", [])
    time_value = True if not time_columns else None
    if time_columns and joined is not None and set(time_columns) <= set(joined):
        times = joined[time_columns].copy()
        try:
            for column in times:
                if not pd.api.types.is_numeric_dtype(times[column]):
                    times[column] = pd.to_datetime(
                        times[column], errors="raise", utc=True
                    )
            if not times.isna().any().any():
                keys = pd.Series(
                    list(times.itertuples(index=False, name=None)), index=joined.index
                )
                bounds = [
                    keys[joined.__partition__ == part]
                    for part in ["train", "val", "test"]
                ]
                if all(len(b) for b in bounds):
                    time_value = bool(
                        bounds[0].max() < bounds[1].min()
                        and bounds[1].max() < bounds[2].min()
                    )
        except (ValueError, TypeError):
            pass
    results["time_ordered"] = {
        "value": time_value,
        "applicable": bool(time_columns),
        "evidence": evidence,
    }
    values = [r["value"] for r in results.values()]
    clean = False if False in values else None if None in values else True
    return {
        "producer": "external_leakage_scorer",
        "agent_work": False,
        "criteria": results,
        "leakage_clean": clean,
    }


def score(project, expected):
    from mlca.log.replay import replay
    from mlca.log.store import Store
    from mlca.paths import sha256
    from mlca.project.inventory import read_table

    if expected["role"] not in {"development", "heldout", "evaluation"}:
        raise ValueError("Unknown table role")
    project = Path(project)
    store = Store(project)
    store.verify()
    state = replay(store.read())
    fits = [
        fit
        for step in state.steps
        if step["status"] == "completed"
        for fit in step.get("fits", [])
    ]
    evidence = [".mlca/log/events.jsonl"]
    frame = membership = None
    if state.split:
        population = state.prepared_data or state.data[state.primary_data]
        path = Path(population.get("original_path", population.get("path", "")))
        if not path.is_absolute():
            path = project / path
        split_path = project / state.split["manifest"]["path"]
        if (
            path.is_file()
            and sha256(path) == population["sha256"]
            and sha256(split_path) == state.split["manifest"]["sha256"]
        ):
            frame = read_table(path)
            membership = pd.read_csv(split_path, dtype={"row_id": str})
            evidence += [str(path), str(split_path)]
    return evaluate(fits, frame, membership, expected, evidence)
