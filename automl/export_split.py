"""Export verified MLCA partitions, using MLCA's interpreter and table reader."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

try:
    from .common import (
        checked,
        data_path,
        digest,
        disjoint,
        fingerprint,
        project_path,
        task_spec,
        write_json,
    )
except ImportError:
    from common import (
        checked,
        data_path,
        digest,
        disjoint,
        fingerprint,
        project_path,
        task_spec,
        write_json,
    )


def read_events(project):
    from mlca.log.store import Store

    if not (project / ".mlca/log/events.jsonl").is_file():
        raise ValueError("MLCA event log is missing")
    store = Store(project)
    verification = store.verify()
    return [
        {
            "type": r.type.value,
            "payload": r.payload,
            "branch": r.branch,
            "step_id": r.step_id,
            "id": r.id,
            "parent_id": r.parent_id,
        }
        for r in store.read()
    ], verification


def select_events(events):
    # A comparison must not accidentally combine different branch histories.
    unlock = next(
        (i for i, r in enumerate(events) if r["type"] == "TestUnlocked"), len(events)
    )
    prefix = events[:unlock]
    splits = [r for r in prefix if r["type"] == "SplitFrozen"]
    if not splits:
        return [], None
    chosen = splits[-1]
    if len({r.get("branch", "main") for r in events}) > 1:
        raise ValueError(
            "Branched project requires an explicitly selected history; export unavailable"
        )
    prefix = prefix[: prefix.index(chosen) + 1]
    return prefix, chosen["payload"]


def verified_frame(frame, descriptor, project):
    path = project_path(project, descriptor["path"])
    if path.is_file():
        checked(path, descriptor["sha256"])
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
        return "stored_parquet"
    # Git exports omit .mlca/data. Reproduce the *bytes*, not merely the row count.
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=True)
    if hashlib.sha256(buffer.getvalue()).hexdigest() != descriptor["sha256"]:
        raise ValueError(
            "Missing stored partition; reconstructed parquet hash differs (use original MLCA environment)"
        )
    return "reconstructed_parquet_hash"


def extract_predictions(events, project, output, spec, test_index):
    starts = {
        r["payload"]["step_id"]: r["payload"]
        for r in events
        if r["type"] == "StepStarted"
    }
    candidates = []
    for record in events:
        if record["type"] != "StepCompleted":
            continue
        payload = record["payload"]
        start = starts.get(payload.get("step_id", record.get("step_id")), {})
        # score_test saves them in MLCA's standard schema; audit_test is the older optional action.
        if start.get("action_id") not in {"audit_test", "score_test"}:
            continue
        for artifact in payload.get("outputs", []):
            if artifact.get("name") == "test_predictions.parquet":
                candidates.append(artifact)
    if not candidates:
        return {
            "status": "unavailable",
            "reason": "No completed test predictions",
        }
    artifact = candidates[-1]
    path = checked(project_path(project, artifact["path"]), artifact["sha256"])
    rows = pd.read_parquet(path)
    if "row_id" not in rows or rows.row_id.duplicated().any():
        raise ValueError("Invalid MLCA prediction row identifiers")
    if rows.row_id.isna().any() or set(rows.row_id.map(str)) != {
        str(v) for v in test_index
    }:
        raise ValueError("MLCA predictions do not cover exactly the frozen test rows")
    if "partition" in rows and not rows["partition"].eq("test").all():
        raise ValueError("MLCA test predictions contain a non-test partition")
    # Only explicitly named fields: the task spec's schema, or MLCA's standard one (y_true, y_pred,
    # proba__<class> in the model's class order), whose class names are in the column names.
    schema = spec.get("mlca_prediction_schema", {})
    standard = None
    single = [c for c in rows.columns if c.startswith("y_pred__")]
    prediction_column = (
        "y_pred" if "y_pred" in rows else single[0] if len(single) == 1 else None
    )
    if not schema and prediction_column:
        probabilities = [c for c in rows.columns if c.startswith("proba__")]
        standard = {
            "prediction_column": prediction_column,
            "probability_columns": probabilities,
            "classes": [c[len("proba__") :] for c in probabilities],
            "standard": True,
        }
        schema = standard
    requested = schema.get("probability_columns", [])
    requested = [requested] if isinstance(requested, str) else list(requested)
    if schema.get("prediction_column"):
        requested.append(schema["prediction_column"])
    if not set(requested) <= set(rows.columns):
        raise ValueError("Explicit MLCA prediction schema columns are missing")
    columns = ["row_id"] + [
        c
        for c in rows
        if c != "row_id"
        and (
            c in requested
            or c in {"prediction", "y_pred", "probability", "proba"}
            or c.startswith(("y_pred__", "proba_"))
        )
    ]
    if len(columns) == 1:
        return {
            "status": "unavailable",
            "reason": "Undocumented prediction column schema",
            "artifact_sha256": artifact["sha256"],
        }
    destination = output / "mlca_test_predictions.parquet"
    rows[columns].to_parquet(destination, index=False)
    return {
        "status": "copied",
        "file": destination.name,
        "sha256": digest(destination),
        "columns": columns,
        "source_sha256": artifact["sha256"],
        "schema": standard,
        "scoring": "Task spec schema or MLCA's standard schema; recorded metric retained otherwise",
    }


def export(spec, project, output, run_label):
    from mlca.project.inventory import read_table

    project, output = Path(project).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = {
        "task": spec["id"],
        "run_label": run_label,
        "project": str(project),
        "spec_sha256": fingerprint(spec),
    }
    try:
        events, verification = read_events(project)
        prefix, split = select_events(events)
        base["log_verification"] = verification
        if split is None:
            base.update(status="no_mlca_split")
            write_json(output / "split.json", base)
            return base
        registered = [
            r["payload"]
            for r in prefix
            if r["type"] == "DataRegistered"
            and r["payload"]["name"] == split["primary_data"]
        ]
        if not registered:
            raise ValueError("Split primary_data is not registered")
        registered = registered[-1]
        # MLCA splits its prepared copy of the primary table (deduplicated, cleaned, Parquet),
        # so the partitions are rebuilt from that copy, verified by its recorded hash.
        prepared = [r["payload"] for r in prefix if r["type"] == "DataPrepared"]
        prepared = prepared[-1] if prepared else None
        if (
            prepared is not None
            and prepared.get("source_sha256") != registered["sha256"]
        ):
            raise ValueError("Prepared table does not derive from the registered table")
        configured = [r["payload"] for r in prefix if r["type"] == "TaskConfigured"]
        if not configured:
            raise ValueError("No TaskConfigured before split")
        task = configured[-1]["task"]
        base["primary_metric"] = task["primary_metric"]
        base["mlca_recorded_metrics"] = [
            dict(m, step_id=r["payload"].get("step_id"), source="recorded")
            for r in events
            if r["type"] == "StepCompleted"
            for m in r["payload"].get("metrics", [])
            if m.get("split") == "test"
        ]
        path = checked(data_path(spec["data"]), spec["sha256"])
        if spec["sha256"] not in {
            registered["sha256"],
            registered.get("source_sha256"),
        }:
            raise ValueError("Task data differ from the table MLCA registered")
        if prepared is not None:
            table = project_path(project, prepared["path"])
            if not table.is_file():
                raise ValueError("MLCA prepared table is unavailable")
            frame = read_table(checked(table, prepared["sha256"]))
            base["table"] = {"source": "mlca_prepared", "sha256": prepared["sha256"]}
        else:
            frame = read_table(path)
            base["table"] = {"source": "registered", "sha256": spec["sha256"]}
        if not task.get("autonomous"):
            frame = frame.reset_index(drop=True)
        if not frame.index.is_unique or frame.index.hasnans:
            raise ValueError("Source row identifiers are not unique and complete")
        manifest = project_path(project, split["manifest"]["path"])
        if not manifest.is_file():
            manifest = project.parent / f"split_manifest-{project.name}.csv"
        checked(manifest, split["sha256"])
        checked(manifest, split["manifest"]["sha256"])
        membership = pd.read_csv(manifest, dtype={"row_id": str}, keep_default_na=False)
        if (
            list(membership.columns) != ["row_id", "partition"]
            or membership.row_id.duplicated().any()
        ):
            raise ValueError("Duplicate or invalid manifest row identifiers")
        mapping = {str(index): index for index in frame.index}
        if len(mapping) != len(frame) or set(mapping) != set(membership.row_id):
            raise ValueError("Manifest does not cover exactly the registered rows")
        if set(membership.partition) != {"train", "val", "test"}:
            raise ValueError("Manifest must contain train, val and test only")
        if membership.partition.value_counts().to_dict() != split["counts"]:
            raise ValueError("Partition counts differ from SplitFrozen")
        parts = {
            name: frame.loc[
                [
                    mapping[v]
                    for v in membership.loc[membership.partition == name, "row_id"]
                ]
            ]
            for name in ["train", "val", "test"]
        }
        disjoint(parts.values())
        index_hash = hashlib.sha256(
            json.dumps(
                list(parts["train"].index), default=str, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if index_hash != split["train_index_sha256"]:
            raise ValueError("Training index hash differs")
        verification_modes = {
            name: verified_frame(parts[name], split[name], project)
            for name in ["train", "val"]
        }
        buffer = io.BytesIO()
        parts["test"].to_parquet(buffer, index=True)
        if hashlib.sha256(buffer.getvalue()).hexdigest() != split["test_plain_sha256"]:
            raise ValueError("Reconstructed test parquet hash differs")
        outputs = {}
        for name, partition in parts.items():
            target = output / f"{name}.parquet"
            partition.to_parquet(target, index=True)
            outputs[name] = {"file": target.name, "sha256": digest(target)}
        snapshots = [
            r["payload"]
            for r in events
            if r["type"] in {"ProjectCreated", "RunSnapshot"}
        ]
        batch_file = (
            project.parents[3] / "batch.json" if len(project.parents) > 3 else None
        )
        batch_state = (
            json.loads(batch_file.read_text())
            if batch_file and batch_file.is_file()
            else {}
        )
        base["mlca_code_commit"] = next(
            (
                r.get("git_commit", r.get("code_commit"))
                for r in reversed(snapshots)
                if r.get("git_commit") or r.get("code_commit")
            ),
            batch_state.get("code_commit"),
        )
        base["rails"] = next(
            (r["rails"] for r in reversed(snapshots) if "rails" in r),
            batch_state.get("rails"),
        )
        metrics = [
            dict(m, step_id=r["payload"].get("step_id"), source="recorded")
            for r in events
            if r["type"] == "StepCompleted"
            for m in r["payload"].get("metrics", [])
            if m.get("split") == "test"
        ]
        base.update(
            status="verified",
            split_id=split["split_id"],
            parameters=split["parameters"],
            counts=split["counts"],
            primary_metric=task["primary_metric"],
            hashes={
                "data": spec["sha256"],
                "manifest": split["sha256"],
                "train_index": index_hash,
                "train": split["train"]["sha256"],
                "val": split["val"]["sha256"],
                "test": split["test_plain_sha256"],
            },
            verification_modes=verification_modes,
            partitions=outputs,
            snapshots=snapshots,
            mlca_recorded_metrics=metrics,
            mlca_predictions=extract_predictions(
                events, project, output, spec, parts["test"].index
            ),
        )
        if spec.get("external"):
            if not any(r["type"] == "TestUnlocked" for r in events):
                raise ValueError(
                    "External cohort remains locked: no TestUnlocked event"
                )
            external = spec["external"]
            external_path = checked(data_path(external["data"]), external["sha256"])
            regs = [
                r["payload"]
                for r in events
                if r["type"] == "DataRegistered"
                and r["payload"].get("role") == "external"
                and r["payload"]["sha256"] == external["sha256"]
            ]
            if len(regs) != 1:
                raise ValueError("External file is not uniquely registered as external")
            sealed = [
                r["payload"]
                for r in events
                if r["type"] == "ExternalSealed"
                and r["payload"]["name"] == regs[0]["name"]
            ]
            if not sealed:
                raise ValueError("External cohort sealing evidence missing")
            cohort = read_table(external_path)
            buffer = io.BytesIO()
            cohort.to_parquet(buffer, index=False)
            if (
                hashlib.sha256(buffer.getvalue()).hexdigest()
                != sealed[-1]["plain_sha256"]
            ):
                raise ValueError("External cohort plaintext hash mismatch")
            cohort.to_parquet(output / "external.parquet", index=True)
            base["external"] = {
                "file": "external.parquet",
                "sha256": digest(output / "external.parquet"),
                "source_sha256": external["sha256"],
            }
    except Exception as exc:  # noqa: BLE001 -- fail closed and record export errors
        base.update(
            status="failed",
            error_type=type(exc).__name__,
            message=str(exc).splitlines()[0] if str(exc) else "",
        )
    write_json(output / "split.json", base)
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    export(task_spec(args.spec, args.task), args.project, args.work, args.run_label)
