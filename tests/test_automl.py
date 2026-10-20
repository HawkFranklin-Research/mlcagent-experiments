"""Synthetic integrity and adapter-contract tests: no framework fitting or network."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlca.log.store import Store
from sklearn.metrics import mean_squared_error, roc_auc_score

from automl import common, export_split, make_specs, run_automl
from automl import summarize as summarize_module
from automl.common import digest, label, write_json
from automl.score import score
from automl.summarize import summarize


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "attempt-001/run-001"
    directory = root / ".mlca/data/splits/split-1"
    directory.mkdir(parents=True)
    frame = pd.DataFrame(
        {"x": np.arange(300), "target": ["a", "b"] * 150, "id": np.arange(300)}
    )
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    membership = pd.DataFrame(
        {
            "row_id": frame.index,
            "partition": ["train"] * 180 + ["val"] * 60 + ["test"] * 60,
        }
    )
    manifest = directory / "split_manifest.csv"
    membership.to_csv(manifest, index=False)
    parts = {
        "train": frame.iloc[:180],
        "val": frame.iloc[180:240],
        "test": frame.iloc[240:],
    }
    descriptors = {}
    for name in ["train", "val"]:
        path = directory / f"{name}.parquet"
        parts[name].to_parquet(path, index=True)
        descriptors[name] = {
            "path": str(path.relative_to(root)),
            "sha256": digest(path),
        }
    buffer = io.BytesIO()
    parts["test"].to_parquet(buffer, index=True)
    payload = {
        "split_id": "split-1",
        "primary_data": "source",
        "sha256": digest(manifest),
        "manifest": {
            "path": str(manifest.relative_to(root)),
            "sha256": digest(manifest),
        },
        "train": descriptors["train"],
        "val": descriptors["val"],
        "vault": {},
        "test_plain_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "train_index_sha256": hashlib.sha256(
            json.dumps(
                list(parts["train"].index), default=str, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "parameters": {"method": "synthetic", "seed": 42},
        "counts": {"train": 180, "val": 60, "test": 60},
    }
    store = Store(root)
    store.append(
        "ProjectCreated",
        {
            "project_id": "synthetic",
            "mlca_version": "test",
            "template_id": "test",
            "template_version": "1",
            "template_sha256": "x",
        },
    )
    store.append(
        "DataRegistered",
        {
            "name": "source",
            "original_path": str(source),
            "sha256": digest(source),
            "size": source.stat().st_size,
            "read_only": True,
        },
    )
    store.append(
        "TaskConfigured", {"task": {"primary_metric": "roc_auc", "autonomous": True}}
    )
    store.append("SplitFrozen", payload)
    spec = {
        "id": "synthetic",
        "data": str(source),
        "sha256": digest(source),
        "problem_type": "binary",
        "targets": ["target"],
        "excluded": ["id"],
        "external": None,
    }
    return root, spec, tmp_path / "work", store, manifest


def test_export_real_hash_chain_and_partitions(project):
    root, spec, work, store, _manifest = project
    result = export_split.export(spec, root, work, "synthetic-001/attempt-001")
    assert result["status"] == "verified", result
    assert result["counts"] == {"train": 180, "val": 60, "test": 60}
    assert pd.read_parquet(work / "test.parquet").index.tolist() == list(
        range(240, 300)
    )
    assert store.verify()["ok"]


@pytest.mark.parametrize(
    "corruption",
    [
        "manifest",
        "data",
        "overlap",
        "prepared",
        "log",
        "train",
        "train_index",
        "counts",
        "test_hash",
    ],
)
def test_export_refuses_corruption(project, monkeypatch, corruption):
    root, spec, work, _store, manifest = project
    if corruption == "manifest":
        manifest.write_text(manifest.read_text() + "\n")
    elif corruption == "data":
        Path(spec["data"]).write_text("changed\n")
    elif corruption == "log":
        path = root / ".mlca/log/events.jsonl"
        path.write_bytes(path.read_bytes().replace(b"synthetic", b"tampered", 1))
    elif corruption == "train":
        (manifest.parent / "train.parquet").write_bytes(b"bad")
    else:
        events, verification = export_split.read_events(root)
        split = events[-1]["payload"]
        if corruption == "overlap":
            rows = pd.read_csv(manifest)
            rows.loc[200, "row_id"] = 0
            rows.to_csv(manifest, index=False)
            split["sha256"] = split["manifest"]["sha256"] = digest(manifest)
        elif corruption == "prepared":
            events.insert(-1, {"type": "DataPrepared", "payload": {}})
        elif corruption == "train_index":
            split["train_index_sha256"] = "bad"
        elif corruption == "test_hash":
            split["test_plain_sha256"] = "bad"
        elif corruption == "counts":
            split["counts"]["train"] -= 1
        monkeypatch.setattr(
            export_split, "read_events", lambda _: (events, verification)
        )
    result = export_split.export(spec, root, work, "run")
    assert result["status"] == "failed"
    assert not (work / "test.parquet").exists()


def test_no_split_is_unavailable(project, monkeypatch):
    root, spec, work, _, _ = project
    events, verification = export_split.read_events(root)
    monkeypatch.setattr(
        export_split, "read_events", lambda _: (events[:-1], verification)
    )
    assert export_split.export(spec, root, work, "run")["status"] == "no_mlca_split"


def test_github_manifest_fallback_checks_reconstructed_bytes(project):
    root, spec, work, _, manifest = project
    (root.parent / "split_manifest-run-001.csv").write_bytes(manifest.read_bytes())
    import shutil

    shutil.rmtree(root / ".mlca/data")
    result = export_split.export(spec, root, work, "run")
    assert result["status"] == "verified", result
    assert result["verification_modes"]["train"] == "reconstructed_parquet_hash"


def test_first_unlock_pins_split(project, monkeypatch):
    root, spec, work, _, _ = project
    events, verification = export_split.read_events(root)
    events += [
        {"type": "TestUnlocked", "payload": {}},
        {"type": "SplitFrozen", "payload": {}},
    ]
    monkeypatch.setattr(export_split, "read_events", lambda _: (events, verification))
    assert export_split.export(spec, root, work, "run")["status"] == "verified"


def test_binary_score_positive_and_class_alignment():
    truth = ["b", "a", "b", "a"]
    probabilities = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    result = score("binary", "roc_auc", truth, truth, probabilities, ["b", "a"])
    assert result["primary"]["value"] == roc_auc_score(
        np.array(truth) == "b", probabilities[:, 0]
    )
    assert result["secondary"]["accuracy"] == 1
    assert (
        score("binary", "roc_auc", ["a"] * 2, proba=[0.2, 0.3], classes=["a", "b"])[
            "primary"
        ]["value"]
        is None
    )


@pytest.mark.parametrize("absent", [False, True])
def test_multiclass_auc(absent):
    truth = np.array([0, 1, 0, 1] if absent else [0, 1, 2, 1])
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.1, 0.8, 0.1]]
    )
    result = score("multiclass", "roc_auc_ovr", truth, proba=proba, classes=[0, 1, 2])
    expected = (
        np.mean([roc_auc_score(truth == c, proba[:, c]) for c in [0, 1]])
        if absent
        else roc_auc_score(
            truth, proba, labels=[0, 1, 2], multi_class="ovr", average="macro"
        )
    )
    assert result["primary"]["value"] == pytest.approx(expected)
    assert bool(result["notes"]) == absent


def test_regression_scoring_and_unsupported_metric():
    result = score("regression", "rmse", [1.0, 2.0, 3.0], [1.1, 2.2, 3.3])
    assert result["primary"]["value"] == pytest.approx(
        np.sqrt(mean_squared_error([1, 2, 3], [1.1, 2.2, 3.3]))
    )
    assert set(result["secondary"]) == {"mae", "r2"}
    with pytest.raises(ValueError, match="Unsupported"):
        score("regression", "made_up", [1], [1])


def test_one_hot_labels_and_invalid_rows():
    spec = {"targets": ["a", "b", "c"], "label_from_one_hot": True}
    frame = pd.DataFrame(np.eye(3)[np.arange(300) % 3], columns=spec["targets"])
    assert label(frame, spec).tolist()[:3] == ["a", "b", "c"]
    frame.loc[0, "b"] = 1
    with pytest.raises(ValueError, match="exactly one"):
        label(frame, spec)


def test_specs_use_metadata_only(tmp_path, monkeypatch):
    manifest = (
        Path(__file__).resolve().parents[2] / "mlca-revized/benchmarks/manifest.yaml"
    )
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path == manifest or path.is_relative_to(tmp_path), (
            "Dataset files must not be opened"
        )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    groups = make_specs.generate(manifest, tmp_path / "specs")
    assert {s["id"] for s in groups["heldout"]} == {
        "bank_marketing",
        "steel_plates_faults",
        "wine_quality",
    }
    assert next(s for s in groups["heldout"] if s["id"] == "steel_plates_faults")[
        "label_from_one_hot"
    ]
    assert all(
        s["problem_type"] in {"binary", "multiclass", "regression"}
        for s in groups["development"]
    )


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append(frame.index.tolist())
        return np.array(
            [[0.7, 0.3] if index % 2 == 0 else [0.2, 0.8] for index in frame.index]
        )


def test_mock_runner_never_refits_or_repeats_test_predictions(
    project, monkeypatch, tmp_path
):
    root, spec, work, _, _ = project
    export_split.export(spec, root, work, "synthetic-001/attempt-001")
    model = FakePredictor()

    def fit(framework, kind, metric, X, y, X_val, y_val, *args):
        assert X.index.tolist() == list(range(180))
        assert X_val.index.tolist() == list(range(180, 240))
        assert list(X) == ["x"]
        return model, ["a", "b"], {"best_model": "fake", "settings": {}}, None

    monkeypatch.setattr(run_automl, "fit_framework", fit)
    output = tmp_path / "public/synthetic/synthetic-001/attempt-001/flaml"
    result = run_automl.run(spec, work, "flaml", 20, 2, 42, output)
    assert result["status"] == "completed", result
    assert model.calls == [list(range(180, 240)), list(range(240, 300))]
    assert set(pd.read_parquet(output / "test_predictions.parquet")) == {
        "row_id",
        "label",
        "prediction",
        "proba_0",
        "proba_1",
    }
    run_automl.run(spec, work, "flaml", 20, 2, 42, output)
    assert len(model.calls) == 2
    assert summarize(tmp_path / "public")[0]["status"] == "completed"


def test_failure_retained_and_models_cleaned(project, monkeypatch, tmp_path):
    root, spec, work, _, _ = project
    export_split.export(spec, root, work, "run")

    def fail(*args):
        (work / "model-flaml").mkdir()
        raise RuntimeError("first line\nprivate traceback")

    monkeypatch.setattr(run_automl, "fit_framework", fail)
    result = run_automl.run(spec, work, "flaml", 20, 2, 42, tmp_path / "result")
    assert result["status"] == "failed" and result["message"] == "first line"
    assert not (work / "model-flaml").exists()
    assert run_automl.run(spec, work, "flaml", 20, 2, 42, tmp_path / "result") == result


def test_metric_mappings():
    assert run_automl.METRICS["autogluon"] == {
        "roc_auc": "roc_auc",
        "roc_auc_ovr": "roc_auc_ovr_macro",
        "rmse": "root_mean_squared_error",
    }
    assert run_automl.METRICS["flaml"]["roc_auc_ovr"] == "roc_auc_ovr"


def test_summary_retains_failed_pairs_and_ambiguous_agent_metrics(tmp_path):
    for framework, status in [("flaml", "failed"), ("autogluon", "completed")]:
        write_json(
            tmp_path / "task/study/attempt-001" / framework / "result.json",
            {
                "task": "task",
                "framework": framework,
                "status": status,
                "test": {"primary": {"name": "rmse", "value": 2.0}},
                "mlca_recorded_metrics": [
                    {"name": "rmse", "value": 3.0},
                    {"name": "rmse", "value": 4.0},
                ],
            },
        )
    rows = summarize(tmp_path)
    assert len(rows) == 2
    assert all(r["difference_automl_minus_mlca"] is None for r in rows)
    assert all(r["mlca_source"] == "ambiguous_recorded" for r in rows)


def test_autogluon_adapter_contract_without_training(monkeypatch, tmp_path):
    import sys
    import types

    calls = {}

    class Generator:
        def fit_transform(self, frame, y):
            calls["generator_train_ids"] = frame.index.tolist()
            return frame

    class Predictor:
        class_labels = ("a", "b")
        model_best = "synthetic"

        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def fit(self, frame, **kwargs):
            calls["train_ids"] = frame.index.tolist()
            calls["fit"] = kwargs

        def leaderboard(self, **kwargs):
            calls["leaderboard"] = kwargs
            return pd.DataFrame({"model": ["synthetic"]})

    monkeypatch.setitem(
        sys.modules,
        "autogluon.features.generators",
        types.SimpleNamespace(AutoMLPipelineFeatureGenerator=Generator),
    )
    monkeypatch.setitem(
        sys.modules,
        "autogluon.tabular",
        types.SimpleNamespace(TabularPredictor=Predictor),
    )
    frame = pd.DataFrame({"x": range(10)})
    y = pd.Series(["a", "b"] * 5)
    run_automl.fit_framework(
        "autogluon",
        "binary",
        "roc_auc",
        frame.iloc[:6],
        y.iloc[:6],
        frame.iloc[6:],
        y.iloc[6:],
        tmp_path / "model",
        20,
        2,
        42,
    )
    assert calls["generator_train_ids"] == calls["train_ids"] == list(range(6))
    settings = calls["fit"]
    assert settings["tuning_data"].index.tolist() == list(range(6, 10))
    assert settings["refit_full"] is False and settings["num_bag_folds"] == 0
    assert (
        settings["num_stack_levels"] == 0 and settings["fit_weighted_ensemble"] is False
    )
    assert settings["excluded_model_types"] == ["NN_TORCH", "FASTAI"]
    assert "hyperparameters" not in settings
    assert calls["init"]["learner_kwargs"]["random_state"] == 42
    assert calls["leaderboard"] == {"silent": True}


def test_flaml_adapter_contract_without_training(monkeypatch, tmp_path):
    import sys
    import types

    calls = {}

    class AutoML:
        classes_ = ("a", "b")
        best_estimator = "fake"
        best_config = None
        best_loss = 0.25

        def fit(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "flaml", types.SimpleNamespace(AutoML=AutoML))
    frame = pd.DataFrame({"x": range(10)})
    y = pd.Series(["a", "b"] * 5)
    run_automl.fit_framework(
        "flaml",
        "binary",
        "roc_auc",
        frame.iloc[:6],
        y.iloc[:6],
        frame.iloc[6:],
        y.iloc[6:],
        tmp_path / "model",
        20,
        2,
        42,
    )
    assert calls["X_train"].index.tolist() == list(range(6))
    assert calls["X_val"].index.tolist() == list(range(6, 10))
    assert calls["retrain_full"] is False and calls["eval_method"] == "holdout"
    assert calls["time_budget"] == 20 and calls["n_jobs"] == 2 and calls["seed"] == 42


def test_runner_partition_hash_failure_does_not_fit(project, monkeypatch, tmp_path):
    root, spec, work, _, _ = project
    export_split.export(spec, root, work, "run")
    (work / "test.parquet").write_bytes(b"tampered")
    monkeypatch.setattr(
        run_automl, "fit_framework", lambda *_: pytest.fail("Must not fit")
    )
    result = run_automl.run(spec, work, "flaml", 20, 2, 42, tmp_path / "result")
    assert result["status"] == "failed" and "SHA-256" in result["message"]


def test_shared_rescoring_of_explicit_mlca_prediction_schema(tmp_path):
    work = tmp_path
    frame = pd.DataFrame(
        {
            "row_id": [2, 1],
            "prediction": ["b", "a"],
            "proba_0": [0.2, 0.8],
            "proba_1": [0.8, 0.2],
        }
    )
    frame.to_parquet(work / "predictions.parquet", index=False)
    spec = {
        "problem_type": "binary",
        "mlca_prediction_schema": {
            "prediction_column": "prediction",
            "probability_columns": ["proba_0", "proba_1"],
            "classes": ["a", "b"],
        },
    }
    metadata = {
        "primary_metric": "roc_auc",
        "mlca_predictions": {
            "status": "copied",
            "file": "predictions.parquet",
            "sha256": digest(work / "predictions.parquet"),
        },
    }
    result = run_automl.rescore_mlca(
        work, metadata, spec, pd.Series(["a", "b"], index=[1, 2]), "roc_auc"
    )
    assert (
        result["source"] == "shared_scorer" and result["score"]["primary"]["value"] == 1
    )


@pytest.mark.parametrize("kind", ["multiclass_one_hot", "regression"])
def test_export_other_supported_synthetic_tasks(project, monkeypatch, kind):
    root, spec, work, _, manifest = project
    events, verification = export_split.read_events(root)
    frame = pd.DataFrame({"x": np.arange(300), "id": np.arange(300)})
    if kind == "multiclass_one_hot":
        for i, name in enumerate(["a", "b", "c"]):
            frame[name] = (np.arange(300) % 3 == i).astype(int)
        spec.update(
            problem_type="multiclass", targets=["a", "b", "c"], label_from_one_hot=True
        )
        metric = "roc_auc_ovr"
    else:
        frame["target"] = np.arange(300) * 0.17
        spec.update(problem_type="regression")
        metric = "rmse"
    frame.to_csv(spec["data"], index=False)
    # Use MLCA's reader just as registration does (float parsing included).
    from mlca.project.inventory import read_table

    frame = read_table(Path(spec["data"]))
    spec["sha256"] = digest(spec["data"])
    events[1]["payload"]["sha256"] = spec["sha256"]
    events[2]["payload"]["task"]["primary_metric"] = metric
    split = events[-1]["payload"]
    for name, positions in [
        ("train", slice(0, 180)),
        ("val", slice(180, 240)),
        ("test", slice(240, 300)),
    ]:
        buffer = io.BytesIO()
        frame.iloc[positions].to_parquet(buffer, index=True)
        sha = hashlib.sha256(buffer.getvalue()).hexdigest()
        if name == "test":
            split["test_plain_sha256"] = sha
        else:
            (manifest.parent / f"{name}.parquet").write_bytes(buffer.getvalue())
            split[name]["sha256"] = sha
    monkeypatch.setattr(export_split, "read_events", lambda _: (events, verification))
    result = export_split.export(spec, root, work, "run")
    assert result["status"] == "verified", result
    assert result["primary_metric"] == metric
    assert len(label(pd.read_parquet(work / "train.parquet"), spec)) == 180


def test_batch_keeps_export_failures_and_resume_does_not_repeat(tmp_path, monkeypatch):
    import argparse
    import types

    import yaml

    study = tmp_path / "runs/studies/synthetic-001/attempt-001/run-001"
    study.mkdir(parents=True)
    specs = tmp_path / "spec.yaml"
    specs.write_text(
        yaml.safe_dump(
            [{"id": "synthetic", "problem_type": "binary", "targets": ["target"]}]
        )
    )
    calls = []

    def export(command, **kwargs):
        calls.append(command)
        assert command[0] == "mlca-interpreter"
        work = Path(command[command.index("--work") + 1])
        write_json(
            work / "split.json",
            {"status": "no_mlca_split", "run_label": "synthetic-001/attempt-001"},
        )
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(run_automl.subprocess, "run", export)
    monkeypatch.setattr(
        run_automl,
        "fit_framework",
        lambda *_: pytest.fail("No split: no training permitted"),
    )
    args = argparse.Namespace(
        spec=str(specs),
        runs_root=str(tmp_path / "runs"),
        work_root=str(tmp_path / ".work"),
        out=str(tmp_path / "results"),
        mlca_python="mlca-interpreter",
        time_budget=20,
        cpus=2,
        seed=42,
    )
    run_automl.batch(args)
    run_automl.batch(args)
    assert len(calls) == 1
    rows = summarize(tmp_path / "results")
    assert len(rows) == 2 and all(r["status"] == "unavailable" for r in rows)


def test_external_requires_unlock(project):
    root, spec, work, _, _ = project
    spec["external"] = {"data": "/must/not/be/opened", "sha256": "x"}
    result = export_split.export(spec, root, work, "run")
    assert result["status"] == "failed" and "remains locked" in result["message"]


@pytest.mark.parametrize("chosen", ["average_precision", "poisson_deviance"])
def test_comparison_uses_fixed_metric_whatever_mlca_chose(
    project, monkeypatch, tmp_path, chosen
):
    root, spec, work, _, _ = project
    export_split.export(spec, root, work, "synthetic-001/attempt-001")
    metadata = json.loads((work / "split.json").read_text())
    metadata["primary_metric"] = chosen
    metadata["mlca_recorded_metrics"] = [
        {"name": chosen, "value": 0.4, "split": "test"},
        {"name": "roc_auc", "value": 0.9, "split": "test"},
    ]
    (work / "split.json").write_text(json.dumps(metadata))
    model = FakePredictor()
    monkeypatch.setattr(
        run_automl,
        "fit_framework",
        lambda *a, **k: (
            model,
            ["a", "b"],
            {"best_model": "fake", "settings": {}},
            None,
        ),
    )
    output = tmp_path / "public/synthetic/synthetic-001/attempt-001/flaml"
    result = run_automl.run(spec, work, "flaml", 20, 2, 42, output)
    assert result["status"] == "completed", result
    assert (
        result["comparison_metric"] == "roc_auc"
        and result["mlca_primary_metric"] == chosen
    )
    assert result["test"]["primary"]["name"] == "roc_auc"
    assert result["mlca"]["metrics"] == [
        {"name": "roc_auc", "value": 0.9, "split": "test"}
    ]
    row = summarize(tmp_path / "public")[0]
    assert row["mlca_test"] == 0.9 and row["mlca_source"] == "recorded"
    assert row["mlca_primary_metric"] == chosen


def test_missing_comparison_metric_on_mlca_side_is_reported_not_guessed():
    result = {
        "comparison_metric": "rmse",
        "mlca_recorded_metrics": [{"name": "poisson_deviance", "value": 1.2}],
    }
    assert summarize_module.comparison(result) == (
        None,
        None,
        "mlca_metric_unavailable",
    )


def test_split_on_mlca_prepared_table_is_rebuilt_from_that_table(tmp_path):
    # MLCA splits its prepared copy (here deduplicated), never the raw CSV.
    root = tmp_path / "attempt-001/run-001"
    raw = pd.DataFrame(
        {"x": np.arange(300), "target": ["a", "b"] * 150, "id": np.arange(300)}
    )
    raw = pd.concat([raw, raw.iloc[:10]], ignore_index=True)
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)
    prepared = raw.drop_duplicates()
    table = root / ".mlca/data/prepared/p.parquet"
    table.parent.mkdir(parents=True)
    prepared.to_parquet(table, index=True)
    directory = root / ".mlca/data/splits/split-1"
    directory.mkdir(parents=True)
    labels = ["train"] * 180 + ["val"] * 60 + ["test"] * 60
    manifest = directory / "split_manifest.csv"
    pd.DataFrame({"row_id": prepared.index, "partition": labels}).to_csv(
        manifest, index=False
    )
    parts = {
        "train": prepared.iloc[:180],
        "val": prepared.iloc[180:240],
        "test": prepared.iloc[240:],
    }
    descriptors = {}
    for name in ["train", "val"]:
        path = directory / f"{name}.parquet"
        parts[name].to_parquet(path, index=True)
        descriptors[name] = {
            "path": str(path.relative_to(root)),
            "sha256": digest(path),
        }
    buffer = io.BytesIO()
    parts["test"].to_parquet(buffer, index=True)
    store = Store(root)
    store.append(
        "ProjectCreated",
        {
            "project_id": "synthetic",
            "mlca_version": "test",
            "template_id": "test",
            "template_version": "1",
            "template_sha256": "x",
        },
    )
    store.append(
        "DataRegistered",
        {
            "name": "source",
            "original_path": str(source),
            "sha256": digest(source),
            "size": source.stat().st_size,
            "read_only": True,
        },
    )
    store.append(
        "DataPrepared",
        {
            "path": str(table.relative_to(root)),
            "sha256": digest(table),
            "source_sha256": digest(source),
            "decision_sha256": "d",
        },
    )
    store.append(
        "TaskConfigured", {"task": {"primary_metric": "roc_auc", "autonomous": True}}
    )
    store.append(
        "SplitFrozen",
        {
            "split_id": "split-1",
            "primary_data": "source",
            "sha256": digest(manifest),
            "manifest": {
                "path": str(manifest.relative_to(root)),
                "sha256": digest(manifest),
            },
            "train": descriptors["train"],
            "val": descriptors["val"],
            "vault": {},
            "test_plain_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
            "train_index_sha256": hashlib.sha256(
                json.dumps(
                    list(parts["train"].index), default=str, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "parameters": {"method": "synthetic", "seed": 42},
            "counts": {"train": 180, "val": 60, "test": 60},
        },
    )
    spec = {
        "id": "synthetic",
        "data": str(source),
        "sha256": digest(source),
        "problem_type": "binary",
        "targets": ["target"],
        "excluded": ["id"],
        "external": None,
    }
    result = export_split.export(
        spec, root, tmp_path / "work", "synthetic-001/attempt-001"
    )
    assert result["status"] == "verified", result
    assert result["table"]["source"] == "mlca_prepared"
    table.unlink()
    missing = export_split.export(
        spec, root, tmp_path / "work2", "synthetic-001/attempt-001"
    )
    assert missing["status"] != "verified"


def test_excluded_columns_dropped_by_mlca_preparation_are_not_an_error():
    frame = pd.DataFrame({"x": [1, 2], "target": ["a", "b"]})
    spec = {"targets": ["target"], "excluded": ["leak", "id"]}
    assert common.feature_columns(frame, spec) == ["x"]
    with pytest.raises(ValueError, match="Target columns absent"):
        common.feature_columns(frame.drop(columns="target"), spec)
