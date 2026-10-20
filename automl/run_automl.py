"""Direct AutoGluon/FLAML comparison; frameworks are imported only when fitting."""

import argparse
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

try:
    from .common import (
        checked,
        disjoint,
        feature_columns,
        fingerprint,
        label,
        task_spec,
        write_json,
    )
    from .score import score
except ImportError:
    from common import (
        checked,
        disjoint,
        feature_columns,
        fingerprint,
        label,
        task_spec,
        write_json,
    )
    from score import score

# Both sides are scored on one fixed metric per task type, whatever MLCA's navigator chose.
COMPARISON_METRIC = {
    "binary": "roc_auc",
    "multiclass": "roc_auc_ovr",
    "regression": "rmse",
}

METRICS = {
    "autogluon": {
        "roc_auc": "roc_auc",
        "roc_auc_ovr": "roc_auc_ovr_macro",
        "rmse": "root_mean_squared_error",
    },
    "flaml": {"roc_auc": "roc_auc", "roc_auc_ovr": "roc_auc_ovr", "rmse": "rmse"},
}


def versions():
    result = {"python": sys.version, "executable": sys.executable}
    for package in [
        "autogluon.tabular",
        "flaml",
        "scikit-learn",
        "numpy",
        "pandas",
        "lightgbm",
    ]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def fit_framework(
    framework, kind, metric, X, y, X_val, y_val, model_dir, seconds, cpus, seed
):
    mapped = METRICS[framework][metric]
    if framework == "autogluon":
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from autogluon.tabular import TabularPredictor

        # AG otherwise learns feature generation jointly on train + tuning_data.
        preprocessing_started = time.monotonic()
        generator = AutoMLPipelineFeatureGenerator()
        generator.fit_transform(X, y=y)
        target = "__automl_label__"
        if target in X:
            raise ValueError("Reserved label column collides with a feature")
        predictor = TabularPredictor(
            label=target,
            problem_type=kind,
            eval_metric=mapped,
            path=str(model_dir),
            learner_kwargs={"random_state": seed},
        )
        settings = {
            "eval_metric": mapped,
            "presets": "medium_quality",
            "num_cpus": cpus,
            "refit_full": False,
            "num_bag_folds": 0,
            "num_stack_levels": 0,
            "fit_weighted_ensemble": False,
            "feature_generator_fit": "training_only",
            "excluded_model_types": ["NN_TORCH", "FASTAI"],
            "seed": seed,
            "determinism": "AutoGluon may not be fully deterministic",
        }
        predictor.fit(
            X.assign(**{target: y}),
            tuning_data=X_val.assign(**{target: y_val}),
            time_limit=max(0.01, seconds - (time.monotonic() - preprocessing_started)),
            presets="medium_quality",
            num_cpus=cpus,
            refit_full=False,
            num_bag_folds=0,
            num_stack_levels=0,
            fit_weighted_ensemble=False,
            feature_generator=generator,
            excluded_model_types=settings["excluded_model_types"],
            ag_args_fit={"num_cpus": cpus, "num_gpus": 0},
        )
        classes = list(predictor.class_labels) if kind != "regression" else None
        # Cached held-out leaderboard avoids a second validation prediction pass.
        details = {"best_model": predictor.model_best, "settings": settings}
        return predictor, classes, details, predictor.leaderboard(silent=True)
    if framework != "flaml":
        raise ValueError("Unknown framework")
    from flaml import AutoML

    predictor = AutoML()
    settings = {
        "task": "regression" if kind == "regression" else "classification",
        "metric": mapped,
        "time_budget": seconds,
        "n_jobs": cpus,
        "seed": seed,
        "eval_method": "holdout",
        "retrain_full": False,
        "estimator_list": "auto",
    }
    predictor.fit(X_train=X, y_train=y, X_val=X_val, y_val=y_val, **settings)
    classes = list(predictor.classes_) if kind != "regression" else None
    settings["resolved_estimator_list"] = getattr(predictor, "estimator_list", None)
    details = {
        "best_model": predictor.best_estimator,
        "best_config": predictor.best_config,
        "validation_loss": float(predictor.best_loss),
        "settings": settings,
    }
    return predictor, classes, details, None


def predict_once(model, frame, kind, classes):
    if kind == "regression":
        return np.asarray(model.predict(frame)), None
    raw = model.predict_proba(frame)
    if isinstance(raw, pd.DataFrame):
        if set(raw.columns) != set(classes):
            raise ValueError("Probability class columns differ from fitted classes")
        probabilities = raw[classes].to_numpy()
    else:
        probabilities = np.asarray(raw)
    if probabilities.shape != (len(frame), len(classes)):
        raise ValueError("Framework probability shape mismatch")
    # Fixed argmax decision, not a threshold tuned using test labels.
    return np.asarray(classes)[probabilities.argmax(axis=1)], probabilities


def prediction_table(frame, truth, prediction, probabilities):
    rows = pd.DataFrame(
        {"row_id": frame.index, "label": np.asarray(truth), "prediction": prediction}
    )
    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            rows[f"proba_{i}"] = probabilities[:, i]
    return rows


def rescore_mlca(work, metadata, spec, truth, metric):
    prediction = metadata.get("mlca_predictions", {})
    schema = spec.get("mlca_prediction_schema") or prediction.get("schema")
    if not schema or prediction.get("status") != "copied":
        recorded = [
            m for m in metadata.get("mlca_recorded_metrics", []) if m["name"] == metric
        ]
        return {
            "source": "recorded" if recorded else "mlca_metric_unavailable",
            "metrics": recorded,
            "reason": "No explicit saved-prediction schema; class ordering is never guessed",
        }
    rows = pd.read_parquet(checked(work / prediction["file"], prediction["sha256"]))
    expected = {str(v): v for v in truth.index}
    if (
        rows.row_id.isna().any()
        or rows.row_id.duplicated().any()
        or set(rows.row_id.map(str)) != set(expected)
    ):
        raise ValueError("MLCA prediction rows do not equal frozen test membership")
    rows.index = [expected[str(v)] for v in rows.row_id]
    rows = rows.loc[truth.index]
    proba_columns = schema.get("probability_columns")
    probabilities = rows[proba_columns].to_numpy() if proba_columns else None
    predicted = (
        rows[schema["prediction_column"]] if schema.get("prediction_column") else None
    )
    if schema.get("standard") and spec["problem_type"] != "regression":
        # Standard-schema class names come from column names, so labels are compared as text.
        truth = truth.astype(str)
        predicted = predicted.astype(str) if predicted is not None else None
    positive = spec.get("positive")
    if schema.get("standard") and positive is not None:
        positive = str(positive)
    result = score(
        spec["problem_type"],
        metric,
        truth,
        y_pred=predicted,
        proba=probabilities,
        classes=schema.get("classes"),
        positive=positive,
    )
    return {"source": "shared_scorer", "score": result}


def run(spec, work, framework, seconds, cpus, seed, output):
    work, output = Path(work), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / "result.json"
    if result_file.exists():
        return json.loads(result_file.read_text())
    result = {
        "status": "started",
        "task": spec["id"],
        "framework": framework,
        "spec_sha256": fingerprint(spec),
        "time_budget_s": seconds,
        "cpus": cpus,
        "seed": seed,
        "versions": versions(),
    }
    # Existing started results are retained after interruption, never silently retried.
    write_json(result_file, result)
    model_dir = work / f"model-{framework}"
    try:
        if seconds <= 0 or cpus <= 0:
            raise ValueError("Time budget and CPUs must be positive")
        metadata = json.loads((work / "split.json").read_text())
        result.update(
            run_label=metadata["run_label"],
            split_hashes=metadata.get("hashes"),
            mlca_recorded_metrics=metadata.get("mlca_recorded_metrics", []),
            mlca_code_commit=metadata.get("mlca_code_commit"),
            rails=metadata.get("rails"),
            split_id=metadata.get("split_id"),
            split_parameters=metadata.get("parameters"),
            mlca_log_verification=metadata.get("log_verification"),
        )
        if metadata["status"] != "verified":
            result.update(
                status="unavailable",
                reason=metadata["status"],
                export_error=metadata.get("message"),
            )
            write_json(result_file, result)
            return result
        if metadata["spec_sha256"] != fingerprint(spec):
            raise ValueError("Spec changed after export")
        parts = {
            name: pd.read_parquet(
                checked(
                    work / metadata["partitions"][name]["file"],
                    metadata["partitions"][name]["sha256"],
                )
            )
            for name in ["train", "val", "test"]
        }
        disjoint(parts.values())
        if {name: len(frame) for name, frame in parts.items()} != metadata["counts"]:
            raise ValueError("Partition counts changed after export")
        columns = feature_columns(parts["train"], spec)
        if any(
            list(frame.columns) != list(parts["train"].columns)
            for frame in parts.values()
        ):
            raise ValueError("Partition columns differ")
        labels = {name: label(frame, spec) for name, frame in parts.items()}
        kind = spec["problem_type"]
        metric = COMPARISON_METRIC[kind]
        result.update(
            comparison_metric=metric, mlca_primary_metric=metadata["primary_metric"]
        )
        result["features"], result["feature_count"] = columns, len(columns)
        result["excluded_absent"] = sorted(
            set(spec.get("excluded", [])) - set(parts["train"].columns)
        )
        random.seed(seed)
        np.random.seed(seed)
        for variable in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            os.environ[variable] = str(cpus)
        started = time.monotonic()
        with threadpool_limits(limits=cpus):
            model, classes, details, leaderboard = fit_framework(
                framework,
                kind,
                metric,
                parts["train"][columns],
                labels["train"],
                parts["val"][columns],
                labels["val"],
                model_dir,
                seconds,
                cpus,
                seed,
            )
        result.update(
            details,
            fit_seconds=time.monotonic() - started,
            classes=classes,
            positive=(
                spec.get("positive", sorted(classes)[1]) if kind == "binary" else None
            ),
        )
        if leaderboard is not None:
            leaderboard.to_csv(output / "leaderboard.csv", index=False)
        result["predict_seconds"] = {}
        for partition in ["val", "test"]:
            started = time.monotonic()
            with threadpool_limits(limits=cpus):
                prediction, probabilities = predict_once(
                    model, parts[partition][columns], kind, classes
                )
            result["predict_seconds"][partition] = time.monotonic() - started
            result[partition] = score(
                kind,
                metric,
                labels[partition],
                prediction,
                probabilities,
                classes,
                spec.get("positive"),
            )
            if partition == "test":
                prediction_table(
                    parts[partition], labels[partition], prediction, probabilities
                ).to_parquet(output / "test_predictions.parquet", index=False)
        result["mlca"] = rescore_mlca(work, metadata, spec, labels["test"], metric)
        if metadata.get("external"):
            descriptor = metadata["external"]
            cohort = pd.read_parquet(
                checked(work / descriptor["file"], descriptor["sha256"])
            )
            started = time.monotonic()
            with threadpool_limits(limits=cpus):
                prediction, probabilities = predict_once(
                    model, cohort[columns], kind, classes
                )
            result["predict_seconds"]["external"] = time.monotonic() - started
            truth = label(cohort, spec)
            result["external"] = score(
                kind,
                metric,
                truth,
                prediction,
                probabilities,
                classes,
                spec.get("positive"),
            )
            prediction_table(cohort, truth, prediction, probabilities).to_parquet(
                output / "external_predictions.parquet", index=False
            )
        result["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 -- keep failed comparisons as research evidence
        result.update(
            status="failed",
            error_type=type(exc).__name__,
            message=str(exc).splitlines()[0] if str(exc) else "",
        )
    finally:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        write_json(result_file, result)
    return result


def batch(args):
    import yaml

    specs = yaml.safe_load(Path(args.spec).read_text())
    for spec in specs:
        # Bound discovery to the public study layout, never recurse into environments.
        projects = sorted(
            Path(args.runs_root).glob(
                f"studies/{spec['id']}-[0-9][0-9][0-9]/attempt-*/run-001"
            )
        )
        for project in projects:
            run_label = f"{project.parent.parent.name}/{project.parent.name}"
            work = Path(args.work_root) / spec["id"] / run_label
            destinations = [
                Path(args.out) / spec["id"] / run_label / name for name in METRICS
            ]
            if all(
                (destination / "result.json").exists() for destination in destinations
            ):
                continue
            work.mkdir(parents=True, exist_ok=True)
            if not (work / "split.json").exists():
                command = [
                    args.mlca_python,
                    str(Path(__file__).with_name("export_split.py")),
                    "--spec",
                    str(Path(args.spec).resolve()),
                    "--task",
                    spec["id"],
                    "--project",
                    str(project.resolve()),
                    "--work",
                    str(work.resolve()),
                    "--run-label",
                    run_label,
                ]
                completed = subprocess.run(
                    command, check=False, capture_output=True, text=True
                )
                if completed.returncode or not (work / "split.json").exists():
                    write_json(
                        work / "split.json",
                        {
                            "status": "failed",
                            "run_label": run_label,
                            "message": "Export subprocess failed; inspect export.stderr.txt",
                        },
                    )
                    (work / "export.stderr.txt").write_text(completed.stderr)
            for framework in METRICS:
                run(
                    spec,
                    work,
                    framework,
                    args.time_budget,
                    args.cpus,
                    args.seed,
                    Path(args.out) / spec["id"] / run_label / framework,
                )
    try:
        from .summarize import summarize
    except ImportError:
        from summarize import summarize
    summarize(Path(args.out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--task")
    parser.add_argument("--work")
    parser.add_argument("--framework", choices=list(METRICS))
    parser.add_argument("--time-budget", type=float, required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--runs-root")
    parser.add_argument("--mlca-python")
    parser.add_argument("--work-root")
    args = parser.parse_args()
    if args.batch:
        if not all([args.runs_root, args.mlca_python, args.work_root]):
            parser.error("Batch requires --runs-root --mlca-python --work-root")
        batch(args)
    else:
        if not all([args.task, args.work, args.framework]):
            parser.error("Single run requires --task --work --framework")
        run(
            task_spec(args.spec, args.task),
            args.work,
            args.framework,
            args.time_budget,
            args.cpus,
            args.seed,
            args.out,
        )


if __name__ == "__main__":
    main()
