"""Small analysis-side API. No model clients, credentials or engine access."""

import hashlib
import json
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any


def _step() -> Path:
    return Path(os.environ["MLCA_STEP_DIR"]).resolve()


class InputKeyError(KeyError):
    """A safe contract error; never includes the attempted key or source values."""

    def __str__(self) -> str:
        return str(self.args[0])


class _Inputs(dict):
    def __missing__(self, key):
        raise InputKeyError("MLCA: Use the declared input dictionary keys exactly; filenames are not keys")


def inputs() -> dict[str, Path]:
    return _Inputs(
        {
            name: Path(value["path"])
            for name, value in sorted(
                json.loads((_step() / "inputs.json").read_text(encoding="utf-8")).items(),
                key=lambda item: item[0] == "_protocol",
            )
        }
    )


def output_dir() -> Path:
    path = _step() / "outputs"
    path.mkdir(exist_ok=True)
    return path


def _output(name: str) -> Path:
    path = (output_dir() / name).resolve()
    if not path.is_relative_to(output_dir().resolve()):
        raise ValueError("Output must be inside the step output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_scalar(value: Any) -> Any:
    """Allow scientific scalar values while keeping strict, finite JSON serialization."""
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _record(kind: str, value: dict) -> None:
    path = _step() / "manifest.json"
    data = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"outputs": [], "metrics": [], "facts": [], "fits": []}
    )
    data.setdefault(kind, []).append(value)
    scratch = _step() / "scratch"
    scratch.mkdir(exist_ok=True)
    fd, name = tempfile.mkstemp(dir=scratch)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, allow_nan=False, sort_keys=True, default=_json_scalar)
    os.replace(name, path)


def _artifact(path: Path, kind: str, caption: str = "") -> None:
    _record("outputs", {"name": str(path.relative_to(output_dir())), "kind": kind, "caption": caption})


def save_figure(fig: Any, name: str, caption: str = "") -> Path:
    from .style import apply

    apply()
    path = _output(name)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    svg = path.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    _artifact(path, "figure", caption)
    _artifact(svg, "figure", caption)
    return path


def save_table(df: Any, name: str, caption: str = "") -> Path:
    path = _output(name)
    if path.suffix.lower() not in {".csv", ".parquet", ".json"}:
        path = path.with_suffix(".csv")
    if path.suffix == ".json":
        path.write_text(df.to_json(orient="table"), encoding="utf-8")
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=True)
    else:
        df.to_csv(path, index=False)
    _artifact(path, "table", caption)
    if path.suffix == ".csv":
        parquet = path.with_suffix(".parquet")
        df.to_parquet(parquet, index=True)
        _artifact(parquet, "table", caption)
    return path


def save_model(model: Any, name: str) -> Path:
    path = _output(name)
    with path.open("wb") as stream:
        pickle.dump(model, stream)
    _artifact(path, "model")
    return path


def save_json(value: object, name: str, kind: str = "report") -> Path:
    path = _output(name)
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False, default=_json_scalar), encoding="utf-8"
    )
    _artifact(path, kind)
    return path


def record_metric(name: str, value: float, split: str, ci: Any = None) -> None:
    import math

    if split not in {"train", "val", "test", "external"}:
        raise ValueError(
            "MLCA: Metric partition must be train, val, test or external; "
            "use sourced fact candidates for counts before the split, not event names."
        )
    if not math.isfinite(float(value)):
        raise ValueError("MLCA: Metrics require a finite scalar value")
    _record("metrics", {"name": name, "value": float(value), "split": split, "ci": ci})


def record_fact_candidate(key: str, value: Any, unit: str | None = None, source: str = "") -> None:
    _record("facts", {"key": key, "value": value, "unit": unit, "source": source})


def index_sha256(index: Any) -> str:
    return hashlib.sha256(json.dumps(list(index), default=str, separators=(",", ":")).encode()).hexdigest()


class FitRuleError(ValueError):
    """Frozen-training instrumentation rejected a fit. Messages start with "MLCA:" and carry no data,
    so the repair prompt may quote them."""


def fit_record(estimator_or_pipeline: Any, X: Any, partition: str, y: Any = None, **kwargs: Any) -> Any:
    if partition != "train":
        raise FitRuleError("MLCA: Fit is allowed only on training data")
    if not hasattr(X, "index"):
        # The frozen row index is the evidence that only training rows were fitted; arrays lose it.
        raise FitRuleError(
            "MLCA: fit_record needs the training pandas DataFrame with its frozen row index; "
            "pass X itself, not X.values or a NumPy array (put transforms inside a Pipeline)"
        )
    expected = os.environ.get("MLCA_TRAIN_INDEX_SHA256")
    digest = index_sha256(X.index)
    if not expected or digest != expected:
        raise FitRuleError(
            f"MLCA: Fit rows must match the frozen training index: expected {os.environ.get('MLCA_TRAIN_ROW_COUNT', 'unknown')} rows, received {len(X.index)} rows"
        )
    columns = (
        [str(c) for c in X.columns]
        if hasattr(X, "columns")
        else [str(X.name)]
        if getattr(X, "name", None) is not None
        else None
    )
    protected = set(json.loads(os.environ.get("MLCA_PROTECTED_COLUMNS") or "[]"))
    reached = sorted(protected.intersection(columns or []))
    if reached:
        # Outcomes, excluded columns and the group key are never model inputs; group
        # values reach internal folds through groups=, not through X.
        raise FitRuleError(
            f"MLCA: Target, excluded or group columns cannot be fitted as inputs: {reached}; "
            "select the feature columns only (pass group values with groups=)"
        )
    result = (
        estimator_or_pipeline.fit(X, y, **kwargs) if y is not None else estimator_or_pipeline.fit(X, **kwargs)
    )
    _record(
        "fits",
        {
            "partition": partition,
            "index_sha256": digest,
            "estimator": type(estimator_or_pipeline).__name__,
            "columns": columns,
        },
    )
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def save_row_outputs(frame: Any, name: str) -> Path:
    """Save local per-row results for an external evaluator, never an LLM summary."""
    if not {"row_id", "partition"} <= set(frame.columns):
        raise ValueError("MLCA: row outputs require row_id and partition")
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("MLCA: row identifiers must be unique and nonmissing")
    if not set(frame["partition"]) <= {"train", "val", "test"}:
        raise ValueError("MLCA: row outputs have invalid partitions")
    path = _output(name)
    if path.suffix != ".parquet":
        raise ValueError("MLCA: per-row outputs must use Parquet")
    frame.to_parquet(path, index=False)
    _artifact(path, "data")
    return path
