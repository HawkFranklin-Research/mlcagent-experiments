"""Small, framework-independent provenance and task helpers."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def task_spec(file, task):
    entries = yaml.safe_load(Path(file).read_text())
    found = [entry for entry in entries if entry["id"] == task]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one spec for {task}")
    spec = found[0]
    if spec["problem_type"] not in {"binary", "multiclass", "regression"}:
        raise ValueError("Unsupported task type")
    if not spec.get("targets") or (
        len(spec["targets"]) != 1 and not spec.get("label_from_one_hot")
    ):
        raise ValueError("Expected one target or an explicit one-hot label")
    return spec


def data_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def checked(path, expected):
    path = Path(path)
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    return path


def project_path(project, relative):
    path = (Path(project) / relative).resolve()
    if not path.is_relative_to(Path(project).resolve()):
        raise ValueError("Artifact path escapes project")
    return path


def label(frame, spec):
    columns = spec["targets"]
    if spec.get("label_from_one_hot"):
        values = frame[columns]
        if not values.isin([0, 1]).all().all() or not values.sum(axis=1).eq(1).all():
            raise ValueError("One-hot targets must contain exactly one 1 per row")
        return values.idxmax(axis=1).rename("__automl_label__")
    result = frame[columns[0]]
    if result.isna().any():
        raise ValueError("Missing target labels")
    return result


def feature_columns(frame, spec):
    excluded = set(spec["targets"]) | set(spec.get("excluded", []))
    # Targets must be present; excluded columns may already be absent, because MLCA's
    # prepared table can drop declared leakage or identifier columns before the split.
    missing = set(spec["targets"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Target columns absent: {sorted(missing)}")
    columns = [c for c in frame.columns if c not in excluded]
    if not columns:
        raise ValueError("No eligible features")
    return columns


def disjoint(frames):
    seen = set()
    for frame in frames:
        ids = set(frame.index)
        if not frame.index.is_unique or frame.index.hasnans or seen.intersection(ids):
            raise ValueError("Duplicate, missing or overlapping row identifiers")
        seen.update(ids)
