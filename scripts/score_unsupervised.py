"""External evaluator, NOT agent work. Labels never enter MLCA projects."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(source, outputs, labels, kind):
    source, outputs = Path(source), Path(outputs)
    original = (
        pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    )
    rows = (
        pd.read_parquet(outputs)
        if outputs.suffix == ".parquet"
        else pd.read_csv(outputs)
    )
    if not labels or not set(labels) <= set(original.columns):
        raise ValueError("Explicit evaluation labels must exist in the external source")
    if not {"row_id", "partition"} <= set(rows.columns):
        raise ValueError("Outputs require row_id and partition")
    ids = rows["row_id"]
    if (
        ids.isna().any()
        or ids.duplicated().any()
        or not pd.api.types.is_integer_dtype(ids)
    ):
        raise ValueError("Row identifiers must be unique, nonmissing integers")
    if (ids < 0).any() or (ids >= len(original)).any():
        raise ValueError("Output row identifiers are outside the original source")
    if not set(rows.partition) <= {"train", "val", "test"}:
        raise ValueError("Unknown partition")
    truth = original.loc[ids, labels].reset_index(drop=True)
    if truth.isna().any().any():
        raise ValueError("Evaluation labels are missing; no rows are silently dropped")
    test = rows.partition.eq("test").to_numpy()
    if not test.any():
        raise ValueError("Test outputs are required")
    metrics = {}
    if kind == "clustering":
        if "cluster" not in rows:
            raise ValueError("Cluster assignments are missing")
        for label in labels:
            metrics[label] = {
                "ari": adjusted_rand_score(
                    truth.loc[test, label], rows.loc[test, "cluster"]
                ),
                "nmi": normalized_mutual_info_score(
                    truth.loc[test, label], rows.loc[test, "cluster"]
                ),
            }
    elif kind == "representation":
        dimensions = [c for c in rows if c.startswith("embedding_")]
        if not dimensions:
            raise ValueError("Embedding dimensions must be named embedding_*")
        train = rows.partition.eq("train").to_numpy()
        X = rows[dimensions].to_numpy(dtype=float)
        if not train.any() or not np.isfinite(X).all():
            raise ValueError("Finite train and test embeddings are required")
        for label in labels:
            probe = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
            )
            probe.fit(X[train], truth.loc[train, label])
            predicted = probe.predict(X[test])
            metrics[label] = {
                "accuracy": accuracy_score(truth.loc[test, label], predicted),
                "macro_f1": f1_score(
                    truth.loc[test, label], predicted, average="macro", zero_division=0
                ),
            }
    else:
        raise ValueError("Unknown evaluation kind")
    return {
        "schema_version": "1",
        "producer": "external_evaluation_scorer",
        "agent_work": False,
        "kind": kind,
        "source_sha256": checksum(source),
        "outputs_sha256": checksum(outputs),
        "rows": {
            "train": int(rows.partition.eq("train").sum()),
            "test": int(test.sum()),
        },
        "metrics": metrics,
        "method": "Test-only ARI/NMI"
        if kind == "clustering"
        else "External linear probe: training embeddings/labels only; test embeddings scored once. Seed 42.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument(
        "--kind", choices=["clustering", "representation"], required=True
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.source, args.outputs, args.labels, args.kind)
    if args.out.exists():
        raise ValueError("Use a fresh scoring output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
