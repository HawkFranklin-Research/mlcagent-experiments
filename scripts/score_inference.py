"""Research-side comparison of logged inference with researcher metadata, never an agent input."""

import argparse
import json
from pathlib import Path

import yaml


def score(proposal, expected):
    return {
        "producer": "external_inference_scorer",
        "agent_work": False,
        "problem_type": proposal.get("problem_type") == expected["problem_type"],
        "targets": sorted(proposal.get("targets", []))
        == sorted(expected.get("targets", [])),
        "group_column": proposal.get("group_column") == expected.get("group_column"),
        "time_columns": proposal.get("time_columns", [])
        == expected.get("time_columns", []),
        "scope": "Exact metadata agreement, not proof that alternative scientific aims are invalid.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Verify the log before using it, without launching a Session or making a model request.
    from mlca.log.store import Store

    store = Store(args.project)
    store.verify()
    proposals = [
        r.payload["proposal"]
        for r in store.read()
        if r.type.value == "InferenceRecorded"
    ]
    if not proposals:
        raise ValueError("No recorded inference")
    entries = yaml.safe_load(args.manifest.read_text())
    expected = next(e for e in entries if e["id"] == args.dataset_id)
    if expected["role"] not in {"development", "heldout", "evaluation"}:
        raise ValueError("Unknown table role")
    result = score(proposals[-1], expected)
    if args.out.exists():
        raise ValueError("Use a fresh scoring output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
