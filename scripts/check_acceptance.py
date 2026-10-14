"""Verify curated reference evidence and build an acceptance table without inference."""

import argparse
import csv
import json
from pathlib import Path

from mlca.paths import sha256

EXPECTED = [("S1_anemia_cbc", "task_card")] + [
    ("S2_brca_receptors", "task_card_" + modality + "_" + receptor)
    for modality in ("rna", "proteome")
    for receptor in ("er", "pr", "her2")
]


def check(records, root, out):
    rows = []
    for study, task in EXPECTED:
        row = json.loads((records / (study + "_" + task + ".json")).read_text())
        assert row["model_requests"] == 0
        export = root / "bio-bench" / study / "runs/reference/2026-10-14-final" / task
        with (export / "EXPORT_MANIFEST.csv").open() as f:
            entries = list(csv.DictReader(f))
        listed = {entry["path"] for entry in entries}
        actual = {
            str(file.relative_to(export))
            for file in export.rglob("*")
            if file.is_file() and file.name != "EXPORT_MANIFEST.csv"
        }
        assert listed == actual, "Export has missing or unlisted files"
        for entry in entries:
            path = Path(entry["path"])
            assert not path.is_absolute() and ".." not in path.parts
            assert sha256(export / path) == entry["sha256"], entry["path"]
        for file in export.rglob("*"):
            parts = file.relative_to(export).parts
            assert not any(
                parts[i : i + 2]
                in {
                    (".mlca", "private"),
                    (".mlca", "data"),
                    (".mlca", "env"),
                    (".mlca", "vault"),
                }
                for i in range(len(parts) - 1)
            )
            assert not file.is_symlink()
            assert not file.name.endswith((".local.md", ".local.yaml"))
        # Derive outcomes from retained evidence even when bench stopped on failure.
        project = export / "run-001"
        events = [
            json.loads(line)
            for line in (project / ".mlca/log/events.jsonl").read_text().splitlines()
        ]
        failures = [event for event in events if event["type"] == "StepFailed"]
        last_failure = failures[-1] if failures else None
        instructions = []
        instruction_file = project / "custom_instructions.md"
        if instruction_file.exists():
            for line in instruction_file.read_text().splitlines():
                if line.startswith('{"event":'):
                    instructions.append(json.loads(line))
        row["not_achieved"] = sum(
            item["outcome"] != "StepCompleted" for item in instructions
        )
        row["custom_instructions_submitted"] = len(instructions)
        row["last_failure_category"] = (
            last_failure["payload"].get("category") if last_failure else None
        )
        row["last_failure_event"] = last_failure["event_id"] if last_failure else None
        row["completed_steps"] = sum(
            event["type"] == "StepCompleted" for event in events
        )
        row["outcomes_source"] = (
            "derived from exported event log and custom instruction records"
        )
        row["export_path"] = str(export.relative_to(root))
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = check(args.records, args.root, args.out)
    print([(r["task"], r["status"], r["core_completed"]) for r in rows])
