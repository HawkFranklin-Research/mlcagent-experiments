"""Generate comparison specs from metadata only; never opens dataset files."""

import argparse
import os
from pathlib import Path

import yaml

try:
    from .common import ROOT
except ImportError:
    from common import ROOT


def generate(manifest, output):
    manifest, output = Path(manifest), Path(output)
    entries = yaml.safe_load(manifest.read_text())
    groups = {"heldout": [], "development": []}
    for entry in entries:
        kind, targets = entry["problem_type"], entry.get("targets", [])
        one_hot = kind == "multiclass" and len(targets) > 1
        if kind not in {"binary", "multiclass", "regression"} or not targets:
            continue
        if len(targets) != 1 and not one_hot:
            continue
        excluded = []
        for key in ["leakage_columns", "identifier_columns", "evaluation_only"]:
            excluded.extend(entry.get(key, []))
        if entry.get("group_column"):
            excluded.append(entry["group_column"])
        spec = {
            "id": entry["id"],
            "data": os.path.relpath(manifest.parent / entry["file"], ROOT),
            "sha256": entry["sha256"],
            "problem_type": kind,
            "targets": targets,
            "label_from_one_hot": one_hot,
            "excluded": sorted(set(excluded)),
            "external": None,
        }
        role = entry["role"].replace("-", "")
        if role in groups:
            groups[role].append(spec)
    output.mkdir(parents=True, exist_ok=True)
    for group, specs in groups.items():
        (output / f"{group}.yaml").write_text(yaml.safe_dump(specs, sort_keys=False))
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT.parent / "mlca-revized/benchmarks/manifest.yaml",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "specs")
    args = parser.parse_args()
    generate(args.manifest, args.out)
