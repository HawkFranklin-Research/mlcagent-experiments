"""Deterministic preparation; inputs are never changed."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=ROOT / "paths.local.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "data_prep")
    args = parser.parse_args()
    manifest = prepare("S4", args.paths, args.out)
    print(
        "Prepared",
        manifest["study"],
        [(o["file"], o["rows"]) for o in manifest["outputs"]],
    )
