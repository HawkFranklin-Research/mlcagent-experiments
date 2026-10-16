"""Hash allowlisted setup evidence separately from immutable original attempts."""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def curate(folder):
    folder = Path(folder).resolve()
    destination = folder / "setup-evidence"
    destination.mkdir(exist_ok=False)
    rows = []
    for dataset in sorted(folder.iterdir()):
        if not dataset.is_dir() or dataset.name in {"curated", "setup-evidence"}:
            continue
        for setup in sorted(dataset.glob("setup-[0-9][0-9][0-9]")):
            for relative in [
                "setup-events.json",
                "decision-windows.json",
                ".mlca/requests",
                ".mlca/llm",
                ".mlca/usage.jsonl",
                ".mlca/egress.jsonl",
            ]:
                path = setup / relative
                for source in sorted(path.rglob("*")) if path.is_dir() else [path]:
                    if source.is_symlink() or any(
                        p.is_symlink() for p in source.parents
                    ):
                        raise ValueError("Symlink in setup evidence")
                    if not source.is_file():
                        continue
                    target = destination / source.relative_to(folder)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    digest = hashlib.sha256(source.read_bytes()).hexdigest()
                    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                        raise ValueError("Evidence copy differs")
                    rows.append(
                        {"source": str(source.relative_to(folder)), "sha256": digest}
                    )
    with (destination / "MANIFEST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["source", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=Path)
    print(curate(parser.parse_args().round))
