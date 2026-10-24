"""Build each study environment offline before a batch, exactly as a worker would.

A worker creates `.mlca/env` with MLCA's own `env.create`, the controller's interpreter and
`UV_OFFLINE=1`. If the uv cache lacks a wheel (wrong Python tag, missing package), every study
fails at setup. This check fails once, before any study starts, and compares the resulting
package versions with the reference locks in `bio-bench/env-reference/`.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parent / "mlca-revized"
sys.path.insert(0, str(CODE / "src"))
REFERENCE = ROOT / "bio-bench/env-reference"


def package_sets():
    """Distinct runtime package lists over all MLCA workflow templates, keyed by extras."""
    lists = {
        tuple(yaml.safe_load(f.read_text()).get("default_runtime_packages") or [])
        for f in sorted((CODE / "templates").glob("*/template.yaml"))
    }
    base = set.intersection(*(set(p) for p in lists)) if lists else set()
    return {
        "-".join(sorted(set(p) - base)) or "default": list(p) for p in sorted(lists)
    }


def build(packages, offline=True):
    from mlca.project.env import create

    previous = os.environ.get("UV_OFFLINE")
    if offline:
        os.environ["UV_OFFLINE"] = "1"
    else:
        os.environ.pop("UV_OFFLINE", None)
    try:
        with tempfile.TemporaryDirectory(prefix="mlca-env-preflight-") as tmp:
            root = Path(tmp)
            (root / ".mlca").mkdir()
            return create(root, packages)["packages"]
    finally:
        if previous is None:
            os.environ.pop("UV_OFFLINE", None)
        else:
            os.environ["UV_OFFLINE"] = previous


def preflight(write_reference=False):
    results = {}
    for key, packages in package_sets().items():
        try:
            lines = build(packages)
        except Exception as exc:  # report which set failed, then stop
            raise ValueError(
                f"Environment preflight failed for '{key}' ({sys.executable}, "
                f"Python {sys.version.split()[0]}): {exc}"
            ) from exc
        reference = REFERENCE / f"{key}.requirements.txt"
        text = "\n".join(lines) + "\n"
        if write_reference:
            REFERENCE.mkdir(parents=True, exist_ok=True)
            reference.write_text(text)
        elif reference.exists() and reference.read_text() != text:
            expected = set(reference.read_text().split())
            raise ValueError(
                f"Environment preflight for '{key}' differs from {reference.relative_to(ROOT)}: "
                f"missing {sorted(expected - set(lines))}, unexpected {sorted(set(lines) - expected)}"
            )
        results[key] = {
            "packages": packages,
            "lock_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "matches_reference": reference.exists(),
        }
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "sets": results,
    }


def seed():
    """Fill the uv cache online with exactly the reference versions, using this interpreter."""
    for key in package_sets():
        pinned = (REFERENCE / f"{key}.requirements.txt").read_text().split()
        build(pinned, offline=False)
    return preflight()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Record this machine's locks as the reference (only on the reference machine)",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Download the reference versions into the uv cache (online), then run the offline check",
    )
    args = parser.parse_args()
    result = seed() if args.seed else preflight(args.write_reference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
