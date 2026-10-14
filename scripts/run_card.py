"""Resolve local input paths without committing them; run only explicitly selected cards."""

import argparse
from pathlib import Path

import yaml


def resolve_card(path, paths_file):
    path = Path(path).resolve()
    settings = yaml.safe_load(Path(paths_file).read_text())
    card = yaml.safe_load(path.read_text())
    data = []
    for item in card["data"]:
        for key, value in settings.items():
            item = item.replace("${" + key + "}", str(value))
        if "${" in item:
            raise ValueError("Unresolved local input variable")
        data.append(str((path.parent / item).resolve()))
    card["data"] = data
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("card", type=Path)
    parser.add_argument(
        "--paths",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paths.local.yaml",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    card = resolve_card(args.card, args.paths)
    if args.reference:
        card.update(mode="driver", client=None, model=None, measurement_arm="reference")
        card["config"] = {"providers": {}, "roles": {}}
    # Resolved paths are local-only. The original card remains the shareable protocol.
    local = args.out.parent / (args.out.name + ".local.yaml")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(yaml.safe_dump(card, sort_keys=False))
    from mlca.evaluation.bench import run

    result = run(local, 1, args.out, bio="bio-bench" in args.card.parts)
    print([(r["project"], r["status"]) for r in result["runs"]])
