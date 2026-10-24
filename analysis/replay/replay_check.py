"""Replay check for R1-2b: re-execute a finished study's stored scripts without the model.

For each selected eval-02 study, the run folder is copied to a scratch directory (replay opens the
project for writing and logs test exposure, so originals are never used), then `mlca replay --verify`
re-runs every generated script in a fresh environment built from the recorded package lock and
compares outputs and metrics with the record. MLCA's split check compares row ids after resetting the
index, which fails when rows were removed before the split (duplicate removal); the split is therefore
also re-checked here by comparing partition assignments in the stored row order.

No model calls. The recorded test key is read from MLCA's user data folder (as during the run).
Only studies run on this laptop can be replayed here: VM runs record VM file paths.
Usage: ../mlca-revized/.venv/bin/python analysis/replay/replay_check.py
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MLCA = REPO.parent / "mlca-revized/.venv/bin/mlca"
SCRATCH = Path(os.environ.get("MLCA_REPLAY_SCRATCH", Path.home() / ".cache/mlca-replay-check"))
RUNS = REPO / "bio-bench/runs/deepseek-flash/eval-02/studies"
STUDIES = ["s1_anemia-001", "s2_rna_er-001", "s2_rna_pr-001", "s2_proteome_er-003"]


def split_check(project):
    import hashlib

    import pandas as pd
    from mlca.api import Session
    from mlca.engine.split import partitions
    from mlca.project.init import column_for
    from mlca.project.inventory import read_table

    with Session.open(project) as session:
        state = session.state
        step = next(s for s in state.steps if "_split_" in s["step_id"])
        record = next(r.payload for r in session.branches.records(session.branch_name) if r.type.value == "SplitFrozen")
        source = Path(step["inputs"][state.primary_data]["path"])
        frame = read_table(source)
        assigned = partitions(frame.reset_index(drop=True), record["parameters"], column_for(state))
        stored = pd.read_csv(project / record["manifest"]["path"])
        return {
            "rows": len(frame),
            "partitions_identical": stored["partition"].tolist() == list(assigned),
            "row_ids_as_stored": stored["row_id"].tolist() == list(frame.index),
            "mlca_hash_check": hashlib.sha256(
                pd.DataFrame({"row_id": range(len(frame)), "partition": assigned}).to_csv(index=False).encode()
            ).hexdigest() == record["sha256"],
        }


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "cfg").mkdir(exist_ok=True)
    env = {**os.environ, "XDG_CONFIG_HOME": str(SCRATCH / "cfg"), "UV_OFFLINE": "1"}
    results = {}
    for study in STUDIES:
        source = RUNS / study / "attempt-001/run-001"
        copy = SCRATCH / study
        if copy.exists():
            subprocess.run(["chmod", "-R", "u+w", str(copy)], check=False)
            shutil.rmtree(copy)
        shutil.copytree(source, copy, symlinks=True)
        subprocess.run(["chmod", "-R", "u+w", str(copy)], check=True)
        done = subprocess.run([str(MLCA), "--project", str(copy), "replay", "--verify", "--json"],
                              capture_output=True, text=True, env=env, timeout=5400, check=False)
        try:
            report = json.loads(done.stdout)
        except json.JSONDecodeError:
            results[study] = {"error": done.stderr[-800:]}
            continue
        steps = report.get("steps", [])
        os.environ["XDG_CONFIG_HOME"] = str(SCRATCH / "cfg")
        results[study] = {
            "mlca_ok": report.get("ok"),
            "generated_scripts": sum(1 for s in steps if "exit_status" in s),
            "generated_scripts_reproduced": sum(1 for s in steps if "exit_status" in s and s.get("ok")),
            "engine_steps": [(s["step_id"], s.get("comparison"), s.get("ok")) for s in steps if "exit_status" not in s],
            "differences": [(s["step_id"], s.get("differences")) for s in steps if s.get("differences") not in (None, [], "[]")],
            "split_recheck": split_check(copy),
        }
        print(study, json.dumps(results[study])[:400], flush=True)
    (HERE / "replay_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO.parent / "mlca-revized/src"))
    main()
