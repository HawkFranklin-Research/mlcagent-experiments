"""Ask MLCA's own scope classifier whether each instruction is accepted at its stage, without running it.

Works on a scratch copy of a finished eval-02 HNSCC project: the stage is set in memory and only the
Navigator's scope classification is requested (one model call per instruction). No analysis runs and no
outcome of any study is seen. Usage (laptop, repository root):
  scripts/with_deepseek_key.sh ../mlca-revized/.venv/bin/python assisted/scope_check.py \\
      assisted/instructions-hnscc.yaml [assisted/scope-variants.yaml]
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent / "mlca-revized/src"))
SOURCE = REPO / "bio-bench/runs/deepseek-flash/eval-02-vm3/studies/s4_hnscc-002/attempt-001/run-001"
SCRATCH = Path(os.environ.get("MLCA_SCOPE_SCRATCH", Path.home() / ".cache/mlca-scope-check"))


def main():
    items = []
    for f in sys.argv[1:]:
        plan = yaml.safe_load(Path(f).read_text())
        items += [(f, i["id"], i["at_stage"], i["text"]) for i in plan.get("instructions", [])]
        for stage in plan.get("repair_stages", ["features", "model", "validate", "test"]):
            if "repair" in plan:
                items.append((f, plan["repair"]["id"], stage, plan["repair"]["text"]))
    copy = SCRATCH / "project"
    if copy.exists():
        subprocess.run(["chmod", "-R", "u+w", str(copy)], check=False)
        shutil.rmtree(copy)
    shutil.copytree(SOURCE, copy, symlinks=True)
    subprocess.run(["chmod", "-R", "u+w", str(copy)], check=True)
    os.environ["XDG_CONFIG_HOME"] = str(SCRATCH / "cfg")
    from mlca.api import Session
    from mlca.roles.navigator import Navigator

    config = {"providers": {"development": {"api_key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com",
              "kind": "deepseek", "structured_outputs": True, "timeout_s": 90, "call_deadline_s": 300}},
              "roles": {"navigator": {"provider": "development", "model": "deepseek-flash", "reasoning_effort": "low",
                                      "temperature": 0, "max_output_tokens": 32768}}}
    (SCRATCH / "cfg/mlca").mkdir(parents=True, exist_ok=True)
    (SCRATCH / "cfg/mlca/config.yaml").write_text(yaml.safe_dump(config))
    results = []
    with Session.open(copy) as session:
        for source, ident, stage, text in items:
            session.state.current_event = stage
            decision = Navigator(session).classify_scope(" ".join(text.split()))
            row = {"file": Path(source).name, "id": ident, "stage": stage, "decision": decision.decision,
                   "target_event": getattr(decision, "target_event", None), "reason": decision.reason[:240]}
            results.append(row)
            print(json.dumps(row), flush=True)
    (REPO / "assisted/scope_check_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
