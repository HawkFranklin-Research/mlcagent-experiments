"""Assisted runs: the released agent runs autonomously, and pre-written instructions are given at fixed
stages through MLCA's own human-on-the-loop path (no change to MLCA).

At a decision in a listed stage, the scripted window halts the agent's proposed choice once and returns
a custom instruction, which MLCA executes as the user's step (`session.custom`). If a step fails, the
recovery question is answered "Pause" and the repair instruction is given the same way; the agent then
continues. Every instruction counts toward the per-run cap. Everything else is identical to an eval-02
study: data, brief, external cohort, model settings, rails, sandbox, and the controller's scoring.

Usage (from the experiments repository root, with the DeepSeek key launcher):
  scripts/with_deepseek_key.sh ../mlca-revized/.venv/bin/python assisted/run_assisted.py \\
      --instructions assisted/instructions-hnscc.yaml --out bio-bench/runs/deepseek-flash/assisted-hnscc
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CODE = REPO.parent / "mlca-revized"
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Identical to the eval-02 study configuration (bio-bench/runs/deepseek-flash/eval-02-vm3/.control/*.json).
ROLE = {
    "max_output_tokens": 32768,
    "model": "deepseek-flash",
    "provider": "development",
    "reasoning_effort": "low",
    "temperature": 0,
}


def study_config(rate_dir):
    return {
        "budgets": {
            "session_max_usd": None,
            "turn_max_usd": None,
            "worker_max_attempts": 3,
        },
        "measurement_arm": "worker",
        "providers": {
            "development": {
                "api_key_env": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",
                "call_deadline_s": 300,
                "kind": "deepseek",
                "rate_limit_dir": str(rate_dir),
                "requests_per_minute": 60.0,
                "structured_outputs": True,
                "timeout_s": 90,
            }
        },
        "rails_enabled": True,
        "roles": {"navigator": dict(ROLE), "worker": dict(ROLE)},
        "sandbox": {"allow_level_b": False},
    }


def make_window(plan, record):
    from mlca.autonomous.windows import DecisionWindow, principal
    from ulid import ULID

    class ScriptedWindow(DecisionWindow):
        """A human who halts once per listed stage and gives the pre-written instruction."""

        def __init__(self):
            super().__init__(0)
            self.applied = set()

        def remaining(self):
            return plan["cap_per_run"] - len(record)

        def intervene(self, session, kind, proposed, options, item, trigger):
            wid = str(ULID())
            session.emit(
                "DecisionWindowOpened",
                {
                    "window_id": wid,
                    "kind": kind,
                    "seconds": 0,
                    "proposed": proposed,
                    "options": options,
                    "principal": session.principal,
                },
            )
            with principal(session, True):
                session.emit(
                    "DecisionWindowHalted",
                    {"window_id": wid, "wait_s": 0.0},
                    actor="user",
                )
            record.append(
                {
                    "id": item["id"],
                    "trigger": trigger,
                    "stage": session.state.current_event,
                    "words": len(item["text"].split()),
                    "at": time.time(),
                }
            )

        def decide(self, session, kind, proposed, options):
            if kind == "choice" and self.remaining() > 0:
                stage = session.state.current_event
                item = next(
                    (
                        i
                        for i in plan["instructions"]
                        if i["at_stage"] == stage and i["id"] not in self.applied
                    ),
                    None,
                )
                if item is not None:
                    self.applied.add(item["id"])
                    self.intervene(session, kind, proposed, options, item, "stage")
                    return {"custom": " ".join(item["text"].split())}, True
            return super().decide(session, kind, proposed, options)

        def repair(self, session):
            """After a failure: pause instead of the automatic retry, then give the repair instruction."""
            if self.remaining() <= 0:
                return False
            from mlca.autonomous.windows import principal as as_user

            for question in session.questions():
                kinds = [o.get("effect", {}).get("kind") for o in question["options"]]
                if "pause" in kinds:
                    with as_user(session, True):
                        session.answer_question(
                            question["question_id"], kinds.index("pause")
                        )
            item = plan["repair"]
            self.intervene(
                session, "failure", 0, [{"title": "repair"}], item, "failure"
            )
            with as_user(session, True):
                outcome = session.custom(" ".join(item["text"].split()))
            record[-1]["custom_decision"] = (
                outcome.get("decision") if isinstance(outcome, dict) else None
            )
            return True

    return ScriptedWindow()


def assisted_drive(session, max_turns=100):
    """MLCA's autonomous drive, except that a failure or a refused instruction does not end the run
    while the scripted human still has instructions left."""
    from mlca.errors import MlcaUserError

    window = session.decision_window
    pauses = 0
    for _ in range(max_turns):
        result = session.autopilot(max_turns)
        if result["reason"] == "completed":
            return
        if result["reason"] == "failure" and window.repair(session):
            continue
        if result["reason"] == "paused" and not session.questions() and pauses < 5:
            # A refused custom instruction pauses the autopilot; the scripted human lets it continue.
            pauses += 1
            continue
        if result["reason"] in {"question", "approval", "paused"}:
            answer_like_the_autonomous_runner(session, result)
            continue
        failure = next(
            (
                r.payload["category"]
                for r in reversed(session.store.read())
                if r.payload.get("category")
            ),
            "agent_failure",
        )
        raise MlcaUserError(
            "Autonomous run stopped: " + result["reason"], category=failure
        )
    raise MlcaUserError("Autonomous turn limit reached", category="budget_stop")


def answer_like_the_autonomous_runner(session, result):
    """Questions and preauthorised approvals are resolved exactly as mlca.autonomous.runner.drive does."""
    from mlca.autonomous.windows import principal

    window = session.decision_window
    if session.questions():
        for question in session.questions():
            chosen, human = window.decide(
                session, "question", question["recommended"], question["options"]
            )
            if not human:
                session.emit(
                    "QuestionAutoResolved",
                    {
                        "question_id": question["question_id"],
                        "answer": chosen,
                        "reason": "Autonomous policy selected the recorded recommendation",
                    },
                )
            with principal(session, human):
                session.answer_question(question["question_id"], chosen)
    else:
        for choice in result["turn"]:
            if not choice["id"].startswith("human:"):
                continue
            _, human = window.decide(session, "approval", 0, [choice])
            with principal(session, human):
                if choice["id"] == "human:unlock":
                    session.unlock_test(
                        "UNLOCK TEST",
                        "Preauthorized autonomous policy; validation is frozen",
                    )
                elif choice["id"].startswith("human:confirm:"):
                    session.confirm(
                        choice["id"].split(":", 2)[2], "Preauthorized autonomous policy"
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--runs",
        type=int,
        nargs="*",
        help="Run numbers to do (default: all in the plan)",
    )
    args = parser.parse_args()
    plan = yaml.safe_load(args.instructions.read_text())
    out = args.out.resolve()
    (out / ".control").mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "UV_OFFLINE": "1",
            **{
                k: "2"
                for k in [
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                ]
            },
        }
    )
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("Run through scripts/with_deepseek_key.sh")

    from env_preflight import preflight

    environment = preflight()
    from mlca.autonomous import runner
    from mlca.autonomous.runner import run
    from mlca.paths import atomic_json

    runner.drive = assisted_drive  # in this process only; MLCA's files are unchanged
    data = REPO / plan["data"]
    external = [REPO / plan["external"]] if plan.get("external") else []
    summary = {
        "plan": str(args.instructions),
        "plan_text": plan,
        "environment": environment,
        "studies": [],
    }
    for number in args.runs or range(1, plan["runs"] + 1):
        study = f"{plan['task']}-assisted-{number:03d}"
        relative = Path("studies") / study / "attempt-001"
        attempt = out / relative
        if (attempt / "worker-result.json").exists():
            print(f"{study}: already done, kept", flush=True)
            continue
        attempt.parent.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CONFIG_HOME"] = str(out / ".control/config" / study)
        record = []
        window = make_window(plan, record)
        started = time.time()
        try:
            result = run(
                data,
                attempt,
                brief=plan["brief"],
                config=study_config(out / ".control/rate"),
                exclude_columns=plan.get("exclude_columns", []),
                halt_window=0,
                window=window,
                external=external,
            )["runs"][0]
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - retain the outcome of every study
            result = {
                "status": "failed",
                "category": "worker_exception",
                "error_type": type(exc).__name__,
            }
        result["assisted"] = {
            "instructions_used": record,
            "count": len(record),
            "words": sum(r["words"] for r in record),
            "cap": plan["cap_per_run"],
        }
        atomic_json(attempt / "worker-result.json", result)
        job = {
            "id": study,
            "dataset": plan["task"],
            "replicate": number,
            "attempts": [str(relative)],
            "status": "finished",
        }
        if plan.get("score", True):
            from run_batch import entries, finish_job

            expected = {
                e["id"]: e for e in entries(REPO / plan["manifest"], plan["role"])
            }[plan["task"]]
            finish_job(out, job, expected)
        scored = job.get("result", result)
        line = {
            "study": study,
            "success": scored.get("success"),
            "status": scored.get("status"),
            "category": scored.get("category"),
            "furthest_event": scored.get("furthest_event"),
            "instructions": result["assisted"],
            "wall_s": round(time.time() - started, 1),
        }
        atomic_json(attempt / "assisted-result.json", {**line, "scored": scored})
        summary["studies"].append(line)
        print(json.dumps(line), flush=True)
    atomic_json(out / "assisted-summary.json", summary)


if __name__ == "__main__":
    main()
