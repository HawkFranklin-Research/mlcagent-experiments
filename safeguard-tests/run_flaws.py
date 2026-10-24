#!/usr/bin/env python3
"""Run pre-registered planted flaws through the frozen MLCA step runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ENGINE = REPO.parent / "mlca-revized"
sys.path.insert(0, str(ENGINE / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, cwd: Path = ENGINE) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def rail_digest() -> str:
    archive = subprocess.check_output(["git", "archive", "freeze-v2", "rails"], cwd=ENGINE)
    return hashlib.sha256(archive).hexdigest()


def fixture_data(path: Path, imbalanced: bool = False) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 4))
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, 5)])
    latent = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=240)
    frame["y"] = (latent > np.quantile(latent, 0.85 if imbalanced else 0.5)).astype(int)
    frame.to_csv(path, index=False)


def protocol_script(action, template: str) -> str:
    """A small, generic clean worker script for completing paired controls."""
    outputs = [o.name for o in action.outputs]
    event = action.event
    if event in {"data", "eda"}:
        name = outputs[0]
        return f"""import pandas as pd, mlca_runtime as m
df=pd.read_csv(list(m.inputs().values())[0])
m.save_table(df.describe(include='all').transpose().reset_index(),{name!r})
"""
    if event in {"preprocess", "features"}:
        name = outputs[0]
        return """import pandas as pd, json, mlca_runtime as m
from sklearn.preprocessing import StandardScaler, FunctionTransformer
i=m.inputs(); p=json.load(open(i['_protocol'])); tr=pd.read_parquet(i['train'])
drop=set(p['targets'])|set(p.get('excluded_columns',[]))|({p['group_column']} if p.get('group_column') else set())
X=tr.drop(columns=[c for c in drop if c in tr])
obj=StandardScaler() if EVENT == 'preprocess' else FunctionTransformer()
fit=m.fit_record(obj,X,'train')
m.save_model(fit,OUTPUT)
""".replace("EVENT", repr(event)).replace("OUTPUT", repr(name))
    if event == "model":
        name = outputs[0]
        estimator = "LogisticRegression(max_iter=1000)" if template in {"tabular_classification", "multilabel_classification"} else "LinearRegression()"
        fit = "MultiOutputClassifier(" + estimator + ")" if template == "multilabel_classification" else estimator
        yexpr = "tr[p['targets']]" if template == "multilabel_classification" else "tr[p['targets'][0]]"
        return f"""import pandas as pd, json, mlca_runtime as m
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputClassifier
i=m.inputs(); p=json.load(open(i['_protocol'])); tr=pd.read_parquet(i['train'])
drop=set(p['targets'])|set(p.get('excluded_columns',[]))|({p['group_column']} if p.get('group_column') else set())
X=tr.drop(columns=[c for c in drop if c in tr]); y={yexpr}
model=m.fit_record({fit},X,'train',y=y); m.save_model(model,{name!r})
"""
    if event in {"validate", "test"}:
        table = next((o.name for o in action.outputs if o.kind == "table"), outputs[0])
        data_output = next((o.name for o in action.outputs if o.kind == "data"), None)
        part = "val" if event == "validate" else "test"
        score = "roc_auc_score" if template in {"tabular_classification", "multilabel_classification"} else "mean_squared_error"
        metric = "p['task'].get('primary_metric','roc_auc')" if template in {"tabular_classification", "multilabel_classification"} else "p['task'].get('primary_metric','rmse')"
        target = "p['targets'][0]"
        pred_expr = "model.predict_proba(X)[:,1]" if template == "tabular_classification" else "model.predict(X)"
        metric_expr = "roc_auc_score(yy,pred)" if template == "tabular_classification" else "mean_squared_error(yy,pred)**0.5"
        save_rows = ""
        if data_output:
            save_rows = f"""m.save_row_outputs(pd.DataFrame({{'row_id':df.index,'partition':'test','y_true':yy.to_numpy(),'y_pred':model.predict(X)}}),{data_output!r})"""
        return f"""import pandas as pd, json, pickle, mlca_runtime as m
from sklearn.metrics import roc_auc_score, mean_squared_error
i=m.inputs(); p=json.load(open(i['_protocol'])); df=pd.read_parquet(i[{part!r}]); model=pickle.load(open(i['latest_model'],'rb'))
y={target}; drop=set(p['targets'])|set(p.get('excluded_columns',[]))|({{p['group_column']}} if p.get('group_column') else set())
X=df.drop(columns=[c for c in drop if c in df]); yy=df[y]; pred={pred_expr}; value=float({metric_expr})
m.record_metric({metric},value,{('val' if event == 'validate' else 'test')!r})
m.save_table(pd.DataFrame({{'metric':[str({metric})],'value':[value]}}),{table!r})
{save_rows}
"""
    if event in {"interpret", "report"}:
        output = outputs[0]
        obj = next((o for o in action.outputs if o.kind in {"json", "report"}), None)
        if obj:
            return f"import mlca_runtime as m\nm.save_json({{'status':'fixture control'}},{obj.name!r})\n"
        return f"import pandas as pd, mlca_runtime as m\nm.save_table(pd.DataFrame({{'status':['fixture control']}}),{output!r})\n"
    return "import mlca_runtime as m\n"


def build_responses(case: dict, variant: str, workflow) -> list[dict]:
    script_field = "planted_script" if variant == "flaw" else "control_script"
    result = []
    for action in workflow.actions.values():
        if action.builtin or action.role != "worker":
            continue
        text = protocol_script(action, case["template"])
        if variant == "flaw" and action.id == case["action"] and case["id"] not in {"F5", "F6"}:
            text = case[script_field]
        result.append({"match": {"role": "worker", "action_id": action.id}, "text": f"```python\n{text}\n```"})
    if case["id"] not in {"F5", "F6"}:
        return result
    return result


def matrix_markdown(rows: list[dict]) -> str:
    columns = ["flaw", "table", "outcome", "check", "message", "run_path"]
    lines = ["# Frozen-engine safeguard test matrix", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = [str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_matrix(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = ["flaw", "table", "outcome", "check", "message", "run_path"]
    with (out / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    (out / "MATRIX.md").write_text(matrix_markdown(rows), encoding="utf-8")


def create_synthetic_file(path: Path, imbalanced: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fixture_data(path, imbalanced=imbalanced)


def run_one(case: dict, variant: str, out: Path, manifest: dict) -> dict:
    from mlca.api import Session
    from mlca.errors import MlcaUserError
    from mlca.project.init import initialise
    from mlca.templates.registry import get
    from score_leakage import score

    case_id = case["id"] + ("-flaw" if variant == "flaw" else "-control")
    project = out / "runs" / case_id / "project"
    project.parent.mkdir(parents=True, exist_ok=True)
    table = case["table"]
    if table == "synthetic":
        data = out / "inputs" / f"synthetic-{case['id']}.csv"
        create_synthetic_file(data, imbalanced=bool(case.get("imbalance")))
        expected = {"role": "development", "leakage_columns": [], "identifier_columns": [], "targets": ["y"], "group_column": None, "time_columns": []}
    else:
        meta = next(row for row in manifest if row["id"] == table)
        data = ENGINE / "benchmarks" / meta["file"]
        expected = {"role": "development", "leakage_columns": meta.get("leakage_columns", []), "identifier_columns": meta.get("identifier_columns", []), "targets": meta.get("targets", []), "group_column": meta.get("group_column"), "time_columns": meta.get("time_columns", [])}
    workflow = get(case["template"])
    responses = build_responses(case, variant, workflow)
    script_path = out / "runs" / case_id / "fake-provider.yaml"
    script_path.write_text(yaml.safe_dump({"responses": responses}, sort_keys=False), encoding="utf-8")
    config = {
        "providers": {"offline": {"kind": "fake", "script": str(script_path), "request_cost_bound_usd": 0}},
        "roles": {"worker": {"provider": "offline", "model": "scripted"}},
        "sandbox": {"allow_level_b": False},
    }
    brief = dict(case["brief"])
    root = initialise(project, [data], case["template"], brief, config)
    start = time.monotonic()
    exception = None
    reached_report = False
    with Session.open(root) as session:
        guard_steps = 0
        while guard_steps < 80:
            guard_steps += 1
            event = session.state.current_event
            if event == "onboard":
                opts = session.options()
                option = next(o for o in opts if o.get("action_id") == "confirm_brief")
                result = session.choose(option["id"])
                if result.get("status") != "completed":
                    exception = {"type": "ActionResult", "category": result.get("category", "unknown"), "message": result.get("message", "")}
                    break
                session.confirm("brief_confirmed", "Setup fixed in the preregistered case")
            elif event == "data" and not session.state.task:
                setup = case["setup"]
                task_type = {"tabular_classification": "binary", "regression": "regression", "multilabel_classification": "multilabel"}[case["template"]]
                targets = json.loads(brief.get("targets", "[]")) or ([brief["target"]] if brief.get("target") else [])
                session.emit("TaskConfigured", {"task": {"type": task_type, "targets": targets, "primary_metric": "roc_auc" if task_type == "binary" else ("rmse" if task_type == "regression" else "macro_f1"), "time_columns": setup.get("time_columns", [])}, "excluded_columns": setup.get("excluded_columns", []), "group_column": setup.get("group_column")})
            if event == "validate" and not any(r.type.value == "UserConfirmed" and r.payload.get("subgoal") == "model_selected" for r in session.store.read()):
                pass
            if event == "test" and not session.state.test_unlocked:
                session.unlock_test("UNLOCK TEST", "Planned test-scoring step")
            options = session.options()
            action_id = case["action"] if event == next((a.event for a in workflow.actions.values() if a.id == case["action"]), "") else None
            option = next((o for o in options if o.get("action_id") == action_id), None) if action_id else None
            if option is None:
                option = next((o for o in options if o.get("action_id") and workflow.actions[o["action_id"]].role != "supporting"), None)
            if option is None:
                if event == "report":
                    reached_report = True
                    break
                try:
                    session.next()
                    continue
                except Exception as exc:
                    exception = {"type": type(exc).__name__, "category": getattr(exc, "category", "runtime_error"), "message": str(exc)[:240]}
                    break
            params = dict(option.get("params") or {})
            if case["id"] == "F5" and event == "split":
                params["method"] = "stratified" if variant == "flaw" else "group"
                params["group_column"] = "RecordID"
            if case["id"] == "F6" and event == "split":
                params["method"] = "random" if variant == "flaw" else "time"
            if params != option.get("params"):
                replacement = {**option, "params": params}
                session.emit("OptionsProposed", {"options": [replacement], "source": "safeguard-test"})
                option = replacement
            try:
                result = session.choose(option["id"])
                if result.get("status") not in {None, "completed"}:
                    exception = {"type": "ActionResult", "category": result.get("category", "unknown"), "message": result.get("message", "")}
                    break
                for goal in workflow.events[event].subgoals:
                    if goal.check.type == "user_confirm":
                        session.confirm(goal.id, "Preregistered safeguard test")
                if session.state.event_statuses.get(event) != "done":
                    exception = {"type": "StepNotCompleted", "category": "step_incomplete", "message": "Engine did not complete the selected step"}
                    break
                if event == "validate":
                    session.confirm("model_selected", "Frozen clean or planted validation result")
                if event == "report":
                    reached_report = True
                    break
                session.next()
            except Exception as exc:
                exception = {"type": type(exc).__name__, "category": getattr(exc, "category", "runtime_error:" + type(exc).__name__), "message": str(exc)[:240]}
                break
        records = session.store.read()
        warnings = [r.payload for r in records if r.type.value in {"Warning", "CheckWarning", "Note"}]
        steps = [s for s in session.state.steps if s.get("action_id") == case["action"]]
        target_action = steps[-1] if steps else None
        split_step = next((s for s in session.state.steps if s.get("event_id") == "split" and s.get("status") == "completed"), None)
        split_method = (split_step or {}).get("params", {}).get("method")
        try:
            scored = score(project, expected)
        except Exception as exc:
            scored = {"leakage_clean": None, "error": type(exc).__name__}
    elapsed = round(time.monotonic() - start, 3)
    records_data = [json.loads(line) for line in (project / ".mlca/log/events.jsonl").read_text().splitlines()]
    fit_events = [r for r in records_data if r.get("type") == "StepCompleted" and r.get("payload", {}).get("fits")]
    target_step_status = target_action.get("status") if target_action else None
    check = ""
    message = ""
    if variant == "control":
        outcome = "control_passed" if reached_report or case["id"] in {"F5", "F6"} and split_step else "control_failed"
        check = "clean script / declared split control"
        message = "reached report" if reached_report else "clean split completed" if split_step else (exception or {}).get("category", "")
    elif case["id"] in {"F5", "F6"}:
        expected_method = "group" if case["id"] == "F5" else "time"
        if split_step and split_method == expected_method:
            outcome, check, message = "refused", "frozen split metadata enforcement", f"requested {params['method']}; engine recorded {split_method}"
        else:
            outcome, check, message = "undetected", "", f"engine recorded {split_method or 'no split'}"
    elif exception:
        category = exception.get("category", "")
        lower_message = exception.get("message", "").lower()
        if any(term in category for term in ["fit_rule", "static_check", "test_lock", "MlcaUserError"]) or "fit rows must match the frozen training index" in lower_message:
            outcome, check = "refused", category or "fit_record training-index enforcement"
        elif case["id"] in {"F2", "F7"} and "test" in lower_message:
            outcome, check = "refused", "locked test input is not supplied to this action"
        else:
            outcome, check = "undetected", "runtime failure, not a safeguard refusal"
        message = exception.get("message", "")
    elif warnings:
        outcome, check, message = "warned", "event warning", json.dumps(warnings[0], sort_keys=True)[:200]
    elif scored.get("leakage_clean") is False:
        outcome, check, message = "external_only", "external leakage scorer", json.dumps(scored.get("criteria", {}), sort_keys=True)[:200]
    else:
        outcome, check, message = "undetected", "", "step completed with no warning or external detection"
    result = {
        "flaw": case["id"] + ("/control" if variant == "control" else ""),
        "table": table,
        "outcome": outcome,
        "check": check,
        "message": message,
        "run_path": str(project.relative_to(REPO)),
        "reached_report": reached_report,
        "target_step_status": target_step_status,
        "exception": exception,
        "warnings": warnings,
        "leakage": scored,
        "split_method": split_method,
        "fit_events": len(fit_events),
        "elapsed_s": elapsed,
        "engine_crash": bool(exception and not any(r.get("type") == "StepFailed" for r in records_data)),
    }
    (project.parent / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_environment(out: Path) -> None:
    from mlca.sandbox.level import detect

    lock = ENGINE / "uv.lock"
    environment = {
        "experiment_commit": git("rev-parse", "HEAD", cwd=REPO),
        "engine_head": git("rev-parse", "HEAD"),
        "engine_freeze_v2": git("rev-parse", "freeze-v2^{commit}"),
        "rails_version": "rails-0.15",
        "rails_sha256_git_archive": rail_digest(),
        "python": sys.version,
        "package_lock": str(lock),
        "package_lock_sha256": sha256(lock),
        "sandbox_level": detect(),
        "host": {"system": platform.platform(), "machine": platform.machine(), "hostname": platform.node()},
        "run_date": date.today().isoformat(),
        "model_calls": 0,
        "provider": "scripted fake",
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty output folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load((HERE / "flaws.yaml").read_text(encoding="utf-8"))
    manifest_doc = yaml.safe_load((ENGINE / "benchmarks/manifest.yaml").read_text(encoding="utf-8"))
    manifest = manifest_doc if isinstance(manifest_doc, list) else manifest_doc["datasets"]
    rows = []
    for case in spec["cases"]:
        for variant in ("flaw", "control"):
            result = run_one(case, variant, out, manifest)
            rows.append(result)
            write_matrix(rows, out)
            print(json.dumps({k: result[k] for k in ["flaw", "outcome", "check", "elapsed_s"]}), flush=True)
    write_matrix(rows, out)
    write_environment(out)
    crashes = [r for r in rows if r.get("engine_crash")]
    summary = {
        "rows": len(rows),
        "controls_passed": sum(r["outcome"] == "control_passed" for r in rows),
        "controls_total": sum(r["flaw"].endswith("/control") for r in rows),
        "undetected": [r["flaw"] for r in rows if r["outcome"] == "undetected"],
        "engine_crashes": [r["flaw"] for r in crashes],
        "elapsed_s": round(sum(r["elapsed_s"] for r in rows), 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
