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

import socket

import numpy as np
import pandas as pd
import yaml

PROVIDER_KEYS = [
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_API_KEY",
    "HF_TOKEN",
]
NETWORK_ATTEMPTS: list[str] = []


def isolate(out: Path) -> None:
    """Empty MLCA user config (no navigator or provider from ~/.config), no keys, offline uv.

    Mirrors the isolated_user fixture in MLCA's tests/conftest.py. The 24 October partial run
    used the real user config, whose navigator called openrouter/free.
    """
    base = out / ".isolated"
    for name, folder in [("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "userdata")]:
        (base / folder).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(base / folder)
    os.environ["UV_OFFLINE"] = "1"
    for key in PROVIDER_KEYS:
        os.environ.pop(key, None)


def block_network() -> None:
    """Refuse every non-Unix socket connection in this process; worker scripts run in the network-free sandbox."""

    def refuse(self, address, *args, **kwargs):
        if self.family == socket.AF_UNIX:
            return original(self, address, *args, **kwargs)
        NETWORK_ATTEMPTS.append(repr(address))
        raise ConnectionRefusedError(f"safeguard test: network blocked ({address!r})")

    original = socket.socket.connect
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse


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
    """Return a clean script matching the selected action's declared output contract.

    The simple binary-classification operations below follow the text in MLCA's
    synthetic_fake_llm fixture; the task-aware extensions keep the same fit_record,
    save_model, scoring and row-output pattern for the other registered templates.
    """
    outputs = [o.name for o in action.outputs]
    output = outputs[0]
    event = action.event
    if action.id == "audit_schema":
        return '''import json, pandas as pd, mlca_runtime as m
i=m.inputs(); path=next(v for k,v in i.items() if k != "_protocol")
df=pd.read_csv(path)
schema=pd.DataFrame({"name":df.columns.astype(str),"dtype":df.dtypes.astype(str),
 "n_unique":[int(df[c].nunique(dropna=True)) for c in df],
 "n_missing":[int(df[c].isna().sum()) for c in df]})
schema["pct_missing"]=schema["n_missing"] / max(len(df),1) * 100
m.save_table(schema,"schema.json")
m.record_fact_candidate("n_rows",int(len(df)),source="schema.json")
m.record_fact_candidate("n_columns",int(df.shape[1]),source="schema.json")
m.record_fact_candidate("missing.total",int(df.isna().sum().sum()),source="schema.json")
'''
    if action.id == "count_target":
        return f'''import json, pandas as pd, mlca_runtime as m
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); path=next(v for k,v in i.items() if k != "_protocol")
df=pd.read_csv(path); rows=[]
for target in p["targets"]:
 for label,count in df[target].value_counts(dropna=False).items():
  rows.append({{"target":target,"class":str(label),"n":int(count),"share":float(count/len(df))}})
m.save_table(pd.DataFrame(rows,columns=["target","class","n","share"]),"target_counts.csv")
'''
    if action.id == "describe_distributions":
        return '''import json, pandas as pd, numpy as np, mlca_runtime as m
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); path=next(v for k,v in i.items() if k != "_protocol")
df=pd.read_csv(path); numeric=df.select_dtypes(include="number"); categorical=df.select_dtypes(exclude="number"); rows=[]
for c in numeric:
 s=numeric[c]; rows.append({"column":c,"kind":"numeric","mean":s.mean(),"sd":s.std(),"min":s.min(),"median":s.median(),"max":s.max(),"n_missing":int(s.isna().sum())})
for c in categorical:
 s=categorical[c]; counts=s.value_counts(normalize=True,dropna=True); rows.append({"column":c,"kind":"categorical","n_levels":int(s.nunique(dropna=True)),"top_level_share":float(counts.iloc[0]) if len(counts) else np.nan,"n_missing":int(s.isna().sum())})
m.save_table(pd.DataFrame(rows),"distributions.csv")
m.record_fact_candidate("n_numeric",int(len(numeric.columns)),source="distributions.csv")
m.record_fact_candidate("n_categorical",int(len(categorical.columns)),source="distributions.csv")
'''
    if action.id == "fit_preprocessing":
        return f'''import json, pandas as pd, mlca_runtime as m
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
drop=set(p["targets"])|set(p.get("excluded_columns",[]))|set(p["task"].get("time_columns",[]))|({{p["group_column"]}} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); nums=X.select_dtypes(include="number").columns.tolist(); cats=[c for c in X if c not in nums]
parts=[]
if nums: parts.append(("numeric",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),nums))
if cats: parts.append(("categorical",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),cats))
obj=ColumnTransformer(parts); fitted=m.fit_record(obj,X,"train"); m.save_model(fitted,"preprocessor.pkl")
'''
    if action.id == "select_features":
        return '''import json, pandas as pd, mlca_runtime as m
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
drop=set(p["targets"])|set(p.get("excluded_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]).select_dtypes(include="number")
selector=m.fit_record(make_pipeline(SimpleImputer(strategy="median"),SelectKBest(mutual_info_classif,k=min(20,X.shape[1]))),X,"train",y=tr[p["targets"][0]])
m.save_model(selector,"selector.pkl")
'''
    if action.id == "retain_features":
        return '''import json, pandas as pd, mlca_runtime as m
from sklearn.preprocessing import FunctionTransformer
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
drop=set(p["targets"])|set(p.get("excluded_columns",[]))|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); selector=m.fit_record(FunctionTransformer(),X,"train")
m.save_model(selector,"selector_all.pkl")
'''
    if event == "model":
        if template == "multilabel_classification":
            estimator = "MultiOutputClassifier(make_pipeline(preprocessor, LogisticRegression(max_iter=2000)))"
            imports = "from sklearn.multioutput import MultiOutputClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler"
            Xbuild = '''targets=p["targets"]; drop=set(targets)|set(p.get("excluded_columns",[]))|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); nums=X.select_dtypes(include="number").columns.tolist(); cats=[c for c in X if c not in nums]
parts=[]
if nums: parts.append(("numeric",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),nums))
if cats: parts.append(("categorical",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),cats))
preprocessor=ColumnTransformer(parts); y=tr[targets]'''
        elif template == "regression":
            estimator = "make_pipeline(preprocessor, LinearRegression())"
            imports = "from sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler"
            Xbuild = '''targets=p["targets"]; drop=set(targets)|set(p.get("excluded_columns",[]))|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); nums=X.select_dtypes(include="number").columns.tolist(); cats=[c for c in X if c not in nums]
parts=[]
if nums: parts.append(("numeric",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),nums))
if cats: parts.append(("categorical",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),cats))
preprocessor=ColumnTransformer(parts); y=tr[targets] if len(targets)>1 else tr[targets[0]]'''
        else:
            estimator = "make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))"
            imports = "from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression"
            Xbuild = '''targets=p["targets"]; drop=set(targets)|set(p.get("excluded_columns",[]))|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); y=tr[targets[0]]'''
        return f'''import json, pandas as pd, mlca_runtime as m
{imports}
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
{Xbuild}
model=m.fit_record({estimator},X,"train",y=y); m.save_model(model,{output!r})
'''
    if event in {"validate", "test"}:
        part = "val" if event == "validate" else "test"
        metric_default = "roc_auc" if template == "tabular_classification" else ("macro_f1" if template == "multilabel_classification" else "rmse")
        row_output = next((o.name for o in action.outputs if o.kind == "data"), None)
        if template == "tabular_classification":
            score_code = '''from sklearn.metrics import roc_auc_score
proba=model.predict_proba(X); classes=list(model.classes_); positive=1 if 1 in classes else classes[-1]; j=classes.index(positive)
pred=model.predict(X); value=float(roc_auc_score(df[targets[0]],proba[:,j]))'''
            prediction_columns = '''"y_true":df[targets[0]].to_numpy(),"y_pred":pred,
 **{f"proba__{c}":proba[:,j] for j,c in enumerate(classes)}'''
        elif template == "multilabel_classification":
            score_code = '''from sklearn.metrics import f1_score
pred=model.predict(X); value=float(np.mean([f1_score(df[t],pred[:,j],average="macro") for j,t in enumerate(targets)]))'''
            prediction_columns = '''"y_true__"+targets[0]:df[targets[0]].to_numpy(),"y_pred__"+targets[0]:pred[:,0]'''
        else:
            score_code = '''from sklearn.metrics import mean_squared_error
pred=model.predict(X); actual=df[targets].to_numpy() if len(targets)>1 else df[targets[0]].to_numpy()
value=float(mean_squared_error(actual,pred)**0.5)'''
            prediction_columns = '''"y_true__"+targets[0]:actual[:,0] if actual.ndim>1 else actual,"y_pred__"+targets[0]:pred[:,0] if getattr(pred,"ndim",1)>1 else pred'''
        row_save = f'''m.save_row_outputs(pd.DataFrame({{"row_id":df.index,"partition":"test",{prediction_columns}}}),{row_output!r})''' if row_output else ""
        return f'''import json, pickle, numpy as np, pandas as pd, mlca_runtime as m
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); df=pd.read_parquet(i[{part!r}]); targets=p["targets"]
drop=set(targets)|set(p.get("excluded_columns",[]))|set(p["task"].get("time_columns",[]))|({{p["group_column"]}} if p.get("group_column") else set())
model=pickle.load(open(i["latest_model"],"rb")); names=getattr(model,"feature_names_in_",None)
X=df[list(names)] if names is not None else df.drop(columns=[c for c in drop if c in df])
{score_code}
metric=p["task"].get("primary_metric") or {metric_default!r}; m.record_metric(metric,value,{part!r})
m.save_table(pd.DataFrame({{"metric":[metric],"value":[value]}}),{output!r})
{row_save}
'''
    if event == "interpret":
        return f'''import json, pickle, pandas as pd, mlca_runtime as m
i=m.inputs(); model=pickle.load(open(i["latest_model"],"rb")); final=model[-1] if hasattr(model,"steps") else model
names=list(getattr(final,"feature_names_in_",[])); values=getattr(final,"coef_",getattr(final,"feature_importances_",[]))
import numpy as np
values=np.asarray(values); values=np.abs(values[0]) if values.ndim>1 else np.abs(values)
if len(names)!=len(values): names=[f"feature_{{j}}" for j in range(len(values))]
m.save_table(pd.DataFrame({{"feature":names,"importance_mean":values,"importance_sd":np.zeros(len(values))}}).head(15),{output!r})
'''
    return "import mlca_runtime as m\n"


def build_responses(case: dict, variant: str, workflow) -> list[dict]:
    script_field = "planted_script" if variant == "flaw" else "control_script"
    result = []
    # Template actions are labelled primary or supporting; the fixture matches the
    # Worker role plus action_id, which is the request context used by MLCA.
    for action in workflow.actions.values():
        if action.builtin:
            continue
        text = protocol_script(action, case["template"])
        # Clean surrounding steps must not independently introduce the planted flaw.
        manifest = yaml.safe_load((ENGINE / "benchmarks/manifest.yaml").read_text())
        entries = manifest if isinstance(manifest,list) else manifest["datasets"]
        meta = next((row for row in entries if row["id"] == case["table"]), {})
        clean_exclusions = sorted(set(meta.get("leakage_columns",[])) | set(meta.get("identifier_columns",[])))
        text = text.replace('set(p.get("excluded_columns",[]))', 'set(p.get("excluded_columns",[]))|set(' + repr(clean_exclusions) + ')')
        # The planted step runs the case's own script in both variants: the flaw, or its correct control.
        # F5 and F6 plant the flaw in the split parameters instead (see run_one).
        if action.id == case["action"] and case.get(script_field) and case["id"] not in {"F5", "F6"}:
            text = case[script_field]
        result.append({"match": {"role": "worker", "action_id": action.id}, "text": f"```python\n{text}\n```"})
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
    # Worker only: with no navigator role, MLCA proposes its deterministic fallback options.
    config = {
        "providers": {"offline": {"kind": "fake", "script": str(script_path), "request_cost_bound_usd": 0}},
        "roles": {"worker": {"provider": "offline", "model": "scripted"}},
        "sandbox": {"allow_level_b": False},
        "budgets": {"worker_max_attempts": 1},
    }
    brief = dict(case["brief"])
    root = initialise(project, [data], case["template"], brief, config)
    start = time.monotonic()
    exception = None
    reached_report = False
    requested_split_method = None
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
            if event == "test" and not session.state.test_unlocked:
                session.unlock_test("UNLOCK TEST", "Planned test-scoring step")
            options = session.options()
            action_id = case["action"] if event == next((a.event for a in workflow.actions.values() if a.id == case["action"]), "") else None
            option = next((o for o in options if o.get("action_id") == action_id), None) if action_id else None
            preferred = {"validate": "score_validation", "test": "score_test"}.get(event)
            if option is None and preferred:
                option = next((o for o in options if o.get("action_id") == preferred), None)
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
            if event == "split":
                requested_split_method = params.get("method")
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
        split_method = (session.state.split or {}).get("parameters", {}).get("method")
        last_action = session.state.steps[-1].get("action_id") if session.state.steps else None
        try:
            scored = score(project, expected)
        except Exception as exc:
            scored = {"leakage_clean": None, "error": type(exc).__name__}
    elapsed = round(time.monotonic() - start, 3)
    records_data = [json.loads(line) for line in (project / ".mlca/log/events.jsonl").read_text().splitlines()]
    fit_events = [r for r in records_data if r.get("type") == "StepCompleted" and r.get("payload", {}).get("fits")]
    target_step_status = target_action.get("status") if target_action else None
    execution_errors = "\n".join(r.get("payload", {}).get("stderr", "") for r in records_data if r.get("type") == "RunFinished" and r.get("payload", {}).get("exit_status"))
    check = ""
    message = ""
    if variant == "control":
        outcome = "control_passed" if reached_report else "control_failed"
        check = "clean script / declared split control"
        message = "reached report" if reached_report else "clean split completed" if split_step else (exception or {}).get("category", "")
    elif case["id"] in {"F5", "F6"}:
        expected_method = "group" if case["id"] == "F5" else "time"
        if split_step and split_method == expected_method:
            outcome, check, message = "refused", "frozen split metadata enforcement", f"requested {requested_split_method}; engine recorded {split_method}"
        else:
            outcome, check, message = "undetected", "", f"engine recorded {split_method or 'no split'}"
    elif exception:
        category = exception.get("category", "")
        lower_message = exception.get("message", "").lower()
        if any(term in category for term in ["fit_rule", "static_check", "test_lock", "MlcaUserError"]) or "fit rows must match the frozen training index" in lower_message or "FitRuleError:" in execution_errors:
            outcome, check = "refused", "fit_record enforcement" if "FitRuleError:" in execution_errors else category
        elif case["id"] in {"F2", "F7"} and ("test" in lower_message or "InputKeyError:" in execution_errors):
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
        "run_path": str(project.relative_to(out)),
        "reached_report": reached_report,
        "target_step_status": target_step_status,
        "exception": exception,
        "warnings": warnings,
        "leakage": scored,
        "split_method": split_method,
        "requested_split_method": requested_split_method,
        "fit_events": len(fit_events),
        "elapsed_s": elapsed,
        "engine_crash": bool(exception and not any(r.get("type") == "StepFailed" for r in records_data)),
    }
    (project.parent / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Unexpected failures must stop the official test; expected safeguard refusals
    # at the planted action are the measured outcomes, not harness errors.
    if exception and (variant == "control" or outcome != "refused" or last_action != case["action"]):
        write_environment(out)
        raise RuntimeError(f"Stopped at {case_id}: {exception['category']}: {exception['message']}; action={last_action}")
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
        "provider": "scripted fake (worker only; no navigator role; MLCA fallback options)",
        "isolation": "empty XDG_CONFIG_HOME/XDG_DATA_HOME, provider keys removed, UV_OFFLINE=1, non-Unix sockets refused",
        "network_attempts": NETWORK_ATTEMPTS,
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", help="Run only these case IDs (harness checks); the official run uses all")
    parser.add_argument("--variants", nargs="*", default=["flaw", "control"], choices=["flaw", "control"])
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty output folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    isolate(out)
    block_network()
    spec = yaml.safe_load((HERE / "flaws.yaml").read_text(encoding="utf-8"))
    manifest_doc = yaml.safe_load((ENGINE / "benchmarks/manifest.yaml").read_text(encoding="utf-8"))
    manifest = manifest_doc if isinstance(manifest_doc, list) else manifest_doc["datasets"]
    rows = []
    cases = [c for c in spec["cases"] if not args.cases or c["id"] in args.cases]
    for case in cases:
        for variant in args.variants:
            result = run_one(case, variant, out, manifest)
            if NETWORK_ATTEMPTS:
                result["outcome"] = "harness_error"
                result["message"] = "network attempt blocked: " + "; ".join(NETWORK_ATTEMPTS)
            rows.append(result)
            write_matrix(rows, out)
            print(json.dumps({k: result[k] for k in ["flaw", "outcome", "check", "elapsed_s"]}), flush=True)
            if NETWORK_ATTEMPTS:
                raise SystemExit("Stopped: a network connection was attempted (blocked). The harness, not MLCA, must be fixed.")
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
