import mlca_runtime
import pickle, json
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary_metric = task["primary_metric"]
time_col, event_col = protocol["targets"][0], protocol["targets"][1]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def to_binary(s):
    digits = s.astype(str).str.extract(r'(\d+)')[0]
    nums = pd.to_numeric(digits, errors='coerce')
    if nums.notna().all():
        return nums.astype(int)
    m = {}
    for v in s.dropna().unique():
        sv = str(v).lower()
        if 'dead' in sv or 'deceased' in sv or sv in ('1',): m[v] = 1
        elif 'alive' in sv or 'living' in sv or sv in ('0',): m[v] = 0
    return s.map(m).astype(int)

val = val.dropna(subset=[time_col, event_col]).copy()
val['_event'] = to_binary(val[event_col])
val['_time'] = pd.to_numeric(val[time_col], errors='coerce')
val = val.dropna(subset=['_time'])

def load(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

def risk(model, df):
    if isinstance(model, Pipeline):
        last = model.steps[-1][1]
        if isinstance(last, CoxPHFitter):
            return last.predict_partial_hazard(model[:-1].transform(df)).values.ravel()
        if hasattr(model, "predict_proba"):
            return model.predict_proba(df)[:, 1]
        return np.asarray(model.predict(df)).ravel()
    if isinstance(model, CoxPHFitter):
        return model.predict_partial_hazard(df).values.ravel()
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df)[:, 1]
    return np.asarray(model.predict(df)).ravel()

cands = {
    "cox": "07_model_train_cox_DV6BJ4W8/model.pkl",
    "alternative": "08_model_train_alternative_V100ZYV3/alternative_model.pkl",
}
rows, models = [], {}
for name, key in cands.items():
    try:
        m = load(key)
        models[name] = m
        r = risk(m, val)
        ci = float(concordance_index(val['_time'].values, -r, val['_event'].values))
        rows.append({"model": name, "metric": primary_metric, "value": round(ci, 4)})
        mlca_runtime.record_metric(primary_metric, ci, split="val")
    except Exception as e:
        print(f"{name} failed: {type(e).__name__}")

mlca_runtime.save_table(pd.DataFrame(rows), "validation_metrics.csv")
if rows:
    best = max(rows, key=lambda x: x["value"])
    mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")
    mlca_runtime.record_fact_candidate("best_val_c_index", best["value"], source="validation_metrics.csv")
    print(f"best={best['model']} c_index={best['value']:.4f}")
else:
    print("no candidates produced metrics")

cox = models.get("cox")
if cox is not None:
    cox_fit = cox.steps[-1][1] if isinstance(cox, Pipeline) else cox
    if isinstance(cox_fit, CoxPHFitter):
        try:
            Xt = train.dropna(subset=[time_col, event_col]).copy()
            if isinstance(cox, Pipeline):
                Xt = cox[:-1].transform(Xt)
            ph = proportional_hazard_test(cox_fit, Xt)
            print(f"PH Schoenfeld min p={ph.summary['p'].min():.3g} across {len(ph.summary)} covariates")
        except Exception as e:
            print(f"PH test skipped: {type(e).__name__}")
