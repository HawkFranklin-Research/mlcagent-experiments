import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing protocol key {key!r}; available: {list(proto.keys())}")

targets = proto["targets"]
task = proto["task"]
event_col = targets[0] if len(targets) == 1 else "Overall Survival Status"

duration_candidates = list(task.get("time_columns") or [])
duration_candidates += ["Overall Survival (Months)", "Overall Survival (Months)"]
duration_candidates += [c for c in proto.get("duration_columns", []) or []]

df = pd.read_parquet(inputs["discovery"])
if event_col not in df.columns:
    raise KeyError(f"Event column {event_col!r} not in data columns: {list(df.columns)}")

duration_col = next((c for c in duration_candidates if c in df.columns), None)
if duration_col is None:
    raise KeyError(f"Duration column not found among {duration_candidates}; data has: {list(df.columns)}")

def event_indicator(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).where(s.notna())
    t = s.astype("string").str.strip().str.lower()
    ev = t.str.contains(r"dead|deceased|event|yes|^1($|:)", regex=True, na=False)
    cens = t.str.contains(r"alive|living|censored|no|^0($|:)", regex=True, na=False)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[ev] = 1.0
    out[cens & ~ev] = 0.0
    return out

event = event_indicator(df[event_col])
duration = pd.to_numeric(df[duration_col], errors="coerce")
valid = event.notna() & duration.notna()
event = event[valid].astype(int)
duration = duration[valid].astype(float)
total = int(len(event))
if total == 0:
    raise ValueError("No rows with both event status and duration.")

n_event = int(event.sum())
n_cens = int(total - n_event)

def reverse_km_median(times, event_obs):
    ev = 1 - event_obs
    data = pd.DataFrame({"time": times, "ev": ev}).sort_values("time")
    at_risk = len(data)
    surv = 1.0
    for t, grp in data.groupby("time", sort=True):
        d = int(grp["ev"].sum())
        n = len(grp)
        if at_risk > 0:
            surv *= (1 - d / at_risk)
        if surv <= 0.5:
            return float(t)
        at_risk -= n
    return np.nan

med = reverse_km_median(duration.values, event.values)

def sig4(x):
    return float(f"{x:.4g}") if pd.notna(x) and np.isfinite(x) else np.nan

med_val = sig4(med)
out = pd.DataFrame([
    {"status": "event", "n": n_event, "share": sig4(n_event / total), "median_follow_up": med_val},
    {"status": "censored", "n": n_cens, "share": sig4(n_cens / total), "median_follow_up": med_val},
])
mlca_runtime.save_table(out, "target_counts.csv", caption="Status counts and reverse Kaplan-Meier median follow-up")

print(f"rows={total}, events={n_event}, censored={n_cens}")
print(f"median_follow_up={med:.4g}" if np.isfinite(med) else "median_follow_up=NA")
