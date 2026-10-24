import mlca_runtime
import json
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise KeyError(f"Missing protocol key {key!r}; available: {list(proto.keys())}")

targets = proto["targets"]
event_col = targets[0] if len(targets) == 1 else "Overall Survival Status"
duration_col = "Overall Survival (Months)"

df = pd.read_parquet(inputs["discovery"])
if event_col not in df.columns:
    raise KeyError(f"Event column {event_col!r} not in data columns: {list(df.columns)}")
if duration_col not in df.columns:
    raise KeyError(f"Duration column {duration_col!r} not in data columns: {list(df.columns)}")

def event_indicator(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).where(s.notna())
    t = s.astype("string").str.strip().str.lower()
    ev = t.str.contains(r"dead|deceased|event|yes|^1($|:)|^1\b", regex=True, na=False)
    cens = t.str.contains(r"alive|living|censored|no|^0($|:)|^0\b", regex=True, na=False)
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
share_event = n_event / total
share_cens = n_cens / total

km = KaplanMeierFitter()
km.fit(duration, event_observed=1 - event)
med = float(km.median_survival_time_)
if not np.isfinite(med):
    med = np.nan

out = pd.DataFrame([
    {"status": "event", "n": n_event, "share": round(share_event, 4), "median_follow_up": round(med, 4) if np.isfinite(med) else np.nan},
    {"status": "censored", "n": n_cens, "share": round(share_cens, 4), "median_follow_up": round(med, 4) if np.isfinite(med) else np.nan},
])
mlca_runtime.save_table(out, "target_counts.csv", caption="Status counts and reverse Kaplan-Meier median follow-up")

mlca_runtime.record_metric("n_events", float(n_event), split="discovery")
mlca_runtime.record_metric("n_censored", float(n_cens), split="discovery")
if np.isfinite(med):
    mlca_runtime.record_metric("median_follow_up", med, split="discovery")

print(f"rows={total}, events={n_event}, censored={n_cens}")
print(f"median_follow_up={med:.4g}" if np.isfinite(med) else "median_follow_up=NA")
