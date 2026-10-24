import json
import numpy as np
import pandas as pd
import mlca_runtime
from lifelines import KaplanMeierFitter

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
if "targets" not in protocol:
    raise ValueError(f"Missing 'targets'. Available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
task = protocol.get("task", {})

df = pd.read_parquet(inputs["discovery"])
df = df.copy()

# Identify duration and event columns from protocol/task/name or numeric type.
def find_col(keys, words, numeric=False):
    for k in keys:
        if k in task:
            return task[k]
        if k in protocol:
            return protocol[k]
    for c in targets:
        if any(w in c.lower() for w in words):
            return c
    if numeric:
        for c in targets:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    return None

duration_col = find_col(("duration_col", "duration_column", "time_col", "time_column"),
                         ("month", "time", "duration"), numeric=True)
event_col = find_col(("event_col", "event_column", "status_col", "status_column"),
                      ("status", "event", "dead", "censor"))
if duration_col is None or event_col is None:
    raise ValueError(f"Could not identify duration/event columns among targets={targets}")

# Report and normalize placeholder tokens in the two outcome columns.
ph_tokens = {"?", "-", "na", "n/a", ""}
ph_cols = []
for c in (duration_col, event_col):
    if df[c].dtype == object:
        s = df[c].astype(str).str.strip().str.lower()
        if s.isin(ph_tokens).any():
            ph_cols.append(c)
        df[c] = df[c].astype(str).str.strip()
        df.loc[s.isin(ph_tokens), c] = np.nan
print(f"Placeholders found in {len(ph_cols)} outcome column(s): {ph_cols}")

duration = pd.to_numeric(df[duration_col], errors="coerce")
raw_event = df[event_col]
if pd.api.types.is_numeric_dtype(raw_event):
    event = (raw_event.astype(float) > 0).astype(float)
    event[raw_event.isna()] = np.nan
else:
    s = raw_event.astype(str).str.strip().str.lower()
    s = s.replace({"nan": np.nan, "none": np.nan, "?": np.nan, "-": np.nan,
                   "na": np.nan, "n/a": np.nan, "": np.nan})
    event = s.map({"dead": 1, "deceased": 1, "1": 1, "true": 1, "yes": 1, "event": 1,
                   "alive": 0, "living": 0, "0": 0, "false": 0, "no": 0, "censored": 0})
    event = pd.to_numeric(event, errors="coerce")

valid = duration.notna() & event.notna() & (duration >= 0)
duration = duration[valid]
event = event[valid]
total = len(duration)
n_event = int((event == 1).sum())
n_cens = int((event == 0).sum())

median_fu = np.nan
if total > 0:
    km_df = pd.DataFrame({duration_col: duration, "_reverse_event": 1 - event})
    kmf = KaplanMeierFitter()
    mlca_runtime.fit_record(kmf, km_df, "discovery", duration_col=duration_col,
                            event_col="_reverse_event")
    median_fu = kmf.median_survival_time_
    if not np.isfinite(median_fu):
        median_fu = np.nan

def sig(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

out = pd.DataFrame([
    {"status": "event", "n": n_event, "share": sig(n_event / total) if total else np.nan,
     "median_follow_up": sig(median_fu)},
    {"status": "censored", "n": n_cens, "share": sig(n_cens / total) if total else np.nan,
     "median_follow_up": sig(median_fu)},
])
mlca_runtime.save_table(out, "target_counts.csv",
                        caption="Event/censoring counts and reverse-KM median follow-up")

mlca_runtime.record_metric("n_events", float(n_event), "discovery")
mlca_runtime.record_metric("n_censored", float(n_cens), "discovery")
if pd.notna(median_fu):
    mlca_runtime.record_metric("median_follow_up", float(median_fu), "discovery")

print(f"Rows: {total}, events: {n_event}, censored: {n_cens}")
print(f"Median follow-up (reverse KM): {sig(median_fu)}")
