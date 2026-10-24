import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
if "targets" not in protocol:
    raise ValueError(f"Missing 'targets'. Available keys: {list(protocol)}")
targets = list(protocol["targets"])

df = pd.read_parquet(inputs["discovery"])

duration_col = next((c for c in targets if any(w in c.lower() for w in ("month", "time", "duration"))), None)
event_col = next((c for c in targets if any(w in c.lower() for w in ("status", "event", "dead", "censor"))), None)
if duration_col is None or event_col is None:
    raise ValueError(f"Cannot infer duration/event from targets={targets}")

ph = {"?", "-", "na", "n/a", "", "nan", "none"}
ph_cols = []
for c in df.columns:
    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype(str).str.strip().str.lower()
        if s.isin(ph).any():
            ph_cols.append(c)
print(f"Placeholders in {len(ph_cols)} column(s): {ph_cols}")

duration = pd.to_numeric(df[duration_col], errors="coerce")
raw = df[event_col]
if pd.api.types.is_numeric_dtype(raw):
    event = (raw.astype(float) > 0).astype(float)
    event[raw.isna()] = np.nan
else:
    s = raw.astype(str).str.strip().str.lower()
    s = s.where(~s.isin(ph), np.nan)
    event = s.map({
        "dead": 1, "deceased": 1, "1": 1, "1:dead": 1, "1:deceased": 1,
        "alive": 0, "living": 0, "0": 0, "0:alive": 0, "0:living": 0,
        "true": 1, "yes": 1, "event": 1,
        "false": 0, "no": 0, "censored": 0
    })
    event = pd.to_numeric(event, errors="coerce")

valid = duration.notna() & event.notna() & (duration >= 0)
duration = duration[valid].to_numpy()
event = event[valid].to_numpy()
total = len(duration)
n_event = int((event == 1).sum())
n_cens = int((event == 0).sum())

def reverse_km_median(times, events):
    if len(times) == 0:
        return np.nan
    rev_event = 1.0 - events
    order = np.argsort(times)
    t = times[order]
    d = rev_event[order]
    uniq_t, idx = np.unique(t, return_index=True)
    n_at_risk = len(t) - idx
    d_uniq = np.add.reduceat(d, idx)
    surv = np.cumprod(1.0 - d_uniq / n_at_risk)
    below = np.where(surv <= 0.5)[0]
    return float(uniq_t[below[0]]) if len(below) else np.nan

median_fu = reverse_km_median(duration, event)

def sig(x):
    return float(f"{x:.4g}") if pd.notna(x) else np.nan

out = pd.DataFrame([
    {"status": "event", "n": n_event,
     "share": sig(n_event / total) if total else np.nan,
     "median_follow_up": sig(median_fu)},
    {"status": "censored", "n": n_cens,
     "share": sig(n_cens / total) if total else np.nan,
     "median_follow_up": sig(median_fu)},
])

mlca_runtime.save_table(out, "target_counts.csv",
                        caption="Event/censoring counts and reverse-KM median follow-up")

mlca_runtime.record_metric("n_events", float(n_event), "train")
mlca_runtime.record_metric("n_censored", float(n_cens), "train")
if pd.notna(median_fu):
    mlca_runtime.record_metric("median_follow_up", float(median_fu), "train")

print(f"Rows: {total}, events: {n_event}, censored: {n_cens}")
print(f"Median follow-up (reverse KM): {sig(median_fu)}")
