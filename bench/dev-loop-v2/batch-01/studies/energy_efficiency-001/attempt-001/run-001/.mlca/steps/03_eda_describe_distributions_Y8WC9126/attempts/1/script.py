import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["energy_efficiency_uci"])

for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"Missing protocol key '{k}'. Available: {sorted(proto)}")
targets = proto["targets"]
excluded = list(proto["excluded_columns"])

# RL-0023 placeholder scan and whitespace strip
ph = {"", "?", "-", "NA", "N/A"}
str_cols = [c for c in df.columns
            if df[c].dtype == object or isinstance(df[c].dtype, pd.StringDtype)]
ph_cols = []
for c in str_cols:
    s = df[c].astype("string").str.strip()
    mask = s.isin(ph) & s.notna()
    if mask.any():
        ph_cols.append(c)
    df[c] = s.mask(s.isin(ph), pd.NA)
print(f"placeholders {sorted(ph)} in {len(ph_cols)}/{len(str_cols)} string columns")

num_cols = [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]

def sig4(x):
    if x is None:
        return np.nan
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(xf):
        return np.nan
    return float(f"{xf:.4g}")

rows = []
for c in num_cols:
    s = pd.to_numeric(df[c], errors="coerce")
    rows.append({"variable": c, "type": "numeric", "n": int(s.notna().sum()),
                 "n_missing": int(s.isna().sum()),
                 "mean": sig4(s.mean()), "sd": sig4(s.std()), "min": sig4(s.min()),
                 "median": sig4(s.median()), "max": sig4(s.max()),
                 "n_levels": np.nan, "top_level": np.nan, "top_share": np.nan})
for c in cat_cols + dt_cols:
    s = df[c].astype("string")
    vc = s.value_counts(dropna=True)
    n = int(s.notna().sum())
    top = str(vc.index[0]) if len(vc) else np.nan
    share = sig4(vc.iloc[0] / n) if n else np.nan
    rows.append({"variable": c, "type": "categorical", "n": n,
                 "n_missing": int(s.isna().sum()),
                 "mean": np.nan, "sd": np.nan, "min": np.nan,
                 "median": np.nan, "max": np.nan,
                 "n_levels": int(vc.shape[0]), "top_level": top, "top_share": share})

if len(rows) > 50:
    print(f"truncating {len(rows)} rows to 50")
    rows = rows[:50]
cols = ["variable", "type", "n", "n_missing", "mean", "sd", "min",
        "median", "max", "n_levels", "top_level", "top_share"]
out = pd.DataFrame(rows, columns=cols)
mlca_runtime.save_table(out, "distributions.csv", caption="Column distributions summary")

n_numeric = int((out["type"] == "numeric").sum())
n_categorical = int((out["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"n_numeric={n_numeric} n_categorical={n_categorical}")
