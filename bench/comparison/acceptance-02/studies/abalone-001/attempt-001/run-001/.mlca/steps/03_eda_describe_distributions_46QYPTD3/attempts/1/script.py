import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
gcol = protocol.get("group_column")
task = protocol["task"]
df = pd.read_parquet(inputs["abalone_uci"])

feature_cols = []
for c in df.columns:
    if c in targets or c in excluded or c == gcol:
        continue
    s = df[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=True) == len(df):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(df) and (s.max() - s.min()) == len(df) - 1:
        continue
    feature_cols.append(c)

types = {}
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        types[c] = "datetime"
    elif pd.api.types.is_numeric_dtype(s):
        types[c] = "numeric"
    else:
        n = s.nunique(dropna=True)
        types[c] = "categorical" if (n <= 20 or n <= 0.05 * len(df)) else "text"

rows = []
def add(name, typ, s, group):
    if typ == "numeric":
        rows.append({"variable": name, "type": typ, "group": group, "n": int(s.notna().sum()),
                     "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(),
                     "max": s.max(), "n_missing": int(s.isna().sum()),
                     "n_levels": np.nan, "top_share": np.nan})
    else:
        vc = s.dropna().value_counts(normalize=True)
        rows.append({"variable": name, "type": typ, "group": group, "n": int(s.notna().sum()),
                     "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
                     "max": np.nan, "n_missing": int(s.isna().sum()),
                     "n_levels": int(s.nunique(dropna=True)),
                     "top_share": float(vc.iloc[0]) if len(vc) else np.nan})

tgt = targets[0]
bins = None
if pd.api.types.is_numeric_dtype(df[tgt]) and df[tgt].nunique() > 4:
    try:
        bins = pd.qcut(df[tgt].rank(method="first"), 4, labels=False, duplicates="drop")
    except Exception:
        bins = None

for c in feature_cols:
    typ = types[c]
    if typ == "datetime":
        d = pd.to_datetime(df[c], errors="coerce")
        add(c + "_elapsed_days", "numeric", (d - d.min()).dt.days.astype(float), "all")
        continue
    add(c, typ, df[c], "all")
    if bins is not None and len(rows) < 40:
        for b in sorted(pd.Series(bins).dropna().unique()):
            add(c, typ, df[c][bins == b], f"{tgt}_q{int(b) + 1}")

res = pd.DataFrame(rows).head(50)[["variable", "type", "group", "n", "mean", "sd", "min",
                                    "median", "max", "n_missing", "n_levels", "top_share"]]
mlca_runtime.save_table(res.round(4), "distributions.csv",
                        caption="Distribution summaries by variable and target quartile.")

n_numeric = sum(1 for c in feature_cols if types[c] == "numeric")
n_categorical = sum(1 for c in feature_cols if types[c] in ("categorical", "text"))
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"features: n_numeric={n_numeric}, n_categorical={n_categorical}")
print(f"distribution rows: {len(res)}")
