import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in proto:
        raise KeyError(f"missing '{k}'; available top-level keys: {sorted(proto)}")
targets = list(proto["targets"] or [])
excluded = list(proto["excluded_columns"] or [])
group = proto.get("group_column")

df = pd.read_parquet(inp["optical_digits_uci"])
print(f"rows={len(df)} cols={df.shape[1]}")

blocked = set(targets) | set(excluded) | ({group} if group else set())
feat = [c for c in df.columns if c not in blocked]

PH = ["?", "-", "NA", "N/A"]
ph_cols, kept = [], []
for c in feat:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        d = s.dropna()
        if len(s) > 1 and d.nunique() == len(d) == len(s) and (
                d.is_monotonic_increasing or d.is_monotonic_decreasing):
            continue
        kept.append(c)
        continue
    st = s.astype("string").str.strip()
    if st.isin(PH + [""]).any():
        ph_cols.append(c)
    if s.nunique(dropna=True) > 0.05 * len(df):
        continue
    kept.append(c)

if ph_cols:
    print(f"placeholder tokens {PH} found in {len(ph_cols)} text columns")

num_rows, cat_rows = [], []
for c in kept:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num_rows.append({"column": c, "kind": "numeric", "mean": s.mean(), "sd": s.std(),
                         "min": s.min(), "median": s.median(), "max": s.max(),
                         "n_missing": int(s.isna().sum())})
    else:
        vc = s.astype("string").value_counts(dropna=True, normalize=True)
        cat_rows.append({"column": c, "kind": "categorical",
                         "n_levels": int(s.nunique(dropna=True)),
                         "top_share": float(vc.iloc[0]) if len(vc) else np.nan,
                         "n_missing": int(s.isna().sum())})

summary = pd.DataFrame(num_rows + cat_rows)
for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    if col in summary.columns:
        summary[col] = summary[col].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)
n_rows_total = len(summary)
summary = summary.head(50)
mlca_runtime.save_table(summary, "distributions.csv",
                        caption="Per-feature summary (numeric and categorical statistics); no data rows.")

n_numeric, n_categorical = len(num_rows), len(cat_rows)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"numeric_features={n_numeric} categorical_features={n_categorical}")
print(f"summary_rows_written={len(summary)} of {n_rows_total} feature rows")
