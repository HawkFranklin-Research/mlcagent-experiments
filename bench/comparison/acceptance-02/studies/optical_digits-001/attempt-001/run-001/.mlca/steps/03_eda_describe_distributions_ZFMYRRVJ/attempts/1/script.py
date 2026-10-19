import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
missing = [k for k in ("targets", "excluded_columns", "group_column") if k not in proto]
if missing:
    raise KeyError(f"Protocol missing {missing}; available top-level keys: {sorted(proto)}")

targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto["group_column"]
df = pd.read_parquet(inputs["optical_digits_uci"])

exclude_cols = set(targets) | set(excluded)
if group is not None:
    exclude_cols.add(group)
features = [c for c in df.columns if c not in exclude_cols]

def is_text(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

keylike = []
for c in features:
    s = df[c]
    if is_text(s) and s.nunique(dropna=True) == len(df):
        keylike.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) == len(df):
        keylike.append(c)
features = [c for c in features if c not in keylike]

work = df.copy()
found_tokens = set()
placeholder_cols = 0
for c in features:
    if is_text(work[c]):
        stripped = work[c].astype("string").str.strip()
        mask = stripped.isin(["?", "-", "NA", "N/A"]) | stripped.eq("")
        mask = mask.fillna(False)
        if mask.any():
            placeholder_cols += 1
            for tok in ["?", "-", "NA", "N/A"]:
                if (stripped == tok).fillna(False).any():
                    found_tokens.add(tok)
            if (stripped == "").fillna(False).any():
                found_tokens.add("whitespace")
            work[c] = work[c].where(~mask, np.nan)

def sig4(x):
    if pd.isna(x):
        return np.nan
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

rows = []
for i, c in enumerate(features):
    s = work[c]
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            "feature": c, "type": "numeric", "n_missing": int(s.isna().sum()),
            "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(),
            "max": s.max(), "n_levels": np.nan, "top_level_share": np.nan,
            "_priority": float(s.var()) if s.notna().sum() > 1 else 0.0, "_idx": i,
        })
    else:
        levels = int(s.nunique(dropna=True))
        vc = s.value_counts(dropna=True, normalize=True)
        top = float(vc.iloc[0]) if len(vc) else np.nan
        rows.append({
            "feature": c, "type": "categorical", "n_missing": int(s.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
            "max": np.nan, "n_levels": levels, "top_level_share": top,
            "_priority": float(levels), "_idx": i,
        })

if len(rows) > 50:
    rows = sorted(rows, key=lambda r: r["_priority"], reverse=True)[:50]
    rows = sorted(rows, key=lambda r: r["_idx"])

summary = pd.DataFrame(rows).drop(columns=["_priority", "_idx"], errors="ignore")
for col in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[col] = summary[col].map(sig4)
summary = summary[["feature", "type", "mean", "sd", "min", "median", "max",
                   "n_missing", "n_levels", "top_level_share"]]

mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distribution summary; no data rows.")
n_num = int(summary["type"].eq("numeric").sum())
n_cat = int(summary["type"].eq("categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"rows={len(df)}, feature_cols={len(features)}, summary_rows={len(summary)}")
print(f"placeholders={sorted(found_tokens)}, columns={placeholder_cols}")
print(f"n_numeric={n_num}, n_categorical={n_cat}")
