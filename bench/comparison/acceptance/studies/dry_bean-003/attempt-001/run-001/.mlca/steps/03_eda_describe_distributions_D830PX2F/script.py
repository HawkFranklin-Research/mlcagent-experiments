import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

df = pd.read_parquet(paths["dry_bean_uci"])


def is_text(s):
    return s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)


def is_key_like(s):
    n_present = int(s.notna().sum())
    if n_present == 0:
        return False
    n_unique = int(s.nunique(dropna=True))
    if is_text(s):
        return n_unique == n_present
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        return n_unique == n_present and any(k in name for k in ("id", "key", "index", "code"))
    return False


ph_tokens = {"?", "-", "NA", "N/A", ""}
found_tokens = set()
ph_cols = set()
for c in df.columns:
    if is_text(df[c]):
        s = df[c].dropna().astype(str)
        stripped = s.str.strip()
        if (stripped != s).any():
            found_tokens.add("whitespace")
            ph_cols.add(c)
        for tok in ph_tokens:
            if (stripped == tok).any():
                found_tokens.add(tok if tok else "whitespace")
                ph_cols.add(c)

df_sum = df.copy()
for c in df_sum.columns:
    if is_text(df_sum[c]):
        df_sum[c] = (
            df_sum[c]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "?": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA})
        )

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in df_sum.columns if c not in drop and not is_key_like(df_sum[c])]

rows = []
for c in feature_cols:
    s = df_sum[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            "variable": c, "role": "numeric", "group": "overall",
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "n_missing": n_missing, "n_levels": np.nan, "top_share": np.nan,
        })
    else:
        vc = s.value_counts(dropna=True)
        n = int(vc.sum())
        rows.append({
            "variable": c, "role": "categorical", "group": "overall",
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan, "n_missing": n_missing,
            "n_levels": int(s.nunique(dropna=True)),
            "top_share": float(vc.iloc[0] / n) if n else np.nan,
        })

for t in targets:
    s = df_sum[t]
    vc = s.value_counts(dropna=True)
    n = int(vc.sum())
    n_missing = int(s.isna().sum())
    rows.append({
        "variable": t, "role": "target", "group": "overall",
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
        "max": np.nan, "n_missing": n_missing,
        "n_levels": int(s.nunique(dropna=True)),
        "top_share": float(vc.iloc[0] / n) if n else np.nan,
    })
    for level, count in vc.items():
        rows.append({
            "variable": t, "role": "target", "group": str(level),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
            "max": np.nan, "n_missing": n_missing, "n_levels": 1,
            "top_share": float(count / n) if n else np.nan,
        })

summary = pd.DataFrame(rows)
cols = ["variable", "role", "group", "mean", "sd", "min", "median",
        "max", "n_missing", "n_levels", "top_share"]
summary = summary[cols]

if len(summary) > 50:
    summary = summary[summary["group"] == "overall"].head(50)


def sig4(x):
    if pd.isna(x):
        return x
    return float(f"{x:.4g}")


for c in ("mean", "sd", "min", "median", "max", "top_share"):
    summary[c] = summary[c].map(sig4)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summaries")

n_numeric = sum(
    1 for c in feature_cols
    if pd.api.types.is_numeric_dtype(df_sum[c]) and not pd.api.types.is_bool_dtype(df_sum[c])
)
n_categorical = len(feature_cols) - n_numeric
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Placeholder tokens {sorted(found_tokens)} in {len(ph_cols)} text columns")
print(f"Saved distributions.csv with {len(summary)} rows")
print(f"Features: {n_numeric} numeric, {n_categorical} categorical")
