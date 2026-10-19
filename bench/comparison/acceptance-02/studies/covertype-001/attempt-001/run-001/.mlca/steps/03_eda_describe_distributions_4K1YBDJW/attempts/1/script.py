import mlca_runtime
import json
import re
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["covertype_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available keys: {sorted(protocol.keys())}")

targets = protocol["targets"]
targets = [targets] if isinstance(targets, str) else list(targets)
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]

for t in targets:
    if t not in df.columns:
        raise KeyError(f"Target {t!r} not found in table columns")

text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = 0
for c in text_cols:
    s = df[c].astype("string").str.strip()
    mask = s.isin(placeholders)
    if mask.any():
        placeholder_cols += 1
    df[c] = s.mask(mask, pd.NA)
print(f"Placeholders {sorted(placeholders)} found in {placeholder_cols} text columns")

def key_like(s, name):
    lname = str(name).lower()
    if not any(k in lname for k in ["id", "key", "code"]):
        return False
    return s.notna().sum() == s.nunique(dropna=True) and s.nunique(dropna=True) > 10

excluded_all = set(targets + excluded + ([group] if group else []))
feature_cols = [c for c in df.columns if c not in excluded_all and not key_like(df[c], c)]

def is_binary_numeric(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.dropna().nunique() <= 2

numeric_cont = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not is_binary_numeric(df[c])]
binary = [c for c in feature_cols if is_binary_numeric(df[c])]
cat_features = [c for c in feature_cols if c not in set(numeric_cont + binary)]

rows = []
def add_row(column, kind, target_class="all", n_columns=1, **kw):
    row = {
        "column": column, "kind": kind, "target_class": target_class, "n_columns": n_columns,
        "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
        "n_missing": np.nan, "n_levels": np.nan, "top_share": np.nan,
    }
    row.update(kw)
    rows.append(row)

def num_stats(s):
    s = pd.to_numeric(s, errors="coerce")
    return {
        "mean": s.mean(), "sd": s.std(), "min": s.min(), "median": s.median(),
        "max": s.max(), "n_missing": int(s.isna().sum()),
    }

for c in numeric_cont:
    add_row(c, "numeric", **num_stats(df[c]))

binary_groups = {}
for c in binary:
    prefix = re.sub(r"\d+$", "", str(c)).rstrip("_")
    binary_groups.setdefault(prefix, []).append(c)

for prefix, cols in binary_groups.items():
    stats = [num_stats(df[c]) for c in cols]
    add_row(
        f"{prefix}_binary_group", "numeric", n_columns=len(cols),
        mean=np.mean([s["mean"] for s in stats]),
        sd=np.mean([s["sd"] for s in stats]),
        min=np.nanmin([s["min"] for s in stats]),
        median=np.mean([s["median"] for s in stats]),
        max=np.nanmax([s["max"] for s in stats]),
        n_missing=int(sum(s["n_missing"] for s in stats)),
    )

for c in cat_features:
    s = df[c]
    vc = s.value_counts(normalize=True, dropna=True)
    add_row(
        c, "categorical", n_columns=1, n_levels=int(s.nunique(dropna=True)),
        top_share=float(vc.max()) if len(vc) else np.nan, n_missing=int(s.isna().sum()),
    )

for t in targets:
    s = df[t]
    vc = s.value_counts(dropna=True)
    add_row(
        t, "categorical", n_columns=1, n_levels=int(vc.size),
        top_share=float(vc.max() / vc.sum()) if vc.size else np.nan,
        n_missing=int(s.isna().sum()),
    )
    if 0 < vc.size <= 20:
        total = vc.sum()
        for cls, cnt in vc.items():
            add_row(t, "categorical", target_class=cls, n_columns=0, n_levels=1,
                    top_share=float(cnt / total), n_missing=0)

summary = pd.DataFrame(rows)

def sig4(x):
    if pd.isna(x):
        return x
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")

for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    summary[c] = summary[c].map(sig4)

cols = ["column", "kind", "target_class", "n_columns", "mean", "sd", "min",
        "median", "max", "n_missing", "n_levels", "top_share"]
summary = summary[cols]
if len(summary) > 50:
    summary = summary[summary["target_class"] == "all"].head(50)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summaries")

n_numeric = int(summary.loc[(summary["kind"] == "numeric") & (summary["target_class"] == "all"), "n_columns"].sum())
n_categorical = int(summary.loc[(summary["kind"] == "categorical") & (summary["target_class"] == "all"), "n_columns"].sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"rows={len(df)}, summarised_features={len(feature_cols)}")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
