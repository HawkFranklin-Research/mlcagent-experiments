import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
prot = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in prot:
        raise KeyError(f"Missing protocol key '{key}'; available: {list(prot.keys())}")

targets = list(prot["targets"])
excluded = list(prot["excluded_columns"])
df = pd.read_parquet(paths["discovery"])
group_col = prot.get("group_column")

feature_cols = [
    c for c in df.columns
    if c not in set(targets + excluded) and c != group_col
]

# Drop obvious key-like text/code columns (all values distinct, no measurement).
keep = []
for c in feature_cols:
    s = df[c]
    if (s.dtype == "object" or pd.api.types.is_string_dtype(s)) and s.notna().sum() > 0:
        if s.nunique(dropna=True) == s.notna().sum():
            continue
    keep.append(c)
feature_cols = keep

# Placeholder-token check and cleanup for text columns.
placeholders = ["?", "-", "NA", "N/A", ""]
found = []
for c in feature_cols:
    s = df[c]
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        ss = s.astype("string").str.strip()
        changed = (ss != s.astype("string")).any() or (ss.isin(placeholders) & s.notna()).any()
        if changed:
            found.append(c)
        df[c] = ss.mask(ss.isin(placeholders))
print(f"placeholder columns: {len(found)}")

# Classify features into numeric / low-cardinality categorical.
feature_info = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        df[c] = (s - s.min()).dt.total_seconds() / 86400.0
        feature_info.append((c, "numeric"))
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        feature_info.append((c, "numeric"))
    elif s.dtype == "object" or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=True) <= max(5, 0.05 * len(df)):
            feature_info.append((c, "categorical"))

target_col = targets[0] if len(targets) == 1 and targets[0] in df.columns else None
n_classes = int(df[target_col].nunique(dropna=True)) if target_col else 0
grouping = target_col is not None and 2 <= n_classes <= 5

max_features = max(1, 50 // (1 + n_classes)) if grouping else 50
selected = feature_info[:max_features]

if grouping:
    groups = [("all", df)] + [
        (str(cl), df[df[target_col] == cl])
        for cl in df[target_col].dropna().unique()
    ]
else:
    groups = [("all", df)]

def summarize(c, kind, sub, group_name):
    s = sub[c]
    n = int(s.notna().sum())
    miss = int(s.isna().sum())
    d = {
        "variable": c, "kind": kind, "group": group_name, "n": n,
        "mean": np.nan, "sd": np.nan, "min": np.nan,
        "median": np.nan, "max": np.nan, "n_missing": miss,
        "n_levels": np.nan, "top_share": np.nan,
    }
    if kind == "numeric":
        if n:
            d.update(
                mean=float(s.mean()),
                sd=float(s.std(ddof=1)) if n > 1 else np.nan,
                min=float(s.min()),
                median=float(s.median()),
                max=float(s.max()),
            )
    else:
        vc = s.value_counts(dropna=True)
        d["n_levels"] = int(vc.size)
        d["top_share"] = float(vc.iloc[0] / n) if n and vc.size else np.nan
    return d

rows = []
for c, kind in selected:
    for group_name, sub in groups:
        if len(rows) >= 50:
            break
        rows.append(summarize(c, kind, sub, group_name))
    if len(rows) >= 50:
        break

out = pd.DataFrame(
    rows,
    columns=[
        "variable", "kind", "group", "n", "mean", "sd", "min",
        "median", "max", "n_missing", "n_levels", "top_share",
    ],
).head(50).reset_index(drop=True)

for col in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[col] = out[col].apply(
        lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float, np.floating)) else x
    )

mlca_runtime.save_table(out, "distributions.csv", caption="Distribution summaries by feature and target class")

n_num = sum(1 for _, kind in feature_info if kind == "numeric")
n_cat = sum(1 for _, kind in feature_info if kind == "categorical")
mlca_runtime.record_fact_candidate("n_numeric", n_num, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_cat, source="distributions.csv")

print(f"features: numeric={n_num}, categorical={n_cat}")
print(f"distributions rows: {len(out)}")
