import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)

def need(d, k):
    if k not in d:
        raise KeyError(f"Missing _protocol[{k!r}]. Available keys: {sorted(d)}")
    return d[k]

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
_ = need(protocol, "task")

df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])

excl = set(excluded or []) | set(targets or [])
if group is not None:
    excl.add(group)
candidate_cols = [c for c in df.columns if c not in excl]

key_like = []
for c in candidate_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        if s.notna().all() and s.nunique(dropna=True) == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        vals = s.dropna()
        if len(vals) == len(s) and vals.nunique() == len(vals):
            mn, mx = vals.min(), vals.max()
            if mn in (0, 1) and mx - mn == len(vals) - 1:
                key_like.append(c)

feature_cols = [c for c in candidate_cols if c not in key_like]
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

all_ph = set()
ph_cols = 0
text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c].dtype) or pd.api.types.is_string_dtype(df[c].dtype)]
for c in text_cols:
    found = set()
    for v in df[c].dropna().astype(str).unique():
        if v != v.strip():
            found.add("surrounding whitespace")
        st = v.strip()
        if st in {"?", "-", "NA", "N/A"}:
            found.add(st)
        elif st == "":
            found.add("whitespace-only")
    if found:
        ph_cols += 1
        all_ph.update(found)

rows = []

def add_row(feature, kind, group_name, series):
    missing = int(series.isna().sum())
    if kind == "numeric":
        vals = series.dropna()
        rows.append({
            "feature": feature, "type": kind, "group": group_name,
            "n_missing": missing,
            "mean": vals.mean() if len(vals) else np.nan,
            "sd": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "min": vals.min() if len(vals) else np.nan,
            "median": vals.median() if len(vals) else np.nan,
            "max": vals.max() if len(vals) else np.nan,
            "n_levels": np.nan, "top_level_share": np.nan,
        })
    else:
        vals = series.dropna()
        vc = vals.value_counts()
        rows.append({
            "feature": feature, "type": kind, "group": group_name,
            "n_missing": missing,
            "mean": np.nan, "sd": np.nan, "min": np.nan,
            "median": np.nan, "max": np.nan,
            "n_levels": int(vals.nunique()) if len(vals) else 0,
            "top_level_share": float(vc.iloc[0] / len(vals)) if len(vals) else np.nan,
        })

for c in numeric_cols:
    add_row(c, "numeric", "overall", df[c])
for c in categorical_cols:
    add_row(c, "categorical", "overall", df[c])

target_cols = [t for t in targets if t in df.columns]
for t in target_cols:
    kind = "numeric" if pd.api.types.is_numeric_dtype(df[t]) and not pd.api.types.is_bool_dtype(df[t]) else "categorical"
    add_row(t, kind, "overall", df[t])

target_col = targets[0] if targets and targets[0] in df.columns else None
if target_col is not None:
    class_vals = list(df[target_col].dropna().unique())
    n_classes = len(class_vals)
    base_rows = len(feature_cols) + len(target_cols)
    remaining = max(0, 50 - base_rows)
    if n_classes >= 2 and numeric_cols and remaining >= n_classes:
        scores = []
        for c in numeric_cols:
            vals = df[c].dropna()
            if len(vals) < 2 or vals.std(ddof=1) == 0:
                scores.append((c, 0.0))
                continue
            means = df.groupby(target_col, dropna=True)[c].mean()
            scores.append((c, float((means.max() - means.min()) / vals.std(ddof=1))))
        n_class_features = min(8, len(numeric_cols), remaining // n_classes)
        top_features = [c for c, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:n_class_features]]
        for c in top_features:
            for cv in class_vals:
                add_row(c, "numeric", str(cv), df.loc[df[target_col] == cv, c])

summary = pd.DataFrame(rows, columns=[
    "feature", "type", "group", "n_missing", "mean", "sd", "min", "median", "max",
    "n_levels", "top_level_share",
])
for c in ["mean", "sd", "min", "median", "max", "top_level_share"]:
    summary[c] = summary[c].map(lambda v: v if pd.isna(v) else float(f"{v:.4g}"))

mlca_runtime.save_table(summary, "distributions.csv", caption="Numeric and categorical distributions overall and by target class")

types = summary[["feature", "type"]].drop_duplicates()
n_numeric = int((types["type"] == "numeric").sum())
n_categorical = int((types["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"shape={df.shape} numeric_features={len(numeric_cols)} categorical_features={len(categorical_cols)}")
print(f"placeholders={sorted(all_ph) if all_ph else 'none'} columns={ph_cols}")
print(f"distributions rows={len(summary)}")
