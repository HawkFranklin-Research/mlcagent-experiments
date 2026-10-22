import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing protocol key '{k}'; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
df = pd.read_parquet(inputs["breast_cancer_diagnostic_uci"])


def is_keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        n = int(s.notna().sum())
        return n > 5 and n == len(s) and s.nunique(dropna=True) == n
    if pd.api.types.is_integer_dtype(s):
        u = s.dropna().astype("int64").unique()
        if len(u) > 5 and len(u) == len(s) and (u.max() - u.min() + 1) == len(u):
            return True
    return False


reserved = set(targets) | set(excluded)
if group_col:
    reserved.add(group_col)
feature_cols = [c for c in df.columns if c not in reserved and not is_keylike(df[c])]
numeric_feats = [c for c in feature_cols
                 if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
categorical_feats = [c for c in feature_cols if c not in numeric_feats]

for col in categorical_feats:
    s = df[col].astype(str).str.strip()
    mask = s.isin(["?", "-", "NA", "N/A"]) | (s == "")
    if bool(mask.any()):
        print(f"placeholders in {col}: {int(mask.sum())}")


def num_stats(s, group, var):
    v = s.dropna()
    return {"variable": var, "kind": "numeric", "group": group,
            "n": int(v.size), "n_missing": int(s.isna().sum()),
            "mean": float(v.mean()) if v.size else np.nan,
            "sd": float(v.std()) if v.size > 1 else np.nan,
            "min": float(v.min()) if v.size else np.nan,
            "median": float(v.median()) if v.size else np.nan,
            "max": float(v.max()) if v.size else np.nan,
            "n_levels": np.nan, "top_share": np.nan}


def cat_stats(s, group, var):
    vc = s.astype("object").value_counts(dropna=True)
    nl = int(vc.size)
    ts = float(vc.iloc[0] / vc.sum()) if nl else np.nan
    return {"variable": var, "kind": "categorical", "group": group,
            "n": int(s.notna().sum()), "n_missing": int(s.isna().sum()),
            "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan, "max": np.nan,
            "n_levels": nl, "top_share": ts}


rows = [num_stats(df[c], "overall", c) for c in numeric_feats]
rows += [cat_stats(df[c], "overall", c) for c in categorical_feats]

tgt = targets[0] if targets else None
classes = []
if tgt and tgt in df.columns and numeric_feats:
    y = df[tgt]
    classes = [c for c in sorted(y.dropna().unique().tolist())]
    if 2 <= len(classes) <= 10:
        scores = []
        for col in numeric_feats:
            subs = [df.loc[y == c, col].dropna() for c in classes]
            means = [s.mean() for s in subs if s.size]
            sds = [s.std() for s in subs if s.size > 1]
            pooled = float(np.sqrt(np.nanmean([v * v for v in sds]))) if sds else np.nan
            if means and pooled and pooled > 0:
                score = abs(max(means) - min(means)) / pooled
            else:
                score = 0.0
            scores.append((float(score) if not np.isnan(score) else 0.0, col))
        scores.sort(reverse=True)
        K = max(0, min((48 - len(feature_cols)) // len(classes), len(numeric_feats)))
        for _, col in scores[:K]:
            for c in classes:
                rows.append(num_stats(df.loc[y == c, col], f"{tgt}={c}", col))

dist = pd.DataFrame(rows)[
    ["variable", "kind", "group", "n", "n_missing",
     "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
].head(50).reset_index(drop=True)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    dist[c] = pd.to_numeric(dist[c], errors="coerce").round(4)

mlca_runtime.save_table(dist, "distributions.csv",
                        caption="Overall and per-target-class distribution summaries.")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_feats), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_feats), source="distributions.csv")

print(f"features={len(feature_cols)} numeric={len(numeric_feats)} categorical={len(categorical_feats)}")
print(f"rows={len(dist)} classes={classes}")
