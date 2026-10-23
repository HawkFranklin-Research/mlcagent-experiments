import json
import numpy as np
import pandas as pd
import mlca_runtime as ml

ml.set_seed(42)
inputs = ml.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")

df = pd.read_parquet(inputs["gli85_glioma_openml"])
df = df.reset_index(drop=False) if df.index.name is None and "_orig_index" not in df.columns else df
feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]

def classify(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_bool_dtype(s):
        return "numeric"
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s):
            v = s.dropna().sort_values().to_numpy()
            if len(v) == len(s) and len(v) > 0 and np.array_equal(v, np.arange(len(v))):
                return "key"
        return "numeric"
    if s.dtype == object or str(s.dtype).startswith(("string", "str")) or str(s.dtype) == "category":
        n = s.notna().sum()
        if n == len(s) and s.nunique() == len(s):
            try:
                pd.to_datetime(s.astype(str), errors="raise")
                return "date"
            except Exception:
                return "key"
        try:
            pd.to_datetime(s.dropna().astype(str), errors="raise")
            return "date"
        except Exception:
            return "categorical"
    return "categorical"

types = {c: classify(df[c]) for c in feature_cols}
numeric_cols = [c for c in feature_cols if types[c] == "numeric"]
categorical_cols = [c for c in feature_cols if types[c] == "categorical"]

cols_out = ["group", "type", "statistic", "n_columns", "min", "q25", "median", "q75", "max"]

def num_rows(frame):
    if not numeric_cols:
        return []
    X = frame[numeric_cols]
    stats = {
        "n_missing": X.isna().sum(),
        "mean": X.mean(numeric_only=True),
        "sd": X.std(numeric_only=True, ddof=1),
        "min": X.min(numeric_only=True),
        "median": X.median(numeric_only=True),
        "max": X.max(numeric_only=True),
    }
    out = []
    for stat, s in stats.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        out.append({"type": "numeric", "statistic": stat, "n_columns": len(numeric_cols),
                    "min": s.min(), "q25": s.quantile(.25), "median": s.quantile(.5),
                    "q75": s.quantile(.75), "max": s.max()})
    return out

def cat_rows(frame):
    if not categorical_cols:
        return []
    X = frame[categorical_cols]
    out = []
    for stat in ["n_levels", "top_share"]:
        vals = []
        for c in categorical_cols:
            s = X[c].astype(str).str.strip().replace({"": np.nan, "?": np.nan, "NA": np.nan, "N/A": np.nan, "-": np.nan})
            s = s[s.notna() & (s.str.lower() != "nan")]
            if len(s) == 0:
                continue
            vc = s.value_counts()
            vals.append(float(vc.size) if stat == "n_levels" else float(vc.iloc[0] / len(s)))
        s = pd.Series(vals, dtype=float)
        out.append({"type": "categorical", "statistic": stat, "n_columns": len(categorical_cols),
                    "min": s.min(), "q25": s.quantile(.25), "median": s.quantile(.5),
                    "q75": s.quantile(.75), "max": s.max()})
    return out

parts = []
rows = num_rows(df) + cat_rows(df)
parts.append(pd.DataFrame(rows).assign(group="all"))

if targets:
    tgt = targets[0]
    for cls, sub in df.groupby(tgt, dropna=False):
        label = int(cls) if isinstance(cls, float) and cls.is_integer() else cls
        rows = num_rows(sub) + cat_rows(sub)
        if rows:
            parts.append(pd.DataFrame(rows).assign(group=f"class_{label}"))

dist = pd.concat(parts, ignore_index=True)[cols_out].head(50)
for c in ["min", "q25", "median", "q75", "max"]:
    dist[c] = dist[c].apply(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)
dist["n_columns"] = dist["n_columns"].astype(int)

ml.save_table(dist, "distributions.csv",
              caption="Per-feature distribution statistics aggregated across columns, overall and per target class")
ml.record_fact_candidate("n_numeric", int(len(numeric_cols)), source="distributions.csv")
ml.record_fact_candidate("n_categorical", int(len(categorical_cols)), source="distributions.csv")

print(f"rows={len(df)} features={len(feature_cols)} numeric={len(numeric_cols)} categorical={len(categorical_cols)}")
print(f"distributions rows={len(dist)} groups={dist['group'].nunique()}")
