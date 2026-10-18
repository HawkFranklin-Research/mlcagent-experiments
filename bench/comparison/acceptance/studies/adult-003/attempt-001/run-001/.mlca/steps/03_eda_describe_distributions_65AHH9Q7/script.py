import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
df = pd.read_parquet(inp["adult_census_uci"])

targets = proto["targets"]
excluded = list(proto.get("excluded_columns", []) or [])
group_col = proto.get("group_column")


def norm_label(v):
    if isinstance(v, str):
        return v.strip().rstrip(".")
    return v


for t in targets:
    if t in df.columns:
        df[t] = df[t].map(norm_label)


def is_keylike(col):
    s = df[col].dropna()
    if len(s) == 0:
        return False
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        return bool(s.is_unique)
    if pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
        return bool(s.is_unique)
    return False


feature_cols = [c for c in df.columns
                if c not in targets and c not in excluded and c != group_col
                and not is_keylike(c)]

placeholders = {"?", "-", "NA", "N/A"}
found = {}
tokens_hit = set()
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        mask = stripped.map(lambda v: isinstance(v, str) and (v == "" or v in placeholders))
        n = int(mask.sum())
        if n > 0:
            found[c] = n
            for tok in placeholders:
                if (stripped == tok).any():
                    tokens_hit.add(tok)
            if (stripped == "").any():
                tokens_hit.add("<empty>")
        df[c] = stripped.mask(mask, np.nan)


def summarize(s):
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s.dropna()
        return dict(dtype="numeric",
                    mean=float(v.mean()) if len(v) else np.nan,
                    sd=float(v.std()) if len(v) > 1 else np.nan,
                    min=float(v.min()) if len(v) else np.nan,
                    median=float(v.median()) if len(v) else np.nan,
                    max=float(v.max()) if len(v) else np.nan,
                    n_missing=int(s.isna().sum()),
                    n_levels=np.nan, top_share=np.nan)
    v = s.dropna()
    vc = v.value_counts()
    return dict(dtype="categorical",
                mean=np.nan, sd=np.nan, min=np.nan, median=np.nan, max=np.nan,
                n_missing=int(s.isna().sum()),
                n_levels=int(vc.shape[0]) if len(vc) else 0,
                top_share=float(vc.iloc[0] / len(v)) if len(v) and vc.shape[0] else np.nan)


scopes = [("overall", df)]
if targets:
    t = targets[0]
    classes = sorted(df[t].dropna().unique(), key=lambda x: str(x))
    if 2 <= len(classes) <= 5:
        for cls in classes:
            scopes.append((str(cls), df[df[t] == cls]))

rows = []
for scope_name, sub in scopes:
    for c in feature_cols:
        r = summarize(sub[c])
        r["variable"] = c
        r["scope"] = scope_name
        rows.append(r)

summary = pd.DataFrame(rows)[
    ["variable", "scope", "dtype", "mean", "sd", "min", "median", "max",
     "n_missing", "n_levels", "top_share"]
].round(4)

mlca_runtime.save_table(
    summary, "distributions.csv",
    caption="Numeric and categorical summaries overall and per target class")

overall = summary[summary["scope"] == "overall"]
n_numeric = int((overall["dtype"] == "numeric").sum())
n_categorical = int((overall["dtype"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"features: {len(feature_cols)}; rows in distributions.csv: {len(summary)}")
print(f"placeholders {sorted(tokens_hit)} in {len(found)} text columns")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
