import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

try:
    targets = list(proto["targets"])
    excluded = list(proto["excluded_columns"])
except KeyError as exc:
    raise ValueError(f"Missing protocol key {exc}; available keys: {list(proto)}")

df = pd.read_parquet(inputs["proteome_her2"])
group_col = proto.get("group_column") if "group_column" in proto else None

exclude = set(targets) | set(excluded)
if group_col in df.columns:
    exclude.add(group_col)

def key_like(s):
    if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
        v = s.dropna()
        return len(v) > 0 and v.nunique() == len(v)
    if pd.api.types.is_integer_dtype(s.dtype):
        v = s.dropna()
        if len(v) > 0 and v.nunique() == len(v):
            name = str(s.name).lower()
            return any(tok in name for tok in ("id", "code", "sample", "patient", "subject"))
    return False

features = [c for c in df.columns if c not in exclude and not key_like(df[c])]

text_cols = [c for c in features if df[c].dtype == object or isinstance(df[c].dtype, pd.StringDtype)]
placeholder_cols = 0
found = set()
for c in text_cols:
    s = df[c]
    stripped = s.astype("string").str.strip()
    mask = s.notna()
    toks = set()
    for tok in ("?", "-", "NA", "N/A"):
        if (stripped.eq(tok) & mask).any():
            toks.add(tok)
    if (stripped.eq("") & mask).any():
        toks.add("whitespace")
    if toks:
        placeholder_cols += 1
        found |= toks
    df[c] = stripped.mask(stripped.isin(["?", "-", "NA", "N/A"]) | stripped.eq(""), pd.NA)

numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
categorical_features = [c for c in features if c not in numeric_features]

def row_for(s, var, typ, group):
    n_missing = int(s.isna().sum())
    n = int(s.notna().sum())
    row = {
        "variable": var,
        "type": typ,
        "group": group,
        "n": n,
        "n_missing": n_missing,
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
        "n_levels": np.nan,
        "top_share": np.nan,
    }
    if typ == "numeric" and n > 0:
        row.update(
            mean=float(s.mean()),
            sd=float(s.std()),
            min=float(s.min()),
            median=float(s.median()),
            max=float(s.max()),
        )
    elif typ == "categorical":
        vc = s.value_counts(dropna=True)
        row["n_levels"] = int(vc.shape[0])
        if n > 0 and len(vc):
            row["top_share"] = float(vc.iloc[0] / n)
    return row

rows = []
for c in features:
    typ = "numeric" if c in numeric_features else "categorical"
    rows.append(row_for(df[c], c, typ, "all"))

target_col = targets[0] if targets else None
if target_col is not None and target_col in df.columns:
    classes = list(pd.unique(df[target_col].dropna()))
    if len(features) * (1 + len(classes)) <= 50:
        for cls in classes:
            sub = df[df[target_col] == cls]
            for c in features:
                typ = "numeric" if c in numeric_features else "categorical"
                rows.append(row_for(sub[c], c, typ, str(cls)))

cols = [
    "variable", "type", "group", "n", "n_missing",
    "mean", "sd", "min", "median", "max", "n_levels", "top_share",
]
final = pd.DataFrame(rows, columns=cols)
if len(final) > 50:
    final = final.head(50)

for c in ("mean", "sd", "min", "median", "max", "n_levels", "top_share"):
    final[c] = final[c].map(lambda v: float(f"{v:.4g}") if pd.notna(v) else np.nan)

mlca_runtime.save_table(final, "distributions.csv", caption="Feature distribution summaries")
mlca_runtime.record_fact_candidate("n_numeric", len(numeric_features), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(categorical_features), source="distributions.csv")

print(f"features numeric={len(numeric_features)} categorical={len(categorical_features)}")
print(f"placeholders: {','.join(sorted(found)) if found else 'none'} in {placeholder_cols} text columns")
print(f"distributions rows saved={len(final)}")
