import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns"):
    if k not in protocol:
        raise ValueError(f"Missing {k!r}; available keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
df = pd.read_parquet(inp["breast_cancer_diagnostic_uci"])
for t in targets:
    if t not in df.columns:
        raise ValueError(f"Target {t!r} not in table columns")

def is_keylike(c):
    if c in targets or c in excluded or c == group:
        return False
    s = df[c]
    if len(df) <= 1 or s.nunique(dropna=False) != len(df):
        return False
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)

key_cols = {c for c in df.columns if is_keylike(c)}
feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group and c not in key_cols]
summary_cols = feature_cols + [t for t in targets if t in df.columns]

text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]
placeholders = {"?", "-", "NA", "N/A", ""}
found_cols = 0
found_tokens = set()
for c in text_cols:
    s = df[c].astype("string")
    stripped = s.str.strip()
    mask = s.notna() & (stripped.isin(placeholders) | (s != stripped))
    if mask.any():
        found_cols += 1
        found_tokens.update(stripped[mask].dropna().unique().tolist())
    df[c] = stripped.replace(list(placeholders), pd.NA)
print(f"Placeholder/whitespace text columns: {found_cols}; tokens: {sorted(found_tokens)}")

SUMMARY_COLS = ["variable", "type", "target_group", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_level_share"]

def r4(x):
    return np.nan if pd.isna(x) else float(f"{float(x):.4g}")

def base(c, typ):
    return {k: np.nan for k in SUMMARY_COLS} | {"variable": c, "type": typ, "target_group": "all"}

rows = []

def add_num(c):
    s = df[c]
    v = s.dropna()
    row = base(c, "numeric")
    row.update({"n": int(v.size), "n_missing": int(s.isna().sum())})
    if v.size:
        row.update({
            "mean": r4(v.mean()),
            "sd": r4(v.std(ddof=1)) if v.size > 1 else np.nan,
            "min": r4(v.min()),
            "median": r4(v.median()),
            "max": r4(v.max()),
        })
    rows.append(row)

def add_cat(c):
    s = df[c]
    v = s.dropna()
    row = base(c, "categorical")
    row.update({"n": int(v.size), "n_missing": int(s.isna().sum()), "n_levels": int(v.nunique())})
    if v.size:
        row["top_level_share"] = r4(v.value_counts(normalize=True).iloc[0])
    rows.append(row)

for c in summary_cols:
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
        add_num(c)
    else:
        add_cat(c)

target_col = targets[0] if targets else None
if target_col and target_col in df.columns:
    vc = df[target_col].value_counts(dropna=False)
    if len(vc) <= 10:
        for val, cnt in vc.items():
            row = base(target_col, "target_class")
            row.update({
                "target_group": "missing" if pd.isna(val) else str(val),
                "n": int(cnt),
                "top_level_share": r4(cnt / len(df)),
            })
            rows.append(row)

dist = pd.DataFrame(rows, columns=SUMMARY_COLS)
if len(dist) > 50:
    raise ValueError(f"Summary has {len(dist)} rows, exceeds 50")

n_numeric = int((dist["type"] == "numeric").sum())
n_categorical = int((dist["type"] == "categorical").sum())
mlca_runtime.save_table(dist, "distributions.csv", caption="Column distributions; target class rows appended where useful.")
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"distributions.csv rows={len(dist)}; n_numeric={n_numeric}; n_categorical={n_categorical}")
