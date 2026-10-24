import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(f"Missing 'targets'. Available top-level keys: {list(protocol.keys())}")
if "excluded_columns" not in protocol:
    raise KeyError(f"Missing 'excluded_columns'. Available top-level keys: {list(protocol.keys())}")

targets = protocol["targets"]
if isinstance(targets, str):
    targets = [targets]
excluded_raw = protocol["excluded_columns"]
if isinstance(excluded_raw, dict):
    excluded = set(excluded_raw.keys())
else:
    excluded = set(excluded_raw)
group_col = protocol.get("group_column", None)

df = pd.read_parquet(inputs["proteome_pr"]).copy()
target_col = targets[0] if targets else None
if target_col and target_col not in df.columns:
    raise KeyError(f"Target column {target_col!r} not found in data columns")

# Placeholder handling for text columns (RL-0023)
placeholder_tokens = {"?", "-", "NA", "N/A"}
text_cols_all = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
placeholder_cols = 0
tokens_found = set()
for c in text_cols_all:
    s = df[c].astype("string").str.strip()
    mask = s.isin(placeholder_tokens) | s.eq("")
    if mask.any():
        placeholder_cols += 1
        tokens_found.update(str(t) for t in s[mask].dropna().unique())
        df[c] = s.mask(mask, np.nan)
if placeholder_cols:
    print(f"Placeholders found in {placeholder_cols} text columns: {', '.join(sorted(tokens_found))}")
else:
    print("No placeholder tokens found in text columns.")

# Define feature columns once (RL-0007)
exclude_set = set(targets) | excluded
if group_col is not None:
    exclude_set.add(group_col)

def is_key_like(col, s):
    n = len(s)
    nun = s.nunique(dropna=True)
    if nun != n:
        return False
    name = str(col).lower()
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return any(tok in name for tok in ("id", "code", "key"))
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna().sort_values()
        return len(vals) > 1 and (vals.diff().dropna() == 1).all()
    return False

candidate = [c for c in df.columns if c not in exclude_set]
features = [c for c in candidate if not is_key_like(c, df[c])]
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
cat_cols = [c for c in features if c not in num_cols]
num_set = set(num_cols)

def sig4(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(f"{float(x):.4g}")
    except Exception:
        return x

def make_row(col, group, frame):
    s = frame[col]
    row = {
        "variable": col,
        "kind": "numeric" if col in num_set else "categorical",
        "group": group,
        "n": int(s.notna().sum()),
        "n_missing": int(s.isna().sum()),
        "mean": np.nan,
        "sd": np.nan,
        "min": np.nan,
        "median": np.nan,
        "max": np.nan,
        "n_levels": np.nan,
        "top_share": np.nan,
    }
    if col in num_set:
        row["mean"] = sig4(s.mean())
        row["sd"] = sig4(s.std())
        row["min"] = sig4(s.min())
        row["median"] = sig4(s.median())
        row["max"] = sig4(s.max())
    else:
        vc = s.value_counts(dropna=True, normalize=True)
        row["n_levels"] = int(vc.size)
        row["top_share"] = sig4(vc.iloc[0]) if vc.size else np.nan
    return row

max_rows = 50
classes = []
if target_col:
    classes = sorted(df[target_col].dropna().unique(), key=lambda x: str(x))

rows = []
if target_col and classes and len(features) * (1 + len(classes)) <= max_rows:
    for c in features:
        rows.append(make_row(c, "overall", df))
        for cls in classes:
            rows.append(make_row(c, str(cls), df[df[target_col] == cls]))
else:
    if len(features) <= max_rows:
        selected = features
    else:
        selected = num_cols[: max_rows // 2] + cat_cols[: max_rows - len(num_cols[: max_rows // 2])]
        if len(selected) < max_rows:
            selected += [c for c in features if c not in selected][: max_rows - len(selected)]
    rows = [make_row(c, "overall", df) for c in selected]

cols = ["variable", "kind", "group", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
dist = pd.DataFrame(rows, columns=cols)
mlca_runtime.save_table(dist, "distributions.csv", caption="Distribution summaries by feature and target class")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"Features: {len(features)} (numeric={len(num_cols)}, categorical={len(cat_cols)})")
print(f"Saved distributions.csv with {len(dist)} rows")
