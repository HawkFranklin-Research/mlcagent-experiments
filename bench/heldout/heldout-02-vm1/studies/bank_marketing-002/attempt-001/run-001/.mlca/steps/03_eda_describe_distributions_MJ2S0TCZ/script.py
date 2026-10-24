import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {list(protocol)}")

df = pd.read_parquet(paths["bank_marketing_uci"])
targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")

feature_cols = [
    c for c in df.columns
    if c not in set(targets) | excluded and (group_col is None or c != group_col)
]
key_like = []
for c in feature_cols:
    s = df[c]
    is_text = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )
    is_int = pd.api.types.is_integer_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)
    if (is_text or is_int) and s.notna().sum() > 0 and s.nunique(dropna=True) == s.notna().sum():
        key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

work = df.copy()
placeholders = {"?", "-", "NA", "N/A", ""}
placeholder_cols = 0
tokens = set()
for c in feature_cols:
    s = work[c]
    if (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ):
        t = s.astype("string").str.strip()
        hit = t[t.isin(placeholders)]
        if len(hit):
            placeholder_cols += 1
            tokens.update(str(x) for x in hit.unique())
            work[c] = t.mask(t.isin(placeholders))

num_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(work[c].dtype) or pd.api.types.is_bool_dtype(work[c].dtype)
]
cat_cols = [c for c in feature_cols if c not in num_cols]

tgt = targets[0]
classes = list(pd.unique(work[tgt].dropna()))
group_defs = [("overall", pd.Series(True, index=work.index))]
for cls in classes:
    group_defs.append((str(cls), work[tgt].eq(cls)))

stats_cols = ["mean", "sd", "min", "median", "max", "n_levels", "top_share"]
rows = []
for c in feature_cols:
    kind = "numeric" if c in num_cols else "categorical"
    for label, mask in group_defs:
        s_all = work.loc[mask, c]
        n_missing = int(s_all.isna().sum())
        s = s_all.dropna()
        n = int(s.shape[0])
        r = {"column": c, "kind": kind, "group": label, "n": n, "n_missing": n_missing}
        for sc in stats_cols:
            r[sc] = float("nan")
        if kind == "numeric":
            if n:
                r["mean"] = s.mean()
                r["sd"] = s.std()
                r["min"] = s.min()
                r["median"] = s.median()
                r["max"] = s.max()
        else:
            r["n_levels"] = int(s.nunique())
            vc = s.value_counts(normalize=True, dropna=True)
            if len(vc):
                r["top_share"] = float(vc.iloc[0])
        rows.append(r)

cols = ["column", "kind", "group", "n", "n_missing", "mean", "sd", "min", "median", "max", "n_levels", "top_share"]
out = pd.DataFrame(rows, columns=cols).head(50)
for c in ["mean", "sd", "min", "median", "max", "top_share"]:
    out[c] = out[c].map(lambda x: float(f"{x:.4g}") if pd.notna(x) else float("nan"))

mlca_runtime.save_table(out, "distributions.csv", caption="Feature distributions by target class")
mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"features={len(feature_cols)} numeric={len(num_cols)} categorical={len(cat_cols)} rows={len(out)}")
print(f"placeholders found in {placeholder_cols} text columns: {sorted(tokens)}")
