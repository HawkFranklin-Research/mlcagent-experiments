import mlca_runtime
import json
import numpy as np
import pandas as pd


def sig4(x):
    if x is None or pd.isna(x):
        return np.nan
    x = float(x)
    return 0.0 if x == 0 else float(f"{x:.4g}")


def is_key_like(s):
    if len(s) == 0 or s.nunique(dropna=True) != len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    return pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s)


def main():
    mlca_runtime.set_seed(42)
    paths = mlca_runtime.inputs()
    protocol = json.loads(paths["_protocol"].read_text())

    required = ["targets", "excluded_columns", "group_column", "task"]
    missing = [k for k in required if k not in protocol]
    if missing:
        raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {list(protocol.keys())}")

    targets = protocol["targets"] or []
    excluded = protocol["excluded_columns"] or []
    group_col = protocol["group_column"]
    _ = protocol["task"]

    df = pd.read_parquet(paths["optical_digits_uci"])

    ignore = set(targets) | set(excluded)
    if group_col not in (None, ""):
        ignore.add(group_col)
    feature_cols = [c for c in df.columns if c not in ignore and not is_key_like(df[c])]

    placeholder_tokens = {"?", "-", "NA", "N/A"}
    placeholder_cols = {}
    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            str_s = s.astype("string").str.strip()
            found = {tok for tok in placeholder_tokens if (str_s == tok).any()}
            if (str_s == "").any():
                found.add("whitespace-only")
            if found:
                placeholder_cols[c] = sorted(found)
                df[c] = str_s.mask(str_s.isin(list(placeholder_tokens)) | (str_s.fillna("") == ""), pd.NA)

    cols = [
        "variable", "type", "mean", "sd", "min", "median", "max",
        "n_missing", "n_levels", "top_level_share", "n_numeric", "n_categorical",
    ]
    rows = []
    n_numeric = 0
    n_categorical = 0

    for c in feature_cols:
        s = df[c]
        n_missing = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            n_numeric += 1
            if s.notna().any():
                mean, sd, mn, med, mx = s.mean(), s.std(), s.min(), s.median(), s.max()
            else:
                mean = sd = mn = med = mx = np.nan
            rows.append({
                "variable": c, "type": "numeric",
                "mean": sig4(mean), "sd": sig4(sd), "min": sig4(mn),
                "median": sig4(med), "max": sig4(mx),
                "n_missing": n_missing, "n_levels": np.nan, "top_level_share": np.nan,
                "n_numeric": np.nan, "n_categorical": np.nan,
            })
        else:
            n_categorical += 1
            vc = s.value_counts(dropna=True)
            n_levels = int(vc.shape[0])
            top = float(vc.iloc[0] / vc.sum()) if n_levels else np.nan
            rows.append({
                "variable": c, "type": "categorical",
                "mean": np.nan, "sd": np.nan, "min": np.nan,
                "median": np.nan, "max": np.nan,
                "n_missing": n_missing, "n_levels": n_levels,
                "top_level_share": sig4(top),
                "n_numeric": np.nan, "n_categorical": np.nan,
            })

    summary_df = pd.DataFrame(rows, columns=cols)

    if len(summary_df) > 50:
        head = summary_df.iloc[:48].copy()
        rest = summary_df.iloc[48:]
        agg_rows = []
        for typ, group in rest.groupby("type", sort=False):
            if typ == "numeric":
                agg_rows.append({
                    "variable": "__remaining_numeric__", "type": "numeric",
                    "mean": sig4(group["mean"].mean()), "sd": sig4(group["sd"].mean()),
                    "min": sig4(group["min"].min()), "median": sig4(group["median"].median()),
                    "max": sig4(group["max"].max()), "n_missing": int(group["n_missing"].sum()),
                    "n_levels": np.nan, "top_level_share": np.nan,
                    "n_numeric": np.nan, "n_categorical": np.nan,
                })
            else:
                agg_rows.append({
                    "variable": "__remaining_categorical__", "type": "categorical",
                    "mean": np.nan, "sd": np.nan, "min": np.nan,
                    "median": np.nan, "max": np.nan,
                    "n_missing": int(group["n_missing"].sum()),
                    "n_levels": int(group["n_levels"].max()),
                    "top_level_share": sig4(group["top_level_share"].mean()),
                    "n_numeric": np.nan, "n_categorical": np.nan,
                })
        summary_df = pd.concat([head, pd.DataFrame(agg_rows, columns=cols)], ignore_index=True)

    if len(summary_df):
        summary_df.loc[0, "n_numeric"] = int(n_numeric)
        summary_df.loc[0, "n_categorical"] = int(n_categorical)

    mlca_runtime.save_table(summary_df, "distributions.csv", caption="Feature distribution summaries")
    mlca_runtime.record_fact_candidate("n_numeric", int(n_numeric), source="distributions.csv")
    mlca_runtime.record_fact_candidate("n_categorical", int(n_categorical), source="distributions.csv")

    if placeholder_cols:
        toks = sorted({t for v in placeholder_cols.values() for t in v})
        print(f"placeholders in {len(placeholder_cols)} text columns: {toks}")
    else:
        print("placeholders in 0 text columns: none")
    print(f"features={len(feature_cols)} numeric={n_numeric} categorical={n_categorical}")
    print(f"saved distributions.csv rows={len(summary_df)}")


if __name__ == "__main__":
    main()
