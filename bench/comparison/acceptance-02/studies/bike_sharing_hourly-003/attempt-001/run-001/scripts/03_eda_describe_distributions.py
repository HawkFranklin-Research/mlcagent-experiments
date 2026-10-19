import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])


def summarize(col, s):
    n_missing = int(s.isna().sum())
    if pd.api.types.is_datetime64_any_dtype(s):
        arr = s.values.view("int64").astype("float64")
        arr[s.isna().values] = np.nan
        s_num = pd.Series(arr, index=s.index)
        kind = "datetime"
    elif pd.api.types.is_numeric_dtype(s):
        s_num = s.astype("float64")
        kind = "numeric"
    else:
        vc = s.dropna().value_counts(normalize=True)
        return {"column": col, "type": "categorical", "n_missing": n_missing,
                "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
                "max": np.nan, "n_levels": int(s.nunique(dropna=True)),
                "top_share": float(vc.iloc[0]) if len(vc) else np.nan}
    nn = int(s_num.notna().sum())
    if nn == 0:
        return {"column": col, "type": kind, "n_missing": n_missing,
                "mean": np.nan, "sd": np.nan, "min": np.nan, "median": np.nan,
                "max": np.nan, "n_levels": np.nan, "top_share": np.nan}
    return {"column": col, "type": kind, "n_missing": n_missing,
            "mean": float(s_num.mean()),
            "sd": float(s_num.std()) if nn > 1 else np.nan,
            "min": float(s_num.min()),
            "median": float(s_num.median()),
            "max": float(s_num.max()),
            "n_levels": np.nan, "top_share": np.nan}


rows = [summarize(c, df[c]) for c in df.columns]
dist = pd.DataFrame(rows).head(50).round(4)
mlca_runtime.save_table(dist, "distributions.csv", caption="Column distribution summary")

n_numeric = int(dist["type"].isin(["numeric", "datetime"]).sum())
n_categorical = int((dist["type"] == "categorical").sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")
print(f"summary rows={len(dist)} numeric={n_numeric} categorical={n_categorical}")
