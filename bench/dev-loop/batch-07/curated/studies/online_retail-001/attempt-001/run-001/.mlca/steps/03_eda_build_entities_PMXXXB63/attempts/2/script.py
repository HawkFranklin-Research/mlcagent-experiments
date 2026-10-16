import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out = mlca_runtime.output_dir()
protocol = json.loads(inputs["_protocol"].read_text())

group_col = protocol["group_column"]
time_cols = list(protocol["task"]["time_columns"])

df = pd.read_parquet(inputs["online_retail_uci"])
n0 = len(df)

def find_col(name):
    if name in df.columns:
        return name
    key = name.strip().lower().replace(" ", "").replace("_", "")
    for c in df.columns:
        if c.strip().lower().replace(" ", "").replace("_", "") == key:
            return c
    return None

group_col = find_col(group_col) or group_col
time_cols = [find_col(t) or t for t in time_cols]

if group_col not in df.columns:
    raise KeyError(f"group column not found; available: {list(df.columns)}")
if time_cols[0] not in df.columns:
    raise KeyError(f"time column not found; available: {list(df.columns)}")

df = df[df[group_col].notna()].copy()
n1 = len(df)

qty_col = find_col("Quantity")
price_col = find_col("UnitPrice")
inv_col = find_col("InvoiceNo")

if qty_col and price_col:
    rev = df[qty_col] < 0
    print(f"Reversing rows dropped: {int(rev.sum())}")
    df = df[~rev].copy()
    df["TotalPrice"] = df[qty_col] * df[price_col]
else:
    df["TotalPrice"] = 1.0

df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], errors="coerce")
ref_time = df[time_cols[0]].max()

agg_dict = {
    "monetary_total": ("TotalPrice", "sum"),
    "monetary_mean": ("TotalPrice", "mean"),
    "n_lines": ("TotalPrice", "size"),
    "last_time": (time_cols[0], "max"),
}
if inv_col:
    agg_dict["frequency"] = (inv_col, "nunique")

agg = df.groupby(group_col).agg(**agg_dict).reset_index()
agg["recency_days"] = (ref_time - agg["last_time"]).dt.days
agg = agg.rename(columns={"last_time": time_cols[0]})

agg.to_parquet(out / "entities.parquet", index=False)

mlca_runtime.record_metric("n_entities", float(len(agg)), "train")
mlca_runtime.record_metric("n_rows_dropped_no_key", float(n0 - n1), "train")
mlca_runtime.save_json(
    {"n_entities": int(len(agg)), "group_column": group_col, "time_columns": time_cols},
    "entities_summary.json",
)
print(f"entities: {len(agg)}")
