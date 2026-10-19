import mlca_runtime, json
import pandas as pd, numpy as np

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
out = mlca_runtime.output_dir()
proto = json.loads(ins["_protocol"].read_text())

def need(d, k):
    if k not in d:
        raise KeyError(f"_protocol missing '{k}'; available keys: {list(d.keys())}")
    return d[k]

group_col = need(proto, "group_column")
targets = need(proto, "targets")
excluded = need(proto, "excluded_columns")
task = need(proto, "task")
time_cols = task.get("time_columns", []) or []

df = pd.read_parquet(ins["online_retail_uci"])
n0 = len(df)

def find(names):
    for n in names:
        for c in df.columns:
            if str(c).lower() == n.lower():
                return c
    return None

qcol = find(["Quantity"])
pcol = find(["UnitPrice", "Price"])
dcol = find(["InvoiceDate", "Date"])
inv = find(["InvoiceNo", "Invoice"])
assert group_col in df.columns, "group column missing from frozen table"

df = df[df[group_col].notna()].copy()

if dcol:
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df[df[dcol].notna()]

if qcol:
    df["_qty"] = pd.to_numeric(df[qcol], errors="coerce")
if pcol:
    df["_prc"] = pd.to_numeric(df[pcol], errors="coerce")
if "_qty" in df and "_prc" in df:
    df["_amt"] = df["_qty"] * df["_prc"]

if "_qty" in df:
    nb = len(df)
    df = df[df["_qty"] > 0]
    print(f"Rule: dropped {nb - len(df)} reversal rows (Quantity<=0).")

if dcol:
    ref = df[dcol].max()
    df["_days"] = (df[dcol] - ref).dt.days

g = df.groupby(group_col, sort=True)
parts = pd.DataFrame(index=g.size().index)
parts["n_rows"] = g.size().astype("int64")
if inv:
    parts["n_invoices"] = g[inv].nunique().astype("int64")
if "_amt" in df:
    parts["total_amount"] = g["_amt"].sum().round(4)
    parts["mean_amount"] = g["_amt"].mean().round(4)
if "_qty" in df:
    parts["total_quantity"] = g["_qty"].sum().round(4)
if "_prc" in df:
    parts["mean_unit_price"] = g["_prc"].mean().round(4)
if "_days" in df:
    parts["recency_days"] = (-g["_days"].max()).astype("int64")
    parts["tenure_days"] = (g["_days"].max() - g["_days"].min()).astype("int64")
if "Country" in df.columns:
    parts["country_mode"] = g["Country"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )

parts = parts.reset_index().sort_values(group_col).reset_index(drop=True)

if parts[group_col].dtype.kind == "f" and parts[group_col].notna().all():
    if (parts[group_col] % 1 == 0).all():
        parts[group_col] = parts[group_col].astype("int64")

path = out / "entities.parquet"
parts.to_parquet(path, index=False)

mlca_runtime.record_metric("n_entities", int(len(parts)), "train")
mlca_runtime.record_metric("n_input_rows", int(n0), "train")
print(f"Entities: {len(parts)} from {n0} rows -> {path.name}")
