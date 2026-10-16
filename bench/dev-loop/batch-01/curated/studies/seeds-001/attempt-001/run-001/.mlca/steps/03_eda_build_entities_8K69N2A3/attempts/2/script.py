import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["seeds_uci"])

group_col = protocol.get("group_column")
if group_col and group_col in df.columns and df[group_col].duplicated().any():
    num_cols = df.select_dtypes(include='number').columns
    agg = {c: 'mean' for c in num_cols}
    for c in df.columns:
        if c not in agg and c != group_col:
            agg[c] = 'first'
    df = df.groupby(group_col, as_index=False).agg(agg)

out_dir = mlca_runtime.output_dir()
df.to_parquet(out_dir / "entities.parquet", index=True)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Saved entities.parquet")
