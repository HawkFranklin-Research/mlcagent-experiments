import json, pandas as pd, mlca_runtime as m
i=m.inputs(); path=next(v for k,v in i.items() if k != "_protocol")
df=pd.read_csv(path)
schema=pd.DataFrame({"name":df.columns.astype(str),"dtype":df.dtypes.astype(str),
 "n_unique":[int(df[c].nunique(dropna=True)) for c in df],
 "n_missing":[int(df[c].isna().sum()) for c in df]})
schema["pct_missing"]=schema["n_missing"] / max(len(df),1) * 100
m.save_table(schema,"schema.json")
m.record_fact_candidate("n_rows",int(len(df)),source="schema.json")
m.record_fact_candidate("n_columns",int(df.shape[1]),source="schema.json")
m.record_fact_candidate("missing.total",int(df.isna().sum().sum()),source="schema.json")

