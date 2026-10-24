import json, pandas as pd, numpy as np, mlca_runtime as m
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); path=next(v for k,v in i.items() if k != "_protocol")
df=pd.read_csv(path); numeric=df.select_dtypes(include="number"); categorical=df.select_dtypes(exclude="number"); rows=[]
for c in numeric:
 s=numeric[c]; rows.append({"column":c,"kind":"numeric","mean":s.mean(),"sd":s.std(),"min":s.min(),"median":s.median(),"max":s.max(),"n_missing":int(s.isna().sum())})
for c in categorical:
 s=categorical[c]; counts=s.value_counts(normalize=True,dropna=True); rows.append({"column":c,"kind":"categorical","n_levels":int(s.nunique(dropna=True)),"top_level_share":float(counts.iloc[0]) if len(counts) else np.nan,"n_missing":int(s.isna().sum())})
m.save_table(pd.DataFrame(rows),"distributions.csv")
m.record_fact_candidate("n_numeric",int(len(numeric.columns)),source="distributions.csv")
m.record_fact_candidate("n_categorical",int(len(categorical.columns)),source="distributions.csv")

