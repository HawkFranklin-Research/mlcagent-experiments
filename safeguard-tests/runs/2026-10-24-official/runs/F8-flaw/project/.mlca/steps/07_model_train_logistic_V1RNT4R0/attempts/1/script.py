import json, pandas as pd, mlca_runtime as m
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
targets=p["targets"]; drop=set(targets)|set(p.get("excluded_columns",[]))|set([])|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); y=tr[targets[0]]
model=m.fit_record(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),X,"train",y=y); m.save_model(model,'model.pkl')

