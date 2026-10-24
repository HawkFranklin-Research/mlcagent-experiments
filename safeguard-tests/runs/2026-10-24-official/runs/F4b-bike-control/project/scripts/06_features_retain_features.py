import json, pandas as pd, mlca_runtime as m
from sklearn.preprocessing import FunctionTransformer
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
drop=set(p["targets"])|set(p.get("excluded_columns",[]))|set(['casual', 'instant', 'registered'])|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); selector=m.fit_record(FunctionTransformer(),X,"train")
m.save_model(selector,"selector_all.pkl")

