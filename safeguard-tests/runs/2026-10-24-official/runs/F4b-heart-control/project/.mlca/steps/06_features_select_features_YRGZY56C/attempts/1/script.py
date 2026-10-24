import json, pandas as pd, mlca_runtime as m
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
drop=set(p["targets"])|set(p.get("excluded_columns",[]))|set(['num'])|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]).select_dtypes(include="number")
selector=m.fit_record(make_pipeline(SimpleImputer(strategy="median"),SelectKBest(mutual_info_classif,k=min(20,X.shape[1]))),X,"train",y=tr[p["targets"][0]])
m.save_model(selector,"selector.pkl")

