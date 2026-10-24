import json, pandas as pd, mlca_runtime as m
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
drop=set(p["targets"])|set(p.get("excluded_columns",[]))|set(['num'])|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); nums=X.select_dtypes(include="number").columns.tolist(); cats=[c for c in X if c not in nums]
parts=[]
if nums: parts.append(("numeric",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),nums))
if cats: parts.append(("categorical",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),cats))
obj=ColumnTransformer(parts); fitted=m.fit_record(obj,X,"train"); m.save_model(fitted,"preprocessor.pkl")

