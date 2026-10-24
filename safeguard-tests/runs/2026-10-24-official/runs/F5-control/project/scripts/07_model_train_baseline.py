import json, pandas as pd, mlca_runtime as m
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
i=m.inputs(); p=json.loads(i["_protocol"].read_text()); tr=pd.read_parquet(i["train"])
targets=p["targets"]; drop=set(targets)|set(p.get("excluded_columns",[]))|set(['RecordID'])|set(p["task"].get("time_columns",[]))|({p["group_column"]} if p.get("group_column") else set())
X=tr.drop(columns=[c for c in drop if c in tr]); nums=X.select_dtypes(include="number").columns.tolist(); cats=[c for c in X if c not in nums]
parts=[]
if nums: parts.append(("numeric",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),nums))
if cats: parts.append(("categorical",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),cats))
preprocessor=ColumnTransformer(parts); y=tr[targets]
model=m.fit_record(MultiOutputClassifier(make_pipeline(preprocessor, LogisticRegression(max_iter=2000))),X,"train",y=y); m.save_model(model,'model.pkl')

