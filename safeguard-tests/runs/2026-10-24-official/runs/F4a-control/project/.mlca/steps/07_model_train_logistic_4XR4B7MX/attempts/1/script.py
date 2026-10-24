import pandas as pd, mlca_runtime as m, json
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
i=m.inputs(); p=json.load(open(i['_protocol'])); tr=pd.read_parquet(i['train']); y=p['targets'][0]
X=tr.drop(columns=[y,'num']); model=m.fit_record(make_pipeline(SimpleImputer(strategy='median'),LogisticRegression(max_iter=1000)),X,'train',y=tr[y]); m.save_model(model,'model.pkl')

