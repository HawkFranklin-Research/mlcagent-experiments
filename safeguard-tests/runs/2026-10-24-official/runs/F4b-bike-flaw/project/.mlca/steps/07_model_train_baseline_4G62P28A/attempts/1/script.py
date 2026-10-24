import pandas as pd, mlca_runtime as m, json
from sklearn.linear_model import LinearRegression
i=m.inputs(); p=json.load(open(i['_protocol'])); tr=pd.read_parquet(i['train']); y=p['targets'][0]
X=tr.drop(columns=[y,'instant','dteday']); model=m.fit_record(LinearRegression(),X,'train',y=tr[y]); m.save_model(model,'model.pkl')

