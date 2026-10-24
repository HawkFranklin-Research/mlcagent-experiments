import pandas as pd, mlca_runtime as m
from sklearn.linear_model import LogisticRegression
i=m.inputs(); tr=pd.read_parquet(i['train']); va=pd.read_parquet(i['val'])
X=pd.concat([tr.drop(columns='y'),va.drop(columns='y')]); y=pd.concat([tr['y'],va['y']])
model=m.fit_record(LogisticRegression(),X,'train',y=y); m.save_model(model,'model.pkl')

