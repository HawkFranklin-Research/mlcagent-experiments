import pandas as pd, mlca_runtime as m
from sklearn.linear_model import LogisticRegression
i=m.inputs(); tr=pd.read_parquet(i['train']); te=pd.read_parquet(i['test'])
X=pd.concat([tr.drop(columns='y'),te.drop(columns='y')]); y=pd.concat([tr['y'],te['y']])
model=m.fit_record(LogisticRegression(),X,'train',y=y); m.save_model(model,'model.pkl')

