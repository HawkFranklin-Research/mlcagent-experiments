import pandas as pd, mlca_runtime as m
from sklearn.linear_model import LogisticRegression
tr=pd.read_parquet(m.inputs()['train']); X=tr.drop(columns='y')
model=m.fit_record(LogisticRegression(max_iter=1000),X,'train',y=tr['y']); m.save_model(model,'model.pkl')

