import json, pickle, pandas as pd, mlca_runtime as m
i=m.inputs(); model=pickle.load(open(i["latest_model"],"rb")); final=model[-1] if hasattr(model,"steps") else model
names=list(getattr(final,"feature_names_in_",[])); values=getattr(final,"coef_",getattr(final,"feature_importances_",[]))
import numpy as np
values=np.asarray(values); values=np.abs(values[0]) if values.ndim>1 else np.abs(values)
if len(names)!=len(values): names=[f"feature_{j}" for j in range(len(values))]
m.save_table(pd.DataFrame({"feature":names,"importance_mean":values,"importance_sd":np.zeros(len(values))}).head(15),'interpretation.csv')

