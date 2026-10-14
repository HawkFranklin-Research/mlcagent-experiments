import mlca_runtime as rt
from mlca_runtime.reference import run
def fit_record(*args, **kwargs):
    return rt.fit_record(*args, **kwargs)
run("retain_features", fit_record)
