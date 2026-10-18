import os
from pathlib import Path


def apply() -> None:
    import matplotlib.pyplot as plt

    path = os.environ.get("MLCA_STYLE")
    if path and Path(path).is_file():
        plt.style.use(path)
