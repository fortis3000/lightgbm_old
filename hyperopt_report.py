"""Visualize hyperopt optimization results"""

import re
import json
import pandas as pd
from pandas.plotting import parallel_coordinates

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

if __name__ == "__main__":

    results = pd.DataFrame()

    with open(r"PATH_TO_MODEL_FOLDER/logs.txt", "r",) as f:

        for line in f.readlines():

            line = " ".join(line.split()[2:])

            if line.startswith("Metrics"):
                metrics, params = line.split(", params: ")

                metrics = float(metrics.rstrip().split()[-1].rstrip())

                params = json.loads(params.strip().replace("'", '"'))

                params["metrics"] = metrics

                params["n_estimators"] /= 100
                params["num_leaves"] /= 10

                results = results.append(pd.DataFrame([params]))

    results = results.sort_values("metrics", ascending=False)
    results = results.iloc[-10:, :]

    # Make the plot
    numeric_columns = [
        "max_depth",
        "min_child_samples",
        "n_estimators",
        "num_leaves",
        "reg_alpha",
        "reg_lambda",
        "tweedie_variance_power",
        "metrics",
    ]
    parallel_coordinates(
        results[numeric_columns], "metrics", colormap=plt.get_cmap("Blues")
    )
    plt.show()
