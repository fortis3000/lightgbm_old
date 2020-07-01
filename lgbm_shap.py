"""Module for choosing the most important features using SHAP"""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

import matplotlib.pyplot as plt

import shap


def save_shap_importances(df, shap_values):
    vals = np.abs(shap_values).mean(0)
    print(vals.shape)
    feature_importance = pd.DataFrame(
        list(zip(df.drop("target", axis=1).columns, vals)),
        columns=["col_name", "feature_importance_vals"],
    )
    feature_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )

    feature_importance.to_csv("shap_importance.csv", index=False, header=True)


if __name__ == "__main__":

    path = os.path.join("models", "lgbm_2020-06-16_20:21:23")

    model = lgb.Booster(model_file=os.path.join(path, "booster.txt"))

    #####
    # All importances
    #####

    print(model.feature_name())
    print(model.feature_importance())

    lgb.plot_importance(model)
    plt.show()

    #####
    # Best features
    #####

    max_importance = max(model.feature_importance())
    features_important = list(
        filter(
            lambda x: x[1] >= 15,  # 0.1 * max_importance,
            zip(model.feature_name(), model.feature_importance()),
        )
    )

    print(sorted([x[0] for x in features_important]))

    #####
    # DATA
    #####

    # shap.initjs()

    from transfer_df import df

    df = df.iloc[: 30490 * 14, :]

    # workaround
    model.params["objective"] = "tweedie"

    #####
    # SHAP
    #####
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        df.drop("target", axis=1)
    )  # , approximate=True)

    print("SHAP values calculated")

    with open("lgbm_shap_importances_ext_poisson.pkl", "wb") as f:
        pickle.dump(shap_values, f)

    with open("lgbm_shap_importances_ext.pkl", "rb") as f:
        shap_values = pickle.load(f)

    save_shap_importances(df, shap_values)

    shap.summary_plot(
        shap_values, df.drop("target", axis=1), max_display=30, plot_size="auto"
    )

    shap.dependence_plot("week_mean", shap_values, df.drop("target", axis=1))

    # needs a lot of memory
    shap.force_plot(explainer.expected_value, shap_values, df.drop("target", axis=1))

    plt.savefig("shap30.jpg")

    ####
    # LGBM SHAP
    #####
    shaps = model.predict(df.drop("target", axis=1), pred_contrib=True)

    shap_importances = np.median(np.abs(shaps[:, :-1]), axis=0)

    df_importances = pd.DataFrame(
        {"feature": df.drop("target", axis=1).keys(), "importance": shap_importances}
    ).sort_values("importance", ascending=False)

    df_importances.to_csv(
        os.path.join(path, "lgbm_shap_importances_median.csv"),
        index=False,
        header=True,
        sep=",",
    )
