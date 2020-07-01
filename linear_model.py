"""Train linear model"""
import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import RidgeCV, Ridge, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from data_config import *
from initial_data import d_cat_id

from lgbm_regr import get_dt_str

##########
# Data upload
##########


def upload_data(data_path, data_type="csv", features=None, remove_target_zeros=False):

    # features = pd.read_csv(
    #     "linear_model_importances_2020-05-03_17:29:52.csv", index_col=0
    # )

    # df = (
    #     pd.read_csv(
    #         "data/73_days_clear.csv",
    #         index_col=0,
    #         usecols=list(features[features.importance >= 0.05].index)
    #         + ["store_id", "item_id", "target"]
    #         # + ["target"],
    #     ).fillna(0)
    #     # .astype(pd.SparseDtype(np.float32, fill_value=0))
    # )

    # df = pd.read_pickle(data_path, compression="xz")  # .iloc[
    #     :2000000, :
    # ]  # last 30490*14days = 426860 records for validation

    from transfer_df import df

    base_length = 30490

    X_TEST, Y_TEST = cut_target(df.iloc[0 : base_length * 14, :])  # last two week
    X_TRAIN, Y_TRAIN = cut_target(df.iloc[base_length * 14 : base_length * 28, :])

    if remove_target_zeros:
        X_TRAIN = X_TRAIN[Y_TRAIN != 0]
        Y_TRAIN = Y_TRAIN[Y_TRAIN != 0]

    return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST


#####
# LEARING
#####
tss = TimeSeriesSplit(n_splits=10, max_train_size=None)

metrics = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "r2": r2_score,
    "rmsse": rmsse,
}


def learn_model(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, random_seed=42, save=True):

    # Init model
    model = ElasticNetCV(
        l1_ratio=np.arange(0, 0.8, 0.05),
        eps=1e-4,
        alphas=np.arange(0, 1.5, 0.05),
        fit_intercept=True,
        normalize=True,  # try to False
        max_iter=2000,
        cv=tss,
        n_jobs=-1,
        random_state=random_seed,
    )

    # Learning
    print("Start learning with CV")

    model = model.fit(X_TRAIN.iloc[::-1], Y_TRAIN.iloc[::-1])

    best_params = {"alpha": model.alpha_, "l1_ratio": model.l1_ratio_}

    print(f"Best model params: {best_params}")

    if save:
        with open(f"models/elastic_model_{get_dt_str()}", "wb") as f:
            pickle.dump(model, file=f)
    return model


def evaluate(preds, Y_TEST, metrics):
    """Calculate metrics provided"""
    return {x: y(y_pred=preds, y_true=Y_TEST) for x, y in metrics.items()}


def get_importances(model, list_of_features, save=True):
    ##########
    # Features importances
    ##########

    feature_importances = pd.DataFrame(
        {"feature": list_of_features, "importance": model.coef_},
    )

    feature_importances["importance"] = feature_importances["importance"].apply(np.abs)

    feature_importances.sort_values("importance", ascending=False).to_csv(
        f"linear_model_importances_{get_dt_str()}.csv",
        sep=",",
        encoding="utf-8",
        index=False,
        header=True,
    )

    if save:
        with open("models/elastic_model_" + get_dt_str(), "wb") as f:
            pickle.dump(model, f)


def get_predictions(model, x_test):
    ##########
    # Prediction
    ##########
    return model.predict(x_test)


def main(random_seed, remove_target_zeros=False):

    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = upload_data(
        data_path=r"data/100_days_ext.pkl.compress",
        remove_target_zeros=remove_target_zeros,
    )

    model = learn_model(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, random_seed=random_seed)

    best_metrics = evaluate(model.predict(X_TEST), Y_TEST, metrics)
    print(best_metrics)
    get_importances(model, X_TRAIN.keys())


if __name__ == "__main__":

    main(random_seed=42, remove_target_zeros=False)
    # main(random_seed=42, remove_target_zeros=False)
