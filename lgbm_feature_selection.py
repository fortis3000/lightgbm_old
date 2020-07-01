import numpy as np
import pandas as pd

# import lightgbm as lgb
import optuna.integration.lightgbm as lgb

from sklearn.model_selection import TimeSeriesSplit

from transfer_df import df
from data_config import cut_target

import pickle
import json
from lgbm_regr import get_dt_str

if __name__ == "__main__":

    base_length = 30490
    n_days_test = 14

    X_TEST, Y_TEST = cut_target(df.iloc[0 : base_length * n_days_test, :])  # last days
    X_TRAIN, Y_TRAIN = cut_target(df.iloc[base_length * n_days_test :, :])  # using CV

    ds_train = lgb.Dataset(data=X_TRAIN, label=Y_TRAIN)
    ds_val = lgb.Dataset(data=X_TEST, label=Y_TEST)

    params = {
        "objective": "poisson",  # regression,
        "learning_rate": 0.01,
        "random_state": 42,
        "seed": 42,
        "boosting": "gbdt",
        "metric": "rmse",  # ["rmse", "tweedie"],
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "gpu_use_dp": False,  # float32 instead of 64
        "n_jobs": 2,
        # "early_stopping_rounds": 50,
    }

    max_depth = int(np.sqrt(len(X_TRAIN.keys())))
    num_leaves = 2 ** max_depth if max_depth <= 10 else 2 ** 10

    SPACE = {
        # "learning_rate": hyperopt.hp.uniform("learning_rate", 0.00001, 0.01),
        "max_depth": -1,  # max_depth,
        "num_leaves": 128,  # num_leaves,
        "n_estimators": 1000,
        # "max_bin": 512,
        # "boosting": hyperopt.hp.choice("boosting", ["gbdt", "dart", "goss"]),
        # "colsample_bytree": hyperopt.hp.uniform("colsample_bytree", 0.1, 6.0),
        # "neg_bagging_fraction": hyperopt.hp.uniform("neg_bagging_fraction", 0.1, 0.9),
        # "pos_bagging_fraction": hyperopt.hp.uniform("pos_bagging_fraction", 0.1, 0.9),
        "reg_alpha": 5,
        "reg_lambda": 10,
        # "min_data_in_leaf": hyperopt.hp.choice("min_data_in_leaf", range(100, 10000)),
        # "min_child_samples": 5,
        # "min_child_weight": hyperopt.hp.uniform("min_child_weight", 0.001, 0.9),
        # "min_split_gain": hyperopt.hp.uniform("min_split_gain", 0.0, 0.9),
        # "tweedie_variance_power": 1.5,
    }

    params.update(SPACE)

    TSCV = TimeSeriesSplit(n_splits=3)

    best_params, tuning_history = dict(), list()
    model = lgb.train(
        params=params,
        train_set=ds_train,
        valid_sets=[ds_train, ds_val],
        num_boost_round=1000000,
        # folds=TSCV,
        # shuffle=False,
        early_stopping_rounds=100,
        best_params=best_params,
        tuning_history=tuning_history,
    )

    print("Best Params:", best_params)
    print("Tuning history:", tuning_history)

    with open(f"booster_{get_dt_str()}.json", "w") as f:
        f.write(json.dumps(model.dump_model()))
    model.save_model(f"booster_{get_dt_str()}.txt")
