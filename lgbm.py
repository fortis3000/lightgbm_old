"""The module realizes LightGBM model creating, validating, learning and predicting"""

##### Imports
import datetime as dt
import pickle
from copy import copy
import logging
import os
import shutil
import os.path as osp
import json

import numpy as np
import pandas as pd

import lightgbm as lgb
import hyperopt

import yaml

from sklearn.model_selection import (
    ShuffleSplit,
    TimeSeriesSplit,
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
from sklearn.exceptions import ConvergenceWarning

from data_config import cut_target, rmsse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


##### Service functions
def get_dt_str():
    """Returns current date and time in string format"""
    return str(dt.datetime.now()).split(".")[0].replace(" ", "_")


def load_config(config_file):
    """Loading YAML config file and parsing to the dictionary"""
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


#####
# Model folder
#####
path = "models"
folder_name = f"lgbm_{get_dt_str()}"
path = os.path.join(path, folder_name)
os.mkdir(path)

##########
# LOAD CONFIGS
##########
CONFIG_FILE = "config.yml"
CONFIG = load_config(CONFIG_FILE)
LOG_FILE = osp.join(path, "logs.txt")
logging.basicConfig(
    filename=LOG_FILE,
    level=CONFIG["LOGS"]["LOGS_LEVEL"],
    format="%(asctime)s: %(message)s",
)

shutil.copy2(CONFIG_FILE, path)

##########
# Functions
##########
def fit_model(params, ds_train, ds_val=None):

    if ds_val is None:
        return lgb.train(
            params=params,
            train_set=ds_train,
            num_boost_round=CONFIG["LEARNING"]["NUM_BOOST_ROUND"],
        )
    else:
        return lgb.train(
            params=params,
            train_set=ds_train,
            valid_sets=[ds_val],
            num_boost_round=CONFIG["LEARNING"]["NUM_BOOST_ROUND"],
            early_stopping_rounds=CONFIG["LEARNING"]["EARLY_STOPPING_ROUNDS"],
        )


def predict_model(model, X_TEST, Y_TEST):
    return model.predict(X_TEST)


SERVICE_PARAMS = {
    "objective": CONFIG["OBJECTIVE"],
    "learning_rate": CONFIG["LEARNING"]["LEARNING_RATE"],
    "random_state": CONFIG["RANDOM_SEED"],
    "boosting": CONFIG["LEARNING"]["BOOSTING"],
    "metric": [CONFIG["LEARNING"]["LEARN_METRICS_STRING"], "tweedie"],
    "device": CONFIG["DEVICE"],
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "n_jobs": CONFIG["LEARNING"]["N_JOBS"],
    "first_metric_only": False,
}


def objective(params):
    """Creates objective function to optimize"""

    params.update(SERVICE_PARAMS)

    res = lgb.cv(
        params=params,
        train_set=ds_train,
        num_boost_round=CONFIG["LEARNING"]["NUM_BOOST_ROUND"],
        folds=TSCV,
        shuffle=False,
        early_stopping_rounds=CONFIG["LEARNING"]["EARLY_STOPPING_ROUNDS"],
        seed=CONFIG["RANDOM_SEED"],
        metrics=list(
            set([CONFIG["LEARNING"]["LEARN_METRICS_STRING"], "rmse", "tweedie"])
        ),
    )

    score = np.median(res["rmse-mean"])

    logging.info(f"CV results: {params}")
    logging.info(f"Metrics: {score}, params: {params}")

    return score


if __name__ == "__main__":

    ##############################
    ### LOGGING
    ##############################
    logging.info("Script starts properly")
    logging.info(
        "Learning metrics: {}".format(CONFIG["LEARNING"]["LEARN_METRICS_STRING"])
    )
    logging.info(
        "Validation metrics: {}".format(CONFIG["VALIDATION"]["VALIDATE_METRICS_STRING"])
    )
    logging.info("Random seed: {}".format(str(CONFIG["RANDOM_SEED"])))
    logging.info("Hyperopt cycles: {}".format(str(CONFIG["LEARNING"]["HYPER_EVALS"])))

    ##############################
    ### DATASETS UPLOAD
    ##############################
    from transfer_df import df

    base_length = 30490
    n_days_test = 14

    X_TEST, Y_TEST = cut_target(df.iloc[0 : base_length * n_days_test, :])  # last days
    X_TRAIN, Y_TRAIN = cut_target(df.iloc[base_length * n_days_test :, :])  # using CV

    ds_train = lgb.Dataset(data=X_TRAIN, label=Y_TRAIN)
    ds_val = lgb.Dataset(data=X_TEST, label=Y_TEST)

    max_depth = max(3, int(np.sqrt(len(X_TRAIN.keys()))))
    num_leaves = 2 ** max_depth if max_depth <= 10 else 2 ** 10

    # To save RAM during training
    del df, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

    ##############################
    ## HYPERPARAMETERS TUNING
    ##############################

    ### Metrics and losses
    METRICS = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "r2": r2_score,
        "rmsse": rmsse,
    }

    # max_depth = int(np.sqrt(len(X_TRAIN.keys())))
    # num_leaves = 2 ** max_depth if max_depth <= 10 else 2 ** 10

    ### Hyperopt space
    SPACE = {
        # "learning_rate": hyperopt.hp.uniform("learning_rate", 0.00001, 0.01),
        "max_depth": hyperopt.hp.choice("max_depth", range(max_depth // 2, max_depth)),
        "num_leaves": hyperopt.hp.choice(
            "num_leaves", list(range(num_leaves // 4, num_leaves))
        ),
        "n_estimators": hyperopt.hp.choice("n_estimators", range(500, 2000)),
        # "boosting": hyperopt.hp.choice("boosting", ["gbdt", "dart", "goss"]),
        # "colsample_bytree": hyperopt.hp.uniform("colsample_bytree", 0.1, 6.0),
        # "neg_bagging_fraction": hyperopt.hp.uniform("neg_bagging_fraction", 0.1, 0.9),
        # "pos_bagging_fraction": hyperopt.hp.uniform("pos_bagging_fraction", 0.1, 0.9),
        "feature_fraction": hyperopt.hp.uniform("feature_fraction", 0.2, 0.9),
        "reg_alpha": hyperopt.hp.uniform("reg_alpha", 2, 10),
        "reg_lambda": hyperopt.hp.uniform("reg_lambda", 5, 20),
        # "min_data_in_leaf": hyperopt.hp.choice("min_data_in_leaf", range(100, 10000)),
        "min_child_samples": hyperopt.hp.choice("min_child_samples", range(2, 30)),
        # "min_child_weight": hyperopt.hp.uniform("min_child_weight", 0.001, 0.9),
        # "min_split_gain": hyperopt.hp.uniform("min_split_gain", 0.0, 0.9),
        "tweedie_variance_power": hyperopt.hp.uniform(
            "tweedie_variance_power", 1.0, 2.0
        ),
    }

    ### Cross-validator
    CV = ShuffleSplit(
        n_splits=CONFIG["VALIDATION"]["CV"]["N_SPLITS"],
        train_size=CONFIG["VALIDATION"]["CV"]["TRAIN_SIZE"],
        test_size=CONFIG["VALIDATION"]["CV"]["TEST_SIZE"],
        random_state=CONFIG["RANDOM_SEED"],
    )

    TSCV = TimeSeriesSplit(n_splits=CONFIG["VALIDATION"]["CV"]["N_SPLITS"])

    ##############################
    ##  LEARNING MODEL
    ##############################

    logging.info("Hyperopt optimization")

    # visibility bug in hyperopt
    PARAMS_OPTIMAL = hyperopt.fmin(
        fn=objective,
        space=SPACE,
        algo=hyperopt.tpe.suggest,
        max_evals=CONFIG["LEARNING"]["HYPER_EVALS"],
        rstate=np.random.RandomState(CONFIG["RANDOM_SEED"]),
    )

    SERVICE_PARAMS.update(PARAMS_OPTIMAL)
    PARAMS_OPTIMAL = SERVICE_PARAMS

    logging.info("Optimal parameters obtained")
    with open(os.path.join(path, f"params_optimal.pkl"), "wb") as f:
        pickle.dump(PARAMS_OPTIMAL, f)

    logging.info("Learning model with optimal parameters")
    MODEL = fit_model(PARAMS_OPTIMAL, ds_train=ds_train, ds_val=ds_val,)

    ### Saving model
    logging.info("Saving model")
    with open(os.path.join(path, f"booster.json"), "w") as f:
        f.write(json.dumps(MODEL.dump_model()))
    MODEL.save_model(os.path.join(path, f"booster.txt"))

    ### predictions on train and test
    logging.info("Get model predictions")
    from transfer_df import df

    base_length = 30490
    n_days_test = 14
    X_TEST, Y_TEST = cut_target(df.iloc[0 : base_length * n_days_test, :])  # last days
    X_TRAIN, Y_TRAIN = cut_target(df.iloc[base_length * n_days_test :, :])  # using CV

    preds_train = MODEL.predict(X_TRAIN)
    preds_test = MODEL.predict(X_TEST)

    train_metrics = {key: val(Y_TRAIN, preds_train) for key, val in METRICS.items()}
    test_metrics = {key: val(Y_TEST, preds_test) for key, val in METRICS.items()}

    logging.info(f"Final metrics on train set: {train_metrics}")
    logging.info(f"Final metrics on test set: {test_metrics}")

    if CONFIG["SAVE_ERRORS"]:
        with open(os.path.join(path, "errors_test.pkl"), "wb") as f:
            pickle.dump(preds_test - Y_TEST, f)
        with open(os.path.join(path, "errors_train.pkl"), "wb") as f:
            pickle.dump(preds_train - Y_TRAIN, f)

    ##############################
    ## Saving results
    ##############################
    logging.info("")
    if CONFIG["PREDICTION"]["SAVE_PREDICTIONS_TRAIN"]:
        logging.info("Saving model predictions on train")
        with open(os.path.join(path, f"lgmb_preds"), "wb") as f:
            pickle.dump(preds_train, f)

    if CONFIG["PREDICTION"]["SAVE_PREDICTIONS_TEST"]:
        logging.info("Saving model predictions on test")
        with open(os.path.join(path, f"lgmb_preds"), "wb") as f:
            pickle.dump(preds_test, f)
