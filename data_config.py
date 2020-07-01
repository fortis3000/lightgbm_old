"""Prepare data for correct import from other scripts"""

import re

import numpy as np
import pandas as pd


#####
# SERVICE
#####


def reduce_mem_usage(props):
    """Reduce memory spended on pandas DataFrame
    https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    """
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = props[col] - asint
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist


def safe_drop(df, columns):
    return df[set(df.keys()) - set(columns)]


#####
# SALES
#####


def get_category(column_name):
    return column_name.split("_")[0]


def get_days(df):
    """Get day columns with pattern provided"""
    return list(filter(lambda x: re.match(r"d_\d", x), df.columns))


def get_firstday(timseries):
    """Find first day of the product sale"""
    for num, i in enumerate(timseries):
        if i != 0:
            return num


def get_day_number(column_name):
    """Returns day number from column name"""
    return int(column_name.split("_")[-1])


def get_column_name(day_number):
    """Returns column name from day number provided"""
    return "d_" + str(int(day_number))


def get_max_day(df):
    """Returns column name for latest day in dataframe"""
    sorted(get_days(df), key=lambda x: int(x.split("_")[-1]), reverse=True)[0]


def get_prev(day_string, days_ago=1, period=False):
    """Returns day column name from period provided

    Use days_ago = 
        1 for previous day,
        6 or 7 for previous week,
        28 for previous month
    """
    temp = day_string.split("_")

    if period:
        "_".join([temp[0], str(int(temp[1]) - days_ago)])
        return [
            "_".join(["d", str(x)])
            for x in range(
                int(day_string.split("_")[1]) - days_ago, int(day_string.split("_")[1])
            )
        ]
    else:
        return "_".join([temp[0], str(int(temp[1]) - days_ago)])


def cumsum_except_zeros(ts):
    """Accumulated total for non-zero sales to exclude warehouse imfluence"""
    cumsum = 0
    out = []
    for i in ts:
        cumsum = 0 if i == 0 else cumsum + i
        out.append(cumsum)
    return out


#####
# PRICES
#####


#####
# MODELLING
#####


def read_csv(filename):
    return (
        pd.read_csv(filename, index_col=["store_id", "item_id"])
        .fillna(0)
        .astype(pd.SparseDtype("float32", 0))
    )


def cut_target(df):
    return df.drop("target", axis=1), df["target"]


def rmsse(y_true, y_pred):

    assert len(y_true) == len(y_pred), "y_true and y_pred is not of the same length"

    h = len(y_true)

    n = 100  # 1914

    squared_error_sum = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)

    return np.sqrt(1 / h * squared_error_sum / (1 / (n - 1) * squared_error_sum))


#####
# PREDICTION
#####
# TODO: create prediction module
def custom_round(float_value, threshold=0.5):
    """Round float_value to integer using custom threshold"""

    if float_value <= 0:
        return 0

    int_part = int(float_value)

    res = -1 * (int_part - float_value)

    return int_part if res <= threshold else int_part + 1


def find_thres(preds, real_values, threshold_range, metric_to_follow="mse"):
    """Brute-force method to choose the best round threshold"""

    best_metric_to_follow = np.inf  # -np.inf dor maximizing task
    best_t = 0

    best_metrics = {}

    for t in threshold_range:

        current_metrics = evaluate(
            list(map(lambda x: custom_round(x, t), preds)), real_values, metrics
        )

        if current_metrics[metric_to_follow] <= best_metric_to_follow:
            best_metrics = current_metrics
            best_t = t

    return best_t, best_metrics


def postprocess_predictions(preds_df, choose_threshold=False):
    # step by step predictions using sales dataframe
    global sales

    if choose_threshold:
        thresholds = []

        for col in preds_df.keys():
            t, _ = find_thres(preds_df[col], sales[col], np.arange(0.1, 1, 0.05))
            thresholds.append(t)

        threshold_found = np.median(thresholds)

        print("Thresholds", thresholds)
        print(f"Applying threshold {threshold_found}")

        preds_df = preds_df.applymap(
            lambda x: custom_round(x, threshold=threshold_found)
        )

    return preds_df

