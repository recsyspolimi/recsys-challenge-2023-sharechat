import os
import sys
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

#sys.path.append("..")
from utils.rerank_predictions import rerank_predictions


def exec_light():
    print("importing datasets and preprocessing...")
    print(os.listdir())
    df: pd.DataFrame = pd.read_parquet("data/train_val.parquet")
    df = df.astype({f"f_{i}": "category" for i in range(2, 33)})
    df = df.astype({"f_1": "int"})

    df["f_9"] = df["f_9"].astype(str)
    df["f_9"].mask(
        (df["f_1"] == 45) | (df["f_1"] == 52) | (df["f_1"] == 59) | (df["f_1"] == 66),
        "6675",
        inplace=True,
    )
    df["f_9"].mask((df["f_1"] == 46) | (df["f_1"] == 53) | (df["f_1"] == 60), "14659", inplace=True)
    df["f_9"].mask((df["f_1"] == 47) | (df["f_1"] == 54) | (df["f_1"] == 61), "9638", inplace=True)
    df["f_9"].mask((df["f_1"] == 48) | (df["f_1"] == 55) | (df["f_1"] == 62), "23218", inplace=True)
    df["f_9"].mask((df["f_1"] == 49) | (df["f_1"] == 56) | (df["f_1"] == 63), "869", inplace=True)
    df["f_9"].mask((df["f_1"] == 50) | (df["f_1"] == 57) | (df["f_1"] == 64), "21533", inplace=True)
    df["f_9"].mask((df["f_1"] == 51) | (df["f_1"] == 58) | (df["f_1"] == 65), "31372", inplace=True)
    df["f_9"] = df["f_9"].astype("category")
    df["f_new43-51"] = df["f_43"] * df["f_51"]
    df["f_new43-66"] = df["f_43"] * df["f_66"]
    df["f_new43-70"] = df["f_43"] * df["f_70"]
    df["f_new51-70"] = df["f_51"] * df["f_70"]
    df["f_new51-66"] = df["f_51"] * df["f_66"]
    df["f_new66-70"] = df["f_66"] * df["f_70"]

    booleans = range(33, 41 + 1)
    for i in booleans:
        df = df.astype({"f_{}".format(i): "int"})

    df = df.astype({"is_clicked": "int"})
    df = df.astype({"is_installed": "int"})

    categorical_columns: List[str] = [f"f_{i}" for i in range(2, 33)]
    to_delete = ["f_0", "f_7", "f_11", "f_27", "f_28", "f_29"]
    df = df.drop(columns=to_delete)

    X = df.drop(columns=["is_clicked", "is_installed"])
    y = df[["is_clicked", "is_installed"]]

    light_params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "xentropy",
        "learning_rate": 0.025199403190325702,
        "max_depth": 284,
        "num_leaves": 300,
        "num_iterations": 473,
        "cat_smooth": 150,
        "cat_l2": 0,
        "verbose": -1,
    }

    X_rem = X[(X["f_1"] != 52) & (X["f_1"] != 48) & (X["f_1"] != 50)]
    y_rem = y[(X["f_1"] != 52) & (X["f_1"] != 48) & (X["f_1"] != 50)]["is_installed"]
    print("fitting light model...")
    lgb_train = lgb.Dataset(X_rem, y_rem)
    gbm_installed_old = lgb.train(light_params, lgb_train)

    test: pd.DataFrame = pd.read_parquet("data/test.parquet")
    test = test.astype({f"f_{i}": "category" for i in range(2, 33)})
    test = test.drop(columns=to_delete)

    test["f_new43-51"] = test["f_43"] * test["f_51"]
    test["f_new43-66"] = test["f_43"] * test["f_66"]
    test["f_new43-70"] = test["f_43"] * test["f_70"]
    test["f_new51-70"] = test["f_51"] * test["f_70"]
    test["f_new51-66"] = test["f_51"] * test["f_66"]
    test["f_new66-70"] = test["f_66"] * test["f_70"]

    test["f_9"] = test["f_9"].astype(str)
    test["f_9"].mask((test["f_1"] == 67), "14659", inplace=True)
    test["f_9"] = test["f_9"].astype("category")

    test = test.astype({"f_1": "int"})

    booleans = range(33, 41 + 1)
    for i in booleans:
        test = test.astype({"f_{}".format(i): "int"})

    print("generating predictions...")
    results_installed_old = gbm_installed_old.predict(test)

    test1: pd.DataFrame = pd.read_parquet("data/test.parquet")
    submission = pd.DataFrame()
    submission["RowId"] = test1["f_0"]
    submission["is_clicked"] = 0.11
    submission["is_installed"] = pd.Series(results_installed_old)
    submission = rerank_predictions(submission)
    submission.to_csv("predictions/light_predictions.csv", sep="\t", index=False)
