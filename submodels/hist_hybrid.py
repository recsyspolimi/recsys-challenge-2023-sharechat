import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm.notebook import tqdm

#sys.path.append("..")
from utils.rerank_predictions import rerank_predictions

TRAIN_VAL_DATA_PATH = "data/train_valEncoded.parquet"
TEST_DATA_PATH = "data/testEncoded.parquet"


def exec_hist_hybrid():
    print("importing datasets and preprocessing...")
    df: pd.DataFrame = pd.read_parquet(TRAIN_VAL_DATA_PATH)
    df = df.astype({f"f_{i}": "category" for i in range(2, 33)})
    df = df.astype({"f_1": "int"})

    booleans = range(33, 41 + 1)
    for i in booleans:
        df = df.astype({"f_{}".format(i): "int"})

    df = df.astype({"is_clicked": "int"})
    df = df.astype({"is_installed": "int"})

    df["f_9"] = df["f_9"].astype(str)
    df["f_9"].mask(
        (df["f_1"] == 45) | (df["f_1"] == 52) | (df["f_1"] == 59) | (df["f_1"] == 66),
        "1",
        inplace=True,
    )
    df["f_9"].mask((df["f_1"] == 46) | (df["f_1"] == 53) | (df["f_1"] == 60), "2", inplace=True)
    df["f_9"].mask((df["f_1"] == 47) | (df["f_1"] == 54) | (df["f_1"] == 61), "3", inplace=True)
    df["f_9"].mask((df["f_1"] == 48) | (df["f_1"] == 55) | (df["f_1"] == 62), "4", inplace=True)
    df["f_9"].mask((df["f_1"] == 49) | (df["f_1"] == 56) | (df["f_1"] == 63), "5", inplace=True)
    df["f_9"].mask((df["f_1"] == 50) | (df["f_1"] == 57) | (df["f_1"] == 64), "6", inplace=True)
    df["f_9"].mask((df["f_1"] == 51) | (df["f_1"] == 58) | (df["f_1"] == 65), "0", inplace=True)
    df["f_9"] = df["f_9"].astype("category")

    df["f_new43-51"] = df["f_43"] * df["f_51"]
    df["f_new43-66"] = df["f_43"] * df["f_66"]
    df["f_new43-70"] = df["f_43"] * df["f_70"]
    df["f_new51-70"] = df["f_51"] * df["f_70"]
    df["f_new51-66"] = df["f_51"] * df["f_66"]
    df["f_new66-70"] = df["f_66"] * df["f_70"]

    categorical_columns: List[str] = [f"f_{i}" for i in range(2, 33)]

    light_preds = pd.read_csv("predictions/light_train_predictions_all_days.csv", sep="\t")
    hist_preds = pd.read_csv("predictions/hist_train_predictions_all_days.csv", sep="\t")

    # change rowId to f_0
    light_preds = light_preds.rename(columns={"RowId": "f_0"})
    hist_preds = hist_preds.rename(columns={"RowId": "f_0"})

    # change is_installed to light and hist
    light_preds = light_preds.rename(columns={"is_installed": "light"})
    hist_preds = hist_preds.rename(columns={"is_installed": "hist"})

    df = df.merge(hist_preds, on="f_0")
    df = df.merge(light_preds, on="f_0")

    to_delete = ["f_0", "f_7", "f_11", "f_27", "f_28", "f_29", "f_59", "f_64"]
    df = df.drop(columns=to_delete)

    # remove rows with f_1=48,50,52
    df = df[df.f_1 != 48]
    df = df[df.f_1 != 50]
    df = df[df.f_1 != 52]

    # remove those columns from the list categorical_columns:['f_7','f_11','f_27','f_28','f_29'])
    for col in ["f_7", "f_11", "f_27", "f_28", "f_29"]:
        categorical_columns.remove(col)

    # get unique values for each categorical column
    categorical_dims = {}
    for col in categorical_columns:
        categorical_dims[col] = df[col].nunique()

    categorical_index = {}
    for col in categorical_columns:
        categorical_index[col] = df.columns.get_loc(col)
    # keep only the values of the dictionary
    categorical_index = list(categorical_index.values())

    X = df.drop(columns=["is_clicked", "is_installed"])
    y = df[["is_clicked", "is_installed"]]

    max_bins = 229
    hyper_params = {
        "loss": "log_loss",
        "l2_regularization": 79.97112528034731,
        "learning_rate": 0.10659053116784906,
        "max_depth": 12,
        "max_leaf_nodes": 130,
        "min_samples_leaf": 184,
        "max_bins": max_bins,
        "max_iter": 261,
        "verbose": 0,
        "random_state": 0,
    }

    categorical_columns1 = categorical_columns.copy()
    # remove from categorical_columns1 those columns with more features than max_bins
    for col in categorical_columns:
        if categorical_dims[col] >= max_bins:
            categorical_columns1.remove(col)

    print("fitting hist model...")
    clf1 = HistGradientBoostingClassifier(**hyper_params, categorical_features=categorical_columns1)
    clf1.fit(X, y["is_installed"])

    hist_preds_test = pd.read_csv("predictions/hist_predictions.csv", sep="\t")
    light_preds_test = pd.read_csv("predictions/light_predictions.csv", sep="\t")

    # change rowId to f_0
    light_preds_test = light_preds_test.rename(columns={"RowId": "f_0"})
    hist_preds_test = hist_preds_test.rename(columns={"RowId": "f_0"})

    # change is_installed to light and hist
    light_preds_test = light_preds_test.rename(columns={"is_installed": "light"})
    hist_preds_test = hist_preds_test.rename(columns={"is_installed": "hist"})

    light_preds_test = light_preds_test.drop(columns=["is_clicked"])
    hist_preds_test = hist_preds_test.drop(columns=["is_clicked"])

    test: pd.DataFrame = pd.read_parquet(TEST_DATA_PATH)
    test = test.astype({f"f_{i}": "category" for i in range(2, 33)})
    test["f_new43-51"] = test["f_43"] * test["f_51"]
    test["f_new43-66"] = test["f_43"] * test["f_66"]
    test["f_new43-70"] = test["f_43"] * test["f_70"]
    test["f_new51-70"] = test["f_51"] * test["f_70"]
    test["f_new51-66"] = test["f_51"] * test["f_66"]
    test["f_new66-70"] = test["f_66"] * test["f_70"]

    test["f_9"] = test["f_9"].astype(str)
    test["f_9"].mask((test["f_1"] == 67), "2", inplace=True)
    test["f_9"] = test["f_9"].astype("category")

    test = test.astype({"f_1": "int"})

    test = test.merge(hist_preds_test, on="f_0")
    test = test.merge(light_preds_test, on="f_0")

    test = test.drop(columns=to_delete)

    booleans = range(33, 41 + 1)
    for i in booleans:
        test = test.astype({"f_{}".format(i): "int"})

    print("generating predictions...")
    results_installed = clf1.predict_proba(test)
    test1: pd.DataFrame = pd.read_parquet(TEST_DATA_PATH)
    submission = pd.DataFrame()
    submission["RowId"] = test1["f_0"]
    submission["is_clicked"] = 0.33
    submission["is_installed"] = pd.Series(results_installed[:, 1])
    submission = rerank_predictions(submission)
    submission.to_csv("predictions/hist_hybrid_predictions.csv", sep="\t", index=False)
