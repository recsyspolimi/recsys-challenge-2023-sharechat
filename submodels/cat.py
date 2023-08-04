import os
from typing import Callable, Dict, List, Set, Tuple

import pandas as pd
from catboost import CatBoostClassifier

from utils.preprocessing import (
    remove_outliers,
    trigonometric_date_encoding,
)


def exec_cat():
    GPU = True

    task_type = "GPU" if GPU else "CPU"

    print("importing datasets...")
    df_train_val: pd.DataFrame = pd.read_parquet("data/train_val.parquet").reset_index(drop=True)
    df_train_val = df_train_val.astype({f"f_{i}": "category" for i in range(2, 33)})
    df_train_val = df_train_val.astype({"f_1": "int"})

    df_train_val = df_train_val.astype({"is_clicked": "int"})
    df_train_val = df_train_val.astype({"is_installed": "int"})

    df_test = pd.read_parquet("data/test.parquet").reset_index(drop=True)
    df_test = df_test.astype({f"f_{i}": "category" for i in range(2, 33)})
    df_test = df_test.astype({"f_1": "int"})

    df_train_val["f_30"] = df_train_val["f_30"].astype("str")
    df_train_val["f_31"] = df_train_val["f_31"].astype("str")
    df_train_val["f_30"] = df_train_val["f_30"].replace("0.0", "0")
    df_train_val["f_31"] = df_train_val["f_31"].replace("0.0", "0")
    df_train_val["f_30"] = df_train_val["f_30"].replace("1.0", "1")
    df_train_val["f_31"] = df_train_val["f_31"].replace("1.0", "1")
    df_train_val["f_30"] = df_train_val["f_30"].astype("category")
    df_train_val["f_31"] = df_train_val["f_31"].astype("category")
    df_train_val["f_30"] = df_train_val["f_30"].replace("nan", -2)
    df_train_val["f_31"] = df_train_val["f_31"].replace("nan", -2)
    df_train_val["f_30"] = df_train_val["f_30"].replace("0", 0)
    df_train_val["f_30"] = df_train_val["f_30"].replace("1", 1)
    df_train_val["f_31"] = df_train_val["f_31"].replace("0", 0)
    df_train_val["f_31"] = df_train_val["f_31"].replace("1", 1)

    df_test["f_30"] = df_test["f_30"].astype("str")
    df_test["f_31"] = df_test["f_31"].astype("str")
    df_test["f_30"] = df_test["f_30"].replace("0.0", "0")
    df_test["f_31"] = df_test["f_31"].replace("0.0", "0")
    df_test["f_30"] = df_test["f_30"].replace("1.0", "1")
    df_test["f_31"] = df_test["f_31"].replace("1.0", "1")
    df_test["f_30"] = df_test["f_30"].astype("category")
    df_test["f_31"] = df_test["f_31"].astype("category")
    df_test["f_30"] = df_test["f_30"].replace("nan", -2)
    df_test["f_31"] = df_test["f_31"].replace("nan", -2)
    df_test["f_30"] = df_test["f_30"].replace("0", 0)
    df_test["f_30"] = df_test["f_30"].replace("1", 1)
    df_test["f_31"] = df_test["f_31"].replace("0", 0)
    df_test["f_31"] = df_test["f_31"].replace("1", 1)

    boolean_columns: List[str] = [f"f_{i}" for i in range(33, 42)]

    # convert boolean columns to bool otherwise catboost will throw an error
    for col in boolean_columns:
        df_train_val[col] = df_train_val[col].astype(bool)
        df_test[col] = df_test[col].astype(bool)

    CATEGORICAL_TO_DROP: list = [
        "f_7",
        "f_9",
        "f_11",
        "f_23",
        "f_24",
        "f_25",
        "f_26",
        "f_27",
        "f_28",
        "f_29",
    ]

    NUMERICAL_TO_DROP: list = [
        "f_55",
        "f_59",
        "f_64",
        "f_65",
        "f_66",
    ]

    NUMERICAL_NON_COUNTERS: List[str] = [
        "f_43",
        "f_51",
        "f_58",
        "f_59",
        "f_64",
        "f_65",
        "f_66",
        "f_67",
        "f_68",
        "f_69",
        "f_70",
    ]

    categorical_columns: List[str] = [f"f_{i}" for i in range(2, 32 + 1)]
    numerical_columns: List[str] = [f"f_{i}" for i in range(42, 79 + 1)]
    categorical_columns = [col for col in categorical_columns if col not in CATEGORICAL_TO_DROP]
    numerical_columns = [
        col
        for col in numerical_columns
        if col not in NUMERICAL_TO_DROP and col in NUMERICAL_NON_COUNTERS
    ]
    counter_columns: List[str] = [f"f_{i}" for i in range(42, 79 + 1)]
    counter_columns = [
        col
        for col in counter_columns
        if col not in NUMERICAL_TO_DROP and col not in NUMERICAL_NON_COUNTERS
    ]

    def preprocess_data_nico(
        df_train: pd.DataFrame, df_val: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Drop bad columns...")
        df_train = df_train.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
        df_val = df_val.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)

        print("Encoding data...")
        df_train = trigonometric_date_encoding(df_train, column="f_1")
        df_val = trigonometric_date_encoding(df_val, column="f_1")

        print("Removing outliers from numerical columns...")
        means: pd.Series = df_train[numerical_columns].mean()
        stds: pd.Series = df_train[numerical_columns].std()
        df_train = remove_outliers(
            df=df_train,
            columns=numerical_columns,
            coefficient=4,
            means=means,
            stds=stds,
        )
        df_val = remove_outliers(
            df=df_val,
            columns=numerical_columns,
            coefficient=4,
            means=means,
            stds=stds,
        )

        return df_train, df_val

    cat_params = {
        "iterations": 999,
        "learning_rate": 0.09976360026551613,
        "depth": 7,
        "l2_leaf_reg": 0.00036508650695930443,
        "bootstrap_type": "Bayesian",
        "random_strength": 1.3686373913335717e-07,
        "bagging_temperature": 0.5157698860311479,
        "od_type": "Iter",
        "od_wait": 21,
        "border_count": 89,
        "has_time": True,
    }

    print("Preprocessing data...")
    df_train_val, df_test = preprocess_data_nico(df_train_val, df_test)

    if cat_params["has_time"]:
        df_train_val = df_train_val.sort_values(by="f_1", ascending=True).reset_index(drop=True)

    y_train = df_train_val["is_installed"]
    test_row_id = df_test["f_0"]
    X_train = df_train_val.drop(columns=["f_0", "f_1", "is_installed", "is_clicked"])
    df_test = df_test.drop(columns=["f_0", "f_1"])

    cat_clf = CatBoostClassifier(**cat_params, task_type=task_type, eval_metric="CrossEntropy")

    cat_clf.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        verbose=True,
    )
    cat_preds_proba = cat_clf.predict_proba(df_test)

    print("CatBoost model is fitted: " + str(cat_clf.is_fitted()))
    print("CatBoost model parameters:")
    print(cat_clf.get_params())

    submission = pd.DataFrame(
        {"RowId": test_row_id, "is_clicked": 0.0, "is_installed": cat_preds_proba[:, -1]}
    )
    os.system("mkdir predictions")
    submission.to_csv("predictions/cat_predictions.csv", index=False, sep="\t")
