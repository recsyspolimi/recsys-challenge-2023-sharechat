import os
import pickle
from typing import Callable, Dict, List, Set, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from dotenv import load_dotenv

from tqdm import tqdm

from utils.normalized_cross_entropy_loss import normalized_cross_entropy_loss
from utils.preprocessing import (
    encode_counters,
    remove_categories_not_in_both,
    remove_outliers,
    trigonometric_date_encoding,
)

load_dotenv()
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "default")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "default")
regiorn_name = "eu-west-1"
bucket_name = "challenge23"

s3 = boto3.resource(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    config=boto3.session.Config(region_name=regiorn_name),
)
bucket = s3.Bucket(bucket_name)

GPU = False

task_type = "GPU" if GPU else "CPU"

CATEGORICAL_TO_DROP: List[str] = [
    "f_7",
    "f_23",
    "f_24",
    "f_25",
    "f_26",
    "f_27",
    "f_28",
    "f_29",
]

NUMERICAL_TO_DROP: List[str] = [
    "f_59",
    "f_64",
    "f_66",
]


df = pd.read_parquet("data/train_val.parquet")
df = df.astype({f"f_{i}": "category" for i in range(2, 33)})
df = df.astype({'f_1': 'int'})

df = df.astype({'is_clicked': 'int'})
df = df.astype({'is_installed': 'int'})


boolean_columns = list(df.select_dtypes(['boolean']).columns)
for col in boolean_columns:
    df[col] = df[col].astype(bool)

cat_params = {
    "bagging_temperature": 8.453537008132315, 
    "bootstrap_type": "Bayesian", 
    "border_count": 39, 
    "depth": 10, 
    "drop_useless": False, 
    "iterations": 2422, 
    "l2_leaf_reg": 0.01711518718043709, 
    "learning_rate": 0.03231569998879843, 
    "random_strength": 1.9800642768375248e-05, 
    "start_day": 48,
    "has_time": True,
    "od_type": "Iter",
    "od_wait": 30,
}

start_day = cat_params.pop("start_day")

all_predictions = pd.DataFrame()
for day in tqdm(range(52, 67)):
    print("splitting data...")
    df_train = df[df["f_1"].between(start_day, day, inclusive="left")]
    df_val = df[df["f_1"].between(day, day + 1, inclusive="left")]

    print("Preprocessing data...")
    df_train = df_train.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
    df_val = df_val.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)

    if cat_params["has_time"]:
        X_train = X_train.sort_values(by="f_1", ascending=True).reset_index(drop=True)

    y_train = X_train["is_installed"]
    y_val = X_val["is_installed"]
    val_row_ids = df_val["f_0"]

    X_train = X_train.drop(columns=["f_0", "f_1", "is_installed", "is_clicked"])
    X_val = X_val.drop(columns=["f_0", "f_1", "is_installed", "is_clicked"])

    from catboost import CatBoostClassifier

    cat_features = list(X_train.select_dtypes(object).columns)
    count_0 = y_train.value_counts()[0]
    count_1 = y_train.value_counts()[1]
    rate = count_0/count_1

    print("Rate: ", rate)

    clf_train = CatBoostClassifier(
        **cat_params, class_weights={0: 1.0, 1: rate}, task_type=task_type, eval_metric="CrossEntropy"
    )

    cat_features = list(X_train.select_dtypes('category').columns)
    for col in cat_features:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    clf_train.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True,
        verbose=True,
    )

    cat_preds_train_proba = clf_train.predict_proba(X_val)

    print("loss:", normalized_cross_entropy_loss(cat_preds_train_proba[:, -1], y_val, 0.5))

    day_predictions = pd.DataFrame(
        {"row_id": val_row_ids, "catboost_weighted": cat_preds_train_proba[:, -1]}
    )

    all_predictions = pd.concat([all_predictions, day_predictions], axis=0)


print("END OF TRAINING FOR ALL DAYS")

all_predictions = all_predictions.to_csv(
    "catboost_weighted_predictions.csv", index=False, sep="\t"
)

bucket.upload_file(
    "catboost_weighted_predictions.csv",
    "hybrid_predictions/catboost_weighted_predictions.csv",
)
