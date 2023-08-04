import os
import pickle
from typing import Callable, Dict, List, Set, Tuple

from xgboost import XGBClassifier

import boto3
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from dotenv import load_dotenv

from notify_run import Notify

from tqdm import tqdm

from utils.normalized_cross_entropy_loss import normalized_cross_entropy_loss

notify = Notify()

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

tree_method = "gpu_hist" if GPU else "hist"

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


df_train_val = pd.read_parquet("data/train_valEncoded.parquet")
df_train_val = df_train_val.astype({f"f_{i}": "category" for i in range(2, 33)})
df_train_val = df_train_val.astype({'f_1': 'int'})

df_test = pd.read_parquet("data/testEncoded.parquet")
df_test = df_test.astype({f"f_{i}": "category" for i in range(2, 33)})
df_test = df_test.astype({'f_1': 'int'})

df_train_val = df_train_val.astype({'is_clicked': 'int'})
df_train_val = df_train_val.astype({'is_installed': 'int'})

print("converting boolean columns to bool...")
boolean_columns = list(df_train_val.select_dtypes(['boolean']).columns)
for col in boolean_columns:
    df_train_val[col] = df_train_val[col].astype(bool)
    df_test[col] = df_test[col].astype(bool)

print("converting categorical columns to int...")
cat_columns = list(df_train_val.select_dtypes(['category']).columns)
for col in cat_columns:
    df_train_val[col] = df_train_val[col].astype(int)
    df_test[col] = df_test[col].astype(int)


################## HYPERPARAMETERS ##################

params =  {
    "drop_useless": True, 
    "grow_policy": "depthwise", 
    "n_estimators": 400, 
    "max_depth": 6, 
    "learning_rate": 0.08067255120460029, 
    "subsample": 0.632939654030012, 
    "colsample_bytree": 0.6199993443064554, 
    "gamma": 0.002287053762127034, 
    "reg_alpha": 0.00017939742017654073, 
    "reg_lambda": 0.00045867039830469483, 
    "min_child_weight": 168
}

drop_useless = params.pop("drop_useless")

################## TRAINING ##################

all_predictions = pd.DataFrame()
for day in tqdm(range(52, 67)):
    print("splitting data...")
    df_train = df_train_val[df_train_val["f_1"].between(0, day, inclusive="left")]
    df_val = df_train_val[df_train_val["f_1"].between(day, day + 1, inclusive="left")]

    print("Preprocessing data...")
    if drop_useless:
        df_train = df_train.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
        df_val = df_val.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
        print("Dropped useless columns")
    else:
        print("Keeping all columns")

    y_train = df_train["is_installed"]
    y_val = df_val["is_installed"]
    val_row_ids = df_val["f_0"]

    X_train = df_train.drop(columns=["f_0", "is_installed", "is_clicked"])
    X_val = df_val.drop(columns=["f_0", "is_installed", "is_clicked"])

    model = XGBClassifier(
        **params, 
        tree_method=tree_method,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    model.fit(
        X_train, 
        y_train, 
        eval_set = [(X_val, y_val)], 
        verbose=True
    )
    
    # Calculate the cross-entropy loss on the testing data
    xgb_probs = model.predict_proba(X_val)[:,-1]

    print("loss:", normalized_cross_entropy_loss(xgb_probs, y_val, 0.5))

    day_predictions = pd.DataFrame(
        {"f_0": val_row_ids, "xgb_catasnum": xgb_probs}
    )

    all_predictions = pd.concat([all_predictions, day_predictions], axis=0)

print("END OF TRAINING FOR ALL DAYS")


############ TEST SET ############

print("Start generating predictions for test set...")
X_train_val = df_train_val.drop(columns=["is_installed", "is_clicked"])
y_train_val = df_train_val["is_installed"]
X_test = df_test
f_0 = df_test["f_0"]

if drop_useless:
    X_train_val = X_train_val.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
    X_test = X_test.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
    print("Dropped useless columns")
else:
    print("Keeping all columns")

X_train_val = X_train_val.drop(columns=["f_0"])
X_test = X_test.drop(columns=["f_0"])

model = XGBClassifier(
    **params,
    tree_method=tree_method,
    objective='binary:logistic',
    eval_metric='logloss',
)
model.fit(
    X_train_val,
    y_train_val,
    verbose=True
)

xgb_probs = model.predict_proba(X_test)[:,-1]

test_predictions = pd.DataFrame(
    {"f_0": f_0, "xgb_catasnum": xgb_probs}
)


############ UPLOAD TO S3 ############

all_predictions = all_predictions.to_csv("xgb_catasnum_trainval_incremental.csv", index=False, sep="\t")
test_predictions.to_csv("xgb_catasnum_test.csv", index=False, sep="\t")

print("Uploading predictions to s3...")
bucket.upload_file(
    "xgb_catasnum_trainval_incremental.csv",
    "hybrid_predictions/xgb_catasnum_trainval_incremental.csv",
)
print("Uploaded trainval predictions")

bucket.upload_file(
    "xgb_catasnum_test.csv",
    "hybrid_predictions/xgb_catasnum_test.csv",
)
print("Uploaded test predictions")

notify.send("Finished training and uploading xgb_catasnum model to s3")