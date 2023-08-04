import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsRegressor

import optuna
from utils.preprocessing import (
    CATEGORICAL_TO_DROP,
    NUMERICAL_NON_COUNTERS,
    NUMERICAL_TO_DROP,
    encode_counters,
    remove_categories_not_in_both,
    remove_outliers,
    trigonometric_date_encoding,
)
from utils.upload_predictions_to_s3 import upload_predictions_to_s3


def preprocess_data(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Drop bad CATEGORICAL columns
    X_train = X_train.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
    X_val = X_val.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)

    print("Encoding day of the week...")
    X_train = trigonometric_date_encoding(X_train, column="f_1")
    X_val = trigonometric_date_encoding(X_val, column="f_1")
    categorical_columns: List[str] = [
        f"f_{i}" for i in range(2, 32 + 1) if f"f_{i}" in X_train.columns
    ]
    boolean_columns: List[str] = [
        f"f_{i}" for i in range(33, 41 + 1) if f"f_{i}" in X_train.columns
    ]
    numerical_columns: List[str] = [col for col in NUMERICAL_NON_COUNTERS if col in X_train.columns]
    counter_columns: List[str] = [
        f"f_{i}"
        for i in range(42, 79 + 1)
        if f"f_{i}" in X_train.columns and f"f_{i}" not in numerical_columns
    ]

    print("Filling NA categorical columns...")
    X_train[categorical_columns] = X_train[categorical_columns].fillna(-100)
    X_val[categorical_columns] = X_val[categorical_columns].fillna(-100)

    print("Normalizing counter columns...")
    X_train, mins_train, steps_train = encode_counters(
        df=X_train,
        columns=counter_columns,
        mins=None,
        steps=None,
    )
    X_val, _, _ = encode_counters(
        df=X_val,
        columns=counter_columns,
        mins=mins_train,
        steps=steps_train,
    )
    for col in counter_columns:
        n_zeros: int = (X_train[col] == 0).sum()
        if n_zeros > X_train.shape[0] * COUNTERS_TO_BINARY_THRESHOLD:
            X_train[col] = np.where(X_train[col].values, 1, 0)
            X_train = X_train.astype({col: "bool"})
            boolean_columns.append(col)
            X_val[col] = np.where(X_val[col].values, 1, 0)
            X_val = X_val.astype({col: "bool"})
        else:
            X_train[col] = np.log(X_train[col] + 0.5)
            X_val[col] = np.log(X_val[col] + 0.5)

    print("Removing outliers from numerical columns...")
    means: pd.Series = X_train[numerical_columns].mean()
    stds: pd.Series = X_train[numerical_columns].std()
    X_train = remove_outliers(
        df=X_train,
        columns=numerical_columns,
        coefficient=OUTLIER_COEFFICIENT,
        means=means,
        stds=stds,
    )
    X_val = remove_outliers(
        df=X_val,
        columns=numerical_columns,
        coefficient=OUTLIER_COEFFICIENT,
        means=means,
        stds=stds,
    )

    print("Normalizing numerical columns...")
    means_no_outliers: pd.Series = X_train[numerical_columns].mean()
    stds_no_outliers: pd.Series = X_train[numerical_columns].std()
    X_train.loc[:, numerical_columns] = (
        X_train.loc[:, numerical_columns] - means_no_outliers
    ) / stds_no_outliers
    X_val.loc[:, numerical_columns] = (
        X_val.loc[:, numerical_columns] - means_no_outliers
    ) / stds_no_outliers
    X_train = X_train.fillna(means_no_outliers)
    X_val = X_val.fillna(means_no_outliers)

    print("Removing categories not in both train and val...")
    X_train, X_val = remove_categories_not_in_both(X_train, X_val, categorical_columns)

    print("Encode categorical features...")
    encoder = CatBoostEncoder(verbose=1)
    X_train.loc[:, categorical_columns] = encoder.fit_transform(
        X_train.loc[:, categorical_columns], y_train
    )
    X_val.loc[:, categorical_columns] = encoder.transform(X_val.loc[:, categorical_columns])

    return X_train, X_val


def build_fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int,
    weights: str,
    leaf_size: int,
    metric: str,
) -> KNeighborsRegressor:
    model: KNeighborsRegressor = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        leaf_size=leaf_size,
        metric=metric,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: KNeighborsRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    y_pred = model.predict(X_val)
    log_loss_score: float = log_loss(y_val, y_pred)
    return log_loss_score


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    # Defining parameters
    n_neighbors: int = trial.suggest_int("n_neighbors", 3, 1000)
    weights: str = trial.suggest_categorical("weights", ["uniform", "distance"])
    leaf_size: int = trial.suggest_int("leaf_size", 20, 100)
    metric: str = trial.suggest_categorical(
        "metric",
        [
            "euclidean",
            "cosine",
            "manhattan",
        ],
    )

    print("Trial parameters:", trial.params)

    # Building model
    print("Building and training model...")
    model: KNeighborsRegressor = build_fit_model(
        X_train=X_train,
        y_train=y_train,
        n_neighbors=n_neighbors,
        weights=weights,
        leaf_size=leaf_size,
        metric=metric,
    )

    # Calculating performance
    print("Calculating performance...")
    log_loss_score: float = evaluate_model(
        model=model,
        X_val=X_val,
        y_val=y_val,
    )

    return log_loss_score


def run_study(
    df_train_val: pd.DataFrame,
    df_test: pd.DataFrame,
    validation_split_day: int,
    model_name: str,
    script_start: datetime,
    optuna_storage: str,
    n_trials: int,
    study_name: Optional[str] = None,
) -> None:
    df_train: pd.DataFrame = df_train_val.loc[df_train_val["f_1"] <= validation_split_day]
    df_train = df_train.drop(columns=["f_0"])
    X_train = df_train.drop(columns=["is_installed", "is_clicked"])
    y_train = df_train["is_installed"]

    df_val: pd.DataFrame = df_train_val.loc[df_train_val["f_1"] > validation_split_day]
    df_val = df_val.drop(columns=["f_0"])
    X_val = df_val.drop(columns=["is_installed", "is_clicked"])
    y_val = df_val["is_installed"]

    X_test = df_test.drop(columns=["f_0"])

    print("Preprocessing data...")
    X_val_test = pd.concat([X_val, X_test], axis=0)
    X_train, X_val_test = preprocess_data(X_train, y_train, X_val_test)
    X_val = X_val_test.iloc[: len(X_val), :]
    X_test = X_val_test.iloc[len(X_val) :, :]

    print("Creating Optuna study...")
    if study_name is None:
        study_name = model_name + "_" + script_start.strftime("%Y-%m-%dT%H:%M:%S")
    study: optuna.Study = optuna.create_study(
        directions=["minimize"],
        storage=optuna_storage,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial=trial,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )


def predict_with_params(
    df_train_val: pd.DataFrame,
    df_test: pd.DataFrame,
    best_params: Dict[str, Any],
    model_name: str,
    script_start: datetime,
    local_predictions_path: Path,
) -> Tuple[Path, Path]:
    raise NotImplementedError


if "__main__" == __name__:
    script_start: datetime = datetime.now()
    script_name: str = os.path.basename(__file__).split(".")[0]

    print("Loading environment variables and parameters...")
    load_dotenv()
    OPTIMIZE_OR_PREDICT: str = os.getenv("OPTIMIZE_OR_PREDICT", "optimize")
    STUDY_NAME: Optional[str] = os.getenv("STUDY_NAME")
    OPTUNA_STORAGE: str = os.getenv("OPTUNA_STORAGE", "sqlite://optuna.db")
    N_TRIALS: int = int(os.getenv("N_TRIALS", 500))
    THRESHOLD_DAY: int = int(os.getenv("THRESHOLD_DAY", 63))
    OUTLIER_COEFFICIENT: int = int(os.getenv("OUTLIER_COEFFICIENT", 4))
    SIZE_LIMIT_ENCODING: int = int(os.getenv("SIZE_LIMIT_ENCODING", 10))
    COUNTERS_TO_BINARY_THRESHOLD: float = 0.95
    CONSTANT: float = 0.145

    print("Loading and splitting data...")
    df_train_val: pd.DataFrame = pd.read_parquet("data/train_val.parquet").reset_index(drop=True)
    df_test: pd.DataFrame = pd.read_parquet("data/test.parquet").reset_index(drop=True)
    best_params: Dict[str, Any] = {
        "num_clusters": 15,
        "max_iter": 800,
        "batch_size": 9472,
        "init": "k-means++",
        "max_no_improvement": 190,
        "positive_weight_multiplier": 1.6,
    }

    if OPTIMIZE_OR_PREDICT == "predict":
        train_val_csv: Path
        test_csv: Path
        train_val_csv, test_csv = predict_with_params(
            df_train_val=df_train_val,
            df_test=df_test,
            best_params=best_params,
            model_name=script_name,
            script_start=script_start,
            local_predictions_path=Path("data/predictions"),
        )

        print("Uploading predictions to S3...")
        aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
        if aws_access_key_id is not None and aws_secret_access_key is not None:
            upload_predictions_to_s3(
                train_val_csv=train_val_csv,
                test_csv=test_csv,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        else:
            print("AWS credentials not found in env variables. Skipping upload to S3.")

    elif OPTIMIZE_OR_PREDICT == "optimize":
        run_study(
            df_train_val=df_train_val,
            df_test=df_test,
            validation_split_day=THRESHOLD_DAY,
            model_name=script_name,
            script_start=script_start,
            study_name=STUDY_NAME,
            optuna_storage=OPTUNA_STORAGE,
            n_trials=N_TRIALS,
        )

    print("Done!")
