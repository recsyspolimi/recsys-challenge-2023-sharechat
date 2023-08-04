import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

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
    num_clusters: int,
    init: str,
    max_iter: int,
    max_no_improvement: int,
    batch_size: int,
    positive_weight_multiplier: float,
) -> MiniBatchKMeans:
    sample_weight: NDArray[np.float32] = np.where(y_train == 1, positive_weight_multiplier, 1)

    model: MiniBatchKMeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        init=init,
        n_init="auto",
        random_state=42,
        max_iter=max_iter,
        max_no_improvement=max_no_improvement,
        batch_size=batch_size,
        verbose=0,
    )

    model.fit(X_train, sample_weight=sample_weight)

    return model


def evaluate_model(
    model: MiniBatchKMeans,
    X_train_val: pd.DataFrame,
) -> Tuple[float, float]:
    clusters: NDArray[np.int32] = model.predict(X_train_val)

    ch_score: float = calinski_harabasz_score(X_train_val, clusters)
    db_score: float = davies_bouldin_score(X_train_val, clusters)
    return db_score, ch_score


def objective(
    trial: optuna.Trial,
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    cpu_count: int,
) -> Tuple[float, float]:
    # Defining parameters
    num_clusters: int = 20
    init: str = trial.suggest_categorical("init", ["k-means++", "random"])
    max_iter: int = trial.suggest_int("max_iter", 100, 1000, 100)
    max_no_improvement: int = trial.suggest_int("max_no_improvement", 50, 200, 10)
    batch_size: int = trial.suggest_int("batch_size", cpu_count * 256, cpu_count * 256 * 4, 256)

    positive_weight_multiplier: float = trial.suggest_float(
        "positive_weight_multiplier", 0.5, 4, step=0.1
    )

    print("Trial parameters:", trial.params)

    # Building model
    print("Building and training model...")
    model: MiniBatchKMeans = build_fit_model(
        X_train=X_train_val,
        y_train=y_train_val,
        num_clusters=num_clusters,
        init=init,
        max_iter=max_iter,
        max_no_improvement=max_no_improvement,
        batch_size=batch_size,
        positive_weight_multiplier=positive_weight_multiplier,
    )

    # Calculating performance
    print("Calculating performance...")
    db_score, ch_score = evaluate_model(
        model=model,
        X_train_val=X_train_val,
    )

    return db_score, ch_score


def run_study(
    df_train_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_name: str,
    script_start: datetime,
    optuna_storage: str,
    n_trials: int,
    study_name: Optional[str] = None,
) -> None:
    X_train_val: pd.DataFrame = df_train_val.drop(columns=["f_0", "is_installed", "is_clicked"])
    y_train_val: pd.Series = df_train_val["is_installed"]
    X_test: pd.DataFrame = df_test.drop(columns=["f_0"])

    print("Preprocessing data...")
    X_train_val, X_test = preprocess_data(X_train_val, y_train_val, X_test)

    print("Creating Optuna study...")
    if study_name is None:
        study_name = model_name + "_" + script_start.strftime("%Y-%m-%dT%H:%M:%S")
    study: optuna.Study = optuna.create_study(
        directions=["minimize", "maximize"],
        storage=optuna_storage,
        study_name=study_name,
        load_if_exists=True,
    )
    cpu_count: int = os.cpu_count() or 8
    study.optimize(
        lambda trial: objective(
            trial=trial,
            X_train_val=X_train_val,
            y_train_val=y_train_val,
            cpu_count=cpu_count,
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
    print("Saving best trial predictions...")
    print("Best trial parameters:", best_params)

    X_train_val: pd.DataFrame = df_train_val.drop(columns=["is_installed", "is_clicked"])
    X_test: pd.DataFrame = df_test.loc[:]
    y_train_val: pd.Series = df_train_val["is_installed"]

    print("Preprocessing...")
    X_train_val, X_test = preprocess_data(X_train_val, y_train_val, X_test)

    print("Building and training model with best parameters...")
    best_model: MiniBatchKMeans = build_fit_model(
        X_train=X_train_val.drop(columns=["f_0"]),
        y_train=y_train_val,
        num_clusters=best_params["num_clusters"],
        init=best_params["init"],
        max_iter=best_params["max_iter"],
        max_no_improvement=best_params["max_no_improvement"],
        batch_size=best_params["batch_size"],
        positive_weight_multiplier=best_params["positive_weight_multiplier"],
    )

    print("Calculating predictions...")
    X_train_val_cluster_distances: NDArray[np.float32] = best_model.transform(
        X_train_val.drop(columns=["f_0"])
    )
    X_test_cluster_distances: NDArray[np.float32] = best_model.transform(
        X_test.drop(columns=["f_0"])
    )

    distance_col: str = model_name + "_distance"
    X_train_val[model_name] = np.argmin(X_train_val_cluster_distances, axis=1)
    X_test[model_name] = np.argmin(X_test_cluster_distances, axis=1)
    X_train_val[distance_col] = np.min(X_train_val_cluster_distances, axis=1)
    X_test[distance_col] = np.min(X_test_cluster_distances, axis=1)

    print("Saving predictions locally...")
    train_val_predictions: pd.DataFrame = X_train_val[["f_0", model_name, distance_col]]
    test_predictions: pd.DataFrame = X_test[["f_0", model_name, distance_col]]

    if not local_predictions_path.exists():
        local_predictions_path.mkdir(parents=True, exist_ok=True)
    train_val_csv: Path = local_predictions_path / f"{model_name}_trainval.csv"
    test_csv: Path = local_predictions_path / f"{model_name}_test.csv"
    train_val_predictions.to_csv(train_val_csv, sep="\t", index=False)
    test_predictions.to_csv(test_csv, sep="\t", index=False)

    return train_val_csv, test_csv


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
        "num_clusters": 20,
        "max_iter": 400,
        "batch_size": 22016,
        "init": "random",
        "max_no_improvement": 200,
        "positive_weight_multiplier": 1.5,
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
            model_name=script_name,
            script_start=script_start,
            study_name=STUDY_NAME,
            optuna_storage=OPTUNA_STORAGE,
            n_trials=N_TRIALS,
        )

    print("Done!")
