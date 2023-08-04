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
from sklearn.metrics import log_loss

import optuna
from utils.preprocessing import (
    encode_counters,
    remove_categories_not_in_both,
    remove_outliers,
    trigonometric_date_encoding,
)

CATEGORICAL_TO_DROP: List[str] = [
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
NUMERICAL_TO_DROP: List[str] = [
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


def preprocess_data(
    df_train: pd.DataFrame, df_val: pd.DataFrame, is_test: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Drop bad CATEGORICAL columns
    df_train = df_train.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
    df_val = df_val.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)

    print("Encoding day of the week...")
    df_train = trigonometric_date_encoding(df_train, column="f_1")
    df_val = trigonometric_date_encoding(df_val, column="f_1")
    categorical_columns: List[str] = [
        f"f_{i}" for i in range(2, 32 + 1) if f"f_{i}" in df_train.columns
    ]
    boolean_columns: List[str] = [
        f"f_{i}" for i in range(33, 41 + 1) if f"f_{i}" in df_train.columns
    ]
    numerical_columns: List[str] = [
        col for col in NUMERICAL_NON_COUNTERS if col in df_train.columns
    ]
    counter_columns: List[str] = [
        f"f_{i}"
        for i in range(42, 79 + 1)
        if f"f_{i}" in df_train.columns and f"f_{i}" not in numerical_columns
    ]
    date_encoding_columns: List[str] = ["sin_date", "cos_date"]

    df_train = df_train.drop(columns=["f_1", "is_clicked"])
    if not is_test:
        df_val = df_val.drop(columns=["f_1", "is_clicked"])
    else:
        df_val = df_val.drop(columns=["f_1"])

    print("Filling NA categorical columns...")
    df_train[categorical_columns] = df_train[categorical_columns].fillna(-100)
    df_val[categorical_columns] = df_val[categorical_columns].fillna(-100)

    print("Normalizing counter columns...")
    df_train, mins_train, steps_train = encode_counters(
        df=df_train,
        columns=counter_columns,
        mins=None,
        steps=None,
    )
    df_val, _, _ = encode_counters(
        df=df_val,
        columns=counter_columns,
        mins=mins_train,
        steps=steps_train,
    )
    counter_modes: pd.Series = df_train[counter_columns].mode()
    df_train = df_train.fillna(counter_modes)
    df_val = df_val.fillna(counter_modes)
    for col in counter_columns:
        n_zeros: int = (df_train[col] == 0).sum()
        if n_zeros > df_train.shape[0] * COUNTERS_TO_BINARY_THRESHOLD:
            df_train[col] = np.where(df_train[col].values, 1, 0)
            df_train = df_train.astype({col: "bool"})
            boolean_columns.append(col)
            df_val[col] = np.where(df_val[col].values, 1, 0)
            df_val = df_val.astype({col: "bool"})
        else:
            df_train[col] = np.log(df_train[col] + 0.5)
            df_val[col] = np.log(df_val[col] + 0.5)

    print("Removing outliers from numerical columns...")
    means: pd.Series = df_train[numerical_columns].mean()
    stds: pd.Series = df_train[numerical_columns].std()
    df_train = remove_outliers(
        df=df_train,
        columns=numerical_columns,
        coefficient=OUTLIER_COEFFICIENT,
        means=means,
        stds=stds,
    )
    df_val = remove_outliers(
        df=df_val,
        columns=numerical_columns,
        coefficient=OUTLIER_COEFFICIENT,
        means=means,
        stds=stds,
    )

    print("Normalizing numerical columns...")
    means_no_outliers: pd.Series = df_train[numerical_columns].mean()
    stds_no_outliers: pd.Series = df_train[numerical_columns].std()
    df_train.loc[:, numerical_columns] = (
        df_train.loc[:, numerical_columns] - means_no_outliers
    ) / stds_no_outliers
    df_val.loc[:, numerical_columns] = (
        df_val.loc[:, numerical_columns] - means_no_outliers
    ) / stds_no_outliers
    df_train = df_train.fillna(means_no_outliers)
    df_val = df_val.fillna(means_no_outliers)

    print("Removing categories not in both train and val...")
    df_train, df_val = remove_categories_not_in_both(df_train, df_val, categorical_columns)

    print("Encode categorical features...")
    encoder = CatBoostEncoder(verbose=1)
    df_train.loc[:, categorical_columns] = encoder.fit_transform(
        df_train.loc[:, categorical_columns], df_train["is_installed"]
    )
    df_val.loc[:, categorical_columns] = encoder.transform(df_val.loc[:, categorical_columns])

    return df_train, df_val


def build_fit_model(
    df_train: pd.DataFrame,
    num_clusters: int,
    init: str,
    max_iter: int,
    max_no_improvement: int,
    batch_size: int,
    positive_weight_multiplier: float,
) -> MiniBatchKMeans:
    sample_weight: NDArray[np.float32] = np.where(
        df_train["is_installed"] == 1, positive_weight_multiplier, 1
    )

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

    model.fit(df_train.drop(columns=["is_installed"]), sample_weight=sample_weight)

    return model


def evaluate_model(
    model: MiniBatchKMeans,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    constant: float,
    adjustment_weight: float,
) -> float:
    df_train_no_label: pd.DataFrame = df_train.drop(columns=["is_installed"])
    df_val_no_label: pd.DataFrame = df_val.drop(columns=["is_installed"])

    # Get cluster statistics from train
    clusters_train: NDArray[np.int32] = model.predict(df_train_no_label)
    results_train = pd.DataFrame(
        {"is_installed": df_train["is_installed"], "cluster": clusters_train}
    )
    n_elements_in_clusters: pd.Series = results_train.groupby("cluster").count()["is_installed"]
    is_installed_in_clusters: pd.Series = results_train.groupby("cluster").sum()["is_installed"]
    is_installed_percentage_in_clusters: pd.Series = (
        is_installed_in_clusters / n_elements_in_clusters
    )

    # Performance on validation
    clusters_val: NDArray[np.int32] = model.predict(df_val_no_label)
    results_val = pd.DataFrame({"is_installed": df_val["is_installed"], "cluster": clusters_val})
    results_val["predicted_is_installed"] = results_val["cluster"].map(
        is_installed_percentage_in_clusters
    )

    # Transform prediction
    n_elements_in_clusters_max: int = n_elements_in_clusters.max()
    results_val["adjustment"] = results_val["cluster"].map(
        adjustment_weight * (n_elements_in_clusters / n_elements_in_clusters_max)
    )
    results_val["predicted_is_installed"] = (
        constant + (results_val["predicted_is_installed"] - constant) * results_val["adjustment"]
    )
    results_val.loc[results_val["predicted_is_installed"] > 1, "predicted_is_installed"] = 1
    results_val.loc[results_val["predicted_is_installed"] < 0, "predicted_is_installed"] = 0

    score: float = log_loss(results_val["is_installed"], results_val["predicted_is_installed"])
    return score


def objective(
    trial: optuna.Trial,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    constant: float,
    cpu_count: int,
) -> float:
    # Defining parameters
    num_clusters: int = trial.suggest_int("num_clusters", 15, 100)
    init: str = trial.suggest_categorical("init", ["k-means++", "random"])
    max_iter: int = trial.suggest_int("max_iter", 100, 1000, 100)
    max_no_improvement: int = trial.suggest_int("max_no_improvement", 50, 200, 10)
    batch_size: int = trial.suggest_int("batch_size", cpu_count * 256, cpu_count * 256 * 4, 256)

    positive_weight_multiplier: float = trial.suggest_float(
        "positive_weight_multiplier", 0.5, 4, step=0.1
    )

    adjustment_weight: float = trial.suggest_float("adjustment_weight", 0.1, 2, step=0.01)
    print("Trial parameters:", trial.params)

    # Building model
    print("Building and training model...")
    model: MiniBatchKMeans = build_fit_model(
        df_train=df_train,
        num_clusters=num_clusters,
        init=init,
        max_iter=max_iter,
        max_no_improvement=max_no_improvement,
        batch_size=batch_size,
        positive_weight_multiplier=positive_weight_multiplier,
    )

    # Calculating performance
    print("Calculating performance...")
    score: float = evaluate_model(
        model=model,
        df_train=df_train,
        df_val=df_val,
        constant=constant,
        adjustment_weight=adjustment_weight,
    )

    return score


if "__main__" == __name__:
    script_start: datetime = datetime.now()

    print("Loading environment variables and parameters...")
    load_dotenv()
    STUDY_NAME: Optional[str] = os.getenv("STUDY_NAME")
    OPTUNA_STORAGE: str = os.getenv("OPTUNA_STORAGE", "sqlite://optuna.db")
    N_TRIALS: int = int(os.getenv("N_TRIALS", 500))
    THRESHOLD_DAY: int = int(os.getenv("THRESHOLD_DAY", 60))
    OUTLIER_COEFFICIENT: int = int(os.getenv("OUTLIER_COEFFICIENT", 4))
    SIZE_LIMIT_ENCODING: int = int(os.getenv("SIZE_LIMIT_ENCODING", 10))
    COUNTERS_TO_BINARY_THRESHOLD: float = 0.95
    CONSTANT: float = 0.145

    print("Loading and splitting data...")
    df_train_val: pd.DataFrame = pd.read_parquet("data/train_val.parquet")
    df_train_val = df_train_val.sample(frac=1).reset_index(drop=True)

    # TODO: df_val needs to be <= 60 for boosting
    df_train = df_train_val[df_train_val["f_1"] <= THRESHOLD_DAY].reset_index(drop=True)
    df_val = df_train_val[df_train_val["f_1"] > THRESHOLD_DAY].reset_index(drop=True)

    print("Preprocessing data...")
    df_train, df_val = preprocess_data(df_train, df_val)
    df_train = df_train.drop(columns=["f_0"])
    df_val = df_val.drop(columns=["f_0"])

    print("Creating Optuna study...")
    script_name: str = os.path.basename(__file__).split(".")[0]
    study_name: str = script_name + "_" + script_start.strftime("%Y-%m-%dT%H:%M:%S")
    if STUDY_NAME is not None:
        study_name = STUDY_NAME
    study: optuna.Study = optuna.create_study(
        directions=["minimize"],
        storage=OPTUNA_STORAGE,
        study_name=study_name,
        load_if_exists=True,
    )
    cpu_count: Optional[int] = os.cpu_count()
    if cpu_count is None:
        cpu_count = 8
    study.optimize(
        lambda trial: objective(
            trial=trial,
            df_train=df_train,
            df_val=df_val,
            constant=CONSTANT,
            cpu_count=cpu_count,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("Saving best trial predictions...")
    df_train = df_train_val[df_train_val["f_1"] <= THRESHOLD_DAY].reset_index(drop=True)
    df_val = df_train_val[df_train_val["f_1"] > THRESHOLD_DAY].reset_index(drop=True)
    df_test: pd.DataFrame = pd.read_parquet("data/test.parquet").reset_index(drop=True)
    df_val_test = pd.concat([df_val, df_test], axis=0).reset_index(drop=True)

    df_train, df_val_test = preprocess_data(df_train, df_val_test, is_test=True)

    best_trial: optuna.trial.FrozenTrial = study.best_trial
    best_params: Dict[str, Any] = best_trial.params
    print("Best trial parameters:", best_params)

    print("Building and training model with best parameters...")
    best_model: MiniBatchKMeans = build_fit_model(
        df_train=df_train.drop(columns=["f_0"]),
        num_clusters=best_params["num_clusters"],
        init=best_params["init"],
        max_iter=best_params["max_iter"],
        max_no_improvement=best_params["max_no_improvement"],
        batch_size=best_params["batch_size"],
        positive_weight_multiplier=best_params["positive_weight_multiplier"],
    )

    print("Calculating predictions...")
    df_test["prediction"] = best_model.predict(df_test.drop(columns=["f_0"]))
    df_train_val["prediction"] = best_model.predict(
        df_train_val.drop(columns=["is_installed", "f_0"])
    )

    print("Saving predictions...")
    train_val_predictions: pd.DataFrame = df_train_val[["f_0", "prediction"]].rename(
        columns={"f_0": "row_id"}
    )
    test_predictions: pd.DataFrame = df_test[["f_0", "prediction"]].rename(
        columns={"f_0": "row_id"}
    )

    predictions_folder: Path = Path("data/predictions")
    if not predictions_folder.exists():
        predictions_folder.mkdir(parents=True, exist_ok=True)
    train_val_predictions.to_parquet(
        predictions_folder / f"{script_name}_train_val_{script_start.strftime('%Y-%m-%d')}.parquet",
        index=False,
    )
    test_predictions.to_parquet(
        predictions_folder / f"{script_name}_test_{script_start.strftime('%Y-%m-%d')}.parquet",
        index=False,
    )
