import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import log_loss

import optuna


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    coefficient: int,
    means: pd.DataFrame,
    stds: pd.DataFrame,
    percentage: float = 0.001,
) -> pd.DataFrame:
    n_to_delete: int = int(len(df) * percentage)
    for col in columns:
        mean: float = means[col]
        std: float = stds[col]
        df_col: pd.DataFrame = pd.DataFrame(df[col])
        df_col["is_outlier"] = abs(df[col] - mean) > (coefficient * std)
        df_col["distance_from_mean"] = abs(df[col] - mean)
        df_col = df_col.sort_values("distance_from_mean", ascending=False)
        mask_is_outlier = df_col[df_col["is_outlier"] == True]  # noqa: E712
        candidates_for_averaging: pd.Index = mask_is_outlier[0:n_to_delete].index
        df.loc[candidates_for_averaging, col] = df[col].mean()
        print(f"Removed {len(candidates_for_averaging)} outliers from {col} column")

    return df


def objective(trial: optuna.Trial, df_train: pd.DataFrame, df_val: pd.DataFrame) -> float:
    CPU_COUNT = os.cpu_count()
    if CPU_COUNT is None:
        CPU_COUNT = 8
    num_clusters = trial.suggest_int("num_clusters", 2, 100)
    init = trial.suggest_categorical("init", ["k-means++", "random"])
    # n_init = trial.suggest_int("n_init", 1, 5)
    max_iter = trial.suggest_int("max_iter", 100, 1000, 100)
    max_no_improvement = trial.suggest_int("max_no_improvement", 50, 300, 10)
    batch_size = trial.suggest_int("batch_size", CPU_COUNT * 256, CPU_COUNT * 256 * 4, 256)
    CONSTANT: float = 0.145
    adjustment_weight: float = trial.suggest_float("adjustment_weight", 0.1, 2, step=0.01)
    weight_positive_samples: bool = trial.suggest_categorical(
        "weight_positive_samples", [True, False]
    )

    sample_weight: NDArray[np.float32] = np.repeat(1, len(df_train))
    if weight_positive_samples:
        positive_weight_multiplier: float = trial.suggest_float(
            "positive_weight_multiplier", 1, 10, step=0.1
        )
        sample_weight = np.where(df_train["is_installed"] == 1, positive_weight_multiplier, 1)

    df_train_numerical: pd.DataFrame = df_train.select_dtypes(include=["float32"])
    df_val_numerical: pd.DataFrame = df_val.select_dtypes(include=["float32"])

    model = MiniBatchKMeans(
        n_clusters=num_clusters,
        init=init,
        n_init="auto",
        max_iter=max_iter,
        max_no_improvement=max_no_improvement,
        batch_size=batch_size,
        verbose=1,
    )
    model.fit(df_train_numerical, sample_weight=sample_weight)

    clusters_train: NDArray[np.int32] = model.predict(df_train_numerical)
    results_train = pd.DataFrame(
        {"is_installed": df_train["is_installed"], "cluster": clusters_train}
    )
    n_elements_in_clusters: pd.Series = results_train.groupby("cluster").count()["is_installed"]
    is_installed_in_clusters: pd.Series = results_train.groupby("cluster").sum()["is_installed"]
    is_installed_percentage_in_clusters: pd.Series = (
        is_installed_in_clusters / n_elements_in_clusters
    )

    # Performance on validation
    clusters_val: NDArray[np.int32] = model.predict(df_val_numerical)
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
        CONSTANT + (results_val["predicted_is_installed"] - CONSTANT) * results_val["adjustment"]
    )
    results_val.loc[results_val["predicted_is_installed"] > 1, "predicted_is_installed"] = 1
    results_val.loc[results_val["predicted_is_installed"] < 0, "predicted_is_installed"] = 0

    score: float = log_loss(results_val["is_installed"], results_val["predicted_is_installed"])
    return score


if "__main__" == __name__:
    load_dotenv()
    OPTUNA_STORAGE: str = os.getenv("OPTUNA_STORAGE", "sqlite://optuna.db")
    N_TRIALS: int = int(os.getenv("N_TRIALS", 100))
    THRESHOLD_DAY: int = int(os.getenv("TRESHOLD_DAY", 63))
    OUTLIER_COEFFICIENT: int = int(os.getenv("OUTLIER_COEFFICIENT", 4))
    STUDY_NAME: str = os.getenv("STUDY_NAME", "clustering_log_loss")

    df_train_val: pd.DataFrame = pd.read_parquet("data/train_valEncoded.parquet")
    df_train_val = df_train_val.astype({f"f_{i}": "category" for i in range(2, 33)})
    df_train_val = df_train_val.sample(frac=1).reset_index(drop=True)

    df_train = df_train_val[df_train_val["f_1"] < THRESHOLD_DAY]
    df_val = df_train_val[df_train_val["f_1"] >= THRESHOLD_DAY]

    numerical_columns: List[str] = df_train.select_dtypes(include=["float32"]).columns.tolist()

    means: pd.Series = df_train[numerical_columns].mean()
    stds: pd.Series = df_train[numerical_columns].std()

    # fill the missing values with the mean
    df_train = df_train.fillna(means)
    df_val = df_val.fillna(means)

    # remove the outliers
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

    # normalize the data
    df_train.loc[:, numerical_columns] = (df_train.loc[:, numerical_columns] - means) / stds
    df_val.loc[:, numerical_columns] = (df_val.loc[:, numerical_columns] - means) / stds

    study: optuna.Study = optuna.create_study(
        study_name=STUDY_NAME, direction="minimize", storage=OPTUNA_STORAGE, load_if_exists=True
    )
    study.optimize(
        lambda trial: objective(trial, df_train, df_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
