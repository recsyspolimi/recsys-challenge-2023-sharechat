import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

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


def objective(trial: optuna.Trial, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[float, float]:
    CPU_COUNT = os.cpu_count()
    if CPU_COUNT is None:
        CPU_COUNT = 8
    '''
    num_clusters = trial.suggest_int("num_clusters", 2, 100)
    init = trial.suggest_categorical("init", ["k-means++", "random"])
    # n_init = trial.suggest_int("n_init", 1, 5)
    max_iter = trial.suggest_int("max_iter", 100, 1000, 100)
    max_no_improvement = trial.suggest_int("max_no_improvement", 50, 200, 10)
    batch_size = trial.suggest_int("batch_size", CPU_COUNT * 256, CPU_COUNT * 256 * 4, 256)

    model = MiniBatchKMeans(
        n_clusters=num_clusters,
        init=init,
        n_init="auto",
        random_state=42,
        max_iter=max_iter,
        max_no_improvement=max_no_improvement,
        batch_size=batch_size,
        verbose=0,
    )
    '''
    eps = trial.suggest_float("eps", 0.1, 1.0)
    min_samples = trial.suggest_int("min_samples", 2, 50)
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan"])
    algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    leaf_size = trial.suggest_int("leaf_size", 10, 100, 10)

    model = DBSCAN(
        eps= eps,
        min_samples= min_samples,
        metric= metric,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs= CPU_COUNT,
    )

    df_train_numerical: pd.DataFrame = df_train.select_dtypes(include=["float32"])
    df_val_numerical: pd.DataFrame = df_val.select_dtypes(include=["float32"])
    if model.__class__.__name__ == "MiniBatchKMeans":
        model.fit(df_train_numerical)
        clusters = model.predict(df_train_numerical)
    else:
        clusters = model.fit_predict(df_train_numerical)
    
    print("Starting compute metrics")
    if len(np.unique(clusters)) == 1:
        return 1000.0, 0.0
    ch_score: float = calinski_harabasz_score(df_train_numerical, clusters)
    db_score: float = davies_bouldin_score(df_train_numerical, clusters)
    return db_score, ch_score


if "__main__" == __name__:
    load_dotenv()
    OPTUNA_STORAGE: str = os.getenv("OPTUNA_STORAGE", "sqlite://optuna.db")
    N_TRIALS: int = int(os.getenv("N_TRIALS", 250))
    THRESHOLD_DAY: int = int(os.getenv("TRESHOLD_DAY", 63))
    OUTLIER_COEFFICIENT: int = int(os.getenv("OUTLIER_COEFFICIENT", 4))
    N_SAMPLES_FOR_PREDICT: int = int(os.getenv("N_SAMPLES_FOR_PREDICT", 3e5))

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
        directions= ["minimize", "maximize"],
        storage=OPTUNA_STORAGE,
        study_name="dbscan_db_ch",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, df_train, df_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    print("END")
