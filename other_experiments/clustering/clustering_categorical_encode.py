import os
from datetime import datetime
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from dotenv import load_dotenv
from kmodes.util.dissim import matching_dissim, ng_dissim
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

import optuna

DISSIM_NAME_CLASS_MAP: Dict[str, Callable] = {
    "matching": matching_dissim,
    "ng": ng_dissim,
}


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

    return df


def objective(
    trial: optuna.Trial,
    df_train: pd.DataFrame,
) -> Tuple[float, float]:
    CPU_COUNT = os.cpu_count()
    if CPU_COUNT is None:
        CPU_COUNT = 8
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

    print("Using parameters:", trial.params)

    model.fit(df_train)
    clusters = model.predict(df_train)

    print("Starting compute metrics")
    if len(np.unique(clusters)) == 1:
        return 1000.0, 0.0

    ch_score: float = calinski_harabasz_score(df_train, clusters)
    db_score: float = davies_bouldin_score(df_train, clusters)
    return db_score, ch_score


def trigonometric_date_encoding(df: pd.DataFrame, column: str = "f_train") -> pd.DataFrame:
    """Encode date as sin and cos of the day of the week.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str, optional): The column name with the date to encode. Defaults to "f_train".

    Returns:
        pd.DataFrame: The dataframe with the encoded date.
    """
    day_of_week: pd.Series = df[column] % 7
    date_sin: pd.Series = np.sin(day_of_week * (2.0 * np.pi / 7.0))
    date_cos: pd.Series = np.cos(day_of_week * (2.0 * np.pi / 7.0))
    encoded_dates: pd.DataFrame = pd.DataFrame({"sin_date": date_sin, "cos_date": date_cos})
    result_df = pd.concat([df, encoded_dates], axis=1)
    return result_df


def remove_categories_not_in_both(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove categories that are not present in both dataframes.
    In train the categories are replaced with -1 while in val they are replcaed with the most common
    category in the train set which is also present in the val set.

    Args:
        df_train (pd.DataFrame): The train dataframe.
        df_val (pd.DataFrame): The val dataframe.
        categorical_columns (List[str]): The list of categorical columns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The two dataframes with the removed categories.
    """
    for col in categorical_columns:
        categories_train: Set[int] = set(df_train[col].unique())
        categories_val: Set[int] = set(df_val[col].unique())
        categories_both: Set[int] = categories_train.intersection(categories_val)

        most_common_categories_train: List[int] = df_train[col].value_counts().index
        most_common_train: int = -1
        for cat in most_common_categories_train:
            if cat in categories_both:
                most_common_train = cat
                break

        df_train[col] = df_train[col].apply(lambda x: -1 if x not in categories_both else x)
        df_val[col] = df_val[col].apply(
            lambda x: most_common_train if x not in categories_both else x
        )

    df_train = df_train.astype({col: "category" for col in categorical_columns})
    df_val = df_val.astype({col: "category" for col in categorical_columns})
    return df_train, df_val


if "__main__" == __name__:
    load_dotenv()
    OPTUNA_STORAGE: str = os.getenv("OPTUNA_STORAGE", "sqlite://optuna.db")
    N_TRIALS: int = int(os.getenv("N_TRIALS", 250))
    THRESHOLD_DAY: int = int(os.getenv("TRESHOLD_DAY", 63))
    OUTLIER_COEFFICIENT: int = int(os.getenv("OUTLIER_COEFFICIENT", 4))
    SIZE_LIMIT_ENCODING: int = int(os.getenv("SIZE_LIMIT_ENCODING", 10))
    EXISTING_DATE_TIME: str = None
    CATEGORICAL_TO_DROP: list = [
        "f_7",
        "f_27",
        "f_28",
        "f_29",
    ]

    if EXISTING_DATE_TIME is not None:
        studyName = "clustering_all_features_" + EXISTING_DATE_TIME
    else:
        studyName = "clustering_all_features_" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    print("Loading and splitting data...")
    df_train_val: pd.DataFrame = pd.read_parquet("data/train_valEncoded.parquet")
    df_train_val = df_train_val.astype({f"f_{i}": "category" for i in range(2, 33)})
    # ATTENZIONE STAI SHUFFLANDO
    df_train_val = df_train_val.sample(frac=1).reset_index(drop=True)

    df_train = df_train_val[df_train_val["f_1"] < THRESHOLD_DAY]
    df_val = df_train_val[df_train_val["f_1"] >= THRESHOLD_DAY]

    print("Encoding day of the week...")
    df_train = trigonometric_date_encoding(df_train, column="f_1")
    df_val = trigonometric_date_encoding(df_val, column="f_1")
    categorical_columns: List[str] = [f"f_{i}" for i in range(2, 41 + 1)]
    for str in CATEGORICAL_TO_DROP:
        categorical_columns.remove(str)
    train_is_installed = df_train["is_installed"]
    if CATEGORICAL_TO_DROP is not None:
        df_train = df_train.drop(columns=CATEGORICAL_TO_DROP)
    df_train = df_train.drop(columns=["f_0", "f_1", "is_installed", "is_clicked"])
    df_val = df_val.drop(columns=["f_0", "f_1", "is_installed", "is_clicked"])

    df_train = df_train.loc[:, categorical_columns]
    df_val = df_val.loc[:, categorical_columns]

    print("Removing categories not in both train and val...")
    df_train, df_val = remove_categories_not_in_both(df_train, df_val, categorical_columns)

    print("Encode categorical and boolean...")
    # create a list of categorical variables to encode
    # CAT-BOOSTING
    encoder = CatBoostEncoder(verbose=1)
    df_train = encoder.fit_transform(df_train, train_is_installed)
    print("Creating optuna study...")
    study: optuna.Study = optuna.create_study(
        directions=["minimize", "maximize"],
        storage=OPTUNA_STORAGE,
        study_name=studyName,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial=trial,
            df_train=df_train,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("END")
