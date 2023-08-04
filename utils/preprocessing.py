from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing

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


def labelEncodeCats(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Label encode categorical variables.

    Args:
        train_val_df (pd.DataFrame): The train and val dataframe.
        test_df (pd.DataFrame): The test (or validation) dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The dataframes with the encoded categorical variables.
    """
    train_df = train_df.reset_index(drop=True)
    le = preprocessing.LabelEncoder()
    concat_df = pd.concat([train_df, test_df])
    for i in range(2, 33):
        le.fit(concat_df[f"f_{i}"])
        train_df[f"f_{i}"] = le.transform(train_df[f"f_{i}"])
        test_df[f"f_{i}"] = le.transform(test_df[f"f_{i}"])
    return train_df, test_df


def trigonometric_date_encoding(df: pd.DataFrame, column: str = "f_1") -> pd.DataFrame:
    """Encode date as sin and cos of the day of the week.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str, optional): The column name with the date to encode. Defaults to "f_1".

    Returns:
        pd.DataFrame: The dataframe with the encoded date.
            The new columns are called sin_date and cos_date.
            The original column is not dropped.
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


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    coefficient: int,
    means: pd.Series,
    stds: pd.Series,
    percentage: float = 0.001,
) -> pd.DataFrame:
    """Remove outliers from the dataframe in the specified columns (use numerical columns!).

    Args:
        df (pd.DataFrame): The dataframe.
        columns (List[str]): The columns from which to remove outliers.
        coefficient (int): The coefficient to use to determine if a value is an outlier.
        means (pd.Series): The means of the columns.
        stds (pd.Series): The stds of the columns.
        percentage (float, optional): The maximum percentage of values to replace. Defaults to 0.001.

    Returns:
        pd.DataFrame: The dataframe without outliers.
    """
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


def encode_counters(
    df: pd.DataFrame,
    columns: List[str],
    mins: Optional[pd.Series] = None,
    steps: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Make the counter columns start from zero and have a distance of 1 between each point.

    Args:
        df (pd.DataFrame): The dataframe.
        columns (List[str]): The columns to encode.
        mins (Optional[pd.Series): The minimum values of the columns.
            This is useful if you want to encode the val set with the mins of the train set.
        steps (Optional[pd.Series]): The distance between each point.
            If None, it is calculated. Defaults to None.
            This is useful if you want to encode the val set with the distances of the train set.

    Returns:
        pd.DataFrame: The dataframe with the encoded columns.
        pd.Series: The minimum values of the columns. Useful for encoding the val set.
        pd.Series: The distance between each point. Useful for encoding the val set.
    """
    if mins is None:
        mins = df[columns].min()
    if steps is None:
        steps = pd.Series({col: np.min(np.diff(np.sort(df[col].unique()))) for col in columns})
    for col in columns:
        df[col] = (df[col] - mins[col]) / steps[col]
    return df, mins, steps

def drop_useless(df):
    d = df.drop(columns=CATEGORICAL_TO_DROP)
    d = d.drop(columns=NUMERICAL_TO_DROP)
    # df.drop(columns=BINARY_TO_DROP)
    return d

def preprocessingNN(trainVal_df: pd.DataFrame, threshold_day: int, test_df: pd.DataFrame):
    
    
    """ Preprocessing for all the features in the dataframe.

    Args:
        trainVal_df (pd.DataFrame): The training+validation dataframe.
        threshold_day (int): The day we use to split between training and validation, if set to None we are using all the data to train
        test_df (pd.DataFrame): The test dataframe

    Returns:
        X_train: Preprocessed dataframe with the training data.
        X_val (Optional): Preprocessed dataframe with the validation data.
        X_test: Preprocessed dataframe with the test data.
        Y_train: Dataframe with the train labels
        Y_val (Optional): Dataframe with the validation labels
    """
    max_day = trainVal_df['f_1'].max()
    
    if threshold_day is not None:
        if threshold_day <= 45 or threshold_day >= max_day:
            raise Exception("Incorrect threshold day!")
    
    global NUMERICAL_NON_COUNTERS
    dataset_cat = trainVal_df.drop(trainVal_df.columns[range(33, 80)], axis = 1)
    dataset_cat = dataset_cat.drop(columns=CATEGORICAL_TO_DROP)
    dataset_cat = dataset_cat.drop(columns=['f_0'])
    test_cat = test_df.drop(test_df.columns[range(33, 80)], axis = 1)
    test_cat = test_cat.drop(columns=CATEGORICAL_TO_DROP)
    test_cat = test_cat.drop(columns=['f_0','f_1'])
    
    dataset_num = trainVal_df.drop(trainVal_df.columns[range(2, 42)], axis = 1)
    dataset_num = dataset_num.drop(columns=NUMERICAL_TO_DROP)
    dataset_num = dataset_num.drop(columns=['f_0'])
    test_num = test_df.drop(test_df.columns[range(2, 42)], axis = 1)
    test_num = test_num.drop(columns=NUMERICAL_TO_DROP)
    test_num = test_num.drop(columns=['f_0','f_1'])
    
    dataset_bin = trainVal_df.drop(trainVal_df.columns[42:80], axis = 1)
    dataset_bin = dataset_bin.drop(dataset_bin.columns[2:33], axis = 1)
    dataset_bin = dataset_bin.drop(columns=["f_0"])
    test_bin = test_df.drop(test_df.columns[42:80], axis = 1)
    test_bin = test_bin.drop(test_bin.columns[2:33], axis = 1)
    test_bin = test_bin.drop(columns=["f_0","f_1"])
    
    if threshold_day is None:
        X_train_cat = dataset_cat.drop(['is_clicked','is_installed'], axis = 1)
        X_train_num = dataset_num.drop(['is_clicked','is_installed'], axis = 1)
        X_train_bin = dataset_bin.drop(['is_clicked','is_installed'], axis = 1)
        Y_train = dataset_cat[['is_installed']]
    else:
        X_train_cat = dataset_cat[dataset_cat["f_1"]<threshold_day].drop(columns=['is_clicked','is_installed'])
        Y_train = dataset_cat[dataset_cat["f_1"]<threshold_day]
        Y_train = Y_train[['is_installed']]
        X_val_cat = dataset_cat[dataset_cat["f_1"]>=threshold_day].drop(columns=['is_clicked','is_installed'])
        Y_val = dataset_cat[dataset_cat["f_1"]>=threshold_day]
        Y_val = Y_val[['is_installed']]
        X_train_cat, X_val_cat = remove_categories_not_in_both(X_train_cat, X_val_cat, X_train_cat.columns)
        
        X_train_num = dataset_num[dataset_num["f_1"]<threshold_day].drop(columns=['is_clicked','is_installed'])
        X_val_num = dataset_num[dataset_num["f_1"]>=threshold_day].drop(columns=['is_clicked','is_installed'])
        
        X_train_bin = dataset_bin[dataset_bin["f_1"]<threshold_day].drop(columns=['is_clicked','is_installed'])
        X_val_bin = dataset_bin[dataset_bin["f_1"]>=threshold_day].drop(columns=['is_clicked','is_installed'])
        X_val_cat = X_val_cat.drop(columns=["f_1"])
        X_val_num = X_val_num.drop(columns=["f_1"])
        X_val_bin = X_val_bin.drop(columns=["f_1"])

        
    X_train_cat = X_train_cat.drop(columns=["f_1"])
    X_train_num = X_train_num.drop(columns=["f_1"])
    X_train_bin = X_train_bin.drop(columns=["f_1"])

    #Categorical features preprocess
    cb_encoder = ce.CatBoostEncoder()
    cb_encoder.fit(X_train_cat, Y_train)
    X_train_cat = cb_encoder.transform(X_train_cat)
    if threshold_day is not None:
        X_val_cat = cb_encoder.transform(X_val_cat)
    test_cat = cb_encoder.transform(test_cat)
    
    #Numerical features preprocess
    NUMERICAL_COUNTERS = set(X_train_num.columns).difference(set(NUMERICAL_NON_COUNTERS))
    means = X_train_num.mean()
    stds = X_train_num.std()

    X_train_num = X_train_num.fillna(means) #fill NANs
    X_train_num = remove_outliers(
        df=X_train_num, columns = X_train_num.columns, coefficient = 4, means=means, stds=stds
    )
    
    if threshold_day is not None:
        X_val_num = X_val_num.fillna(means)
        X_val_num = remove_outliers(
        df=X_val_num, columns = X_train_num.columns, means=means, stds=stds, coefficient=4
        )
        
    test_num = test_num.fillna(means)
    test_num = remove_outliers(
        df=test_num, columns = X_train_num.columns, means=means, stds=stds, coefficient=4
    )
    test_num = (test_num - means) / stds
    
    NUMERICAL_COUNTERS = list(NUMERICAL_COUNTERS)
    X_train_num[NUMERICAL_COUNTERS] = X_train_num[NUMERICAL_COUNTERS].applymap(lambda x: np.log(x) if x > 0 else x)
    if threshold_day is not None:
        X_val_num[NUMERICAL_COUNTERS] = X_val_num[NUMERICAL_COUNTERS].applymap(lambda x: np.log(x) if x > 0 else x)
    test_num[NUMERICAL_COUNTERS] = test_num[NUMERICAL_COUNTERS].applymap(lambda x: np.log(x) if x > 0 else x)
    
    scaler = MinMaxScaler()
    X_train_num[NUMERICAL_COUNTERS] = scaler.fit_transform(X_train_num[NUMERICAL_COUNTERS])
    if threshold_day is not None:
        X_val_num[NUMERICAL_COUNTERS] = scaler.transform(X_val_num[NUMERICAL_COUNTERS])
    test_num[NUMERICAL_COUNTERS] = scaler.transform(test_num[NUMERICAL_COUNTERS])
    
    scaler = StandardScaler()
    NUMERICAL_NON_COUNTERS = list(set(X_train_num.columns).intersection(set(NUMERICAL_NON_COUNTERS)))
    X_train_num[NUMERICAL_NON_COUNTERS] = scaler.fit_transform(X_train_num[NUMERICAL_NON_COUNTERS])
    if threshold_day is not None:
        X_val_num[NUMERICAL_NON_COUNTERS] = scaler.transform(X_val_num[NUMERICAL_NON_COUNTERS])
    test_num[NUMERICAL_NON_COUNTERS] = scaler.transform(test_num[NUMERICAL_NON_COUNTERS])
    
    scaler = MinMaxScaler()
    X_train_num[NUMERICAL_NON_COUNTERS] = scaler.fit_transform(X_train_num[NUMERICAL_NON_COUNTERS])
    if threshold_day is not None:
        X_val_num[NUMERICAL_NON_COUNTERS] = scaler.transform(X_val_num[NUMERICAL_NON_COUNTERS])
    test_num[NUMERICAL_NON_COUNTERS] = scaler.transform(test_num[NUMERICAL_NON_COUNTERS])    
        
    X_train = pd.concat([X_train_cat, X_train_bin, X_train_num], axis = 1)
    if threshold_day is not None:
        X_val = pd.concat([X_val_cat, X_val_bin, X_val_num], axis = 1)
    X_test = pd.concat([test_cat, test_bin, test_num], axis = 1)
    
    if threshold_day is not None:
        return X_train, X_val, X_test, Y_train, Y_val
    else:
        return X_train, None, X_test, Y_train, None
