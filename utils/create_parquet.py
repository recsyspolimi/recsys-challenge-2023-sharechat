"""
Convert csv data files to parquet files.
The parquet files are much smaller than the csv files and are faster to load.
They also keep the data types of the columns.
"""
import gc
from pathlib import Path
from typing import Dict
import os
import pandas as pd

def convert_to_parquet():

    root_dir: Path = Path('data')

    """
    Define the data types for each column.
    The data types of different columns are:
    a. RowId(f_0)
    b. Date(f_1)
    c. Categorical features(f_2 to f_32)
    d. Binary features(f_33 to f_41)
    e. Numerical features(f_42 to f_79)
    f. Labels(is_clicked, is_installed)
    """
    
    columns_dtypes_test: Dict[str, str] = {
        "f_0": "UInt32",
        "f_1": "UInt8",
    }
    for i in range(2, 33):
        columns_dtypes_test[f"f_{i}"] = "category"
    for i in range(33, 42):
        columns_dtypes_test[f"f_{i}"] = "boolean"
    for i in range(42, 80):
        columns_dtypes_test[f"f_{i}"] = "float32"


    columns_dtypes_train: Dict[str, str] = columns_dtypes_test.copy()
    columns_dtypes_train.update({"is_clicked": "boolean", "is_installed": "boolean"})

    paths = list((root_dir / "train").rglob("*.csv"))
    paths.sort()
    # Train
    print("Reading train/val data...")
    train_val_df: pd.DataFrame = pd.concat(
        [pd.read_csv(csv_file, sep="\t") for csv_file in paths]
    )
    print("Converting train data to parquet...")
    train_val_df = train_val_df.astype(columns_dtypes_train)
    train_val_df.to_parquet("data/train_val.parquet")
    del train_val_df
    gc.collect()

    # Test
    print("Reading test data...")
    test_df: pd.DataFrame = pd.read_csv(root_dir / "test" / "000000000000.csv", sep="\t")
    print("Converting test data to parquet...")
    test_df = test_df.astype(columns_dtypes_test)
    test_df.to_parquet("data/test.parquet")
