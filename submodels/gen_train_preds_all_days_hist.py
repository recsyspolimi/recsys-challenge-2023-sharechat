import pandas as pd
from pathlib import Path
from typing import Dict, List
from IPython.display import display, HTML
import numpy as np
from tqdm.notebook import tqdm
import os
from sklearn.ensemble import HistGradientBoostingClassifier

TRAIN_VAL_DATA_PATH = 'data/train_valEncoded.parquet'
TEST_DATA_PATH = 'data/testEncoded.parquet'

def gen_preds_hist():
    df: pd.DataFrame = pd.read_parquet(TRAIN_VAL_DATA_PATH)

    df = df.astype({f"f_{i}": "category" for i in range(2, 33)})
    df = df.astype({'f_1': 'int'})

    booleans = range(33, 41+1)
    for i in booleans:
        df = df.astype({'f_{}'.format(i): 'int'})

    df = df.astype({'is_clicked': 'int'})
    df = df.astype({'is_installed': 'int'})

    df['f_9'] = df['f_9'].astype(str)
    df['f_9'].mask((df['f_1'] ==45) | (df['f_1'] ==52) | (df['f_1'] ==59) | (df['f_1'] ==66),'1', inplace=True)
    df['f_9'].mask((df['f_1'] ==46) | (df['f_1'] ==53) | (df['f_1'] ==60),'2', inplace=True)
    df['f_9'].mask((df['f_1'] ==47) | (df['f_1'] ==54) | (df['f_1'] ==61),'3', inplace=True)
    df['f_9'].mask((df['f_1'] ==48) | (df['f_1'] ==55) | (df['f_1'] ==62),'4', inplace=True)
    df['f_9'].mask((df['f_1'] ==49) | (df['f_1'] ==56) | (df['f_1'] ==63),'5', inplace=True)
    df['f_9'].mask((df['f_1'] ==50) | (df['f_1'] ==57) | (df['f_1'] ==64),'6', inplace=True)
    df['f_9'].mask((df['f_1'] ==51) | (df['f_1'] ==58) | (df['f_1'] ==65),'0', inplace=True)
    df['f_9'] = df['f_9'].astype("category")

    df["f_new43-51"]=df["f_43"]*df["f_51"]
    df["f_new43-66"]=df["f_43"]*df["f_66"]
    df["f_new43-70"]=df["f_43"]*df["f_70"]
    df["f_new51-70"]=df["f_51"]*df["f_70"]
    df["f_new51-66"]=df["f_51"]*df["f_66"]
    df["f_new66-70"]=df["f_66"]*df["f_70"]

    categorical_columns: List[str] = [f"f_{i}" for i in range(2, 33)]

    to_delete = ['f_0','f_7','f_11','f_27','f_28','f_29', 'f_59', 'f_64']
    df=df.drop(columns=to_delete)

    for col in ['f_7','f_11','f_27','f_28','f_29']:
        categorical_columns.remove(col)

    #get unique values for each categorical column
    categorical_dims =  {}
    for col in categorical_columns:
        categorical_dims[col] = df[col].nunique()

    categorical_index =  {}
    for col in categorical_columns:
        categorical_index[col] = df.columns.get_loc(col)
    #keep only the values of the dictionary
    categorical_index = list(categorical_index.values())

    X = df.drop(columns=['is_clicked','is_installed'])
    y = df[['is_clicked','is_installed']]

    max_bins = 229
    hyper_params ={
        'loss': 'log_loss',
        'l2_regularization':79.97112528034731,
        'learning_rate':  0.10659053116784906,
        "max_depth": 12,  
        "max_leaf_nodes": 130,
        "min_samples_leaf": 184,
        "max_bins": max_bins, 
        "max_iter": 261,
        "verbose": 0,
        "random_state":0}
    categorical_columns1 = categorical_columns.copy()
    #remove from categorical_columns1 those columns with more features than max_bins
    for col in categorical_columns:
        if categorical_dims[col] >= max_bins:
            categorical_columns1.remove(col)

    df: pd.DataFrame = pd.read_parquet(TRAIN_VAL_DATA_PATH) 
    row_ids_all = []
    predictions_all = []
    for i in range(45,67):
        row_ids = df[df['f_1']==i]['f_0']
        X_train = X[X['f_1'] != i]
        y_train = y[X['f_1'] != i]
        clf = HistGradientBoostingClassifier(**hyper_params, categorical_features=categorical_columns1)
        clf.fit(X_train, y_train["is_installed"])
        results_installed = clf.predict_proba(X[X['f_1']==i])
        assert len(row_ids)==len(results_installed[:,1])
        for row in row_ids:
            row_ids_all.append(row)
        for pred in results_installed[:,1]:
            predictions_all.append(pred)
        assert len(row_ids_all) == len(predictions_all)
        print("Done day ",i)

    submission = pd.DataFrame()
    submission["f_0"] = row_ids_all
    submission["hist"] = predictions_all
    submission.to_csv('predictions/hist_train_predictions_all_days.csv', sep ='\t', index=False)