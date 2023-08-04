import pandas as pd
from pathlib import Path
from typing import Dict, List
from IPython.display import display, HTML
import numpy as np
from tqdm.notebook import tqdm
import os
import lightgbm as lgb

TRAIN_VAL_DATA_PATH = 'data/train_val.parquet'
TEST_DATA_PATH = 'data/test.parquet'

def gen_preds_light():
    df: pd.DataFrame = pd.read_parquet(TRAIN_VAL_DATA_PATH) 
    df = df.astype({f"f_{i}": "category" for i in range(2, 33)})

    df = df.astype({'f_1': 'int'})

    booleans = range(33, 41+1)
    for i in booleans:
        df = df.astype({'f_{}'.format(i): 'int'})

    df = df.astype({'is_clicked': 'int'})
    df = df.astype({'is_installed': 'int'})

    df['f_9'] = df['f_9'].astype(str)
    df['f_9'].mask((df['f_1'] ==45) | (df['f_1'] ==52) | (df['f_1'] ==59) | (df['f_1'] ==66),'6675', inplace=True)
    df['f_9'].mask((df['f_1'] ==46) | (df['f_1'] ==53) | (df['f_1'] ==60),'14659', inplace=True)
    df['f_9'].mask((df['f_1'] ==47) | (df['f_1'] ==54) | (df['f_1'] ==61),'9638', inplace=True)
    df['f_9'].mask((df['f_1'] ==48) | (df['f_1'] ==55) | (df['f_1'] ==62),'23218', inplace=True)
    df['f_9'].mask((df['f_1'] ==49) | (df['f_1'] ==56) | (df['f_1'] ==63),'869', inplace=True)
    df['f_9'].mask((df['f_1'] ==50) | (df['f_1'] ==57) | (df['f_1'] ==64),'21533', inplace=True)
    df['f_9'].mask((df['f_1'] ==51) | (df['f_1'] ==58) | (df['f_1'] ==65),'31372', inplace=True)
    df['f_9'] = df['f_9'].astype("category")
    df["f_new43-51"]=df["f_43"]*df["f_51"]
    df["f_new43-66"]=df["f_43"]*df["f_66"]
    df["f_new43-70"]=df["f_43"]*df["f_70"]
    df["f_new51-70"]=df["f_51"]*df["f_70"]
    df["f_new51-66"]=df["f_51"]*df["f_66"]
    df["f_new66-70"]=df["f_66"]*df["f_70"]

    X = df.drop(columns=['is_clicked','is_installed'])
    y = df[['is_clicked','is_installed']]

    to_delete = ['f_0','f_7','f_11','f_27','f_28','f_29']
    X = X.drop(columns=to_delete)

    light_params= {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'xentropy',
    'learning_rate': 0.025199403190325702,
    'max_depth': 284,
    'num_leaves': 300,
    'max_iter': 473,
    'cat_smooth': 150,
    'cat_l2': 0,
    'verbose': -1
  }

    df: pd.DataFrame = pd.read_parquet(TRAIN_VAL_DATA_PATH)
    row_ids_all = []
    predictions_all = []
    for i in range(45,67):
        row_ids = df[df['f_1']==i]['f_0']
        X_actual = X[X['f_1'] != i]
        y_actual = y[X['f_1'] != i]
        lgb_train = lgb.Dataset(X_actual, y_actual['is_installed'])
        gbm_installed = lgb.train(light_params, lgb_train)
        predictions = gbm_installed.predict(X[X['f_1']==i])
        assert len(row_ids)==len(predictions)
        for row in row_ids:
            row_ids_all.append(row)
        for pred in predictions:
            predictions_all.append(pred)
        assert len(row_ids_all) == len(predictions_all)
        print("Done day ",i)

    submission = pd.DataFrame()
    submission["f_0"] = row_ids_all
    submission["light"] = predictions_all
    submission.to_csv('predictions/light_train_predictions_all_days.csv', sep ='\t', index=False)