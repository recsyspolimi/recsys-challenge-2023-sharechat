#!/usr/bin/env python
# coding: utf-8

# # Model analysis

# ## Import dataset & libraries

# In[1]:


import os
import pickle
from typing import Callable, Dict, List, Set, Tuple

from xgboost import XGBClassifier

import boto3
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "default")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "default")
regiorn_name = "eu-west-1"
bucket_name = "challenge23"

s3 = boto3.resource(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    config=boto3.session.Config(region_name=regiorn_name),
)
bucket = s3.Bucket(bucket_name)


# In[3]:


threshold_day_train = 62


# ### Choose the dataset

# In[4]:


df_train_val = pd.read_parquet('data/train_valEncoded.parquet').reset_index(drop=True)

df_train_val = df_train_val.astype({f"f_{i}": "category" for i in range(2, 33)})
df_train_val = df_train_val.astype({'f_1': 'int'})

df_train_val = df_train_val.astype({'is_clicked': 'int'})
df_train_val = df_train_val.astype({'is_installed': 'int'})

df_train_val.info()


# In[5]:


df_test = pd.read_parquet('data/testEncoded.parquet').reset_index(drop=True)
df_test = df_test.astype({f"f_{i}": "category" for i in range(2, 33)})
df_test = df_test.astype({'f_1': 'int'})
df_test.info()


# In[6]:


df_train_val


# In[7]:


df_test


# ## Preprocessing for all

# In[8]:


from utils.preprocessing import *


# In[9]:


boolean_columns: List[str] = [f"f_{i}" for i in range(33, 42)]
for col in boolean_columns:
    df_train_val[col] = df_train_val[col].astype(bool)
    df_test[col] = df_test[col].astype(bool)


# In[10]:


CATEGORICAL_TO_DROP: list = [
    "f_7",
    "f_9",
    'f_11',
    "f_23",
    "f_24",
    "f_25",
    "f_26",
    "f_27",
    "f_28",
    "f_29",
]

NUMERICAL_TO_DROP: list = [
    "f_55",
    "f_59",
    "f_64",
    "f_65",
    "f_66",
]


# In[11]:


df_train = df_train_val[df_train_val["f_1"] < threshold_day_train].reset_index(drop=True)
df_val = df_train_val[df_train_val["f_1"] >= threshold_day_train].reset_index(drop=True)


# In[12]:


df_train_val = df_train_val.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
df_train = df_train.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
df_val = df_val.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)
df_test = df_test.drop(columns=CATEGORICAL_TO_DROP + NUMERICAL_TO_DROP)

# In[13]:


cat_features = list(df_train_val.select_dtypes('category').columns)
for col in cat_features:
    df_train_val[col] = df_train_val[col].astype(str)
    df_train[col] = df_train[col].astype(str)
    df_val[col] = df_val[col].astype(str)
    df_test[col] = df_test[col].astype(str)

df_train_val.info()

# ## Model specific preprocessing

# In[14]:


cat_params = {
    "bootstrap_type": "No",
    "border_count": 150,
    "depth": 7,
    "has_time": True,
    "iterations": 320,
    "l2_leaf_reg": 0.0008990664831889606,
    "learning_rate": 0.026171005504059278,
    "random_strength": 0.002795826517884825
}


# ## Model train

# In[15]:


y_train = df_train["is_installed"]
y_train_val = df_train_val["is_installed"]
y_val = df_val["is_installed"]
test_row_id = df_test["f_0"]

X_train = df_train.drop(columns=["is_installed", "is_clicked"])
X_val = df_val.drop(columns=["is_installed", "is_clicked"])
X_train_val = df_train_val.drop(columns=["is_installed", "is_clicked"])
X_test = df_test.drop(columns='f_0')


# In[16]:


X_train = X_train.sort_values(by="f_1", ascending=True)

X_train_val_cat = df_train_val.sort_values(by="f_1", ascending=True).drop(columns=["is_installed", "is_clicked"])
y_train_val_cat = df_train_val.sort_values(by="f_1", ascending=True)["is_installed"]

# In[17]:

X_train_val = X_train_val.drop(columns=['f_0'])
X_train = X_train.drop(columns=['f_0', 'f_1'])
X_val = X_val.drop(columns=['f_0', 'f_1'])


# In[ ]:


from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(**cat_params, od_type="Iter", od_wait=30, task_type='GPU', eval_metric="CrossEntropy")
cat_clf.fit(
    X_train[cat_features],
    y_train,
    eval_set=(X_val[cat_features], y_val),
    cat_features=cat_features,
    use_best_model=True,
    verbose=True,
)


# In[ ]:


cat_preds_train = cat_clf.predict_proba(X_train_val.drop(columns='f_1')[cat_features])[:,-1]


# In[ ]:


cat_train_val = CatBoostClassifier(**cat_params, task_type='GPU', eval_metric="CrossEntropy")
cat_train_val.fit(
    X_train_val_cat[cat_features],
    y_train_val_cat,
    cat_features=cat_features,
    verbose=True,
)


# In[ ]:


cat_preds_train_val = cat_train_val.predict_proba(X_test.drop(columns='f_1')[cat_features])[:,-1]


# In[ ]:


X_train_val['cat_preds'] = cat_preds_train
X_test['cat_preds'] = cat_preds_train_val


# In[ ]:


xgb_params = {
    'lambda': 157.91428030085532,
    'alpha': 4.117893558091652,
    'eta': 0.093412940224152,
    'gamma': 80.65484597175121,
    'scale_pos_weight': 0.7517712977891217,
    "max_depth": 43,
    "objective": "binary:logistic",
    "sampling_method": 'uniform',
    'min_child_weight': 58,
    "max_leaves": 84,
    "max_bin": 184,
    "random_state":0
}


# In[ ]:


from xgboost import XGBClassifier

model = XGBClassifier(
    **xgb_params,
    tree_method='gpu_hist',
)
model.fit(
    X_train_val.drop(columns=cat_features),
    y_train_val,
    verbose=True
)

xgb_probs = model.predict_proba(X_test.drop(columns=cat_features))[:,-1]


# ## Create submission

# In[ ]:


filename = 'vamos.csv'


# In[ ]:


submission = pd.DataFrame({"row_id": test_row_id, "is_clicked": 0.0 ,"is_installed": xgb_probs})
submission.to_csv(filename, index=False, sep='\t')

print("Uploading predictions to s3...")
bucket.upload_file(
    'vamos.csv',
    "submissions/vamos.csv",
)
print("Uploaded predictions")