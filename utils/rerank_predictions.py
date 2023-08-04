import numpy as np
import pandas as pd


def rerank_predictions(submission: pd.DataFrame)->pd.DataFrame:
    """
    Rerank predictions of the model.
    """
    submission.sort_values(by=['is_installed'], ascending=True, inplace=True)

    #put the ordered rowid in a list
    rowids = submission["RowId"].tolist()
    final = pd.read_csv("predictions/cat_predictions.csv", sep="\t")
    final.sort_values(by=['is_installed'], inplace=True)
    inst = final["is_installed"]

    inst = inst.values
    inst = pd.Series(inst)

    subinst = submission["is_installed"]
    subinst = subinst.values
    subinst = pd.Series(subinst)

    rowids = pd.Series(rowids)
    rerank = pd.DataFrame()
    rerank["RowId"]=rowids
    rerank["is_installed"]=inst
    rerank["install"]=subinst
    
    rerank.drop(columns=['install'], inplace=True)

    rerank["is_clicked"]=0.33

    return rerank