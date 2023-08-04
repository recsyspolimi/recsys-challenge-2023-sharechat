import sys

import pandas as pd

#sys.path.append("..")
from utils.rerank_predictions import rerank_predictions


def final_hybrid_gen():
    """
    Create the final hybrid model.
    """
    # read the predictions of the single models
    cat = pd.read_csv("predictions/cat_predictions.csv", sep="\t")
    light = pd.read_csv("predictions/light_predictions.csv", sep="\t")
    hist = pd.read_csv("predictions/hist_predictions.csv", sep="\t")
    hist_hybrid = pd.read_csv("predictions/hist_hybrid_predictions.csv", sep="\t")
    light_hybrid = pd.read_csv("predictions/light_hybrid_predictions.csv", sep="\t")

    light.rename(columns={"is_installed": "light"}, inplace=True)
    light.drop(columns=["is_clicked"], inplace=True)

    hist.rename(columns={"is_installed": "hist"}, inplace=True)
    hist.drop(columns=["is_clicked"], inplace=True)

    hist_hybrid.rename(columns={"is_installed": "hist_hybrid"}, inplace=True)
    hist_hybrid.drop(columns=["is_clicked"], inplace=True)

    light_hybrid.rename(columns={"is_installed": "light_hybrid"}, inplace=True)
    light_hybrid.drop(columns=["is_clicked"], inplace=True)

    final = cat.merge(light, on="RowId")
    final = final.merge(hist, on="RowId")
    final = final.merge(hist_hybrid, on="RowId")
    final = final.merge(light_hybrid, on="RowId")

    final["is_installed"] = (
        0.46 * final["is_installed"]
        + 0.28 * final["light_hybrid"]
        + 0.08 * final["hist_hybrid"]
        + 0.14 * final["light"]
        + 0.04 * final["hist"]
    )
    final.drop(columns=["light", "hist", "hist_hybrid", "light_hybrid"], inplace=True)

    final = rerank_predictions(final)

    # save the final hybrid model
    final.to_csv("predictions/final_hybrid.csv", sep="\t", index=False)
