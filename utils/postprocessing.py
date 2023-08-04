import pandas as pd
import numpy as np

"""applies this piece-wise linear function to remodule the predictions
"""
def piecewise(x, a = 0.0012233892209209916, b = 1.0060693574363284):
    x = np.clip(x, a, b)
    return (x - a) / (b - a)

def apply_postprocessing():
    submission = pd.read_csv("predictions/final_hybrid.csv", sep="\t")

    predictions = piecewise(submission['is_installed'])
    submission['is_installed'] = predictions

    submission.to_csv("predictions/final_predictions.csv", sep="\t", index=False)
