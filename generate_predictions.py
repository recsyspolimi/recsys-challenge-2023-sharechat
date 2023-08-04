import os

import numpy as np
import pandas as pd

from submodels.cat import exec_cat
from submodels.final_hybrid import final_hybrid_gen
from submodels.gen_train_preds_all_days_hist import gen_preds_hist
from submodels.gen_train_preds_all_days_light import gen_preds_light
from submodels.hist import exec_hist
from submodels.hist_hybrid import exec_hist_hybrid
from submodels.light import exec_light
from submodels.light_hybrid import exec_light_hybrid
from utils.create_parquet import convert_to_parquet
from utils.create_parquet_encoded import convert_to_parquet_enc
from utils.postprocessing import apply_postprocessing

if __name__ == "__main__":
    print("Generating predictions...")
    # convert data to parquet
    convert_to_parquet()
    convert_to_parquet_enc()
    print("Data converted to parquet")

    # generate predictions of single models
    exec_cat()  # remember: exec_cat at first time to create predictions directory
    print("Cat predictions generated")

    exec_light()
    print("Light predictions generated")

    exec_hist()
    print("Hist predictions generated")

    # train_predictions_for_hybrid_creation
    gen_preds_hist()
    gen_preds_light()
    print("Hist and Light predictions for hybrid creation generated")

    # generate predictions of hybrid models
    exec_hist_hybrid()
    exec_light_hybrid()
    print("Hist and Light hybrid predictions generated")

    # generate final hybrid model merging the predictions of the single models
    final_hybrid_gen()

    # apply a post-processing function
    apply_postprocessing()

    print("Predictions generated, final results are in final_predictions")

