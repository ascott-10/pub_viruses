# main.py

#### Libraries ####
import os
import sys

import pandas as pd
import numpy as np

from config import *
from codes import main_classify_pipeline

#### Functions ####
def choose_pipeline(pipeline):
    if pipeline == "classification":

        main_classify_pipeline.run(FORCE_TRAIN_CLASSIFIER=True)



#### Driver ####
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pipeline = sys.argv[1]
        choose_pipeline(pipeline)
    else:
        print("Please choose a pipeline to begin: classification, morphology, statistics")
        print("Usage: python main.py <pipeline>")

