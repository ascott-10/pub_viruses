# setup.py

#### Libraries ####
import os
import sys

import pandas as pd
import numpy as np

from config import *
from utils import load_images

#### Functions ####

def set_working_project():
    """This function allows the user to set the working data"""

def set_project_directory():
    """This function will set up the project directory after the user initializes the starting directory in config.py"""

"""
mut_files_df = load_images(SEGMENTED_MUT_DIR, label = "mut")
wt_files_df = load_images(SEGMENTED_WT_DIR, label = "wt")
test_files_df = load_images(SEGMENTED_TEST_DIR, label = "test")

all_files_df = pd.concat([mut_files_df, wt_files_df, test_files_df])
all_files_df.to_csv(os.path.join(REFERENCES_DIR, "all_segmented_images.csv"))

"""
