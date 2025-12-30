#codes/utils.py

#### Libraries ####
import os
from pathlib import Path

import pandas as pd
import numpy as np

from config import *

#### Functions ####

def load_images(input_path, label = None):
    """This is a generic function that will load images from a specified input path and return an organized dataframe
    Usage: files_df = load_images(input_path, label = "label")"""
    image_filepaths = []
    image_labels = []
    image_ids = []

    for files in os.listdir(input_path):
        if files.endswith((".png", ".tif", ".jpg")):
            if label is not None:
                image_labels.append(label)

            image_filepaths.append(os.path.join(input_path, files))
            image_ids.append(Path(files).stem)

    if len(image_labels) > 0:
        all_files = pd.DataFrame([image_ids, image_filepaths, image_labels], index=['im_id', 'file_path', 'class']).T
    else:
        all_files = pd.DataFrame([image_ids, image_filepaths], index=['im_id', 'file_path']).T
    
    all_files_df = all_files.copy()

    print(all_files_df.head())
    return all_files_df



# End codes/utils.py