import os
import torch
from torchvision import models

#### Directories ####
BASE_DIR = "/home/ascott10/documents/projects/pub_viruses"
REFERENCES_DIR = os.path.join(BASE_DIR, "references")

WORKING_DATA_DIR = os.path.join(BASE_DIR, "working_data")
SAM_2_WORKING_DATA_DIR = os.path.join(BASE_DIR, "SAM_2_working_data")
WORKING_DATA_CLASSIFIER = os.path.join(SAM_2_WORKING_DATA_DIR, "classifier")
CLASSIFIER_WEIGHTS_PATH = os.path.join(WORKING_DATA_CLASSIFIER, "resnet_weights")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAM_2_RESULTS_DIR = os.path.join(BASE_DIR, "SAM_2_results")
CLASSIFIER_TRAINING_RESULTS = os.path.join(RESULTS_DIR, "classifier", "training")
CLASSIFIER_TESTING_RESULTS = os.path.join(RESULTS_DIR, "classifier", "testing") 


SEGMENTED_TEST_DIR = os.path.join(BASE_DIR, "segmented_images/test_images")
SEGMENTED_WT_DIR = os.path.join(BASE_DIR,"segmented_images/wildtype_manual_correction")
SEGMENTED_MUT_DIR = os.path.join(BASE_DIR,"segmented_images/mutant_manual_correction")

SAM_2_SEGMENTED_DIR = os.path.join(BASE_DIR,"sam_masks")

#### File paths ####
ALL_SEGMENTED_IMAGES = "/home/ascott10/documents/projects/pub_viruses/references/all_segmented_images.csv"
SAM_2_SEGMENTED_IMAGES = "/home/ascott10/documents/projects/pub_viruses/references/sam_2_segmented_images.csv"


PRE_TRANED_MODEL = models.resnet18(pretrained=False)

#### Parameters ####

# fixed class mapping
CLASS_TO_IDX = {"mut": 0, "wt": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


FORCE_TRAIN_CLASSIFIER = True   # flip to False if you want to just load weights



# ======== TRAINING SETTINGS ========

NUM_EPOCHS = 20
NUM_EPOCHS_CLASSIFY = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# ======== DEVICE SETTING ========

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== OTHER SETTINGS ========

SUBSAMPLE = 25
IMAGE_SIZE = (256,256)
SEED = 42
NEW_CLASSIFY = True

# Color scheme for plotting
TRAIN_COLOR = 'mediumseagreen'
VAL_COLOR = 'darkgreen'
# General green-themed plotting colors
FONT_SIZE = 14  # or whatever size you want
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
COLOR_GENERAL_1 = "#4CAF50"       # green
COLOR_GENERAL_2 = "#81C784"       # lighter green
COLOR_MUTANT = "#f28cb1"          # default pink
COLOR_MUTANT_BRIGHT = "#ffb6c1"   # brighter pink
COLOR_WILDTYPE = "#66bb6a"        # green
SAVE_DPI = 300
