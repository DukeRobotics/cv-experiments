import os

# This file will contain any constants or configurations
# This should have any variables that someone might want to change so that the other scripts don't have to be edited

CLASSES = ['start_gate', 'start_tick']
IMAGE_PREPROCESSING_ALGORITHMS = ['no_preprocessing', 'CLAHE', 'DCP', 'Fusion-Matlab', 'GBdehazingRCorrection', 'GC', 'HE', 'IBLA', 'LowComplexityDCP', 'MIP', 'NewOpticalModel', 'RayleighDistribution', 'RGHS', 'RoWS', 'UCM', 'UDCP', 'ULAP']

# Values
DEFAULT_EPOCHS = 50

# Paths
BASE_DATA_DIR = os.path.join('data')
BASE_OUTPUT_DIR = os.path.join('output')
TRAINING_FOLDER_NAME = 'train_images'
VALIDATION_FOLDER_NAME = 'validation_images'
