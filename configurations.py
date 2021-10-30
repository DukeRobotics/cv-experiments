import os
from image_preprocessing import *

# This file will contain any constants or configurations
# This should have any variables that someone might want to change so that the other scripts don't have to be edited

CLASSES = ['start_gate', 'start_tick']

IMAGE_PREPROCESSING_ALGORITHMS = [no_preprocessing, amine_rhone, clahe, dcp, gbrc, gc]
DEFAULT_IMAGE_PREPROCESSING_ALGORITHMS = [no_preprocessing, amine_rhone, clahe, gc]
ORIGINAL_IMAGE_FOLDER = 'original'

# Values
DEFAULT_EPOCHS = 50
DEFAULT_RUNS = 3
DEFAULT_SEEDS = [0, 1, 2]  # The length of this should be equal to the value of DEFAULT_RUNS

# Paths
BASE_DATA_DIR = os.path.join('data')
BASE_OUTPUT_DIR = os.path.join('output')

TRAINING_FOLDER_NAME = 'train'
VALIDATION_FOLDER_NAME = 'validation'

# File formats
IMAGE_EXT = '.jpg'
LABEL_EXT = '.xml'

if __name__ == '__main__':
    # When running this as main, this will verify that the configurations are valid
    assert len(DEFAULT_SEEDS) == DEFAULT_RUNS, 'The length of DEFAULT_SEEDS should equal the value of DEFAULT_RUNS'
    assert DEFAULT_EPOCHS > 0, 'DEFAULT_EPOCHS should be positive'

