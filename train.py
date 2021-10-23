from detecto import core, utils, visualize
from configurations import *

import matplotlib.pyplot as plt

import random
import argparse
import pandas as pd

from datetime import date

# Directory Structure:
# data
#   no_preprocessing
#       train
#       validation
#   some_other_preprocessing
#       train
#       validation

# output
#   no_preprocessing
#   some_other_preprocessing


def main(args):

    epochs = args.epochs
    image_preprocessing = args.image_preprocessing
    seed = args.seed

    assert epochs > 0, 'Epochs should be positive'
    assert image_preprocessing in IMAGE_PREPROCESSING_ALGORITHMS, f'Couldnt find {image_preprocessing} in ' \
                                                                  f'IMAGE_PREPROCESSING_ALGORITHMS '
    random.seed(seed)

    # Will need to apply image preprocessing at some point
    # Either create folder beforehand or incorporate into custom dataset

    # Get training dataset
    train_dataset = core.Dataset(os.path.join(BASE_DATA_DIR, image_preprocessing, TRAINING_FOLDER_NAME))
    # Get validation dataset
    val_dataset = core.Dataset(os.path.join(BASE_DATA_DIR, image_preprocessing, VALIDATION_FOLDER_NAME))

    # Create model
    model = core.Model(CLASSES)

    # Train the model
    losses = model.fit(dataset=train_dataset,
                       val_dataset=val_dataset,
                       epochs=epochs,
                       verbose=True)

    # Save the results in .png and probably .txt format as well
    # Will save results in output/{image_preprocessing}
    output_dir = os.path.join(BASE_OUTPUT_DIR, image_preprocessing)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    plt.figure()
    plt.plot(losses)
    
    # Get today's date for model identification
    today = date.today.strftime("%d/%m/%Y %H:%M:%S")

    # Save loss graph to the proepr directory
    plt.savefig(os.path.join(output_dir, f'losses-{image_preprocessing}-epochs{epochs}-seed{seed}-date{today}.png'))
    plt.close()

    df = pd.DataFrame(losses, columns=['loss'])
    df.to_csv(os.path.join(output_dir, f'losses-{image_preprocessing}-epochs{epochs}-seed{seed}-date{today}.txt'), header=None, index=None, sep=' ', mode='a')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-preprocessing', type=str, default='no_preprocessing')
    parser.add_argument('--savename', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    main(args)