from detecto import core, utils, visualize
from configurations import *
from data import *

import matplotlib.pyplot as plt

import random
import argparse
import pandas as pd

from datetime import datetime


def main(args):

    epochs = args.epochs
    image_preprocessing = args.image_preprocessing
    seed = args.seed

    assert epochs > 0, 'Epochs should be positive'
    assert image_preprocessing in IMAGE_PREPROCESSING_ALGORITHMS, f'Couldnt find {image_preprocessing.__name__} in ' \
                                                                  f'IMAGE_PREPROCESSING_ALGORITHMS '
    random.seed(seed)

    preprocessing_dir = os.path.join(BASE_DATA_DIR, image_preprocessing.__name__)
    train_dir = os.path.join(preprocessing_dir, TRAINING_FOLDER_NAME)
    val_dir = os.path.join(preprocessing_dir, VALIDATION_FOLDER_NAME)

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f'Creating the folders for image preprocessing: {image_preprocessing.__name__}')
        create_image_processing_folders(image_preprocessing)
    else:
        print(f'Using the existing training and validation folders at {preprocessing_dir}')

    print('Getting training and validation datasets')
    train_dataset = core.Dataset(train_dir)
    val_dataset = core.Dataset(val_dir)

    model = core.Model(CLASSES)

    print('Training the model!')
    losses = model.fit(dataset=train_dataset,
                       val_dataset=val_dataset,
                       epochs=epochs,
                       verbose=True)

    output_dir = os.path.join(BASE_OUTPUT_DIR, image_preprocessing.__name__)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get today's date for model identification
    date_and_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    print(f'Saving results into {output_dir}')
    plt.figure()
    plt.plot(losses)
    plt.title(f'Validation Losses for {image_preprocessing.__name__}, Epochs: {epochs}, Seed: {seed}')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Losses')
    plt.savefig(os.path.join(output_dir, f'losses-{image_preprocessing.__name__}-epochs{epochs}-seed{seed}'
                                         f'-datetime{date_and_time}.png'))
    plt.close()

    df = pd.DataFrame(losses, columns=['loss'])
    df.to_csv(os.path.join(output_dir, f'losses-{image_preprocessing.__name__}-epochs{epochs}-seed{seed}'
                                       f'-datetime{date_and_time}.txt'), header=None, index=None, sep=' ', mode='a')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-preprocessing', type=str, default=no_preprocessing)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    main(args)
