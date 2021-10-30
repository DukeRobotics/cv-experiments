from train import main as train
from configurations import *
import argparse
from typing import List
from image_preprocessing import *


class ModelConfig:
    def __init__(self, image_preprocessing=no_preprocessing, epochs=DEFAULT_EPOCHS, seed=0):
        self.image_preprocessing = image_preprocessing
        self.epochs = epochs
        self.seed = seed


def main(args):
    image_preprocessing_conditions = args.image_preprocessing_conditions
    runs = args.runs

    seeds = args.seeds
    epochs = args.epochs

    assert len(seeds) == runs, 'List of seeds must have same number of elements as runs'

    # For each image_preprocessing_condition, train the model runs number of times and save the results
    for image_preprocessing_condition in image_preprocessing_conditions:
        for i in range(runs):
            print(f'---Running {image_preprocessing_condition.__name__}, run {i}---')
            # Train the model with the image_preprocessing_condition
            curr_config = ModelConfig(image_preprocessing_condition, epochs, seeds[i])
            train(curr_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-preprocessing-conditions', type=List, default=DEFAULT_IMAGE_PREPROCESSING_ALGORITHMS)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS)
    parser.add_argument('--seeds', type=List, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    main(args)
