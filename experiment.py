from train import main as train
from configurations import *
import argparse
import random
from typing import List

# This file will contain code to run the various experiments
# Will want to train the model multiple times on images from each preprocessing technique

class ModelConfig():

    def __init__(self, image_preprocessing=no_preprocessing, epochs=DEFAULT_EPOCHS, seed=0):
        self.image_preprocessing = image_preprocessing
        self.epochs = epochs
        self.seed = seed

def main(args):

    assert len(args.seeds) == args.runs, 'List of seeds must have same number of elements as runs'

    image_preprocessing_conditions = args.image_preprocessing_conditions
    runs = args.runs
    seeds = args.seeds

    # For each image_preprocessing_condition, train the model runs number of times and save the results
    for image_preprocessing_condition in image_preprocessing_conditions:
        for i in range(runs):
            # Train the model with the image_preprocessing_condition
            curr_config = ModelConfig(image_preprocessing_condition, DEFAULT_EPOCHS, seeds[i])
            train(curr_config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-preprocessing-conditions', type=List, default=[no_preprocessing])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seeds', type=List, default=[0, 1, 2])
    args = parser.parse_args()

    main(args)