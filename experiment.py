from train import main as train
from configurations import *
import argparse
from typing import List

# This file will contain code to run the various experiments
# Will want to train the model multiple times on images from each preprocessing technique


def main(args):
    image_preprocessing_conditions = args.image_preprocessing_conditions
    runs = args.runs
    # For each image_preprocessing_condition, train the model runs number of times and save the results

    for image_preprocessing_condition in image_preprocessing_conditions:
        for i in range(runs):
            # Train the model with the image_preprocessing_condition
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-preprocessing-conditions', type=List, default=[no_preprocessing])
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()

    main(args)