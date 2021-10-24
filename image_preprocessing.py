import numpy as np
import cv2


# This file will contain all of the different image processing algorithms
# Each algorithm should have its own separate function

def no_preprocessing(image_path):
    # Read in image
    img = cv2.imread(image_path)

    # Apply algorithm
    img = img

    # Return image
    return img


if __name__ == '__main__':
    # Test an image processing algorithm
    pass
