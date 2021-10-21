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


#Credit to this research paper: http://www.lsis.org/rov3d/article/art_AmineRhone2012.html
def amine_rhone(image_path):

    #Read Image
    image = cv2.imread(image_path)

    #Get Color Information
    blue_min = np.percentile(image[:, :, 0], 4)
    blue_max = np.percentile(image[:, :, 0], 96)
    green_min = np.percentile(image[:, :, 1], 4)
    green_max = np.percentile(image[:, :, 1], 96)
    red_min = np.percentile(image[:, :, 2], 4)
    red_max = np.percentile(image[:, :, 2], 96)

    #Edit Image
    image = image.astype(np.int64) #To avoid overflow issues
    image[:, :, 0] = (255*(image[:, :, 0] - blue_min))/(blue_max-blue_min)
    image[:, :, 1] = (255*(image[:, :, 1] - green_min))/(green_max-green_min)
    image[:, :, 2] = (255*(image[:, :, 2] - red_min))/(red_max-red_min)
    
    #Return Image
    return image


if __name__ == '__main__':
    # Test an image processing algorithm
    pass
