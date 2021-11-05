import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse, math, time
from os.path import exists
from image_preprocessing_pkg.DCP import main as DCP
from image_preprocessing_pkg.CLAHE import main as CLAHE
from image_preprocessing_pkg.GBdehazingRCorrection import main as GBRC
from image_preprocessing_pkg.GC import main as GC

# This file will contain all of the different image processing algorithms
# Each algorithm should have its own separate function

def main():
    # Use ArgumentParser to get functions and image optional arguments
    parser = argparse.ArgumentParser(description='Image Preprocessing')
    parser.add_argument('--functions', type=str, help='An optional argument to specify which image preprocessing functions to run (comma-seperated). If not specified, only no_preprocessing will be run. Ex: "--functions amine_rhone,clahe,gc" NOTE: no_preprocessing is always run, regardless of the specified functions.')
    parser.add_argument('--images', type=str, help='An optional argument to specify which images in the test_images folder should be used (comma-seperated). Ex: "--images frame220.jpg,frame223.jpg,frame230.jpg" If not specified, frame216.jpg will be used.')
    args = parser.parse_args()
    
    # Get list of functions
    possibles = globals().copy()
    possibles.update(locals())

    # If the functions argument was specified, check if the function exists and add it to the functions list
    # If one of the functions does not exist, let user know via command line and stop execution
    # If the functions argument was not specified, functions list only contains no_preprocessing as the default
    functionsStr = args.functions
    functions = [no_preprocessing]
    if functionsStr:
        for functionName in functionsStr.split(","):
            function = possibles.get(functionName)
            if function:
                functions.append(function)
            else:
                print("There is no image preprocessing function with name " + functionName + ".")
                return
    
    # If the images argument was specified, check if the image exists and replace the value of imgPaths with it
    # If one of the images does not exist, let the user know via command line and stop execution
    # If the images argument was not specified, imgPaths contains frame216.jpg as the default
    imgPaths = ["test_images/frame216.jpg"]
    if args.images:
        for img in args.images.split(","):
            if exists("test_images/" + img):
                imgPaths.append("test_images/" + img)
            else:
                print("There is no file with name " + img + " in the test_images directory.")
                return
        imgPaths.pop(0)
    
    # Process each image and display a figure for comparison
    for imgPath in imgPaths:
        # Print which image is being processed
        print("Running " + imgPath)

        # Create subplots a fixed 3 columns and necessary number of rows to display all images
        cols = 3 if len(functions) >=3 else len(functions)
        rows = math.ceil(len(functions) / cols)
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6), squeeze=False)

        # Loop through all preprocessing functions that need to be executed
        # Apply each function to the image, time its executin, and show it on the matplotlib
        for i in range(len(functions)):

            processingStartTime = time.time()
            processedImg = functions[i](imgPath)
            processingTime = time.time() - processingStartTime
            print(f'{functions[i].__name__:>20}' + " %.2f secs" % processingTime)

            ax = axes.ravel()[i]
            ax.imshow(cv2.cvtColor(processedImg, cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_xlabel(functions[i].__name__)

        # Delete any subplots that do not contain an image
        for i in range((rows * cols) - len(functions)):
            fig.delaxes(axes[rows - 1][cols - 1 - i])

        # Set the figure title to the filename of the image
        imgPathSplit = imgPath.split("/")
        fig.suptitle(imgPathSplit[len(imgPathSplit) - 1])

        # Show the plot with all the images
        fig.tight_layout()

        # Newline for spacing
        print()

        # Show each figure
        # User will have to close currently displayed figure to see figure of next image
        plt.show()
    

def no_preprocessing(image_path):
    return cv2.imread(image_path)


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


    # Edit Image
    image = image.astype(np.int64)  # To avoid overflow issues
    image[:, :, 0] = (255 * (image[:, :, 0] - blue_min)) / (blue_max - blue_min)
    image[:, :, 1] = (255 * (image[:, :, 1] - green_min)) / (green_max - green_min)
    image[:, :, 2] = (255 * (image[:, :, 2] - red_min)) / (red_max - red_min)
    image = np.clip(image, 0, 255)  # Formula will cause some values to go slightly outside of range
    image = image.astype(np.uint8)

    return image


# Works, fast
def clahe(image_path):
    img = cv2.imread(image_path)
    return CLAHE.RecoverCLAHE(img)


# Works, but takes 1-2 minutes per image
def dcp(image_path):
    img = cv2.imread(image_path)
    transmission, sceneRadiance = DCP.getRecoverScene(img)

    # Can return transmission or sceneRadiance
    # return np.uint8(transmission * 255)
    return sceneRadiance


# Works, but takes 1-2 minutes per image
def gbrc(image_path):
    img = cv2.imread(image_path)
    img = (img - img.min()) / (img.max() - img.min()) * 255

    blockSize = 9

    largestDiff = GBRC.determineDepth(img, blockSize)
    AtomsphericLight, AtomsphericLightGB, AtomsphericLightRGB = GBRC.getAtomsphericLight(largestDiff, img)
    transmission = GBRC.getTransmission(img, AtomsphericLightRGB, blockSize)
    transmission = GBRC.refinedtransmission(transmission, img)
    # return np.uint8(transmission[:, :, 0] * 255)

    sceneRadiance_GB = GBRC.sceneRadianceGB(img, transmission, AtomsphericLightRGB)
    sceneRadiance = GBRC.sceneradiance(img, sceneRadiance_GB)
    S_x = GBRC.AdaptiveExposureMap(img, sceneRadiance, Lambda=0.3, blockSize=blockSize)
    sceneRadiance = GBRC.AdaptiveSceneRadiance(sceneRadiance, S_x)

    return sceneRadiance


# Works, fast
def gc(image_path):
    img = cv2.imread(image_path)
    return GC.RecoverGC(img)

  
if __name__ == '__main__':
    main()
