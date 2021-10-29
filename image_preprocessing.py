import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from image_preprocessing.DCP import main as DCP
from image_preprocessing.CLAHE import main as CLAHE
from image_preprocessing.GBdehazingRCorrection import main as GBRC
from image_preprocessing.GC import main as GC

# This file will contain all of the different image processing algorithms
# Each algorithm should have its own separate function

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

    #Edit Image
    image = image.astype(np.int64) #To avoid overflow issues
    image[:, :, 0] = (255*(image[:, :, 0] - blue_min))/(blue_max-blue_min)
    image[:, :, 1] = (255*(image[:, :, 1] - green_min))/(green_max-green_min)
    image[:, :, 2] = (255*(image[:, :, 2] - red_min))/(red_max-red_min)
    image = np.clip(image, 0, 255) #Formula will cause some values to go slightly outside of range
    image = image.astype(np.uint8)
    
    
    #Return Image
    print(type(image))
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
    #return np.uint8(transmission * 255)
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
        return np.uint8(transmission[:, :, 0] * 255)

        sceneRadiance_GB = GBRC.sceneRadianceGB(img, transmission, AtomsphericLightRGB)
        sceneRadiance = GBRC.sceneradiance(img, sceneRadiance_GB)
        S_x = GBRC.AdaptiveExposureMap(img, sceneRadiance, Lambda=0.3, blockSize=blockSize)
        sceneRadiance = GBRC.AdaptiveSceneRadiance(sceneRadiance, S_x)
        
        # return sceneRadiance

# Works, fast
def gc(image_path):
    img = cv2.imread(image_path)
    return GC.RecoverGC(img)

if __name__ == '__main__':
    # Test an image processing algorithm
    
    imgPaths = ["test_images/frame216.jpg"]
    # imgPaths = ["test_images/frame216.jpg", "test_images/frame220.jpg", "test_images/frame223.jpg", "test_images/frame230.jpg", "test_images/frame235.jpg", "test_images/frame242.jpg", "test_images/frame258.jpg", ]
    functions = [amine_rhone, clahe, dcp, gbrc, gc]
    imgs = []

    for imgPath in imgPaths:
        imgs.append(no_preprocessing(imgPath))
        #imgs.append(amine_rhone(imgPath))
        #imgs.append(clahe(imgPath))
        #imgs.append(dcp(imgPath))
        #imgs.append(gbrc(imgPath))
        imgs.append(gc(imgPath))

    before = True
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for img, ax in zip(imgs, axes.ravel()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="nearest")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_xlabel("Before" if before else "After")
        before = not before
    fig.tight_layout()
    plt.show()

