import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse, math, time
from os.path import exists
from os import walk

from image_preprocessing_pkg.CLAHE                  import main as CLAHE
from image_preprocessing_pkg.DCP                    import main as DCP
from image_preprocessing_pkg.GBdehazingRCorrection  import main as GBRC
from image_preprocessing_pkg.GC                     import main as GC
from image_preprocessing_pkg.HE                     import main as HE
from image_preprocessing_pkg.IBLA                   import main as IBLA
from image_preprocessing_pkg.ICM                    import main as ICM
from image_preprocessing_pkg.LowComplexityDCP       import main as LCDCP
from image_preprocessing_pkg.MIP                    import main as MIP
from image_preprocessing_pkg.NewOpticalModel        import main as NOM
from image_preprocessing_pkg.RayleighDistribution   import main as RD
from image_preprocessing_pkg.RGHS                   import main as RGHS
from image_preprocessing_pkg.RoWS                   import main as RWS
from image_preprocessing_pkg.UCM                    import main as UCM
from image_preprocessing_pkg.UDCP                   import main as UDCP
from image_preprocessing_pkg.ULAP                   import main as ULAP

# This file will contain all of the different image processing algorithms
# Each algorithm should have its own separate function

def main():
    # Use ArgumentParser to get functions and image optional arguments
    parser = argparse.ArgumentParser(description='Image Preprocessing')
    parser.add_argument('--function-timings', action='store_true', help='An optional argument to print how long each image preprocessing function takes on average. Say "yes" or "no" (case-insensitive). If not specified, no timings will be printed.')
    parser.add_argument('--functions', type=str, help='An optional argument to specify which image preprocessing functions to run (comma-seperated). If not specified, only no_preprocessing will be run. If "all" is specified, all functions (fast and slow) will be run. If "all-fast" is specified, all functions that run in a short amount of time (<1 second) are run. If "all-slow" is specified, all functions that run in a long amount of time are run. Ex: "--functions amine_rhone,clahe,gc" NOTE: no_preprocessing is always run, regardless of the specified functions.')
    parser.add_argument('--images', type=str, help='An optional argument to specify which images in the test_images folder should be used (comma-seperated). If not specified, frame216.jpg will be used. If "all" is specified, all the test_images folder will be used. images in Ex: "--images frame220.jpg,frame223.jpg,frame230.jpg". ')
    parser.add_argument('--view-images-together', action='store_true', help="An optional argument to view the processing results of all images over all functions in one figure. This is useful for comparing the results of functions across images. If not specified, processing results will be shown in a seperate figure for each image.")
    args = parser.parse_args()

    # If the function_timings flag was true, print timings of all functions and don't do anything else
    if args.function_timings:
        # The names of all functions
        functionNames =          ["no_preprocessing", "amine_rhone",  "clahe",        "dcp",       "gbrc",      "gc",           "he",           "ibla",      "icm",        "lcdcp",       "mip",         "nom",          "rd",        "rghs",       "rws",       "ucm",      "udcp",      "ulap"       ]
        
        # The timngs of all functions in natural language
        functionTimingsNatural = ["0.05 seconds",     "0.20 seconds", "0.07 seconds", "2 minutes", "2 minutes", "0.35 seconds", "0.05 seconds", "5 minutes", "30 seconds", "1.5 minutes", "2.5 minutes", "4.25 minutes", "2 minutes", "45 seconds", "2 minutes", "1 minute", "2 minutes", "15 seconds" ]
        
        # The timings of all functions in seconds
        functionTimingsSeconds = [ 0.05,               0.20,           0.07,           120,         120,         0.35,           0.05,           300,         30,           90,            150,           255,            120,         45,           120,         60,         120,         15          ]

        # Print timings in natural language
        print("\nFunction Timings (natural)")
        for i in range(len(functionNames)):
            print("%20s %12s" % (functionNames[i], functionTimingsNatural[i]))
        
        # Print timings in seconds
        print("\nFunction Timings (seconds)")
        for i in range(len(functionNames)):
            print("%20s %6.2f" % (functionNames[i], functionTimingsSeconds[i]))

        # Print timings in seconds, sorted in ascending order
        print("\nFunction Timings (seconds, sorted)")
        functionTimingsSecondsSorted, functionNamesSorted = [list(tuple) for tuple in zip(*sorted(zip(functionTimingsSeconds, functionNames)))]
        for i in range(len(functionNamesSorted)):
            print("%20s %6.2f" % (functionNamesSorted[i], functionTimingsSecondsSorted[i]))

        return

    # Get list of functions
    possibles = globals().copy()
    possibles.update(locals())

    # If the functions argument was specified, check if the function exists and add it to the functions list
    # If one of the functions does not exist, let user know via command line and stop execution
    # If a special keyword was specified for the functions argument (all, all-fast, all-slow), add those functions to the functions list
    # If the functions argument was not specified, functions list only contains no_preprocessing as the default
    
    all      = [amine_rhone, clahe, dcp, gbrc, gc, he, ibla, icm, lcdcp, mip, nom, rd, rghs, rws, ucm, udcp, ulap]
    allFast  = [amine_rhone, clahe,            gc, he                                                            ]
    allSlow  = [                    dcp, gbrc,         ibla, icm, lcdcp, mip, nom, rd, rghs, rws, ucm, udcp, ulap]
    
    functionsStr = args.functions
    functions = [no_preprocessing]
    if functionsStr:
        if functionsStr == "all":
            functions.extend(all)
        elif functionsStr == "all-fast":
            functions.extend(allFast)
        elif functionsStr == "all-slow":
            functions.extend(allSlow)
        else:
            for functionName in functionsStr.split(","):
                function = possibles.get(functionName)
                if function:
                    functions.append(function)
                else:
                    print("There is no image preprocessing function with name " + functionName + ".")
                    return
    
    # If the images argument was specified, check if each image exists and add it to imgPaths
    # If one of the images does not exist, let the user know via command line and stop execution
    # If the all special keyword was specified, get all images in the test_images folder, sorted alphabetically, and all all of them to imgPaths
    # If the images argument was not specified, imgPaths contains frame216.jpg as the default
    imgPaths = []
    if args.images:
        
        if args.images == "all":
            fileNames = next(walk("test_images"), (None, None, []))[2]
            fileNames.sort()
            for i in range(len(fileNames)):
                fileNames[i] = "test_images/" + fileNames[i]
            imgPaths.extend(fileNames)
        
        else:
            for img in args.images.split(","):
                if exists("test_images/" + img):
                    imgPaths.append("test_images/" + img)
                else:
                    print("There is no file with name " + img + " in the test_images directory.")
                    return
    else:
        imgPaths = ["test_images/frame216.jpg"] 
    
    
    # List to hold the sum of the execution times of each function over all images
    functionTimingSums = [0] * len(functions)
    
    # If the user wants to view the all images together (for cross-image cross-function comparison), display them in one figure
    # Else, display the processed versions of each image in seperate figures
    if args.view_images_together:
        # Create subplots such that each column corresponds to one preprocessing function and each row corresponds to one image
        cols = len(functions)
        rows = len(imgPaths)
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6), squeeze=False)

        # List of processed images
        processedImgs = []

        # Process each image
        for imgPath in imgPaths:

            # Print which image is being processed
            print("Running " + imgPath)

            # Loop through all preprocessing functions that need to be executed
            # Apply each function to the image and print how long it takes to execute
            for i in range(len(functions)):
                processingStartTime = time.time()
                processedImg = functions[i](imgPath)
                processingTime = time.time() - processingStartTime
                print(f'{functions[i].__name__:>20}' + " %6.2f secs" % processingTime)
                processedImgs.append(processedImg)
                functionTimingSums[i] += processingTime 

            # Newline for spacing
            print()

        # Add each processed image to the figure
        for i in range(len(processedImgs)):
            ax = axes.ravel()[i]
            ax.imshow(cv2.cvtColor(processedImgs[i], cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            # Label image with the image file name and processing function name
            imgPathSplit = imgPaths[(int) (i / len(functions))].split("/")
            ax.set_xlabel(imgPathSplit[len(imgPathSplit) - 1] + " | " + functions[i % len(functions)].__name__)

        # Set title and layout of figure
        fig.suptitle("Image Preprocessing Comparison")
        fig.tight_layout()

        # Show the figure with all processed images
        plt.show()
    
    else:
        # Process each image and display a figure for comparison
        for imgPath in imgPaths:
            # Print which image is being processed
            print("Running " + imgPath)

            # Create subplots a fixed 3 columns and necessary number of rows to display all images
            cols = 3 if len(functions) >= 3 else len(functions)
            rows = math.ceil(len(functions) / cols)
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6), squeeze=False)

            # Loop through all preprocessing functions that need to be executed
            # Apply each function to the image, print how long it takes to execute, and show it on the matplotlib
            for i in range(len(functions)):

                processingStartTime = time.time()
                processedImg = functions[i](imgPath)
                processingTime = time.time() - processingStartTime
                print(f'{functions[i].__name__:>20}' + " %6.2f secs" % processingTime)
                functionTimingSums[i] += processingTime 

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
    
    # Print the average runtime of each of the function over all the images
    if(len(functions) > 1):
        print("Average runtimes")
        for i in range(len(functions)):
            print(f'{functions[i].__name__:>20}' + " %6.2f secs" % (functionTimingSums[i] / len(imgPaths)))
    print()

    
# Works, fast (0.05 seconds)
def no_preprocessing(image_path):
    return cv2.imread(image_path)


# Credit to this research paper: http://www.lsis.org/rov3d/article/art_AmineRhone2012.html
# Works, fast (0.20 seconds)
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


# Works, fast (0.07 seconds)
def clahe(image_path):
    img = cv2.imread(image_path)
    return CLAHE.RecoverCLAHE(img)


# Works, but takes 2 minutes per image
def dcp(image_path):
    img = cv2.imread(image_path)
    transmission, sceneRadiance = DCP.getRecoverScene(img)

    # Can return transmission or sceneRadiance
    # return np.uint8(transmission * 255)
    return sceneRadiance


# Works, but takes 2 minutes per image
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


# Works, fast (0.35 seconds)
def gc(image_path):
    img = cv2.imread(image_path)
    return GC.RecoverGC(img)

# Works, fast (0.05 seconds)
def he(image_path):
    img = cv2.imread(image_path)
    return HE.RecoverHE(img) 

# Works, but takes 10 minutes per image
def ibla(image_path):
        img = cv2.imread(image_path)

        blockSize = 9
        n = 5
        RGB_Darkchannel = IBLA.getRGB_Darkchannel(img, blockSize)
        BlurrnessMap = IBLA.blurrnessMap(img, blockSize, n)
        AtomsphericLightOne = IBLA.getAtomsphericLightDCP_Bright(RGB_Darkchannel, img, percent=0.001)
        AtomsphericLightTwo = IBLA.getAtomsphericLightLv(img)
        AtomsphericLightThree = IBLA.getAtomsphericLightLb(img, blockSize, n)
        AtomsphericLight = IBLA.ThreeAtomsphericLightFusion(AtomsphericLightOne, AtomsphericLightTwo, AtomsphericLightThree, img)

        R_map = IBLA.max_R(img, blockSize)
        mip_map = IBLA.R_minus_GB(img, blockSize, R_map)
        bluriness_map = BlurrnessMap

        d_R = 1 - IBLA.StretchingFusion(R_map)
        d_D = 1 - IBLA.StretchingFusion(mip_map)
        d_B = 1 - IBLA.StretchingFusion(bluriness_map)

        d_n = IBLA.Scene_depth(d_R, d_D, d_B, img, AtomsphericLight)
        d_n_stretching = IBLA.global_stretching(d_n)
        d_0 = IBLA.closePoint(img, AtomsphericLight)
        d_f = 8  * (d_n +  d_0)

        transmissionR = IBLA.getTransmission(d_f)
        transmissionB, transmissionG = IBLA.getGBTransmissionESt(transmissionR, AtomsphericLight)
        transmissionB, transmissionG, transmissionR = IBLA.Refinedtransmission(transmissionB, transmissionG, transmissionR, img)

        sceneRadiance = IBLA.sceneRadianceRGB(img, transmissionB, transmissionG, transmissionR, AtomsphericLight)

        # Depth Map d_D
        # return np.uint8((d_D)*255)

        # Depth Map
        # return np.uint8((d_f/d_f.max())*255)

        # Transmission Map
        # return np.uint8(np.clip(transmissionR * 255, 0, 255))

        # Scene radiance
        return sceneRadiance

# Works, but takes 30 seconds per image
def icm(image_path):
    img = cv2.imread(image_path)
    img = ICM.stretching(img)
    sceneRadiance = ICM.sceneRadianceRGB(img)
    sceneRadiance = ICM.HSVStretching(sceneRadiance)
    sceneRadiance = ICM.sceneRadianceRGB(sceneRadiance)
    return sceneRadiance

# Works, but takes 1.5 minutes per image
def lcdcp(image_path):
    img = cv2.imread(image_path)

    blockSize = 9
    imgGray = LCDCP.getDarkChannel(img, blockSize)
    AtomsphericLight = LCDCP.getAtomsphericLight(imgGray, img, meanMode=True, percent=0.001)
    transmission = LCDCP.getTransmissionMap(img, AtomsphericLight, blockSize)
    sceneRadiance = LCDCP.SceneRadiance(img, AtomsphericLight, transmission)
    sceneRadiance = LCDCP.ColorContrastEnhancement(sceneRadiance)

    # Transmission map
    # return np.uint8(transmission * 255)

    # Scene radiance
    return sceneRadiance

# Works, but takes 2 minutes per image
def mip(image_path):
    img = cv2.imread(image_path)

    blockSize = 9
    largestDiff = MIP.DepthMap(img, blockSize)
    transmission = MIP.getTransmission(largestDiff)
    transmission = MIP.Refinedtransmission(transmission,img)
    AtomsphericLight = MIP.getAtomsphericLight(transmission, img)
    sceneRadiance = MIP.sceneRadianceRGB(img, transmission, AtomsphericLight)

    # Transmission map
    # return np.uint8(transmission * 255)

    # Scene radiance
    return sceneRadiance

# Works, but takes 5 minutes per image
def nom(image_path):
    img = cv2.imread(image_path)

    blockSize = 9
    largestDiff = NOM.determineDepth(img, blockSize)
    AtomsphericLight = NOM.getAtomsphericLight(largestDiff, img)
    sactterRate = NOM.ScatteringRateMap(img, AtomsphericLight, blockSize)
    transmissionGB = NOM.TransmissionGB(sactterRate)
    transmissionR = NOM.TransmissionR(transmissionGB, img, blockSize)
    transmissionGB, transmissionR = NOM.Refinedtransmission(transmissionGB, transmissionR, img)
    sceneRadiance = NOM.SceneRadiance(img, transmissionGB, transmissionR, sactterRate, AtomsphericLight)

    # Red Transmission map
    # return np.uint8(transmissionR * 255)

    # Green/Blue Transmission map
    # return np.uint8(transmissionGB * 255)

    # Scene radiance
    return sceneRadiance

# Works, but takes 2 minutes per image
def rd(image_path):
    img = cv2.imread(image_path)
    height = len(img)
    width = len(img[0])

    sceneRadiance = RD.RGB_equalisation(img, height, width)
    sceneRadiance = RD.stretching(sceneRadiance)
    sceneRadiance_Lower, sceneRadiance_Upper = RD.rayleighStretching(sceneRadiance, height, width)
    sceneRadiance = (np.float64(sceneRadiance_Lower) + np.float64(sceneRadiance_Upper)) / 2
    sceneRadiance = RD.HSVStretching(sceneRadiance)
    sceneRadiance = RD.sceneRadianceRGB(sceneRadiance)

    return sceneRadiance

# Works, but takes 30 seconds per image
def rghs(image_path):
    img = cv2.imread(image_path)

    sceneRadiance = RGHS.RGB_equalisation(img)
    sceneRadiance = RGHS.stretching(img)
    sceneRadiance = RGHS.LABStretching(img)
    sceneRadiance = sceneRadiance.astype(np.uint8)

    return sceneRadiance

# Works, but takes 1.75 minutes per image
def rws(image_path):
    img = cv2.imread(image_path)
    blockSize = 9
    
    RGB_Darkchannel = RWS.getDarkChannel(img, blockSize)
    AtomsphericLight = RWS.getAtomsphericLight(RGB_Darkchannel, img)
    transmission = RWS.getTransmission(img, AtomsphericLight, blockSize)
    transmission = RWS.Refinedtransmission(transmission, img)
    sceneRadiance = RWS.sceneRadianceRGB(img, transmission, AtomsphericLight)

    # Transmission map
    # return np.uint8(transmission * 255)

    # Scene radiance
    return sceneRadiance

# Works, but takes 1 minute per image
def ucm(image_path):
    img = cv2.imread(image_path)

    sceneRadiance = UCM.RGB_equalisation(img)
    sceneRadiance = UCM.stretching(sceneRadiance)

    sceneRadiance = UCM.HSVStretching(sceneRadiance)
    sceneRadiance = UCM.sceneRadianceRGB(sceneRadiance)

    return sceneRadiance

# Works, but takes 2 minutes per image
def udcp(image_path):
        img = cv2.imread(image_path)

        blockSize = 9
        GB_Darkchannel = UDCP.getDarkChannel(img, blockSize)
        AtomsphericLight = UDCP.getAtomsphericLight(GB_Darkchannel, img)
        transmission = UDCP.getTransmission(img, AtomsphericLight, blockSize)
        transmission = UDCP.Refinedtransmission(transmission, img)
        sceneRadiance = UDCP.sceneRadianceRGB(img, transmission, AtomsphericLight)

        #return np.uint8(transmission* 255)
        return sceneRadiance

# Works, but takes 20 seconds per image
def ulap(image_path):
    img = cv2.imread(image_path)

    gimfiltR = 50
    eps = 10 ** -3

    DepthMap = ULAP.depthMap(img)
    DepthMap = ULAP.global_stretching(DepthMap)
    guided_filter = ULAP.GuidedFilter(img, gimfiltR, eps)
    refineDR = guided_filter.filter(DepthMap)
    refineDR = np.clip(refineDR, 0,1)

    # Depth map
    # return np.uint8(refineDR * 255)

    AtomsphericLight = ULAP.BLEstimation(img, DepthMap) * 255

    d_0 = ULAP.minDepth(img, AtomsphericLight)
    d_f = 8 * (DepthMap + d_0)
    transmissionB, transmissionG, transmissionR = ULAP.getRGBTransmissionESt(d_f)

    transmission = ULAP.refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img)
    sceneRadiance = ULAP.sceneRadianceRGB(img, transmission, AtomsphericLight)

    # Transmission map
    # return np.uint8(transmission[:, :, 2] * 255)
    
    # Scene radiance
    return sceneRadiance
  
if __name__ == '__main__':
    main()
