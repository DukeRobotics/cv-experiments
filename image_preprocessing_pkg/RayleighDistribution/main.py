import math
import os
import natsort
import numpy as np
import datetime
import cv2
from skimage.color import rgb2hsv


from image_preprocessing_pkg.RayleighDistribution.color_equalisation import RGB_equalisation
from image_preprocessing_pkg.RayleighDistribution.global_stretching_RGB import stretching
from image_preprocessing_pkg.RayleighDistribution.hsvStretching import HSVStretching

from image_preprocessing_pkg.RayleighDistribution.histogramDistributionLower import histogramStretching_Lower
from image_preprocessing_pkg.RayleighDistribution.histogramDistributionUpper import histogramStretching_Upper
from image_preprocessing_pkg.RayleighDistribution.rayleighDistribution import rayleighStretching
from image_preprocessing_pkg.RayleighDistribution.rayleighDistributionLower import rayleighStretching_Lower
from image_preprocessing_pkg.RayleighDistribution.rayleighDistributionUpper import rayleighStretching_Upper
from image_preprocessing_pkg.RayleighDistribution.sceneRadiance import sceneRadianceRGB

# e = np.e
# esp = 2.2204e-16
# np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# # folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/NonPhysical/RayleighDistribution"
# folder = "C:/Users/Administrator/Desktop/Databases/Dataset"

# path = folder + "/InputImages"
# files = os.listdir(path)
# files =  natsort.natsorted(files)

# for i in range(len(files)):
#     file = files[i]
#     filepath = path + "/" + file
#     prefix = file.split('.')[0]
#     if os.path.isfile(filepath):
#         print('********    file   ********',file)
#         # img = cv2.imread('InputImages/' + file)
#         img = cv2.imread(folder + '/InputImages/' + file)
#         prefix = file.split('.')[0]
#         height = len(img)
#         width = len(img[0])

#         sceneRadiance = RGB_equalisation(img, height, width)
#         # sceneRadiance = stretching(img)
#         sceneRadiance = stretching(sceneRadiance)
#         sceneRadiance_Lower, sceneRadiance_Upper = rayleighStretching(sceneRadiance, height, width)

#         sceneRadiance = (np.float64(sceneRadiance_Lower) + np.float64(sceneRadiance_Upper)) / 2

#         # cv2.imwrite('OutputImages/' + prefix + 'Lower0.jpg', sceneRadiance_Lower)
#         # cv2.imwrite('OutputImages/' + prefix + 'Upper0.jpg', sceneRadiance_Upper)

#         sceneRadiance = HSVStretching(sceneRadiance)
#         sceneRadiance = sceneRadianceRGB(sceneRadiance)
#         cv2.imwrite('OutputImages/' + prefix + '_RayleighDistribution.jpg', sceneRadiance)



# endtime = datetime.datetime.now()
# time = endtime-starttime
# print('time',time)
