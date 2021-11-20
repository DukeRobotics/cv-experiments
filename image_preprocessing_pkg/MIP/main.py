import os
import numpy as np
import cv2
import natsort

from image_preprocessing_pkg.MIP.BL import getAtomsphericLight
from image_preprocessing_pkg.MIP.EstimateDepth import DepthMap
from image_preprocessing_pkg.MIP.getRefinedTramsmission import Refinedtransmission
from image_preprocessing_pkg.MIP.TM import getTransmission
from image_preprocessing_pkg.MIP.sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# # folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/MIP"
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
#         img = cv2.imread(folder +'/InputImages/' + file)

#         blockSize = 9

#         largestDiff = DepthMap(img, blockSize)
#         transmission = getTransmission(largestDiff)
#         transmission = Refinedtransmission(transmission,img)
#         AtomsphericLight = getAtomsphericLight(transmission, img)
#         sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

#         cv2.imwrite('OutputImages/' + prefix + '_MIP_TM.jpg', np.uint8(transmission * 255))
#         cv2.imwrite('OutputImages/' + prefix + '_MIP.jpg', sceneRadiance)
