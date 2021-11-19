import os
import numpy as np
import cv2
import natsort

from image_preprocessing_pkg.RoWS.RefinedTramsmission import Refinedtransmission
from image_preprocessing_pkg.RoWS.getAtomsphericLight import getAtomsphericLight
from image_preprocessing_pkg.RoWS.getRGBDarkChannel import getDarkChannel
from image_preprocessing_pkg.RoWS.getTM import getTransmission
from image_preprocessing_pkg.RoWS.sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/RoWS"
# # folder = "C:/Users/Administrator/Desktop/Databases/Dataset"
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
        
#         RGB_Darkchannel = getDarkChannel(img, blockSize)
#         AtomsphericLight = getAtomsphericLight(RGB_Darkchannel, img)
#         print('AtomsphericLight', AtomsphericLight)
#         transmission = getTransmission(img, AtomsphericLight, blockSize)
#         print('transmission',transmission)
#         print('np.mean(transmission)',np.mean(transmission))
#         transmission = Refinedtransmission(transmission, img)
#         sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)
#         # # print('AtomsphericLight',AtomsphericLight)
#         #
#         #
#         #
#         cv2.imwrite('OutputImages/' + prefix + '_RoWS_TM.jpg', np.uint8(transmission* 255))
#         cv2.imwrite('OutputImages/' + prefix + '_RoWS.jpg', sceneRadiance)


