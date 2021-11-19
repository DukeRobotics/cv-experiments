import os
import numpy as np
import cv2
import natsort


from image_preprocessing_pkg.NewOpticalModel.DetermineDepth import determineDepth
from image_preprocessing_pkg.NewOpticalModel.getAtomsphericLight import getAtomsphericLight
from image_preprocessing_pkg.NewOpticalModel.getRefinedTramsmission import Refinedtransmission
from image_preprocessing_pkg.NewOpticalModel.getScatteringRate import ScatteringRateMap
from image_preprocessing_pkg.NewOpticalModel.getSceneRadiance import SceneRadiance
from image_preprocessing_pkg.NewOpticalModel.getTransmissionGB import TransmissionGB
from image_preprocessing_pkg.NewOpticalModel.getTransmissionR import TransmissionR

np.seterr(over='ignore')
if __name__ == '__main__':
    pass


# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/NewOpticalModel"
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
#         largestDiff = determineDepth(img, blockSize)
#         AtomsphericLight = getAtomsphericLight(largestDiff, img)
#         print('AtomsphericLight',AtomsphericLight)
#         sactterRate = ScatteringRateMap(img, AtomsphericLight, blockSize)
#         print('sactterRate',sactterRate)
#         transmissionGB = TransmissionGB(sactterRate)
#         transmissionR = TransmissionR(transmissionGB, img, blockSize)
#         transmissionGB, transmissionR = Refinedtransmission(transmissionGB, transmissionR, img)
#         sceneRadiance = SceneRadiance(img, transmissionGB, transmissionR, sactterRate, AtomsphericLight)

#         cv2.imwrite('OutputImages/' + prefix + '_NewOpticalModel_TM.jpg', np.uint8(transmissionR * 255))
#         # cv2.imwrite('OutputImages/' + prefix + '_NewOpticalModel_TM_GB.jpg', np.uint8(transmissionGB * 255))
#         cv2.imwrite('OutputImages/' + prefix + '_NewOpticalModel.jpg', sceneRadiance)



#         # cv2.imwrite('OutputImages/' + prefix + '_MIP_TM.jpg', np.uint8(transmission * 255))
#         # cv2.imwrite('OutputImages/' + prefix + '_MIP.jpg', sceneRadiance)