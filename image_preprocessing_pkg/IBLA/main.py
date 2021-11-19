import os
import datetime
import numpy as np
import cv2
import natsort
from image_preprocessing_pkg.IBLA.CloseDepth import closePoint
from image_preprocessing_pkg.IBLA.F_stretching import StretchingFusion
from image_preprocessing_pkg.IBLA.MapFusion import Scene_depth
from image_preprocessing_pkg.IBLA.MapOne import max_R
from image_preprocessing_pkg.IBLA.MapTwo import R_minus_GB
from image_preprocessing_pkg.IBLA.blurrinessMap import blurrnessMap
from image_preprocessing_pkg.IBLA.getAtomsphericLightFusion import ThreeAtomsphericLightFusion
from image_preprocessing_pkg.IBLA.getAtomsphericLightOne import getAtomsphericLightDCP_Bright
from image_preprocessing_pkg.IBLA.getAtomsphericLightThree import getAtomsphericLightLb
from image_preprocessing_pkg.IBLA.getAtomsphericLightTwo import getAtomsphericLightLv
from image_preprocessing_pkg.IBLA.getRGbDarkChannel import getRGB_Darkchannel
from image_preprocessing_pkg.IBLA.getRefinedTransmission import Refinedtransmission
from image_preprocessing_pkg.IBLA.getTransmissionGB import getGBTransmissionESt
from image_preprocessing_pkg.IBLA.getTransmissionR import getTransmission
from image_preprocessing_pkg.IBLA.global_Stretching import global_stretching
from image_preprocessing_pkg.IBLA.sceneRadiance import sceneRadianceRGB
from image_preprocessing_pkg.IBLA.sceneRadianceHE import RecoverHE

# np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# starttime = datetime.datetime.now()
# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/IBLA"
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
#         n = 5
#         RGB_Darkchannel = getRGB_Darkchannel(img, blockSize)
#         BlurrnessMap = blurrnessMap(img, blockSize, n)
#         AtomsphericLightOne = getAtomsphericLightDCP_Bright(RGB_Darkchannel, img, percent=0.001)
#         AtomsphericLightTwo = getAtomsphericLightLv(img)
#         AtomsphericLightThree = getAtomsphericLightLb(img, blockSize, n)
#         AtomsphericLight = ThreeAtomsphericLightFusion(AtomsphericLightOne, AtomsphericLightTwo, AtomsphericLightThree, img)
#         print('AtomsphericLight',AtomsphericLight)   # [b,g,r]


#         R_map = max_R(img, blockSize)
#         mip_map = R_minus_GB(img, blockSize, R_map)
#         bluriness_map = BlurrnessMap

#         d_R = 1 - StretchingFusion(R_map)
#         d_D = 1 - StretchingFusion(mip_map)
#         d_B = 1 - StretchingFusion(bluriness_map)

#         d_n = Scene_depth(d_R, d_D, d_B, img, AtomsphericLight)
#         d_n_stretching = global_stretching(d_n)
#         d_0 = closePoint(img, AtomsphericLight)
#         d_f = 8  * (d_n +  d_0)

#         # cv2.imwrite('OutputImages/' + prefix + '_IBLADepthMapd_D.jpg', np.uint8((d_D)*255))
#         # cv2.imwrite('OutputImages/' + prefix + '_IBLADepthMap.jpg', np.uint8((d_f/d_f.max())*255))

#         transmissionR = getTransmission(d_f)
#         transmissionB, transmissionG = getGBTransmissionESt(transmissionR, AtomsphericLight)
#         transmissionB, transmissionG, transmissionR = Refinedtransmission(transmissionB, transmissionG, transmissionR, img)

#         # cv2.imwrite('OutputImages/' + prefix + '_IBLA_TM.jpg', np.uint8(np.clip(transmissionR * 255, 0, 255)))

#         sceneRadiance = sceneRadianceRGB(img, transmissionB, transmissionG, transmissionR, AtomsphericLight)
#         # cv2.imwrite('OutputImages/' + prefix + '_IBLA.jpg', sceneRadiance)

#         # sceneRadiance =  RecoverHE(sceneRadiance)
#         # cv2.imwrite('OutputImages/' + prefix + '_IBLA_HE.jpg', sceneRadiance)
#         cv2.imwrite('OutputImages/' + prefix + '_IBLA.jpg', sceneRadiance)


# Endtime = datetime.datetime.now()
# Time = Endtime - starttime
# print('Time', Time)


