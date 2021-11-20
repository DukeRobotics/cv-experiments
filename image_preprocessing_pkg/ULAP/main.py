import os

import datetime
import numpy as np
import cv2
import natsort

from image_preprocessing_pkg.ULAP.GuidedFilter import GuidedFilter
from image_preprocessing_pkg.ULAP.backgroundLight import BLEstimation
from image_preprocessing_pkg.ULAP.depthMapEstimation import depthMap
from image_preprocessing_pkg.ULAP.depthMin import minDepth
from image_preprocessing_pkg.ULAP.getRGBTransmission import getRGBTransmissionESt
from image_preprocessing_pkg.ULAP.global_Stretching import global_stretching
from image_preprocessing_pkg.ULAP.refinedTransmissionMap import refinedtransmissionMap

from image_preprocessing_pkg.ULAP.sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# starttime = datetime.datetime.now()

# # folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/ULAP"
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
#         gimfiltR = 50  # 引导滤波时半径的大小
#         eps = 10 ** -3  # 引导滤波时epsilon的值

#         DepthMap = depthMap(img)
#         DepthMap = global_stretching(DepthMap)
#         guided_filter = GuidedFilter(img, gimfiltR, eps)
#         refineDR = guided_filter.filter(DepthMap)
#         refineDR = np.clip(refineDR, 0,1)

#         cv2.imwrite('OutputImages/' + prefix + '_ULAPDepthMap.jpg', np.uint8(refineDR * 255))

#         AtomsphericLight = BLEstimation(img, DepthMap) * 255

#         d_0 = minDepth(img, AtomsphericLight)
#         d_f = 8 * (DepthMap + d_0)
#         transmissionB, transmissionG, transmissionR = getRGBTransmissionESt(d_f)

#         transmission = refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img)
#         sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)


#         cv2.imwrite('OutputImages/' + prefix + '_ULAP_TM.jpg', np.uint8(transmission[:, :, 2] * 255))


#         # print('AtomsphericLight',AtomsphericLight)

#         cv2.imwrite('OutputImages/' + prefix + '_ULAP.jpg', sceneRadiance)


# Endtime = datetime.datetime.now()
# Time = Endtime - starttime
# print('Time', Time)


