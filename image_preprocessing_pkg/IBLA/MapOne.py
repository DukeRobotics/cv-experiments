import numpy as np

from image_preprocessing_pkg.IBLA.getOneChannelMax import getMaxChannel


def max_R(img,blockSize):
    img = img[:,:,2]
    R_map  = getMaxChannel(img, blockSize)
    return R_map

