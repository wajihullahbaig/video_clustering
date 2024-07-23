# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:47:25 2024

@author: Acer
"""

from abc import ABC, abstractmethod
from numpy import ndarray
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from array_utils import remap_range

class Feature(ABC):
    @abstractmethod
    def compute(self, input_image:ndarray)-> ndarray:
        pass
    
class HOGFeature(Feature):
    win_size=(16,16)
    block_size=(8,8)
    block_stride = (4,4)
    cell_size=(4,4)
    nbins = 32
    hog = cv2.HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins)
    eps = np.finfo(float).eps
    def compute(self,input_image:ndarray)->ndarray:
        try:
            features = self.hog.compute(input_image)
            # remap range
            features = np.trunc(remap_range(features, 255.0, 0.0))
            (hist, _) = np.histogram(features.ravel(),bins=np.arange(0,257),density=True)
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() +self.eps)
            return hist
        except Exception as ex:
            print(ex)
            return None
        
class LBPFeature(Feature):
    radius = 8
    n_points = 4 * radius
    eps = np.finfo(float).eps
    def compute(self,input_image:ndarray) -> ndarray:
        try:
            lbp = local_binary_pattern(input_image, self.n_points,self.radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0,257),density=True)
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() +self.eps)
            return hist
        except Exception as ex:
            print(ex)
            return None
    
class VGG16Feature(Feature):

    def compute(self,input_image:ndarray) -> ndarray:
        try:
           pass
        except Exception as ex:
            print(ex)
            return None  