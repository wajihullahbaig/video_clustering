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
from array_utils import remap_range,normalize_min_max,standardize_zero_mean

class Feature(ABC):
    @abstractmethod
    def compute(self, input_image:ndarray)-> ndarray:
        pass
    
class HOGFeature(Feature):
    win_size=(16,16)
    block_size=(8,8)
    block_stride = (4,4)
    cell_size=(4,4)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins)
    eps = np.finfo(float).eps
    def compute(self,input_image:ndarray)->ndarray:
        try:
            features = self.hog.compute(input_image)
            (hist, _) = np.histogram(features,bins=256,range=(0,1),density=True)
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
            features = normalize_min_max(lbp.ravel())
            (hist, _) = np.histogram(features,bins=256,range = (0,1),density=True)
            return hist
        except Exception as ex:
            print(ex)
            return None
    
class VGG16Feature(Feature):

    def compute(self,input_image:ndarray) -> ndarray:
        try:
            raise NotImplementedError("VGG16 features!")
        except Exception as ex:
            print(ex)
            return None  