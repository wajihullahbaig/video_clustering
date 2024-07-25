# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:46:23 2024

@author: Acer
"""

import numpy as np
from itertools import pairwise
from kneed import KneeLocator
import matplotlib.pyplot as plt


def remap_range(input_array:np.ndarray , new_max,new_min):
    old_min = min(input_array)
    old_max = max(input_array)
    old_range = (old_max - old_min)
    if (old_range == 0):
        new_array = new_min
    else:
        new_range = (new_max - new_min)  
        new_array = (((input_array - old_min) * new_range) / old_range) + new_min
    return new_array

def normalize_min_max(input_array:np.ndarray):
    return (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array))

def normalize_standard_deviation(input_array:np.ndarray):
    return (input_array - input_array.mean()) / (input_array.std())

def find_kneed(arr)->int:
    if len(arr) < 2:
        return None  # Not enough elements to compare
    x = range(len(arr))
    y = arr
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    return kn.knee
    
def visualize_kneed(k,arr):
    x = range(len(arr))
    y = arr
    plt.xlabel('number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.plot(x, y, 'bx-')
    plt.vlines(k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()