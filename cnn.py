#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:12:16 2019

@author: georgebarker, andrezeromski, juliolopez
"""

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def loadTrainData():
    path = os.getcwd() + "/Test-Images/"
    fileList = os.listdir(path)
    
    x_train = []
    y_train = []
    for file in fileList:    
        #read in and convert DICOM file
        
        #read CSV for y_train
        
    return (x_train, y_train)

def window_image(dc, window, level):
    intercept = dc[('0028','1052')].value
    slope = dc[('0028','1053')].value
    pixels = (dc.pixel_array*slope+intercept)
    pixel_min = level - window // 2
    pixel_max = level + window // 2
    pixels[pixels<pixel_min] = pixel_min
    pixels[pixels>pixel_max] = pixel_max
    pixels = (pixels - np.min(pixels)) / (pixel_max - pixel_min)
    return pixels

def composite_image(dc):
    brain_window = window_image(dc, 70, 40)
    vascular_window = window_image(dc, 90, 68)
    subdural_window = window_image(dc, 200, 80)
    image = np.zeros((dc.pixel_array.shape[0], dc.pixel_array.shape[1], 3))
    image[:, :, 0] = brain_window
    image[:, :, 1] = vascular_window
    image[:, :, 2] = subdural_window
    return image



transformedImageList = []
for filename in imageList:
    ds = pydicom.dcmread(path+filename)
    transformedImageList.append(composite_image(ds))

(x_train, y_train) = loadTrainData()
