#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:05:00 2019

@author: georgebarker
"""

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def window_image(dc, window, level):
    intercept = dc[('0028','1052')].value
    slope = dc[('0028','1053')].value
    pixels = (dc.pixel_array*slope+intercept)
    pixel_min = level - window // 2
    pixel_max = level + window // 2
    pixels[pixels<pixel_min] = pixel_min
    pixels[pixels>pixel_max] = pixel_max
    pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))
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

path = os.getcwd() + "/Test-Images/"
imageList = os.listdir(path)

transformedImageList = []
i = 0
for filename in imageList:
    if i == 1 or i == 4:
        print()
    else:
        ds = pydicom.dcmread(path+filename)
        transformedImageList.append(ds)
    i += 1
    
    
width = 5
height = 2
fig, axs = plt.subplots(height, width, figsize=(15,5))
for widthPosition in range(0,width):
    image = transformedImageList[widthPosition].pixel_array
    axs[0,widthPosition].imshow(image, cmap=plt.cm.bone)
    axs[0,widthPosition].axis('off')
    title = 'Before Processing'
    axs[0,widthPosition].set_title(title)
    image = composite_image(transformedImageList[widthPosition])
    axs[1,widthPosition].imshow(image, cmap=plt.cm.bone)
    axs[1,widthPosition].axis('off')
    title = 'After Processing'
    axs[1,widthPosition].set_title(title)
plt.show()

    

