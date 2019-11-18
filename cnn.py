#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:12:16 2019

@author: georgebarker, andrezeromski, juliolopez
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cupy as cp
import pydicom
import os

TRAIN_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"
TEST_IMG_PATH = "Test-Images"
BASE_PATH = '/'
TRAIN_DIR = 'stage_1_train_images/'

train = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")
train_images = os.listdir("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/")

train['filename'] = train['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".dcm")
train['type'] = train['ID'].apply(lambda st: st.split('_')[2])
train = train[['Label', 'filename', 'type']].drop_duplicates().pivot(index='filename', columns='type', values='Label').reset_index()

hem_types = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def load_random_images():
    image_names = [list(train[train[h_type] == 1].sample(1)['filename'])[0] for h_type in hem_types]
    image_names += list(train[train['any'] == 0].sample(5)['filename'])
    return [pydicom.read_file(os.path.join(TRAIN_IMG_PATH, img_name)) for img_name in image_names]


def view_images(images):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in range(0, height * width):
        image = images[im]
        i = im // width
        j = im % width
        axs[i,j].imshow(image, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        title = hem_types[im] if im < len(hem_types) else 'normal'
        axs[i,j].set_title(title)

    plt.show()
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def sigmoid_window(dcm, img, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):
    img = cp.array(np.array(img))
    _, _, intercept, slope = get_windowing(dcm)
    img = img * slope + intercept
    ue = cp.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + cp.power(np.e, -1.0 * z))
    img = (img - cp.min(img)) / (cp.max(img) - cp.min(img))
    return cp.asnumpy(img)

def map_to_gradient_sig(grey_img):
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4*grey_img - 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4*grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4*grey_img + 2, 0, 1.0) * (grey_img > 0.01) * (grey_img <= 1.0)
    return rainbow_img

def sigmoid_rainbow_bsb_window(img):
    brain_img = sigmoid_window(img, 40, 80)
    subdural_img = sigmoid_window(img, 80, 200)
    bone_img = sigmoid_window(img, 600, 2000)
    combo = (brain_img*0.35 + subdural_img*0.5 + bone_img*0.15)
    combo_norm = (combo - np.min(combo)) / (np.max(combo) - np.min(combo))
    return map_to_gradient_sig(combo_norm)

view_images([sigmoid_rainbow_bsb_window(img) for img in imgs])