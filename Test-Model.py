#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:12:16 2019

This python file is used only for testing puproses to show the following:
1) An example of our model's predictions on a positive case (brain hemorrhaging)
2) An example of our model's predictions on a negative cases (no brain hemorrhaging)
3) Our model successfully uses a data generator to load in x and y training data to the fit_generator function
4) Our model successfully uses a data generator to load in x and y validation data to the evaluate_generator function


@author: georgebarker, andrezeromski, juliolopez, zachfrancis
"""

import os
import pickle
import keras
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

def create_partition_and_labels(data_location):
    # Constants
    size_of_train_set = .8 # Percent size of training set
    # Make DF from .csv
    df = pd.read_csv(data_location+"/stage_2_train.csv")
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.drop_duplicates(subset=None, keep='first', inplace=False)    
    df = df.reset_index(drop=True)    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    print("DF extracted.")
    # Make labels dictionary
    labels = {}
    for i in df.index.values:
        labels[i] = df.loc[i].values.tolist()
    print("Labels created.")
    # Make partition dictionary
    partition_train = df.index.values
    partition_len = len(partition_train)
    partition_validation = partition_train[int(partition_len*size_of_train_set):]
    partition_train = partition_train[:int(partition_len*size_of_train_set)]
    np.random.shuffle(partition_train)
    partition = {}
    partition['train']=partition_train
    partition['validation']=partition_validation
    print("Partition created.")
    return (partition, labels)

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
    try:
        brain_window = window_image(dc, 70, 40)
        vascular_window = window_image(dc, 90, 68)
        subdural_window = window_image(dc, 200, 80)
        image = np.zeros((dc.pixel_array.shape[0], dc.pixel_array.shape[1], 3))
        image[:, :, 0] = brain_window
        image[:, :, 1] = vascular_window
        image[:, :, 2] = subdural_window
        if (dc.pixel_array.shape[0] != 512) or (dc.pixel_array.shape[1] != 512):
            # padding to 512,512,3
            image = np.zeros((512,512,3))
            image[:image.shape[0],:image.shape[1],:image.shape[2]] = image
        return image
    except ValueError:
        print("A ValueError exception occurred: returned a (512,512,3) array of zeros")
        return np.zeros((512, 512, 3))

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_x = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]
        batch_y = []
        for item in batch_x:
            batch_y.append(self.labels[item])
        x, y = np.array([composite_image(pydicom.dcmread('data/' + ID + '.dcm')) for ID in batch_x]), np.array(batch_y)
        return x, y

# Datasets
# (partition,labels) = create_partition_and_labels(data_location) # IDs, Labels

# Loading labels
pickle_in = open("data/dataLabels","rb")
(partition,labels) = pickle.load(pickle_in)
print("Partition and labels loaded.")

batch_size=8

# Generators
training_generator = DataGenerator(partition['train'], labels, batch_size)
validation_generator = DataGenerator(partition['validation'], labels, batch_size)
print("Training and validation generators created.")

# Load and compile new model
with open('model/model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('model/model_weights.h5')
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy',tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.AUC()])

# Prints out positive cases of bleeding
positive_batch_x = []
positive_batch_y = []
i=0
for ID in partition['validation']:
    if labels[ID][0] == 1:
        positive_batch_x.append(ID)
        positive_batch_y.append(labels[ID])
        i+=1
        if i == 5:
            break

prediction = model.predict(np.array([composite_image(pydicom.dcmread('data/' + ID + '.dcm')) for ID in positive_batch_x]))
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print("\nTesting predictions on positive cases of brain hemorrhaging (press enter to continue):")
for y_actual, y_predict in zip(positive_batch_y,prediction):
    print("For y_actual: "+str(y_actual)+', our model made y_predict: '+str(y_predict))

# Prints out negative cases of bleeding
positive_batch_x = []
positive_batch_y = []
i=0
for ID in partition['validation']:
    if labels[ID][0] == 0:
        positive_batch_x.append(ID)
        positive_batch_y.append(labels[ID])
        i+=1
        if i == 5:
            break

prediction = model.predict(np.array([composite_image(pydicom.dcmread('data/' + ID + '.dcm')) for ID in positive_batch_x]))
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print("\nTesting predictions on negative cases of brain hemorrhaging (press enter to continue):")
for y_actual, y_predict in zip(positive_batch_y,prediction):
    print("For y_actual: "+str(y_actual)+', our model made y_predict: '+str(y_predict))

# Training model on sample training set
print("\nTraining model on sample training set:")
model.fit_generator(training_generator, epochs = 3, verbose = 2)

# Evaluating model on sample validation set
print("\nEvaluating model on sample testing set:")
prediction = model.evaluate_generator(generator = validation_generator, verbose = 1)
print()
for metric, score in zip(model.metrics_names, prediction):
    print(str(metric)+": "+str(score))



