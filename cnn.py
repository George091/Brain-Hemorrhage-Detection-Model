#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:12:16 2019

@author: georgebarker, andrezeromski, juliolopez
"""

import os
import numpy as np
import pandas as pd
import keras
import pydicom
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import pickle

## GLOBAL CONSTANTS

data_location = "/Volumes/My Passport/RSNA dataset/rsna-intracranial-hemorrhage-detection"

def create_partition_and_labels(data_location):
    # Constants
    size_of_train_set = .7 # Percent size of training set
    
    df = pd.read_csv(data_location+"/stage_2_train.csv")
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.drop_duplicates(subset=None, keep='first', inplace=False)    
    df = df.reset_index(drop=True)    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    # Make labels dictionary
    labels = {}
    for i in df.index.values:
        labels[i] = df.loc[i].values.tolist()
    
    # Make partition dictionary
    partition_train = df.index.values
    np.random.shuffle(partition_train)
    partition_len = len(partition_train)
    partition_validation = partition_train[int(partition_len*size_of_train_set):]
    partition_train = partition_train[:int(partition_len*size_of_train_set)]
    partition = dict({'train': partition_train, 'validation': partition_validation})

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
    brain_window = window_image(dc, 70, 40)
    vascular_window = window_image(dc, 90, 68)
    subdural_window = window_image(dc, 200, 80)
    image = np.zeros((dc.pixel_array.shape[0], dc.pixel_array.shape[1], 3))
    image[:, :, 0] = brain_window
    image[:, :, 1] = vascular_window
    image[:, :, 2] = subdural_window
    return image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(512,512,1), n_channels=3,
                 n_classes=6, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = composite_image(pydicom.dcmread(data_location+'/stage_2_train/' + ID + '.dcm'))

            # Store class
            y[i] = self.labels[ID]

        return X, y#keras.utils.to_categorical(y, num_classes=self.n_classes)

def main():    
    # Datasets
#    (partition,labels) = create_partition_and_labels(data_location) # IDs, Labels
#
#    pickle_out = open("partitionAndLabelInput","wb")
#    pickle.dump((partition,labels), pickle_out)
#    pickle_out.close()

    pickle_in = open("partitionAndLabelInput","rb")
    (partition,labels) = pickle.load(pickle_in)
    
    # Parameters
    params = {'dim': (512,512,3),
              'batch_size': 64,
              'n_classes': 6,
              'n_channels': 3,
              'shuffle': True}
    
    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)
    
    # Design model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(512, 512, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)
    
if __name__ == "__main__": 
    main() 