import numpy as np
import pandas as pd
import keras
import pydicom
from keras.models import model_from_json
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3

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
    partition_set = df.index.values
    partition_len = len(partition_set)
    partition_validation = partition_set[int(partition_len*size_of_train_set):]
    partition_train = partition_set[:int(partition_len*size_of_train_set)]
    np.random.shuffle(partition_train)
    partition = {}
    partition['train']=partition_train
    partition['validation']=partition_validation
    print("Partition created.")
    return (partition, labels)

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
        x, y = np.array([composite_image(pydicom.dcmread(data_location+'/stage_2_train/' + ID + '.dcm')) for ID in batch_x]), np.array(batch_y)
        return x, y

## GLOBAL CONSTANTS
data_location = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'

# Datasets
(partition,labels) = create_partition_and_labels(data_location) # IDs, Labels

print("Partition and labels loaded.")

batch_size=32

# Generators
training_generator = DataGenerator(partition['train'], labels, batch_size)
validation_generator = DataGenerator(partition['validation'], labels, batch_size)

print("Training and validation generators created.")

# """
# # Design model
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=(512, 512, 3)))
# model.add(MaxPooling2D(pool_size=(5,5), strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(6, activation='sigmoid'))
# """

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(512,512,3))
for layer in base_model.layers[:-2]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x)
#x = Dense(256, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)

adam = keras.optimizers.Adam(learning_rate=0.00001,
 beta_1=0.9,
 beta_2=0.999,
 amsgrad=False)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy',tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives(), tf.keras.metrics.AUC()])

# Train model on dataset
model.fit_generator(training_generator, epochs = 1, verbose = 1)

# Save Weights and architecture
model.save_weights('model_weights-BC.h5')

# Save the model architecture
with open('model_architecture-BC.json', 'w') as f:
    f.write(model.to_json())

print("Model weights and architecture saved.")

# Evaluating model
prediction = model.evaluate_generator(generator = validation_generator, verbose = 2)
print("Model Evaluation: ")
for metric, score in zip(model.metrics_names, prediction):
    print(str(metric)+": "+str(score))
