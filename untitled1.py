# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:08:34 2019

@author: Abhi
"""

#import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
import gc; gc.enable() # memory is tight

# Diretory for labels
train_image_dir_l = 'C:\\Users\\Abhi\\Desktop\\KITTI\\DATA_DIR\\training\\label_2'

# Diretory for images

train_image_dir = 'C:\\Users\\Abhi\\Desktop\\KITTI\\DATA_DIR\\training\\image_2'




images =  [(train_image_dir+f) for f in listdir(train_image_dir) if isfile(join(train_image_dir, f))]
masks = [(train_image_dir_l+f) for f in listdir(train_image_dir_l) if isfile(join(train_image_dir_l, f))]



'''for f in listdir(train_image_dir_l):
      fo = open("f.txt", "r")

      file_contents = fo.read()
      typedPassword = "Cyclist"
      for i in file_contents.split('\n'):
          if typedPassword == i:
              df_train['masks'] = 1
          else:
            df_train['masks'] = 0'''

labels=[]
for data in listdir(train_image_dir_l):
      obj_name = 'Cyclist'
      os.chdir(r'C:\\Users\\Abhi\\Desktop\\KITTI\\DATA_DIR\\training\\label_2')
      fo = open(data, 'r')
      file_contents = fo.read()
      if obj_name in file_contents:
            labels.append(1)
      else:
            labels.append(0)
masks = labels      
df = pd.DataFrame(np.column_stack([images, masks]), columns=['images', 'masks'])

'''df1 = df.sort_values(by='images')['images'].reset_index()
# df1 = df.sort_values(by='a')['a']
df2 = df.sort_values(by='masks')['masks'].reset_index()
# df2 = df.sort_values(by='b')['b']
df['images'] = df1['images']
df['masks'] = df2['masks']
del df1, df2'''
      

                  
df_train, df_val = train_test_split(df, test_size=0.25, shuffle=False)

pos=0
neg = 0

for i in labels:
      if i == 1:
            pos = pos + 1
      else:
            neg = neg + 1


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images


from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


y_train = to_categorical(y_train, num_classes = None)
y_test = to_categorical(y_test, num_classes = None)

datagen = ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(df_train)

# fits the model on batches with real-time data augmentation:
classifier.fit_generator(datagen.flow(df_train, y_train, batch_size=32),
                    steps_per_epoch=len(df_train) / 32, epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(df_train, y_train, batch_size=32):
        classifier.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

































'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''


      


