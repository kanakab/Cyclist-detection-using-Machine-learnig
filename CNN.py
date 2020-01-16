# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:08:34 2019

@author: Abhi
"""

#import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
#from parser import load_data

from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
import gc; gc.enable() # memory is tight
from sklearn.externals import joblib

# Diretory for labels
train_image_dir_l = 'C:\\Users\\Abhi\\Desktop\\KITTI\\DATA_DIR\\training\\label_2'

# Diretory for images

train_image_dir = 'C:\\Users\\Abhi\\Desktop\\KITTI\\DATA_DIR\\training\\image_2'


#Creating a lists of images

images =  [(train_image_dir +'/' +f) for f in listdir(train_image_dir) if isfile(join(train_image_dir, f))]
masks = [(train_image_dir_l+f) for f in listdir(train_image_dir_l) if isfile(join(train_image_dir_l, f))]

l=0
for x in images:
    images[l] = x.replace(os.sep, '/')
    l+=1

'''for f in listdir(train_image_dir_l):
      fo = open("f.txt", "r")

      file_contents = fo.read()
      typedPassword = "Cyclist"
      for i in file_contents.split('\n'):
          if typedPassword == i:
              df_train['masks'] = 1
          else:
            df_train['masks'] = 0'''
#Creating lists of labels for Cyclist
labels=[]
for data in listdir(train_image_dir_l):
      obj_name = 'Cyclist'
      os.chdir('C:\\Users\\Abhi\\Desktop\\KITTI\\DATA_DIR\\training\\label_2')
      fo = open(data, 'r')
      file_contents = fo.read()
      if obj_name in file_contents:
            labels.append(1)
      else:
            labels.append(0)
#df = pd.DataFrame(np.column_stack([images, masks]), columns=['images', 'masks'])
masks = labels
'''df1 = df.sort_values(by='images')['images'].reset_index()
# df1 = df.sort_values(by='a')['a']
df2 = df.sort_values(by='masks')['masks'].reset_index()
# df2 = df.sort_values(by='b')['b']
df['images'] = df1['images']
df['masks'] = df2['masks']
del df1, df2'''
  

#Dividing training data for validation only    
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, shuffle = False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, shuffle = False)

             
#df_train, df_val = train_test_split(df, test_size=0.25, shuffle=False)





#Creating the directories for pos and neg

k=0
for i in y_train:
      fromDirectory = X_train[k]
      if i ==1:
            toDirectory = "C:/Users/Abhi/Desktop/KITTI/DATA_DIR/train_dir/pos"
      else:
            toDirectory = "C:/Users/Abhi/Desktop/KITTI/DATA_DIR/train_dir/neg"
      shutil.copy(fromDirectory, toDirectory)
      k+=1


k2=0
for i in y_val:
      fromDirectory = X_val[k2]
      if i ==1:
            toDirectory = "C:/Users/Abhi/Desktop/KITTI/DATA_DIR/val_dir/pos"
      else:
            toDirectory = "C:/Users/Abhi/Desktop/KITTI/DATA_DIR/val_dir/neg"
      shutil.copy(fromDirectory, toDirectory)
      k2 = k2+1


k1 = 0
for i in y_test:
      from1Directory = X_test[k1]
      if i ==1:
            to1Directory = "C:/Users/Abhi/Desktop/KITTI/DATA_DIR/test_dir/pos"
      else:
            to1Directory = "C:/Users/Abhi/Desktop/KITTI/DATA_DIR/test_dir/neg"
      shutil.copy(from1Directory, to1Directory)
      k1 = k1+1
      
      
'''pos=0
neg = 0

for i in labels:
      if i == 1:
            pos = pos + 1
      else:
            neg = neg + 1'''


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



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/train_dir',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/val_dir',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

history_model_1 =        classifier.fit_generator(training_set,
                                     samples_per_epoch = 4787,
                                     nb_epoch = 25,
                                     validation_data = test_set,
                                     nb_val_samples = 1197)


'''TRAINING_LOGS_FILE = "training_logs_3_1.csv"
MODEL_SUMMARY_FILE = "model_summary_3.txt"
TEST_FILE = "test_file_model_3_1.txt"
MODEL_FILE = "CNN_model_with_graph.h5"
classifier.save_weights(MODEL_FILE)'''


test_data_dir = 'C:/Users/Abhi/Desktop/KITTI/DATA_DIR/test_dir'

test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(64, 64), shuffle = True, batch_size= 20, class_mode="binary")
pred=classifier.predict_generator(test_generator, steps=len(test_generator), verbose=1)

# Get classes by np.round
cl = np.round(pred)
# Get filenames (set shuffle=false in generator is important)
filenames=test_generator.filenames
results=pd.DataFrame({"file":filenames,"pr":pred[:,0], "class":cl[:,0]})
y_predicted_for_real=[]
y_predicted_for_real = results['class']



#predicting the results
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_predicted_for_real)
acc = accuracy_score(y_test, y_predicted_for_real, normalize=True, sample_weight=None)
print(acc)
print(classification_report(y_test, y_predicted_for_real))

import time
from keras.preprocessing import image

test_image = image.load_img('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/test_dir/pos/005985.png', target_size = (64, 64)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
tstart = time.time()
#predict the result
result = classifier.predict(test_image)
print(time.time()-tstart)
print(result)
'''
predIdxs = classifier.predict_generator(test_generator)
predIdxs = np.argmax(predIdxs, axis=1)

for i in predIdxs:
      if predIdxs[i]>= 0.5:
            predIdxs[i] =1
      else:
            predIdxs[i]=0'''
            
            







classifier.save_weights('trained_CNN_model.h5')

from keras.models import load_model
classifier.load_weights('trained_CNN_model.h5')

'''X_test1 = X_test.reshape(-1,28, 28, 1)

y_pred = classifier.predict_classes(X_test)

model_path = "C:\\Users\\Abhi\\Desktop\\KITTI\\DATA_DIR\\cnn1.model"
if not os.path.isdir(os.path.split(model_path)[0]):
      os.makedirs(os.path.split(model_path)[0])'''


'''
import matplotlib.image as mpimg
TEST_SIZE = 20
open(TEST_FILE,"w")
probabilities = classifier.predict_generator(test_generator, TEST_SIZE)
for index, probability in enumerate(probabilities):
    image_path = test_data_dir + "/" +test_generator.filenames[index]
    img = mpimg.imread(image_path)
    with open(TEST_FILE,"a") as fh:
        fh.write(str(probability[0]) + " for: " + image_path + "\n")
    plt.imshow(img)
    if probability > 0.5:
        plt.title("%.2f" % (probability[0]*100) + "% dog")
    else:
        plt.title("%.2f" % ((1-probability[0])*100) + "% cat")
    plt.show()'''
from keras.preprocessing import image
imagespos = []
for img in os.listdir('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/test_dir/pos'):
    img = os.path.join('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/test_dir/pos', img)
    img = imagespos.load_img(img, target_size=(64, 64))
    img = imagespos.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    imagespos.append(img)

# stack up images list to pass for prediction
imagespos = np.vstack(images)
classes = model.predict_classes(images, batch_size=10)
