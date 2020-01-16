# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:07:50 2019

@author: Abhi
"""

import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog 


#import training dataset of vehicles/non-vehicles
import glob
cyc = glob.glob('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/HOG/train_dir/pos/*.png')
no_cyc = glob.glob('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/HOG/train_dir/neg/*.png')


height = 250
width = 250

len(cyc)
len(no_cyc)
image_color = cv2.imread('forhog3.jpg') 
plt.imshow(image_color)
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
image_resize = cv2.resize(image_color, (width, height))


plt.imshow(image_resize, cmap = 'gray')



#GET HOG FEATURES
features, hog_image = hog(image_gray, 
                          orientations = 11, 
                          pixels_per_cell = (16,16), 
                          cells_per_block = (2, 2), 
                          transform_sqrt = False, 
                          visualize = True, 
                          feature_vector = True)

features.shape 

hog_image.shape
plt.imshow(hog_image, cmap = 'gray')



#HOG FEATURE EXTRACTION AND TRAINING DATASET CREATION
cyc_hog_accum = []

for i in cyc:
    image_color = mpimg.imread(i)
    image_resize = cv2.resize(image_color, (width, height))
    image_gray  = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    cyc_hog_feature, cyc_hog_img = hog(image_gray, 
                                    orientations = 11, 
                                    pixels_per_cell = (16, 16), 
                                    cells_per_block = (2, 2), 
                                    transform_sqrt = False, 
                                    visualize = True, 
                                    feature_vector = True)
                
    cyc_hog_accum.append(cyc_hog_feature)


X_cyc = np.vstack(cyc_hog_accum).astype(np.float64)  
y_cyc = np.ones(len(X_cyc))
X_cyc.shape
y_cyc.shape
y_cyc

nocyc_hog_accum = []

for i in no_cyc:
    image_color = mpimg.imread(i)
    image_resize = cv2.resize(image_color, (width, height))
    image_gray  = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    nocyc_hog_feature, cyc_hog_img = hog(image_gray, 
                                    orientations = 11, 
                                    pixels_per_cell = (16, 16), 
                                    cells_per_block = (2, 2), 
                                    transform_sqrt = False, 
                                    visualize = True, 
                                    feature_vector = True)
                
    nocyc_hog_accum.append(nocyc_hog_feature)
    
    
X_nocyc = np.vstack(nocyc_hog_accum).astype(np.float64)  
y_nocyc = np.zeros(len(X_nocyc))
X_nocyc.shape
y_nocyc.shape


X_train = np.vstack((X_cyc, X_nocyc))
X_train.shape
y_train = np.hstack((y_cyc, y_nocyc))                       
y_train.shape









#HOG FEATURE EXTRACTION AND TESTING DATASET CREATION

cyc1 = glob.glob('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/HOG/test_dir/pos/*.png')
no_cyc1 = glob.glob('C:/Users/Abhi/Desktop/KITTI/DATA_DIR/HOG/test_dir/neg/*.png')

cyc_hog_accum1 = []

for i in cyc1:
    image_color = mpimg.imread(i)
    image_resize = cv2.resize(image_color, (width, height))
    image_gray  = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    cyc_hog_feature1, cyc_hog_img1 = hog(image_gray, 
                                    orientations = 11, 
                                    pixels_per_cell = (16, 16), 
                                    cells_per_block = (2, 2), 
                                    transform_sqrt = False, 
                                    visualize = True, 
                                    feature_vector = True)
                
    cyc_hog_accum1.append(cyc_hog_feature1)
    
    


X_cyc1 = np.vstack(cyc_hog_accum1).astype(np.float64)  
y_cyc1 = np.ones(len(X_cyc1))
X_cyc1.shape
y_cyc1.shape
y_cyc1

nocyc_hog_accum1 = []

for i in no_cyc1:
    image_color = mpimg.imread(i)
    image_resize = cv2.resize(image_color, (width, height))
    image_gray  = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    nocyc_hog_feature1, car_hog_img1 = hog(image_gray, 
                                    orientations = 11, 
                                    pixels_per_cell = (16, 16), 
                                    cells_per_block = (2, 2), 
                                    transform_sqrt = False, 
                                    visualize = True, 
                                    feature_vector = True)
                
    nocyc_hog_accum1.append(nocyc_hog_feature1)
    
    
X_nocyc1 = np.vstack(nocyc_hog_accum1).astype(np.float64)  
y_nocyc1 = np.zeros(len(X_nocyc1))
X_nocyc1.shape
y_nocyc1.shape




X_test = np.vstack((X_cyc1, X_nocyc1))
X_test.shape
y_test = np.hstack((y_cyc1, y_nocyc1))
y_test.shape




#SVM MODEL CLASSIFIER TRAINING



from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
svc_model = LinearSVC()
svc_model.fit(X_train,y_train)

y_predict = svc_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_predict)
acc = accuracy_score(y_test, y_predict, normalize=True, sample_weight=None)

sns.heatmap(cm, vmin = 10, vmax = 1500, annot=True, fmt="d")
print(classification_report(y_test, y_predict))
Model_prediction = svc_model.predict(X_test[0:50])
Model_prediction
Model_TrueLabel = y_test[0:50]
Model_TrueLabel


#IMPROVE THE MODEL Linear kernel
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear']} 

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
grid_linear = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid_linear.fit(X_train,y_train)
print(grid_linear.best_estimator_)
print(grid_linear.best_params_)

grid_linear_predictions = grid_linear.predict(X_test)
cm_grid_linear = confusion_matrix(y_test, grid_linear_predictions)
acc_linear = accuracy_score(y_test,  grid_linear_predictions, normalize=True, sample_weight=None)

#sns.heatmap(cm_grid, annot=True)
print(classification_report(y_test,grid_linear_predictions))






#IMPROVE THE MODEL kernel rbf
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


grid_rbf = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid_rbf.fit(X_train,y_train)
grid_rbf.best_params_
print(grid_rbf.best_estimator_)
print(grid_rbf.best_params_)

grid_rbf_predictions = grid_rbf.predict(X_test)
cm_grid = confusion_matrix(y_test, grid_rbf_predictions)
acc_rbf = accuracy_score(y_test, grid_rbf_predictions, normalize=True, sample_weight=None)

#sns.heatmap(cm_grid, annot=True)
#print(classification_report(y_test,grid_predictions))


#IMPROVE THE MODEL kernel poly
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],'degree': [1,2, 3, 4], 'kernel': ['poly']} 


grid_poly = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid_poly.fit(X_train,y_train)
grid_poly.best_params_
print(grid_poly.best_estimator_)
print(grid_poly.best_params_)

grid_poly_predictions = grid_poly.predict(X_test)
cm_grid_poly = confusion_matrix(y_test, grid_poly_predictions)
acc_poly = accuracy_score(y_test, grid_poly_predictions, normalize=True, sample_weight=None)

#IMPROVE THE MODEL kernel sigmoid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['sigmoid']} 


grid_sigmoid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid_sigmoid.fit(X_train,y_train)
grid_sigmoid.best_params_
print(grid_sigmoid.best_estimator_)
print(grid_sigmoid.best_params_)

grid_sigmoid_predictions = grid_sigmoid.predict(X_test)
cm_grid_sigmoid = confusion_matrix(y_test, grid_sigmoid_predictions)
acc_sigmoid = accuracy_score(y_test, grid_sigmoid_predictions, normalize=True, sample_weight=None)



#Creating a output for presentation

from prettytable import PrettyTable
t = PrettyTable(['Kernel', 'Accuracy'])
t.add_row(['Linear SVM', acc, ])
t.add_row(['Linear SVC', acc_linear])
t.add_row(['Rbf', acc_rbf])
t.add_row(['Poly', acc_poly])
t.add_row(['Sigmoid', acc_sigmoid])
print(t)


print(classification_report(y_test, y_predict))
print(classification_report(y_test, grid_linear_predictions))
print(classification_report(y_test, grid_rbf_predictions))
print(classification_report(y_test, grid_poly_predictions))
print(classification_report(y_test, grid_sigmoid_predictions))
#Saving the models


from sklearn.externals import joblib

filename = 'finalized_model.sav'
joblib.dump(svc_model, filename)

filename2 = 'finalized_model_optimized.sav'
joblib.dump(grid, filename2)

#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
svc_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)


#finding the time interval for best parameters
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
svc_model = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)



svc_model.fit(X_train,y_train)
import time
tstart = time.time()
cyc_hog_accum_time = []
image_color = mpimg.imread('006039.png')
image_resize = cv2.resize(image_color, (width, height))
image_gray  = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
cyc_hog_featuretime, cyc_hog_imgtime = hog(image_gray, 
                                    orientations = 11, 
                                    pixels_per_cell = (16, 16), 
                                    cells_per_block = (2, 2), 
                                    transform_sqrt = False, 
                                    visualize = True, 
                                    feature_vector = True)
cyc_hog_accum_time.append(cyc_hog_featuretime)
X_cyc_time = np.vstack(cyc_hog_accum_time).astype(np.float64)

result = svc_model.predict(X_cyc_time)
print(time.time()-tstart)
print(result)