# Cyclist-detection-using-Machine-learning:(Work In Progress)

Objective of this project is to compare various Machine learning algorithms/networks to detect the cyclist.

#Data Prep:

I am using the benchmark KITTI dataset for training and validation. The images in the training directory are classified as training set and validation data set. In the program untitled1.py I have preprocessed the data into pos and negative folders. To classify the images as Cyclists I have used the KITTI labels.
  Pos - Cyclist present in the image
  neg - For all other cases(Car, Van ...don't care)
  
  
 #Different classifiers:

1. Classification using CNN
   In this program I have built my own CNN model and trained the model.The file untitled1.py is the code for CNN and the trained model is    saved as cnn.model. The testing is in progress.
