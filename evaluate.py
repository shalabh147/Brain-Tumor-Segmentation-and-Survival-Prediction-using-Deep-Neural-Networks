import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow as tf
import keras.backend as K


from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Maximum
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D,MaxPooling3D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize

import os
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from medpy.io import load
import numpy as np

import cv2
from sklearn import metrics
#from main import one_hot_encode

from utils import one_hot_encode,dice_coef_loss,dice_coef,f1_score

model_predict = load_model('survival_pred_240_155_1.h5',custom_objects={'dice_coef_loss':dice_coef_loss, 'f1_score':f1_score})

# data preprocessing starts here
path = '../Brats17TrainingData/HGG'
all_images = os.listdir(path)
#print(len(all_images))
all_images.sort()

#Y = np.zeros((240,240))
#X = np.zeros((240,240,4))
data = np.zeros((240,240,155,4))
#data2 = np.zeros((240,240,155,5))

for i in range(100,101):
  print(i)
  x_to = []
  y_to = []
  x = all_images[i]
  folder_path = path + '/' + x;
  modalities = os.listdir(folder_path)
  modalities.sort()
  #data = []
  w = 0
  for j in range(len(modalities)-1):
    #print(modalities[j])
    
    image_path = folder_path + '/' + modalities[j]
    if(image_path[-7:-1] + image_path[-1] == 'seg.nii'):
      image_data2, image_header2 = load(image_path);
      print(np.unique(image_data2))
      print("Entered ground truth")
    else:
      image_data, image_header = load(image_path);
      
      data[:,:,:,w] = image_data
      print("Entered modality")
      w = w+1
    
  print(data.shape)
  print(image_data2.shape)  

    
  for slice_no in range(0,240):
    a = slice_no
    X = data[:,slice_no,:,:]

    Y = image_data2[:,slice_no,:]

    #if(X.any()!=0 and Y.any()!=0):
      #print(slice_no)
    x_to.append(X)
    y_to.append(Y)    
      

  #if len(x_to) <= 24:
  #  continue;

  x_to = np.asarray(x_to)
  y_to = np.asarray(y_to)
  print(x_to.shape)
  print(y_to.shape)

  
  y_to[y_to==4] = 3


  print("Number of classes",np.unique(y_to))

  pred = model_predict.predict(x=x_to)
  pred = np.around(pred)
  print(pred.shape)
  #pred = np.around(pred)

  #y_to = one_hot_encode(y_to)

  '''
  pred = pred.reshape(-1,5)
  #print(pred1[1000:1100][:])
  #print(y_to)

  pred1 = np.argmax(pred[:,1:5],axis=1)
  pred1 = pred1 + 1
  y2 = y_to.reshape(-1)
  #print(y2[100000:100100])

  print(y2.shape)
  print(pred1.shape)

  pred1[pred[:,0]>0.97] = 0
'''

  pred1 = np.argmax(pred.reshape(-1,4),axis = 1)
  y2 = y_to.reshape(-1)
  f1 = metrics.f1_score(y2,pred1,average='macro') 

  print(f1)
 # print(pred1[y2==2])
  print("Originally 0" , y2[y2==0].shape)
  print("Predicted 0" , pred1[pred1==0].shape)

  print("Originally 1" , y2[y2==1].shape)
  print("Predicted 1" , pred1[pred1==1].shape)

  print("Originally 2" , y2[y2==2].shape)
  print("Predicted 2" , pred1[pred1==2].shape)

  print("Originally 3" , y2[y2==3].shape)
  print("Predicted 3" , pred1[pred1==3].shape)

  #print("Originally 4" , y2[y2==4].shape)
  #print("Predicted 4" , pred1[pred1==4].shape)

 # f1 = metrics.f1_score(y2,pred1,average='macro')
  #print(f1)

