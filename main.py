import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow as tf
import keras.backend as K

from keras import metrics
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
from sklearn.utils import class_weight
from models import Unet_with_slice,Unet_with_inception,Inception

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

import os
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from medpy.io import load
import numpy as np

import cv2

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_true, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def one_hot_encode(a):
  m = (np.arange(4) == a[...,None]).astype(int)
  return m

checkpoint = ModelCheckpoint('new/weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,min_delta = 0.01, patience = 3, mode = 'min')

callbacks_list = [checkpoint, earlystopping]

input_img = Input((240, 240, 4))
model = Inception(input_img,16,0.1,True)
model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=dice_coef_loss, metrics=[f1_score])
model.summary()





# data preprocessing starts here
path = '../Brats17TrainingData/HGG'
all_images = os.listdir(path)
#print(len(all_images))
all_images.sort()

#data = np.zeros((240,240,155,4),dtype='uint8')
#data2 = np.zeros((240,240,155,1),dtype='uint8')
Y = np.zeros((240,240))
X = np.zeros((240,240,4))
data = np.zeros((240,240,155,4))
#data2 = np.zeros((240,240,155,5))

for i in range(5):
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
      print("Entered ground truth")
    else:
      image_data, image_header = load(image_path);
      data[:,:,:,w] = image_data
      print("Entered modality")
      w = w+1
    
  print(data.shape)
  print(image_data2.shape)  

    
  for slice_no in range(0,155):
    a = slice_no
    X = data[:,:,slice_no,:]

    Y = image_data2[:,:,slice_no]

    if(X.any()!=0 and Y.any()!=0 and len(np.unique(Y)==4)):
      #print(slice_no)
      x_to.append(X)
      y_to.append(Y)    
      

  x_to = np.asarray(x_to)
  y_to = np.asarray(y_to)
  print(x_to.shape)
  print(y_to.shape)

  
  y_to[y_to==4] = 3
  hello = y_to.flatten()
  #print(hello[hello==3].shape)
  print("Number of classes",np.unique(hello))
  class_weights = class_weight.compute_class_weight('balanced',np.unique(hello),hello)
  
  #class_weights.insert(3,0)
  print("class_weights",class_weights)
  y_to = one_hot_encode(y_to)
  print(y_to.shape)


  

  model.fit(x=x_to, y=y_to, batch_size=10, epochs=5,class_weight = class_weights)



model.save('survival_pred_240_240.h5')

