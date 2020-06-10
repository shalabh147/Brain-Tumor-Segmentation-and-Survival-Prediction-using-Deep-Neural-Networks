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

from utils import dice_coef_loss,dice_coef,one_hot_encode,standardize

#checkpoint = ModelCheckpoint('new/weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,min_delta = 0.01, patience = 3, mode = 'min')
#callbacks_list = [checkpoint, earlystopping]

input_img = Input((240, 155, 4))
model = Unet_with_inception(input_img,16,0.1,True)
learning_rate = 0.0000999
epochs = 5000
decay_rate = learning_rate / epochs
model.compile(optimizer=Adam(lr=learning_rate, decay = decay_rate), loss=dice_coef_loss, metrics=[dice_coef])
model.summary()





# data preprocessing starts here
path = '../Brats17TrainingData/HGG'
all_images = os.listdir(path)
#print(len(all_images))
all_images.sort()
data = np.zeros((240,240,155,4))
x_to = []
y_to = []

for i in range(25):
  print(i)

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
      image_data = standardize(image_data)
      data[:,:,:,w] = image_data
      print("Entered modality")
      w = w+1
    
  print(data.shape)
  print(image_data2.shape)  

    
  for slice_no in range(0,240):
    a = slice_no
    X = data[slice_no,:,:,:]

    Y = image_data2[slice_no,:,:]

    if(X.any()!=0 and Y.any()!=0 and len(np.unique(Y)) == 4):
      #print(slice_no)
      x_to.append(X)
      y_to.append(Y)    
      

  
  #hello = y_to.flatten()
  #print(hello[hello==3].shape)
  #print("Number of classes",np.unique(hello))
  #class_weights = class_weight.compute_class_weight('balanced',np.unique(hello),hello)
  
  #class_weights.insert(3,0)
  #print("class_weights",class_weights)
x_to = np.asarray(x_to)
y_to = np.asarray(y_to)
print(x_to.shape)
print(y_to.shape)

  
y_to[y_to==4] = 3         #since label 4 was missing in Brats dataset , changing all labels 4 to 3.
y_to = one_hot_encode(y_to)
print(y_to.shape)


  

model.fit(x=x_to, y=y_to, batch_size=20, epochs=50)



model.save('first_240_155v2.h5')

