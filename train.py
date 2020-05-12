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
from models import Unet_with_slice

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

import os
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from medpy.io import load
import numpy as np

import cv2
from utils import f1_score,dice_coef_loss,dice_coef,one_hot_encode

model_train = load_model('Models/survival_pred_240_155_2.h5',custom_objects={'dice_coef_loss':dice_coef_loss, 'f1_score':f1_score})

# data preprocessing starts here
path = '../Brats17TrainingData/HGG'
all_images = os.listdir(path)
#print(len(all_images))
all_images.sort()
data = np.zeros((240,240,155,4))

for i in range(60,70):
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

    
  for slice_no in range(0,240):
    a = slice_no
    X = data[slice_no,:,:,:]

    Y = image_data2[slice_no,:,:]

    if(X.any()!=0 and Y.any()!=0 and len(np.unique(Y))==4):
      #print(slice_no)
      x_to.append(X)
      y_to.append(Y)    
      

  if len(x_to) <= 27:
  	continue;

  x_to = np.asarray(x_to)
  y_to = np.asarray(y_to)
  print(x_to.shape)
  print(y_to.shape)

  
  y_to[y_to==4] = 3         #since label 4 was missing in Brats dataset , changing all labels 4 to 3.
  hello = y_to.flatten()
  #print(hello[hello==3].shape)
  print("Number of classes",np.unique(hello))
  class_weights = class_weight.compute_class_weight('balanced',np.unique(hello),hello)
  
  #class_weights.insert(3,0)
  print("class_weights",class_weights)
  y_to = one_hot_encode(y_to)
  print(y_to.shape)


  

  model_train.fit(x=x_to, y=y_to, batch_size=9, epochs=4,class_weight = class_weights)



model_train.save('survival_pred_240_155_2.h5')