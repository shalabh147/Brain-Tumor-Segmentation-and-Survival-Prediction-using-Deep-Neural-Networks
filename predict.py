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
from medpy.io import save
import numpy as np
import cv2

def reverse_encode(a):
	return np.argmax(a,axis=-1)

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

model_to_predict = load_model('full_image_model.h5',custom_objects={'dice_coef_loss':dice_coef_loss})

path = '../HGG_LGG'
all_images = os.listdir(path)
#print(len(all_images))
all_images.sort()

data = np.zeros((1,240,240,155,4),dtype='uint8')
#data2 = np.zeros((240,240,155,1),dtype='uint8')
#Y = np.zeros((240,155,5),dtype='uint8')

for i in range(80,110):
  print(i)
  x = all_images[i]
  folder_path = path + '/' + x;
  modalities = os.listdir(folder_path)
  modalities.sort()
  #print(i)
  for j in range(len(modalities)-1):
    #print(modalities[j])
    if(j==0):
    	lst = modalities[j].split('.')
    	image_id = lst[-1]

    image_path = folder_path + '/' + modalities[j]
    image_and_lic = os.listdir(image_path)
    image_and_lic.sort()
    if(len(image_and_lic) > 1):
      actual_image_path = image_path + '/' + image_and_lic[1]
    else:
      actual_image_path = image_path + '/' + image_and_lic[0]
    data[0,:,:,:,j], image_header2 = load(actual_image_path);

  new_image = np.zeros((240,240,155),dtype='uint8')
    #mod_im = resize(image_data2[:,126,:],(128,128), mode = 'constant', preserve_range = True)
  for slice_no in range(0,240):
    #print(slice_no)
    data2 = model_to_predict.predict(data[:,:,slice_no,:,:])
    data2 = reverse_encode(data2[0,:,:,:])
    #print(data2.shape)
    new_image[:,slice_no,:] = data2

  name = '../all_images/VSD.Seg_001.'+ image_id + '.mha'
  save(new_image,name)
