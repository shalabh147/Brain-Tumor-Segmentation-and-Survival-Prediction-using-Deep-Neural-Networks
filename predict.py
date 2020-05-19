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

from utils import f1_score,dice_coef,dice_coef_loss
def reverse_encode(a):
	return np.argmax(a,axis=-1)



model_to_predict1 = load_model('Models/survival_pred_240_240.h5',custom_objects={'dice_coef_loss':dice_coef_loss})
model_to_predict2 = load_model('Models/survival_pred_240_155_1.h5')
model_to_predict3 = load_model('Models/survival_pred_240_155_2.h5')
path = '../TestData'
all_images = os.listdir(path)
#print(len(all_images))
all_images.sort()

data = np.zeros((240,240,155,4))

for i in range(100,101):
  new_image = np.zeros((240,240,155,5))
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
  #print(image_data2.shape)  

    
  for slice_no in range(0,240):
    a = slice_no
    X = data[slice_no,:,:,:]
	X = X.reshape(1,240,155,4)
	Y_hat = model_to_predict3.predict(X)
	new_image[a,:,:,:] = Y_hat[0,:,:,:]

  for slice_no in range(0,155):
    a = slice_no
    X = data[:,:slice_no,:]
	X = X.reshape(1,240,240,4)
	Y_hat = model_to_predict1.predict(X)
	new_image[:,:,slice_no,:] += Y_hat[0,:,:,:]

  for slice_no in range(0,240):
    a = slice_no
    X = data[:,slice_no,:,:]
	X = X.reshape(1,240,155,4)
	Y_hat = model_to_predict2.predict(X)
	new_image[:,a,:,:] += Y_hat[0,:,:,:]
      

  new_image = new_image/3
  

  name = '../all_images/VSD.Seg_001.'+ image_id + '.mha'
  save(new_image,name)
