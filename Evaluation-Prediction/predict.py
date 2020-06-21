import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#import tensorflow as tf
import keras.backend as K
import keras

from keras.models import Model, load_model
#from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Maximum
#from keras.layers.core import Lambda, RepeatVector, Reshape
#from keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose
#from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D,MaxPooling3D
#from keras.layers.merge import concatenate, add
#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#from skimage.io import imread, imshow, concatenate_images
#from skimage.transform import resize

import os
#from skimage.io import imread, imshow, concatenate_images
#from skimage.transform import resize
from medpy.io import load
from medpy.io import save
import numpy as np
#import time
#import sys
#sys.path.insert(1, '~/Brain_Segmentation/utils.py')
#import cv2

from utils import f1_score,dice_coef,dice_coef_loss,standardize,compute_class_sens_spec,get_sens_spec_df,one_hot_encode
def reverse_encode(a):
	return np.argmax(a,axis=-1)


import nibabel as nib

def dice_coef_2(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#model_to_predict1 = load_model('../first_240_155v5.h5',custom_objects={'dice_coef_loss':dice_coef_loss , 'dice_coef':dice_coef})
#model_to_predict2 = load_model('../second_240_155v5.h5',custom_objects={'dice_coef_loss':dice_coef_loss , 'dice_coef':dice_coef})
model_to_predict3 = load_model('../new_model_4 (1).h5',custom_objects={'dice_coef_2':dice_coef_2})
path = '../../Brats17TrainingData/HGG'
all_images = os.listdir(path)
#print(len(all_images))
all_images.sort()

data = np.zeros((240,240,155,4))

for i in range(106,108):
  new_image = np.zeros((240,240,155,4))
  print(i)
  x_to = []
  y_to = []
  x = all_images[i]
  folder_path = path + '/' + x;
  modalities = os.listdir(folder_path)
  modalities.sort()
  #data = []
  w = 0
  print(len(modalities))
  for j in range(len(modalities)-1):
    #print(modalities[j])
    
    image_path = folder_path + '/' + modalities[j]
    if(image_path[-7:-1] + image_path[-1] == 'seg.nii'):
      img = nib.load(image_path);
      image_data2 = img.get_data()
      image_data2 = np.asarray(image_data2)
    else:
      img = nib.load(image_path);
      image_data = img.get_data()
      image_data = np.asarray(image_data)
      image_data = standardize(image_data)
      data[:,:,:,w] = image_data
      print("Entered modality")
      print(w)
      w = w+1

  

  print(data.shape)
  new_image = np.zeros((155,240,240))
  image_data3 = np.zeros((155,240,240))
  for slice_no in range(0,155):
    a = slice_no
    X = data[:,:,slice_no,:]
    X = X.reshape(1,240,240,4)
    Y_hat = model_to_predict3.predict(X)
    #Y_hat = np.argmax(Y_hat,axis=-1)
    new_image[slice_no,:,:] = Y_hat[:,:,0]
    #new_image[new_image > 0.1] = 1
    #new_image[new_image <= 0.1] = 0
    #new_image[new_image==0] = 1
    print(slice_no)
    imgplot = plt.imshow(new_image[slice_no,:,:])
    plt.show(block=False)
    #time.sleep(1)
    plt.pause(0.1)
    plt.close()



    #new_image[slice_no,:,:,:] = data[:,:,slice_no,:]
    #image_data3[slice_no,:,:] = image_data2[:,:,slice_no]

  
  image_data2[image_data2==4] = 3
  #image_data2[image_data2==2] = 1
  #image_data2[image_data2==1] = 1
  image_data2 = image_data2.astype('float64')
  print(np.unique(image_data2[:,:,70]))
  print(image_data2.dtype)
  print(new_image.dtype)
  print(np.unique(new_image[70,:,:]))
  imgplot2 = plt.imshow(image_data2[:,:,70])
  plt.show(block=False)
  #time.sleep(1)
  plt.pause(4)
  #plt.close()

  imgplot = plt.imshow(new_image[70,:,:])
  plt.show(block=False)
  #time.sleep(1)
  plt.pause(4)
  plt.close()

  #image_data3[image_data3==4] = 3
  #image_data3 = one_hot_encode(image_data3)
  #image_data3 = image_data3.astype('float64')

  #print(model_to_predict3.evaluate(new_image,image_data3))
  #print(new_image.dtype)
  #print(image_data2.dtype)

  #print(K.eval(dice_coef_loss(new_image,image_data2)))
  #Y_hat = model_to_predict1.predict(data)
  #print(Y_hat.shape)
  #image_data2[image_data2==4] = 3
  #Y_hat[Y_hat > 0.6] = 1.0
  #Y_hat[Y_hat <= 0.6] = 0.0
  #
  #Y_hat = keras.utils.to_categorical(Y_hat,num_classes=4)
  #print(Y_hat[0,100,100])
  #print(len(Y_hat[:,:,:,0]==1))
  #print(len(Y_hat[:,:,:,1]==1))
  #print(len(Y_hat[:,:,:,2]==1))
  #print(len(Y_hat[:,:,:,3]==1))
  #image_data2 = keras.utils.to_categorical(image_data2, num_classes = 4)
  #
  #print(K.eval(dice_coef_loss(Y_hat,image_data2)))
  #print(get_sens_spec_df(Y_hat,image_data2))
  #print(model_to_predict1.evaluate(x=data,y=image_data2)) 
  #print(model_to_predict1.metrics_names)
  #Combining results from all 3 dimensions

'''
  for slice_no in range(0,240):
    a = slice_no
    X = data[slice_no,:,:,:]
    X = X.reshape(1,240,155,4)
    Y_hat = model_to_predict1.predict(X)
    new_image[a,:,:,:] = Y_hat[0,:,:,:]

  
'''


'''
  for slice_no in range(0,240):
    a = slice_no
    X = data[:,slice_no,:,:]
    X = X.reshape(1,240,155,4)
    Y_hat = model_to_predict2.predict(X)
    new_image[:,a,:,:] += Y_hat[0,:,:,:]
      

  image_data2[image_data2==4] = 3
  Y_hat = new_image/3.0
  Y_hat = np.argmax(Y_hat,axis=-1)
  Y_hat = keras.utils.to_categorical(Y_hat,num_classes=4)
  image_data2 = keras.utils.to_categorical(image_data2, num_classes = 4)
  print(K.eval(dice_coef_loss(Y_hat,image_data2)))
  print(get_sens_spec_df(Y_hat,image_data2))
  #print(new_image[100,100,100])
  #pred = pred.reshape(-1,5)
  #pred1 = np.argmax(new_image[:,:,:,1:],axis=3)
  #new_image = np.argmax(new_image,axis=3)
  #pred1[new_image[:,:,:,0] > 0.56] = 0         #average of probabilities from 3 directions
  #pred1 = pred1.astype('int64') 
  #image_data2 = image_data2.astype('int64')
'''  
'''
  for slice_no in range(0,155):
    print(slice_no)
    img = pred1[:,:,slice_no]
    imgplot = plt.imshow(img)
    plt.show(block=False)
    #time.sleep(1)
    plt.pause(0.1)
    plt.close()


  for slice_no in range(0,155):
    print(slice_no)
    img = image_data2[:,:,slice_no]
    imgplot = plt.imshow(img)
    plt.show(block=False)
    #time.sleep(1)
    plt.pause(0.1)
    plt.close()
  '''
  

  #name = '../all_images/VSD.Seg_001.'+ image_id + '.mha'
  #save(new_image,name)
