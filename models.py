import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow as tf
import keras.backend as K
# import keras
# from ensorflow import keras
from keras import layers
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Maximum
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose,UpSampling2D
# from tensorflow.keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose,UpSampling2D
# from tensorflow.keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose,UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D,MaxPooling3D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from sklearn.utils import class_weight
from keras.models import Sequential




def conv_block(input_mat,num_filters,kernel_size,batch_norm):
  X = Conv2D(num_filters,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same')(input_mat)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)

  X = Conv2D(num_filters,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same')(X)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
  
  return X



def Unet(input_img, n_filters = 16, dropout = 0.2, batch_norm = True):

  c1 = conv_block(input_img,n_filters,3,batch_norm)
  p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)
  p1 = Dropout(dropout)(p1)
  
  c2 = conv_block(p1,n_filters*2,3,batch_norm);
  p2 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c2)
  p2 = Dropout(dropout)(p2)

  c3 = conv_block(p2,n_filters*4,3,batch_norm);
  p3 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c3)
  p3 = Dropout(dropout)(p3)
  
  c4 = conv_block(p3,n_filters*8,3,batch_norm);
  p4 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c4)
  p4 = Dropout(dropout)(p4)
  
  c5 = conv_block(p4,n_filters*16,3,batch_norm);

  u6 = Conv2DTranspose(n_filters*8, (3,3), strides=(2, 2), padding='same')(c5);
  u6 = concatenate([u6,c4]);
  c6 = conv_block(u6,n_filters*8,3,batch_norm)
  c6 = Dropout(dropout)(c6)
  u7 = Conv2DTranspose(n_filters*4,(3,3),strides = (2,2) , padding= 'same')(c6);

  u7 = concatenate([u7,c3]);
  c7 = conv_block(u7,n_filters*4,3,batch_norm)
  c7 = Dropout(dropout)(c7)
  u8 = Conv2DTranspose(n_filters*2,(3,3),strides = (2,2) , padding='same')(c7);
  u8 = concatenate([u8,c2]);

  c8 = conv_block(u8,n_filters*2,3,batch_norm)
  c8 = Dropout(dropout)(c8)
  u9 = Conv2DTranspose(n_filters,(3,3),strides = (2,2) , padding='same')(c8);

  u9 = concatenate([u9,c1]);

  c9 = conv_block(u9,n_filters,3,batch_norm)
  outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

  model = Model(inputs=input_img, outputs=outputs)

  return model

def UnetRes(input_img, n_filters = 32, dropout = 0.4, batch_norm = True):

  c1 = conv_block(input_img,n_filters,3,batch_norm)
  p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)
  p1 = Dropout(dropout)(p1)
  
  c2 = conv_block(p1,n_filters*2,3,batch_norm);
  p2 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c2)
  p2 = Dropout(dropout)(p2)

  c3 = conv_block(p2,n_filters*4,3,batch_norm);
  p3 = MaxPooling2D(pool_size=(2,2) ,strides=2)(c3)
  p3 = Dropout(dropout)(p3)
  
  c5 = conv_block(p3,n_filters*8,3,batch_norm);

  u7 = Conv2DTranspose(n_filters*4,(3,3),strides = (2,2) , padding= 'same')(c5);

  u7 = concatenate([u7,c3]);
  c7 = conv_block(u7,n_filters*4,3,batch_norm)
  c7 = Dropout(dropout)(c7)
  u8 = Conv2DTranspose(n_filters*2,(3,3),strides = (2,2) , padding='same')(c7);
  u8 = concatenate([u8,c2]);

  c8 = conv_block(u8,n_filters*2,3,batch_norm)
  c8 = Dropout(dropout)(c8)
  u9 = Conv2DTranspose(n_filters,(3,3),strides = (2,2) , padding='same')(c8);

  u9 = concatenate([u9,c1]);

  c9 = conv_block(u9,n_filters,3,batch_norm)
  outputs = Conv2D(4, (1, 1), activation='softmax')(c9)

  model = Model(inputs=input_img, outputs=outputs)

  return model



def Unet_with_slice(input_img, n_filters = 16 , dropout = 0.3 , batch_norm = True):
  c1 = Conv2D(16,kernel_size = (1,6) , strides = (1,1) ,padding = 'valid')(input_img)
  if batch_norm:
    c1 = BatchNormalization()(c1)
  #print(c1.shape)
  c1 = Activation('relu')(c1)

  c1 = Conv2D(n_filters,kernel_size=(3,3),strides=(1,1),padding='same')(c1)
  if batch_norm:
    c1 = BatchNormalization()(c1)
  
  c1 = Activation('relu')(c1)

  p1 = MaxPooling2D(pool_size = (2,2) , strides = 2)(c1)
  p1 = Dropout(dropout)(p1)

  #print(p1.shape)
  c2 = conv_block(p1 , n_filters*2,3,batch_norm)
  p2 = MaxPooling2D(pool_size=(3,3), strides=3)(c2)
  p2 = Dropout(dropout)(p2)
  #print(p2.shape)

  c3 = conv_block(p2, n_filters*4,3,batch_norm)
  #print(c3.shape)
  p3 = MaxPooling2D(pool_size = (2,1) , strides = (2,1))(c3)
  p3 = Dropout(dropout)(p3)
  #print(p3.shape)

  c4 = conv_block(p3, n_filters*8,3,batch_norm)
  p4 = MaxPooling2D(pool_size = (4,4) , strides = (4,5))(c4)
  p4 = Dropout(dropout)(p4)

  c5 = conv_block(p4,n_filters*16,3,batch_norm)

  u6 = Conv2DTranspose(n_filters*8,kernel_size = (4,4) , strides = (4,5) , padding = 'same')(c5)
  u6 = concatenate([u6,c4])
  c6 = conv_block(u6,n_filters*8,3,batch_norm)
  c6 = Dropout(dropout)(c6)

  u7 = Conv2DTranspose(n_filters*4,kernel_size = (3,3) , strides = (2,1) , padding = 'same')(c6)
  u7 = concatenate([u7,c3])
  c7 = conv_block(u7,n_filters*4,3,batch_norm)
  c7 = Dropout(dropout)(c7)

  u8 = Conv2DTranspose(n_filters*2,kernel_size = (3,3) , strides = (3,3) , padding = 'same')(c7)
  u8 = concatenate([u8,c2])
  c8 = conv_block(u8,n_filters*2,3,batch_norm)
  c8 = Dropout(dropout)(c8)

  u9 = Conv2DTranspose(n_filters,kernel_size = (3,3) , strides = (2,2) , padding = 'same')(c8)
  u9 = concatenate([u9,c1])
  c9 = conv_block(u9,n_filters,3,batch_norm)
  c9 = Dropout(dropout)(c9)

  c10 = Conv2DTranspose(n_filters, kernel_size = (1,6) , strides = (1,1), padding = 'valid')(c9)

  outputs = Conv2D(4, kernel_size = (1,1), activation = 'softmax')(c10)

  model = Model(inputs = input_img , outputs = outputs)

  return model


def convA(input_mat,num_filters,batch_norm):

  c1 = Conv2D(num_filters,kernel_size = (1,1) , strides = 1 , padding = 'same')(input_mat)

  if batch_norm:
    c1 = BatchNormalization()(c1)
  
  c1 = Activation('relu')(c1)

  c2 = Conv2D(num_filters,kernel_size = (1,1) , strides = 1 , padding = 'same')(input_mat)
  c2 = Conv2D(num_filters,kernel_size = (3,3) , strides = 1 , padding = 'same')(c2)

  if batch_norm:
    c2 = BatchNormalization()(c2)
  
  c2 = Activation('relu')(c2)

  c3 = Conv2D(num_filters,kernel_size = (1,1) , strides = 1 , padding = 'same')(input_mat)
  c3 = Conv2D(num_filters,kernel_size = (3,3) , strides = 1 , padding = 'same')(c3)
  c3 = Conv2D(num_filters,kernel_size = (3,3) , strides = 1 , padding = 'same')(c3)

  if batch_norm:
    c3 = BatchNormalization()(c3)
  
  c3 = Activation('relu')(c3)

  c = concatenate([c1,c2,c3])
  
  return c


def convB(input_mat,num_filters,stride1,stride2,batch_norm = True):
  
  a1 = MaxPooling2D(pool_size = (1,1) , strides = (stride1,stride2))(input_mat)
  a1 = Dropout(0.3)(a1)

  a2 = Conv2D(num_filters,kernel_size = (1,1) , strides = (1,1) , padding = 'same')(input_mat) 
  print(a2.shape)
  a2 = MaxPooling2D(pool_size = (3,3) , strides = (stride1,stride2),padding='same')(a2)
  print(a2.shape)

  if batch_norm:
    a2 = BatchNormalization()(a2)
  
  a2 = Activation('relu')(a2)

  a3 = Conv2D(num_filters,kernel_size = (1,1) , strides = (1,1) , padding = 'same')(input_mat)
  a3 = Conv2D(num_filters,kernel_size = (3,3) , strides = (1,1) , padding = 'same')(a3)
  a3 = MaxPooling2D(pool_size = (3,3) , strides = (stride1,stride2),padding = 'same')(a3)

  if batch_norm:
    a3 = BatchNormalization()(a3)
  
  a3 = Activation('relu')(a3)


  a = concatenate([a1,a2,a3])

  return a


def convC(input_mat,num_filters,stride1,stride2):
  
  a1 = UpSampling2D(size = (stride1,stride2) ,interpolation = 'nearest')(input_mat)

  a2 = Conv2DTranspose(num_filters,kernel_size = (1,1) , strides = (1,1) , padding = 'same')(input_mat)  
  a2 = Conv2DTranspose(num_filters,kernel_size = (3,3) , strides = (stride1,stride2) , padding = 'same')(a2)
  a2 = Dropout(0.3)(a2)

  a3 = Conv2DTranspose(num_filters,kernel_size = (1,1) , strides = (1,1) , padding = 'same')(input_mat)
  a3 = Conv2DTranspose(num_filters,kernel_size = (3,3) , strides = (1,1) , padding = 'same')(a3)
  a3 = Conv2DTranspose(num_filters,kernel_size = (3,3) , strides = (stride1,stride2) , padding = 'same')(a3)
  a3 = Dropout(0.3)(a3)
  
  a = concatenate([a1,a2,a3])

  return a



def Unet_with_inception(input_img, n_filters = 16 , dropout = 0.3 , batch_norm = True):                        # for 240*155 dimensional images
  c1 = Conv2D(16,kernel_size = (1,6) , strides = (1,1) ,padding = 'valid')(input_img)
  if batch_norm:
    c1 = BatchNormalization()(c1)
  #print(c1.shape)
  c1 = Activation('relu')(c1)


  c1 = convB(c1,10,2,2)
  c1 = convA(c1,10,batch_norm)

  c2 = convB(c1,10,3,3)
  c2 = convA(c2,10,batch_norm)

  c3 = convB(c2,10,2,1)
  c3 = convA(c3,10,batch_norm)

  c4 = convB(c3,10,4,5)
  c4 = convA(c4,10,batch_norm)

  c4 = convA(c4,10,batch_norm)

  c5 = convC(c4,10,4,5)
  c5 = concatenate([c5,c3])
  c5 = convA(c5,10,batch_norm)

  c6 = convC(c5,10,2,1)
  c6 = concatenate([c6,c2])
  c6 = convA(c6,10,batch_norm)

  c7 = convC(c6,10,3,3)
  c7 = concatenate([c7,c1])
  c7 = convA(c7,10,batch_norm)

  c8 = convC(c7,10,2,2)

  c9 = Conv2DTranspose(n_filters, kernel_size = (1,6) , strides = (1,1), padding = 'valid')(c8)

  outputs = Conv2D(4, kernel_size = (1,1), activation = 'softmax')(c9)

  model = Model(inputs = input_img , outputs = outputs)

  return model





def Inception(input_img, n_filters = 16 , dropout = 0.3 , batch_norm = True):             # for 240*240 dimensional slices

  c1 = convB(input_img,10,2,2)
  c1 = convA(c1,10,batch_norm)

  c2 = convB(c1,15,2,2)
  c2 = convA(c2,15,batch_norm)

  c3 = convB(c2,15,2,2)
  c3 = convA(c3,15,batch_norm)

  c4 = convA(c3,20,batch_norm)
  
  c5 = convC(c4,15,2,2)
  c5 = concatenate([c5,c2])
  c5 = convA(c5,15,batch_norm)

  c6 = convC(c5,15,2,2)
  c6 = concatenate([c6,c1])
  c6 = convA(c6,15,batch_norm)

  c7 = convC(c6,10,2,2)
  c7 = concatenate([c7,input_img])
  #c7 = convA(c7,10,batch_norm)

  outputs = Conv2D(4, kernel_size = (1,1), activation = 'softmax')(c7)

  model = Model(inputs = input_img , outputs = outputs)

  return model




def survival_model():
  model = Sequential()
  model.add(Dense(32,input_shape=(20,),kernel_regularizer=keras.regularizers.l1(0.4) , activation='relu'))
  model.add(Dense(10,activation = 'relu'))
  model.add(Dense(1,activation = 'sigmoid'))

  return model


def dense_without_bottleneck(input_mat , kernel_dim , growth_rate):
  a = Conv2D(growth_rate,kernel_size = (kernel_dim,kernel_dim),strides = (1,1),padding = 'same')(input_mat)
  a = Activation('elu')(a)

  a = BatchNormalization()(a)


  a = concatenate([input_mat,a])

  return a

def dense_with_bottleneck(input_mat,kernel_dim,dilation,growth_rate):

  if not(dilation):
    a = Conv2D(growth_rate,kernel_size = (1,1) , strides = (1,1), padding = 'same')(input_mat)
    a = Activation('elu')(a)

    a = BatchNormalization()(a)

    a = Conv2D(growth_rate,kernel_size = (kernel_dim,kernel_dim),strides = (1,1),padding = 'same')(input_mat)
    a = Activation('elu')(a)

    a = BatchNormalization()(a)

    a = concatenate([input_mat,a])
  else:
    a = Conv2D(growth_rate,kernel_size = (1,1) , strides = (1,1), padding = 'same')(input_mat)
    a = Activation('elu')(a)

    a = BatchNormalization()(a)

    a = Conv2D(growth_rate,kernel_size = (kernel_dim,kernel_dim),strides = (1,1),padding = 'same',dilation_rate = dilation)(input_mat)
    a = Activation('elu')(a)

    a = BatchNormalization()(a)

    a = concatenate([input_mat,a])

  return a

def DeepScan(input_mat):
  a = Conv2D(24,kernel_size = (3,3) , padding = 'same')(input_mat)
  print(a.shape)

  a = dense_without_bottleneck(a,3,12)
  a = dense_without_bottleneck(a,3,12)
  a = dense_without_bottleneck(a,3,12)
  a = dense_without_bottleneck(a,3,12)
  a = dense_without_bottleneck(a,3,12)
  a = dense_without_bottleneck(a,3,12)
  a = dense_without_bottleneck(a,3,12)
  a = dense_without_bottleneck(a,3,12)

  a = Dropout(0.3)(a)

  a = dense_with_bottleneck(a,5,2,24)
  a = dense_with_bottleneck(a,5,2,24)
  a = dense_with_bottleneck(a,5,2,24)
  a = dense_with_bottleneck(a,5,2,24)

  a = Dropout(0.3)(a)

  a = dense_with_bottleneck(a,5,4,24)
  a = dense_with_bottleneck(a,5,4,24)
  a = dense_with_bottleneck(a,5,4,24)
  a = dense_with_bottleneck(a,5,4,24)

  a = Dropout(0.3)(a)

  a = dense_with_bottleneck(a,5,8,24)
  a = dense_with_bottleneck(a,5,8,24)
  a = dense_with_bottleneck(a,5,8,24)
  a = dense_with_bottleneck(a,5,8,24)

  a = Dropout(0.3)(a)

  a = dense_with_bottleneck(a,5,16,24)
  a = dense_with_bottleneck(a,5,16,24)
  a = dense_with_bottleneck(a,5,16,24)
  a = dense_with_bottleneck(a,5,16,24)

  a = Dropout(0.3)(a)

  a = dense_with_bottleneck(a,3,False,12)
  a = dense_with_bottleneck(a,3,False,12)
  a = dense_with_bottleneck(a,3,False,12)
  a = dense_with_bottleneck(a,3,False,12)
  a = dense_with_bottleneck(a,3,False,12)
  a = dense_with_bottleneck(a,3,False,12)
  a = dense_with_bottleneck(a,3,False,12)
  a = dense_with_bottleneck(a,3,False,12)

  a = Dropout(0.5)(a)

  a = Conv2D(200,kernel_size = (1,1) , activation = 'elu')(a)
  a = BatchNormalization()(a)
  a = Conv2D(50,kernel_size = (1,1) , activation = 'elu')(a)
  a = BatchNormalization()(a)
  a = Conv2D(5,kernel_size = (1,1) , activation = 'sigmoid')(a)

  model = Model(inputs = input_mat , outputs = a)

  return model
#def 
#from utils import one_hot_encode,dice_coef_loss,dice_coef,f1_score

#base_model = load_model('survival_pred.h5',custom_objects={'dice_coef_loss':dice_coef_loss, 'f1_score':f1_score})
#base_model.summary()
#input_img1 = Input((240,240,4))
#input_img2 = Input((240,155,4))

#model1 = Inception(input_img1,16,0.3,True)
#model2 = Unet_with_inception(input_img2,16,0.3,True)

#model1.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss="categorical_crossentropy", metrics=["accuracy"])
#model1.summary()


#model2.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss="categorical_crossentropy", metrics=["accuracy"])
#model2.summary()
#input_img = Input((240,240,4))
#model = DeepScan(input_img)
#model.summary()