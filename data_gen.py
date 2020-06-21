from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import numpy as np
import cv2
from keras.utils import Sequence
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow as tf
import keras.backend as K
import keras
import os
import nibabel as nib
from PIL import Image
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Maximum
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose,UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D,MaxPooling3D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import os
def standardize(image):

    standardized_image = np.zeros(image.shape)

    #
   
        # iterate over the `z` dimension
    for z in range(image.shape[2]):
        # get a slice of the image
        # at channel c and z-th dimension `z`
        image_slice = image[:,:,z]

        # subtract the mean from image_slice
        centered = image_slice - np.mean(image_slice)
       
        # divide by the standard deviation (only if it is different from zero)
        if(np.std(centered)!=0):
            centered = centered/np.std(centered)

        # update  the slice of standardized image
        # with the scaled centered and scaled image
        standardized_image[:, :, z] = centered

    ### END CODE HERE ###

    return standardized_image


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self,
                 to_fit=True, batch_size=1, dim=(240, 240),
                 n_channels=4, n_classes=4, shuffle=False):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        #self.list_IDs = list_IDs
        #self.labels = labels
        #self.image_path = image_path
        #self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.x_to = []
        self.y_to = []
        #self.on_epoch_end()
       
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return 180//self.batch_size
   
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
       
        #x = all_images[index]
        path = '../BRATS2017/Brats17TrainingData/HGG'
        all_images = os.listdir(path)
        #print(len(all_images))
        all_images.sort()
        data = np.zeros((240,240,155,4))

        #x_to = []
        #y_to = []
        for i in range(index*self.batch_size,index*self.batch_size + self.batch_size):
            print(i)

            x = all_images[i]
            folder_path = path + '/' + x;
            modalities = os.listdir(folder_path)
            modalities.sort()
            #data = []
            w = 0
            for j in range(len(modalities)-1):
                image_path = folder_path + '/' + modalities[j]
                print(image_path)
                if not(image_path.find('seg.nii') == -1):
                    img = nib.load(image_path);
                    image_data2 = img.get_data()
                    image_data2 = np.asarray(image_data2)
                    print("Entered ground truth")
                else:
                    img = nib.load(image_path);
                    image_data = img.get_data()
                    image_data = np.asarray(image_data)
                    image_data = standardize(image_data)
                    data[:,:,:,w] = image_data
                    print("Entered modality")
                    w = w+1
   
            print(data.shape)
            print(image_data2.shape)  

   
            for slice_no in range(0,155):
                #a = slice_no
                X = data[:,:,slice_no,:]

                Y = image_data2[:,:,slice_no]


                if(X.any()!=0 and Y.any()!=0 and len(np.unique(Y)) == 4):
                    self.x_to.append(X)
                    self.y_to.append(Y.reshape(240,240,1))
                    if len(self.x_to)>=60:
                        break;

        x_to = np.asarray(self.x_to)
        y_to = np.asarray(self.y_to)
        x_to,y_to = shuffle(x_to,y_to)
       
        print(x_to.shape)
        print(y_to.shape)
           


        return x_to,y_to


training_generator = DataGenerator()

def conv_block(input_mat,num_filters,kernel_size,batch_norm):
  X = Conv2D(num_filters,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same' , kernel_initializer = 'he_normal')(input_mat)
  if batch_norm:
    X = BatchNormalization()(X)
 
  X = Activation('relu')(X)

  X = Conv2D(num_filters,kernel_size=(kernel_size,kernel_size),strides=(1,1),padding='same' , kernel_initializer = 'he_normal')(X)
  if batch_norm:
    X = BatchNormalization()(X)
 
  X = Activation('relu')(X)
   
  #X = add([X,Y])
 
  return X


def Unet(input_img, n_filters = 32, dropout = 0.4, batch_norm = True):

  c1 = conv_block(input_img,n_filters,3,batch_norm)
  p1 = Conv2D(n_filters,kernel_size = (3,3), strides=2, padding = 'same' , kernel_initializer = 'he_normal')(c1)
  #p1 = Dropout(dropout)(p1)
 
  c2 = conv_block(p1,n_filters*2,3,batch_norm);
  p2 = Conv2D(n_filters,kernel_size = (3,3), strides=2 , padding = 'same' , kernel_initializer = 'he_normal')(c2)
  #p2 = Dropout(dropout)(p2)

  c3 = conv_block(p2,n_filters*4,3,batch_norm);
  p3 = Conv2D(n_filters,kernel_size = (3,3), strides=2 , padding = 'same' , kernel_initializer = 'he_normal')(c3)
  #p3 = Dropout(dropout)(p3)
 
  c4 = conv_block(p3,n_filters*8,3,batch_norm);
  p4 = Conv2D(n_filters,kernel_size = (3,3), strides=2 , padding = 'same' , kernel_initializer = 'he_normal')(c4)
  #p4 = Dropout(dropout)(p4)
 
  c5 = conv_block(p4,n_filters*16,3,batch_norm);

  u6 = Conv2DTranspose(n_filters*8, (3,3), strides=(2, 2), padding='same')(c5);
  u6 = concatenate([u6,c4]);
  c6 = conv_block(u6,n_filters*8,3,batch_norm)
  #c6 = Dropout(dropout)(c6)
  u7 = Conv2DTranspose(n_filters*4,(3,3),strides = (2,2) , padding= 'same')(c6);

  u7 = concatenate([u7,c3]);
  c7 = conv_block(u7,n_filters*4,3,batch_norm)
  #c7 = Dropout(dropout)(c7)
  u8 = Conv2DTranspose(n_filters*2,(3,3),strides = (2,2) , padding='same')(c7);
  u8 = concatenate([u8,c2]);

  c8 = conv_block(u8,n_filters*2,3,batch_norm)
  #c8 = Dropout(dropout)(c8)
  u9 = Conv2DTranspose(n_filters,(3,3),strides = (2,2) , padding='same')(c8);

  u9 = concatenate([u9,c1]);

  c9 = conv_block(u9,n_filters,3,batch_norm)
  outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

  model = Model(inputs=input_img, outputs=outputs)

  return model

def dice_coef_2(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

input_img = Input((240, 240, 4))
model = Unet(input_img,32,0.4,True)
model.summary()
#model = load_model('../working/bce_model.h5',custom_objects = {'bce_dice_loss' : bce_dice_loss , 'dice_coef_2' : dice_coef_2 })
#earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,min_delta = 0.001, patience = 4, mode = 'min')
#callbacks_list = [earlystopping]
learning_rate = 0.00015
decay_rate = 0.0000001
#decay_rate = 0.000001
#epochs =

model.compile(optimizer=Adam(lr=learning_rate,decay = decay_rate),loss='binary_crossentropy', metrics=[dice_coef_2])

history = model.fit_generator(generator=training_generator,epochs = 50 , use_multiprocessing=True,workers=11,steps_per_epoch = 60)
model.save("data_Gen.h5")
plt.plot(history.history['dice_coef_2'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('../working/accuracy_plot1')
plt.show()
plt.close()

plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('../working/loss_plot1')
plt.show()
plt.close()


