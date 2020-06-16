import cv2
import h5py
import imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import (
    Activation,
    Conv3D,
    Deconvolution3D,
    MaxPooling3D,
    UpSampling3D,
)
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.compat.v1.logging import INFO, set_verbosity
from keras.models import load_model
import csv
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import os

from sklearn_extra.cluster import KMedoids
import pandas as pd
from lifelines import CoxPHFitter
#from pickle import dumps,loads
from joblib import load
set_verbosity(INFO)
cph_new = load('cox.joblib')
K.set_image_data_format("channels_first")

age_dict = {}
days_dict = {}
from joblib import dump
from medpy.io import load
with open('survival_data.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file,delimiter = ',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(row)
            key = row[0]
            age = row[1]
            days = row[2]
            age_dict[key] = age
            days_dict[key] = days
            line_count+=1

    print(f'Processed {line_count} lines.')



def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)



#model2 = load_model('tumor_segmentation_model.h5',custom_objects={'dice_coefficient_loss':dice_coefficient_loss , 'dice_coefficient':dice_coefficient})
model3 = load_model('isensee_2017_model.h5',custom_objects={'weighted_dice_coefficient':weighted_dice_coefficient, 'weighted_dice_coefficient_loss':weighted_dice_coefficient_loss, 'InstanceNormalization':InstanceNormalization})
#model = load_model('tumor_segmentation_weights.h5')
#model3.summary()

layer_name = 'add_5'

intermediate_layer_model = Model(inputs=model3.get_layer('input_1').input,outputs=model3.get_layer(layer_name).output)

path = '../Brats17TrainingData/HGG'
all_images = os.listdir(path)

final_X = []
final_X_2 = []
ground_truth = []
data = np.zeros((4,128,128,128))


for i in range(0,2):
    print(i)
    survival_features = []
    model_data = []
    x_to = []
    y_to = []
    m = all_images[i]
    if m in days_dict.keys():
        print("He survived ",days_dict[m])
        folder_path = path + '/' + m;
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
              data[w,:,:,:] = image_data[40:168,40:168,10:138]
              print("Entered modality")
              w = w+1

        print(data.shape)
        #print(image_data2.shape)  
        data2 = data.reshape(1,4,128,128,128)
        features = intermediate_layer_model.predict(data2)
        print(features.shape)

        features = features.reshape((1*256*8*8*8))
        features = features[0:256*8*8*8:8]
        features = np.unique(features)
        print(features.shape)
        image_features = np.zeros((features.shape[0],2))
        for x in range(len(features)):
            image_features[x,0] = features[x]

        kmedoids = KMedoids(n_clusters = 19,random_state=0).fit(image_features)

        for x in kmedoids.cluster_centers_:
            survival_features.append(x[0])

        model_data = survival_features
        model_data.append(age_dict[m])
        ground_truth.append(days_dict[m])

        survival_features.append(days_dict[m])
        #survival_features = np.asarray(survival_features)
        #final_X.append(survival_features)
        #ground_truth.append(days_dict[m])

        survival_features.append(age_dict[m])
        #survival_features.append(1)

        final_X.append(survival_features)
        final_X_2.append(model_data)
        '''
        
'''

final_X = np.asarray(final_X)
columns = ["column1","column2","column3","column4","column5","column6","column7","column8","column9","column10","column11","column12","column13","column14","column15","column16","column17","column18","column19","T","Age"]
df =  pd.DataFrame(data=final_X, columns=columns)
print(df)

cph = CoxPHFitter(penalizer = 0.1)
cph.fit(df,duration_col = 'T')
cph.print_summary()

dump(cph,'cox.joblib')
'''

final_X_2 = np.asarray(final_X_2)
ground_truth = np.asarray(ground_truth)

from models import survival_model

nn_model = survival_model()
nn_model.compile(Adam(),loss=survival_loss)
y = nn_model(final_X_2)
print(y.shape)
print(final_X_2.shape)
nn_model.fit(x=final_X_2,y=ground_truth)
#cph_new = loads(s_cph)

#ground_truth = np.asarray(ground_truth)
#print(final_X.shape)
#print(ground_truth.shape)
'''
'''
to_train = np.asarray(to_train)
ground_truth = np.asarray(ground_truth)

print(to_train.shape)
print(ground_truth.shape)
'''
