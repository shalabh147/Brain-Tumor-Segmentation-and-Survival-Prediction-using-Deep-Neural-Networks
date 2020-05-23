#this file is for training survival prediction model using features extracted from segmentation model

#import random
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow as tf
import keras.backend as K
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Maximum
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D,MaxPooling3D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize

import os

from medpy.io import load
import numpy as np

from sklearn import metrics
from sklearn_extra.cluster import KMedoids

from ../models import survival_model

import csv
import pickle

from joblib import dump

age_dict = {}
days_dict = {}


def encode(truth):
	#print(truth)
	truth = int(truth)

	if(truth < 300):
		return 1
	elif(truth >= 300 and truth <= 450):
		return 2
	else:
		return 3


with open('../survival_data.csv', mode='r') as csv_file:
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

from ../utils import one_hot_encode,dice_coef_loss,dice_coef,f1_score

base_model = load_model('../Models/survival_pred.h5',custom_objects={'dice_coef_loss':dice_coef_loss, 'f1_score':f1_score})
layer_name = 'dropout_4'

intermediate_layer_model = Model(inputs=base_model.get_layer('input_1').input,outputs=base_model.get_layer(layer_name).output)

path = '../../Brats17TrainingData/HGG'
all_images = os.listdir(path)
#print(len(all_images))

model_train = survival_model()
model_train.compile(optimizer=Adam(),loss='mean_squared_logarithmic_error')

#import xgboost as xgb
#xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.01, max_depth = 5, alpha = 10)
#early_stopping_monitor = EarlyStopping(patience=3)

to_train = []
ground_truth = []
data = np.zeros((240,240,155,4))
for i in range(0,174):
	print(i)
	final_image_features = []
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
			  data[:,:,:,w] = image_data
			  print("Entered modality")
			  w = w+1

		print(data.shape)
		print(image_data2.shape)  


		for slice_no in range(0,240):
			a = slice_no
			X = data[:,slice_no,:,:]

			Y = image_data2[:,slice_no,:]
			X = X.reshape(1,240,155,4)

			if(X.any()!=0 and Y.any()!=0 and len(np.unique(Y))==4):
				#print(X.shape)
				new_features = intermediate_layer_model.predict(X)
				#print(slice_no)
				new_features = new_features.reshape(1*5*5*128)
				new_features = np.unique(new_features)

				features = np.zeros((new_features.shape[0],2))
				for x in range(len(new_features)):
					features[x,0] = new_features[x]

				kmedoids = KMedoids(n_clusters=8, random_state=0).fit(features)

				for x in kmedoids.cluster_centers_:
					final_image_features.append(x[0])

		


		reduced_features = []
		final_image_features = np.asarray(final_image_features)
		final_image_features = np.unique(final_image_features)
		image_features = np.zeros((final_image_features.shape[0],2))
		for x in range(len(final_image_features)):
			image_features[x,0] = new_features[x]

		kmedoids = KMedoids(n_clusters=19, random_state=0).fit(image_features)

		for x in kmedoids.cluster_centers_:
			reduced_features.append(x[0])

		
		reduced_features.append(age_dict[m])
		reduced_features = np.asarray(reduced_features)	

		truth = days_dict[m]


		to_train.append(reduced_features)
		print("truth",truth)
		ground_truth.append(encode(truth))


to_train = np.asarray(to_train)
ground_truth = np.asarray(ground_truth)

print(to_train.shape)
print(ground_truth.shape)

#xg_reg.fit(to_train,ground_truth)									#An XGBoostREgressor model to try out
#pickle.dump(xg_reg, open("pima.pickle.dat", "wb"))					#saving model to file pima.pickle.dat

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))			#An SVM classifier to classify into 3 categories
clf.fit(to_train,ground_truth)

dump(clf,'SVMfit.joblib')											#Saving model to file SVMfit.joblib


model_train.fit(x=to_train,y=ground_truth,epochs = 1500,batch_size = 15)
model_train.save('Models/dense_survival_classifier.h5')				#A neural networks based dense regression model


