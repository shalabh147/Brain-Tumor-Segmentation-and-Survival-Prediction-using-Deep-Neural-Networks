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
import numpy as np

import cv2
from sklearn import metrics
from sklearn_extra.cluster import KMedoids

import csv

age_dict = {}
days_dict = {}
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

from utils import one_hot_encode,dice_coef_loss,dice_coef,f1_score

base_model = load_model('survival_pred.h5',custom_objects={'dice_coef_loss':dice_coef_loss, 'f1_score':f1_score})
layer_name = 'dropout_4'

intermediate_layer_model = Model(inputs=base_model.get_layer('input_1').input,outputs=base_model.get_layer(layer_name).output)

path = '../Brats17TrainingData/HGG'
all_images = os.listdir(path)
#print(len(all_images))

data = np.zeros((240,240,155,4))
for i in range(1):
	print(i)
	final_image_features = []
	x_to = []
	y_to = []
	x = all_images[i]
	if x in days_dict.keys():
		print("He survived ",days_dict[x])
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
			X = data[:,slice_no,:,:]

			Y = image_data2[:,slice_no,:]
			X = X.reshape(1,240,155,4)

			if(X.any()!=0 and Y.any()!=0 and len(np.unique(Y))==4):
				#print(X.shape)
				new_features = intermediate_layer_model.predict(X)
				print(new_features.shape)
				new_features = new_features.reshape(1*5*5*128)
				new_features = np.unique(new_features)

				features = np.zeros((new_features.shape[0],2))
				for x in range(len(new_features)):
					features[x,0] = new_features[x]

				kmedoids = KMedoids(n_clusters=8, random_state=0).fit(features)

				for x in kmedoids.cluster_centers_:
					final_image_features.append(x[0])

	final_image_features = np.asarray(final_image_features)
	print(final_image_features)
	# now take 19 of these total final_image_features,append age to it and feed it to survival model to train


	
'''
new_image = new_image.reshape(1,128,128,4)
new_image = tf.cast(new_image,tf.float32)
print(new_image.shape)


print(base_model.summary())


new_features = base_model(new_image)
print(new_features[0][100][100][2])




new_features = intermediate_layer_model(new_image)
print(new_features.shape)

#proto_tensor = tf.compat.v1.make_tensor_proto(new_features)  # convert `tensor a` to a proto tensor
#hello = tf.make_ndarray(new_features)
from keras import backend as K
new = K.eval(new_features)
new_damn = new.reshape(1*8*8*256)

new_damn = np.unique(new_damn)
print(new_damn.shape)
#from k_medoids import kmedoids
#a = kmedoids(new_damn,20,2)
#print(new_damn.shape)



kmedoids = KMedoids(n_clusters=20, random_state=0).fit(features)

print(kmedoids.cluster_centers_)

model = survival_model()
model.compile(optimizer=Adam(),loss='mean_squared_error',metrics='accuracy')
model.fit(x=features,y=survival)
'''