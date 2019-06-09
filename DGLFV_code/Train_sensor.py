#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:57:13 2019

@author: user
"""

import os
import skimage.io
import skimage.transform
import numpy as np
from keras.layers import Conv2D,Input,Dense,GlobalAveragePooling2D,Dropout,Activation,BatchNormalization
from keras.optimizers import SGD
from keras.models import Model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt;
from skimage.color import gray2rgb
from skimage.filters import threshold_niblack
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from PIL import Image
# ori data path
data_path = '850nm'
#data_path = 'Finger Vein Database'
# target data path
dae_path = '1'

X = []
y = []
h = os.listdir(data_path)
count = 0
for filename in h:
    im = skimage.io.imread(os.path.join(data_path,filename))
    resized_img = skimage.transform.resize(im, (224, 224),mode='reflect')
    img = resized_img.flatten()
    img = img[np.newaxis,:]
    if count == 0 :
        X = img
        count = 1
    else:
        X = np.vstack((X,img))
    label = filename.replace('.bmp', '').split('_')
    y = np.append(y,label[0])
print (X.shape)
print (y.shape)
print ("Complicate!")

X = np.reshape(X, (len(X),224, 224))
X = gray2rgb(X)
def conbine_img(x,y):
    comb_x = []
    comb_y = []
    count = 0
    for i in range(len(x)):
        if i == (len(x)-1):
            break;
        else:
            for j in range(len(x)-i-1):
                ori_img = x[i]
                compare_img = x[j+i+1]        
#                img = np.zeros((1,224,224,2))
                img = ori_img + compare_img
                img = img.reshape([1,224,224,3])
                if count == 0 :
                    comb_x = img
                    count = 1
                else :
                    comb_x = np.vstack((comb_x,img))
                if y[i] == y[j+i+1] :
                    comb_y.append(1)
                else :
                    comb_y.append(0)
    return comb_x,comb_y  

c = X[:60]
d = y[:60]
Comb_x,Comb_y = conbine_img(c,d)  

#
#X_train, X_test, y_train, y_test = train_test_split(Comb_x, Comb_y)
#
#
#base_model = VGG16(weights='imagenet', include_top=False)
base_model = VGG19(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(input=base_model.input, output=predictions)

model.compile(optimizer=SGD(lr=0.001,momentum=0.9), loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(Comb_x,Comb_y,epochs=320,
                    nb_epoch=20,
                    validation_data=(Comb_x,Comb_y))  
model_save = os.path.join('roc/sensors_SDU','SDU_'+model_name+'_'+str(Epochs)+'.h5')
model.save(model_save)