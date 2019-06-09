# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:14:26 2018

@author: Ashiki
"""

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt;
import os
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from inception_resnet_v2 import InceptionResNetV2
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn import preprocessing  
import numpy as np

#from keras.backend.tensorflow_backend import set_session
#import tensorflow as tf
#import os
##os.environ["CUDA_VISIBLE_DEVICES"]="2"
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
##config.gpu_options.allow_growth =True 
#set_session(tf.Session(config=config))

# input data

data_path = 'LNTU_data/LNTU_ori_data_massive'

train_path = os.path.join(data_path,'train')
test_path = os.path.join(data_path,'test')

# data for preview
preview_path = 'preview'
if os.listdir(preview_path):
    paths = os.listdir(preview_path)
    for path in paths:
        filePath = os.path.join(preview_path,path)
        os.remove(filePath)

xdata = ImageDataGenerator(
        #rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
data = xdata.flow_from_directory(train_path,batch_size=32,target_size=(224, 224),
                                save_to_dir=preview_path,
                                save_prefix='finger', save_format='bmp',
                                class_mode='categorical')
val_data = xdata.flow_from_directory(test_path,batch_size=32,target_size=(224,224),
                                     class_mode='categorical')                                    
                                      
model_path = 'roc/LNTU_518/IRV2_LNTU_500.h5'
model = load_model(model_path)

sgd = SGD(lr=0.001,momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit_generator(data,samples_per_epoch=320,nb_epoch=100,
                   validation_data=val_data,class_weight='auto',validation_steps=20
                   )
model.save('roc/LNTU_519_massive/IRV2_LNTU_600.h5')











