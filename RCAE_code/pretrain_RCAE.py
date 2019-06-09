#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:58:05 2018

@author: Ashiki
"""
import numpy as np
from skimage import io, transform
import os
import matplotlib.pyplot as plt
from keras.layers import Dense,Conv2D,Input,MaxPooling2D,UpSampling2D,merge,Activation,BatchNormalization
from keras.models import Model
from skimage.color import gray2rgb,rgb2gray
from skimage import util,exposure
import random
from keras.optimizers import Adadelta, SGD
import datetime
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras
import copy

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def overlapping_patches(pic,stride,patchsize):
    # stride = 2
    # patchsize = 16
    h = pic.shape[0]
    w = pic.shape[1]
   
    # padding
    if h % patchsize != 0:
        ex_h = int(h/patchsize) + 1
        height = ex_h * patchsize
        temp_h = height - h
        pic = np.vstack((pic,np.zeros([temp_h,w])))
    else:
        height = h
    
    if w % patchsize != 0:
        ex_w = int(w/patchsize) + 1
        width = ex_w * patchsize
        temp_w = width - w
        pic = np.hstack((pic,np.zeros([height,temp_w])))
    else:
        width = w 
    # get patches
    start_h = 0
    result = []
    while start_h < height:
        start_w = 0
        while start_w + patchsize <= width:
            crop = pic[start_h:start_h+patchsize,start_w:start_w+patchsize]
            if start_w==0 and start_h==0:
                result = crop[np.newaxis,:,:]
            else:
                result = np.append(result,crop[np.newaxis,:,:],axis=0)
            
            if start_w + patchsize == width:
                start_h += patchsize    
            
            start_w += stride       
    return result

def build_model():
    
    input_img = Input(shape=(16, 16,3))
    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x1 = MaxPooling2D((2, 2), border_mode='same')(x)
    x2 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(x1)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x2)
    
    # branch 1
    ir1 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(encoded)
    # branch 2
    ir2 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(encoded)
    ir2 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(ir2)
    # branch 3
    ir3 = Conv2D(32, 1, 1, activation='relu', border_mode='same')(encoded)
    ir3 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(ir3)
    ir3 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(ir3)
    
    ir_merge = merge([ir1, ir2, ir3], concat_axis=-1, mode='concat')
    
    ir_conv = Conv2D(32, 1, 1, activation='linear', border_mode='same')(ir_merge)
    
    out = merge([encoded, ir_conv], mode='sum')
    out = BatchNormalization(axis=-1)(out)
    out = Activation("relu")(out)
    
    x3 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(out)
    x4 = UpSampling2D((2, 2))(x3)
    
    merge1 = merge([x2, x4], mode='sum')
    x5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(merge1)
    x6 = UpSampling2D((2, 2))(x5)
    merge2 = merge([x, x6], mode='sum')
    decoded = Conv2D(3, 3, 3, activation='sigmoid', border_mode='same')(merge2)
    
    autoencoder = Model(input_img, decoded)
    
    return autoencoder

def build_com():
    input_img = Input(shape=(16, 16, 3))
    x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
    autoencoder = Model(input_img, decoded)
    return autoencoder

def build_LLCNN():
    input_img = Input(shape=(16, 16,3))
    x1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x2 = MaxPooling2D((2, 2), border_mode='same')(x1)
    x3 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(x2)
    x4 = MaxPooling2D((2, 2), border_mode='same')(x3)
    x5 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x6 = MaxPooling2D((4, 4), border_mode='same')(x5)
    merge1 = merge([x6, x4], concat_axis=-1, mode='concat')
    relu1 = Activation("relu")(merge1)
    x7 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(relu1)
    x8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(x7)
    merge2 = merge([relu1, x8], mode='sum')
    relu2 = Activation("relu")(merge2)
    x9 = UpSampling2D((4, 4))(relu2 )
    decoded = Conv2D(3, 3, 3, activation='sigmoid', border_mode='same')(x9)		
    autoencoder = Model(input_img, decoded)  
    return autoencoder
		
def build_llnet():
    
    encoding_dim = 1200
    input_img = Input(shape=(256,))
    encoded1 = Dense(2000,activation = 'relu')(input_img)
    encoded2 = Dense(1600,activation = 'relu')(encoded1)   
    bottleneck = Dense(encoding_dim,activation = 'relu')(encoded2)  
    decoded1 = Dense(1600,activation = 'relu')(bottleneck)
    decoded2 = Dense(2000,activation = 'relu')(decoded1)
    output_img = Dense(256,activation = 'sigmoid')(decoded2)    
    llnet = Model(input=input_img,output=output_img)    
    return llnet

def add_coeff(patch,mode):
    
    coeff_patches = []
    
    row = patch.shape[1]
    col = patch.shape[2]
    
    if mode == 'both':
        for i in patch:
            var = random.uniform(0.5,1.5)
            rvar = random.uniform(0.005,0.01)
            temp = util.random_noise(i, mode='gaussian',var=rvar)
            coeff_patches.append(exposure.adjust_gamma(temp, var))
    elif mode == 'noisy':
        for i in patch:
            rvar = random.uniform(0,0.0000)
            temp_n = util.random_noise(i, mode='gaussian',var=rvar)
            coeff_patches.append(temp_n)
    elif mode == 'dark':        
        for i in patch:
            var = random.uniform(1,2.2)
            temp_d = exposure.adjust_gamma(i, var)
            coeff_patches.append(temp_d)
    elif mode == 'light':        
        for i in patch:
            var = random.uniform(0.3,1)
            temp_d = exposure.adjust_gamma(i, var)
            coeff_patches.append(temp_d)
    else:
        print("Please input correct mode") 
                       
    coeff_patches = np.reshape(coeff_patches, (len(coeff_patches),row, col))
    
    return coeff_patches


if __name__== '__main__':
       
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    
    # read images -> patches -> split for train&test -> add_coeff -> to_rgb 
    # Choices : LNTU_finger or SDU_finger or MGLI or ROI_finger
    which_database = 'ROI_finger'
    train_mode = 'light'
    epoch = 20
    
    X = []
    if which_database == 'SDU_finger':        # ori_image
        path = 'train_data/SDU_data/train_set/*.png'
        h = 240
        w = 320
    elif which_database == 'LNTU_finger':        
        path = 'train_data/LNTU_850nm/train_set/*.bmp'
        h = 415
        w = 176
    elif which_database == 'ROI_finger':      # ROI_image  
        path = 'train_data/SDU_ROI/train_set/*.png'
        h = 320
        w = 240
    elif which_database == 'MGLI':        
        path = 'train_data/MGLI_data/train_set/*.png'
        h = 512
        w = 512
    else:
        print("Error!")
    
    images = io.ImageCollection(path)
    
    for i in range(len(images)):
        if which_database == 'SDU_finger' or 'LNTU_finger' or 'ROI_finger':
            mid_img = copy.deepcopy(images[i])
            mid_img = rgb2gray(mid_img)
        else : 
            mid_img = copy.deepcopy(images[i])
        resized_img = transform.resize(mid_img, (h, w),mode='reflect')
        temp = np.reshape(resized_img,(1,h,w))
        if i == 0:
            X = temp
        else:
            X = np.vstack((X,temp))
    print ("Complicate to read image.")    
            
    # use directly
    patches = []
    for i in range(len(X)):
        temp_patches = overlapping_patches(X[i],2,16)
        if i == 0:
            patches = temp_patches
        else:
            patches = np.vstack((patches,temp_patches))
    print ("Get patches.")       
    
    # SDU_finger database gets 40680 patches
    # LNTU_finger database gets  56862 patches
    # MGLI database gets  143424 patches
    if which_database == 'SDU_finger' or 'ROI_finger':        
        X_train = patches[:40000]
        X_test = patches[40000:]
    elif which_database == 'LNTU_finger':        
        X_train = patches[:50000]
        X_test = patches[50000:]
    elif which_database == 'MGLI':        
        X_train = patches[:100000]
        X_test = patches[100000:]
    else:
        print("Error!")                        
    
    # Have 3 choices : both noisy dark 
    x_train_both = add_coeff(X_train,train_mode)    
    x_test_both = add_coeff(X_test,train_mode) 
    
    
    X_train = gray2rgb(X_train)
    X_test = gray2rgb(X_test)             
    x_test_both = gray2rgb(x_test_both)
    x_train_both = gray2rgb(x_train_both)
    
    print ("Ready to train model.")
    
    model = build_model()
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)#'adadelta'
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    history = LossHistory()
    model.fit(x_train_both,X_train,
                    nb_epoch = epoch,
                    batch_size = 32,
                    shuffle = True,
                    validation_data = (x_test_both,X_test),
                    callbacks=[history])    
    # name: database_coeff_date.h5 
    print(datetime.datetime.now())
    weight_path = './new_weight/' 
    weight_name = which_database+'_'+train_mode+'_'+str(datetime.datetime.now().month)+str(datetime.datetime.now().day)
    
    model.save(weight_path+weight_name+'.h5')  
#    
#    loss_path = './new_loss/'    
#    np.save(loss_path+weight_name+'3.npy',history.losses)    

