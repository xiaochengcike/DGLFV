#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:44:01 2018

@author: Ashiki
"""
import numpy as np
from skimage import io, transform
import os
import matplotlib.pyplot as plt
from keras.layers import Conv2D,Input,MaxPooling2D,UpSampling2D,merge,Activation,BatchNormalization
from skimage.color import gray2rgb,rgb2gray
from skimage import util, exposure, measure, filters, morphology
import random
import warnings
from keras.models import load_model
from keras.optimizers import Adadelta, SGD
from sklearn.feature_extraction import image
import datetime
from skimage.color import gray2rgb,rgb2gray

def cal_standard(ori,denoise):
    ssim = measure.compare_ssim(ori, denoise ,multichannel=True) 
    psnr = measure.compare_psnr(ori, denoise)
    return ssim,psnr

def add_coeff(patch,coeff_noise,coeff_dark):
    # Add Noisy
    temp = util.random_noise(patch, mode='gaussian',var=coeff_noise)
    # Change light
    coeff_pathes = exposure.adjust_gamma(temp, coeff_dark)
    
    return coeff_pathes

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    # read -> add_coeff -> patches -> to_rgb -> predict -> to_gray -> reconstruction 
    # './open_data/train_set/17.png' './open_finger/001_left_index_1.png'
    file_name = './open_ROI/train_set/001_left_index_1.png'     # './850nm/2_5.bmp'
    model_name = './new_weight/finger_noisy_594.h5'   #'finger_dark_66.h5'
    gamma = 0.8
    cor = 0   
    
    
    # Step1 : read
    img = io.imread(file_name,as_gray=True)
    img = rgb2gray(img)    
    img = img / 255
    h = img.shape[0]
    w = img.shape[1]    

    # Step2 : add_coeff
    add_img = add_coeff(img,cor,gamma)
    
    # Step3 : patches
    patches = image.extract_patches_2d(add_img,(16,16))
    
    # Step4 : to_rgb
    patches = gray2rgb(patches)
    
    # Step5 : load model and predict
    model = load_model(model_name)   
    print("load model...")    
    start_time = datetime.datetime.now()
    preds = model.predict(patches)
    end_time = datetime.datetime.now()
    print('Cost: '+str((end_time-start_time).seconds)+' Seconds')
    
    # Step6 : to_gray
    preds = rgb2gray(preds)
    
    # Step7 : reconstruction 
    de_img = image.reconstruct_from_patches_2d(preds,(h,w))

    ssim,psnr = cal_standard(img,de_img)
    # io.imsave('./save_image/open_LLNet_noisy_0.01.png',de_img)
    plt.subplot(1,3,1)
    plt.imshow(img,cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(add_img,cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(de_img,cmap='gray')
    plt.show()
    io.imsave('1.png',de_img)