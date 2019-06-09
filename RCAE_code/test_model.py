#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:44:01 2018

@author: Ashiki
"""
import numpy as np
from skimage import io, transform
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
import skimage
import copy
from PIL import Image

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

def trans_ycbcr(gray_image):
    rgb_image = skimage.color.gray2rgb(gray_image)
    ycbcr_image = skimage.color.rgb2ycbcr(rgb_image)
    return ycbcr_image

def Norm_luma(gray_image):
    gray_image = gray_image*255
    min_image = gray_image*0
    max_image = copy.deepcopy(gray_image)
    max_image[max_image>=0] = 255    
    ycbcr_min = trans_ycbcr(min_image)
    ycbcr_max = trans_ycbcr(max_image)
    min_luma = np.mean(ycbcr_min[0])
    max_luma = np.mean(ycbcr_max[0])    
    ycbcr_image = trans_ycbcr(gray_image)
    ave_luma = np.mean(ycbcr_image[0])
    weight_luma = (ave_luma-min_luma)/(max_luma-min_luma)
    return weight_luma

def test_model(test_image,modelname,gamma,noisy):
    
    h = test_image.shape[0]                                     #  ./850nm/2_5.bmp
    w = test_image.shape[1]
    # Step2 : add_coeff
    add_img = add_coeff(test_image,noisy,gamma)  # gamma and noisy
    # Step3 : patches
    patches = image.extract_patches_2d(add_img,(16,16))
    # Step4 : to_rgb
    patches = gray2rgb(patches)
    # Step5 : load model and predict
    model = load_model(modelname)   
    print("load model...")    
    start_time = datetime.datetime.now()
    preds = model.predict(patches)
    end_time = datetime.datetime.now()
    print('Cost: '+str((end_time-start_time).seconds)+' Seconds')
    # Step6 : to_gray
    preds = rgb2gray(preds)    
    # Step7 : reconstruction 
    de_img = image.reconstruct_from_patches_2d(preds,(h,w))
    ssim,psnr = cal_standard(test_image,de_img)
    plot_compare(test_image,add_img,de_img)
    return de_img,ssim,psnr

def plot_compare(imag,add_imag,de_imag):    
    plt.subplot(1,3,1)
    plt.imshow(imag,cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(add_imag,cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(de_imag,cmap='gray')
    plt.show()


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    # read -> add_coeff -> patches -> to_rgb -> predict -> to_gray -> reconstruction 
    # './open_finger/001_left_index_1.png'
    file_name = './open_finger/001_left_index_1.png'
    de_model = './new_weight/open_noisy_RACE_419.h5'  
    dark_model = './new_weight/open_dark_RACE_419.h5'
    light_model = './new_weight/open_light_RACE_419.h5'   
    
    set_gamma = 1.5
    set_cor = 0
    ori_gamma = 1
    ori_cor = 0
    
    img = io.imread(file_name,as_gray=True)    # ./open_data/1.png  
#    img = rgb2gray(img)
    img = img / 255                                       #  ./850nm/2_5.bmp
    
    # denoisy model
    index1 = test_model(img,de_model,set_gamma,set_cor)
    # io.imsave('1.png',index1[0])        
    # dark model    
    index2 = test_model(index1[0],dark_model,ori_gamma,ori_cor)
        
    # light model 
    index3 = test_model(index1[0],light_model,ori_gamma,ori_cor)  
    
    weight = Norm_luma(index1[0])
    # use open_data
    result = index2[0]*(1-weight)+index3[0]*weight  
    # use open_finger
#    if set_gamma >1:
#        result = index2[0]*(1-weight)+index3[0]*weight  
#    else:
#        result = index2[0]*weight+index3[0]*(1-weight)
    ssim,psnr = cal_standard(img,result)
    plot_compare(index2[0],index3[0],result)
    
    plot_compare(img,index1[0],result)        
    io.imsave('./save_light/open_RACE_light_ori'+str(set_gamma)+'.png',index1[0])
    io.imsave('./save_light/open_RACE_light_'+str(set_gamma)+'.png',result)

