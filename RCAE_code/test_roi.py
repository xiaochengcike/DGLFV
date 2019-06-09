#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:44:01 2018

@author: Ashiki
"""
import os
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

def test_model(test_image,model,gamma,noisy):
    
    h = test_image.shape[0]
    w = test_image.shape[1]
    # Step2 : add_coeff
    add_img = add_coeff(test_image,noisy,gamma)  # gamma and noisy
    # Step3 : patches
    patches = image.extract_patches_2d(add_img,(16,16))
    # Step4 : to_rgb
    patches = gray2rgb(patches)
#    # Step5 : load model and predict
#    model = load_model(modelname)   
#    print("load model...")    
    start_time = datetime.datetime.now()
    preds = model.predict(patches)
    end_time = datetime.datetime.now()
    print('Cost: '+str((end_time-start_time).seconds)+' Seconds')
    # Step6 : to_gray
    preds = rgb2gray(preds)    
    # Step7 : reconstruction 
    de_img = image.reconstruct_from_patches_2d(preds,(h,w))
#    ssim,psnr = cal_standard(test_image,de_img)
    plot_compare(test_image,add_img,de_img)
    return de_img

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

    IMAGE_DIR = "/home/user/test/Ashiki_DLGFV/SDU_ROI_rotate"
    denoise_model_path = './new_weight/SDU_noisy_59.h5'  
    dark_model_path = './new_weight/SDU_dark_59.h5'
    light_model_path = './new_weight/SDU_light_59.h5'   
    save_path = "enhance_SDU/"

    
#    IMAGE_DIR = "/home/user/test/Ashiki_DLGFV/850nm"
#    denoise_model_path = './new_weight/LNTU_finger_noisy_59.h5'  
#    dark_model_path = './new_weight/LNTU_finger_dark_59.h5'
#    light_model_path = './new_weight/LNTU_finger_light_59.h5'   
#    save_path = "enhance_LNTU/"
    
    denoise_model = load_model(denoise_model_path)  
    dark_model = load_model(dark_model_path)
    light_model = load_model(light_model_path)    
        
    set_gamma = 1
    set_cor = 0
    ori_gamma = 1
    ori_cor = 0
        
    filenames = os.listdir(IMAGE_DIR)
    for i in range(len(filenames)):
        img = skimage.io.imread(os.path.join(IMAGE_DIR,filenames[i]))
        
#        img = rgb2gray(img)
        img = img / 255
        
        # denoisy model
        index1 = test_model(img,denoise_model,set_gamma,set_cor)
        # io.imsave('1.png',index1[0])        
        # dark model    
        index2 = test_model(index1,dark_model,ori_gamma,ori_cor)
            
        # light model 
        index3 = test_model(index1,light_model,ori_gamma,ori_cor)  
        
        weight = Norm_luma(index1)
        # use open_data
        result = index2*(1-weight)+index3*weight  
        # use open_finger
        #    if set_gamma >1:
        #        result = index2[0]*(1-weight)+index3[0]*weight  
        #    else:
        #        result = index2[0]*weight+index3[0]*(1-weight)
#        ssim,psnr = cal_standard(img,result)
#        plot_compare(index2[0],index3[0],result)
        
#        plot_compare(img,index1[0],result)        
        io.imsave(os.path.join(save_path,filenames[i]),result)
        print(" Finished one image")


