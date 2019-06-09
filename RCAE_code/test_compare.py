#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:54:01 2019

@author: user
"""
from skimage import io,restoration,transform,util,exposure,measure
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import gray2rgb,rgb2gray


def add_coeff(patch,coeff_noise,coeff_dark):
    
    temp = util.random_noise(patch, mode='gaussian',var=coeff_noise)
    coeff_pathes = exposure.adjust_gamma(temp, coeff_dark)
    return coeff_pathes

def cal_standard(ori,denoise):
    
    ssim = measure.compare_ssim(ori, denoise ,multichannel=True) #
    psnr = measure.compare_psnr(ori, denoise)
    return ssim,psnr   

def plot_histogram(img):
    
    temp = exposure.histogram(img, nbins=200)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.hist(temp, bins=10)
    #plt.title('Age distribution')
    #plt.xlabel('Age')
    #plt.ylabel('Employee')
    #plt.show()
    
 
if __name__ == "__main__":    
    
    img = io.imread('open_data/17.png',as_gray=True)# open_data/17.png ./850nm/51_9.bmp
    img = img/255
    
    add_img = add_coeff(img,0.01,3) 
    # method
    de_tv = restoration.denoise_tv_chambolle(add_img, weight=1,eps=0.02, n_iter_max=100)
    #de_bil = restoration.denoise_bilateral(add_img, sigma_color=0.1, sigma_spatial=5, multichannel=False)
    de_nl = restoration.denoise_nl_means(add_img, patch_size=4, patch_distance=3, h=0.6,multichannel=False)
    de_nl = de_nl.astype('float64')
    de_wev = restoration.denoise_wavelet(add_img, sigma=0.1)
    
    de_adapteq = exposure.equalize_adapthist(add_img, clip_limit=0.03)
