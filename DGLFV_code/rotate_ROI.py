#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:29:46 2019

@author: user
"""

import numpy as np
import skimage 
import os
from skimage import transform
ori_path = '/home/user/test/mask rcnn/ROI_result'
rotate_path = 'SDU_ROI_rotate' 
files = os.listdir(ori_path)
for i in range(len(files)):
    image = skimage.io.imread(os.path.join(ori_path,files[i]))
    rotate = image.T
    skimage.io.imsave(os.path.join(rotate_path,files[i]),rotate)
