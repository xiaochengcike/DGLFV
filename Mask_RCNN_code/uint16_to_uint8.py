#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:58:05 2018

@author: Ashiki
"""

import os
import cv2
import numpy as np
from skimage import io
from matplotlib import pylab as plt

input_file = "train_finger/labelme_json/"  
img_type = ".png"
output_file = "train_finger/cv2_mask"

midfile = os.listdir(input_file)

for f in midfile:
    
    file_name = os.path.join(input_file + f,"label" + img_type)
    img = io.imread(file_name)
    img = img.astype(np.uint8)
    
    
    ff = f.replace('_json','')
    cv2.imwrite(os.path.join(output_file, ff + img_type),img)
    