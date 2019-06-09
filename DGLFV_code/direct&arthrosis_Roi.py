# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:47:13 2018

@author: Ashiki
"""

import os
import numpy as np
from skimage.filters import threshold_niblack
from skimage import io
from skimage.filters import roberts
from skimage.color import rgb2gray
import shutil
import math

ori_path = 'SDU_data/SDU_DirectROI'
data_path = 'SDU_data/SDU_arthrosisROI'

#ori_path = 'SDU_data/SDU_ROI_rotate'
#data_path = 'SDU_data/SDU_ROI_train'

#ori_path = '/home/user/test/Ashiki_RCAE_Net/enhance_SDU'
#data_path = 'SDU_enhancement/SDU_ROI_train_enhancement'

def arthrosis_roi(ori_path,img):
    img = io.imread(os.path.join(ori_path,top_path[0]))
#    img = img/255
    # it not exists output array because of argmax function's charac
    arth_locate = np.argmax(np.sum(img,axis=0))  
    left_bound = (arth_locate*2)/3
    right_bound = (img.shape[1]-arth_locate)*2/3
    arth_roi = img[:,int(left_bound):int(right_bound)]
    return arth_roi

if not os.path.exists(data_path):
    os.mkdir(data_path)
       
train_path = os.path.join(data_path,'train')
test_path = os.path.join(data_path,'test')
notrain_path = os.path.join(data_path,'notrain')

if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(notrain_path)):
    os.mkdir(train_path)
    os.mkdir(test_path)
    os.mkdir(notrain_path)

top_path = os.listdir(ori_path)
top_path.sort()

for i in range(len(top_path)):
    temp = top_path[i].split('_')
    if i < 536*6 :     # 400 train label + 136 train no label     
        doc = math.floor(i/6)
        if i < 400*6:  # 400 train label
            if int(temp[3].split('.')[0]) <= 4:
#                file_name = temp[0]+'_'+temp[1]+'_'+temp[2]
                file_name = str(doc).zfill(3)
                temp_name = os.path.join(train_path,file_name)
                if not os.path.exists(temp_name):
                    os.mkdir(temp_name)
                img = arthrosis_roi(ori_path,top_path[i])
                io.imsave(os.path.join(temp_name,top_path[i]),img)
#                shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
            else : 
#                file_name = temp[0]+'_'+temp[1]+'_'+temp[2]
                file_name = str(doc).zfill(3)
                temp_name = os.path.join(test_path,file_name)
                if not os.path.exists(temp_name):
                    os.mkdir(temp_name)
                img = arthrosis_roi(ori_path,top_path[i])
                io.imsave(os.path.join(temp_name,top_path[i]),img)
#                shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
        else :         # 136 train no label
            if int(temp[3].split('.')[0]) <= 4:
                file_name = '400'
                temp_name = os.path.join(train_path,file_name)
                if not os.path.exists(temp_name):
                    os.mkdir(temp_name)
                img = arthrosis_roi(ori_path,top_path[i])
                io.imsave(os.path.join(temp_name,top_path[i]),img)
#                shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
            else : 
                file_name = '400'
                temp_name = os.path.join(test_path,file_name)
                if not os.path.exists(temp_name):
                    os.mkdir(temp_name)
                img = arthrosis_roi(ori_path,top_path[i])
                io.imsave(os.path.join(temp_name,top_path[i]),img)
#                shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
    else :
        file_name = '400'
        temp_name = os.path.join(notrain_path,file_name)
        if not os.path.exists(temp_name):
            os.mkdir(temp_name)
        img = arthrosis_roi(ori_path,top_path[i])
        io.imsave(os.path.join(temp_name,top_path[i]),img)
#        shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
        
#for kinds in top_path :
#   hands = os.listdir(os.path.join(ori_path,kinds))
#   for h in hands:
#       orri_path = os.path.join(os.path.join(ori_path,kinds),h)
#       filenames = os.listdir(orri_path)
#       for files in filenames:
#           img = io.imread(os.path.join(orri_path,files))
#           #img = rgb2gray(img)
#           #edge_roberts = roberts(img)
#           #line = edge_roberts.sum(axis=1)
#           #line1 = np.argmax(line[:int(line.size/2)])
#           #line2 = np.argmax(line[int(line.size/2):])
#           #img = img[(line1+10):int(line2+line.size/2-10),:]
#           #nib_img = threshold_niblack(img, window_size=window_size, k=0.5)
#           #binary_img = img > nib_img
#           #binary_img = binary_img.astype('float')
##           file_index = files.replace('.bmp','').split('_')
##
##           ff = kinds+'_'+h+'_'+ files
##           io.imsave(os.path.join(open_alldata,ff),img)
#
#           file_ind = int(file_index[1])
#           if file_ind <= 4 :
#               label = kinds+'_'+h+'_'+file_index[0]
#               temp_path = os.path.join(train_path,label)
#               if not os.path.exists(temp_path):
#                   os.mkdir(temp_path)
#               io.imsave(os.path.join(temp_path,files),img)
#           else :
#               label = kinds + '_' + h + '_' + file_index[0]
#               temp_path = os.path.join(test_path, label)
#               if not os.path.exists(temp_path):
#                   os.mkdir(temp_path)
#               io.imsave(os.path.join(temp_path, files), img)
