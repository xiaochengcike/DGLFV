# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:58:05 2018

@author: Ashiki
"""

import os
import shutil
import numpy as np
from skimage import io

#ori_path = '850nm'
#data_path = 'LNTU_data/LNTU_ori_data_massive'

ori_path = '/home/user/test/Ashiki_RCAE_Net/enhance_LNTU'
data_path = 'LNTU_enhancement/LNTU_train_enhancement'


if not os.path.exists(data_path):
    os.mkdir(data_path)
    
train_path = os.path.join(data_path,'train')
test_path = os.path.join(data_path,'test')
test_path2 = os.path.join(data_path,'notrain')

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    os.mkdir(train_path)
    os.mkdir(test_path)
    os.mkdir(test_path2)
    
h = os.listdir(ori_path)
 
for filename in h:
    label = filename.replace('.bmp','').split('_')
    #label = map(int,label) 
    label = np.array(label).astype('int')
    if label[0] <= 100:
        if label[0] < 80:
            if label[1] > 8:
                temp_tename = os.path.join(test_path,str(label[0]).zfill(3))
                if not os.path.exists(temp_tename):
                    os.mkdir(temp_tename)
                shutil.copy(os.path.join(ori_path,filename),temp_tename)
            else :
                temp_trname = os.path.join(train_path,str(label[0]).zfill(3))
                if not os.path.exists(temp_trname):
                    os.mkdir(temp_trname)
                shutil.copy(os.path.join(ori_path,filename),temp_trname)
        else :
            if label[1] > 8:
                temp_tename = os.path.join(test_path,str(80).zfill(3))
                if not os.path.exists(temp_tename):
                    os.mkdir(temp_tename)
                shutil.copy(os.path.join(ori_path,filename),temp_tename)
            else :
                temp_trname = os.path.join(train_path,str(80).zfill(3))
                if not os.path.exists(temp_trname):
                    os.mkdir(temp_trname)
                shutil.copy(os.path.join(ori_path,filename),temp_trname)
    else :
       temp_tename = os.path.join(test_path2,str(label[0]).zfill(3))
       if not os.path.exists(temp_tename):
           os.mkdir(temp_tename)
       shutil.copy(os.path.join(ori_path,filename),temp_tename) 
       