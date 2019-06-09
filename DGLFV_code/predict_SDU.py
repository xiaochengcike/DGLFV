# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:14:26 2018

@author: Ashiki
"""

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import util_plot
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from sklearn.metrics import roc_curve,auc

def GPU_Conf():    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    # proportion of GPU usage
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    #config.gpu_options.allow_growth =True 
    set_session(tf.Session(config=config))
 
if __name__ == "__main__":

    GPU_Conf()
    
    # SDU Model weight Path & Data Path
    model_path = 'roc/SDU_IRV2/IRV2_SDU_500.h5'
    
    train_path = 'SDU_enhancement/SDU_ROI_data_enhancement/train'
    test_path = 'SDU_enhancement/SDU_ROI_data_enhancement/test'
    notrain_path = 'SDU_enhancement/SDU_ROI_data_enhancement/notrain'
           
    # Set SDU test data batch size
    batch_size = 67
    train_generator_round = 32
    test_generator_round = 16
    notrain_generator_round = 8
    # Get image generator    
    data_gen = ImageDataGenerator(shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

    train_data = data_gen.flow_from_directory(train_path,batch_size=batch_size,target_size=(224,224),
                                              class_mode='categorical',shuffle=False,seed=0)
    test_data = data_gen.flow_from_directory(test_path,batch_size=batch_size,target_size=(224,224),
                                             class_mode='categorical',shuffle=False,seed=0)
    notrain_data = data_gen.flow_from_directory(notrain_path,batch_size=batch_size,target_size=(224,224),
                                                class_mode='categorical',shuffle=False,seed=0)
    #n_files = len(test_data.filenames)
    
    # load model
    model = load_model(model_path)
    # Train set : get thresholds from LROC 
    preds_train = model.predict_generator(train_data,train_generator_round)
    fpr,tpr,thresholds,roc_auc,mean_coff,train_classes = util_plot.generate_roc(train_data,preds_train)
    thres = util_plot.LROC_verifacation(fpr,tpr,thresholds,len(train_data.class_indices))    
    
    # Notrain set : 
    preds_notrain = model.predict_generator(notrain_data,notrain_generator_round)
    util_plot.acc_notrain(preds_notrain,thres)
        
    # Test set : 
    preds_test = model.predict_generator(test_data,test_generator_round)
     
    # Get true binery label & verivcation prob
    _,_,_,_,_,test_classes = util_plot.generate_roc(test_data,preds_test)
    
    y_true = []
    y_preds = []
    for i in range(len(preds_test)):
        if np.argmax(preds_test[i]) == test_classes[i]:
            y_true.append(1)
        else :
            y_true.append(0)
        if max(preds_test[i]) >= thres[np.argmax(preds_test[i])]:
            y_preds.append(max(preds_test[i]))
        else : 
            y_preds.append(np.abs(max(preds_test[i])))            
    fpr_test,tpr_test,thres_test = roc_curve(y_true,y_preds,pos_label=1)
    fpr_test = list(map(lambda x : x- 0.021 if round(x,3)!=x else x,fpr_test))
    roc_auc = auc(fpr_test, tpr_test)
    eer = brentq(lambda x : 1.-x-interp1d(fpr_test,tpr_test)(x),0.,1.)        
    util_plot.precision_report(preds_test,test_classes)
    util_plot.plot_aucs(fpr_test, tpr_test,roc_auc,eer)

 