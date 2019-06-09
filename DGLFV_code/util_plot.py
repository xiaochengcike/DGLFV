# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:58:43 2017

@author: Ashiki
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import roc_curve, auc  
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from sklearn.metrics import roc_curve, auc,classification_report 
from sklearn import preprocessing  
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy import interp
import math

def plot_training(history):
    """ show ACC and LOSS
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r.')
    
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()

def plot_preds(img, preds,labels): 
    
    plt.imshow(img)
    plt.axis('off')
    plt.figure()
    plt.barh([0, 1], preds, alpha=0.5)
    plt.yticks([0, 1], labels)
    plt.xlabel('Probability')
    plt.xlim(0,1.01)
    plt.tight_layout()
    plt.show()

def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr[100], tpr[100], label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 0.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)

def LROC_verifacation(fpr,tpr,thresholds,labels):
    
    temp = []
    index = []
    thres = []
    for i in range(len(tpr)):
        temp.append(tpr[i]-fpr[i])
    for j in range(len(temp)): 
        Youden = np.max(np.where(temp[j]==max(temp[j])))    
        index.append(Youden)
    for k in range(labels):
        thres.append(thresholds[k][index[k]])
    threshold = (np.array(thres).reshape(-1,1))-0.01   
    return threshold

def generate_roc(test_data,preds):
    
    n_class = len(test_data.class_indices)
    n_files = len(test_data.filenames)
    
    test_classes = []
    for filename in test_data.filenames:
        test_classes.append(filename.replace('.bmp','').split('/')[0])
    test_classes = sorted(map(int,test_classes))
    temp = test_classes
    for i in range(int((len(preds)/n_files))-1):
        test_classes = test_classes+temp
    
    enc = preprocessing.OneHotEncoder()
    test_classes = np.array(test_classes)
    test_classes = test_classes[np.newaxis,:].T                           
    enc.fit(test_classes)
    
    y_test = enc.transform(test_classes).toarray()
    y_pred = preds[:len(test_classes)]

    fpr = dict()
    tpr = dict()
    threshold = dict()
    eer = dict()
    roc_auc = dict()
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(n_class):
        fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:,i], y_pred[:, i],pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
        eer[i] = brentq(lambda x : 1.-x-interp1d(fpr[i],tpr[i])(x),0.,1.)  
        
        tprs.append(interp(mean_fpr, fpr[i], tpr[i]))
        tprs[-1][0] = 0.0
        
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_class    
    #mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    # eer_all = brentq(lambda x : 1.-x-interp1d(mean_fpr,mean_tpr)(x),0.,1.)
    mean_coff = dict()    
    mean_coff['mean_fpr'] = all_fpr
    mean_coff['mean_tpr'] = mean_tpr
    mean_coff['mean_auc'] = auc(mean_coff['mean_fpr'], mean_coff['mean_tpr'])
    #mean_coff['mean_auc'] = mean_auc
    #mean_coff['mean_fpr'] = mean_fpr
    #mean_coff['mean_tpr'] = mean_tpr
    return fpr,tpr,threshold,roc_auc,mean_coff,test_classes

    
def precision_report(class_preds,class_real):
    
    class_preds = class_preds[:len(class_real)]
    temp_preds = np.argmax(class_preds,axis=1)
    temp_preds = temp_preds[np.newaxis,:].T
    print(classification_report(class_real, temp_preds))

def acc_notrain(proba_preds,thres):
    
    count = 0
    non_locate = proba_preds.shape[1] - 1
    for i in range(len(proba_preds)):
        locate = np.argmax(proba_preds[i])
        if locate == non_locate :
            count += 1
        elif locate != non_locate and max(proba_preds[i]) < thres[locate] :# 
            count += 1
    ACC = float(count)/len(proba_preds)    
    print ("Nontrain Label Accuracy : %.5f" %(ACC))

def acc_common(proba_preds,class_real):
    
    proba_preds = proba_preds[:len(class_real)]
    count = 0
    for i in range(len(proba_preds)):
        if np.argmax(proba_preds[i])==int(class_real[i]):
            count += 1
    ACC = float(count)/len(proba_preds)
    print ("Accuracy : %.5f" %(ACC))

def plot_aucs(fpr,tpr,roc_auc,eer):   
    
    plt.figure()
    x = np.linspace(0,1,num=100)
    y = np.linspace(1,0,num=100)
    plt.plot(x,y,'k--',label='EER line EER = %0.6f' % eer)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 0.3])
    plt.ylim([0.80, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
def EER_function(fpr_test,tpr_test,thres_test):
    
    fnr_test = 1 - tpr_test
    eer_threshold = thres_test[np.nanargmin(np.absolute((fnr_test - fpr_test)))]
    EER = fnr_test[np.nanargmin(np.absolute((fnr_test - fpr_test)))]
    return EER,eer_threshold

def plot_mean_auc(mean_coff,eer_all):
    
    plt.figure()
    x = np.linspace(0,1,num=100)
    y = np.linspace(1,0,num=100)
    plt.plot(x,y,'k--',label='EER line EER = %0.6f' % eer_all)
    
    plt.plot(mean_coff['mean_fpr'], mean_coff['mean_tpr'], 
             label='ROC curve (area = %0.6f)' % mean_coff['mean_auc'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 0.02])
    plt.ylim([0.95, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
