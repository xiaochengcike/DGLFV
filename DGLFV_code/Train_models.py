# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:14:26 2018

@author: Ashiki
"""
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
import numpy as np
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session

# Define class for loss
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# Setting GPU
def GPU_Conf():    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    # proportion of GPU usage
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    #config.gpu_options.allow_growth =True 
    set_session(tf.Session(config=config))

if __name__ == "__main__":
    
    GPU_Conf()
    # Input data
#    database = "LNTU_data"    
#    database = "LNTU_enhancement"
#    database = "SDU_data"
    database = "SDU_enhancement"
    
#    data = 'LNTU_ori_data_massive'
#    data = 'LNTU_ori_data'
#    data = 'LNTU_data_enhancement'
#    data = 'LNTU_train_enhancement'
#    data = 'SDU_ROI_train'
#    data = 'SDU_DirectROI_train'
    data = 'SDU_ROI_data_enhancement'
#    data = 'SDU_ROI_train_enhancement'
        
    data_path = os.path.join(database,data)
    # Base model : IRV2,ResNet,IncepV3,Xcep,vgg16,vgg19 
    model_name = 'vgg19'
    
    # Epochs & Learning_rate
    Epochs = 500
    Learning_rate = 0.001
    
    # Label 
    Setting_data = database.split('_')[0]
    if Setting_data == 'LNTU' :       
        labels = 81   # 81 known labels + 1 generlized label for unknown label
    elif Setting_data == 'SDU' :
        labels = 401  # 400 known labels + 1 generlized label for unknown label
    else :
        print ("False database!")

    # Setting path for saving Losses and model weights
    save_path = os.path.join('roc',Setting_data + '_' + model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    train_path = os.path.join(data_path,'train')
    test_path = os.path.join(data_path,'test')
    
    # Train data for preview
    preview_path = 'preview'
    
    # Estimate about millions of images for preview, you'd better delete them before training
    if os.listdir(preview_path):
        paths = os.listdir(preview_path)
        for path in paths:
            filePath = os.path.join(preview_path,path)
            os.remove(filePath)
    
    # Image Data Generator : data enhencement (numbers)
    data_gen = ImageDataGenerator( #rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
                
    data = data_gen.flow_from_directory(train_path,batch_size=32,target_size=(224, 224),
                                        save_to_dir=preview_path,
                                        save_prefix='finger', save_format='bmp',
                                        class_mode='categorical')
    
    val_data = data_gen.flow_from_directory(test_path,batch_size=32,target_size=(224,224),
                                            class_mode='categorical')
    
    # Basic models for DGLFV, several models can be selected : 
    # InceptionV3,VGG16,VGG19,ResNet50,Inception-ResNetV2,Xception
    if model_name == 'IRV2':        
        base_model = InceptionResNetV2(include_top=False, weights='imagenet')
    elif model_name == 'IncepV3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif model_name == 'vgg19':
        base_model = VGG19(weights='imagenet', include_top=False)
    elif model_name == 'ResNet':
        base_model = ResNet50(include_top=False, weights='imagenet')    
    elif model_name == 'Xcep':
        base_model = Xception(weights='imagenet', include_top=False)
    else :
        print ("Load model error ! ")
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dense(4096, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(labels, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    
    # Forzen layers 
    # print base_model.layers
    # print (len(base_model.layers))
     
    #for layer in base_model.layers[:450]:
    #    layer.trainable = False
    #for layer in base_model.layers[450:]:
    #    layer.trainable = True
    
    history1 = LossHistory()
    model.compile(optimizer=SGD(lr=Learning_rate,momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    history = model.fit_generator(data,samples_per_epoch=320,
                                  nb_epoch=Epochs,
                                  validation_data=val_data,
                                  class_weight='auto',
                                  validation_steps=20,
                                  callbacks=[history1]                   
                                  )     
    # Save model
    model_save = os.path.join(save_path,Setting_data+'_'+model_name+'_'+str(Epochs)+'.h5')
    model.save(model_save)
    
    # Save loss and acc    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']     
    np.save(os.path.join(save_path,Setting_data+'_'+model_name+'_'+'acc.npy'),acc)
    np.save(os.path.join(save_path,Setting_data+'_'+model_name+'_'+'val_acc.npy'),val_acc)
    np.save(os.path.join(save_path,Setting_data+'_'+model_name+'_'+'loss.npy'),loss)
    np.save(os.path.join(save_path,Setting_data+'_'+model_name+'_'+'val_loss.npy'),val_loss)



