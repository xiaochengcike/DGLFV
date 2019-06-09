# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime 
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco import coco
from max_rectangle import find_max_rectangle 
from skimage import transform



class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
if __name__ == "__main__":	
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    
    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_shapes_0029.h5")
    
    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    
    config = InferenceConfig()
            
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    # Class names
    class_names = ['BG', 'finger']
       
    # Load a random image from the images folder
    
    file_names = next(os.walk(IMAGE_DIR))[2]
    file = random.choice(file_names)
    num = file.split('_')[0]
    image = skimage.io.imread(os.path.join(IMAGE_DIR,file))    
    
    # Run detection
    first_time = datetime.now() 
    results = model.detect([image], verbose=1)
    second_time = datetime.now() 
    print("Time:",(second_time-first_time).seconds)
    
    # Visualize results    
    r = results[0]
    # Get all result
    all_mask = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
    
    # Get box 
    box_mask = visualize.draw_rois(image, r['rois'], r['rois'],r['masks'],r['class_ids'],class_names)
#    skimage.io.imsave("./save_result/box_"+num+".png",box_mask)
    
    # Get mask
    binery_mask = visualize.display_top_masks(image,r['masks'],r['class_ids'],class_names,limit=1)    
    exact_roi = binery_mask * image[:,:,0]
    bimage = binery_mask * 255
    skimage.io.imsave("./save_result/mask_"+num+".png",bimage)
           
    # Find max rectangle from binery mask 
    image_path= "./save_result/mask_"+num+".png"	    
    coors=find_max_rectangle(image_path)
    print(coors)
    
    # Extract ROI from original images
    roi_image = exact_roi[coors[1]:coors[3],coors[0]:coors[2]]
    skimage.io.imshow(roi_image,cmap='gray')    
#    skimage.io.imsave("./save_result/roi_"+num+".png",roi_image)