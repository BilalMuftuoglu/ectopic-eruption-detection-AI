import pickle
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from mrcnn import utils
from matplotlib.figure import Figure 
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
from tensorflow import keras
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from keras import models    

#CLASSES FOR TRAINING PURPOSES
class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # Background + 55, 65, 75, 85, 16, 26, 36, 46
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset,foldNum):

        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("object", 1, "55")
        self.add_class("object", 2, "65")
        self.add_class("object", 3, "75")
        self.add_class("object", 4, "85")
        self.add_class("object", 5, "16")
        self.add_class("object", 6, "26")
        self.add_class("object", 7, "36")
        self.add_class("object", 8, "46")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations for the given fold
        annotations = list(json.load(open(rf'kfold/fold{foldNum}/{subset}/json/birlesik_veri.json')))

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['outputs']['object']]
    
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            
            polygons = [r['polygon'] for r in a['outputs']['object']]
            # polygons = []
            # for r in a['outputs']['object']:
            #     if(r['name'] == "55" or r['name'] == "65" or r['name'] == "75" or r['name'] == "85"):
            #         polygons.append(r['polygon'])
              
            
            objects = [s['name'] for s in a['outputs']['object']]
            # objects=[]
            # for s in a['outputs']['object']:
            #     if(s['name'] == "55" or s['name'] == "65" or s['name'] == "75" or s['name'] == "85"):
            #         objects.append(s['name'])
            
        
            name_dict = {"55": 1,"65": 2,"75":3,"85":4,"16":5,"26":6,"36":7,"46":8}
            #name_dict = {"55": 1,"65": 2,"75":3,"85":4}

            num_ids = [name_dict[a] for a in objects]
            # num_ids = []
            # for el in objects:
            #   if el in name_dict:
            #     num_ids.append(name_dict[el])

            parcalar = a['path'].split("\\")
            # En sondaki parçayı alarak resim ismini elde edin
            resim_ismi = parcalar[-1]

            image_path = os.path.join(dataset_dir, resim_ismi)            
            image = skimage.io.imread(image_path)

            height, width = image.shape[:2]
            self.add_image(
                "object",  ## for a single class just add the name here
                image_id= resim_ismi,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )
      

    def load_mask(self, image_id):
        
        """Generate instance masks for an image.
       Returns:<<<<
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        
        # Convert polygons to a bitmap mask of shape
       
        info = self.image_info[image_id]
        print(info["path"])
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            listOfX= [v for k,v in info["polygons"][i].items() if 'x' in k]
        
            listOfY= [v for k,v in info["polygons"][i].items() if 'y' in k]
          
            # Get indexes of pixels inside the polygon and set them to 1

            rr, cc = skimage.draw.polygon(listOfY, listOfX)
            mask[rr, cc, i] = 1        

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class InferenceConfig(Config):
    NAME = "object"

    NUM_CLASSES = 1 + 8  # Background + 55, 65, 75, 85, 16, 26, 36, 46
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.9

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3
