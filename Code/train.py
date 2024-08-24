
# Her diş için ayrı ayrı bulunan json dosyalarını birleştirme, bu işlem yapılmadan eğitim gerçekleştirilemez.

# import os
# import json

# # Klasördeki JSON dosyalarını bulun
# klasor_yolu = 'ektopik-erupsiyon/dataset/val/val-json'  # Klasörünüzün yolunu buraya girin
# json_verileri = []

# for dosya in os.listdir(klasor_yolu):
#     if dosya.endswith('.json'):
#         with open(os.path.join(klasor_yolu, dosya), 'r') as dosya_oku:
#             veri = json.load(dosya_oku)
#             json_verileri.append(veri)

# # JSON verilerini birleştirin
# birlesik_json = json.dumps(json_verileri)

# # Birleştirilmiş JSON'u yeni bir dosyaya yazın
# with open('ektopik-erupsiyon/dataset/val/val-json/birlesik_veri.json', 'w') as cikti_dosyasi:
#     cikti_dosyasi.write(birlesik_json)

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
from classes import CustomConfig, CustomDataset

def train(model,foldNum):
    
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(f"kfold/fold{foldNum}", "train",foldNum)
    dataset_train.prepare()
    print("Train dataset prepared.\n")
    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(f"kfold/fold{foldNum}", "val",foldNum)
    dataset_val.prepare()
    print("Test dataset prepared.\n")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')

    return dataset_val,dataset_train

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Creates a session with device placement logs
config=tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

sys.path.append("YOUR_MASK_RCNN_PATH") # "C:\Users\Bilal\Desktop\Ara Proje\ara-proje\ektopik-erupsiyon\mrcnn
model = models.Sequential()

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5") 


# K-Fold Cross Validation
for i in range(5):
    # Directory to save logs and model checkpoints, if not provided
    # through the command line argument --logs    
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, f"logs/fold{i+1}")
    config = CustomConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                                    model_dir=DEFAULT_LOGS_DIR)

    weights_path = COCO_WEIGHTS_PATH

    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)

    model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
    
    dataset_val,dataset_train=train(model,i+1)

    # Save the dataset to a file to use it in testing
    with open(f'dataset_val_fold{i+1}.pkl', 'wb') as f:
        pickle.dump(dataset_val, f)
    with open(f'dataset_train_fold{i+1}.pkl', 'wb') as f:
        pickle.dump(dataset_train, f)