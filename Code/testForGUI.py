import warnings
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np
import skimage.draw
import skimage
import cv2
import matplotlib.pyplot as plt
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import model as modellib
from keras import models    
from classes import InferenceConfig
from transformers import pipeline
from PIL import Image

import cv2
import os

from transformers import AutoModelForImageClassification, AutoImageProcessor

def classify_image(image_path,tooth,isBinary):

    if isBinary:
        repo_name = f"model-{tooth}"
    else:
        repo_name = f"hasta-model-{tooth}"

    if not os.path.exists(repo_name):
        os.makedirs(repo_name)
    
    image_processor = AutoImageProcessor.from_pretrained(repo_name)
    model = AutoModelForImageClassification.from_pretrained(repo_name)
    pipe = pipeline("image-classification", model=model,feature_extractor=image_processor)
    image = Image.open(image_path)
    result = pipe(image)
    return result

def crop_combined_images(image, rois, class_ids, class_labels, pairs):
    combined_images = []
    combined_rois = []
    
    for pair in pairs:
        idx1 = np.where(class_ids == class_labels.index(pair[0]))[0]
        idx2 = np.where(class_ids == class_labels.index(pair[1]))[0]

        if len(idx1) > 0 and len(idx2) > 0:
            roi1 = rois[idx1[0]]
            roi2 = rois[idx2[0]]

            # Combine the two ROIs into one bounding box
            y1 = min(roi1[0], roi2[0])
            x1 = min(roi1[1], roi2[1])
            y2 = max(roi1[2], roi2[2])
            x2 = max(roi1[3], roi2[3])

            combined_rois = [y1, x1, y2, x2]
            cropped_image = image[y1:y2, x1:x2]
            
            combined_images.append((cropped_image, combined_rois, f"{pair[0]}_{pair[1]}"))
    
    return combined_images

def replace_turkish_characters(text):
    replacements = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }
    for turkish_char, ascii_char in replacements.items():
        text = text.replace(turkish_char, ascii_char)
    return text

def save_cropped_images(cropped_images, save_dir, image_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    base_name = os.path.splitext(image_name)[0]
    base_name = replace_turkish_characters(base_name)
    
    paths = []
    for img, roi, pair_name in cropped_images:
        pair_name = replace_turkish_characters(pair_name)
        save_path = os.path.join(save_dir, f"{base_name}_combined_{pair_name}.png")
        cv2.imwrite(save_path, img)
        paths.append(save_path)
    
    return paths


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def filterMasks(results,image):
    r = results[0]

    scoresDict={}
    roisDict={}
    masksDict={}
    class_idsDict=[]

    for i in range(len(r['class_ids'])):
        if r['class_ids'][i] not in scoresDict:
            scoresDict[r['class_ids'][i]]=r['scores'][i]
            roisDict[r['class_ids'][i]]=r['rois'][i]
            masksDict[r['class_ids'][i]]=r['masks'][:,:,i]
            class_idsDict.append(r['class_ids'][i])
        elif(scoresDict[r['class_ids'][i]]<r['scores'][i]):
            scoresDict[r['class_ids'][i]]=r['scores'][i]
            roisDict[r['class_ids'][i]]=r['rois'][i]
            masksDict[r['class_ids'][i]]=r['masks'][:,:,i]

    class_ids=np.array(class_idsDict)
    newRois=np.ndarray(shape=(len(roisDict),4),dtype=int)
    newMasks=np.ndarray(shape=(image.shape[0],image.shape[1],len(roisDict)),dtype=bool)
    newScores=np.ndarray(shape=(len(roisDict)),dtype=float)

    for i in range(len(roisDict)):
        newRois[i]=roisDict[class_idsDict[i]]
        newMasks[:,:,i]=masksDict[class_idsDict[i]]
        newScores[i]=scoresDict[class_idsDict[i]]
    return newRois, newMasks, class_ids, newScores

def testOnSingleImage(image_path):


    sys.path.append(r"YOUR_MASK_RCNN_PATH") #Sample  ->  C:\Users\Bilal\Desktop\Ara Proje\ara-proje\ektopik-erupsiyon\mrcnn
    model = models.Sequential()

    ROOT_DIR = os.getcwd()
    sys.path.append(ROOT_DIR)  
    SAVE_DIR = ROOT_DIR + "/output/"
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, f"segmentation-model")
        
    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference",
                                config=inference_config,
                                model_dir=DEFAULT_LOGS_DIR)

    model_path = model.find_last()

    model.load_weights(model_path, by_name=True)

    config= inference_config
    config.USE_MINI_MASK = False
   
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
     
    try:
        results = model.detect([img_arr], verbose=1)
        
    except:
        img=cv2.cvtColor(skimage.io.imread(image_path), cv2.COLOR_GRAY2RGB)
        results = model.detect([np.array(img)], verbose=1)
    
    newRois, newMasks, class_ids, newScores = filterMasks(results,img)
    ax = get_ax(1)
    visualize.display_instances(image_path, SAVE_DIR ,img, newRois, newMasks, class_ids,
                                    ['BG', '55', '65', '75', '85', '16', '26', '36', '46'], newScores,ax=ax)
    
    #if 16 and 55 are detected, crop the image of the tooths together
    pairs = [('16', '55'), ('26', '65'), ('36', '75'), ('46', '85')]
    cropped_images = crop_combined_images(img, newRois, class_ids, ['BG', '55', '65', '75', '85', '16', '26', '36', '46'], pairs)
    paths = save_cropped_images(cropped_images, SAVE_DIR, os.path.basename(image_path))

    pathSaved= SAVE_DIR + os.path.basename(image_path)
    
    if pathSaved.endswith('.bmp'):
        pathSaved=pathSaved[:-4]+'.png'
    
    return pathSaved, paths











