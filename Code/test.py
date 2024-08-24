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
import pandas as pd
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
from classes import CustomDataset,InferenceConfig

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


def randomTest():
    image_id = random.choice(dataset_val.image_ids)
    #config.USE_MINI_MASK = False
    inference_config.USE_MINI_MASK = False
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, image_id) #use_mini_mask=False)
    info = dataset_val.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                           dataset_val.image_reference(image_id)))
    results = model.detect([image], verbose=1)
    newRois, newMasks, class_ids, newScores = filterMasks(results,image)
    ax = get_ax(1)
    image_path= info["path"]
    visualize.display_instances(image_path,SAVE_DIR,image, newRois, newMasks, class_ids, 
                                dataset_val.class_names, newScores, ax=ax,
                                title="Predictions")
    
    drawPrecisionRecallCurve(results,image_id)
    display_images(np.transpose(gt_mask, [2, 0, 1]), cmap="Blues")


def saveAllOutputs(fold_num):
    dataset_val = pickle.load(open(f'dataset_val_fold{fold_num}.pkl', 'rb'))

    real_test_dir = "YOUR_VAL_DATA_DIR" #r'C:\Users\Bilal\Desktop\Ara Proje\ara-proje\ektopik-erupsiyon\dataset\val'
    image_paths = []

    for filename in os.listdir(real_test_dir):
        if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg','.bmp']:
            image_paths.append(os.path.join(real_test_dir, filename))

    for image_path in image_paths:
        filename= os.path.basename(image_path)        
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
                                    dataset_val.class_names, newScores,ax=ax)

def confusionMatrix():
    config=inference_config
    dataset = dataset_val

    gt_tot = np.array([])
    pred_tot = np.array([])

    #mAP list
    mAP_ = []
    config= inference_config
    config.USE_MINI_MASK = False

    # bgBut55 = ""
    # bgBut65 = ""
    # bgBut75 = ""
    # bgBut85 = ""

    #compute gt_tot, pred_tot and mAP for each image in the test dataset
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config, image_id)#, #use_mini_mask=False)
        
        info = dataset.image_info[image_id]

        # Run the model
        results = model.detect([image], verbose=1)
        newRois, newMasks, class_ids, newScores = filterMasks(results,image)

        #compute gt_tot and pred_tot
        gt, pred = utils.gt_pred_lists(gt_class_id, gt_bbox, class_ids, newRois)
        gt_tot = np.append(gt_tot, gt)
        pred_tot = np.append(pred_tot, pred)

        #travel in gt and pred lists and find image paths that predicted 55 or 65 or 75 or 85 but it is actually bg
        # for i in range(len(gt)):
        #     if(gt[i]==0 and pred[i]!=0):
        #         if(pred[i]==1):
        #             bgBut55+=info["path"]+"\n"
        #         elif(pred[i]==2):
        #             bgBut65+=info["path"]+"\n"
        #         elif(pred[i]==3):
        #             bgBut75+=info["path"]+"\n"
        #         elif(pred[i]==4):
        #             bgBut85+=info["path"]+"\n"

        #precision_, recall_, AP_
        AP_, precision_, recall_, overlap_ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            newRois, class_ids, newScores, newMasks)
        #check if the vectors len are equal
        print("the actual len of the gt vect is : ", len(gt_tot))
        print("the actual len of the pred vect is : ", len(pred_tot))

        mAP_.append(AP_)
        print("Average precision of this image : ",AP_)
        print("The actual mean average precision for the whole images", sum(mAP_)/len(mAP_))
        #print("Ground truth object : "+dataset.class_names[gt])

        #print("Predicted object : "+dataset.class_names[pred])
        # for j in range(len(dataset.class_names[gt])):
            # print("Ground truth object : "+j)
            
    #save the paths of the images that predicted 55 or 65 or 75 or 85 but it is actually bg
    # with open("output/bgBut55.txt", "w") as output:
    #     output.write(bgBut55)
    # with open("output/bgBut65.txt", "w") as output:
    #     output.write(bgBut65)
    # with open("output/bgBut75.txt", "w") as output:
    #     output.write(bgBut75)
    # with open("output/bgBut85.txt", "w") as output:
    #     output.write(bgBut85)
    
    gt_tot=gt_tot.astype(int)
    
    pred_tot=pred_tot.astype(int)
    #save the vectors of gt and pred

    save_dir = "output"

    gt_pred_tot_json = {"gt_tot" : gt_tot, "pred_tot" : pred_tot}
    df = pd.DataFrame(gt_pred_tot_json)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_json(os.path.join(save_dir,"gt_pred_test.json"))

    gt_tot = np.array([])
    pred_tot = np.array([])
    #json 2 elements. First is named gt_tot and the second is named pred_tot. First element contains key value pairs. Add values of gt tot to gt_tot array. Add values of pred tot to pred_tot array.
    gt_pred_tot_json = json.load(open("output/gt_pred_test.json"))
    counter1=0
    counter2=0
    for key in gt_pred_tot_json["gt_tot"]:
        counter1+=1
        gt_tot = np.append(gt_tot, gt_pred_tot_json["gt_tot"][key])
    for key in gt_pred_tot_json["pred_tot"]:
        counter2+=1
        pred_tot = np.append(pred_tot, gt_pred_tot_json["pred_tot"][key])

    print("the actual len of the gt vect is : ", len(gt_tot))
    print("the actual len of the pred vect is : ", len(pred_tot))
    print("counter1 : ",counter1)
    print("counter2 : ",counter2)
    
    global all_gt
    global all_pred
    all_gt = np.append(all_gt,gt_tot)
    all_pred = np.append(all_pred,pred_tot)

    tp,fp,fn,tn=utils.plot_confusion_matrix_from_data(gt_tot,pred_tot,columns=["0","55","65","75","85","16","26","36","46"],fz=18, figsize=(16,16), lw=0.5)

    # print("tp for each class :",tp)
    # print("fp for each class :",fp)
    # print("fn for each class :",fn)
    # print("tn for each class :",tn)

    # #eliminate the background class from tps fns and fns lists since it doesn't concern us anymore :
    # del tp[0]
    # del fp[0]
    # del fn[0]
    # del tn[0]

    # print("\n########################\n")
    # print("tp for each class :",tp)
    # print("fp for each class :",fp)
    # print("fn for each class :",fn)    
    # print("tn for each class :",tn)

def drawPrecisionRecallCurve(results,image_id):
    config= inference_config
    config.USE_MINI_MASK = False
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config, image_id)
    newRois, newMasks, class_ids, newScores = filterMasks(results,image)
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                          newRois, class_ids, newScores, newMasks,iou_threshold=0.5)
    
    print(precisions)
    print(recalls)
    print(AP)
    visualize.plot_precision_recall(AP, precisions, recalls)

def calculate_mAP():
    
    #path = os.getcwd()
    #model_tar = "nuclei_datasets.tar.gz"
    #data_path = os.path.join(path + '/dataset')
    #model_path = model.find_last()
    #model_path = os.path.join(path + '\logs\object20240406T1826')
    #weights_path = os.path.join(model_path + '/mask_rcnn_object_0300.h5') #My weights file

    config=inference_config
    config.USE_MINI_MASK = False
    dataset = dataset_val

    # with tf.device(DEVICE):
    #     model = modellib.MaskRCNN(mode="inference", model_dir=DEFAULT_LOGS_DIR, config=config)

    def compute_batch_ap(image_ids):
        APs = []

        for image_id in image_ids:
            # Load image
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id)
            # Run object detection
            results = model.detect([image], verbose=1)
            newRois, newMasks, class_ids, newScores = filterMasks(results,image)
            
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, newRois, class_ids, newScores, newMasks)
            AP = 1 - AP
            APs.append(AP)
            
        return APs, precisions, recalls
    

    #dataset.load_nucleus(data_path, 'val')
    #dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    

    image_ids = np.random.choice(dataset.image_ids, 25)
    APs, precisions, recalls = compute_batch_ap(dataset.image_ids)
    
    print("mAP @ IoU=50: ", APs)
    print(precisions)
    print(recalls)

    AP = np.mean(APs)
    print("mAP: ", AP)
    visualize.plot_precision_recall(AP, precisions, recalls)
    plt.show()
            





# Creates a session with device placement logs
config=tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

sys.path.append("YOUR_MASK_RCNN_PATH") # Sample -> "C:\Users\Bilal\Desktop\Ara Proje\ara-proje\ektopik-erupsiyon\mrcnn"
model = models.Sequential()

# Root directory of the project
#ROOT_DIR = "D:\MRCNN_tensorflow2.7_env\Mask-RCNN"
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

all_gt = np.array([])
all_pred = np.array([])
for i in range(5):
    # Directory to save logs and model checkpoints, if not provided
    # through the command line argument --logs
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, f"logs/fold{i+1}")

    SAVE_DIR = ROOT_DIR + "/test/"

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                            config=inference_config,
                            model_dir=DEFAULT_LOGS_DIR)

    model_path = model.find_last()
    #model_path = 'logs\object20240427T1142\mask_rcnn_object_0010.h5'

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    dataset_val = pickle.load(open(f'dataset_val_fold{i+1}.pkl', 'rb'))
    dataset_train= pickle.load(open(f'dataset_train_fold{i+1}.pkl', 'rb'))

    #saveAllOutputs(fold_num=i+1)
    confusionMatrix()
    #drawPrecisionRecallCurve()
    #randomTest()
    #calculate_mAP()
utils.plot_confusion_matrix_from_data(all_gt,all_pred,columns=["0","55","65","75","85","16","26","36","46"],fz=18, figsize=(16,16), lw=0.5)