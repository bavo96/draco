import os, sys 
from pathlib import Path

SRC = str(Path(os.getcwd())) + "/draco/"
sys.path.insert(0, SRC)

# Detection packages
import BIB_board_detection as BIBbd
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import human_detection as hd
import face_detection as fd

# Python packages
from skimage import io
from PIL import Image
#import tensorflow as tf
import numpy as np
import cv2

# draco libs
import database.dataProcessing as dp
import conf.conf as cfg

import logging
logging.getLogger('tensorflow').disabled = True

def gpu_available():
    """
        Check NVIDIA with nvidia-smi command
        Returning code 0 if no error, it means NVIDIA is installed
        Other codes mean not installed
    """
    code = os.system('nvidia-smi')
    return code == 0

class Pipeline():
    def __init__(self): 
        # Load box detector model
        self.box_detector = BIBbd.bib_detection("draco/models/BIB_board_detection/export_full/frozen_inference_graph.pb")

        # Load human detector model
        self.human_detector = hd.human_detection("draco/models/human_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28/frozen_inference_graph.pb")

        # Load face extraction model
        self.face_detection = fd.face_detection('MTCNN')
        self.face_recognition = fd.face_recognition('VGGFace')

        # Load BIB detector model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = './draco/models/BIB_recognition/weights/transformerocr.pth'
        if gpu_available() == False:
            config['device'] = 'cpu'
        else:
            config['device'] = 'cuda:0'
        config['predictor']['beamsearch'] = False
        self.BIB_detector = Predictor(config)

    def get_human(self, rgb_img):
        human_boxes = self.human_detector.get_box(rgb_img, 0.3)
        
        list_human_boxes = []
        for box in human_boxes:
            crop_img = rgb_img[box[0]:box[2], box[1]:box[3]]
            list_rgb_img.append(crop_img)
        
        return list_human_boxes

    def get_face(self, rgb_img):
        face_boxes = self.face_detection.get_mtcnn_face(rgb_img)
        faces = []
        for box in face_boxes:
            crop_img = rgb_img[box[1]:box[3], box[0]:box[2]]
            face_vector = self.face_recognition.get_single_face_vector(crop_img)

            faces.append([crop_img, face_vector])
        return faces

    def get_box(self, rgb_img):
        # Get box from image
        boxes = self.box_detector.get_bib_box(rgb_img, 0.9)
        return boxes

    def get_BIB_code(self, rgb_img, human_detection=True, face_recognition=True, box_detection=True):
        codes = []
        
        Path("draco/result/human").mkdir(parents=True, exist_ok=True)
        Path("draco/result/faces").mkdir(parents=True, exist_ok=True)
        Path("draco/result/BIB_codes").mkdir(parents=True, exist_ok=True)

        # Detect human
        human_imgs = get_human(rgb_img)
        for i, human in enumerate(human_imgs):
            faces = get_face(human)
            cv2.imwrite("draco/result/human{}.jpg".format(i), human)

            # Get faces per human
            for k, face in enumerate(faces):
                cv2.imwrite("draco/result/faces{}_{}.jpg".format(i, k), face)
                    
            # Get BIB code
            boxes = self.get_box(human)
            for l, box in enumerate(boxes):
                #[ 350,  977,  407, 1071] y1, x1, y2, x2
                crop_img = rgb_img[box[0]:box[2], box[1]:box[3]]    
                cv2.imwrite("draco/result/BIB_codes{}_{}.jpg".format(i, l), crop_img)
                pil_img = Image.fromarray(crop_img)
                code = self.BIB_detector.predict(pil_img)
                codes.append(code)

        return codes

if __name__=="__main__":
    pipeline = Pipeline()
    dataProcessing = dp.dataStructure()
    query = open("draco/database/query_image_url.txt", "r").read()

    # Get image urls
    data = dataProcessing.get_data_mysql(query, cfg.MYSQL)

    # Phase 1: BIB recognition and validate with backend side
    for batch in data:
        for img in batch:
            
            image = io.imread(img['url'])
            if image.shape[2] != 3:
                print("Can't predict image {} with shape {}.".format(img['url'], image.shape[2]))
                continue
            
            l_rgb_img = pipeline.get_human(image)
            
            for i, box_img in enumerate(l_rgb_img):
                codes = pipeline.get_BIB_code(image)            
            break

            cond = {}
            for code in codes:
                cond['BIB_code'] = code
                data_existed = dataProcessing.check_data_exist(cfg.MYSQL, cfg.DB_MYSQL_PREDICTION_TABLE, cond)
                res = {}
                res['image_id'] = img['image_id']
                res['face_vector'] = "\"\"" # detect later
                res['validation_bib_code'] = "\"\""
                if data_existed:
                    res['bib_code'] = code    
                elif data_existed == False:
                    res['bib_code'] = "\"\""                
                dataProcessing.write_data_mysql(res, cfg.MYSQL, cfg.DB_MYSQL_PREDICTION_TABLE)
            
        break
    # Phase 2: double check the BIB code based on face vectors
    