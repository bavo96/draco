import os, sys 
from pathlib import Path
# SRC_DIR = str(Path(os.getcwd()).parent.parent)
SRC = str(Path(os.getcwd())) + "/draco/"
sys.path.insert(0, SRC)
#print(sys.path)

import BIB_board_detection as BIBbd
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import numpy as np
import tensorflow as tf
import cv2

import database.dataProcessing as dp

import conf.conf as cfg
from skimage import io

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
        self.box_detector = BIBbd.bib_box_detection("draco/models/BIB_board_detection/export_full/frozen_inference_graph.pb")
        
        # Load BIB detector model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = './draco/models/BIB_recognition/weights/transformerocr.pth'
        if gpu_available() == False:
            config['device'] = 'cpu'
        else:
            config['device'] = 'cuda:0'
        config['predictor']['beamsearch'] = False
        self.BIB_detector = Predictor(config)

    def get_BIB_code(self, cv_img):
        codes = []

        # Get box from image
        res = self.box_detector.get_bib_box(cv_img, 0.9)
        
        for box in res:
            #[ 350,  977,  407, 1071] y1, x1, y2, x2
            crop_img = cv_img[box[0]:box[2], box[1]:box[3]]    
            
            pil_img = Image.fromarray(crop_img)
            code = self.BIB_detector.predict(pil_img)
            codes.append(code)
        return codes
        
    def get_face(self):
        pass
    
    def get_human(self):
        pass

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
            codes = pipeline.get_BIB_code(image)
            
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
    
    # Phase 2: double check the BIB code based on face vectors
    