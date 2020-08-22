import BIB_board_detection as BIBbd
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import numpy as np
import tensorflow as tf
import cv2

class Pipeline():
    def __init__(self):
        # Load box detector model
        self.box_detector = BIBbd.bib_box_detection("draco/models/BIB_board_detection/export_full/frozen_inference_graph.pb")
        
        # Load BIB detector model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = './draco/models/BIB_recognition/weights/transformerocr.pth'
        config['device'] = 'cuda:0'
        config['predictor']['beamsearch']=False
        self.BIB_detector = Predictor(config)

    def get_BIB_code(self, path):
        cv_img = cv2.imread(path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        codes = []
        # Get box from image
        res = self.box_detector.get_bib_box(cv_img, 0.9)
        print(res)
        for box in res:
            #[ 350,  977,  407, 1071] y1, x1, y2, x2
            crop_img = cv_img[box[0]:box[2], box[1]:box[3]]    
            
            pil_img = Image.fromarray(crop_img)
            code = self.BIB_detector.predict(pil_img)
            codes.append(code)
            

    def get_face(self):
        pass

if __name__=="__main__":
    img_link = "/content/H-XP (184).JPG"
    pipeline = Pipeline()
    
    pipeline.get_BIB_code(img_link)