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
import numpy as np
import cv2
import faiss
import logging
from collections import Counter
logging.getLogger('tensorflow').disabled = True

# draco libs
import database.dataProcessing as dp
import conf.conf as cfg

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
        if cfg.DEVICE == 'cpu':
            config['device'] = 'cpu'
        else:
            config['device'] = 'cuda:0'
        config['predictor']['beamsearch'] = False
        self.BIB_detector = Predictor(config)

    def get_human(self, rgb_img):
        human_boxes = self.human_detector.get_box(rgb_img, 0.9)
        
        list_human_boxes = []
        for box in human_boxes:
            crop_img = rgb_img[box[0]:box[2], box[1]:box[3]]
            list_human_boxes.append(crop_img)
        
        return list_human_boxes

    def get_face(self, rgb_img):
        face_boxes = self.face_detection.get_mtcnn_face(rgb_img)
        faces = []
        for f_box in face_boxes:
            #x1, y1, x2, y2
            crop_img = rgb_img[f_box[1]:f_box[3], f_box[0]:f_box[2]]
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                continue
            face_vector = self.face_recognition.get_single_face_vector(crop_img)
            faces.append([crop_img, face_vector, np.asarray(f_box)])
        return faces

    def get_box(self, rgb_img):
        # Get box from image
        boxes = self.box_detector.get_bib_box(rgb_img, 0.9)
        return boxes

    def get_BIB_code(self, rgb_img, human_detection=True, face_recognition=True, box_detection=True):
        results = [] # [code, face]
        
        Path("draco/result/human").mkdir(parents=True, exist_ok=True)
        Path("draco/result/faces").mkdir(parents=True, exist_ok=True)
        Path("draco/result/BIB_codes").mkdir(parents=True, exist_ok=True)

        # Detect human
        human_imgs = self.get_human(rgb_img)
        for i, human in enumerate(human_imgs):
            print("human:", i)
            cv_human = cv2.cvtColor(human, cv2.COLOR_RGB2BGR)
            cv2.imwrite("draco/result/human/human{}.jpg".format(i), cv_human)

            final_face = []
            final_code = []
            
            # Get faces per human
            faces = self.get_face(human)
            
            # Get BIB code
            boxes = self.get_box(human)
            
            if boxes: # If BIB box exists
                for l, bib_box in enumerate(boxes):
                    print("box:", l)
                    #BIB box: y1, x1, y2, x2
                    #face box: #x1, y1, x2, y2
                    
                    # Detect BIB code from box
                    crop_box = human[bib_box[0]:bib_box[2], bib_box[1]:bib_box[3]]
                    cv_box = cv2.cvtColor(crop_box, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("draco/result/BIB_codes/BIB_codes{}_{}.jpg".format(i, l), cv_box)
                    pil_box = Image.fromarray(crop_box)
                    code = self.BIB_detector.predict(pil_box)
                    if code:
                        final_code = [code, bib_box]

                        for k, face in enumerate(faces):
                            face_center_x = (face[2][0] + face[2][2]) / 2
                            if face_center_x > bib_box[1] and face_center_x < bib_box[3]:
                                final_face = [face[1], face[2]]
                                cv_face = cv2.cvtColor(face[0], cv2.COLOR_RGB2BGR)
                                cv2.imwrite("draco/result/faces/faces{}_{}.jpg".format(i, k), cv_face)
                                break
                    
            elif len(faces) == 1: # If no BIB boxes but face exist
                face = faces[0]
                final_face = [face[1], face[2]]
                cv_face = cv2.cvtColor(face[0], cv2.COLOR_RGB2BGR)
                cv2.imwrite("draco/result/faces/faces{}.jpg".format(i), cv_face)

            results.append([final_code, final_face]) # return face box and bib box also
        return results

def phase1():
    pipeline = Pipeline()
    dataProcessing = dp.dataStructure()
    query = open("draco/database/query_image_url.txt", "r").read()

    # Get image urls
    data = dataProcessing.get_data_mysql(query, cfg.MYSQL)

    # Phase 1: BIB recognition and validate with backend side
    for batch in data:
        for img in batch:
            print("Predict codes.")
            print(img['url'])
            rgb_image = io.imread(img['url'])
            if rgb_image.shape[2] != 3:
                print("Can't predict image {} with color space {}.".format(img['url'], rgb_image.shape[2]))
                continue
            
            results = pipeline.get_BIB_code(rgb_image) # [code, face]

            print("Add codes to DB.")
            
            for person in results:
                if not person[0] and not person[1]:
                    continue
                res = {}
                res['image_id'] = img['image_id']

                # Check BIB code
                if person[0]:
                    cond = {}
                    cond['bib_code'] = person[0][0]
                    data_validation = dataProcessing.check_data_exist(cfg.MYSQL, cfg.DB_MYSQL_CANDIDATE_TABLE, cond)
                    
                    if data_validation:
                        res['bib_code'] = "{}".format(person[0][0])
                        
                        res['bib_box'] = person[0][1].tobytes()
                        

                # Check face vector
                if person[1]:
                    res['face_vector'] = person[1][0].tobytes()
                    
                    res['face_box'] = person[1][1].tobytes()
                dataProcessing.write_data_mysql(res, cfg.MYSQL, cfg.DB_MYSQL_PREDICTION_TABLE)

def phase2():
    dataProcessing = dp.dataStructure()

    # Phase 2: double check the BIB code based on face vectors
    # 1. Read data from BIB_prediction (ID+face_vector) to RAM
    query = open("draco/database/query_face_vector.txt", "r").read()
    predictions = dataProcessing.get_data_mysql(query, cfg.MYSQL)
    face_vectors = []
    list_predictions = []
    
    for batch in predictions:
        for pred in batch:
            if pred['face_vector']:
                list_predictions.append(pred)
                np_face_vector = np.frombuffer(pred['face_vector'],dtype=np.float32)
                face_vectors.append(np_face_vector)
    face_vectors = np.asarray(face_vectors)
    face_vectors = face_vectors.astype('float32')

    # 2. Use faiss to search for top k vector, choose the most frequent BIB code
    dim = 2048
    k = 5
    index = faiss.IndexFlatL2(dim)
    #print(index.is_trained)
    index.add(face_vectors)
    print(index.ntotal)
    D, I = index.search(face_vectors, k)
    for i, vector in enumerate(I):
        print("Checking {}".format(list_predictions[i]['id']))
        topK_BIB_codes = [list_predictions[idx]['bib_code'] for idx in vector]
        print(topK_BIB_codes)
        freq = Counter(topK_BIB_codes)
        new_code, occurence = freq.most_common(1)[0]
        if occurence > 2:
            print("{} code is {}".format(list_predictions[i]['id'], new_code))
            where_condition = {}
            where_condition['id'] = list_predictions[i]['id']
            update_data = {}
            update_data['validation_bib_code'] = "{}".format(new_code)
            dataProcessing.update_data_mysql(update_data, cfg.MYSQL, cfg.DB_MYSQL_PREDICTION_TABLE, where_condition)

if __name__=="__main__":
    phase1()
    phase2()
     