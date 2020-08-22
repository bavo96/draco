import BIB_board_detection as BIBbd
import BIB_recognition as BIBr

class Pipeline():
    def __init__(self):
        # Load box detector model
        box_detector = BIBbd.bib_box_detection("draco/models/BIB_board_detection/export_full/frozen_inference_graph.pb")
        
        # Load BIB detector model
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = './draco/models/weights/transformerocr.pth'
        config['device'] = 'cuda:0'
        config['predictor']['beamsearch']=False
        BIB_detector = Predictor(config)

    def get_BIB_code(self, path):
        cv_img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get box from image
        res = self.box_detector.get_bib_box(img, 0.9)
        for box in res:
            #cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)
            crop_img = img[box[3]:box[2], box[1]:box[0]]    
            im_pil = Image.fromarray(crop_img)
            code = BIB_detector.predict(img_pil)
            print(code)            
            
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.jpg', img) 
    
    def get_face(self):
        pass

if __name__=="__main":
    img_link = ""
    pipeline = Pipeline()
    pipeline.get_BIB_code(img_link)