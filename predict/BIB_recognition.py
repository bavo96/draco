import argparse
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

img_link = "/content/DT-CTD (13)_box_0.JPG"

# Set config
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './draco/models/weights/transformerocr.pth'
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False

# Set predictor
detector = Predictor(config)

# Predict image and show BIB code
img = Image.open(img_link)
s = detector.predict(img)
print(s)
print()
