import argparse
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')

config['weights'] = './weights/transformerocr.pth'
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
detector = Predictor(config)
# with open("box_dataset/box_annotations/test_annotations.txt", "r") as file:
    # data = file.readlines()

#for line in data[10:50]:
#    img = line.rsplit(" ", 1)[0].strip()
img = Image.open("Picture111.jpg")
display(img)
s = detector.predict(img)
print(s)
print()