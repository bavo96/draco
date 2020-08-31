import numpy as np
import tensorflow as tf
import cv2
from skimage import io
import sys

class bib_detection():
    def __init__(self,pb_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def get_bib_box(self, rgb_image, thres):
        rgb_image_shape = rgb_image.shape
        convert_shape = np.array([ rgb_image_shape[0] , rgb_image_shape[1] , rgb_image_shape[0] , rgb_image_shape[1] ])
        output_dict = self.run_inference_for_single_image(rgb_image)
        box_list = []
        for ix in range(output_dict['num_detections']):
            if output_dict['detection_scores'][ix] >= thres:
                #print(output_dict['detection_scores'][ix])
                box = (output_dict['detection_boxes'][ix]*convert_shape).astype(np.uint)
                box_list.append( box )
        return box_list

    def run_inference_for_single_image(self, image):
        graph = self.detection_graph
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

if __name__=="__main__":
    img_link = "/draco/H-XP (184).JPG"
    predictor = bib_box_detection("draco/models/BIB_board_detection/export_full/frozen_inference_graph.pb")
    #img = cv2.imread(img_link)
    img = io.imread("https://i.imgur.com/gwjK1pL.jpg")
    if len(img.shape) != 3:
        print("Can't predict image.")
        sys.exit()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = predictor.get_bib_box(img, 0.9)
    for box in res:
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test.jpg', img)