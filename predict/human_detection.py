import numpy as np
import tensorflow as tf
import cv2
from skimage import io

class human_detection():
    def __init__(self,pb_path):
        '''
        Thử các model sau nhé 
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
        MODEL_NAME = 'mask_rcnn_resnet50_atrous_coco_2018_01_28'
        MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        '''
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(pb_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def get_box(self, rgb_image, thres):
        rgb_image_shape = rgb_image.shape
        convert_shape = np.array([ rgb_image_shape[0] , rgb_image_shape[1] , rgb_image_shape[0] , rgb_image_shape[1] ])
        output_dict = self.run_inference_for_single_image(rgb_image)
        box_list = []   
        for ix in range(output_dict['num_detections']):
            if output_dict['detection_classes'][ix] == 1 and output_dict['detection_scores'][ix] >= thres:
                #print(output_dict['detection_scores'][ix])
                box = (output_dict['detection_boxes'][ix]*convert_shape).astype(np.uint)
                box_list.append(box)
        return box_list

    def run_inference_for_single_image(self, image):
        graph = self.detection_graph
        with graph.as_default():
            with tf.compat.v1.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.compat.v1.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
                
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

if __name__ == "__main__":
    img_link = "/draco/H-XP (184).JPG"
    
    predictor = human_detection("draco/models/human_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28/frozen_inference_graph.pb")
    #img = cv2.imread(img_link)
    img = io.imread("https://i.imgur.com/gwjK1pL.jpg")
    if len(img.shape) != 3:
        print("Can't predict image.")
        sys.exit()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = predictor.get_box(img, 0.0)
    print(res)
    for box in res:
        cv2.rectangle(img, (box[0], box[2]), (box[1], box[3]), (255,0,0), 2)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('draco/test.jpg', img)