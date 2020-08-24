import numpy as np
import tensorflow as tf
import cv2

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
                box = (output_dict['detection_boxes'][ix]*convert_shape).astype(np.uint)
                box_list.append( box  )
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
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
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
    img = cv2.imread(img_link)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = predictor.get_bib_box(img, 0.9)
    for box in res:
        print(box)
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255,0,0), 2)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test.jpg', img) 