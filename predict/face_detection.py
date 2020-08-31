import tensorflow as tf
import keras
#from keras_applications.imagenet_utils import _obtain_input_shape
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import face_alignment
import numpy as np

class face_detection():
    def __init__(self,option):
        '''
        Sẽ thêm nhiều model khác vào đây theo option
        '''
        if option == "MTCNN":
            self.model = MTCNN()
    
    def get_mtcnn_face(self,image):
        face_mtcnn = self.model.detect_faces(image)
        return_list = []
        for face in face_mtcnn:
            xmin, ymin, width, height = face['box']
            xmax, ymax = xmin + width, ymin + height
            return_list.append([xmin,ymin,xmax, ymax])
        return return_list

class face_recognition():
    def __init__(self, option):
        '''
        Sẽ thêm nhiều model khác vào đây theo option
        '''
        if option == "VGGFace":
            self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') 
            self.face_align = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        
    def alignment(self,cv_img, dst, dst_w, dst_h):
        if dst_w == 96 and dst_h == 112:
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041] ], dtype=np.float32)
        elif dst_w == 112 and dst_h == 112:
            src = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041] ], dtype=np.float32)
        elif dst_w == 150 and dst_h == 150:
            src = np.array([
                [51.287415, 69.23612],
                [98.48009, 68.97509],
                [75.03375, 96.075806],
                [55.646385, 123.7038],
                [94.72754, 123.48763]], dtype=np.float32)
        elif dst_w == 160 and dst_h == 160:
            src = np.array([
                [54.706573, 73.85186],
                [105.045425, 73.573425],
                [80.036, 102.48086],
                [59.356144, 131.95071],
                [101.04271, 131.72014]], dtype=np.float32)
        elif dst_w == 224 and dst_h == 224:
            src = np.array([
                [76.589195, 103.3926],
                [147.0636, 103.0028],
                [112.0504, 143.4732],
                [83.098595, 184.731],
                [141.4598, 184.4082]], dtype=np.float32)
        else:
            return None
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        face_img = cv2.warpAffine(cv_img,M,(dst_w,dst_h), borderValue = 0.0)
        return face_img
        
    def get_aligned_face(face_image):
        landmarks = self.face_align.get_landmarks(face_image)
        if landmarks is None:
            return None
        else:
            points = landmarks[0]
            p1 = np.mean(points[36:42,:], axis=0)
            p2 = np.mean(points[42:48,:], axis=0)
            p3 = points[33,:]
            p4 = points[48,:]
            p5 = points[54,:]

            dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
            face_224x224 = alignment(face_image, dst, 224, 224)
        return face_224x224

    def get_single_face_vector(self,face_image):
        '''
        RGB input
        '''
        face_image = cv2.resize(face_image, (224,224), interpolation = cv2.INTER_NEAREST)
        aligned_face = self.get_aligned_face(face_image)
        if aligned_face != None:
            face_image = aligned_face
        face_image = np.asarray(face_image, 'float32')
        preprocessed_face_image = preprocess_input(np.expand_dims(face_image, axis=0))
        face_vector = self.model.predict(preprocessed_face_image)
        return face_vector

    def resize_batch(self,batch_image):
        batch_list = []
        for image in batch_image:
            resized = cv2.resize(image, (224,224), interpolation = cv2.INTER_NEAREST)
            batch_list.append(resized)
        return np.array(batch_list)

    def resize_align_batch(self,batch_image):
        batch_list = []
        for image in batch_image:
            resized = cv2.resize(image, (224,224), interpolation = cv2.INTER_NEAREST)
            aligned_face = self.get_aligned_face(resized)
            if aligned_face != None:
                resized = aligned_face
            batch_list.append(resized)
        return np.array(batch_list)

    def get_batch_face_vector(self,face_image_batch):
        '''
        RGB input
        '''
        resized_batch = self.resize_align_batch(face_image_batch)
        resized_batch = np.asarray(resized_batch, 'float32')
        preprocessed_face_image = preprocess_input(resized_batch)
        batch_face_vector = self.model.predict(preprocessed_face_image)
        return batch_face_vector

