from abc import ABC
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
import numpy as np
from matplotlib.patches import Rectangle, Circle
import imutils
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()


class FaceDetection(ABC):
    re_run = 0

    def detect_face_model(self, img):
        prototype_path = './face_recognition/face_Detection_model/deploy.prototxt'
        model_path = './face_recognition/face_Detection_model/res10_300x300_ssd_iter_140000.caffemodel'
        confidence_param = 0.5
        proto_path = prototype_path
        model_path = model_path
        detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        Detection_result = list()
        image = cv2.imread(img)
        image = imutils.resize(image, width=350)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (250, 250)), 1.0, (250, 250),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(image_blob)
        Detections = detector.forward()

        # loop over the Detections
        for i in range(0, Detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = Detections[0, 0, i, 2]

            # filter out weak Detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = Detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # draw the bounding box of the face along with the associated
                # probability
                text = (confidence * 100)
                Detection_result.append(text)
        return Detection_result

    def detect_face_tensor_flow(self, img):
        self.re_run += 1
        img_result = list()
        image = tf.keras.preprocessing.image.load_img(img, grayscale=False, color_mode='rgb', target_size=None,
                                                      interpolation='bilinear')
        pixels = keras.preprocessing.image.img_to_array(image)
        pyplot.imshow(image)
        ax = pyplot.gca()
        for result in detector.detect_faces(pixels):
            main_color = 'green'
            x, y, width, height = result['box']
            rect = Rectangle((x, y), width, height, fill=False, color=main_color)
            img_result.append(result['confidence'] * 100)
            ax.add_patch(rect)
            for key, value in result['keypoints'].items():
                dot = Circle(value, radius=3, color=main_color)
                ax.add_patch(dot)

        pyplot.show()
        return img_result

    ## THIS IS THE FIRST LEVEL Detection  =>>> THIS IS THE FIRST METHOD THAT THE IMAGE WOULD GO THROUGH

    ## THIS IS THE SECOND LEVEL Detection  =>>> THIS IS THE SECOND METHOD THAT THE IMAGE WOULD GO THROUGH

    def detect_face_tensor_flow_2nd(self, img):
        img_result = list()
        pixels = pyplot.imread(img)
        faces = detector.detect_faces(pixels)
        pyplot.imshow(pixels)
        ax = pyplot.gca()
        for result in faces:
            main_color = 'green'
            x, y, width, height = result['box']
            rect = Rectangle((x, y), width, height, fill=False, color=main_color)
            img_result.append(result['confidence'] * 100)
            ax.add_patch(rect)
            for key, value in result['keypoints'].items():
                dot = Circle(value, radius=3, color=main_color)
                ax.add_patch(dot)

        pyplot.show()
        return img_result
