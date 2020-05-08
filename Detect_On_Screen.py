import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from PIL import ImageGrab
from PIL import ImageDraw



flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/paris.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    fps = 0.0
    #count = 0

    while True:
       
       # capturing a part of the screen with resulation (1200,1200)px 
        img = ImageGrab.grab(bbox=(50,50,950,950),all_screens=True) # x, y, w, h
        img = np.array(img)
        _ = True

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        img, objekt_dim, marra = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                    
        img = cv2.putText(img, "Das Code wird von ersten Gruppe_I erstellt "+str(int(fps))+" (fps)", (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imshow("Gruppe_I",img)
        if cv2.waitKey(1) == ord('q'):
         break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass



