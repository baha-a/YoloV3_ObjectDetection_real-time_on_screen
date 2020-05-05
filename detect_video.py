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

# function to check colour of the detected object...
# inputs: img the captured video or image.
# inputs: coordinate refers to coordinates of the objects.
def colour_finder(img,coordinate):
 
    # corping the image to get the just the object.
    # print(coordinate[0]," =:=> ",coordinate[1],"\n-=============-\n",coordinate[2]," =:=> ",coordinate[3])
    img = img[coordinate[0]:coordinate[1],coordinate[2]:coordinate[3]]

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
     # red color

    red_lower = np.array([136,87,111],np.uint8)
    red_upper = np.array([180,255,255],np.uint8)

    # blue color

    blue_lower = np.array([99,115,150],np.uint8)
    blue_upper = np.array([110,255,255],np.uint8)

    # Green Color

    lower_green = np.array([65,60,60])
    upper_green = np.array([80,255,255])
        
    # yellow color

    yellow_lower = np.array([22,60,200],np.uint8)
    yellow_upper = np.array([60,255,255],np.uint8)

    # all color together

    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    green = cv2.inRange(hsv, lower_green, upper_green)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Morphological Transform, Dilation

    kernal = np.ones((5, 5), "uint8")

    red = cv2.dilate(red, kernal)
    res_red = cv2.bitwise_and(img, img, mask = red)

    blue = cv2.dilate(blue, kernal)
    res_blue = cv2.bitwise_and(img, img, mask = blue)

    green = cv2.dilate(green,kernal)
    res_green = cv2.bitwise_and(img,img,mask = green)
        
        
    yellow = cv2.dilate(yellow, kernal)
    res_yellow = cv2.bitwise_and(img, img, mask = yellow)

    # Tracking red
    (contours, hierarchy)=cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Rote Farbe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
            
    # Tracking blue
    (contours, hierarchy)=cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Blaue Farbe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

    # Tracking red
    (contours, hierarchy)=cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Grune Farbe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
                

    # Tracking yellow
    (contours, hierarchy)=cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Gelbe Farbe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0))

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

    try:
        vid = cv2.VideoCapture(int(pyautogui.screenshot())) #instead of inputing a camera we have input the screen of the pc.
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    fps = 0.0
    count = 0

    while True:
       
       # capturing a part of the screen with resulation (1200,1200)px 
        img = ImageGrab.grab(bbox=(50,50,900,900),all_screens=True) # x, y, w, h
        img = np.array(img)
        _ = True

        
        
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        img, objekt_dim, marra = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        
        for km_marra in range(0,marra):
            
            objekt_coordinate = np.array((objekt_dim[km_marra][0],objekt_dim[km_marra][1],objekt_dim[km_marra][2],objekt_dim[km_marra][3]))
            objekt_coordinate = objekt_coordinate.astype(int)
            

        img = cv2.putText(img, "Das Code wird von ersten Gruppen erstellt", (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        
        if FLAGS.output:
            out.write(img)
        cv2.imshow("output",img)
        if cv2.waitKey(1) == ord('q'):
         break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass



