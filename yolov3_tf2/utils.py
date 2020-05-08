from absl import logging
import numpy as np
import tensorflow as tf
import cv2
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes, nums = outputs

    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]

    wh = np.flip(img.shape[0:2]) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='./data/fonts/futur.ttf',
                              size=(img.size[0] + img.size[1]) // 100)

    # creating an empty array for saving coordinates of objects for the colour classification process
    coordinates = np.zeros((nums,4))
    
    for i in range(nums):
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
       
        obj_color = tuple(color)       
        if(class_names[int(classes[i])] == "cup") or ( class_names[int(classes[i])] == "bottle" ):
            for t in np.linspace(0, 1, thickness):
                x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
                x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
          
            # coordinates of each object [y1 => y2 , x1 => x2]
            coordinates[i] = [ x1y1[1], x2y2[1], x1y1[0], x2y2[0] ] 
            
            obj_color = tuple(color)
            if(coordinates[i][0]>0 and coordinates[i][1]>0 and coordinates[i][2]>0 and coordinates[i][3]>0):
               # obj_color = check_colour(img,coordinates[i])
               obj_color = test(img,coordinates[i],tuple(color))
            else:
                obj_color = tuple(color)

            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]], outline=obj_color,width=10)

            confidence = '{:.2f}%'.format(objectness[i]*100)

        if(class_names[int(classes[i])] == "cup"):
            text = '{} {}'.format(("Tasse"), confidence)
            text_size = draw.textsize(text, font=font)

        elif(( class_names[int(classes[i])] == "bottle" )):
            text = '{} {}'.format(("Flasche"), confidence)
            text_size = draw.textsize(text, font=font)
        else:
            text = ""
            text_size = draw.textsize(text, font=font)

        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill="black")
        draw.text((x0, y0 - text_size[1]), text, fill=obj_color,
                              font=font)

    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    return img, coordinates, nums


def draw_labels(x, y, class_names):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 0), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

            
def check_colour(img,coordinate):
    
    coordinate = np.array((coordinate[0],coordinate[1],coordinate[2],coordinate[3])).astype(int)
    
    img = np.array(img)
    img = img[coordinate[0]:coordinate[1],coordinate[2]:coordinate[3]]
    colour = None
    print(img.shape)
    
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
        

    # all color together

    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    green = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological Transform, Dilation

    kernal = np.ones((3, 3), "uint8")

    red = cv2.dilate(red, kernal)
    res_red = cv2.bitwise_and(img, img, mask = red)

    blue = cv2.dilate(blue, kernal)
    res_blue = cv2.bitwise_and(img, img, mask = blue)

    green = cv2.dilate(green,kernal)
    res_green = cv2.bitwise_and(img,img,mask = green)
        

    # Tracking red
    (contours, hierarchy)=cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #if (colour == "red"):
        #   break
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Rote Farbe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
            colour = "red"

    # Tracking blue
    (contours, hierarchy)=cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #if (colour == "blue"):
        #   break 
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Blaue Farbe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
            colour = "blue"

    # Tracking Green
    (contours, hierarchy)=cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        #if (colour == "green"):
        #   break
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Grune Farbe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
            colour = "green"
            
    # Tracking yellow
    #(contours, hierarchy)=cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for pic, contour in enumerate(contours):
    #    area = cv2.contourArea(contour)
    #    if(area > 300):
    #        # x, y, w, h = cv2.boundingRect(contour)
    #        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #        # cv2.putText(img, "Gelbe Farbe", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0))
    #        colour = "yellow"

    return colour

def test(img,coordinate,color):
    
    coordinate = np.array((coordinate[0],coordinate[1],coordinate[2],coordinate[3])).astype(int)
    
    img = np.array(img)
    img = img[coordinate[0]:coordinate[1],coordinate[2]:coordinate[3]]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("========================")
    print("\nr layer of the image = ",img[:,:,0].mean(), "\nmax vaule is equal to {}".format(img[:,:,0].max()))
    print("\ng layer of the image = ",img[:,:,1].mean(), "\nmax vaule is equal to {}".format(img[:,:,1].max()))
    print("\nb layer of the image = ",img[:,:,2].mean(), "\nmax vaule is equal to {}".format(img[:,:,2].max()))
    print("========================")
    colour = color
    if(img[:,:,0].mean() > img[:,:,1].mean()+20):
        if(img[:,:,0].mean() > img[:,:,2].mean()+20):
            if(int(img[:,:,0].mean()) > 120):
                colour = "red"
    elif(img[:,:,1].mean() > img[:,:,0].mean()+20):
        if(img[:,:,1].mean() > img[:,:,2].mean()+20):
            if(int(img[:,:,1].mean()) > 120):
                colour = "green"
    elif(img[:,:,2].mean() > img[:,:,0].mean()+20):
        if(img[:,:,2].mean() > img[:,:,1].mean()+20):
            if(int(img[:,:,2].mean()) > 120):
                colour = "blue"
    
    return colour