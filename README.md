# YoloV3 Real-Time Object Detection on Screen With GradScreen and Tensorflow Libraries
```bash
This Project has been supported by the Turkish-German Univeristy
Supervised by Dr.-Ing. Soner Emec
Built by Cabbar Serif, Ibrahim Nemmura, Ovais Fakhani, Hasan Güzelmansur
```
### The Project on Youtube
[![IMAGE ALT TEXT](https://github.com/JabSYsEmb/Objekt_erkennung/blob/master/data/images/Thumbnail.png)](https://www.youtube.com/watch?v=P8Ia9LfaVEM&feature=youtu.be "YoloV3 Real-Time Object Detection on Screen")
## Getting started

#### Conda (Recommended)
```bash
# Tensorflow GPU
conda env create -f erste-gruppe.yml
conda activate erste-gruppe
```
#### Pip
```bash

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you haven't set it up already)
```bash
# Ubuntu 18.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```
### Downloading official pretrained weights
For Linux: Let's download official yolov3 weights pretrained on COCO dataset. 

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O weights/yolov3-tiny.weights
```
For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and yolov3-tiny [here](https://pjreddie.com/media/files/yolov3-tiny.weights) then save them to the weights folder.
