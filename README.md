# deep-vision-x-player
This player can detect objects on photo or video file. It's based on Darknet, YOLOv3, OpenCV and CUDA.

The darknet-cpp in this repository is based on https://github.com/pjreddie/darknet.
But original darknet can not support to the OpenCV 3.x.
So I have modified source file and create a build script based CMake to build with C++.

If you don't have GPUs, you can turn off GPU mode in CMakeLists.txt.

You should build with GPU option for more fast object detection.

## How to build
```bash
mkdir build
cd build
cmake ..
make
```

## How to run
DeepVisionXPlayer [cfg file] [data cfg file] [weight file] [phont or video file]

First, You already have the config file for YOLO in the cfg/ subdirectory. 
And you will have to download the pre-trained weight file here (237 MB). Or just run this:

```bash
cd {git}/deep-vision-x-player
mkdir weights
cd weights
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
```

And you have to use parameters below to run the built DeepVisionXPlayer.

```bash
cd {git}/deep-vision-x-player
./build/bin/DeepVisionXPlayer cfg/yolov3.cfg cfg/coco.dat weights/yolov3.weights example.mp4
```
