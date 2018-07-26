#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

#include "ExYoloDetector.h"

int main(int argc, char* argv[]) {
  //  cv::CommandLineParser parser(argc, argv, keys);

  ExYoloDetector  detector;

  if (argc < 4) {
    std::cout << "usage: ./DeepVisionXPlayer [cfg file] [data cfg file] [weight file] [phont or video file]" << std::endl;
    std::cout << std::endl;
    std::cout << "exemple: ./DeepVisionXPlayer cfg/yolov3.cfg cfg/coco.dat weights/yolov3.weights example.mp4" << std::endl;
    return 0;
  }

  int is_fullscreen = 0;
  float thresh = 0.8;
  float hier_thresh = 0.8;

  std::string cfg_file = argv[1];
  std::string data_cfg_file = argv[2];
  std::string weight_file = argv[3];
  std::string source_file = argv[4];

  if (detector.init(cfg_file, weight_file, data_cfg_file, thresh, hier_thresh, is_fullscreen) == false)
    return 0;

  if (source_file.length() == 0)
    return 0;
    
  // When the source file is an image, it must use a predicit(std::string filename) function in ExYoloDetector for image.
  if (source_file.find(".jpeg") != std::string::npos || 
      source_file.find(".jpg") != std::string::npos ||
      source_file.find(".png") != std::string::npos) {

    std::cout << "INFO: predicting image..." << std::endl;

    if (detector.predict(source_file) == false) {
      std::cout << "ERROR: There are some problem during predicting the image!" << std::endl;
      return -1;
    }

    return 0;
  }

  std::cout << "INFO: predicting video..." << std::endl;

  // When the source file is an video it must use a predicit(cv::Mat& src_img) function in ExYoloDetector for each images in video.
  cv::VideoCapture video_cap;

  video_cap.open(argv[3]);

  if (!video_cap.isOpened()) {
    std::cout << "ERROR: I can not open camera or video file." << std::endl;
    return 0;
  }

  cv::Mat source_img;

  for (;;) {
    video_cap >> source_img;

    detector.predict(source_img);

    cv::imshow("Detecting Objects on the video file", source_img);           

    cv::waitKey(1);
  }

  return 0;
}
