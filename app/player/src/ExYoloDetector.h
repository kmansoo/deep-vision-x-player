#pragma once

#include <string>

#include "opencv2/core.hpp"

#include "darknet-cpp/include/darknet.h"

class ExYoloDetector {
public:
  ExYoloDetector();

private:
  bool convert_mat_to_image(cv::Mat& src_img, image& dest_img, int channels);
  bool convert_image_to_mat(image& src_img, cv::Mat& dest_img, int channels);

  bool prediction_impl(image& dest_img);

public:
  bool init(
        const std::string& cfg_file, 
        const std::string& weight_file,
        const std::string& data_cfg_file,
        float thresh,
        float hire_thresh,
        int is_fullscreen);

  bool predict(const std::string& filename);
  bool predict(cv::Mat& src_img);

private:
  float thresh_;
  float hier_thresh_;

  int is_fullscreen_;

  char **names_list_;
  image **alphabet_list_;
  network *net_;

};
