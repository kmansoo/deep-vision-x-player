
#include "ExYoloDetector.h"

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "darknet-cpp/include/darknet.h"
#include "darknet-cpp/include/option_list.h"
#include "darknet-cpp/include/network.h"
#include "darknet-cpp/include/utils.h"
#include "darknet-cpp/include/option_list.h"

ExYoloDetector::ExYoloDetector() : 
  is_fullscreen_(0), 
  names_list_(NULL), 
  alphabet_list_(NULL),
  net_(NULL)  {
}

bool ExYoloDetector::init(
        const std::string& cfg_file, 
        const std::string& weight_file,
        const std::string& data_cfg_file,
        float thresh,
        float hire_thresh,
        int is_fullscreen) {

  thresh_ = thresh;
  hier_thresh_ = hire_thresh;
  is_fullscreen_ = is_fullscreen;

  list *options = read_data_cfg((char *)data_cfg_file.c_str());

  if (options == NULL) {
    std::cout << "ERROR: read_data_cfg retures NULL!" << std::endl;
    return false;
  }

  std::string names_list_file = option_find_str(options, (char *)"names", (char *)"data/names.list");

  names_list_ = get_labels((char*)names_list_file.c_str());

  if (names_list_ == NULL) {
    std::cout << "ERROR: get_labels retures NULL!" << std::endl;
    return false;
  }

  alphabet_list_ = load_alphabet();

  if (alphabet_list_ == NULL) {
    std::cout << "ERROR: load_alphabet retures NULL!" << std::endl;
    return false;
  }

  net_ = load_network((char *)cfg_file.c_str(), (char *)weight_file.c_str(), 0);

  if (net_ == NULL) {
    std::cout << "ERROR: load_network retures NULL!" << std::endl;
    return false;
  }

  set_batch_network(net_, 1);
  srand(2222222);

  gpu_index = 0;

#ifdef GPU
  if (gpu_index >= 0)
    cuda_set_device(gpu_index);
#endif

  return true;
}

bool ExYoloDetector::predict(const std::string& filename) {
  image img = load_image_color((char*)filename.c_str(), 0, 0);

  if (img.data == NULL)
    return false;

  bool ret_val = prediction_impl(img);

  int nboxes = 0;
  
  detection *dets = get_network_boxes(net_, img.w, img.h, thresh_, hier_thresh_, 0, 1, &nboxes);
  
  //printf("%d\n", nboxes);
  //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
  
  layer l = net_->layers[net_->n - 1];
  float nms = .45;

  if (nms)
    do_nms_sort(dets, nboxes, l.classes, nms);
  
  draw_detections(img, dets, nboxes, thresh_, names_list_, alphabet_list_, l.classes);
  free_detections(dets, nboxes);  

  save_image_png(img, "predictions");

  free_image(img);

  return ret_val;
}

bool ExYoloDetector::predict(cv::Mat& source_img) {
  image img;

  convert_mat_to_image(source_img, img, 3);

  if (img.data == NULL)
    return false;

  bool ret_val = prediction_impl(img);

  {
    int nboxes = 0;
    
    detection *dets = get_network_boxes(net_, img.w, img.h, thresh_, hier_thresh_, 0, 1, &nboxes);
    
    layer l = net_->layers[net_->n - 1];
    float nms = .45;

    if (nms)
      do_nms_sort(dets, nboxes, l.classes, nms);
    
    {
      for (int i = 0; i < nboxes; ++i) {
        char labelstr[4096] = {0};
        int class_val = -1;

        for (int j = 0; j < l.classes; ++j) {
          if (dets[i].prob[j] > thresh_) {
            if (class_val < 0) {
              strcat(labelstr, names_list_[j]);
              class_val = j;
            }
            else {
              strcat(labelstr, ", ");
              strcat(labelstr, names_list_[j]);
            }

            std::cout << names_list_[j] << ": " << dets[i].prob[j] * 100 << "%%" << std::endl;
            //  printf("%s: %.0f%%\n", names_list_[j], dets[i].prob[j] * 100);
          }
        }

        if (class_val >= 0) {
          //  printf("%d %s: %.0f%%\n", i, names_list_[class_val], prob*100);          
          int offset = class_val * 123457 % l.classes;
          float red = get_color(2, offset, l.classes) * 255;
          float green = get_color(1, offset, l.classes) * 255;
          float blue = get_color(0, offset, l.classes) * 255;

          box b = dets[i].bbox;

          int left = (b.x - b.w / 2.) * img.w;
          int right = (b.x + b.w / 2.) * img.w;
          int top = (b.y - b.h / 2.) * img.h;
          int bot = (b.y + b.h / 2.) * img.h;

          if (left < 0)
            left = 0;

          if (right > img.w - 1)
            right = img.w - 1;

          if (top < 0)
            top = 0;

          if (bot > img.h - 1)
            bot = img.h - 1;

          //  Draw Object 
          cv::Rect rect(left, top, right - left, bot - top);
          cv::rectangle(source_img, rect, cv::Scalar(blue, green, red), 3);
          
          if (alphabet_list_)
            cv::putText(source_img, labelstr, cv::Point(left, top - 2), cv::FONT_HERSHEY_SIMPLEX , 1, cv::Scalar(blue, green, red), 2, 8);
        }
      }
    }

    free_detections(dets, nboxes);  
  }

  free_image(img);

  return ret_val;
}

bool ExYoloDetector::convert_mat_to_image(cv::Mat& src_img, image& dest_img, int channels) {
  IplImage ipltemp = src_img;

  IplImage* src = &ipltemp;

  dest_img = ipl_to_image(src);

  rgbgr_image(dest_img);

  //  save_image_png(dest_img, "converted_img");
  return true;
}

bool ExYoloDetector::prediction_impl(image& target_image) {
  image sized = letterbox_image(target_image, net_->w, net_->h);

  //image sized = resize_image(im, net->w, net->h);
  //image sized2 = resize_max(im, net->w);
  //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
  //resize_network(net, sized.w, sized.h);

  float *X = sized.data;

  double time = what_time_is_it_now();

  network_predict(net_, X);

  printf("Predicted in %f seconds.\n", what_time_is_it_now() - time);

  free_image(sized);

  return true;
}
