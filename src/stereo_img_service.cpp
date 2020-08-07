//major problem comes up sometimes
#include "ros/ros.h"
#include <nodelet/loader.h>
#include <image_proc/advertisement_checker.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_geometry/stereo_camera_model.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <iostream>

#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <dynamic_reconfigure/server.h>
#include <tf_ros_detection/StereoConfig.h>

#include <stereo_img_service.h>

#include "tf_ros_detection/StereoDepth.h"

using namespace std;
using namespace cv;
using namespace sensor_msgs;
using namespace stereo_msgs;
using namespace Eigen;

bool _init_stereo = false;

void matcher_set_params()
{
    if (config_.prefilter_size % 2 == 0)
      config_.prefilter_size = config_.prefilter_size + 1;
    if (config_.prefilter_size>255)
      config_.prefilter_size = 255;
    if (config_.disparity_range % 16 != 0)
      config_.disparity_range = config_.disparity_range - config_.disparity_range%16;
    if (config_.correlation_window_size%2==0)
      config_.correlation_window_size = config_.correlation_window_size + 1;
    if (config_.correlation_window_size>255)
      config_.correlation_window_size = 255;
    if (config_.stereo_algorithm==0)//BM
    {
    	// block_matcher_.state->preFilterType = config_.prefilter_size;
    	block_matcher_->setPreFilterSize(config_.prefilter_size);
    	block_matcher_->setPreFilterCap(config_.prefilter_cap);
    	block_matcher_->setBlockSize(config_.correlation_window_size);//setSADWindowSize
    	block_matcher_->setMinDisparity(config_.min_disparity);
    	block_matcher_->setNumDisparities(config_.disparity_range);
    	block_matcher_->setTextureThreshold(config_.texture_threshold);
    	block_matcher_->setUniquenessRatio(config_.uniqueness_ratio);
    	block_matcher_->setSpeckleRange(config_.speckle_range);
    	block_matcher_->setSpeckleWindowSize(config_.speckle_size);
    	block_matcher_->setDisp12MaxDiff(config_.disp12MaxDiff);
    }
    else
    {
    	sg_block_matcher_->setMinDisparity(config_.min_disparity);
    	sg_block_matcher_->setNumDisparities(config_.disparity_range);
    	sg_block_matcher_->setBlockSize(config_.correlation_window_size);
    	sg_block_matcher_->setPreFilterCap(config_.prefilter_cap);
    	sg_block_matcher_->setUniquenessRatio(config_.uniqueness_ratio);
    	sg_block_matcher_->setP1(config_.P1);
    	sg_block_matcher_->setP2(config_.P2);
    	sg_block_matcher_->setSpeckleWindowSize(config_.speckle_size);
    	sg_block_matcher_->setSpeckleRange(config_.speckle_range);
    	sg_block_matcher_->setDisp12MaxDiff(config_.disp12MaxDiff);
    	sg_block_matcher_->setMode(config_.fullDP);
    }
    _init_stereo = true;
}

void callback(Config &config, uint32_t level)
{
	  config_ = config;
    matcher_set_params();
}

bool depth_map(tf_ros_detection::StereoDepth::Request  &req,
         tf_ros_detection::StereoDepth::Response &res)
{
   // Mat left_frame = req.left_image;
   // Mat left_frame = cv_bridge::toCvShare(req.left_image, "bgr8")->image;
   // Mat right_frame = cv_bridge::toCvShare(req.right_image, "bgr8")->image;

   sensor_msgs::ImageConstPtr img_ptr_left( new sensor_msgs::Image( req.left_image ) );
   cv::Mat left_frame = cv_bridge::toCvShare(img_ptr_left, "bgr8")->image;
   sensor_msgs::ImageConstPtr img_ptr_right( new sensor_msgs::Image( req.right_image ) );
   cv::Mat right_frame = cv_bridge::toCvShare(img_ptr_right, "bgr8")->image;

   sensor_msgs::CameraInfo left_info_msg = req.left_camera_info;
   sensor_msgs::CameraInfo right_info_msg = req.right_camera_info;

   // Verify camera is actually calibrated
  if (left_info_msg.K[0] == 0.0) {
    //ROS_INFO(30, "Rectified topic requested but camera publishing is uncalibrated");
    cout << "Major Problem" << endl;
    ros::shutdown();
    return false;
  }

  // If zero distortion, just pass the message along
  bool zero_distortion = true;
  for (size_t i = 0; i < left_info_msg.D.size(); ++i)
  {
    if ((left_info_msg.D[i] != 0.0) && (right_info_msg.D[i] != 0.0))
    {
      zero_distortion = false;
      break;
    }
  }

  // This will be true if D is empty/zero sized
  // if (zero_distortion)
  // {
  //   pub_rect_.publish(image_msg);
  //   return;
  // }
  
  // Update the camera model
  left_model_.fromCameraInfo(left_info_msg);
  right_model_.fromCameraInfo(right_info_msg);
  model_.fromCameraInfo(left_info_msg, right_info_msg);
  
 //Image Rectify Colour

  Mat left_rect, right_rect;

  left_model_.rectifyImage(left_frame, left_rect, config_.interpolation);
  right_model_.rectifyImage(right_frame, right_rect, config_.interpolation);

   //Disparity Calculation

  // Allocate new disparity image message
  DisparityImagePtr disp_msg = boost::make_shared<DisparityImage>();
  disp_msg->header         = left_info_msg.header;
  disp_msg->image.header   = left_info_msg.header;
  int border, left, wtf;
  // Compute window of (potentially) valid disparities
  if (config_.stereo_algorithm==0){
    border   = block_matcher_->getBlockSize() / 2;
    left   = block_matcher_->getNumDisparities() + block_matcher_->getMinDisparity() + border - 1;
    wtf = (block_matcher_->getMinDisparity() >= 0) ? border + block_matcher_->getMinDisparity() : std::max(border, -block_matcher_->getMinDisparity());
  }
  else
  {
    border   = sg_block_matcher_->getBlockSize() / 2;
    left   = sg_block_matcher_->getNumDisparities() + sg_block_matcher_->getMinDisparity() + border - 1;
    wtf = (sg_block_matcher_->getMinDisparity() >= 0) ? border + sg_block_matcher_->getMinDisparity() : std::max(border, -sg_block_matcher_->getMinDisparity());
  }

  int right  = disp_msg->image.width - 1 - wtf;
  int top    = border;
  int bottom = disp_msg->image.height - 1 - border;
  disp_msg->valid_window.x_offset = left;
  disp_msg->valid_window.y_offset = top;
  disp_msg->valid_window.width    = right - left;
  disp_msg->valid_window.height   = bottom - top;

  // Fixed-point disparity is 16 times the true value: d = d_fp / 16.0 = x_l - x_r.
  static const int DPP = 16; // disparities per pixel
  static const double inv_dpp = 1.0 / DPP;
  
  

  cvtColor(left_rect,  left_rect,  COLOR_BGR2GRAY);
  cvtColor(right_rect, right_rect,  COLOR_BGR2GRAY);

   // cout << "error" << endl;
  // cout << typeid(left_rect).name() << endl;
  // cout << left_rect.size << endl;
  // cout << left_rect.at<float>(1,1) << endl;
  // cout << block_matcher_->getPreFilterSize() << endl;
  // cout << "use" << config_.prefilter_size << endl;
  cv::Mat_<int16_t> disparity16_;
  

  while(!_init_stereo)
  {

  }

  try
  {    
    if (config_.stereo_algorithm==0)//BM
        block_matcher_->compute(left_rect, right_rect, disparity16_);
    else
        sg_block_matcher_->compute(left_rect, right_rect, disparity16_);
  }
   catch (cv_bridge::Exception& e)
  {   
    // cout << "error" << endl;
    // cout << typeid(left_rect).name() << endl;
    // cout << left_rect.size << endl;
    // cout << left_rect.at<float>(1,1) << endl;
    // ROS_ERROR("error using compute"); 
  }
  
  // Fill in DisparityImage image data, converting to 32-bit float
  sensor_msgs::Image& dimage = disp_msg->image;
  dimage.height = disparity16_.rows;
  dimage.width = disparity16_.cols;
  dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  dimage.step = dimage.width * sizeof(float);
  dimage.data.resize(dimage.step * dimage.height);
  cv::Mat_<float> dmat(dimage.height, dimage.width, (float*)&dimage.data[0], dimage.step);
  // We convert from fixed-point to float disparity and also adjust for any x-offset between
  // the principal points: d = d_fp*inv_dpp - (cx_l - cx_r)
  disparity16_.convertTo(dmat, dmat.type(), inv_dpp, -(model_.left().cx() - model_.right().cx()));
  ROS_ASSERT(dmat.data == &dimage.data[0]);
  

  // Stereo parameters
  disp_msg->f = model_.right().fx();
  disp_msg->T = model_.baseline();


  /// @todo Window of (potentially) valid disparities

  // Disparity search range
  if (config_.stereo_algorithm==0)
  {
    disp_msg->min_disparity = block_matcher_->getMinDisparity();
    disp_msg->max_disparity = block_matcher_->getMinDisparity() + block_matcher_->getNumDisparities() - 1;
  }
  else
  {
    disp_msg->min_disparity = sg_block_matcher_->getMinDisparity();
    disp_msg->max_disparity = sg_block_matcher_->getMinDisparity() + block_matcher_->getNumDisparities() - 1; 
  }
  
  disp_msg->delta_d = inv_dpp;

  // Adjust for any x-offset between the principal points: d' = d - (cx_l - cx_r)
  float cx_l = model_.left().cx();
  float cx_r = model_.right().cx();
  cv::Mat_<float> dispimage;
  if (cx_l != cx_r) {
    cv::Mat_<float> disp_image(disp_msg->image.height, disp_msg->image.width,
                              reinterpret_cast<float*>(&disp_msg->image.data[0]),
                              disp_msg->image.step);
    dispimage = disp_image;
    cv::subtract(disp_image, cv::Scalar(cx_l - cx_r), disp_image);
  }
  
  Mat disp8;
  normalize(disparity16_, disp8, 0, 255, CV_MINMAX, CV_8U);
  // imshow("disparity", disp8);
  // waitKey(2);


  cv::Mat_<cv::Vec3b> disparity_color_;
  // Colormap and display the disparity image
    float min_disparity = disp_msg->min_disparity;
    float max_disparity = disp_msg->max_disparity;
    float multiplier = 255.0f / (max_disparity - min_disparity);

    const cv::Mat_<float> dmat2(disp_msg->image.height, disp_msg->image.width,
                               (float*)&disp_msg->image.data[0], disp_msg->image.step);
    disparity_color_.create(disp_msg->image.height, disp_msg->image.width);

    for (int row = 0; row < disparity_color_.rows; ++row) {
      const float* d = dmat2[row];
      for (int col = 0; col < disparity_color_.cols; ++col) {
        int index = (d[col] - min_disparity) * multiplier + 0.5;
        index = std::min(255, std::max(0, index));
        // Fill as BGR
        disparity_color_(row, col)[2] = colormap[3*index + 0];
        disparity_color_(row, col)[1] = colormap[3*index + 1];
        disparity_color_(row, col)[0] = colormap[3*index + 2];
      }
    }

    // imshow("disparity colour", disparity_color_);
    // waitKey(1);

    res.DisparityValue = *disp_msg;
    res.cx_r.data = cx_r;
    res.cx_l.data = cx_l;
    res.cy_l.data = model_.left().cy();
    res.cy_r.data = model_.right().cy();


   // int x = 
   // double u = left_rect.x, v = left_rect.y;
   // cv::Point3d XYZ(u + Q_(0,3), v + Q_(1,3), Q_(2,3));
   // double W = Q_(3,2)*disparity + Q_(3,3);
   // xyz = XYZ * (1.0/W);

  // res.sum = req.a + req.b;
  // ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
  // ROS_INFO("sending back response: [%ld]", (long int)res.sum);
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "stereo_img_server");
  ros::NodeHandle nh;

  ros::ServiceServer service = nh.advertiseService("get_depth_map", depth_map);
  ROS_INFO("Ready to add two ints.");

  dynamic_reconfigure::Server<tf_ros_detection::StereoConfig> server;
  dynamic_reconfigure::Server<tf_ros_detection::StereoConfig>::CallbackType f;
  f = boost::bind(&callback, _1, _2);
  server.setCallback(f);

  image_transport::Publisher pub_rect_;

  // Processing state (note: only safe because we're using single-threaded NodeHandle!)
  
  // matcher_set_params(block_matcher_, sg_block_matcher_, config_);
  matcher_set_params();

  ros::spin();

  return 0;
}
