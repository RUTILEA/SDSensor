// Tests correctness of models.
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>

#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "basic_engine.h"
#include "posenet_decoder_op.h"
#include "test_utils.h"

#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

#include <stdio.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>
#include <chrono>

#include <tf_ros_detection/keypoints.h>
#include <tf_ros_detection/people.h>

using namespace std;
using namespace cv;

int model_width;
int model_height;
int model_channels;

cv::Mat image, frame;
bool is_first_image = true;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    Mat temp = cv_bridge::toCvShare(msg, "bgr8")->image;
    image = temp.clone();
  }
  catch (cv_bridge::Exception& e)
  {   
    cout << "error" << endl;
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str()); 
  }
  is_first_image = false;
}

namespace coral {

struct Keypoint {
  Keypoint(float _y, float _x, float _score) : y(_y), x(_x), score(_score) {}
  float y;
  float x;
  float score;
};

tf_ros_detection::people TestPoseNetDecoder() 
{
  

 // std::vector<uint8_t> input = GetInputFromImage(
 //     image_path,
 //     {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});
   /*   
  std::vector<uint8_t> input(input_tensor_shape[2]*input_tensor_shape[1]*3);
  for (int y = 0; y < input_tensor_shape[1]; y ++)
    {
        for (int x = 0; x < input_tensor_shape[2]; x ++)
        {
            int r = frame.at<cv::Vec3b>(y,x)[0];//*buf_ui8 ++; 
            int g = frame.at<cv::Vec3b>(y,x)[1];//*buf_ui8 ++;
            int b = frame.at<cv::Vec3b>(y,x)[2];//*buf_ui8 ++;
            int src_pos = y*input_tensor_shape[1] + x*3;//buf_ui8 ++;          // skip alpha 
            input[src_pos]   = (uint8_t)r;
            input[src_pos+1] = (uint8_t)g;
            input[src_pos+2] = (uint8_t)b;            
        }
    }*/
   
  
}

}  // namespace coral

int main(int argc, char** argv) {
  ros::init(argc,argv, "coral_test");
    ros::NodeHandle n;
    ros::NodeHandle private_nh_("~");
    
    image_transport::ImageTransport it(n);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    ros::Publisher pub_target = n.advertise<tf_ros_detection::people>("/people",10);
    ros::Rate r(20);
    
    std::string model_path = "/home/ubuntu/catkin_p3/src/test_p3/src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite";
    private_nh_.getParam("models_dir",model_path);
    
    //tf_ros_detection::people detections = coral::TestPoseNetDecoder();  
    
    // Load the model.
  
  std::string image_path = "/home/ubuntu/Downloads/i4.jpeg";
  
  //  image_path = "/home/ubuntu/Downloads/i4.bmp";
  LOG(INFO) << "Testing model: " << model_path;
  coral::BasicEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  
  //wait till we receive images
  while (ros::ok() && is_first_image) {
       	ros::spinOnce();
       	r.sleep();
  }
  
  while(n.ok()){
  
	  // Read image.
	 // frame =  imread(image_path);
	  frame = image.clone();
	  Mat frame_org = frame.clone();
	  resize(frame,frame,Size(input_tensor_shape[2],input_tensor_shape[1]));
	  cvtColor(frame, frame, COLOR_BGR2RGB);
	  frame.convertTo(frame, CV_8UC3);
	//  cout << input_tensor_shape[2] << "\t" << input_tensor_shape[1] << endl;
	  
	   std::vector<uint8_t> input(frame.data, frame.data + (frame.cols * frame.rows * frame.elemSize()));
	  CHECK(!input.empty()) << "Input image path: " << image_path;
	  // Get result.
	  chrono::steady_clock::time_point Tbegin, Tend;
	  Tbegin = chrono::steady_clock::now();
	  const std::vector<std::vector<float>>& raw_outputs = engine.RunInference(input);
	  Tend = chrono::steady_clock::now();
	  float f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
	//  cout << "Inference time " << f << endl;
	  const auto& poses = raw_outputs[0];
	  const auto& keypoint_scores = raw_outputs[1];
	  const auto& pose_scores = raw_outputs[2];
	  const auto& n_poses = raw_outputs[3];

	  std::cout << "poses: " << n_poses[0] << std::endl;
	  cvtColor(frame, frame, COLOR_RGB2BGR);
	  tf_ros_detection::people detections;
	  detections.num.data = 0;//n_poses[0];
	  
	  for (int i = 0; i < n_poses[0]; ++i)
	  {
	  	tf_ros_detection::keypoints points;
	  	if (pose_scores[i]>0.0f)
	  	{
	  		detections.num.data = detections.num.data + 1;
	  		for (int k = 0; k < 17; k++) 
	  		{
	  			float point_score = keypoint_scores[i * 17 + k];
	  			float y = poses[i * 17 * 2 + 2 * k];
	  			float x = poses[i * 17 * 2 + 2 * k + 1];
	  			//cout << "keypoint score: " << point_score << " x: " << x << " y: " << y << endl;
	  			cv::Point pt;
	  			pt.x=(int)x;   pt.y=(int)y;
	  			points.x.push_back(x/input_tensor_shape[2]);
	  			points.y.push_back(y/input_tensor_shape[1]);
	  			points.scores.push_back(point_score);
	  			circle(frame,pt,4,Scalar( 255, 255, 0 ),FILLED);
	  			pt.x=(int)(x/input_tensor_shape[2]*frame_org.cols); pt.y=(int)(y/input_tensor_shape[1]*frame_org.rows);
	  			circle(frame_org,pt,4,Scalar( 255, 255, 0 ),FILLED);
	  		}
	  		detections.persons.push_back(points);
	  	}
	  }
//	  imshow("results",frame);
//	  imshow("results_org",frame_org);
//	  waitKey(1);
	  
	  pub_target.publish(detections);
	    
	  ros::spinOnce();
	  r.sleep();
    
    }
    
    
    
}
