// followed
// https://github.com/Qengineering/TensorFlow_Lite_SSD_RPi_64-bits/blob/master/MobileNetV1.cpp

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc


#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include <tensorflow/lite/string_util.h>
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <cmath>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

typedef std::chrono::steady_clock measure;

bool is_first_image = true;
bool is_first_depth = true;
double focus_x = 0;
double focus_y = 0;
double center_x = 0;
double center_y = 0;

cv::Mat image, frame;
cv::Mat image_depth_org, image_depth;
std_msgs::Header header_org, header;

double x_mean, y_mean,z_mean,d_mean;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

//#define CAFFE

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
  //cout << "receiving" << endl;
  //imshow("received",image);
  //waitKey(1); 
}

void depth_Callback(const sensor_msgs::ImageConstPtr& ros_depth_image, const sensor_msgs::CameraInfo::ConstPtr& camera_info)
{
    header_org = ros_depth_image->header;
    // convert to OpenCV cv::Mat
    cv_bridge::CvImageConstPtr orig_depth_img;
    cv::Mat depth_image;
    try {
        if(ros_depth_image->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
        {
            orig_depth_img = cv_bridge::toCvShare (ros_depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
            depth_image = orig_depth_img->image;
        }
        else
        {
            orig_depth_img = cv_bridge::toCvShare (ros_depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
            orig_depth_img->image.convertTo(depth_image, CV_32F, 0.001);
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge Exception: %s", e.what());
        return;
    }
    // cout << "hello" << endl;
    image_depth_org = depth_image.clone();

    // imshow("depth",depth_image);
    // waitKey(1);

    focus_x = camera_info->K[0];
    focus_y = camera_info->K[4];
    center_x = camera_info->K[2];
    center_y = camera_info->K[5];
    // cout << "info x" << center_x << endl;

    is_first_depth = false;
    
}

int main(int argc, char** argv)
{
	ros::init(argc,argv, "object_detect_haar");
	ros::NodeHandle n;
	ros::NodeHandle private_nh_("~");

	ros::Rate r(20);
	
	image_transport::ImageTransport it(n);
    	image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    	image_transport::CameraSubscriber depth_sub_ = it.subscribeCamera("/camera/depth/image_rect_raw", 1, &depth_Callback);
    	
    	string model_path;
    	private_nh_.getParam("models_dir",model_path);
    	
	//wait till we receive images
    	while (ros::ok() && (is_first_image || is_first_depth)) {
	
        	ros::spinOnce();
        	r.sleep();
    	}
    	cout << "c1" << endl;
    	string faceCascadePath = "/home/ubuntu/catkin_ws/src/tf_ros_detection/nodes/haarcascade_frontalface_default.xml";
	CascadeClassifier faceCascade;
	cout << "c2" << faceCascadePath << endl;
	faceCascade.load( faceCascadePath );
   	cout << "c3" << endl;
	while(n.ok()){
	
		Mat frame = image.clone();
		Mat frame_n = frame.clone();
    	
    		int frameHeight = frame.rows;
	        int frameWidth = frame.cols;
	        
	        cvtColor( frame, frame, COLOR_BGR2GRAY );
    		equalizeHist( frame, frame );
	        
		std::vector<Rect> faces;
		faceCascade.detectMultiScale(frame, faces);

		for ( size_t i = 0; i < faces.size(); i++ )
		{
		  int x1 = faces[i].x;
		  int y1 = faces[i].y;
		  int x2 = faces[i].x + faces[i].width;
		  int y2 = faces[i].y + faces[i].height;
		  
		  cv::rectangle(frame_n, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
		}
    
		imshow("main frame",frame_n);
		waitKey(1);
    	
    	
        	ros::spinOnce();
        	r.sleep();

	}
}
