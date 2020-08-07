//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//
#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

#include "UltraFace.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>\
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>

using namespace cv;
using namespace std;

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


int main(int argc, char **argv) {
 
    ros::init(argc,argv, "main");
    ros::NodeHandle n;
    ros::NodeHandle private_nh_("~");
    
    ros::Rate r(20);
    
    image_transport::ImageTransport it(n);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 

    std::string bin_path = "/home/ubuntu/catkin_ws/src/tf_ros_detection/data/version-RFB/RFB-320.bin";//argv[1];
    std::string param_path = "/home/ubuntu/catkin_ws/src/tf_ros_detection/data/version-RFB/RFB-320.param";//argv[2];
    UltraFace ultraface(bin_path, param_path, 320, 240, 1, 0.7); // config model input

        std::string image_file = "/home/ubuntu/catkin_ws/src/tf_ros_detection/data/test.jpg";
        std::cout << "Processing " << image_file << std::endl;

     //   cv::Mat frame = cv::imread(image_file);
        
        //wait till we receive images
    while (ros::ok() && is_first_image) {
        	ros::spinOnce();
        	r.sleep();
    } 

    while(ros::ok()){
  //  cout << "running" << endl;
       frame = image.clone();
       cvtColor(frame, frame, COLOR_BGR2RGB);
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
            putText(frame, std::to_string((int)(face.score*100)), Point(face.x1, face.y1+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
        }

        cv::imshow("UltraFace", frame);
        cv::waitKey(1);
      //  cv::imwrite("result.jpg", frame);
      
       ros::spinOnce();
   	r.sleep();
        
        }
destroyAllWindows();
    return 0;
}
