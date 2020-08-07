// followed
// https://github.com/Qengineering/TensorFlow_Lite_SSD_RPi_64-bits/blob/master/MobileNetV1.cpp

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc
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
#include <tf/transform_broadcaster.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <std_msgs/Int32.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <tf_ros_detection/TargetBB.h>
#include <tf_ros_detection/Detect.h>
#include <tf_ros_detection/people.h>
#include <tf_ros_detection/keypoints.h>
#include <tf_ros_detection/PosePass.h>

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

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

bool is_first_image = true;
bool is_first_depth = true;
double focus_x = 0;
double focus_y = 0;
double center_x = 0;
double center_y = 0;

cv::Mat image, frame;
cv::Mat image_depth_org, image_depth;
std_msgs::Header header_org, header, head_detect;

Mat detectImg;
Mat detectDep;
bool is_first_detect = true;

typedef std::chrono::steady_clock measure;

//------------------------------------------------------------------------------------
struct RGB {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};
//------------------------------------------------------------------------------------
const RGB Colors[21] = {{127,127,127} ,  // 0 background
                        {  0,  0,255} ,  // 1 aeroplane
                        {  0,255,  0} ,  // 2 bicycle
                        {255,  0,  0} ,  // 3 bird
                        {255,  0,255} ,  // 4 boat
                        {  0,255,255} ,  // 5 bottle
                        {255,255,  0} ,  // 6 bus
                        {  0,  0,127} ,  // 7 car
                        {  0,127,  0} ,  // 8 cat
                        {127,  0,  0} ,  // 9 chair
                        {127,  0,127} ,  //10 cow
                        {  0,127,127} ,  //11 diningtable
                        {127,127,  0} ,  //12 dog
                        {127,127,255} ,  //13 horse
                        {127,255,127} ,  //14 motorbike
                        {255,127,127} ,  //15 person
                        {255,127,255} ,  //16 potted plant
                        {127,255,255} ,  //17 sheep
                        {255,255,127} ,  //18 sofa
                        {  0, 91,127} ,  //19 train
                        { 91,  0,127} }; //20 tv monitor
const RGB black = {0,0,0};
//-----------------------------------------------------------------------------------------------------------------------

vector<tf_ros_detection::TargetBB> rois;
vector<tf_ros_detection::TargetBB> faces;
void detectCallback(const tf_ros_detection::DetectConstPtr& msg)
{
	head_detect = msg->header;
	rois = msg->boxes;
	
	try
	{
	    sensor_msgs::ImageConstPtr img_ptr(new sensor_msgs::Image(msg->image));
	    Mat temp = cv_bridge::toCvShare(img_ptr, "bgr8")->image;
	    detectImg = temp.clone();
	}
	catch (cv_bridge::Exception& e)
	{   
	    cout << "error" << endl;
	    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->image.encoding.c_str()); 
	}
	is_first_detect = false;
//	cout << "image encoding " << msg->image.encoding.c_str() << endl;
/*	try
	{
	    sensor_msgs::ImageConstPtr dep_ptr(new sensor_msgs::Image(msg->depth));
	    Mat temp = cv_bridge::toCvShare(dep_ptr, sensor_msgs::image_encodings::TYPE_32FC1)->image;
	    detectDep = temp.clone();
	}
	catch (cv_bridge::Exception& e)
	{   
	    cout << "error" << endl;
	    ROS_ERROR("Could not convert from '%s' to 'depth grayscale'.", msg->depth.encoding.c_str()); 
	}
	
	imshow("depth received",detectDep);
    	waitKey(1);*/
	
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

double x_mean, y_mean,z_mean,d_mean;

int person_check(cv::Rect roi)
{
    x_mean = 0;
    y_mean = 0;
    z_mean = 0;
    d_mean = 0;
    return 1;
    int pixel_count = 0;

    Mat img_depth_in = image_depth.clone();

    vector<float> x, y, z;
    float min_x=0,max_x=0,min_y=0,max_y=0,min_z=0,max_z=0;
    for(int row=roi.y; row<(roi.y+roi.height); row++)
    {
        const float* data_ptr = img_depth_in.ptr<float>(row);
        for(int col=roi.x; col<(roi.x+roi.width); col++)
        {
           const float data = data_ptr[col];
           if(data < 0.00001)
            { continue; }
           if (data>0.5)
           {
            const float z_data = data;
            const float x_data = ((float)col-center_x) * z_data / focus_x;
            const float y_data = ((float)row-center_y) * z_data / focus_y;
            x_mean += x_data;
            y_mean += y_data;
            z_mean += z_data;
            x.push_back(x_data);
            y.push_back(y_data);
            z.push_back(z_data);
            d_mean += sqrtf( x_data*x_data + y_data*y_data + z_data*z_data  );
            pixel_count++;
            
            if (x_data>max_x){max_x=x_data;} else if (x_data<min_x){min_x=x_data;}
            if (y_data>max_y){max_y=y_data;} else if (y_data<min_y){min_y=y_data;}
            if (z_data>max_z){max_z=z_data;} else if (z_data<min_z){min_z=z_data;}
           }
        }
    }    

    if(pixel_count == 0)
        return 0;

    x_mean /= pixel_count;
    y_mean /= pixel_count;
    z_mean /= pixel_count;
    d_mean /= pixel_count;
//cout << "distance" << d_mean << endl;
    float x_size = max_x - min_x;
    float y_size = max_y - min_y;
    float z_size = max_z - min_z;

//    cout << "person " << "x: " << x_size << " y: " << y_size << " z: " << z_size << " z_mean " << z_mean << endl;

    if ( fabsf(z_size-z_mean)>1.0 )
    {
        ROS_WARN("current size method not working good");
        return 1;
    }

    return 1;


}

double IOU(Rect2f box1, Rect2f box2) { // box 2 is face

		    float xA = max(box1.tl().x, box2.tl().x);
		    float yA = max(box1.tl().y, box2.tl().y);
		    float xB = min(box1.br().x, box2.br().x);
		    float yB = min(box1.br().y, box2.br().y);
		    
		    float intersectArea = abs((xB - xA) * (yB - yA));
                    //float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;
		    float faceArea = abs(box2.area());

		    return 1. * intersectArea / faceArea; // less than 1. if = 1 then completely inside
}


int main(int argc, char** argv)
{
	ros::init(argc,argv, "poeple_pose_decision");
	ros::NodeHandle n;
	ros::NodeHandle private_nh_("~");

	ros::Rate r(10);
	
    	ros::Subscriber sub_detect = n.subscribe("/detect",1,&detectCallback);
    	
    	ros::Publisher pub_danger = n.advertise<std_msgs::Int32>("/danger",1);

	// Load the model.
  	std::string model_path = "/home/ubuntu/catkin_p3/src/test_p3/src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite";
  	std::string image_path = "/home/ubuntu/Downloads/i4.jpeg";
	
    	private_nh_.getParam("models_dir",model_path);
	//private_nh_.getParam("labels_dir",label_path);
	double thresholdScore = 0.7;
    	double thresholdIOU = 0.8;
    	
    	//  image_path = "/home/ubuntu/Downloads/i4.bmp";
  	LOG(INFO) << "Testing model: " << model_path;
	coral::BasicEngine engine(model_path);
  	std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
    	
	//wait till we receive images
    	while (ros::ok() && is_first_detect) {
        	ros::spinOnce();
        	r.sleep();
    	} 
    	int fwidth = detectImg.size().width;
    	int fheight = detectImg.size().height;
    	
	while(ros::ok()){
	
		frame = detectImg.clone();
		Mat frame_org = frame.clone();
		resize(frame,frame,Size(input_tensor_shape[2],input_tensor_shape[1]));
      		//image_depth = detectDep.clone();	
      		cvtColor(frame, frame, COLOR_BGR2RGB);
	        frame.convertTo(frame, CV_8UC3);
      		
      	//	cout << input_tensor_shape[2] << "\t" << input_tensor_shape[1] << "\t frame elemsize: " << frame.elemSize() << endl;
	       
                vector<vector<float>> people_location;
                vector<int> id_filter;
                tf_ros_detection::people detections;
	  	detections.num.data = 0;//n_poses[0];
                //assuming all faces are correct // find other boxes not included in faces
             /*   if (rois.size()<6)
                	cout << "doing posenet directly" << endl;
                else
                	cout << "running posenet multiple times" << endl;                
             */   
                // For now assuming that much more than 10 people present
                for (int i=0;i<rois.size();i=i+6)
                {
                	cv::Mat frame_copy = frame.clone();
                	cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
                	for (int j=0; ( ((j+i)<rois.size()) && (j<6) ) ;j++)
                	{
                		int k = j+i;
                		sensor_msgs::RegionOfInterest ROI_k = rois[k].roi;
                        	cv::Rect rec_k(ROI_k.x_offset,ROI_k.y_offset,ROI_k.width,ROI_k.height);
                        	int score = rois[k].score.data;
                        	mask(rec_k) = 255;
                	}
                	Mat frame_roi;
                	bitwise_and(frame_copy,frame_copy,frame_roi,mask);
                	imshow("masked frame",frame_roi);
                	waitKey(1);
                	std::vector<uint8_t> input(frame_roi.data, frame_roi.data + (frame_roi.cols * frame_roi.rows * frame_roi.elemSize()));
	        
			// Get result.
		  	chrono::steady_clock::time_point Tbegin, Tend;
		  	Tbegin = chrono::steady_clock::now();
		  	const std::vector<std::vector<float>>& raw_outputs = engine.RunInference(input);
		  	Tend = chrono::steady_clock::now();
		  	float f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
		  //	cout << "Inference time " << f << endl;
		  	const auto& poses = raw_outputs[0];
		  	const auto& keypoint_scores = raw_outputs[1];
		  	const auto& pose_scores = raw_outputs[2];
		  	const auto& n_poses = raw_outputs[3];

		  	std::cout << "poses: " << n_poses[0] << std::endl;
		  	
		  	for (int l = 0; l < n_poses[0]; ++l)
			  {
			  	tf_ros_detection::keypoints points;
			  	if (pose_scores[l]>0.0f)
			  	{
			  		detections.num.data = detections.num.data + 1;
			  		for (int m = 0; m < 17; m++) 
			  		{
			  			float point_score = keypoint_scores[l * 17 + m];
			  			float y = poses[l * 17 * 2 + 2 * m];
			  			float x = poses[l * 17 * 2 + 2 * m + 1];
			  			cv::Point pt;
			  			pt.x=(int)x;   pt.y=(int)y;
			  			points.x.push_back(x/input_tensor_shape[2]);
			  			points.y.push_back(y/input_tensor_shape[1]);
			  			points.scores.push_back(point_score);
			  			//circle(frame,pt,4,Scalar( 255, 255, 0 ),FILLED);
			  			pt.x=(int)(x/input_tensor_shape[2]*frame_org.cols); pt.y=(int)(y/input_tensor_shape[1]*frame_org.rows);
			  			circle(frame_org,pt,4,Scalar( 255, 255, 0 ),FILLED);
			  		}
			  		detections.persons.push_back(points);
			  	}
			  }
                        
                	
                }                
                
                vector<vector<float>> face_location;
		vector<tf_ros_detection::keypoints> people = detections.persons;  
		for(int i=0;i<detections.num.data;i++)
	        {
	        	tf_ros_detection::keypoints person = people[i];
	        	vector<float> scores;// = person.scores;
	        	vector<int> xs;
	        	vector<int> ys;
	        	int facex_max=0; int facey_max=0;
	        	int facex_min=0; int facey_min=0;
	        	bool first = true;
	        	for (int j=0;j<person.x.size();j++)
	        	{
	        		scores.push_back(person.scores[j]);
	        		xs.push_back((int)(fwidth*person.x[j]));
	        		ys.push_back((int)(fheight*person.y[j]));
	        		if (j<5)
	        		{
	        		    if (!first) 
	        		    {
					    if (xs[j]>facex_max) facex_max=xs[j]; else if (xs[j]<facex_min) facex_min=xs[j];
					    if (ys[j]>facey_max) facey_max=ys[j]; else if (ys[j]<facey_min) facey_min=ys[j];		       
	        		    }
	        		    else
	        		    {
	        		    	facex_max = xs[j]; facex_min = xs[j];
	        		    	facey_max = ys[j]; facey_min = ys[j];
	        		    	first = false;   
	        		    }
	        		      		    
	        		}
	        	}
	        	float score_face = (scores[0]+scores[1]+scores[2]+scores[3]+scores[4])/5;
	        	float score_middle = (scores[5]+scores[6]+scores[11]+scores[12])/4;
	        	bool face = true;
	        	if (score_face>0.5)
	        	   {
	        	      int y_size = (facey_max - facey_min)/2;
	        	      int x_size = (facex_max - facex_min)/2;
	                      int size_face = (x_size>y_size) ? x_size : y_size;
	                      int center_x = (int)( ( xs[0]+xs[1]+xs[2]+xs[3]+xs[4] )/5 );
	                      int center_y = (int)( ( ys[0]+ys[1]+ys[2]+ys[3]+ys[4] )/5 );
	              
              	              cv::Rect rec((center_x-size_face),(center_y-size_face),(size_face*2),(size_face*2));
              	              //cv::rectangle(frame_org,rec,Scalar(255,0,255),3);
              	              if (person_check(rec)==1)
	        	      {    
	        	      	vector<float> pos; 
			        pos.push_back(x_mean); pos.push_back(y_mean); 
			        pos.push_back(z_mean); pos.push_back(d_mean);
			    	face_location.push_back(pos);
			    	cv::rectangle(frame_org,rec,Scalar(0,255,255),1);
			    	putText(frame_org, "face" + std::to_string((int)(score_face*100))+" "+std::to_string(d_mean), Point(rec.x, rec.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
	        	      }
	        	      else if (score_middle>0.5)
	        	      {
	        	        int y_left = abs(ys[6] - ys[12])/2;
	        	        int x_top = abs(xs[6] - xs[5])/2;
	        	        int y_right = abs(ys[5]-ys[11])/2;
	        	        int x_bottom = abs(xs[12]-xs[11])/2;
	        	        x_size = (x_top>x_bottom)? x_bottom : x_top;
	        	        y_size = (y_left>y_right)? y_right : y_left;
	                        int size_body = (x_size>y_size) ? x_size : y_size;
	                        center_x = (int)( ( xs[5]+xs[6]+xs[11]+xs[12] )/4 );
	                        center_y = (int)( ( ys[5]+ys[6]+ys[11]+ys[12] )/4 );
	              
              	                cv::Rect rec2((center_x-x_size),(center_y-y_size),(x_size*2),(y_size*2));
              	                //cv::rectangle(frame_org,rec2,Scalar(0,255,255),3);
              	                
              	                if (person_check(rec2)==1)
			        {
				      	vector<float> pos; 
					pos.push_back(x_mean); pos.push_back(y_mean); 
					pos.push_back(z_mean); pos.push_back(d_mean);
				    	face_location.push_back(pos);
				    	cv::rectangle(frame_org,rec2,Scalar(0,255,255),1);
				    	putText(frame_org, "body" + std::to_string((int)(score_middle*100))+" "+std::to_string(d_mean), Point(rec2.x, rec2.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
				}
	        	      }
	        	   }
	        	else if (score_middle>0.5)
	        	{
	        	      int y_left = abs(ys[6] - ys[12])/2;
	        	      int x_top = abs(xs[6] - xs[5])/2;
	        	      int y_right = abs(ys[5]-ys[11])/2;
	        	      int x_bottom = abs(xs[12]-xs[11])/2;
	        	      int x_size = (x_top>x_bottom)? x_bottom : x_top;
	        	      int y_size = (y_left>y_right)? y_right : y_left;
	                      int size_body = (x_size>y_size) ? x_size : y_size;
	                      int center_x = (int)( ( xs[5]+xs[6]+xs[11]+xs[12] )/4 );
	                      center_y = (int)( ( ys[5]+ys[6]+ys[11]+ys[12] )/4 );
	              
              	              cv::Rect rec2((center_x-x_size),(center_y-y_size),(x_size*2),(y_size*2));
              	              //cv::rectangle(frame_org,rec2,Scalar(0,255,255),3);
              	              if (person_check(rec2)==1)
			        {
				      	vector<float> pos; 
					pos.push_back(x_mean); pos.push_back(y_mean); 
					pos.push_back(z_mean); pos.push_back(d_mean);
				    	face_location.push_back(pos);
				    	cv::rectangle(frame_org,rec2,Scalar(0,255,255),1);
				    	putText(frame_org, "body" + std::to_string((int)(score_middle*100))+" "+std::to_string(d_mean), Point(rec2.x, rec2.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
				}
	        	}
	        	   
	        	   
	        }	
   
//              separate person boxes not included in faces
                bool danger =  false;
                for (int i=0;i<face_location.size();i++)
                {
                	for (int j=(i+1);j<face_location.size();j++)
                	{
                		float dist_rel = sqrtf( powf(face_location[i][0]-face_location[j][0],2) + powf(face_location[i][1]-face_location[j][1],2) + powf(face_location[i][2]-face_location[j][2],2) );
                		if (dist_rel<2.0f)
                		{
                		   danger = true;
                		}
                	}
                	
                	for (int j=0;j<people_location.size();j++)
                	{
                		float dist_rel = sqrtf( powf(face_location[i][0]-people_location[j][0],2) + powf(face_location[i][1]-people_location[j][1],2) + powf(face_location[i][2]-people_location[j][2],2) );
                		if (dist_rel<2.0f)
                		{
                		   danger = true;
                		}
                	}
                }

                std_msgs::Int32 msg1;
                if (danger){
                	cout << "WARN: INFECTION SPREAD" << endl;
                	msg1.data = 1;//gpioWrite(17, 0); gpioWrite(25, 1); 
                	}
                else{
                	cout << "SOCIAL DISTANCE MAINTAINED" << endl;
                	msg1.data = 2;//gpioWrite(17, 1); gpioWrite(25, 0); 
                	}
                	
                
                imshow("segment frame",frame_org);
                waitKey(1);   
                
                pub_danger.publish(msg1);
                	

		ros::spinOnce();
           	r.sleep();
	}
}
