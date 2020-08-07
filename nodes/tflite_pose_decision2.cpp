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
#include <std_msgs/Int32.h>
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
#include <numeric>

using namespace std;
using namespace cv;

typedef std::chrono::steady_clock measure;

bool is_first_image = true;
bool is_first_depth = true;
double focus_x = 0;
double focus_y = 0;
double center_x = 0;
double center_y = 0;

bool processed = true;

cv::Mat image, frame;
cv::Mat image_depth_org, image_depth;
std_msgs::Header header_org, header;

double x_mean, y_mean,z_mean,d_mean;

vector<tf_ros_detection::keypoints> people;
//vector<tf_ros_detection::people> people;
int num_detections;
bool is_first_pose = true;
void peopleCallback(const tf_ros_detection::peoplePtr& msg)
{	
	num_detections = msg->num.data;
	people = msg->persons;	
	is_first_pose = false;
	processed = false;
}


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
 // cout << "Image encoding" << msg->encoding.c_str() << endl;
  //cout << "receiving" << endl;
  //imshow("received",image);
  //waitKey(1); 
}
int encoding;
void depth_Callback(const sensor_msgs::ImageConstPtr& ros_depth_image, const sensor_msgs::CameraInfo::ConstPtr& camera_info)
{
//cout << "depth encoding" << ros_depth_image->encoding << endl;
    header_org = ros_depth_image->header;
    // convert to OpenCV cv::Mat
    cv_bridge::CvImageConstPtr orig_depth_img;
    cv::Mat depth_image;
    try {
        if(ros_depth_image->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
        {
            orig_depth_img = cv_bridge::toCvShare (ros_depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
            depth_image = orig_depth_img->image;
            encoding = 1;
         //   cout << encoding << endl;
        }
        else
        {
            orig_depth_img = cv_bridge::toCvShare (ros_depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
            orig_depth_img->image.convertTo(depth_image, CV_32F, 0.001);
            encoding = 2;
         //   cout << encoding << "\t" << ros_depth_image->encoding << endl;
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


int person_check(cv::Rect roi)
{
    x_mean = 0;
    y_mean = 0;
    z_mean = 0;
    d_mean = 0;
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
           if (data>0.4)
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
//        ROS_WARN("current size method not working good");
        return 1;
    }

    return 1;


}


bool getFileContent(std::string fileName, std::vector<std::string> &labels)
{
	// Open the File
	std::ifstream in(fileName.c_str());
	// Check if object is valid
	if(!in.is_open()) return false;

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size()>0) labels.push_back(str);
	}
	// Close The File
	in.close();
	return true;
}

void set_roi(sensor_msgs::RegionOfInterest *roi, cv::Rect Irect)
{
    roi->x_offset = Irect.x;
    roi->y_offset = Irect.y;
    roi->width = Irect.width;
    roi->height = Irect.height;
}

double IOU(Rect2f box1, Rect2f box2) {

    float xA = max(box1.tl().x, box2.tl().x);
    float yA = max(box1.tl().y, box2.tl().y);
    float xB = min(box1.br().x, box2.br().x);
    float yB = min(box1.br().y, box2.br().y);
    
    float intersectArea = abs((xB - xA) * (yB - yA));
    float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;

    return 1. * intersectArea / unionArea;
}

vector<int> filterBoxes(float *scores, float *boxes, double thresholdIOU, double thresholdScore, int num) 
{
           vector<int> sortIdxs(num);
           iota(sortIdxs.begin(), sortIdxs.end(), 0);
           // Create set of "bad" idxs
	    set<int> badIdxs = set<int>();
	    int i = 0;
            while (i < sortIdxs.size()) {
		if (scores[sortIdxs.at(i)] < thresholdScore)
		    badIdxs.insert(sortIdxs[i]);
		if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
		    i++;
		    continue;
		}
		float ymin=boxes[4*i  ];//ymin
	        float xmin=boxes[4*i+1];//xmin
	        float ymax=boxes[4*i+2];//ymax
	        float xmax=boxes[4*i+3];//xmax
		Rect2f box1 = Rect2f(Point2f(xmin,ymin),Point2f(xmax,ymax));
		
                for (int j = i + 1; j < sortIdxs.size(); j++) {
		    if (scores[sortIdxs.at(j)] < thresholdScore) {
			badIdxs.insert(sortIdxs[j]);
			continue;
		    }
		    float ymin=boxes[4*j  ];//ymin
	            float xmin=boxes[4*j+1];//xmin
	            float ymax=boxes[4*j+2];//ymax
	            float xmax=boxes[4*j+3];//xmax
		    Rect2f box2 = Rect2f(Point2f(xmin,ymin),Point2f(xmax,ymax));
		    if (IOU(box1, box2) > thresholdIOU)
			badIdxs.insert(sortIdxs[j]);
		    }
		    i++;
	      }	      
	      // Prepare "good" idxs for return
	      vector<int> goodIdxs = vector<int>();
	      for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
		if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
		    goodIdxs.push_back(*it);

	      return goodIdxs;
}




int main(int argc, char** argv)
{
	ros::init(argc,argv, "pose_decision2");
	ros::NodeHandle n;
	ros::NodeHandle private_nh_("~");

	ros::Rate r(5);
	
	image_transport::ImageTransport it(n);
    	image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    	image_transport::CameraSubscriber depth_sub_ = it.subscribeCamera("/camera/depth/image_rect_raw", 1, &depth_Callback);
    	ros::Subscriber sub_detect = n.subscribe("/people",1,&peopleCallback);
    	
    	ros::Publisher pub_pass = n.advertise<tf_ros_detection::PosePass>("/posepass",1);
    	
    	ros::Publisher pub_danger = n.advertise<std_msgs::Int32>("/danger",1);
    	
    	string model_path;
    	private_nh_.getParam("models_dir",model_path);

	//string model_path = "/home/ubuntu/Downloads/coco_mobile/detect.tflite";
	string label_path = "/home/ubuntu/Downloads/coco_mobile/labelmap.txt";
	
	double thresholdScore = 0.5;
    	double thresholdIOU = 0.8;
    	
    	private_nh_.getParam("threshold_Score",thresholdScore);
	private_nh_.getParam("threshold_IOU",thresholdIOU);	
	
	
	//wait till we receive images
    	while (ros::ok() && (is_first_image || is_first_depth)) {
        	ros::spinOnce();
        	r.sleep();
    	}
    /*	while (ros::ok() && (is_first_pose)) {
        	ros::spinOnce();
        	r.sleep();
    	} */
    	
	cout << "start" << endl;
	std::string person="person";
	measure::time_point begin = measure::now();
	measure::time_point begin2;
	measure::time_point end = measure::now();
	measure::time_point end2;
 //cout << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0 << endl;
// image = imread("/home/ubuntu/Downloads/i4.jpeg");
 int fwidth = image.size().width;
        int fheight = image.size().height;
        bool passed = false;
	while(n.ok()){
	        
	        
	        
		begin = measure::now();
		if (processed){
		    frame = image.clone();
		    image_depth = image_depth_org.clone();	
		    tf_ros_detection::PosePass detections;
      		    detections.header.seq = 1;
      		    detections.header.stamp = ros::Time::now();
      		    detections.header.frame_id = "pose detection";
      		    detections.image = *( cv_bridge::CvImage(detections.header,"rgb8",frame.clone()).toImageMsg() );
      		    pub_pass.publish(detections);    
		 }
		 
 else {
		
		
		
		
		vector<vector<float>> face_location;
			
	        for(int i=0;i<num_detections;i++)
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
	        	if (score_face>0.6)
	        	   {
	        	      int y_size = (facey_max - facey_min)/2;
	        	      int x_size = (facex_max - facex_min)/2;
	                      int size_face = (x_size>y_size) ? x_size : y_size;
	                      int center_x = (int)( ( xs[0]+xs[1]+xs[2]+xs[3]+xs[4] )/5 );
	                      int center_y = (int)( ( ys[0]+ys[1]+ys[2]+ys[3]+ys[4] )/5 );
	              
              	              cv::Rect rec((center_x-size_face),(center_y-size_face),(size_face*2),(size_face*2));
              	              cv::rectangle(frame,rec,Scalar(255,0,255),3);
              	              if (person_check(rec)==1)
	        	      {
	        	        cout << "valid distance" << endl;
	        	      	vector<float> pos; 
			        pos.push_back(x_mean); pos.push_back(y_mean); 
			        pos.push_back(z_mean); pos.push_back(d_mean);
			    	face_location.push_back(pos);
			    	cv::rectangle(frame,rec,Scalar(0,255,255),1);
			    	putText(frame, "face" + std::to_string((int)(score_face*100))+" "+std::to_string(d_mean), Point(rec.x, rec.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
	        	      }
	        	      else
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
              	                cv::rectangle(frame,rec2,Scalar(0,255,255),3);
              	                
              	                if (person_check(rec2)==1)
			        {
				        cout << "valid distance" << endl;
				      	vector<float> pos; 
					pos.push_back(x_mean); pos.push_back(y_mean); 
					pos.push_back(z_mean); pos.push_back(d_mean);
				    	face_location.push_back(pos);
				    	cv::rectangle(frame,rec2,Scalar(0,255,255),1);
				    	putText(frame, "body" + std::to_string((int)(score_middle*100))+" "+std::to_string(d_mean), Point(rec2.x, rec2.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
				}
	        	      }
	        	   }
	        	else if (score_middle>0.6)
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
              	              cv::rectangle(frame,rec2,Scalar(0,255,255),3);
              	              if (person_check(rec2)==1)
			        {
				        cout << "valid distance" << endl;
				      	vector<float> pos; 
					pos.push_back(x_mean); pos.push_back(y_mean); 
					pos.push_back(z_mean); pos.push_back(d_mean);
				    	face_location.push_back(pos);
				    	cv::rectangle(frame,rec2,Scalar(0,255,255),1);
				    	putText(frame, "body" + std::to_string((int)(score_middle*100))+" "+std::to_string(d_mean), Point(rec2.x, rec2.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
				}
	        	}
	        	   
	        	   
	        }	
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
	    
		imshow("main_frame_tf",frame);
		waitKey(1);   	
      		
      		pub_danger.publish(msg1);
      		processed = true;
 }     		
		ros::spinOnce();
           	r.sleep();
	}
}
