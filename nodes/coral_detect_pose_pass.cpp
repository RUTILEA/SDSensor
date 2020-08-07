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
//#include "basic_engine.h"
#include "test_utils.h"
#include "engine.h"

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

#include <tf_ros_detection/TargetBB.h>
#include <tf_ros_detection/Detect.h>
#include <tf_ros_detection/people.h>
#include <tf_ros_detection/keypoints.h>
#include <tf_ros_detection/PosePass.h>
#include <std_msgs/Int32.h>

#include "tensorflow/lite/builtin_op_data.h"

using namespace std;
using namespace cv;

using tflite::ops::builtin::BuiltinOpResolver;

int model_width;
int model_height;
int model_channels;

cv::Mat image, frame;
cv::Mat image_depth_org, image_depth;
std_msgs::Header header_org;

double x_mean, y_mean,z_mean,d_mean;
bool is_first_image = true;
bool is_first_depth = true;
double focus_x = 0;
double focus_y = 0;
double center_x = 0;
double center_y = 0;



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
    image_depth_org = depth_image.clone();

    focus_x = camera_info->K[0];
    focus_y = camera_info->K[4];
    center_x = camera_info->K[2];
    center_y = camera_info->K[5];

    is_first_depth = false;
    
}

namespace coral {

struct Keypoint {
  Keypoint(float _y, float _x, float _score) : y(_y), x(_x), score(_score) {}
  float y;
  float x;
  float score;
};

}  // namespace coral

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

void set_roi(sensor_msgs::RegionOfInterest *roi, cv::Rect Irect)
{
    roi->x_offset = Irect.x;
    roi->y_offset = Irect.y;
    roi->width = Irect.width;
    roi->height = Irect.height;
}

double IOU(Rect2f box1, Rect2f box2, double thresholdSame) {

    float xA = max(box1.tl().x, box2.tl().x);
    float yA = max(box1.tl().y, box2.tl().y);
    float xB = min(box1.br().x, box2.br().x);
    float yB = min(box1.br().y, box2.br().y);
    
    float intersectArea;
    
    if ( (xA>xB) || (yA>yB) )
    	intersectArea = 0;
    else
    	intersectArea = abs((xB - xA) * (yB - yA));
    	
  // check if completely inside
  if ( ( ( intersectArea/abs(box2.area()) ) > thresholdSame ) || ( ( intersectArea/abs(box1.area()) ) > thresholdSame ) )
     return 1;

    float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;

    return 1. * intersectArea / unionArea;
}

vector<int> filterBoxes(std::vector<coral::DetectionCandidate> candidates, double thresholdIOU, double thresholdScore, double thresholdSame, int fwidth, int fheight, vector<tf_ros_detection::TargetBB>& targets, cv::Mat &frame_iou)//float *scores, float *boxes, double thresholdIOU, double thresholdScore, int num) 
{
           vector<int> sortIdxs(candidates.size());
           iota(sortIdxs.begin(), sortIdxs.end(), 0);
           // Create set of "bad" idxs
	    set<int> badIdxs = set<int>();
	    int i = 0;
            while (i < sortIdxs.size()) {
                
		if (candidates[sortIdxs.at(i)].score < thresholdScore)
		    badIdxs.insert(sortIdxs[i]);
		if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
		    i++;
		    continue;
		}
		float ymin=candidates[i].corners.xmin;//boxes[4*i  ];//ymin
	        float xmin=candidates[i].corners.ymin;//boxes[4*i+1];//xmin
	        float ymax=candidates[i].corners.xmax;//boxes[4*i+2];//ymax
	        float xmax=candidates[i].corners.ymax;//boxes[4*i+3];//xmax
		Rect2f box1 = Rect2f(Point2f(xmin,ymin),Point2f(xmax,ymax));
		
                for (int j = i + 1; j < sortIdxs.size(); j++) {
		    if (candidates[sortIdxs.at(j)].score < thresholdScore) {
			badIdxs.insert(sortIdxs[j]);
			continue;
		    }
		    float ymin=candidates[j].corners.xmin;//boxes[4*j  ];//ymin
	            float xmin=candidates[j].corners.ymin;//boxes[4*j+1];//xmin
	            float ymax=candidates[j].corners.xmax;//boxes[4*j+2];//ymax
	            float xmax=candidates[j].corners.ymax;//boxes[4*j+3];//xmax
		    Rect2f box2 = Rect2f(Point2f(xmin,ymin),Point2f(xmax,ymax));
		    if (IOU(box1, box2, thresholdSame) > thresholdIOU)
			badIdxs.insert(sortIdxs[j]);
		    }
		    i++;
	      }	      
	      // Prepare "good" idxs for return
	      vector<int> goodIdxs = vector<int>();
	      for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
		if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
		{
		    goodIdxs.push_back(*it);
		    if (candidates[*it].label!=0) //human
		         continue;
		    tf_ros_detection::TargetBB target;
		    target.score.data = (int)(candidates[*it].score*100);   
		    int xmax = candidates[*it].corners.ymax*fwidth;
	  	    int xmin = candidates[*it].corners.ymin*fwidth;
	  	    int ymax = candidates[*it].corners.xmax*fheight;
	  	    int ymin = candidates[*it].corners.xmin*fheight;
		    cv::Rect rec((int)((xmin>=0)? xmin : 0), (int)((ymin>=0)? ymin : 0), (int)(((xmax-xmin)<=fwidth)? (xmax-xmin) : fwidth), (int)(((ymax-ymin)<=fheight)? (ymax-ymin) : fwidth));
		    set_roi(&target.roi,rec);
		    cv::rectangle(frame_iou,rec,Scalar(0,255,0),2);
		    targets.push_back(target);
		}
	      imshow("result after IoU",frame_iou);
	      waitKey(1);
	      return goodIdxs;
}


int main(int argc, char** argv) {
  ros::init(argc,argv, "coral_detect_pose_pass");
    ros::NodeHandle n;
    ros::NodeHandle private_nh_("~");
    
    image_transport::ImageTransport it(n);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    image_transport::CameraSubscriber depth_sub_ = it.subscribeCamera("/camera/depth/image_rect_raw", 1, &depth_Callback);
    ros::Publisher pub_target = n.advertise<tf_ros_detection::people>("/people",10);
    ros::Publisher pub_danger = n.advertise<std_msgs::Int32>("/danger",1);
    ros::Rate r(30);
    
  // Load the model.
  std::string pose_path = "/home/ubuntu/catkin_p3/src/test_p3/src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite";
  std::string image_path = "/home/ubuntu/Downloads/i4.jpeg";
  std::string ssd_path = "/home/ubuntu/codes/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";//"/home/ubuntu/Downloads/coco_mobile/detect_int_edgetpu.tflite";
  
  private_nh_.getParam("ssd_path",ssd_path);
  private_nh_.getParam("model_path",pose_path);
  
  LOG(INFO) << "Testing detection model: " << ssd_path;
  LOG(INFO) << "Testing posenet model: " << pose_path;
  coral::BasicEngine engine_pose(pose_path);
  coral::DetectionEngine engine_detect(ssd_path);
  std::vector<int> input_tensor_shape_detect = engine_detect.get_input_tensor_shape();
  std::vector<int> input_tensor_shape_pose   = engine_pose.get_input_tensor_shape();
  
  float thresholdScore = 0.3f; 
  float thresholdIOU = 0.4;
  float thresholdSame = 0.95;
  private_nh_.getParam("threshold_Score",thresholdScore);
  private_nh_.getParam("threshold_IOU",thresholdIOU); 
  private_nh_.getParam("threshold_Same",thresholdSame); 
  
  int use_image = 0; int resize_image = 0;
  private_nh_.getParam("use_image",use_image);
  private_nh_.getParam("resize_image",resize_image);
  private_nh_.getParam("image_path",image_path);
  
  int top_k = 1;
  private_nh_.getParam("top_k",top_k);
  
  //wait till we receive images
  if (!use_image) {
  while (ros::ok() && is_first_image && is_first_depth) {
       	ros::spinOnce();
       	r.sleep();
  }
  }
  else
  {
      image = imread(image_path);
      if (resize_image==1)
          resize(image,image,Size(960,720));
  }
  
  int fwidth = image.size().width;
  int fheight = image.size().height;
  
  while(n.ok()){
  
          image_depth = image_depth_org.clone();
          
          chrono::steady_clock::time_point Tbegin, Tend;
          Tbegin = chrono::steady_clock::now(); 
          
	  frame = image.clone();
	  Mat frame_detect = frame.clone();
	  Mat frame_pose = frame.clone();
	  
	  resize(frame_detect,frame_detect,Size(input_tensor_shape_detect[2],input_tensor_shape_detect[1])); // (width,height)
	  cvtColor(frame_detect, frame_detect, COLOR_BGR2RGB);
	  frame_detect.convertTo(frame_detect, CV_8UC3);
	  std::vector<uint8_t> input_detection(frame_detect.data, frame_detect.data + (frame_detect.cols * frame_detect.rows * frame_detect.elemSize()));
	  
	  cvtColor(frame_pose, frame_pose, COLOR_BGR2RGB);
	  frame_pose.convertTo(frame_pose, CV_8UC3);	    
	
	  // Get result Detection	  
	  auto candidates = engine_detect.DetectWithInputTensor(input_detection,thresholdScore,top_k);
	  
	  vector<tf_ros_detection::TargetBB> rois;
	  
	  vector<int> goodIdxs = filterBoxes(candidates, thresholdIOU, thresholdScore, thresholdSame, fwidth, fheight, rois, frame);
	  
	  cout << "Detection Size " << candidates.size() << endl;
	  //continue;
	  tf_ros_detection::people detections;
	  detections.num.data = 0;//n_poses[0];

	  // For now assuming that much more than 10 people present
          for (int i=0;i<rois.size();i=i+6)
          {
        	cv::Mat frame_copy = frame_pose.clone();
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
        	 resize(frame_roi,frame_roi,Size(input_tensor_shape_pose[2],input_tensor_shape_pose[1]));
        	imshow("masked frame",frame_roi);
        	waitKey(1);
        	std::vector<uint8_t> input_pose(frame_roi.data, frame_roi.data + (frame_roi.cols * frame_roi.rows * frame_roi.elemSize()));
        
		// Get result.
	  	
	  	const std::vector<std::vector<float>>& raw_outputs = engine_pose.RunInference(input_pose);

	  	const auto& poses = raw_outputs[0];
	  	const auto& keypoint_scores = raw_outputs[1];
	  	const auto& pose_scores = raw_outputs[2];
	  	const auto& n_poses = raw_outputs[3];

	  	//std::cout << "poses: " << n_poses[0] << std::endl;
	  	
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
		  			points.x.push_back(x/input_tensor_shape_pose[2]);
		  			points.y.push_back(y/input_tensor_shape_pose[1]);
		  			points.scores.push_back(point_score);
		  			//circle(frame,pt,4,Scalar( 255, 255, 0 ),FILLED);
		  			pt.x=(int)(x/input_tensor_shape_pose[2]*frame_pose.cols); pt.y=(int)(y/input_tensor_shape_pose[1]*frame_pose.rows);
		  			circle(frame_pose,pt,4,Scalar( 255, 255, 0 ),FILLED);
		  		}
		  		detections.persons.push_back(points);
		  	}
		  }
                
        	
          }
          imshow("pose detections",frame_pose);
          waitKey(1);
          Tend = chrono::steady_clock::now();
	  float f2 = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
	  cout << "Total Inference time " << f2 << endl;  
          continue;
               	vector<vector<float>> face_location;
               	vector<vector<float>> people_location;
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
              	              //cv::rectangle(frame_pose,rec,Scalar(255,0,255),3);
              	              if (person_check(rec)==1)
	        	      {    
	        	      	vector<float> pos; 
			        pos.push_back(x_mean); pos.push_back(y_mean); 
			        pos.push_back(z_mean); pos.push_back(d_mean);
			    	face_location.push_back(pos);
			    	cv::rectangle(frame_pose,rec,Scalar(0,255,255),1);
			    	putText(frame_pose, "face" + std::to_string((int)(score_face*100))+" "+std::to_string(d_mean), Point(rec.x, rec.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
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
              	                //cv::rectangle(frame_pose,rec2,Scalar(0,255,255),3);
              	                
              	                if (person_check(rec2)==1)
			        {
				      	vector<float> pos; 
					pos.push_back(x_mean); pos.push_back(y_mean); 
					pos.push_back(z_mean); pos.push_back(d_mean);
				    	face_location.push_back(pos);
				    	cv::rectangle(frame_pose,rec2,Scalar(0,255,255),1);
				    	putText(frame_pose, "body" + std::to_string((int)(score_middle*100))+" "+std::to_string(d_mean), Point(rec2.x, rec2.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
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
              	              //cv::rectangle(frame_pose,rec2,Scalar(0,255,255),3);
              	              if (person_check(rec2)==1)
			        {
				      	vector<float> pos; 
					pos.push_back(x_mean); pos.push_back(y_mean); 
					pos.push_back(z_mean); pos.push_back(d_mean);
				    	face_location.push_back(pos);
				    	cv::rectangle(frame_pose,rec2,Scalar(0,255,255),1);
				    	putText(frame_pose, "body" + std::to_string((int)(score_middle*100))+" "+std::to_string(d_mean), Point(rec2.x, rec2.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
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
	  






	  
	
	  
	 // imshow("results_pose",frame_pose);
	//  waitKey(1);
	  
	  
	  Tend = chrono::steady_clock::now();
	  float f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
	  cout << "Total Inference time " << f << endl;  
	  
	  pub_target.publish(detections);
	  pub_danger.publish(msg1);
	    
	  ros::spinOnce();
	  r.sleep();
    
    }
    
    
    
}

