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
#include <tf_ros_detection/TargetBB.h>
#include <tf_ros_detection/Detect.h>

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

cv::Mat image, frame;
cv::Mat image_depth_org, image_depth;
std_msgs::Header header_org, header;

double x_mean, y_mean,z_mean,d_mean;

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
	ros::init(argc,argv, "people_detect_pass");
	ros::NodeHandle n;
	ros::NodeHandle private_nh_("~");

	ros::Rate r(5);
	
	image_transport::ImageTransport it(n);
    	image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    	image_transport::CameraSubscriber depth_sub_ = it.subscribeCamera("/camera/depth/image_rect_raw", 1, &depth_Callback);
    	
    	ros::Publisher pub_target = n.advertise<tf_ros_detection::Detect>("/detect",1);
    	
    	string model_path;
    	int type = 1;//1:int 2:float
    	private_nh_.getParam("models_dir",model_path);
    	private_nh_.getParam("type",type);

	//string model_path = "/home/ubuntu/Downloads/coco_mobile/detect.tflite";
	string label_path = "/home/ubuntu/Downloads/coco_mobile/labelmap.txt";
	
	double thresholdScore = 0.5;
    	double thresholdIOU = 0.8;
    	
    	private_nh_.getParam("threshold_Score",thresholdScore);
	private_nh_.getParam("threshold_IOU",thresholdIOU);	
	
	// Load the model
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
	if (!model) {
	    ROS_ERROR("Failed to load model");
	    //LOG(FATAL) << "Failed to load model " << "\n";
	    //exit(-1);
	 }
	cout << "loaded model" << endl;
	
	std::vector<std::string> Labels;
	bool loaded = getFileContent(label_path, Labels);
	if (loaded){
	     cout << "Loaded labels with " << Labels.size() << "labels" << endl;
	     }
	else{
	     ROS_ERROR("failed to load labels");
	 }
	 cout << Labels[0] << endl;

	//Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*model, resolver)(&interpreter);
	if (!interpreter) {
	    ROS_ERROR("Failed to construct interpreter");
	  }

	//Resize input tensors, if desired
	interpreter->AllocateTensors();
	//tflite::PrintInterpreterState(interpreter.get());

	cout << "tensors size: " << interpreter->tensors_size() << "\n";
    	cout << "nodes size: " << interpreter->nodes_size() << "\n";
    	cout << "inputs: " << interpreter->inputs().size() << "\n";
    	cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
	int t_size = interpreter->tensors_size();
	
	interpreter->SetAllowFp16PrecisionForFp32(true);
	interpreter->SetNumThreads(2);

//	float* input = interpreter->types_input_tensor<float>(0);
	
	int input = interpreter->inputs()[0];
	int output = interpreter->outputs()[0];
	const std::vector<int> inputs = interpreter->inputs();
        const std::vector<int> outputs = interpreter->outputs();
        
        TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  	int height = dims->data[1];
  	int width = dims->data[2];
	int wanted_channels = dims->data[3];
	
	TfLiteIntArray* dims_o = interpreter->tensor(output)->dims;
  	int height_o = dims_o->data[1];
  	int width_o = dims_o->data[2];
	int wanted_channels_o = dims_o->data[3];
	
	cout << "input shape " << height << "\t" << width << "\t" << wanted_channels << endl;
	cout << "output shape " << height_o << "\t" << width_o << "\t" << wanted_channels_o << endl;
	
	//wait till we receive images
    /*	while (ros::ok() && (is_first_image || is_first_depth)) {
        	ros::spinOnce();
        	r.sleep();
    	}*/
    	int fwidth = image.size().width;
        int fheight = image.size().height;
	cout << "start" << endl;
	std::string person="person";
	measure::time_point begin = measure::now();
	measure::time_point begin2;
	measure::time_point end = measure::now();
	measure::time_point end2;
 //cout << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0 << endl;
 	float input_mean = 127.5;
	float input_std = 127.5;
	while(n.ok()){
	
	begin = measure::now();
		//frame = image.clone();
		frame = imread("/home/ubuntu/Downloads/i7.jpg");
		resize(frame,frame,Size(640,480));
		fwidth = frame.size().width;
                fheight = frame.size().height;
      		//image_depth = image_depth_org.clone();
      		
      		cvtColor(frame, frame, COLOR_BGR2RGB);
      		Mat frame_nn;
      		resize(frame,frame_nn,Size(width,height));
	
		if (type==1)
      		    memcpy(interpreter->typed_input_tensor<uchar>(0), frame_nn.data, frame_nn.total() * frame_nn.elemSize());
      		else if (type==2)
      		{
      			Mat frame_n2;
      			frame_nn.convertTo(frame_n2,CV_32FC3);
      			frame_nn = frame_n2 - Vec3b(input_mean,input_mean,input_mean);//temp;
      			frame_nn = frame_nn*(1.0f/input_std);
      			std::memcpy(interpreter->typed_input_tensor<float32_t>(0), frame_nn.data, frame_nn.total() * frame_nn.elemSize());
      		}

      		interpreter->Invoke();
        end = measure::now();
        
//cout << "Inference time " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0 << endl;
      		float* boxes = interpreter->tensor(interpreter->outputs()[0])->data.f;
    		float* classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    		float* scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    		int    num = *interpreter->tensor(interpreter->outputs()[3])->data.f;
    		cout << "detections: " << num << endl;
	begin2 = measure::now();
      		vector<cv::Rect> locations;
      		vector<vector<float>> people_location;
      		tf_ros_detection::Detect detections;
      		vector<tf_ros_detection::TargetBB> targets;
      		detections.header.seq = 1;
      		detections.header.stamp = ros::Time::now();
      		detections.header.frame_id = "human detection";
      		detections.image = *( cv_bridge::CvImage(detections.header,"rgb8",frame.clone()).toImageMsg() );
      	//	detections.depth = *( cv_bridge::CvImage( detections.header,sensor_msgs::image_encodings::TYPE_32FC1,image_depth.clone()).toImageMsg() );
      		/*imshow("original",image_depth.clone());
      		waitKey(1);
      		
      		Mat detectDep;
      		try
		{
		    sensor_msgs::ImageConstPtr dep_ptr(new sensor_msgs::Image(detections.depth));
		    Mat temp = cv_bridge::toCvShare(dep_ptr, sensor_msgs::image_encodings::TYPE_32FC1)->image;
		    detectDep = temp.clone();
		}
		catch (cv_bridge::Exception& e)
		{   
		    cout << "error" << endl;
		    ROS_ERROR("Could not convert from '%s' to 'depth grayscale'.", detections.depth.encoding.c_str()); 
		}
		
		imshow("new",detectDep);
	    	waitKey(1);*/
      		
      		vector<int> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore, num);
      		
      		
      		
      		for (int ii=0;ii<goodIdxs.size();ii=ii+1)
      		{      
      			int i = goodIdxs.at(ii);
      			if ((classes[i]==0) && (scores[i]>thresholdScore))//(strcmp(Labels[classes[i]],person.c_str())==0)
      			{
      		         //   cout << Labels[classes[i]] << "\t" << scores[i] <<  endl;		
      		            float ymin=boxes[4*i  ]*fheight;//ymin
			    float xmin=boxes[4*i+1]*fwidth;//xmin
			    float ymax=boxes[4*i+2]*fheight;//ymax
			    float xmax=boxes[4*i+3]*fwidth;//xmax  (((xmax-xmin)<=fwidth)? (xman-xmin) : fwidth)
			  //  cout << ymin << "\t" << xmin << "\t" << ymax << "\t" << xmax << endl;
			    cv::Rect rec((int)((xmin>=0)? xmin : 0), (int)((ymin>=0)? ymin : 0), (int)(((xmax-xmin)<=fwidth)? (xmax-xmin) : fwidth), (int)(((ymax-ymin)<=fheight)? (ymax-ymin) : fwidth));
			    tf_ros_detection::TargetBB target;
			    target.score.data = (int)(scores[i]*100);
			    set_roi(&target.roi,rec);
			    targets.push_back(target);
			
			    end = measure::now();
       		      //      cout << "One Distance estimation time " << (std::chrono::duration_cast<std::chrono::microseconds>(begin-end).count())/1000000.0 << endl;
			    cv::rectangle(frame,rec,Scalar(0,255,255),1);//
			    putText(frame, Labels[classes[i]] + std::to_string((int)(scores[i]*100))+" "+std::to_string(d_mean), Point(xmin, ymin+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
      			}
      		}
      		detections.boxes = targets;
	end2 = measure::now();
//cout << "Total distance estimation time " << (std::chrono::duration_cast<std::chrono::microseconds>(begin2-end2).count())/1000000.0 << endl;
      		cvtColor(frame, frame, COLOR_RGB2BGR);
      		//putText(frame,"i can see",Point((int)(fwidth/2),(int)(fheight/2)),FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,255),2);


        imshow("frame_people_detect",frame);
        waitKey(1);   	
      		
      		pub_target.publish(detections);
		ros::spinOnce();
           	r.sleep();
	}
}
