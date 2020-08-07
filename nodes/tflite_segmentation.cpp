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
std_msgs::Header header_org, header;

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
//-----------------------------------------------------------------------------------------------------------------------

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


int main(int argc, char** argv)
{
	ros::init(argc,argv, "segment_c");
	ros::NodeHandle n;
	ros::NodeHandle private_nh_("~");

	ros::Rate r(20);
	
	image_transport::ImageTransport it(n);
    	image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    	image_transport::CameraSubscriber depth_sub_ = it.subscribeCamera("/camera/depth/image_rect_raw", 1, &depth_Callback);

	string model_path = "/home/ubuntu/Downloads/deeplabv3_1_default_1.tflite";
	model_path = "/home/ubuntu/Downloads/deeplabv3_257_mv_gpu.tflite";
	string label_path = "/home/ubuntu/Downloads/coco_mobile/labelmap.txt";
	
	double thresholdScore = 0.7;
    	double thresholdIOU = 0.8;

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
	tflite::PrintInterpreterState(interpreter.get());

	cout << "tensors size: " << interpreter->tensors_size() << "\n";
    	cout << "nodes size: " << interpreter->nodes_size() << "\n";
    	cout << "inputs: " << interpreter->inputs().size() << "\n";
    	cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
	int t_size = interpreter->tensors_size();
	
	interpreter->SetAllowFp16PrecisionForFp32(true);
	interpreter->SetNumThreads(2);

	//float* input = interpreter->typed_input_tensor<float>(0);
	
	int input = interpreter->inputs()[0];
	const std::vector<int> inputs = interpreter->inputs();
        const std::vector<int> outputs = interpreter->outputs();
        
        cout << "size" << inputs.size() << endl;
        cout << input << endl;
        
        TfLiteIntArray* dims = interpreter->tensor(input)->dims;
        //cout << interpreter->tensor(input)->shape() << endl;
  	int height = dims->data[1];
  	int width = dims->data[2];
	int wanted_channels = dims->data[3];
	cout << height << "\t" << width << "\t" << wanted_channels << endl;
	
	int output = interpreter->outputs()[0];
	TfLiteIntArray* dims_o = interpreter->tensor(output)->dims;
  	int height_o = dims_o->data[1];
  	int width_o = dims_o->data[2];
	int wanted_channels_o = dims_o->data[3];
	
	cout << "output shape " << height_o << "\t" << width_o << "\t" << wanted_channels_o << endl;
	

	//wait till we receive images
    	while (ros::ok() && (is_first_image || is_first_depth)) {
        	ros::spinOnce();
        	r.sleep();
    	} 
    	//image = imread("/home/ubuntu/Downloads/i4.jpeg");
    	int fwidth = image.size().width;
        int fheight = image.size().height;
	cout << "start" << endl;
	std::string person="person";
	float input_mean = 127.5;
	float input_std = 127.5;
	
		measure::time_point begin = measure::now();
	measure::time_point begin2;
	measure::time_point end = measure::now();
	measure::time_point end2;
 cout << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0 << endl;
 
	while(n.ok()){
	begin = measure::now();
		frame = image.clone();
      		image_depth = image_depth_org.clone();
      	
      		cvtColor(frame, frame, COLOR_BGR2RGB);
      		Mat frame_nn;
      		Mat frame_nt; Mat frame_n2;
      		resize(frame,frame_nt,Size(width,height));
      		frame_nt.convertTo(frame_n2,CV_32FC3);
      		
      		Mat temp = cv::Mat::ones(Size(width,height),CV_32FC3)*input_mean;
      		//cvSub(frame_n2,temp,frame_nn);
      		frame_nn = frame_n2 - Vec3b(input_mean,input_mean,input_mean);//temp;
      		frame_nn = frame_nn*(1.0f/input_std);
      		cout << width << "\t" << height << endl;

//cout << interpreter << endl;
cout << inputs[0] << endl;
cout << "hello" << interpreter->typed_input_tensor<float32_t>(0) << endl;

      		std::memcpy(interpreter->typed_input_tensor<float32_t>(0), frame_nn.data, frame_nn.total() * frame_nn.elemSize());

      		interpreter->Invoke();

//cout <<  interpreter->tensor(interpreter->outputs()[0]) << endl;

      		float* data = interpreter->tensor(interpreter->outputs()[0])->data.f;
    
    		RGB *rgb;
		static Mat frame_n(width,height,CV_8UC3);
    		static Mat blend_n(frame.cols   ,frame.rows    ,CV_8UC3);
    	
    		rgb = (RGB *)frame_n.data;
    		int mi, k;
    		float mx, v;
    		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
			    for(mi=-1,mx=0.0,k=0;k<21;k++){
				v = data[21*(i*width+j)+k];
				if(v>mx){ mi=k; mx=v; }
			    }
			    rgb[j+i*width] = Colors[mi];
			}
	    	}
	    	
    	
    		end = measure::now();
        
	cout << "Inference time " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0 << endl;
	
		//merge output into frame
		cv::resize(frame_n, blend_n, Size(frame.cols,frame.rows),INTER_NEAREST);
		Mat temp2;
		cvtColor(frame, frame, COLOR_RGB2BGR);
		cv::addWeighted(frame, 0.5, blend_n, 0.5, 0.0, temp2);
    		imshow("people?",temp2);
		waitKey(1);
    	
   /*	vector<float> values;
        cout << boxes[(257*257+21)] << endl;
        float max = 0;
    	float min = 0;
		for (int ii=0;ii<(257*257);ii++)
		{
		   values.push_back(boxes[( 21*(ii) + 15 )]);
		   //     cout << ii << "\t" << boxes[( 21*(ii) + 15 )] << endl;
		   if (values.back()>max)
		   	max = values.back();
                   else if (values.back()<min)
                        min = values.back();
		}
		cout << values.size() << endl;     		
      		
      		Mat temp2 = cv::Mat::ones(Size(width,height),CV_8UC1);
   
      		for (int ii=0;ii<(257*257);ii++)
      		{
      		    temp2.at<uchar>(((int)ii/257),(ii%257)) = (uchar)(255/(max-min)*(values[ii]-min));
      		}
  	
      		resize(temp2,temp2,Size(640,480));
      		imshow("people?",temp2);
		waitKey(1);
		
*/
      		//cvtColor(frame, frame, COLOR_RGB2BGR);
      		putText(frame,"i can see",Point((int)(fwidth/2),(int)(fheight/2)),FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,255),2);
      		imshow("main_frame_tf",frame);
                waitKey(1);   		

		ros::spinOnce();
           	r.sleep();
	}
}
