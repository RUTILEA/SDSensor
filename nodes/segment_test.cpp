// followed
// https://github.com/Qengineering/TensorFlow_Lite_SSD_RPi_64-bits/blob/master/MobileNetV1.cpp

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc


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
#include <pigpio.h>

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
 //   imshow("depth received",image_depth_org);
 //   waitKey(1);

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
vector<tf_ros_detection::TargetBB> rois;
vector<tf_ros_detection::TargetBB> faces;
void detectCallback(const tf_ros_detection::DetectConstPtr& msg)
{
	head_detect = msg->header;
	rois = msg->boxes;
	faces = msg->faces;
	
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
	try
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
	
	//imshow("depth received",detectDep);
    	//waitKey(1);
	
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

int person_check(cv::Rect roi)//, Mat& mask)
{
    x_mean = 0;
    y_mean = 0;
    z_mean = 0;
    d_mean = 0;
    int pixel_count = 0;

    Mat img_depth_in = image_depth.clone();

    float min_x=0,max_x=0,min_y=0,max_y=0,min_z=0,max_z=0;
    for(int row=roi.y; row<(roi.y+roi.height); row++)
    {
        const float* data_ptr = img_depth_in.ptr<float>(row);
        for(int col=roi.x; col<(roi.x+roi.width); col++)
        {
           const float data = data_ptr[col];
           if(data < 0.00001)
            { continue; }
           if ( (data>0.5))// && (mask.at<uchar>(row,col)<100) )
           {
            const float z_data = data;
            const float x_data = ((float)col-center_x) * z_data / focus_x;
            const float y_data = ((float)row-center_y) * z_data / focus_y;
            x_mean += x_data;
            y_mean += y_data;
            z_mean += z_data;
            d_mean += sqrtf( x_data*x_data + y_data*y_data + z_data*z_data  );
            pixel_count++;
            
            if (x_data>max_x){max_x=x_data;} else if (x_data<min_x){min_x=x_data;}
            if (y_data>max_y){max_y=y_data;} else if (y_data<min_y){min_y=y_data;}
            if (z_data>max_z){max_z=z_data;} else if (z_data<min_z){min_z=z_data;}
           }
        }
    }    
//    cout << "count " << pixel_count << endl;
    if (pixel_count<20)
    {
    	    pixel_count = 0;
    	    x_mean = 0;
	    y_mean = 0;
	    z_mean = 0;
	    d_mean = 0;
	    min_x=0;max_x=0;min_y=0;max_y=0;min_z=0;max_z=0;
            for(int row=roi.y; row<(roi.y+roi.height); row++)
	    {
		const float* data_ptr = img_depth_in.ptr<float>(row);
		for(int col=roi.x; col<(roi.x+roi.width); col++)
		{
		   const float data = data_ptr[col];
		   if(data < 0.00001)
		    { continue; }
		   if ( (data>0.5)  )
		   {
		    const float z_data = data;
		    const float x_data = ((float)col-center_x) * z_data / focus_x;
		    const float y_data = ((float)row-center_y) * z_data / focus_y;
		    x_mean += x_data;
		    y_mean += y_data;
		    z_mean += z_data;
		    d_mean += sqrtf( x_data*x_data + y_data*y_data + z_data*z_data  );
		    pixel_count++;
		    
		    if (x_data>max_x){max_x=x_data;} else if (x_data<min_x){min_x=x_data;}
		    if (y_data>max_y){max_y=y_data;} else if (y_data<min_y){min_y=y_data;}
		    if (z_data>max_z){max_z=z_data;} else if (z_data<min_z){min_z=z_data;}
		   }
		}
	    }   
	    cout << "pixel calculated again " << pixel_count << endl;
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

double IOU(Rect2f box1, Rect2f box2) { // box 2 is face

		    float xA = max(box1.tl().x, box2.tl().x);
		    float yA = max(box1.tl().y, box2.tl().y);
		    float xB = min(box1.br().x, box2.br().x);
		    float yB = min(box1.br().y, box2.br().y);
		    
		    float intersectArea;
		    		    
		    if ( (xA>xB) || (yA>yB) )
		    	intersectArea = 0;
		    else
		    	intersectArea = abs((xB - xA) * (yB - yA));
                    //float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;
		    float faceArea = abs(box2.area());
		    return ((1.0 * intersectArea) / faceArea); // less than 1. if = 1 then completely inside
		    
}


int main(int argc, char** argv)
{
	ros::init(argc,argv, "segment_test");
	ros::NodeHandle n;
	ros::NodeHandle private_nh_("~");

	ros::Rate r(5);
	
	image_transport::ImageTransport it(n);
    	image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    	image_transport::CameraSubscriber depth_sub_ = it.subscribeCamera("/camera/depth/image_rect_raw", 1, &depth_Callback);
    	ros::Subscriber sub_detect = n.subscribe("/detect",1,&detectCallback);
    	
    	ros::Publisher pub_danger = n.advertise<std_msgs::Int32>("/danger",1);

	string model_path = "/home/ubuntu/Downloads/deeplabv3_1_default_1.tflite";
	model_path = "/home/ubuntu/Downloads/deeplabv3_257_mv_gpu.tflite";
	string label_path = "/home/ubuntu/Downloads/coco_mobile/labelmap.txt";
	
    	private_nh_.getParam("models_dir",model_path);
	private_nh_.getParam("labels_dir",label_path);
	double thresholdScore = 0.7;
    	double thresholdIOU = 0.8;

/*
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
	*/

	//wait till we receive images
    	while (ros::ok() && (is_first_image || is_first_depth || is_first_detect )) {
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
	
	/*measure::time_point begin = measure::now();
	measure::time_point begin2;
	measure::time_point end = measure::now();
	measure::time_point end2;
 cout << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0 << endl;*/
	while(ros::ok()){
	//begin = measure::now();
		frame = detectImg.clone();
      		image_depth = image_depth_org.clone();	
      		cvtColor(frame, frame, COLOR_BGR2RGB);
      	/*	Mat frame_nn;
      		Mat frame_nn2;
      		Mat frame_nt; Mat frame_n2;
      		resize(frame,frame_nt,Size(width,height));
      		frame_nt.convertTo(frame_n2,CV_32FC3);
//       string ty =  type2str( frame_nt.type() );
//printf("Matrix: %s %dx%d \n", ty.c_str(), frame_nt.cols, frame_nt.rows );
      		//Mat temp = cv::Mat::ones(Size(width,height),CV_32FC3)*input_mean;	
      		//cvSub(frame_n2,temp,frame_nn);
      		frame_nn = frame_n2 - Vec3b(input_mean,input_mean,input_mean);//temp;
      		frame_nn = frame_nn*(1.0f/input_std);
      	//	cout << width << "\t" << height << endl;
//cout << inputs[0] << endl;
//cout << "hello" << interpreter->typed_input_tensor<float32_t>(0) << endl;
      		std::memcpy(interpreter->typed_input_tensor<float32_t>(0), frame_nn.data, frame_nn.total() * frame_nn.elemSize());
      		interpreter->Invoke();
      		float* data = interpreter->tensor(interpreter->outputs()[0])->data.f;
    
    		RGB *rgb;
		static Mat frame_n(width,height,CV_8UC3);
    		static Mat blend_n(frame.cols   ,frame.rows    ,CV_8UC3);
    		Mat mask(width,height,CV_8UC1);
    		Mat mask_n(frame.cols   ,frame.rows    ,CV_8UC1);
 //  cout << "c1" << endl; 	
    		rgb = (RGB *)frame_n.data;
   // 		cout << "c2" << endl;   
    		int mi, k;
    		float mx, v;
    		for(int i=0;i<height;i++){
			for(int j=0;j<width;j++){
			    for(mi=-1,mx=0.0,k=0;k<21;k++){
				v = data[21*(i*width+j)+k];
				if(v>mx){ mi=k; mx=v; }
			    }
			    rgb[j+i*width] = Colors[mi];
			    int* data_ptr = mask.ptr<int>(i);
			    if (mi==15)
			        { mask.at<uchar>(i,j) = 0;}
                            else
			        mask.at<uchar>(i,j) = 255;    
			       
			}
	    	}
//	 cout << "c3" << endl;     	
	//    	imshow("Masp person",mask);
	//    	waitKey(1);	    	
    	
    		end = measure::now();
        
	//cout << "Inference time " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0 << endl;
	
		//merge output into frame
		cv::resize(frame_n, blend_n, Size(frame.cols,frame.rows),INTER_NEAREST);
		cv::resize(mask, mask_n, Size(frame.cols,frame.rows),INTER_NEAREST);
		Mat temp2;
		cvtColor(frame, frame, COLOR_RGB2BGR);
		cv::addWeighted(frame, 0.5, blend_n, 0.5, 0.0, temp2);
//    		imshow("people?",temp2);
//		waitKey(1);
    	
      		//cvtColor(frame, frame, COLOR_RGB2BGR);
      		//putText(frame,"i can see",Point((int)(fwidth/2),(int)(fheight/2)),FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,255),2);
      			
          */      
                //rois = msg->boxes;
	        //faces = msg->faces;
	        vector<vector<float>> face_location;
                for (int i=0;i<faces.size();i++)
                {
                	sensor_msgs::RegionOfInterest ROI = faces[i].roi;
                	cv::Rect rec(ROI.x_offset,ROI.y_offset,ROI.width,ROI.height);
                	if (person_check(rec)==1)
			    {
			        vector<float> pos; 
			        pos.push_back(x_mean); pos.push_back(y_mean); 
			        pos.push_back(z_mean); pos.push_back(d_mean);
			    	face_location.push_back(pos);
			    	cv::rectangle(frame,rec,Scalar(0,255,255),1);
			    	putText(frame, "face " + std::to_string(i) + " " + std::to_string((int)((faces[i].score.data)*100))+" "+std::to_string(d_mean), Point(rec.x, rec.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
			    }
                }
                vector<vector<float>> people_location;
                vector<int> id_filter;
                vector<int> id_bad;
                //assuming all faces are correct // find other boxes not included in faces
                for (int i=0;i<rois.size();i++)
                {
                	for (int j=0;j<faces.size();j++)
                	{
                	    
                		sensor_msgs::RegionOfInterest ROI_i = rois[i].roi;
                		cv::Rect rec_i(ROI_i.x_offset,ROI_i.y_offset,ROI_i.width,ROI_i.height);
                		sensor_msgs::RegionOfInterest ROI_j = faces[j].roi;
                		cv::Rect rec_j(ROI_j.x_offset,ROI_j.y_offset,ROI_j.width,ROI_j.height);
                		if (IOU(rec_i,rec_j)>0.6)
                		{
                			id_bad.push_back(i);
                	 		break;
                			//cout << "bbox " << IOU(rec_i,rec_j) << "\t" << i << "\t" << j << endl;
                			//id_filter.push_back(i);
                		}
                		//else
                	//	{
                	//		
                	//	}
                	}
                }
                
                for (int i=0;i<rois.size();i++)
                {
                	bool found = false;
                	for (int i_f=0;i_f<id_bad.size();i_f++)
		        {
		           	if (i==id_bad[i_f]){
		           		found = true;
		           		break;
		           	}                   	   
		        }
		        if (!found)
		        {
		        	sensor_msgs::RegionOfInterest ROI_i = rois[i].roi;
                		cv::Rect rec_i(ROI_i.x_offset,ROI_i.y_offset,ROI_i.width,ROI_i.height);
		        	if (person_check(rec_i)==1)
				{
					vector<float> pos; 
					pos.push_back(x_mean); pos.push_back(y_mean); 
					pos.push_back(z_mean); pos.push_back(d_mean);
				    	people_location.push_back(pos);
				    	cv::rectangle(frame,rec_i,Scalar(0,255,255),1);
				    	putText(frame, "person" + std::to_string((int)((rois[i].score.data)*100))+" "+std::to_string(d_mean), Point(rec_i.x, rec_i.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
				    	//putText(frame, "person" + std::to_string(i)+ " with " +std::to_string(j), Point(rec_i.x, rec_i.y+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
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
                		if (dist_rel<1.5f)
                		{
                		   danger = true;
                		}
                	}
                	
                	for (int j=0;j<people_location.size();j++)
                	{
                		float dist_rel = sqrtf( powf(face_location[i][0]-people_location[j][0],2) + powf(face_location[i][1]-people_location[j][1],2) + powf(face_location[i][2]-people_location[j][2],2) );
                		if (dist_rel<1.5f)
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
                	
                
                imshow("segment frame",frame);
                waitKey(1);   
                
                pub_danger.publish(msg1);
                	

		ros::spinOnce();
           	r.sleep();
	}
}
