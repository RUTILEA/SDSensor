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



int main(int argc, char** argv)
{
	ros::init(argc,argv, "tflite");
	ros::NodeHandle n;
	ros::NodeHandle private_nh_("~");

	ros::Rate r(20);
	
	image_transport::ImageTransport it(n);
    	image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    	image_transport::CameraSubscriber depth_sub_ = it.subscribeCamera("/camera/depth/image_rect_raw", 1, &depth_Callback);
    	
    	string model_path;
    	private_nh_.getParam("models_dir",model_path);

	//string model_path = "/home/ubuntu/Downloads/coco_mobile/detect.tflite";
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
    	while (ros::ok() && (is_first_image || is_first_depth)) {
        	ros::spinOnce();
        	r.sleep();
    	}
    	int fwidth = image.size().width;
        int fheight = image.size().height;
	cout << "start" << endl;
	std::string person="person";
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
      		resize(frame,frame_nn,Size(width,height));

      		memcpy(interpreter->typed_input_tensor<uchar>(0), frame_nn.data, frame_nn.total() * frame_nn.elemSize());

      		interpreter->Invoke();
        end = measure::now();
        
cout << "Inference time " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0 << endl;
      		float* boxes = interpreter->tensor(interpreter->outputs()[0])->data.f;
    		float* classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    		float* scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    		int    num = *interpreter->tensor(interpreter->outputs()[3])->data.f;
	begin2 = measure::now();
      		vector<cv::Rect> locations;
      		vector<vector<float>> people_location;
      		
      		for (int i=0;i<num;i=i+1)
      		{      
      			if (1)//((classes[i]==0) && (scores[i]>0.5f))//(strcmp(Labels[classes[i]],person.c_str())==0)
      			{
      		         //   cout << Labels[classes[i]] << "\t" << scores[i] <<  endl;		
      		            float ymin=boxes[4*i  ]*fheight;//ymin
			    float xmin=boxes[4*i+1]*fwidth;//xmin
			    float ymax=boxes[4*i+2]*fheight;//ymax
			    float xmax=boxes[4*i+3]*fwidth;//xmax  (((xmax-xmin)<=fwidth)? (xman-xmin) : fwidth)
			  //  cout << ymin << "\t" << xmin << "\t" << ymax << "\t" << xmax << endl;
			    cv::Rect rec((int)((xmin>=0)? xmin : 0), (int)((ymin>=0)? ymin : 0), (int)(((xmax-xmin)<=fwidth)? (xmax-xmin) : fwidth), (int)(((ymax-ymin)<=fheight)? (ymax-ymin) : fwidth));
			    locations.push_back(rec);
			    begin = measure::now();
			    if (person_check(rec)==1)
			    {
			        vector<float> pos; 
			        pos.push_back(x_mean); pos.push_back(y_mean); 
			        pos.push_back(z_mean); pos.push_back(d_mean);
			    	people_location.push_back(pos);
			    }
			    end = measure::now();
       		            cout << "One Distance estimation time " << (std::chrono::duration_cast<std::chrono::microseconds>(begin-end).count())/1000000.0 << endl;
			    cv::rectangle(frame,rec,Scalar(0,255,255),1);//Labels[classes[i]]+
			    putText(frame, std::to_string((int)(scores[i]*100))+" "+std::to_string(d_mean), Point(xmin, ymin+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
      			}
      		}
	end2 = measure::now();
cout << "Total distance estimation time " << (std::chrono::duration_cast<std::chrono::microseconds>(begin2-end2).count())/1000000.0 << endl;
      		cvtColor(frame, frame, COLOR_RGB2BGR);
      		putText(frame,"i can see",Point((int)(fwidth/2),(int)(fheight/2)),FONT_HERSHEY_SIMPLEX,1,Scalar(0,255,255),2);
      		imshow("main_frame_tf",frame);
                waitKey(1);   		
      		//TfLiteTensor* output_locations = nullptr;
      		//TfLiteTensor* output_classes = nullptr;
      		//TfLiteTensor* num_detection = nullptr;
		ros::spinOnce();
           	r.sleep();
	}
}
