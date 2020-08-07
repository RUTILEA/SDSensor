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

#include <tf_ros_detection/keypoints.h>
#include <tf_ros_detection/people.h>

#include "tensorflow/lite/builtin_op_data.h"

using namespace std;
using namespace cv;

using tflite::ops::builtin::BuiltinOpResolver;

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


// Resizes BMP image.
void ResizeImage(const coral::ImageDims& in_dims, const uint8_t* in,
                 const coral::ImageDims& out_dims, uint8_t* out) {
  const int image_height = in_dims[0];
  const int image_width = in_dims[1];
  const int image_channels = in_dims[2];
  const int wanted_height = out_dims[0];
  const int wanted_width = out_dims[1];
  const int wanted_channels = out_dims[2];
  const int number_of_pixels = image_height * image_width * image_channels;
  if (image_height == wanted_height && image_width == wanted_width &&
      image_channels == wanted_channels) {
    VLOG(1) << "No resizing needed for input image.";
    std::memcpy(out, in, number_of_pixels);
    return;
  }
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);
  int base_index = 0;
  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});
  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);
  BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  //params->half_pixel_centers = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);
  interpreter->AllocateTensors();
  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }
  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;
  interpreter->Invoke();
  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels =
      wanted_height * wanted_height * wanted_channels;
  for (int i = 0; i < output_number_of_pixels; i++) {
    out[i] = static_cast<uint8_t>(output[i]);
  }
}

int main(int argc, char** argv) {
  ros::init(argc,argv, "coral_test_detect");
    ros::NodeHandle n;
    ros::NodeHandle private_nh_("~");
    
    image_transport::ImageTransport it(n);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 
    ros::Publisher pub_target = n.advertise<tf_ros_detection::people>("/people",10);
    ros::Rate r(20);
    
    tf_ros_detection::people detections = coral::TestPoseNetDecoder();  
    
    // Load the model.
  std::string model_path = "/home/ubuntu/catkin_p3/src/test_p3/src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite";
  std::string image_path = "/home/ubuntu/Downloads/i4.jpeg";
  std::string ssd_path = "/home/ubuntu/codes/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";//"/home/ubuntu/Downloads/coco_mobile/detect_int_edgetpu.tflite";
  
  private_nh_.getParam("ssd_path",ssd_path);
  private_nh_.getParam("model_path",model_path);
  
  //  image_path = "/home/ubuntu/Downloads/i4.bmp";
  LOG(INFO) << "Testing model: " << model_path;
  coral::BasicEngine engine(model_path);
  coral::DetectionEngine engine_detect(ssd_path);
  std::vector<int> input_tensor_shape = engine_detect.get_input_tensor_shape();
  std::vector<int> input_tensor_shape_pose = engine.get_input_tensor_shape();
  
  float score_threshold = 0.3f;
  private_nh_.getParam("threshold_Score",score_threshold);
  
  int use_image = 0;
  private_nh_.getParam("use_image",use_image);
  private_nh_.getParam("image_path",image_path);
  
  int top_k = 1;
  private_nh_.getParam("top_k",top_k);
  
  //wait till we receive images
  if (!use_image) {
  while (ros::ok() && is_first_image) {
       	ros::spinOnce();
       	r.sleep();
  }
  }
  else
  {
      image = imread(image_path);
  }
  
  int fwidth = image.size().width;
  int fheight = image.size().height;
  
  while(n.ok()){
  chrono::steady_clock::time_point Tbegin, Tend;
  Tbegin = chrono::steady_clock::now(); 
	  frame = image.clone();
	  Mat frame_org = frame.clone();
	  Mat frame_pose = frame_org.clone();
	  resize(frame,frame,Size(input_tensor_shape[2],input_tensor_shape[1])); // (width,height)
	  cvtColor(frame, frame, COLOR_BGR2RGB);
	  frame.convertTo(frame, CV_8UC3);
	  
	   std::vector<uint8_t> input(frame.data, frame.data + (frame.cols * frame.rows * frame.elemSize()));
	
	  CHECK(!input.empty()) << "Input image path: " << image_path;
	  // Get result.
	  
	  auto candidates = engine_detect.DetectWithInputTensor(input,score_threshold,top_k);

	  Mat frame1;
	  resize(frame_org,frame1,Size(input_tensor_shape_pose[2],input_tensor_shape_pose[1]));
	  cvtColor(frame_pose, frame_pose, COLOR_BGR2RGB);
	  std::vector<uint8_t> input_pose(frame1.data, frame1.data + (frame1.cols * frame1.rows * frame1.elemSize()));
	  const std::vector<std::vector<float>>& raw_outputs = engine.RunInference(input_pose);

	  for (int i=0;i<candidates.size();i++)
	  {
	  	coral::DetectionCandidate result = candidates[i];
	  	int xmax = result.corners.ymax*fwidth;
	  	int xmin = result.corners.ymin*fwidth;
	  	int ymax = result.corners.xmax*fheight;
	  	int ymin = result.corners.xmin*fheight;
	  //	cout << result.label << "\t" << result.score << "\t" << result.corners.ymax << "\t" << result.corners.ymin << "\t" << result.corners.xmax << "\t" << result.corners.xmin << endl; // result.score;
	  	//cout << result.label << "\t" << result.score << "\t" << xmin << "\t" << ymin << "\t" << xmax << "\t" << ymax << endl;
	  	//if (!(result.label==0))
	  	//	continue;
	  	cv::Rect rec((int)((xmin>=0)? xmin : 0), (int)((ymin>=0)? ymin : 0), (int)(((xmax-xmin)<=fwidth)? (xmax-xmin) : fwidth), (int)(((ymax-ymin)<=fheight)? (ymax-ymin) : fwidth));
	 	cv::rectangle(frame_org,rec,Scalar(0,255,0),2);//
              //  putText(frame, "Person " + std::to_string((int)(scores[i]*100)), Point(xmin, ymin+5) ,FONT_HERSHEY_SIMPLEX,0.7, cv::Scalar(0, 255, 255), 1, 8, 0);
	  }
	  
	  cout << "Size " << candidates.size() << endl;
	  resize(frame_org,frame_org,Size(960,720));
	  imshow("results detection",frame_org);
	//  imshow("results_org",frame_org);
	//  waitKey(1);
	  
	  const auto& poses = raw_outputs[0];
	  const auto& keypoint_scores = raw_outputs[1];
	  const auto& pose_scores = raw_outputs[2];
	  const auto& n_poses = raw_outputs[3];

	  std::cout << "poses: " << n_poses[0] << std::endl;
	  cvtColor(frame, frame, COLOR_RGB2BGR);
	  tf_ros_detection::people detections;
	  detections.num.data = 0;//n_poses[0];
	  cvtColor(frame, frame, COLOR_RGB2BGR);
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
	  			points.x.push_back(x/input_tensor_shape_pose[2]);
	  			points.y.push_back(y/input_tensor_shape_pose[1]);
	  			points.scores.push_back(point_score);
	  			circle(frame,pt,4,Scalar( 255, 255, 0 ),FILLED);
	  			pt.x=(int)(x/input_tensor_shape_pose[2]*frame_pose.cols); pt.y=(int)(y/input_tensor_shape_pose[1]*frame_pose.rows);
	  			circle(frame_pose,pt,4,Scalar( 255, 255, 0 ),FILLED);
	  		}
	  		detections.persons.push_back(points);
	  	}
	  }
	  
	  imshow("results_pose",frame_pose);
	  waitKey(1);
	  
	  
  Tend = chrono::steady_clock::now();
  float f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
  cout << "Total Inference time " << f << endl;
	  
	  
	  
	  pub_target.publish(detections);
	    
	  ros::spinOnce();
	  r.sleep();
    
    }
    
    
    
}

/*   std::vector<uint8_t> input(frame.data, frame.data + (frame.cols * frame.rows * frame.elemSize()));
std::vector<uint8_t> result;
coral::ImageDims image_dims{ {fheight,fwidth,3} };
coral::ImageDims target_dims{ {input_tensor_shape[1],input_tensor_shape[2],3} };

result.resize(input_tensor_shape[2]*input_tensor_shape[1]*input_tensor_shape[3]);
ResizeImage(image_dims, input.data(), target_dims, result.data());
*/
