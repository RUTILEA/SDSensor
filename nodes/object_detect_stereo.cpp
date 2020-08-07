#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <stereo_msgs/DisparityImage.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <time.h>

#include "utils_old.h"

#include "tf_ros_detection/StereoDepth.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;
using namespace tensorflow;


//http://forums.codeguru.com/showthread.php?277995-convert-int32-to-int64-HOW
//error in (int64) data type conversion in original code
#if defined (_MSC_VER)

  typedef signed __int64 myINT64;

#elif defined(__GNUC__)

  typedef signed long long  myINT64;

#else

  #error Compiler not supported.

#endif

bool is_first_image = true;
cv::Mat image_right, image_left, frame_right, frame_left;

sensor_msgs::CameraInfo cil, cir;

bool dist_valid = false;

int height_use;// = disparity_.image.height;
int width_use;// = disparity_.image.width;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    Mat temp = cv_bridge::toCvShare(msg, "bgr8")->image;
    Mat image_raw = temp.clone();

    int width_stereo     = image_raw.size().width; //width = col = x;
    int height_stereo    = image_raw.size().height; //height = row = y;
    int width_single     = width_stereo/2;
    int height_single    = height_stereo;
    
    Mat image_right1, image_left1;
    image_right1  = image_raw(Range(0, height_single), Range(width_single, width_stereo));
    image_left1  = image_raw(Range(0, height_single), Range(0, width_single));

    image_right = image_right1.clone();
    image_left = image_left1.clone();

    if (is_first_image)
       is_first_image = false;
  }
  catch (cv_bridge::Exception& e)
  {   
    cout << "error" << endl;
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str()); 
  }
}

void leftinfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
    cil = *msg;
}

void rightinfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
    cir = *msg;
}

void set_roi(sensor_msgs::RegionOfInterest *roi, int x_off, int y_off, int width, int height)
{
    roi->x_offset = x_off;
    roi->y_offset = y_off;
    roi->width = width;
    roi->height = height;
}
                                                     
float maximum_element(vector<float> xx)
{
    float maximum = xx[0];
    for (int ii = 0; ii < xx.size(); ii++)
    {
        if (xx[ii]>maximum)
            maximum = xx[ii];
    }
    return ((float)maximum);
}

float minimum_element(vector<float> xx)
{
    float minimum = xx[0];
    for (int ii = 0; ii < xx.size(); ii++)
    {
        if (xx[ii]<minimum)
            minimum = xx[ii];
    }
    return ((float)minimum);
}

float find_sum(vector<float> zz)
{
    float sum = 0.0f;
    for (int ii = 0; ii < zz.size(); ii++)
    {
        sum = sum + zz[ii];
    }
    return ((float)sum);
}

vector<float> get_location(cv::Mat_<float> dmat, sensor_msgs::RegionOfInterest ROI, Eigen::MatrixXf Q_, float min_value)
{
    vector<float> x;
    vector<float> y;
    vector<float> z;

    int coun = 0;
    for (int ii=ROI.y_offset; ii<(ROI.y_offset + ROI.height); ii++)
   {
        for (int jj=ROI.x_offset; jj<(ROI.x_offset + ROI.width); jj++)
        {
            if (((dmat[ii][jj]-min_value)>2) && (dmat[ii][jj]>2))
            {
                int v = ii; int u = jj; 
                Eigen::Vector3f XYZ(u + Q_(0,3), v + Q_(1,3), Q_(2,3)); //cv::Point3d
                double W = Q_(3,2)*dmat[v][u] + Q_(3,3);

                Eigen::Vector3f xyz = XYZ * (1.0/W); //Point3d
                x.push_back(xyz[0]);y.push_back(xyz[1]);z.push_back(xyz[2]);
                coun = coun + 1;
            }    
        }
   }
   vector<float> xyz_;
   if (coun>10)
   {
        dist_valid = true;

        xyz_.push_back(maximum_element(x) - minimum_element(x));
        xyz_.push_back(maximum_element(y) - minimum_element(y));
        xyz_.push_back( find_sum(z)/z.size() );
        xyz_.push_back( find_sum(x)/x.size() );
        xyz_.push_back( find_sum(y)/y.size() );
   }
   else
   {
        xyz_.push_back(0.0f);xyz_.push_back(0.0f);xyz_.push_back(0.0f);xyz_.push_back(0.0f);xyz_.push_back(0.0f);
        dist_valid = false;
   }
   return xyz_;
}


int main(int argc, char* argv[]) {

    ros::init(argc, argv, "tf_detect_stereo_node");

    ros::NodeHandle n1;
    ros::NodeHandle private_nh_("~");

    ros::Rate loop_rate(10);

    image_transport::ImageTransport it(n1);
    image_transport::Subscriber sub = it.subscribe("cam_in", 2, imageCallback); // /stereo_cam_node/image_raw
    // image_transport::Subscriber sub = it.subscribe("/stereo_cam_node/left/image_raw", 2, leftimageCallback);
    ros::Subscriber sub_left = n1.subscribe("cam_left_info",1,&leftinfoCallback);// /stereo_cam_node/left/camera_info
    ros::Subscriber sub_right = n1.subscribe("cam_right_info",1,&rightinfoCallback);// /stereo_cam_node/right/camera_info
    ros::ServiceClient client = n1.serviceClient<tf_ros_detection::StereoDepth>("get_depth_map");

    tf_ros_detection::StereoDepth srv;
    std::string root_dir_, labels_dir_, models_dir_;
    double thresholdScore = 0.7;
    double thresholdIOU = 0.8;
    private_nh_.getParam("root_dir",root_dir_);
    private_nh_.getParam("labels_dir",labels_dir_);
    private_nh_.getParam("models_dir",models_dir_);
    private_nh_.getParam("threshold_Score",thresholdScore);
    private_nh_.getParam("threshold_IOU",thresholdIOU);

    // Set dirs variables
    string ROOTDIR = root_dir_;
    string LABELS = labels_dir_;
    string GRAPH = models_dir_;

    // Set input & output nodes names
    string inputLayer = "image_tensor:0";
    vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

    // Load and initialize the model from .pb file
    std::unique_ptr<tensorflow::Session> session;
    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;
    

    // Load labels map from .pbtxt file
    std::map<int, std::string> labelsMap = std::map<int,std::string>();
    Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
    if (!readLabelsMapStatus.ok()) {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;

    // Mat frame;
    Tensor tensor;
    std::vector<Tensor> outputs;

    // FPS count
    int nFrames = 2;
    int iFrame = 0;
    double fps = 0.;
    time_t start, end;
    time(&start);

    //wait till we receive images
    while (ros::ok() && is_first_image) {
        ros::spinOnce();
        loop_rate.sleep();
    }
    Mat temp;
    
    frame_right = image_right.clone();
    frame_left = image_left.clone();

    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim(static_cast<myINT64>(frame_right.size().height));
    shape.AddDim(static_cast<myINT64>(frame_right.size().width));
    shape.AddDim(3);
    while (ros::ok()){    // && (cap.isOpened())) {
       frame_right = image_right.clone();
       frame_left = image_left.clone();
       // resize(frame_left,frame_left,Size(640,480));
       // resize(frame_right,frame_right,Size(640,480));

       std_msgs::Header header;
       header.seq = 1;
       header.frame_id = "tf_object";
       header.stamp = ros::Time::now();
       sensor_msgs::ImagePtr img_left = cv_bridge::CvImage(header, "bgr8", image_left.clone()).toImageMsg();
       sensor_msgs::ImagePtr img_right = cv_bridge::CvImage(header, "bgr8", image_right.clone()).toImageMsg();

       if (frame_right.empty())
        {
            ROS_ERROR("Frame empty: could not initialize properly");
            break;
        }

        srv.request.left_image  = *(cv_bridge::CvImage(header, "bgr8", frame_left).toImageMsg());
        srv.request.right_image =  *(cv_bridge::CvImage(header, "bgr8", frame_right).toImageMsg());//frame;
        cvtColor(frame_left, frame_left, COLOR_BGR2RGB);
        cvtColor(frame_right,frame_right,COLOR_BGR2RGB);

        if (nFrames % (iFrame + 1) == 0) {
            time(&end);
            fps = 1. * nFrames / difftime(end, start);
            time(&start);
        }
        iFrame++;
        // Convert mat to tensor
        tensor = Tensor(tensorflow::DT_FLOAT, shape);
        Status readTensorStatus = readTensorFromMat(frame_left, tensor);
        if (!readTensorStatus.ok()) {
            LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
            return -1;
        }
        // Run the graph on tensor
        outputs.clear();
        Status runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
        if (!runStatus.ok()) {
            LOG(ERROR) << "Running model failed: " << runStatus;
            return -1;
        }       
        // Extract results from the outputs vector
        tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
        tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
        tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
        tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();
        vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
        // for (size_t i = 0; i < goodIdxs.size(); i++)
        //     LOG(INFO) << "score:" << scores(goodIdxs.at(i)) << ",class:" << labelsMap[classes(goodIdxs.at(i))]
        //               << " (" << classes(goodIdxs.at(i)) << "), box:" << "," << boxes(0, goodIdxs.at(i), 0) << ","
        //               << boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
        //               << boxes(0, goodIdxs.at(i), 3);
        
        // Draw bboxes and captions
        cvtColor(frame_left, frame_left, COLOR_RGB2BGR);
        cvtColor(frame_right,frame_right,COLOR_RGB2BGR);
        // boxes(0,idxs.at(j),0) = ymin
        // boxes(0,idxs.at(j),1) = xmin
        // boxes(0,idxs.at(j),2) = ymax
        // boxes(0,idxs.at(j),3) = xmax
    
        srv.request.left_camera_info = cil;
        srv.request.right_camera_info = cir;
        if (!client.call(srv))
        {
            cout << "error" << endl;
            ROS_ERROR("Could not convert get disparity map"); 
        }
        

        Rect2d bbox;
        
        stereo_msgs::DisparityImage disparity_ = srv.response.DisparityValue;

        const cv::Mat_<float> dmat(disparity_.image.height, disparity_.image.width,
                               (float*)&disparity_.image.data[0], disparity_.image.step);

         //Depth Calculation
        Eigen::MatrixXf Q_(4,4);
        Q_ = Eigen::MatrixXf::Zero(4,4);
        Q_(0,0) = 1.0f;
        Q_(0,3) = -srv.response.cx_r.data;
        Q_(1,1) = 1.0f;
        Q_(1,3) = -srv.response.cy_r.data;//model_.left().cy();
        Q_(2,3) = disparity_.f;
        Q_(3,2) = 1.0/disparity_.T;
        Q_(3,3) = 0.0f;//(srv.response.cx_r.data-srv.response.cx_l.data)/disparity_.T;//disparity already subtracted before publishing in StereoDepth service


        float min_disp_value = dmat[0][0];
        for (int i =0;i<disparity_.image.height;i++)
        {
            for (int j=0;j<disparity_.image.width;j++)
            {
                if (((float)dmat[i][j])<min_disp_value)
                    min_disp_value = (float)dmat[i][j];
            }
        }

        height_use = disparity_.image.height;
        width_use = disparity_.image.width;

        sensor_msgs::RegionOfInterest ROI;        
        vector<float> location;
      
        for (size_t i = 0; i < goodIdxs.size(); i++)
        {                   
            set_roi(&ROI,(int)(boxes(0, goodIdxs.at(i), 1)*width_use),(int)(boxes(0, goodIdxs.at(i), 0 )*height_use),(int)(boxes(0,goodIdxs.at(i),3)*width_use-boxes(0,goodIdxs.at(i),1)*width_use),(int)(boxes(0,goodIdxs.at(i),2)*height_use-boxes(0,goodIdxs.at(i),0)*height_use));  
            location = get_location(dmat, ROI, Q_,min_disp_value);
            drawBoundingBoxOnImage(frame_left,
                               boxes(0,goodIdxs.at(i),0), boxes(0,goodIdxs.at(i),1),
                               boxes(0,goodIdxs.at(i),2), boxes(0,goodIdxs.at(i),3),
                               scores(goodIdxs.at(i)), ( labelsMap[classes(goodIdxs.at(i))] + " " + std::to_string(location[2]) + " m"), true );
        } 

        putText(frame_left, to_string(fps).substr(0, 5), Point(0, frame_left.rows), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255));
        imshow("main_frame_tf",frame_left);
        waitKey(1);
    
        ros::spinOnce();
        loop_rate.sleep();
    }

// destroyAllWindows();

    return 0;
}
