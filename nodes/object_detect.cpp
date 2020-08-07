#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Header.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h> // from ROS image type to OpenCV image type

#include <fstream>
#include <utility>
#include <vector>
#include <iostream> // input and output..

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

// standardOpenCV libraries
#include <opencv2/core/mat.hpp>
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
cv::Mat image, frame;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    Mat temp = cv_bridge::toCvShare(msg, "bgr8")->image;
    Mat image_raw = temp.clone();
    image = image_raw.clone();

    int width     = image_raw.size().width; //width = col = x;
    int height    = image_raw.size().height; //height = row = y;

    if (is_first_image)
       is_first_image = false;
  }
  catch (cv_bridge::Exception& e)
  {   
    cout << "error" << endl;
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str()); 
  }
}

int main(int argc, char* argv[]) {

    ros::init(argc, argv, "tf_detect_node");

    ros::NodeHandle n1;
    ros::NodeHandle private_nh_("~");

    ros::Rate loop_rate(10);

    image_transport::ImageTransport it(n1);
    image_transport::Subscriber sub = it.subscribe("cam_in", 2, imageCallback); // /stereo_cam_node/image_raw

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
    
    frame = image.clone();

    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim(static_cast<myINT64>(frame.size().height));
    shape.AddDim(static_cast<myINT64>(frame.size().width));
    shape.AddDim(3);
    while (ros::ok()){    // && (cap.isOpened())) {
       frame = image.clone();
       // resize(frame,frame,Size(640,480));

       if (frame.empty())
        {
            cout << "problem" << endl;
            break;
        }

        cvtColor(frame,frame,COLOR_BGR2RGB);
                                              
        if (nFrames % (iFrame + 1) == 0) {
            time(&end);
            fps = 1. * nFrames / difftime(end, start);
            time(&start);
        }
        iFrame++;
        // Convert mat to tensor
        tensor = Tensor(tensorflow::DT_FLOAT, shape);
        Status readTensorStatus = readTensorFromMat(frame, tensor);
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
        vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore); //Intersection over Union
        // for (size_t i = 0; i < goodIdxs.size(); i++)
        //     LOG(INFO) << "score:" << scores(goodIdxs.at(i)) << ",class:" << labelsMap[classes(goodIdxs.at(i))]
        //               << " (" << classes(goodIdxs.at(i)) << "), box:" << "," << boxes(0, goodIdxs.at(i), 0) << ","
        //               << boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
        //               << boxes(0, goodIdxs.at(i), 3);
        
        // Draw bboxes and captions
        cvtColor(frame,frame,COLOR_RGB2BGR);
        drawBoundingBoxesOnImage(frame, scores, classes, boxes, labelsMap, goodIdxs);
        putText(frame, to_string(fps).substr(0, 5), Point(0, frame.rows), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255));
        imshow("main_frame_tf",frame);
        waitKey(1);
        // boxes(0,idxs.at(j),0) = ymin
        // boxes(0,idxs.at(j),1) = xmin
        // boxes(0,idxs.at(j),2) = ymax
        // boxes(0,idxs.at(j),3) = xmax
    
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
