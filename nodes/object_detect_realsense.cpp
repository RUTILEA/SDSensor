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

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <time.h>

#include <stdlib.h>

#include "utils_old.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;
using namespace tensorflow;

std_msgs::Header header_org, header;

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

sensor_msgs::CameraInfo cil, cir;

bool dist_valid = false;

int height_use;// = disparity_.image.height;
int width_use;// = disparity_.image.width;

Mat image_depth_org, image_depth;
double focus_x = 0;
double focus_y = 0;
double center_x = 0;
double center_y = 0;
bool is_first_depth = true;

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


void set_roi(sensor_msgs::RegionOfInterest *roi, int x_off, int y_off, int width, int height)
{
    roi->x_offset = x_off;
    roi->y_offset = y_off;
    roi->width = width;
    roi->height = height;
}

int count_nonzero(cv::Mat result)
{
    int i,j;
    int count = 0;
    int x = 0;
    int y = 0;
    for (i=0;i<result.size[0];i++)
    {
        for (j=0;j<result.size[1];j++)
        {
            if (result.at<int>(i,j)!=0)
                 count = count + 1;
                 x = x + i;
                 y = y + j;
        }
    }
    if (count>0)
    {
        float centroidx = ((float)x)/count;
        float centroidy = ((float)y)/count;
    }
    return count;
}

vector<float> find_mean(Mat mask)
{
    int x = 0;
    int y = 0;
    int count = 0;
    for (int i=0;i<mask.size[0];i++)
    {
        for (int j=0;j<mask.size[1];j++)
        {   
             if (mask.at<int>(i,j)!=0)
             {
                x = x + i;
                y = y + j;
                count = count + 1;
             }
        }
    }
    vector<float> mean;
    mean.push_back((float)x/count);
    mean.push_back((float)y/count);
    return mean;

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

bool frame_person_pos_check(Mat check, sensor_msgs::RegionOfInterest roi, float w_thr, float h_thr)
{
    int width_check = check.size().width;
    int height_check = check.size().height;
    float thr_width = w_thr*width_check;
    float thr_height = h_thr*height_check;
    if (sqrt( ((roi.x_offset+roi.width/2.0f)-width_check/2.0f)*((roi.x_offset+roi.width/2.0f)-width_check/2.0f)  + ((roi.y_offset+roi.height/2.0f)-height_check/2.0f)*((roi.y_offset+roi.height/2.0f)-height_check/2.0f) ) < 60.0f )
    {
        cout << "passed 1" << endl; 
        if ( (roi.x_offset>=thr_width) && (roi.y_offset>=thr_height) && ( (roi.x_offset+roi.width) <= (width_check-thr_width) ) &&  ( (roi.y_offset+roi.height) <= (height_check-thr_height) )  )
         return true;
        else
        {
         return false;
         cout << "failed 2" << endl;
        }
    }
    else 
    {
        cout << "failed 1" << endl;
        return false;
    }
}

double x_mean, y_mean,z_mean;

int person_check(Mat frame_check, sensor_msgs::RegionOfInterest roi)
{

    Mat image_target_range = Mat(frame_check.rows, frame_check.cols, CV_8UC1, cv::Scalar(0));

    vector<int> r, g, b;

    std::vector<Point2f> scene_corners_p;    
    scene_corners_p.push_back(Point2f(roi.x_offset,roi.y_offset));
    scene_corners_p.push_back(Point2f(roi.x_offset+roi.width,roi.y_offset));
    scene_corners_p.push_back(Point2f(roi.x_offset+roi.width,roi.y_offset+roi.height));
    scene_corners_p.push_back(Point2f(roi.x_offset,roi.y_offset+roi.height));

    cv::Point point2[1][4];
    point2[0][0] = scene_corners_p[0];
    point2[0][1] = scene_corners_p[1];
    point2[0][2] = scene_corners_p[2];
    point2[0][3] = scene_corners_p[3];
    const cv::Point *pts2[] = { point2[0] };
    int npts2[] = { 4 };
    cv::fillPoly(image_target_range, pts2, npts2, 1, cv::Scalar(255), cv::LINE_8);

    // imshow("image_trans", image_target_range);
    // waitKey(1);

    //read the depth from the depth image
    x_mean = 0;
    y_mean = 0;
    z_mean = 0;
    int pixel_count = 0;

    Mat img_depth_in = image_depth.clone();

    vector<float> x, y, z;
    for(int row=0; row<image_target_range.rows; row++)
    {
        const float* pdata_ptr = img_depth_in.ptr<float>(row);
        const unsigned char* rdata_ptr = image_target_range.ptr<unsigned char>(row);
        for(int col=0; col<image_target_range.cols; col++)
        {
            const unsigned char rdata = rdata_ptr[col];
            if(rdata != 255)
            { continue; }
            const float pdata = pdata_ptr[col];
            if(pdata < 0.00001)
            { continue; }
        if (pdata>1)
            continue;
            const float z_pdata = pdata;
            const float x_pdata = ((float)col-center_x) * z_pdata / focus_x;
            const float y_pdata = ((float)row-center_y) * z_pdata / focus_y;
            x_mean += x_pdata;
            y_mean += y_pdata;
            z_mean += z_pdata;
            x.push_back(x_pdata);
            y.push_back(y_pdata);
            z.push_back(z_pdata);
            r.push_back(frame_check.at<cv::Vec3b>(row,col)[0]);
            g.push_back(frame_check.at<cv::Vec3b>(row,col)[1]);
            b.push_back(frame_check.at<cv::Vec3b>(row,col)[2]);
            pixel_count++;
        }
    }    

    if(pixel_count == 0)
        return 0;

    x_mean /= pixel_count;
    y_mean /= pixel_count;
    z_mean /= pixel_count;


    float x_size = maximum_element(x) - minimum_element(x);
    float y_size = maximum_element(y) - minimum_element(y);
    float z_size = maximum_element(z) - minimum_element(z);

    cout << "person " << "x: " << x_size << " y: " << y_size << " z: " << z_size << " z_mean " << z_mean << endl;

    if ( fabsf(z_size-z_mean)>1.0 )
    {
        // ROS_WARN("current size method not working good");
        return 1;
    }

    return 1;

    // if ( (x_size>=0.07) && (x_size<=0.25) && (y_size>=0.20) && (y_size<=0.6) ) //no x and y check for person. just trust
    // {}
    // if ((z_mean<1.5f) && z_mean >0.8f)
    // {
    //     if (frame_person_pos_check(frame_check, roi, 0.2f,0.2f))
    //     {
    //         return 2;  //baby doll!
    //     }
    // }
    // if (z_mean<3.0f) //don't bother with baby doll at this distance
    // {
    //     if ( (roi.width>(frame_check.size().width/2.0f)) || (roi.height>(frame_check.size().height/2.0f)) )
    //     return 1;  //big human
    // }   
    // else
    // {
    //     ROS_INFO("size not within limits");
    //     return 0;
    // }

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



int main(int argc, char* argv[]) {

    ros::init(argc, argv, "tf_detect_realsense_node");

    ros::NodeHandle n1;
    ros::NodeHandle private_nh_("~");

    ros::Rate loop_rate(10);

    image_transport::ImageTransport it(n1);
    image_transport::Subscriber sub = it.subscribe("cam_in", 2, imageCallback); 
    image_transport::CameraSubscriber depth_sub_ = it.subscribeCamera("depth_in", 1, &depth_Callback);

    
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

    cout << "GOING TO START LOOP" << endl;
    while (ros::ok()){    // && (cap.isOpened())) {

       frame = image.clone();
       image_depth = image_depth_org.clone();
       // resize(frame_left,frame_left,Size(640,480));
       // resize(frame_right,frame_right,Size(640,480));

       std_msgs::Header header;
       header.seq = 1;
       header.frame_id = "tf_object";
       header.stamp = ros::Time::now();
       
       
       if (frame.empty())
        {
            ROS_ERROR("Frame empty: could not initialize properly");
            break;
        }

        cvtColor(frame, frame, COLOR_BGR2RGB);

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
        vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
        // for (size_t i = 0; i < goodIdxs.size(); i++)
        //     LOG(INFO) << "score:" << scores(goodIdxs.at(i)) << ",class:" << labelsMap[classes(goodIdxs.at(i))]
        //               << " (" << classes(goodIdxs.at(i)) << "), box:" << "," << boxes(0, goodIdxs.at(i), 0) << ","
        //               << boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
        //               << boxes(0, goodIdxs.at(i), 3);
        
        // Draw bboxes and captions
        cvtColor(frame, frame, COLOR_RGB2BGR);
        Mat frame_copy = frame.clone();
        // boxes(0,idxs.at(j),0) = ymin
        // boxes(0,idxs.at(j),1) = xmin
        // boxes(0,idxs.at(j),2) = ymax
        // boxes(0,idxs.at(j),3) = xmax     

        vector<int> personIdxs;   
        sensor_msgs::RegionOfInterest ROI;

        int width = frame.size().width;
        int height = frame.size().height;

        vector<sensor_msgs::RegionOfInterest> people;
        string person = "person";
        vector<vector<float>> people_location;

        for (size_t i = 0; i < goodIdxs.size(); i++)
        {
            if (strcmp(labelsMap[classes(goodIdxs.at(i))].c_str(),person.c_str())==0)
            {
                set_roi(&ROI,(int)(boxes(0, goodIdxs.at(i), 1)*width),(int)(boxes(0, goodIdxs.at(i), 0 )*height),(int)(boxes(0,goodIdxs.at(i),3)*width-boxes(0,goodIdxs.at(i),1)*width),(int)(boxes(0,goodIdxs.at(i),2)*height-boxes(0,goodIdxs.at(i),0)*height));  
                        if (person_check(frame_copy,ROI)==1)//(check_distance(frame)dist<dist_thr) 
                        {
                                personIdxs.push_back(goodIdxs.at(i));  
                                people.push_back(ROI);  
                                vector<float> pos; pos.push_back(x_mean); pos.push_back(y_mean); pos.push_back(z_mean);
                                people_location.push_back(pos);                                 
                        }
                        else
                        {
                            ROS_ERROR("person parameters wrong");
                        }
            }
        }

        for (size_t i = 0; i < people.size(); i++)
        {           
           ROI = people[i];
           vector<float> location = people_location[i];
           drawBoundingBoxOnImage(frame_copy,
                               boxes(0,personIdxs.at(i),0), boxes(0,personIdxs.at(i),1),
                               boxes(0,personIdxs.at(i),2), boxes(0,personIdxs.at(i),3),
                               scores(personIdxs.at(i)), ( labelsMap[classes(personIdxs.at(i))] + " " + std::to_string(location[2]) + " m"), true );
        } 

        putText(frame_copy, to_string(fps).substr(0, 5), Point(0, frame.rows), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255));
        imshow("main_frame_tf",frame_copy);
        waitKey(1);
    
        ros::spinOnce();
        loop_rate.sleep();
    }

// destroyAllWindows();

    return 0;
}
