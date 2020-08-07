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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <float.h>

#include <list>
#include <iostream>
#include <chrono>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <thread>

#include "posenet_decoder_op.h"

using namespace cv;
using namespace std;
using namespace tflite;
using namespace coral;

#define MAX_POSE_NUM  10

#define POSENET_QUANT_MODEL_PATH  "/home/ubuntu/catkin_p3/src/test_p3/src/models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder.tflite"//"/home/ubuntu/codes/TensorFlow_Lite_Pose_RPi_64-bits/posenet_mobilenet_v1_101_300x220_stride16.tflite"//"/home/ubuntu/codes/TensorFlow_Lite_Pose_RPi_64-bits/posenet_mobilenet_v1_101_200x150_stride8.tflite"//"/home/ubuntu/codes/TensorFlow_Lite_Pose_RPi_64-bits/posenet_mobilenet_v1_101_200x150_stride16.tflite"// "/home/ubuntu/codes/TensorFlow_Lite_Pose_RPi_64-bits/posenet_mobilenet_v1_101_300x220_stride32.tflite"//"/home/ubuntu/codes/TensorFlow_Lite_Pose_RPi_64-bits/posenet_mobilenet_v1_101_300x300.tflite"//"/home/ubuntu/codes/TensorFlow_Lite_Pose_RPi_64-bits/posenet_mobilenet_v1_101_513x513.tflite"
// 330x200_32 fast but not accurate, _16 fast and accurate. almost same speed
// 200x150_32 faster (max 0.3) but fcannot detect far away objects
//200x150_8 is slower. can take upto 0.5s and still cannot detect far away objects
#define POSENET_MODEL_PATH   "./posenet.tflite"

#define USE_QUANT_TFLITE_MODEL 0

cv::Mat image, frame, frame_copy;
bool is_first_image = true;

typedef struct tflite_interpreter_t
{
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter>     interpreter;
    tflite::ops::builtin::BuiltinOpResolver  resolver;
} tflite_interpreter_t;

typedef struct tflite_createopt_t
{
    int gpubuffer;
} tflite_createopt_t;

typedef struct tflite_tensor_t
{
    int         idx;        /* whole  tensor index */
    int         io;         /* [0] input_tensor, [1] output_tensor */
    int         io_idx;     /* in/out tensor index */
    TfLiteType  type;       /* [1] kTfLiteFloat32, [2] kTfLiteInt32, [3] kTfLiteUInt8 */
    void        *ptr;
    int         dims[4];
    float       quant_scale;
    int         quant_zerop;
} tflite_tensor_t;

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_heatmap;
static tflite_tensor_t      s_tensor_offsets;
static tflite_tensor_t      s_tensor_fw_disp;
static tflite_tensor_t      s_tensor_bw_disp;

static int     s_img_w = 0;
static int     s_img_h = 0;
static int     s_hmp_w = 0;
static int     s_hmp_h = 0;
static int     s_edge_num = 0;

typedef struct part_score_t {
    float score;
    int   idx_x;
    int   idx_y;
    int   key_id;
} part_score_t;

typedef struct keypoint_t {
    float pos_x;
    float pos_y;
    float score;
    int   valid;
} keypoint_t;


enum pose_key_id {
    kNose = 0,          //  0
    kLeftEye,           //  1
    kRightEye,          //  2
    kLeftEar,           //  3
    kRightEar,          //  4
    kLeftShoulder,      //  5
    kRightShoulder,     //  6
    kLeftElbow,         //  7
    kRightElbow,        //  8
    kLeftWrist,         //  9
    kRightWrist,        // 10
    kLeftHip,           // 11
    kRightHip,          // 12
    kLeftKnee,          // 13
    kRightKnee,         // 14
    kLeftAnkle,         // 15
    kRightAnkle,        // 16

    kPoseKeyNum


};

static int pose_edges[][2] =
{
    /* parent,        child */
    { kNose,          kLeftEye      },  //  0
    { kLeftEye,       kLeftEar      },  //  1
    { kNose,          kRightEye     },  //  2
    { kRightEye,      kRightEar     },  //  3
    { kNose,          kLeftShoulder },  //  4
    { kLeftShoulder,  kLeftElbow    },  //  5
    { kLeftElbow,     kLeftWrist    },  //  6
    { kLeftShoulder,  kLeftHip      },  //  7
    { kLeftHip,       kLeftKnee     },  //  8
    { kLeftKnee,      kLeftAnkle    },  //  9
    { kNose,          kRightShoulder},  // 10
    { kRightShoulder, kRightElbow   },  // 11
    { kRightElbow,    kRightWrist   },  // 12
    { kRightShoulder, kRightHip     },  // 13
    { kRightHip,      kRightKnee    },  // 14
    { kRightKnee,     kRightAnkle   },  // 15
};


typedef struct _pose_key_t
{
    float x;
    float y;
    float score;
} pose_key_t;

typedef struct _pose_t
{
    pose_key_t key[kPoseKeyNum];
    float pose_score;

    void *heatmap;
    int   heatmap_dims[2];  /* heatmap resolution. (9x9) */
} pose_t;

typedef struct _posenet_result_t
{
    int num;
    pose_t pose[MAX_POSE_NUM];
} posenet_result_t;

typedef struct fvec2
{
    float x, y;
} fvec2;

static tflite_interpreter_t s_detect_interpreter;
static tflite_tensor_t      s_detect_tensor_input;
static tflite_tensor_t      s_detect_tensor_scores;
static tflite_tensor_t      s_detect_tensor_bboxes;



static std::list<fvec2> s_anchors;

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



int
tflite_create_interpreter_from_file (tflite_interpreter_t *p, const char *model_path)
{
    p->model = FlatBufferModel::BuildFromFile (model_path);
    p->resolver.AddCustom(kPosenetDecoderOp, RegisterPosenetDecoderOp());
    InterpreterBuilder(*(p->model), p->resolver)(&(p->interpreter));
    
    
    p->interpreter->SetNumThreads(4);
    if (p->interpreter->AllocateTensors() != kTfLiteOk)
    {
        cout << "ERR: " << endl;
        return -1;
    }
    tflite::PrintInterpreterState(p->interpreter.get());
    return 0;
}


int
tflite_get_tensor_by_name (tflite_interpreter_t *p, int io, const char *name, tflite_tensor_t *ptensor)
{
    std::unique_ptr<Interpreter> &interpreter = p->interpreter;

    memset (ptensor, 0, sizeof (*ptensor));

    int tensor_idx;
    int io_idx = -1;
    int num_tensor = (io == 0) ? interpreter->inputs ().size() :
                                 interpreter->outputs().size();
       //  if (io==0)
        // {
       //  	cout << interpreter->tensor(interpreter->inputs ()[0])->type << endl;
       //  }

    for (int i = 0; i < num_tensor; i ++)
    {
        tensor_idx = (io == 0) ? interpreter->inputs ()[i] :
                                 interpreter->outputs()[i];

        const char *tensor_name = interpreter->tensor(tensor_idx)->name;
        cout << tensor_name << endl;
        if (strcmp (tensor_name, name) == 0)
        {
            cout << "found tensor: " << tensor_name << endl;
            io_idx = i;
            break;
        }
    }

    if (io_idx < 0)
    {
        cout << "can't find tensor:" << name << endl;
        return -1;
    }

    void *ptr = NULL;
    TfLiteTensor *tensor = interpreter->tensor(tensor_idx);
    switch (tensor->type)
    {
    case kTfLiteUInt8:
        ptr = (io == 0) ? interpreter->typed_input_tensor <uint8_t>(io_idx) :
                          interpreter->typed_output_tensor<uint8_t>(io_idx);
        break;
    case kTfLiteFloat32:
        ptr = (io == 0) ? interpreter->typed_input_tensor <float>(io_idx) :
                          interpreter->typed_output_tensor<float>(io_idx);
        break;
    default:
        cout << "ERR: " << endl; //"DBG_LOGE ("ERR: ", __FILE__, __LINE__)" << endl;
        return -1;
    }

    ptensor->idx    = tensor_idx;
    ptensor->io     = io;
    ptensor->io_idx = io_idx;
    ptensor->type   = tensor->type;
    ptensor->ptr    = ptr;
    ptensor->quant_scale = tensor->params.scale;
    ptensor->quant_zerop = tensor->params.zero_point;

    for (int i = 0; (i < 4) && (i < tensor->dims->size); i ++)
    {
        ptensor->dims[i] = tensor->dims->data[i];
    }

    return 0;
}

int
init_tflite_posenet(int use_quantized_tflite)
{
    const char *posenet_model;

    if (use_quantized_tflite)
    {
        posenet_model = POSENET_QUANT_MODEL_PATH;
        tflite_create_interpreter_from_file (&s_interpreter, posenet_model);
        tflite_get_tensor_by_name (&s_interpreter, 0, "image",              &s_tensor_input);
        tflite_get_tensor_by_name (&s_interpreter, 1, "heatmap",            &s_tensor_heatmap);
        tflite_get_tensor_by_name (&s_interpreter, 1, "offset_2",           &s_tensor_offsets);
        tflite_get_tensor_by_name (&s_interpreter, 1, "displacement_fwd_2", &s_tensor_fw_disp);
        tflite_get_tensor_by_name (&s_interpreter, 1, "displacement_bwd_2", &s_tensor_bw_disp);
    }
    else
    {
        posenet_model = POSENET_QUANT_MODEL_PATH;//POSENET_MODEL_PATH;
        tflite_create_interpreter_from_file (&s_interpreter, posenet_model);
        tflite_get_tensor_by_name (&s_interpreter, 0, "sub_2",                                  &s_tensor_input);
        tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/heatmap_2/BiasAdd",          &s_tensor_heatmap);
        tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/offset_2/BiasAdd",           &s_tensor_offsets);
        tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/displacement_fwd_2/BiasAdd", &s_tensor_fw_disp);
        tflite_get_tensor_by_name (&s_interpreter, 1, "MobilenetV1/displacement_bwd_2/BiasAdd", &s_tensor_bw_disp);
    }

    /* input image dimention */
    s_img_w = s_tensor_input.dims[2];
    s_img_h = s_tensor_input.dims[1];
    fprintf (stderr, "input image size: (%d, %d)\n", s_img_w, s_img_h);

    /* heatmap dimention */
    s_hmp_w = s_tensor_heatmap.dims[2];
    s_hmp_h = s_tensor_heatmap.dims[1];
    fprintf (stderr, "heatmap size: (%d, %d)\n", s_hmp_w, s_hmp_h);

    /* displacement forward vector dimention */
    s_edge_num = s_tensor_fw_disp.dims[3] / 2;

    return 0;
}

void *
get_posenet_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[2];
    *h = s_tensor_input.dims[1];
    return s_tensor_input.ptr;
}

static float
get_heatmap_score (int idx_y, int idx_x, int key_id)
{
    int idx = (idx_y * s_hmp_w * kPoseKeyNum) + (idx_x * kPoseKeyNum) + key_id;
    float *heatmap_ptr = (float *)s_tensor_heatmap.ptr;
 //   cout << "heatmap score: " << heatmap_ptr[idx] << endl;
    return heatmap_ptr[idx];
}

static void
get_displacement_vector (void *disp_buf, float *dis_x, float *dis_y, int idx_y, int idx_x, int edge_id)
{
    int idx0 = (idx_y * s_hmp_w * s_edge_num*2) + (idx_x * s_edge_num*2) + (edge_id + s_edge_num);
    int idx1 = (idx_y * s_hmp_w * s_edge_num*2) + (idx_x * s_edge_num*2) + (edge_id);

    float *disp_buf_fp = (float *)disp_buf;
    *dis_x = disp_buf_fp[idx0];
    *dis_y = disp_buf_fp[idx1];
}

static void
get_offset_vector (float *ofst_x, float *ofst_y, int idx_y, int idx_x, int pose_id)
{
    int idx0 = (idx_y * s_hmp_w * kPoseKeyNum*2) + (idx_x * kPoseKeyNum*2) + (pose_id + kPoseKeyNum);
    int idx1 = (idx_y * s_hmp_w * kPoseKeyNum*2) + (idx_x * kPoseKeyNum*2) + (pose_id);
    float *offsets_ptr = (float *)s_tensor_offsets.ptr;

    *ofst_x = offsets_ptr[idx0];
    *ofst_y = offsets_ptr[idx1];
}




/* resize image to DNN network input size and convert to fp32. */
void
feed_posenet_image(int win_w, int win_h, Mat& im)
{

    int x, y, w, h;
#if defined (USE_QUANT_TFLITE_MODEL)
    unsigned char *buf_u8 = (unsigned char *)get_posenet_input_buf (&w, &h);
    cout << "doing quant" << endl;
#else
    float *buf_fp32 = (float *)get_posenet_input_buf (&w, &h);
#endif
 //   unsigned char *buf_ui8 = NULL;
 //   static unsigned char *pui8 = NULL;

 //   if (pui8 == NULL)
 //       pui8 = (unsigned char *)malloc(w * h * 4);

 //   buf_ui8 = pui8;

    /* convert UI8 [0, 255] ==> FP32 [0, 1] */
    float mean =   0.0f;
    float std  = 255.0f;
    for (y = 0; y < h; y ++)
    {
        for (x = 0; x < w; x ++)
        {
            int r = im.at<cv::Vec3b>(y,x)[0];//*buf_ui8 ++; 
            int g = im.at<cv::Vec3b>(y,x)[1];//*buf_ui8 ++;
            int b = im.at<cv::Vec3b>(y,x)[2];//*buf_ui8 ++;
            //buf_ui8 ++;          /* skip alpha */
#if defined (USE_QUANT_TFLITE_MODEL)
            *buf_u8 ++ = r;
            *buf_u8 ++ = g;
            *buf_u8 ++ = b;
#else
            *buf_fp32 ++ = (float)(r - mean) / std;
            *buf_fp32 ++ = (float)(g - mean) / std;
            *buf_fp32 ++ = (float)(b - mean) / std;
#endif
        }
    }
    return;
}


static bool
score_is_max_in_local_window (int key, float score, int idx_y, int idx_x, int max_rad)
{
    int xs = std::max (idx_x - max_rad,     0);
    int ys = std::max (idx_y - max_rad,     0);
    int xe = std::min (idx_x + max_rad + 1, s_hmp_w);
    int ye = std::min (idx_y + max_rad + 1, s_hmp_h);

    for (int y = ys; y < ye; y ++)
    {
        for (int x = xs; x < xe; x ++)
        {
            /* if a higher score is found, return false */
            if (get_heatmap_score (y, x, key) > score)
                return false;
        }
    }
    return true;
}

/* enqueue an item in descending order. */
static void
enqueue_score (std::list<part_score_t> &queue, int x, int y, int key, float score)
{
    std::list<part_score_t>::iterator itr;
    for (itr = queue.begin(); itr != queue.end(); itr++)
    {
        if (itr->score < score)
            break;
    }

    part_score_t item;
    item.score = score;
    item.idx_x = x;
    item.idx_y = y;
    item.key_id= key;
    queue.insert(itr, item);
}

static void
build_score_queue (std::list<part_score_t> &queue, float thresh, int max_rad)
{
    for (int y = 0; y < s_hmp_h; y ++)
    {
        for (int x = 0; x < s_hmp_w; x ++)
        {
            for (int key = 0; key < kPoseKeyNum; key ++)
            {
                float score = get_heatmap_score (y, x, key);

                /* if this score is lower than thresh, skip this pixel. */
                if (score < thresh)
                    continue;

                /* if there is a higher score near this pixel, skip this pixel. */
                if (!score_is_max_in_local_window (key, score, y, x, max_rad))
                    continue;

                enqueue_score (queue, x, y, key, score);
            }
        }
    }
}

static void
get_index_to_pos (int idx_x, int idx_y, int key_id, float *pos_x, float *pos_y)
{
    float ofst_x, ofst_y;
    get_offset_vector (&ofst_x, &ofst_y, idx_y, idx_x, key_id);

    float rel_x = (float)idx_x / (float)(s_hmp_w -1);
    float rel_y = (float)idx_y / (float)(s_hmp_h -1);

    float pos0_x = rel_x * s_img_w;
    float pos0_y = rel_y * s_img_h;

    *pos_x = pos0_x + ofst_x;
    *pos_y = pos0_y + ofst_y;
}

static bool
within_nms_of_corresponding_point (posenet_result_t *pose_result,
                        float pos_x, float pos_y, int key_id, float nms_rad)
{
    for (int i = 0; i < pose_result->num; i ++)
    {
        pose_t *pose = &pose_result->pose[i];
        float prev_pos_x = pose->key[key_id].x * s_img_w;
        float prev_pos_y = pose->key[key_id].y * s_img_h;

        float dx = pos_x - prev_pos_x;
        float dy = pos_y - prev_pos_y;
        float len = (dx * dx) + (dy * dy);

        if (len <= (nms_rad * nms_rad))
            return true;
    }
    return false;
}

/*
 *  0      28.5    57.1    85.6   114.2   142.7   171.3   199.9   228.4   257   [pos_x]
 *  |---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|---+---|
 *     0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0      [hmp_pos_x]
 */
static void
get_pos_to_near_index (float pos_x, float pos_y, int *idx_x, int *idx_y)
{
    float ratio_x = pos_x / (float)s_img_w;
    float ratio_y = pos_y / (float)s_img_h;

    float hmp_pos_x = ratio_x * (s_hmp_w - 1);
    float hmp_pos_y = ratio_y * (s_hmp_h - 1);

    int hmp_idx_x = roundf (hmp_pos_x);
    int hmp_idx_y = roundf (hmp_pos_y);

    hmp_idx_x = std::min (hmp_idx_x, s_hmp_w -1);
    hmp_idx_y = std::min (hmp_idx_y, s_hmp_h -1);
    hmp_idx_x = std::max (hmp_idx_x, 0);
    hmp_idx_y = std::max (hmp_idx_y, 0);

    *idx_x = hmp_idx_x;
    *idx_y = hmp_idx_y;
}

static keypoint_t
traverse_to_tgt_key(int edge, keypoint_t src_key, int tgt_key_id, void *disp)
{
    float src_pos_x = src_key.pos_x;
    float src_pos_y = src_key.pos_y;

    int src_idx_x, src_idx_y;
    get_pos_to_near_index (src_pos_x, src_pos_y, &src_idx_x, &src_idx_y);

    /* get displacement vector from source to target */
    float disp_x, disp_y;
    get_displacement_vector (disp, &disp_x, &disp_y, src_idx_y, src_idx_x, edge);

    /* calculate target position */
    float tgt_pos_x = src_pos_x + disp_x;
    float tgt_pos_y = src_pos_y + disp_y;

    int tgt_idx_x, tgt_idx_y;
    int offset_refine_step = 2;
    for (int i = 0; i < offset_refine_step; i ++)
    {
        get_pos_to_near_index (tgt_pos_x, tgt_pos_y, &tgt_idx_x, &tgt_idx_y);
        get_index_to_pos (tgt_idx_x, tgt_idx_y, tgt_key_id, &tgt_pos_x, &tgt_pos_y);
    }

    keypoint_t tgt_key = {0};
    tgt_key.pos_x = tgt_pos_x;
    tgt_key.pos_y = tgt_pos_y;
    tgt_key.score = get_heatmap_score (tgt_idx_y, tgt_idx_x, tgt_key_id);
    tgt_key.valid = 1;

    return tgt_key;
}

static void
decode_pose (part_score_t &root, keypoint_t *keys)
{
    /* calculate root key position. */
    int idx_x = root.idx_x;
    int idx_y = root.idx_y;
    int keyid = root.key_id;
    float *fw_disp_ptr = (float *)s_tensor_fw_disp.ptr;
    float *bw_disp_ptr = (float *)s_tensor_bw_disp.ptr;

    float pos_x, pos_y;
    get_index_to_pos (idx_x, idx_y, keyid, &pos_x, &pos_y);

    keys[keyid].pos_x = pos_x;
    keys[keyid].pos_y = pos_y;
    keys[keyid].score = root.score;
    keys[keyid].valid = 1;

    for (int edge = s_edge_num - 1; edge >= 0; edge --)
    {
        int src_key_id = pose_edges[edge][1];
        int tgt_key_id = pose_edges[edge][0];

        if ( keys[src_key_id].valid &&
            !keys[tgt_key_id].valid)
        {
            keys[tgt_key_id] = traverse_to_tgt_key(edge, keys[src_key_id], tgt_key_id, bw_disp_ptr);
        }
    }

    for (int edge = 0; edge < s_edge_num; edge ++)
    {
        int src_key_id = pose_edges[edge][0];
        int tgt_key_id = pose_edges[edge][1];

        if ( keys[src_key_id].valid &&
            !keys[tgt_key_id].valid)
        {
            keys[tgt_key_id] = traverse_to_tgt_key(edge, keys[src_key_id], tgt_key_id, fw_disp_ptr);
        }
    }
}

static float
get_instance_score (posenet_result_t *pose_result, keypoint_t *keys, float nms_rad)
{
    float score_total = 0.0f;
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float pos_x = keys[i].pos_x;
        float pos_y = keys[i].pos_y;
        if (within_nms_of_corresponding_point (pose_result, pos_x, pos_y, i, nms_rad))
            continue;

        score_total += keys[i].score;
    }
    return score_total / (float)kPoseKeyNum;
}

static int
regist_detected_pose (posenet_result_t *pose_result, keypoint_t *keys, float score)
{
    int pose_id = pose_result->num;
    if (pose_id >= MAX_POSE_NUM)
    {
        fprintf (stderr, "ERR: %s(%d): pose_num overflow.\n", __FILE__, __LINE__);
        return -1;
    }

    for (int i = 0; i < kPoseKeyNum; i++)
    {
        pose_result->pose[pose_id].key[i].x     = keys[i].pos_x / (float)s_img_w;
        pose_result->pose[pose_id].key[i].y     = keys[i].pos_y / (float)s_img_h;
        pose_result->pose[pose_id].key[i].score = keys[i].score;
    }

    pose_result->pose[pose_id].pose_score = score;
    pose_result->num ++;

    return 0;
}

static void
decode_multiple_poses (posenet_result_t *pose_result)
{
    std::list<part_score_t> queue;
//cout << "decoding" << endl;
    float score_thresh  = 0.1f;
    int   local_max_rad = 1;
    build_score_queue (queue, score_thresh, local_max_rad);

    memset (pose_result, 0, sizeof (posenet_result_t));
    while (pose_result->num < MAX_POSE_NUM && !queue.empty())
    {
        part_score_t &root = queue.front();

        float pos_x, pos_y;
        get_index_to_pos (root.idx_x, root.idx_y, root.key_id, &pos_x, &pos_y);

        float nms_rad = 20.0f;
        if (within_nms_of_corresponding_point (pose_result, pos_x, pos_y, root.key_id, nms_rad))
        {
            queue.pop_front();
            continue;
        }

        keypoint_t key_points[kPoseKeyNum] = {0};
        decode_pose (root, key_points);

        float score = get_instance_score (pose_result, key_points, nms_rad);
        regist_detected_pose (pose_result, key_points, score);

        queue.pop_front();
    }
}

static void
decode_single_pose (posenet_result_t *pose_result)
{
    int   max_block_idx[kPoseKeyNum][2] = {0};
    float max_block_cnf[kPoseKeyNum]    = {0};

    /* find the highest heatmap block for each key */
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        float max_confidence = -FLT_MAX;
        for (int y = 0; y < s_hmp_h; y ++)
        {
            for (int x = 0; x < s_hmp_w; x ++)
            {
                float confidence = get_heatmap_score (y, x, i);
                if (confidence > max_confidence)
                {
                    max_confidence = confidence;
                    max_block_cnf[i] = confidence;
                    max_block_idx[i][0] = x;
                    max_block_idx[i][1] = y;
                }
            }
        }
    }

#if 0
    for (int i = 0; i < kPoseKeyNum; i ++)
    {
        fprintf (stderr, "---------[%d] --------\n", i);
        for (int y = 0; y < s_hmp_h; y ++)
        {
            fprintf (stderr, "[%d] ", y);
            for (int x = 0; x < s_hmp_w; x ++)
            {
                float confidence = get_heatmap_score (y, x, i);
                fprintf (stderr, "%6.3f ", confidence);

                if (x == max_block_idx[i][0] && y == max_block_idx[i][1])
                    fprintf (stderr, "#");
                else
                    fprintf (stderr, " ");
            }
            fprintf (stderr, "\n");
        }
    }
#endif

    /* find the offset vector and calculate the keypoint coordinates. */
    for (int i = 0; i < kPoseKeyNum;i ++ )
    {
        int idx_x = max_block_idx[i][0];
        int idx_y = max_block_idx[i][1];
        float key_posex, key_posey;
        get_index_to_pos(idx_x, idx_y, i, &key_posex, &key_posey);

        pose_result->pose[0].key[i].x     = key_posex / (float)s_img_w;
        pose_result->pose[0].key[i].y     = key_posey / (float)s_img_h;
        pose_result->pose[0].key[i].score = max_block_cnf[i];
    }
    pose_result->num = 1;
    pose_result->pose[0].pose_score = 1.0f;
}


int
invoke_posenet (posenet_result_t *pose_result)
{
    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    /*
     * decode algorithm is from:
     *   https://github.com/tensorflow/tfjs-models/tree/master/posenet/src/multi_pose
     */
    if (1)
        decode_multiple_poses (pose_result);
    else
        decode_single_pose (pose_result);

    pose_result->pose[0].heatmap = s_tensor_heatmap.ptr;
    pose_result->pose[0].heatmap_dims[0] = s_hmp_w;
    pose_result->pose[0].heatmap_dims[1] = s_hmp_h;

    return 0;
}


static void
render_facemask (int x, int y, int w, int h, posenet_result_t *pose_ret)
{
    static int s_mask_texid = 0;
    static int s_mask_w;
    static int s_mask_h;

//    if (s_mask_texid == 0)
//    {
//        load_png_texture ("facemask/facemask.png", &s_mask_texid, &s_mask_w, &s_mask_h);
//    }

    for (int i = 0; i < pose_ret->num; i ++)
    {
        float rx = pose_ret->pose[i].key[kRightEar].x * w + x;
        float ry = pose_ret->pose[i].key[kRightEar].y * h + y;
        float lx = pose_ret->pose[i].key[kLeftEar ].x * w + x;
        float ly = pose_ret->pose[i].key[kLeftEar ].y * h + y;
        float cx = (rx + lx) * 0.5f;
        float cy = (ry + ly) * 0.5f;
        float scale = (rx - lx) / (float)s_mask_w * 2.5;
        float mask_w = s_mask_w * scale;
        float mask_h = s_mask_h * scale;
    //    draw_2d_texture (s_mask_texid,
    //                     cx - mask_w * 0.5f, cy - mask_h * 0.5f,
    //                     mask_w, mask_h, 1);
    }
}

/* render a bone of skelton. */
void
render_bone (int ofstx, int ofsty, int drw_w, int drw_h, 
             posenet_result_t *pose_ret, int pid, 
             enum pose_key_id id0, enum pose_key_id id1,
             float *col, float scale_h, float scale_w)
{
    float x0 = pose_ret->pose[pid].key[id0].x *scale_w;// * drw_w + ofstx;
    float y0 = pose_ret->pose[pid].key[id0].y *scale_h;//* drw_h + ofsty;
    float x1 = pose_ret->pose[pid].key[id1].x *scale_w;//* drw_w + ofstx;
    float y1 = pose_ret->pose[pid].key[id1].y *scale_h;//* drw_h + ofsty;
    float s0 = pose_ret->pose[pid].key[id0].score;
    float s1 = pose_ret->pose[pid].key[id1].score;
    
    cv::Point p1,p2;
    p1.x = (int)x0; p1.y = (int)y0;
    p2.x = (int)x1; p2.y = (int)y1;

    /* if the confidence score is low, draw more transparently. */
    col[3] = (s0 + s1) * 0.5f;
    line(frame,p1,p2,Scalar( 255, 255, 0 ),10);
//    draw_2d_line (x0, y0, x1, y1, col, 5.0f);
    

    col[3] = 1.0f;
}

void
render_posenet_result (int x, int y, int w, int h, posenet_result_t *pose_ret, float scale_h, float scale_w)
{
    float col_red[]    = {1.0f, 0.0f, 0.0f, 1.0f};
    float col_orange[] = {1.0f, 0.6f, 0.0f, 1.0f};
    float col_cyan[]   = {0.0f, 1.0f, 1.0f, 1.0f};
    float col_lime[]   = {0.0f, 1.0f, 0.3f, 1.0f};
    float col_pink[]   = {1.0f, 0.0f, 1.0f, 1.0f};
    float col_blue[]   = {0.0f, 0.5f, 1.0f, 1.0f};
    
    for (int i = 0; i < pose_ret->num; i ++)
    {
        /* draw skelton */

        /* body */
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kRightShoulder, col_cyan, scale_h, scale_w);
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kLeftHip,       col_cyan, scale_h, scale_w);
        render_bone (x, y, w, h, pose_ret, i, kRightShoulder, kRightHip,      col_cyan, scale_h, scale_w);
        render_bone (x, y, w, h, pose_ret, i, kLeftHip,       kRightHip,      col_cyan, scale_h, scale_w);

        /* legs */
        render_bone (x, y, w, h, pose_ret, i, kLeftHip,       kLeftKnee,      col_pink, scale_h, scale_w);
        render_bone (x, y, w, h, pose_ret, i, kLeftKnee,      kLeftAnkle,     col_pink, scale_h, scale_w);
        render_bone (x, y, w, h, pose_ret, i, kRightHip,      kRightKnee,     col_blue, scale_h, scale_w);
        render_bone (x, y, w, h, pose_ret, i, kRightKnee,     kRightAnkle,    col_blue, scale_h, scale_w);
        
        /* arms */
        render_bone (x, y, w, h, pose_ret, i, kLeftShoulder,  kLeftElbow,     col_orange, scale_h, scale_w);
        render_bone (x, y, w, h, pose_ret, i, kLeftElbow,     kLeftWrist,     col_orange, scale_h, scale_w);
        render_bone (x, y, w, h, pose_ret, i, kRightShoulder, kRightElbow,    col_lime, scale_h, scale_w  );
        render_bone (x, y, w, h, pose_ret, i, kRightElbow,    kRightWrist,    col_lime, scale_h, scale_w  );

        /* draw key points */
        for (int j = 0; j < kPoseKeyNum; j ++)
        {
            float keyx = pose_ret->pose[i].key[j].x * scale_w;// * w + x;
            float keyy = pose_ret->pose[i].key[j].y * scale_h;//* h + y;
         //   cout << "Point " << pose_ret->pose[i].key[j].x << "\t" << pose_ret->pose[i].key[j].x << "\t" << scale_w << "\t" << scale_h << endl;
            int r = 9;
            Point p1; p1.x = (int)keyx; p1.y = (int)keyy;
            circle(frame,p1,10,Scalar( 255, 255, 0 ),FILLED);
          //  draw_2d_fillrect (keyx - (r/2), keyy - (r/2), r, r, col_red);
        }
    }

  //  render_facemask (x, y, w, h, pose_ret);
}

int
main(int argc, char *argv[])
{
    ros::init(argc,argv, "posenet");
    ros::NodeHandle n;
    ros::NodeHandle private_nh_("~");

    ros::Rate r(10);
    
    image_transport::ImageTransport it(n);
    image_transport::Subscriber sub = it.subscribe("/camera/color/image_raw", 2, imageCallback); 

    string input_path = "/home/ubuntu/Downloads/i7.jpg";
    char *input_name = NULL;
    int win_w = 960;
    int win_h = 540;
    int texid;
    int texw, texh, draw_x, draw_y, draw_w, draw_h;
   // texture_2d_t captex = {0};
   
    int use_quantized_tflite = 0;//0: for quantized models from edgetpu website
    draw_x = 0;
    draw_y = 0;

    init_tflite_posenet (use_quantized_tflite);
    /* invoke pose estimation using TensorflowLite */
    //Mat frame_copy;
    cout << s_img_w << "\t" << s_img_h << endl;
   chrono::steady_clock::time_point Tbegin, Tend;
   //wait till we receive images
    while (ros::ok() && is_first_image) {
        	ros::spinOnce();
        	r.sleep();
    } 
    
    while(ros::ok()){
       
      //  cap >> frame;
        frame = image.clone();
        cvtColor(frame, frame, COLOR_BGR2RGB);
        int image_height = frame.size().height;
        int image_width = frame.size().width;
    //    cout << image_width << "\t" << image_height << endl;
        
        resize(frame,frame_copy,Size(s_img_w,s_img_h));
    
        float scale_h = (float)image_height/s_img_h*s_img_h;
        float scale_w = (float)image_width/s_img_w*s_img_w;
        Tbegin = chrono::steady_clock::now();
         feed_posenet_image (win_w, win_h, frame_copy);
         posenet_result_t pose_ret = {0};
         invoke_posenet (&pose_ret);

         render_posenet_result (draw_x, draw_y, draw_w, draw_h, &pose_ret, scale_h, scale_w);
        Tend = chrono::steady_clock::now();
        //calculate frame rate
        float f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        cout << "Inference time " << f << endl;
         cout << "Number of people " << pose_ret.num << endl;
      //   resize(frame,frame,Size(640,480));
         cvtColor(frame, frame, COLOR_RGB2BGR);
         imshow("results",frame);
         waitKey(1);
         
         ros::spinOnce();
         r.sleep();
    
    }
    
    
    /*
   
    frame =  imread(input_path);
    cvtColor(frame, frame, COLOR_BGR2RGB);
    int image_height = frame.size().height;
    int image_width = frame.size().width;
    cout << image_width << "\t" << image_height << endl;
    imshow("input", frame);
    waitKey();
    int use_quantized_tflite = 1;
    draw_x = 0;
    draw_y = 0;

    init_tflite_posenet (use_quantized_tflite);
    // invoke pose estimation using TensorflowLite 
    //Mat frame_copy;
    cout << s_img_w << "\t" << s_img_h << endl;
    resize(frame,frame_copy,Size(s_img_w,s_img_h));
    
    float scale_h = (float)image_height/s_img_h*s_img_h;
    float scale_w = (float)image_width/s_img_w*s_img_w;
    
    imshow("input", frame_copy);
    waitKey();
    feed_posenet_image (win_w, win_h, frame_copy);
    posenet_result_t pose_ret = {0};
    invoke_posenet (&pose_ret);

    render_posenet_result (draw_x, draw_y, draw_w, draw_h, &pose_ret, scale_h, scale_w);
    
    cout << "Number of people " << pose_ret.num << endl;
    resize(frame,frame,Size(640,480));
    cvtColor(frame, frame, COLOR_RGB2BGR);
    imshow("results",frame);
    waitKey(0);
*/
    destroyAllWindows();
    cout << "Bye!" << endl;
    return 0;

}
