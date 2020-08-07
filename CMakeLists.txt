cmake_minimum_required(VERSION 3.5.1)
project(tf_ros_detection)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_HAVE_THREADS_LIBRARY 1)
#set(CMAKE_USE_WIN32_THREADS_INIT 0)
#set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rospy
  tensorflow_ros_cpp
  cv_bridge 
  image_transport
  std_msgs std_srvs sensor_msgs stereo_msgs
  dynamic_reconfigure
  #genmsg
  message_generation
  image_geometry
  tf
)	
#SET("OpenCV_DIR" "/home/ubuntu/opencv/opencv4/lib/cmake/opencv4")
#include_directories("/home/ubuntu/opencv/opencv4/lib/cmake/opencv4")
#include_directories("/home/ubuntu/opencv/opencv4/include/opencv4")

# Eigen lib headers
include_directories("/usr/local/include/eigen3" "/usr/lib/aarch64-linux-gnu" "/home/ubuntu/codes/abseil-cpp" "/home/ubuntu/codes/benchmark")# "/usr/src/gtest/build"
include_directories("/home/ubuntu/Downloads/edgetpu_runtime/libedgetpu/")#"/home/ubuntu/edgetpu/libedgetpu")

add_subdirectory(glog)
add_subdirectory(benchmark)
add_subdirectory(abseil-cpp)


add_subdirectory(3rdparty/ncnn ncnn_build)
include_directories(3rdparty/ncnn/src ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
#find_package(GTest REQUIRED)
#find_package(benchmark REQUIRED)
#ind_package(OpenCV REQUIRED)


#benchmark absl::strings absl::flags absl::memory glog::glog gtest gtest_main
#libgtest.a libgtest_main.a

generate_dynamic_reconfigure_options(
  cfg/Stereo.cfg
)

add_service_files(
    FILES
    StereoDepth.srv
)



add_message_files(
  FILES
  ImageDepthCall.msg
  TargetBB.msg
  Detect.msg
  keypoints.msg
  people.msg
  PosePass.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  stereo_msgs
)

catkin_package(
  INCLUDE_DIRS include
 # LIBRARIES ${PROJECT_NAME}
  DEPENDS #eigen3
  message_runtime
)


include_directories(include
  # needs to be first so that the tensorflow versions of includes are always treated first (not needed on all systems, but sometimes it is).
 # ${tensorflow_ros_cpp_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  "/home/ubuntu/tensorflow"
  "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include"
  "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/absl"
  "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build"
  "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/"
  "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build"
)


# Build the utils library
add_library(utilslib src/utils_old.cpp) #${PROJECT_NAME}_lib
add_dependencies(utilslib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(utilslib
  ${catkin_LIBRARIES}
  ${tensorflow_ros_cpp_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# Build the utils library
add_library(posenet_decoder src/posenet_decoder_op.cc src/posenet_decoder.cc src/basic_engine.cc src/basic_engine_native.cc  src/bbox_utils_test.cc src/edgetpu_resource_manager.cc src/error_reporter.cc  src/fake_op.cc  src/posenet_decoder_tflite_plugin.cc src/test_utils.cc src/utils.cc src/version.cc src/engine.cc) #${PROJECT_NAME}_lib
add_dependencies(posenet_decoder ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(posenet_decoder "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES} "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1" benchmark::benchmark absl::strings absl::flags absl::memory glog::glog dl gtest rt
)
#src/models_test.cc src/posenet_decoder_test.cc src/model_random_input_test_main.cc src/basic_engine_test src/error_reporter_test.cc src/version_test.cc

add_executable(tf_detect_realsense_node nodes/object_detect_realsense.cpp)
add_dependencies(tf_detect_realsense_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# again, tensorflow_ros_cpp_LIBRARIES need to go first in some setups
target_link_libraries(tf_detect_realsense_node ${tensorflow_ros_cpp_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} utilslib)

add_executable(tf_camera nodes/tf_camera.cpp)
target_link_libraries(tf_camera ${catkin_LIBRARIES})
add_dependencies(tf_camera ${catkin_EXPORTED_TARGETS})

add_executable(tflite nodes/tflite_detection.cpp)
target_link_libraries(tflite ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(tflite ${catkin_EXPORTED_TARGETS})

add_executable(posenet nodes/posenet.cpp)
target_link_libraries(posenet ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl posenet_decoder rt
)
add_dependencies(posenet ${catkin_EXPORTED_TARGETS})

add_executable(segment_c nodes/tflite_segmentation.cpp)
target_link_libraries(segment_c ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(segment_c ${catkin_EXPORTED_TARGETS})

add_executable(pose_net nodes/Pose_single.cpp)
target_link_libraries(pose_net ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(pose_net ${catkin_EXPORTED_TARGETS})

add_executable(pose_multi nodes/Pose_multi.cpp)
target_link_libraries(pose_multi ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl posenet_decoder rt
)
add_dependencies(pose_multi ${catkin_EXPORTED_TARGETS})

add_executable(unet nodes/Unet.cpp)
target_link_libraries(unet ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(unet ${catkin_EXPORTED_TARGETS})

add_executable(dnn nodes/object_detect_dnn.cpp)
target_link_libraries(dnn ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(dnn ${catkin_EXPORTED_TARGETS})

add_executable(object_detect_haar nodes/object_detect_haar.cpp)
target_link_libraries(object_detect_haar ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(object_detect_haar ${catkin_EXPORTED_TARGETS})

add_executable(detect_pass nodes/tflite_detect_pass.cpp src/UltraFace.cpp)
target_link_libraries(detect_pass ncnn ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(detect_pass ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(segment nodes/segment.cpp)
target_link_libraries(segment ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} pigpio dl rt
)
add_dependencies(segment ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(segment_test nodes/segment_test.cpp)
target_link_libraries(segment_test ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} pigpio dl rt
)
add_dependencies(segment_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


add_executable(main src/main.cpp src/UltraFace.cpp)
target_link_libraries(main ncnn ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_dependencies(main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(led nodes/led.cpp)
target_link_libraries(led ${catkin_LIBRARIES} pigpio dl
)
add_dependencies(led ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(coral_test nodes/coral_test.cc)
target_link_libraries(coral_test ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl posenet_decoder rt
)
add_dependencies(coral_test ${catkin_EXPORTED_TARGETS})

add_executable(pose_decision nodes/tflite_pose_decision.cpp)
target_link_libraries(pose_decision ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(pose_decision ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(pose_decision2 nodes/tflite_pose_decision2.cpp)
target_link_libraries(pose_decision2 ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(pose_decision2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(people_detect_pass nodes/detect_pass_pose.cpp)
target_link_libraries(people_detect_pass ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl rt
)
add_dependencies(people_detect_pass ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(people_pose_decision nodes/people_DetectPose_decision.cpp)
target_link_libraries(people_pose_decision ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl posenet_decoder rt
)
add_dependencies(people_pose_decision ${catkin_EXPORTED_TARGETS})

add_executable(coral_test_detect nodes/coral_test_detect.cc)
target_link_libraries(coral_test_detect ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl posenet_decoder rt
)
add_dependencies(coral_test_detect ${catkin_EXPORTED_TARGETS})

add_executable(coral_detect_pose_pass nodes/coral_detect_pose_pass.cpp)
target_link_libraries(coral_detect_pose_pass ${catkin_LIBRARIES} "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib/libtensorflow-lite.a" "/home/ubuntu/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a" ${OpenCV_LIBRARIES} dl posenet_decoder rt
)
add_dependencies(coral_detect_pose_pass ${catkin_EXPORTED_TARGETS})

