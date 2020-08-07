#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

int main(int argc, char** argv)
{
	ros::init(argc,argv, "tf_camera");
	ros::NodeHandle n;

	ros::Rate r(100);

	tf::TransformBroadcaster broadcaster;

	while(n.ok()){
		broadcaster.sendTransform(
      		tf::StampedTransform(
        		tf::Transform(tf::Quaternion(0, 0, 0, 1), tf::Vector3(0.0, 0.0, 0.0)),
        		ros::Time::now(),"camera_realsense_link", "camera"));
    r.sleep();
	}


}