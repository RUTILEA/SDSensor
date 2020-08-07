#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>
#include <std_msgs/Int32.h>

#include <iostream>
#include <chrono>
#include <pigpio.h>

using namespace std;

int status=1;
int status_prev = 1;

void dangerCallback(const std_msgs::Int32ConstPtr& msg)
{
        status_prev = status;
	status = msg->data;
}


int main(int argc, char** argv)
{
	ros::init(argc,argv, "led");
	ros::NodeHandle n;

	ros::Subscriber sub_detect = n.subscribe("/danger",1,&dangerCallback);

	ros::Rate r(5);

	if (gpioInitialise() >= 0)
    		cout << "GPIO initlialized" << endl;
    	else
  		ROS_ERROR("cannot start GPIO");
	gpioSetMode(18,PI_OUTPUT);
	gpioSetMode(25,PI_OUTPUT);
	
	while(ros::ok()){
	
	bool status_change = false;
	if (status==status_prev)
		status_change = true;
	
	if (status_change)
	{
		if (status==2){
			cout << "WARN: INFECTION SPREAD" << endl;
			gpioWrite(18, 0); gpioWrite(25, 1); 
		}
		else{
			cout << "SOCIAL DISTANCE MAINTAINED" << endl;
			gpioWrite(18, 1); gpioWrite(25, 0); 
		}
	}
	
	ros::spinOnce();
        r.sleep();
	
	}
	
	
	
}
