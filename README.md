# tf_ros_detection

## Ubuntu Server Installation
Download the Ubuntu 18 server image file for Raspberry Pi 4 for Rafrom [official website](https://ubuntu.com/download/raspberry-pi/thank-you?version=20.04&architecture=arm64+raspi)
For connecting to Wifi, edit the system-boot/network-config file from the sd card and edit it as follows while keeping the spaces as shown
``` 
wifis:
  wlan0:
  dhcp4: true
  optional: true
  access-points:
    "home network":
      password: "123456789"
```
Now boot the SD card and install any desktop for GUI:
```
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install xubuntu-desktop
$ sudo reboot
```

**(Optional)** For using neural networks and for development it might also be a good idea to increase swap memory if SD card memory is huge. First install the package zram-config
> $ sudo apt-get install zram-config
and the edit the following file:
> $ sudo nano /usr/bin/init-zram-swapping
to look like this

```
#!/bin/sh

# load dependency modules
NRDEVICES=$(grep -c ^processor /proc/cpuinfo | sed 's/^0$/1/')
if modinfo zram | grep -q ' zram_num_devices:' 2>/dev/null; then
  MODPROBE_ARGS="zram_num_devices=${NRDEVICES}"
elif modinfo zram | grep -q ' num_devices:' 2>/dev/null; then
  MODPROBE_ARGS="num_devices=${NRDEVICES}"
else
  exit 1
fi
modprobe zram $MODPROBE_ARGS

# Calculate memory to use for zram (1/2 of ram)
totalmem=`LC_ALL=C free | grep -e "^Mem:" | sed -e 's/^Mem: *//' -e 's/  *.*//'`
mem=$(((totalmem / 2 / ${NRDEVICES}) * 1024 * 3))

# initialize the devices
for i in $(seq ${NRDEVICES}); do
```
Here, you need to change `1024*i` where 'i' can be adjusted according to available memory and your desired memory.
Now reboot the system and check the swap memory using `htop` command in terminal

## ROS melodic installation
Follow the steps from official documentation. The steps below are repeated but some details like keyserver addresses may change in future and it is better to follow website.
```
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
$ sudo apt update
$ sudo apt install ros-melodic-desktop
$ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
$ sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
$ sudo apt install python-rosdep
$ sudo rosdep init
$ rosdep update
```

## Intel realsense driver
The apt-get installation for raspberry pi is not supported for one of the driver packages [realsense-dkms](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) and you will have to build it from source following instructions from their [website](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md). The steps are repeated below:

```
$ sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
$ sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
$ git clone https://github.com/IntelRealSense/librealsense.git
$ cd librealsense
```
Now run the realsense permissions script from librealsense root directory:
> $ ./scripts/setup_udev_rules.sh

Now run the patch file
> $ ./scripts/patch-realsense-ubuntu-lts.sh

Now we build and install the library

```
$ mkdir build && cd build
$ cmake ../ -DCMAKE_BUILD_TYPE=Release
$ sudo make uninstall && make clean && make -j3 && sudo make install
```

Now connect the camera and start the API from terminal by tying **$realsense-viewer**. Install the recommended driver version thats shows up automatically in the notifications after successfully detetcing the camera. Check whether the image and depth maps are being received properly or not.

## Essential dependent ROS packages installation
You need to install some pre-requisite packages for using this repository. The instructions from their respective official pages are repeated below. First make a new catkin workspace
```
$ cd ~
$ mkdir catkin_ws && cd catkin_ws
$ mkdir src
$ catkin_make
```
1. [dynamic_reconfigure](http://wiki.ros.org/hokuyo_node/Tutorials/UsingReconfigureGUIToChangeHokuyoLaserParameters) and [ddynamic_reconfigure](https://github.com/pal-robotics/ddynamic_reconfigure.git)
```
$ cd ~/catkin_ws
$ rosdep install dynamic_reconfigure
$ rosmake dynamic_reconfigure
$ cd src
$ git clone https://github.com/pal-robotics/ddynamic_reconfigure.git
$ (working with branch kinetic-devel, which is the default master to be cloned)
$ cd ..
$ catkin_make
```

2. [realsense-ros](https://github.com/IntelRealSense/realsense-ros)
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/IntelRealSense/realsense-ros.git
$ cd ..
$ catkin_make
```

3. [vision-opencv](https://github.com/ros-perception/vision_opencv)
This package(particularly cv_bridge) is required for using opencv in ROS
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/ros-perception/vision_opencv.git
$ git checkout melodic
$ cd ..
$ catkin_make
```

## Tensorflow Lite installation
We need to install version of tensorflow that was used to compile current master branch of edgetpu compiler for using the Coral USB accelerator. From current master branch(`commit 7064b94dd5b996189242320359dbab8b52c94a84` at time of writing this documentation) of [google-coral/edgetpu](https://github.com/google-coral/edgetpu), you can see in edgetpu/WORKSPACE file:
`TENSORFLOW_COMMIT = "d855adfc5a0195788bf5f92c3c7352e638aa1109"`

This is the version we need to checkout for. Now follow the following steps for building tensorflow lite:
```
$ git clone https://github.com/tensorflow/tensorflow
$ cd tensorflow
$ git checkout d855adfc5a0195788bf5f92c3c7352e638aa1109
$ ./tensorflow/lite/tools/make/download_dependencies.sh (This will download many packages in ./tensorflow/lite/tools/make/downloads)
$ ./tensorflow/lite/tools/make/build_generic_aarch64_lib.sh

```

While the above steps compile successfully for master branch, there will be errors in the ckecked out branch above. This is because the abseil package downloaded above is incomplete for some reason. Therefore, you can copy the abseil package downloaded from master branch to this branch in ./tensorflow/lite/tools/make/downloads.

There will still be errors when running build_generic_aarch64_lib.sh for compilation. The reason is that the Makefile file in this branch also seems incomplete for building the abseil package and some other tensorflow files. Therefore, following changes are reqiured in the Makefile. Find the following code excerpts and remove the lines marked with `(-)` and add the lines marked with `(+)`.

```
$(wildcard tensorflow/lite/kernels/internal/reference/*.cc) \
(+)$(wildcard tensorflow/lite/tools/optimize/sparsity/*.cc)\ 
$(PROFILER_SRCS) \
tensorflow/lite/tools/make/downloads/farmhash/src/farmhash.cc \
```

```

$(shell find tensorflow/lite/tools/make/downloads/absl/absl/ \
               (-) -type f -name \*.cc | grep -v test | grep -v benchmark
               (+) -type f -name \*.cc | grep -v test | grep -v benchmark **| grep -v synchronization | grep -v debugging | grep -v hash | grep -v flags | grep -v random**)
```
```
LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz \
(+)-ldl
```

```
(+)CFLAGS := -O3 -DNDEBUG -fPIC -pthread $(EXTRA_CFLAGS)
(+)CXXFLAGS := $(CFLAGS) --std=c++11 $(EXTRA_CXXFLAGS)
(-)CXXFLAGS := -O3 -DNDEBUG -fPIC -pthread
(-)CXXFLAGS += $(EXTRA_CXXFLAGS)
(-)CFLAGS := ${CXXFLAGS}
(-)CXXFLAGS += --std=c++11
```
Now compile again using `./tensorflow/lite/tools/make/build_generic_aarch64_lib.sh` and it should compile successfully and you should be able to see these files finally:
```
ubuntu@ubuntu:~/tensorflow/tensorflow/lite/tools/make/gen/generic-aarch64_armv8-a/lib$ ls
benchmark-lib.a  libtensorflow-lite.a
```

Now we need to compile flatbuffers too:
```
$ cd ~/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ sudo make install
$ sudo ldconfig
```

## edgetpu installation
While we can clone the complete edgetpu repository, it is not not necessary. Instead just download the edgetpu_runtime folder from () and run from inside
> ./install.sh

This should install the edgetpu library `/usr/lib/aarch64-linux-gnu/libedgetpu.so.1`

## Clone this repository
Finally everything is setup and you can clone this repository and compile it using `catkin_make`. The repository contains custom written nodes in the tf_ros_detection package. The library files in src folder are all taken from *src/cpp* folder of the master branch of [google-coral/edgetpu](https://github.com/google-coral/edgetpu) (`commit 7064b94dd5b996189242320359dbab8b52c94a84` at time of writing this documentation). However, although the structure of library files in *src* folder of *tf_ros_detection* is same, the code here is condensed version wherein unnecessay files are removed.

### Usage

The repository contains the following 3 nodes that have to be run:
1. **realsense2_camera** *in realsense-ros*:  for start using the realsense camera and publish the image and depth data
2. **coral_posenet** *in tf_ros_detection*: for running tflite posenet model and passing the result to decision making node
3. **pose_decision** *in tf_ros_detection*:: for running the decision making node which receives data from *coral_posenet*, decides if social disyance norms are being followed, and sends signal to led node(introduced below) to control LED color.

The nodes can be launched using a single launch file **camera_decision.launch** in **tf_ros_detection** package. The arguments for running the 3 nodes are as follows:

* **coral_posenet**
  * "cam_in" : camera raw color image topic name
  * "models_dir": location of posenet model
* **pose_decision**
  * cam_in: camera raw color image topic name
  * depth_in: camera depth topic name
  * threshold_Score: threshold minimum allowed distance between people
  * delay: for syncing asynchronous nodes of *coral_posenet* and *pose_decision*. If both nodes run fast enough, then delay=1, if *coral_posenet* works slow, then delay>1. Note: both nodes can be made synchronous by passing raw image and depth from *coral_posenet* node to *pose_decision* using custom messages.		

## LED control
There are some conveninent libraries for controlling WS2812B led strip with very simple python interface. The python libraries can be installed using
> sudo pip3 install rpi_ws281x adafruit-circuitpython-neopixel

The support is only for python3. However, ROS kinetic supports python2 by default. Therefore, you need to build another catkin workspace with python 3.
```
$ sudo apt-get install python-catkin-tools python3-dev python3-catkin-pkg-modules python3-numpy python3-yaml 
$ cd
$ mkdir catkin_led && cd catkin_led
$ mkdir src
$ catkin init
$ catkin config -DPYTHON_EXECUTIBLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/python3.6/config-3.6m-aarch64-linux-gnu/libpython3.6m.so
$ catkin config --install
$ catkin build
$ source install/setup.bash
```

Now clone the **led** package with this repository and place it inside the **src** folder and use `catkin build`. For python codes in ROS, you also need to give them execution permission using:
> sudo chmod a+x ~/catkin_led/src/led/led.py

In order to run the node, do (after sourcing workspace as above from *install/setup.bash*)
> rosrun led led.py

## Auto start Raspberry Pi

First we need to make a startup service and then link that service to a script file which will be run.
> $ sudo nano /etc/systemd/system/realsense_start.service

Then write the following:
```
[Unit]
After=mysql.service

[Service]
ExecStart=/home/ubuntu/realsense.sh

[Install]
WantedBy=default.target
```
This service will run the `realsense.sh` script file which will in turn run the posenet and detection codes and pass output to LED node for controlling the output of LED. This LED node will be started by another service:
> $ sudo nano /etc/systemd/system/led_start.service

```
[Unit]
After=mysql.service

[Service]
ExecStart=/home/ubuntu/catkin_led/st.sh

[Install]
WantedBy=default.target
```

These launch files that are run using the scripts above are already included in the package repository that were built before.

Now we need to give permissions to run these files
```
$ sudo chmod 744 /home/ubuntu/realsense.sh
$ sudo chmod 744 /home/ubuntu/catkin_led/st.sh
$ sudo chmod 664 /etc/systemd/system/realsense_start.service
$ sudo chmod 664 /etc/systemd/system/led_start.service
```
Some useful commands related to services are:
1. See all active services: `$ systemctl list-units --type=service`
2. Start a service: `$ sudo systemctl start led_start.service`
3. Stop a service: `$ sudo systemctl stop led_start.service`


The script files are as follow:
1. */home/ubuntu/realsense.sh*
```
#!/bin/bash
source /home/ubuntu/.bashrc
source /home/ubuntu/catkin_ws/devel/setup.bash
roslaunch realsense2_camera rs_camera.launch
```
2. */home/ubuntu/catkin_led/st.sh*
```
#!/bin/bash
sudo /home/ubuntu/catkin_led/led.sh
```
3. */home/ubuntu/catkin_led/led.sh*
```
#!/bin/bash
source /home/ubuntu/.bashrc
source /home/ubuntu/catkin_led/install/setup.bash --extend
rosrun led led.py
```

The LED code is run using two script files because the python code needs sudo permission to use GPIO pins and cannot be called with sudo from *led.sh* durectly because this script file has different type of permissions.

