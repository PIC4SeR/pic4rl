> Project: "pic4rl"

> Owner: "Enrico Sutera & Mauro Martini"  

> Date: "2020:06" 

---

# Title
Pic4rl
## Description of the project


![alt text](https://github.com/PIC4SeRCentre/pic4rl/blob/master/Screenshot%20from%202020-07-17%2012-09-47.png?raw=true)

## Installation procedure
...
ROS2 is required
On ubuntu 18.04, install folliwing https://index.ros.org/doc/ros2/Installation/Dashing/Linux-Install-Debians/


Gazebo 9 i required.
Install following http://gazebosim.org/tutorials?cat=install&tut=install_ubuntu&ver=9.0

in case of dependencies error related to gdal-abi-2-2-3,
  sudo apt remove libgda20
  sudo apt install libogdi3.2
  sudo apt install libgdal20=2.2.3+dfsg-2
Then proceed installing gazebo9

Gazebo 9 should also be configured, as paths have to be added (or temporarily sourced each time)
gazebo must be sourced
  source /usr/share/gazebo/setup.sh
pic4rl models (or other models) should be added to GAZEBO_MODEL_PATH
  export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/ros_2_workspace/src/pic4rl/pic4rl_gazebo/models


## User Guide
...

P.S. Compile requirements.txt file if needed

More detailed information about markdown style for README.md file [HERE](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
