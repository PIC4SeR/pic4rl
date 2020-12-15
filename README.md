> Project: "pic4rl"

> Owner: "Enrico Sutera & Mauro Martini"  

> Date: "2020:06" 

---

# Pic4rl

## Description of the project
This repository aims to provide a framework for applying Deep Reinforcement Learning to robotics.
Currently we're working on providing some simulation environment for diverse robots, such as Turtlebots 3, Rosbot pro,
Nexus 3wd (named after S7B3) and 4wd (named after ????) platforms.

Next steps include drones and manipulators.

## Installation procedure
...
### ROS2 (Eloquent)
On ubuntu 18.04, install folliwing https://index.ros.org/doc/ros2/Installation/Eloquent/Linux-Install-Debians/

### Gazebo
Gazebo 9 i required.
Install following http://gazebosim.org/tutorials?cat=install&tut=install_ubuntu&ver=9.0

Gazebo 9 should also be configured, as paths have to be added (or temporarily sourced each time).
gazebo must be sourced:
```
source /usr/share/gazebo/setup.sh
```
pic4rl models (or other models) should be added to GAZEBO_MODEL_PATH:
```
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/ros_2_workspace/src/pic4rl/pic4rl_gazebo/models
```
### Other dependencies:
Python3:
OpenCV (3.2.0.8) 
tensorflow 2.3

Ros2 packages:
To be added
## User Guide



