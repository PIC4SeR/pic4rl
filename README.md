> Project: "pic4rl"

> Owner: "Enrico Sutera" 

> Date: "2020:06" 

---

# Title
Pic4rl
## Description of the project
This repository is meant to use reinforcement learning techniques along with Gazebo vith a variety of differnt mobile robots and drones.

The main idea so far is simply to have a open-gym-like structure with main functions being step, reset, render
The repository has many things that needs developing.

So far the actions is sent using the Twist message, which is suitable for mobile ground and aerial vehicles, however a more general message could be used when more dof are needed, e.g. when using manipulator. 

There are 3 files
pic4rl_gazebo: it is the main interface with gazebo, both for sensor reading and service call. 
pic4rl_environment: this file contains a class having all the gym-like functions (step, render, reset), moreover it contains the reward functions and the function related to the transformation of the state to observation (e.g. state processing, reduction, filtering)
pic4rl_trainig: At this state it simply implement a ddpg agent using t2rl package. It uses a class that inherits the class in pic4rl_environment.

only two nodes are hence generated
  Node 1 : one from pic4r_gazebo(node), which is always spinning, thus reading infomration from sensors
  Node 2 : one from pic4rl_training(pic4rl_environment(node))

Node 1 reads the state from gazebo and when requested send the state to Node 2. However the action is not sent by Node 1 but from Node 2, which does it simply pulishing to cmd_vel topic. Than Node 2 waits for 0.1 (it can be changed of course) while Node 1 collects sensors data.

## Installation procedure
...

## User Guide
...

P.S. Compile requirements.txt file if needed

More detailed information about markdown style for README.md file [HERE](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
