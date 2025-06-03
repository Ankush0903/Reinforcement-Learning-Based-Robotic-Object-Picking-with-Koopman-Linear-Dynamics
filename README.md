This project implements a reinforcement learning (RL) framework for robotic object picking, leveraging Koopman operator theory to model nonlinear dynamics as a linear system. The system enables a robotic arm to learn optimal picking strategies in complex environments using data-driven Koopman embeddings. The project integrates RL algorithms (e.g., PPO, DDPG) with a Koopman-based linear model to achieve efficient and robust control for tasks like pick-and-place.

## Key Features:
- RL-based policy learning for robotic object picking
- Koopman operator for linearizing nonlinear robotic dynamics
- Support for robotic arms (e.g.Franka Emika) in simulated environments (e.g., Gazebo, PyBullet)
- Real-time visualization and control in ROS

## Prerequisites
- Operating System: Ubuntu 18.04/20.04 (ROS Melodic/Noetic recommended)
- ROS Version: Melodic or Noetic
  ## Dependencies ##
  - ROS packages: `ros-<distro>-moveit`, `ros-<distro>-rviz`, `ros-<distro>-gazebo-ros`
  - Python: >= 3.6 (with `numpy`, `torch`, `gym`, `stable-baselines3`)
  - PyTorch: >= 1.8.0 (for RL and Koopman model training)
  - OpenCV: >= 3.2 (for object detection)
  - Gazebo or PyBullet (for simulation)
  ## Hardware ##
  - Robotic arm (e.g.Franka Emika) or simulation
  - GPU (recommended for RL training, e.g., NVIDIA RTX 4080 Super or better)
