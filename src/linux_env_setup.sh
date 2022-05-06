#!/usr/bin/bash
# sudo apt-get update && sudo apt-get install fish
cp /sim_ws/src/.tmux.conf /root/.tmux.conf

echo "alias underlay='source /opt/ros/foxy/setup.bash'" >> /root/.bashrc
echo "alias overlay='source /sim_ws/install/setup.bash'" >> /root/.bashrc
echo "alias sim='ros2 launch f1tenth_gym_ros gym_bridge_launch.py'" >> /root/.bashrc

