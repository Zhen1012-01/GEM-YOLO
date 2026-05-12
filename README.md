# GEM-YOLO

YOLO + OpenCV lane-following controller for the GEM ROS2 simulator.

## Contents

- src/gem_lane_yolo: ROS2 lane-following package
- models/lane_yolo/best.pt: trained YOLO segmentation model

## Install

Copy the package into the GEM simulator workspace:

    cp -r src/gem_lane_yolo ~/gem_sim/src/
    mkdir -p ~/lane_models
    cp models/lane_yolo/best.pt ~/lane_models/best.pt

Build:

    cd ~/gem_sim
    source /opt/ros/humble/setup.bash
    colcon build --packages-select gem_lane_yolo
    source install/setup.bash

Run:

    ros2 run gem_lane_yolo lane_follow_node

Visualization:

    ros2 run rqt_image_view rqt_image_view

Then select:

    /lane_follow/vis

## Notes

The lane follower uses YOLO segmentation, OpenCV yellow-line filtering, PID control, safe single-line fallback, adjacent-lane rejection, and visualization.
