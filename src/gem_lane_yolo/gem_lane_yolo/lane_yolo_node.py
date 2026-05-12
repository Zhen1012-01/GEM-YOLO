#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO


class LaneYoloNode(Node):

    def __init__(self):

        super().__init__('lane_yolo_node')

        self.bridge = CvBridge()

        self.model = YOLO("yolov8n-seg.pt")

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            '/lane_detection/image',
            10
        )

        self.get_logger().info("Lane YOLO node started")


    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        results = self.model(frame, verbose=False)

        annotated = results[0].plot()

        out_msg = self.bridge.cv2_to_imgmsg(
            annotated,
            encoding='bgr8'
        )

        self.publisher.publish(out_msg)


def main(args=None):

    rclpy.init(args=args)

    node = LaneYoloNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
