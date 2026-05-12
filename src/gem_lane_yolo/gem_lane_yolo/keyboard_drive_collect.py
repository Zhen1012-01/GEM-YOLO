#!/usr/bin/env python3

import os
import sys
import time
import termios
import tty
import select

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from cv_bridge import CvBridge

import cv2


class KeyboardDriveCollect(Node):

    def __init__(self):
        super().__init__('keyboard_drive_collect')

        self.bridge = CvBridge()
        self.latest_frame = None

        self.speed = 0.0
        self.steer = 0.0

        self.forward_speed = 3.0
        self.reverse_speed = -0.5
        self.steer_angle = 0.75

        self.last_key_time = time.time()
        self.key_timeout = 0.18

        self.image_count = 0
        self.save_dir = os.path.expanduser('~/lane_dataset/images')
        os.makedirs(self.save_dir, exist_ok=True)

        self.cmd_pub = self.create_publisher(
            AckermannDrive,
            '/ackermann_cmd',
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info('Keyboard drive + manual image collection started')
        self.print_help()

    def print_help(self):
        print('')
        print('========== Keyboard Control ==========')
        print('hold w: forward constant speed')
        print('hold s: reverse constant speed')
        print('hold a: steer left')
        print('hold d: steer right')
        print('x: stop')
        print('space: save one image manually')
        print('q: quit')
        print('Images saved to ~/lane_dataset/images')
        print('======================================')
        print('')

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

    def save_image(self):
        if self.latest_frame is None:
            self.get_logger().warn('No image received yet')
            return

        filename = os.path.join(
            self.save_dir,
            f'lane_{self.image_count:06d}.jpg'
        )

        cv2.imwrite(filename, self.latest_frame)
        self.image_count += 1

        self.get_logger().info(f'Saved image: {filename}')

    def publish_cmd(self):
        msg = AckermannDrive()
        msg.speed = float(self.speed)
        msg.steering_angle = float(self.steer)
        self.cmd_pub.publish(msg)

    def control_loop(self):
        if time.time() - self.last_key_time > self.key_timeout:
            self.speed = 0.0
            self.steer = 0.0

        self.publish_cmd()

    def update_key(self, key):
        self.last_key_time = time.time()

        if key == 'w':
            self.speed = self.forward_speed
            self.steer = 0.0

        elif key == 's':
            self.speed = self.reverse_speed
            self.steer = 0.0

        elif key == 'a':
            self.speed = self.forward_speed
            self.steer = self.steer_angle

        elif key == 'd':
            self.speed = self.forward_speed
            self.steer = -self.steer_angle

        elif key == 'x':
            self.speed = 0.0
            self.steer = 0.0

        elif key == ' ':
            self.save_image()

        print(f'speed: {self.speed:.2f}, steer: {self.steer:.2f}')


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.02)

        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return key


def main(args=None):
    rclpy.init(args=args)

    node = KeyboardDriveCollect()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)

            key = get_key()

            if key == 'q':
                break

            if key != '':
                node.update_key(key)

    except KeyboardInterrupt:
        pass

    node.speed = 0.0
    node.steer = 0.0
    node.publish_cmd()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
