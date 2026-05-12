#!/usr/bin/env python3

import os
import sys
import cv2
import math
import tty
import select
import termios
import subprocess

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DatasetCollectKeyboard(Node):

    def __init__(self):
        super().__init__('dataset_collect_keyboard')

        self.bridge = CvBridge()

        self.latest_frame = None
        self.image_count = 0

        self.save_dir = os.path.expanduser('~/lane_dataset/images')
        os.makedirs(self.save_dir, exist_ok=True)

        # Gazebo world/model
        self.world = 'highbay_testbed'
        self.model_name = 'gem'

        # pose
        self.x = 0.0
        self.y = 0.0
        self.z = 0.15
        self.yaw = 0.0

        # movement
        self.step_xy = 0.25
        self.step_yaw = math.radians(8.0)

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.timer = self.create_timer(0.02, self.keyboard_loop)

        print('')
        print('========== DATASET COLLECT ==========')
        print('W/S : forward/backward')
        print('A/D : left/right')
        print('Q/E : rotate')
        print('R/F : z up/down')
        print('SPACE : save image')
        print('+/- : change step')
        print('X : reset pose')
        print('ESC : quit')
        print('=====================================')
        print('')

        self.apply_pose()

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)

            rlist, _, _ = select.select(
                [sys.stdin],
                [],
                [],
                0.02
            )

            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''

        finally:
            termios.tcsetattr(
                fd,
                termios.TCSADRAIN,
                old_settings
            )

        return key

    def save_image(self):

        if self.latest_frame is None:
            print('[WARN] no image')
            return

        filename = os.path.join(
            self.save_dir,
            f'lane_{self.image_count:06d}.jpg'
        )

        ok = cv2.imwrite(filename, self.latest_frame)

        if ok:
            print(f'[SAVED] {filename}')
            self.image_count += 1
        else:
            print('[FAILED] save image')

    def apply_pose(self):

        qz = math.sin(self.yaw / 2.0)
        qw = math.cos(self.yaw / 2.0)

        req = (
            f'name: "{self.model_name}" '
            f'position {{x: {self.x} y: {self.y} z: {self.z}}} '
            f'orientation {{x: 0 y: 0 z: {qz} w: {qw}}}'
        )

        cmd = [
            'ign', 'service',
            '-s', f'/world/{self.world}/set_pose',
            '--reqtype', 'ignition.msgs.Pose',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '1000',
            '--req', req
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print(
            f'pose -> x={self.x:.2f}, y={self.y:.2f}, '
            f'yaw={math.degrees(self.yaw):.1f}'
        )

    def keyboard_loop(self):

        key = self.get_key()

        if key == '':
            return

        if key == '\x1b':
            print('Quit.')
            rclpy.shutdown()
            return

        elif key == 'w':
            self.x += self.step_xy * math.cos(self.yaw)
            self.y += self.step_xy * math.sin(self.yaw)
            self.apply_pose()

        elif key == 's':
            self.x -= self.step_xy * math.cos(self.yaw)
            self.y -= self.step_xy * math.sin(self.yaw)
            self.apply_pose()

        elif key == 'a':
            self.x += self.step_xy * math.cos(self.yaw + math.pi / 2)
            self.y += self.step_xy * math.sin(self.yaw + math.pi / 2)
            self.apply_pose()

        elif key == 'd':
            self.x += self.step_xy * math.cos(self.yaw - math.pi / 2)
            self.y += self.step_xy * math.sin(self.yaw - math.pi / 2)
            self.apply_pose()

        elif key == 'q':
            self.yaw += self.step_yaw
            self.apply_pose()

        elif key == 'e':
            self.yaw -= self.step_yaw
            self.apply_pose()

        elif key == 'r':
            self.z += 0.05
            self.apply_pose()

        elif key == 'f':
            self.z -= 0.05
            self.apply_pose()

        elif key == '+':
            self.step_xy *= 1.25
            self.step_yaw *= 1.25

        elif key == '-':
            self.step_xy *= 0.8
            self.step_yaw *= 0.8

        elif key == 'x':
            self.x = 0.0
            self.y = 0.0
            self.z = 0.15
            self.yaw = 0.0
            self.apply_pose()

        elif key == ' ':
            self.save_image()


def main(args=None):

    rclpy.init(args=args)

    node = DatasetCollectKeyboard()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
