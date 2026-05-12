#!/usr/bin/env python3

import os
import sys
import cv2
import tty
import select
import termios

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageCaptureNode(Node):

    def __init__(self):
        super().__init__('image_capture_node')

        self.bridge = CvBridge()

        self.latest_frame = None
        self.image_count = 0

        # 保存目录
        self.save_dir = os.path.expanduser('~/lane_dataset/images')
        os.makedirs(self.save_dir, exist_ok=True)

        # 订阅原始摄像头
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # 键盘检测
        self.timer = self.create_timer(0.02, self.keyboard_loop)

        print('')
        print('========== IMAGE CAPTURE ==========')
        print('SPACE : save image')
        print('q     : quit')
        print(f'Save dir: {self.save_dir}')
        print('===================================')
        print('')

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
                0.001
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

    def keyboard_loop(self):
        key = self.get_key()

        if key == ' ':
            self.save_image()

        elif key == 'q':
            print('Quit image capture.')
            rclpy.shutdown()

    def save_image(self):

        if self.latest_frame is None:
            print('[WARN] No image received yet.')
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
            print(f'[FAILED] Could not save image')


def main(args=None):

    rclpy.init(args=args)

    node = ImageCaptureNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

