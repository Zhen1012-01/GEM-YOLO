#!/usr/bin/env python3

import sys
import math
import tty
import select
import termios
import subprocess


class ModelTeleportKeyboard:

    def __init__(self):
        self.world = 'highbay_testbed'
        self.model_name = 'gem'

        self.x = 0.0
        self.y = 0.0
        self.z = 0.15
        self.yaw = 0.0

        self.step_xy = 0.25
        self.step_yaw = math.radians(8.0)

        print('')
        print('========== MODEL TELEPORT KEYBOARD ==========')
        print('w/s : forward/backward')
        print('a/d : left/right')
        print('q/e : rotate left/right')
        print('r/f : z up/down')
        print('+/- : change step')
        print('x   : reset pose')
        print('ESC : quit')
        print('=============================================')
        print('')

        self.apply_pose()

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            key = sys.stdin.read(1) if rlist else ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return key

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

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(
            f'x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, '
            f'yaw={math.degrees(self.yaw):.1f} deg'
        )

    def run(self):
        while True:
            key = self.get_key()

            if key == '':
                continue

            if key == '\x1b':
                print('Quit.')
                break

            elif key == 'w':
                self.x += self.step_xy * math.cos(self.yaw)
                self.y += self.step_xy * math.sin(self.yaw)

            elif key == 's':
                self.x -= self.step_xy * math.cos(self.yaw)
                self.y -= self.step_xy * math.sin(self.yaw)

            elif key == 'a':
                self.x += self.step_xy * math.cos(self.yaw + math.pi / 2)
                self.y += self.step_xy * math.sin(self.yaw + math.pi / 2)

            elif key == 'd':
                self.x += self.step_xy * math.cos(self.yaw - math.pi / 2)
                self.y += self.step_xy * math.sin(self.yaw - math.pi / 2)

            elif key == 'q':
                self.yaw += self.step_yaw

            elif key == 'e':
                self.yaw -= self.step_yaw

            elif key == 'r':
                self.z += 0.05

            elif key == 'f':
                self.z -= 0.05

            elif key == '+':
                self.step_xy *= 1.25
                self.step_yaw *= 1.25
                print(f'step_xy={self.step_xy:.2f}, step_yaw={math.degrees(self.step_yaw):.1f}')

            elif key == '-':
                self.step_xy *= 0.8
                self.step_yaw *= 0.8
                print(f'step_xy={self.step_xy:.2f}, step_yaw={math.degrees(self.step_yaw):.1f}')

            elif key == 'x':
                self.x = 0.0
                self.y = 0.0
                self.z = 0.15
                self.yaw = 0.0

            self.apply_pose()


def main():
    node = ModelTeleportKeyboard()
    node.run()


if __name__ == '__main__':
    main()
