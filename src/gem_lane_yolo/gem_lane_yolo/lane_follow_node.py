#!/usr/bin/env python3

import os
import sys
import cv2
import tty
import time
import math
import select
import termios
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class LaneFollowNode(Node):
    def __init__(self):
        super().__init__('lane_follow_node')

        self.bridge = CvBridge()

        # =========================
        # YOLO
        # =========================
        self.model_path = os.path.expanduser('~/lane_models/best.pt')
        self.yolo_model = None
        self.use_yolo = True
        self.yolo_conf = 0.35
        self.yolo_imgsz = 640
        self.yolo_every_n = 2
        self.frame_id = 0
        self.cached_yolo_mask = None
        self.cached_yolo_count = 0

        if YOLO is None:
            self.get_logger().warn('Ultralytics not installed. Running OpenCV-only mode.')
            self.use_yolo = False
        elif not os.path.exists(self.model_path):
            self.get_logger().warn(f'YOLO model not found: {self.model_path}. Running OpenCV-only mode.')
            self.use_yolo = False
        else:
            self.get_logger().info(f'Loading YOLO model: {self.model_path}')
            self.yolo_model = YOLO(self.model_path)
            self.get_logger().info('YOLO model loaded.')

        # =========================
        # State
        # =========================
        self.state = 'ACQUIRE'

        self.acquire_count = 0
        self.acquire_need_count = 6
        self.acquire_error_gate = 90.0
        self.acquire_max_pair_error = 280.0
        self.track_lost_count = 0
        self.max_track_lost_count = 12

        # =========================
        # PID
        # error = lane_center_x - image_center_x
        # steer = -(kp * error + kd * d_error)
        # =========================
        self.kp = 0.0058
        self.ki = 0.0
        self.kd = 0.0022

        self.error_sum = 0.0
        self.last_error = 0.0
        self.last_d_error = 0.0
        self.last_steer = 0.0

        self.max_steer = 0.62
        self.max_steer_delta = 0.060

        # =========================
        # Speed
        # =========================
        self.fast_speed = 3.00
        self.base_speed = 2.40
        self.min_speed = 0.70
        self.acquire_speed_limit = 1.20
        self.safe_single_speed_floor = 1.80

        # lost search speed
        self.lost_speed_1 = 0.65
        self.lost_speed_2 = 0.55
        self.lost_speed_3 = 0.45
        self.lost_speed_4 = 0.30

        # =========================
        # Lane geometry
        # =========================
        self.lane_width_px = 230.0
        self.min_lane_width_px = 120.0
        self.max_lane_width_px = 560.0

        self.lane_center_est = None
        self.had_lane_lock = False

        # Historical own-lane anchors.
        # Only reliable two-line pair updates these.
        self.last_left_x = None
        self.last_right_x = None
        self.last_left_slope = None
        self.last_right_slope = None

        # Gates for anti-adjacent-lane logic.
        self.reacquire_center_gate = 110.0
        self.pair_anchor_gate = 125.0
        self.single_anchor_gate = 120.0
        self.single_slope_gate = 1.50
        self.max_single_abs_slope = 4.2
        self.single_line_gate = 300.0
        self.single_ambiguous_margin = 8.0
        self.max_center_jump = 230.0

        # Locked center update. Single-line does NOT update lock.
        self.lock_update_max_step = 25.0

        # =========================
        # Curve debug / bias
        # =========================
        # Keep curve bias OFF for now.
        # Adjacent-lane hairpins are easier to keep stable without bias.
        self.curve_bias_enable = False
        self.curve_bias_px = 0.0
        self.last_curve_print_time = 0.0
        self.curve_print_interval = 0.35

        # =========================
        # Line contact guard
        # =========================
        self.contact_ratio = 0.28
        self.warning_ratio = 0.40
        self.contact_steer = 0.26
        self.warning_steer = 0.12
        self.contact_speed_limit = 1.20
        self.warning_speed_limit = 2.20

        # =========================
        # Lost search
        # =========================
        self.lost_count = 0
        self.search_dir = 1.0
        self.last_search_flip_time = time.time()

        # =========================
        # Save image
        # =========================
        self.latest_frame = None
        self.image_count = 0
        self.save_dir = os.path.expanduser('~/lane_dataset/images')
        os.makedirs(self.save_dir, exist_ok=True)

        # =========================
        # ROS
        # =========================
        self.cmd_pub = self.create_publisher(AckermannDrive, '/ackermann_cmd', 10)
        self.vis_pub = self.create_publisher(Image, '/lane_follow/vis', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.timer = self.create_timer(0.02, self.keyboard_loop)

        print('')
        print('========== Clean YOLO + OpenCV + PID Lane Follow ==========')
        print(f'Model path : {self.model_path}')
        print(f'YOLO mode  : {self.use_yolo}')
        print('State      : ACQUIRE -> TRACK -> ACQUIRE')
        print('ACQUIRE    : must see TWO lane lines')
        print('TRACK      : pair first, safe single fallback second')
        print('Anti-change: pair and single both checked by own-lane anchors')
        print('Curve bias : OFF')
        print('SPACE      : save image')
        print('q          : quit')
        print('Vis topic  : /lane_follow/vis')
        print('Cmd topic  : /ackermann_cmd')
        print('===========================================================')
        print('')

    # ============================================================
    # Keyboard / ROS
    # ============================================================
    def publish_cmd(self, speed, steer):
        msg = AckermannDrive()
        msg.speed = float(speed)
        msg.steering_angle = float(steer)
        self.cmd_pub.publish(msg)

    def get_key(self):
        if not sys.stdin.isatty():
            return ''

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
            key = sys.stdin.read(1) if rlist else ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return key

    def keyboard_loop(self):
        key = self.get_key()

        if key == ' ':
            self.save_image()
        elif key == 'q':
            print('Quit lane follow node.')
            self.publish_cmd(0.0, 0.0)
            rclpy.shutdown()

    def save_image(self):
        if self.latest_frame is None:
            print('No image received yet.')
            return

        filename = os.path.join(self.save_dir, f'lane_{self.image_count:06d}.jpg')
        ok = cv2.imwrite(filename, self.latest_frame)

        if ok:
            print(f'[SAVED] {filename}')
            self.image_count += 1
        else:
            print(f'[FAILED] {filename}')

    def draw_text(self, img, text, x, y, color=(0, 255, 0), scale=0.55, thick=2):
        cv2.putText(
            img,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick
        )

    # ============================================================
    # Mask
    # ============================================================
    def make_road_roi(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)

        poly = np.array([[
            (0, int(h * 0.40)),
            (w, int(h * 0.40)),
            (w, h),
            (0, h)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, poly, 255)
        return mask

    def get_yellow_mask(self, frame):
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        lower_yellow = np.array([14, 35, 65])
        upper_yellow = np.array([48, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_yellow, upper_yellow)

        _, _, b = cv2.split(lab)
        _, mask_lab = cv2.threshold(b, 145, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(mask_hsv, mask_lab)

        # Remove grass
        lower_green = np.array([35, 35, 35])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        mask[green_mask > 0] = 0

        roi = self.make_road_roi(h, w)
        mask = cv2.bitwise_and(mask, roi)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 5))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        return mask

    def get_yolo_mask(self, frame):
        h, w = frame.shape[:2]

        if not self.use_yolo or self.yolo_model is None:
            return np.zeros((h, w), dtype=np.uint8), 0

        run_now = (
            self.cached_yolo_mask is None or
            self.cached_yolo_mask.shape[:2] != (h, w) or
            self.frame_id % self.yolo_every_n == 0
        )

        if not run_now:
            return self.cached_yolo_mask.copy(), self.cached_yolo_count

        yolo_mask = np.zeros((h, w), dtype=np.uint8)
        count = 0

        try:
            result = self.yolo_model.predict(
                frame,
                imgsz=self.yolo_imgsz,
                conf=self.yolo_conf,
                verbose=False
            )[0]
        except Exception as e:
            self.get_logger().warn(f'YOLO inference failed: {e}')
            return yolo_mask, 0

        if result.masks is not None and result.masks.xy is not None:
            for poly in result.masks.xy:
                if poly is None or len(poly) < 3:
                    continue

                pts = np.array(poly, dtype=np.int32)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

                cv2.fillPoly(yolo_mask, [pts], 255)
                count += 1

        if count > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 7))
            yolo_mask = cv2.dilate(yolo_mask, kernel, iterations=1)

        roi = self.make_road_roi(h, w)
        yolo_mask = cv2.bitwise_and(yolo_mask, roi)

        self.cached_yolo_mask = yolo_mask.copy()
        self.cached_yolo_count = count

        return yolo_mask, count

    def fuse_masks(self, frame):
        yellow_mask = self.get_yellow_mask(frame)
        yolo_mask, yolo_count = self.get_yolo_mask(frame)

        yolo_area = cv2.countNonZero(yolo_mask)

        if yolo_area > 80:
            fused = cv2.bitwise_and(yellow_mask, yolo_mask)
            fused_area = cv2.countNonZero(fused)

            if fused_area > 35:
                final_mask = fused
                mode = 'yolo+cv'
            else:
                final_mask = yellow_mask
                mode = 'cv_fallback'
        else:
            final_mask = yellow_mask
            mode = 'cv_only'

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        info = {
            'mode': mode,
            'yolo_count': yolo_count,
            'yellow_area': cv2.countNonZero(yellow_mask),
            'yolo_area': yolo_area,
            'final_area': cv2.countNonZero(final_mask)
        }

        return final_mask, yellow_mask, yolo_mask, info

    # ============================================================
    # Component extraction
    # ============================================================
    def extract_components(self, mask, y1, y2):
        roi = mask[y1:y2, :]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)

        comps = []
        roi_h = max(y2 - y1, 1)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bx = stats[i, cv2.CC_STAT_LEFT]
            by_local = stats[i, cv2.CC_STAT_TOP]
            by = by_local + y1
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            if area < 25:
                continue

            if bw < 3 and bh < 3:
                continue

            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if area < 60 and aspect < 1.4:
                continue

            comp_mask = (labels == i)
            ys, xs = np.where(comp_mask)

            if len(xs) == 0:
                continue

            # Control x: use lower part of component, less biased on curves.
            lower_cut = int(roi_h * 0.55)
            lower_idx = ys >= lower_cut

            if np.count_nonzero(lower_idx) >= 8:
                use_xs = xs[lower_idx]
                use_ys = ys[lower_idx]
                weights = 1.0 + 2.0 * (use_ys.astype(np.float32) / max(roi_h - 1, 1))
                cx_control = float(np.average(use_xs.astype(np.float32), weights=weights))
                cy_control = float(np.average(use_ys.astype(np.float32), weights=weights)) + y1
            else:
                use_xs = xs
                use_ys = ys
                cx_control = float(centroids[i][0])
                cy_control = float(centroids[i][1]) + y1

            # Direction: x = slope * y + intercept.
            slope = 0.0
            if len(use_xs) >= 8 and np.std(use_ys) > 1.0:
                try:
                    slope = float(np.polyfit(use_ys.astype(np.float32), use_xs.astype(np.float32), 1)[0])
                except Exception:
                    slope = 0.0

            bottom_local = by_local + bh
            bottom_touch = bottom_local > int(roi_h * 0.70)

            comps.append({
                'x': float(cx_control),
                'y': float(cy_control),
                'area': float(area),
                'w': float(bw),
                'h': float(bh),
                'slope': float(slope),
                'bottom_touch': bool(bottom_touch),
                'bbox': (int(bx), int(by), int(bw), int(bh))
            })

        comps = sorted(comps, key=lambda c: c['area'], reverse=True)
        return comps

    # ============================================================
    # Pair selection
    # ============================================================
    def anchor_pair_ok(self, left, right):
        if not self.had_lane_lock:
            return True

        if self.last_left_x is None or self.last_right_x is None:
            return True

        left_err = abs(left['x'] - self.last_left_x)
        right_err = abs(right['x'] - self.last_right_x)

        if left_err > self.pair_anchor_gate:
            return False

        if right_err > self.pair_anchor_gate:
            return False

        # Slope check only if anchors exist.
        if self.last_left_slope is not None:
            if abs(left.get('slope', 0.0) - self.last_left_slope) > self.single_slope_gate * 1.6:
                return False

        if self.last_right_slope is not None:
            if abs(right.get('slope', 0.0) - self.last_right_slope) > self.single_slope_gate * 1.6:
                return False

        return True

    def find_best_pair_acquire(self, comps, image_center):
        if len(comps) < 2:
            return None

        if self.had_lane_lock and self.lane_center_est is not None:
            target_center = self.lane_center_est
            target_gate = self.reacquire_center_gate
        else:
            target_center = image_center
            target_gate = self.acquire_max_pair_error

        valid = []

        for i in range(len(comps)):
            for j in range(len(comps)):
                if i == j:
                    continue

                left = comps[i]
                right = comps[j]

                if left['x'] >= right['x']:
                    continue

                width = right['x'] - left['x']

                if not (self.min_lane_width_px <= width <= self.max_lane_width_px):
                    continue

                if not self.anchor_pair_ok(left, right):
                    continue

                center = (left['x'] + right['x']) / 2.0
                error_to_img = center - image_center
                error_to_target = center - target_center

                if abs(error_to_target) > target_gate:
                    continue

                if abs(error_to_img) > self.acquire_max_pair_error:
                    continue

                area_bonus = min((left['area'] + right['area']) / 400.0, 10.0)

                if self.had_lane_lock:
                    score = abs(error_to_target) + 0.35 * abs(error_to_img) - area_bonus
                else:
                    score = abs(error_to_img) - area_bonus

                valid.append((score, left, right, center, width, error_to_img))

        if not valid:
            return None

        valid = sorted(valid, key=lambda x: x[0])
        return valid[0]

    def find_best_pair_track(self, comps, pred_center):
        if len(comps) < 2:
            return None

        expected_left = pred_center - self.lane_width_px / 2.0
        expected_right = pred_center + self.lane_width_px / 2.0

        valid = []

        for i in range(len(comps)):
            for j in range(len(comps)):
                if i == j:
                    continue

                left = comps[i]
                right = comps[j]

                if left['x'] >= right['x']:
                    continue

                width = right['x'] - left['x']

                if not (self.min_lane_width_px <= width <= self.max_lane_width_px):
                    continue

                if not self.anchor_pair_ok(left, right):
                    continue

                center = (left['x'] + right['x']) / 2.0
                center_dist = abs(center - pred_center)

                if center_dist > self.max_center_jump:
                    continue

                edge_dist = abs(left['x'] - expected_left) + abs(right['x'] - expected_right)
                area_bonus = min((left['area'] + right['area']) / 500.0, 8.0)
                score = center_dist + 0.35 * edge_dist - area_bonus

                valid.append((score, left, right, center, width))

        if not valid:
            return None

        valid = sorted(valid, key=lambda x: x[0])
        return valid[0]

    def update_lane_anchors_from_pair(self, left, right):
        self.last_left_x = float(left['x'])
        self.last_right_x = float(right['x'])
        self.last_left_slope = float(left.get('slope', 0.0))
        self.last_right_slope = float(right.get('slope', 0.0))

    # ============================================================
    # Lane estimate
    # ============================================================
    def estimate_acquire(self, mask, vis, y1, y2, name, color):
        h, w = mask.shape[:2]
        image_center = w / 2.0

        comps = self.extract_components(mask, y1, y2)

        cv2.rectangle(vis, (0, y1), (w - 1, y2), color, 2)
        cv2.line(vis, (int(image_center), y1), (int(image_center), y2), (255, 0, 0), 2)

        if self.had_lane_lock and self.lane_center_est is not None:
            cv2.line(vis, (int(self.lane_center_est), y1), (int(self.lane_center_est), y2), (255, 180, 0), 2)
            self.draw_text(vis, f'{name} RE-ACQUIRE locked', 10, y1 + 20, color, 0.55, 2)
        else:
            self.draw_text(vis, f'{name} ACQUIRE', 10, y1 + 20, color, 0.55, 2)

        for c in comps[:10]:
            x, y, bw, bh = c['bbox']
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (120, 120, 255), 1)

        pair = self.find_best_pair_acquire(comps, image_center)

        if pair is None:
            self.draw_text(vis, f'{name}: no valid TWO lines', 30, y1 + 45, (0, 0, 255))
            return None, None, False, len(comps)

        _, left, right, center, width, error = pair

        self.update_lane_anchors_from_pair(left, right)

        cv2.line(vis, (int(left['x']), y1), (int(left['x']), y2), (0, 255, 255), 3)
        cv2.line(vis, (int(right['x']), y1), (int(right['x']), y2), (0, 255, 255), 3)
        cv2.line(vis, (int(center), y1), (int(center), y2), (0, 0, 255), 3)

        self.draw_text(vis, 'LEFT LINE', left['x'] + 5, y1 + 45, (0, 255, 255), 0.55, 2)
        self.draw_text(vis, 'RIGHT LINE', right['x'] + 5, y1 + 70, (0, 255, 255), 0.55, 2)
        self.draw_text(vis, 'LANE CENTER', center + 5, y1 + 95, (0, 0, 255), 0.55, 2)

        self.draw_text(vis, f'{name}: TWO width={width:.0f}', 30, y2 - 30, (0, 255, 255))
        self.draw_text(vis, f'{name}: err={error:.0f}', 30, y2 - 10, (0, 0, 255))

        return error, center, True, len(comps)

    def single_anchor_ok(self, comp, side):
        line_x = comp['x']
        line_slope = comp.get('slope', 0.0)

        if abs(line_slope) > self.max_single_abs_slope:
            return False, f'slope_abs={line_slope:.2f}'

        if side == 'left':
            if self.last_left_x is not None:
                dx = abs(line_x - self.last_left_x)
                if dx > self.single_anchor_gate:
                    return False, f'L dx={dx:.0f}'

            if self.last_left_slope is not None:
                ds = abs(line_slope - self.last_left_slope)
                if ds > self.single_slope_gate:
                    return False, f'L ds={ds:.2f}'

        if side == 'right':
            if self.last_right_x is not None:
                dx = abs(line_x - self.last_right_x)
                if dx > self.single_anchor_gate:
                    return False, f'R dx={dx:.0f}'

            if self.last_right_slope is not None:
                ds = abs(line_slope - self.last_right_slope)
                if ds > self.single_slope_gate:
                    return False, f'R ds={ds:.2f}'

        return True, 'ok'

    def estimate_track(self, mask, vis, y1, y2, name, color, allow_single=True):
        h, w = mask.shape[:2]
        image_center = w / 2.0

        pred_center = self.lane_center_est if self.lane_center_est is not None else image_center
        expected_left = pred_center - self.lane_width_px / 2.0
        expected_right = pred_center + self.lane_width_px / 2.0

        comps = self.extract_components(mask, y1, y2)

        cv2.rectangle(vis, (0, y1), (w - 1, y2), color, 2)
        cv2.line(vis, (int(image_center), y1), (int(image_center), y2), (255, 0, 0), 2)
        cv2.line(vis, (int(pred_center), y1), (int(pred_center), y2), (255, 180, 0), 2)
        cv2.line(vis, (int(expected_left), y1), (int(expected_left), y2), (180, 0, 255), 1)
        cv2.line(vis, (int(expected_right), y1), (int(expected_right), y2), (180, 0, 255), 1)

        self.draw_text(vis, f'{name} TRACK', 10, y1 + 20, color, 0.55, 2)

        for c in comps[:8]:
            x, y, bw, bh = c['bbox']
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (120, 120, 255), 1)

        # 1) Pair first
        pair = self.find_best_pair_track(comps, pred_center)

        if pair is not None:
            _, left, right, center, width = pair

            self.update_lane_anchors_from_pair(left, right)
            self.lane_width_px = 0.95 * self.lane_width_px + 0.05 * width

            error = center - image_center

            cv2.line(vis, (int(left['x']), y1), (int(left['x']), y2), (0, 255, 255), 3)
            cv2.line(vis, (int(right['x']), y1), (int(right['x']), y2), (0, 255, 255), 3)
            cv2.line(vis, (int(center), y1), (int(center), y2), (0, 0, 255), 3)

            self.draw_text(vis, 'LEFT LINE', left['x'] + 5, y1 + 45, (0, 255, 255), 0.55, 2)
            self.draw_text(vis, 'RIGHT LINE', right['x'] + 5, y1 + 70, (0, 255, 255), 0.55, 2)
            self.draw_text(vis, 'LANE CENTER', center + 5, y1 + 95, (0, 0, 255), 0.55, 2)

            self.draw_text(vis, f'{name}: PAIR width={width:.0f}', 30, y2 - 30, (0, 255, 255))
            self.draw_text(vis, f'{name}: err={error:.0f}', 30, y2 - 10, (0, 0, 255))

            return error, center, 'pair'

        # 2) Safe single fallback
        if not allow_single:
            self.draw_text(vis, f'{name}: single disabled', 30, y1 + 45, (0, 0, 255))
            return None, None, 'none'

        if len(comps) == 0:
            self.draw_text(vis, f'{name}: no line', 30, y1 + 45, (0, 0, 255))
            return None, None, 'none'

        best = None
        best_side = None
        best_dist = 99999.0
        best_dl = 99999.0
        best_dr = 99999.0

        for c in comps:
            x = c['x']
            dl = abs(x - expected_left)
            dr = abs(x - expected_right)

            if dl < best_dist:
                best = c
                best_side = 'left'
                best_dist = dl
                best_dl = dl
                best_dr = dr

            if dr < best_dist:
                best = c
                best_side = 'right'
                best_dist = dr
                best_dl = dl
                best_dr = dr

        if best is None:
            return None, None, 'none'

        if best_dist > self.single_line_gate:
            self.draw_text(vis, f'{name}: reject single dist', 30, y1 + 45, (0, 0, 255))
            return None, None, 'none'

        if abs(best_dl - best_dr) < self.single_ambiguous_margin:
            self.draw_text(vis, f'{name}: reject ambiguous single', 30, y1 + 45, (0, 0, 255))
            return None, None, 'none'

        ok, reason = self.single_anchor_ok(best, best_side)
        if not ok:
            self.draw_text(vis, f'{name}: reject single anchor {reason}', 30, y1 + 70, (0, 0, 255))
            return None, None, 'none'

        line_x = best['x']
        line_slope = best.get('slope', 0.0)

        # Side sanity check
        if best_side == 'left' and line_x > pred_center + 10:
            self.draw_text(vis, f'{name}: reject L side', 30, y1 + 95, (0, 0, 255))
            return None, None, 'none'

        if best_side == 'right' and line_x < pred_center - 10:
            self.draw_text(vis, f'{name}: reject R side', 30, y1 + 95, (0, 0, 255))
            return None, None, 'none'

        if best_side == 'left':
            center = line_x + self.lane_width_px / 2.0
            virtual_x = center + self.lane_width_px / 2.0
            cv2.line(vis, (int(virtual_x), y1), (int(virtual_x), y2), (0, 180, 255), 1)
        else:
            center = line_x - self.lane_width_px / 2.0
            virtual_x = center - self.lane_width_px / 2.0
            cv2.line(vis, (int(virtual_x), y1), (int(virtual_x), y2), (0, 180, 255), 1)

        if abs(center - pred_center) > self.max_center_jump:
            self.draw_text(vis, f'{name}: reject single jump', 30, y1 + 120, (0, 0, 255))
            return None, None, 'none'

        error = center - image_center

        cv2.line(vis, (int(line_x), y1), (int(line_x), y2), (0, 255, 255), 3)
        cv2.line(vis, (int(center), y1), (int(center), y2), (0, 0, 255), 3)

        self.draw_text(vis, f'{name}: SINGLE {best_side.upper()} slope={line_slope:.2f}', 30, y2 - 30, (0, 255, 255))
        self.draw_text(vis, f'{name}: err={error:.0f}', 30, y2 - 10, (0, 0, 255))

        return error, center, 'single'

    # ============================================================
    # Curve debug
    # ============================================================
    def estimate_circle_from_3pts(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        temp = x2 * x2 + y2 * y2
        bc = (x1 * x1 + y1 * y1 - temp) / 2.0
        cd = (temp - x3 * x3 - y3 * y3) / 2.0

        det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

        if abs(det) < 1e-5:
            return None

        cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
        cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
        r = math.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)

        return cx, cy, r

    def curve_debug(self, vis, h, w, near_center, mid_center):
        if near_center is None or mid_center is None:
            return

        image_center = w / 2.0

        p_bottom = (image_center, h * 0.96)
        p_near = (float(near_center), h * 0.83)
        p_mid = (float(mid_center), h * 0.64)

        circle = self.estimate_circle_from_3pts(p_bottom, p_near, p_mid)
        shift = float(mid_center - near_center)

        if shift > 12:
            side = "RIGHT"
        elif shift < -12:
            side = "LEFT"
        else:
            side = "STRAIGHT"

        cv2.circle(vis, (int(p_bottom[0]), int(p_bottom[1])), 5, (255, 0, 0), -1)
        cv2.circle(vis, (int(p_near[0]), int(p_near[1])), 5, (0, 255, 255), -1)
        cv2.circle(vis, (int(p_mid[0]), int(p_mid[1])), 5, (0, 180, 255), -1)

        if circle is None:
            txt = f'CURVE side:{side} shift:{shift:.1f} r:None'
        else:
            cx, cy, r = circle
            if -2 * w < cx < 3 * w and -2 * h < cy < 3 * h:
                cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 255), -1)
            txt = f'CURVE side:{side} shift:{shift:.1f} center:({cx:.1f},{cy:.1f}) r:{r:.1f}'

        self.draw_text(vis, txt, 25, 128, (255, 255, 0), 0.48, 2)

        now = time.time()
        if now - self.last_curve_print_time > self.curve_print_interval:
            print(txt)
            self.last_curve_print_time = now

    # ============================================================
    # Control
    # ============================================================
    def compute_pid_control(self, error):
        self.error_sum += error
        self.error_sum = float(np.clip(self.error_sum, -1500.0, 1500.0))

        d_raw = error - self.last_error
        d_error = 0.35 * d_raw + 0.65 * self.last_d_error

        raw_steer = -(self.kp * error + self.ki * self.error_sum + self.kd * d_error)
        raw_steer = float(np.clip(raw_steer, -self.max_steer, self.max_steer))

        delta = raw_steer - self.last_steer
        delta = float(np.clip(delta, -self.max_steer_delta, self.max_steer_delta))

        steer = self.last_steer + delta
        steer = 0.68 * self.last_steer + 0.32 * steer
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        self.last_error = error
        self.last_d_error = d_error
        self.last_steer = steer

        abs_steer = abs(steer)

        if abs_steer < 0.08:
            speed = self.fast_speed
        elif abs_steer < 0.22:
            speed = self.base_speed
        else:
            ratio = min(abs_steer / self.max_steer, 1.0)
            speed = self.base_speed * (1.0 - 0.55 * ratio)

        speed = float(max(self.min_speed, speed))
        return speed, steer, d_error

    def reset_pid_soft(self):
        self.error_sum = 0.0
        self.last_d_error = 0.0

    def directed_lost_search_control(self):
        self.lost_count += 1
        now = time.time()

        if abs(self.last_error) > 25:
            target_dir = -1.0 if self.last_error > 0 else 1.0
        elif abs(self.last_steer) > 0.05:
            target_dir = 1.0 if self.last_steer > 0 else -1.0
        else:
            if now - self.last_search_flip_time > 1.2:
                self.search_dir *= -1.0
                self.last_search_flip_time = now
            target_dir = self.search_dir

        if self.lost_count <= 3:
            strength = 0.12
            speed = self.lost_speed_1
        elif self.lost_count <= 10:
            strength = 0.20
            speed = self.lost_speed_2
        elif self.lost_count <= 18:
            strength = 0.26
            speed = self.lost_speed_3
        else:
            strength = 0.18
            speed = self.lost_speed_4

        target_steer = target_dir * strength

        if self.lost_count > 5:
            target_steer += 0.05 * math.sin(self.lost_count * 0.35)

        steer = 0.25 * self.last_steer + 0.75 * target_steer
        steer = float(np.clip(steer, -0.30, 0.30))

        self.last_steer = steer
        return speed, steer

    # ============================================================
    # Line contact guard
    # ============================================================
    def line_contact_guard(self, mask, vis):
        h, w = mask.shape[:2]
        image_center = w / 2.0

        y1 = int(h * 0.76)
        y2 = int(h * 0.98)

        x1 = int(w * 0.12)
        x2 = int(w * 0.88)

        roi = mask[y1:y2, x1:x2]
        ys, xs = np.where(roi > 0)

        contact_px = max(50.0, min(78.0, self.contact_ratio * self.lane_width_px))
        warning_px = max(80.0, min(120.0, self.warning_ratio * self.lane_width_px))

        cv2.rectangle(
            vis,
            (int(image_center - contact_px), y1),
            (int(image_center + contact_px), y2),
            (0, 0, 255),
            2
        )

        cv2.rectangle(
            vis,
            (int(image_center - warning_px), y1),
            (int(image_center + warning_px), y2),
            (0, 180, 255),
            1
        )

        if len(xs) == 0:
            return {
                'active': False,
                'contact': False,
                'target_steer': 0.0,
                'speed_limit': 999.0,
                'dist': 999.0,
                'side': 'none'
            }

        xs = xs.astype(np.float32) + x1
        dists = np.abs(xs - image_center)

        idx = int(np.argmin(dists))
        closest_x = float(xs[idx])
        min_dist = float(dists[idx])

        if min_dist > warning_px:
            return {
                'active': False,
                'contact': False,
                'target_steer': 0.0,
                'speed_limit': 999.0,
                'dist': min_dist,
                'side': 'clear'
            }

        line_on_left = closest_x < image_center
        side = 'left' if line_on_left else 'right'

        # If this correction direction is wrong, flip this line.
        steer_sign = -1.0 if line_on_left else 1.0

        contact = min_dist < contact_px

        if contact:
            target_steer = steer_sign * self.contact_steer
            speed_limit = self.contact_speed_limit
            color = (0, 0, 255)
            label = 'LINE CONTACT'
        else:
            target_steer = steer_sign * self.warning_steer
            speed_limit = self.warning_speed_limit
            color = (0, 180, 255)
            label = 'LINE WARNING'

        cv2.circle(vis, (int(closest_x), int((y1 + y2) / 2)), 7, color, -1)
        cv2.line(vis, (int(closest_x), y1), (int(closest_x), y2), color, 2)

        self.draw_text(
            vis,
            f'{label}: {side} dist:{min_dist:.0f} target:{target_steer:.2f}',
            25,
            98,
            color,
            0.58,
            2
        )

        return {
            'active': True,
            'contact': contact,
            'target_steer': target_steer,
            'speed_limit': speed_limit,
            'dist': min_dist,
            'side': side
        }

    def apply_line_guard(self, speed, steer, guard):
        if guard is None or not guard['active']:
            return speed, steer

        target = guard['target_steer']

        if guard['contact']:
            steer = 0.40 * steer + 0.60 * target
            speed = float(min(speed, guard['speed_limit']))
        else:
            steer = 0.75 * steer + 0.25 * target
            speed = float(min(speed, guard['speed_limit']))

        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        return speed, steer

    # ============================================================
    # Visualization
    # ============================================================
    def overlay_mask(self, vis, mask, color):
        overlay = vis.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(vis, 0.80, overlay, 0.20, 0.0)

    # ============================================================
    # Main callback
    # ============================================================
    def image_callback(self, msg):
        self.frame_id += 1

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_frame = frame.copy()

        h, w = frame.shape[:2]

        final_mask, yellow_mask, yolo_mask, info = self.fuse_masks(frame)

        vis = frame.copy()
        vis = self.overlay_mask(vis, final_mask, (0, 255, 0))

        guard = self.line_contact_guard(final_mask, vis)

        acquire_y1 = int(h * 0.52)
        acquire_y2 = int(h * 0.94)

        near_y1 = int(h * 0.72)
        near_y2 = int(h * 0.94)

        mid_y1 = int(h * 0.56)
        mid_y2 = int(h * 0.72)

        # =========================
        # ACQUIRE
        # =========================
        if self.state == 'ACQUIRE':
            error, center, two_lines, comp_count = self.estimate_acquire(
                final_mask,
                vis,
                acquire_y1,
                acquire_y2,
                'main',
                (255, 255, 255)
            )

            if not two_lines or error is None:
                speed, steer = self.directed_lost_search_control()
                speed, steer = self.apply_line_guard(speed, steer, guard)
                self.publish_cmd(speed, steer)

                self.acquire_count = 0

                if self.had_lane_lock:
                    reason = 'need TWO lines near locked lane'
                else:
                    reason = 'need TWO valid lines'

                self.draw_text(
                    vis,
                    f'ACQUIRE SEARCH | {reason} | comps:{comp_count} | steer:{steer:.3f} speed:{speed:.2f}',
                    25,
                    38,
                    (0, 0, 255),
                    0.62,
                    2
                )

            else:
                speed, steer, d_error = self.compute_pid_control(error)
                speed = min(speed, self.acquire_speed_limit)
                speed, steer = self.apply_line_guard(speed, steer, guard)
                self.publish_cmd(speed, steer)

                if abs(error) < self.acquire_error_gate:
                    self.acquire_count += 1
                else:
                    self.acquire_count = 0

                if self.acquire_count >= self.acquire_need_count:
                    self.state = 'TRACK'
                    self.lane_center_est = center
                    self.had_lane_lock = True
                    self.track_lost_count = 0
                    self.lost_count = 0
                    self.reset_pid_soft()
                    print('[LANE ACQUIRED] Two lines detected and car centered. Switch to TRACK.')

                self.draw_text(
                    vis,
                    f'ACQUIRE TWO | err:{error:.1f} count:{self.acquire_count}/{self.acquire_need_count} steer:{steer:.3f} speed:{speed:.2f}',
                    25,
                    38,
                    (0, 255, 255),
                    0.62,
                    2
                )

        # =========================
        # TRACK
        # =========================
        elif self.state == 'TRACK':
            allow_single = (
                self.had_lane_lock and
                self.lane_center_est is not None and
                self.track_lost_count <= 8
            )

            near_error, near_center, near_mode = self.estimate_track(
                final_mask,
                vis,
                near_y1,
                near_y2,
                'near',
                (255, 255, 255),
                allow_single=allow_single
            )

            mid_error, mid_center, mid_mode = self.estimate_track(
                final_mask,
                vis,
                mid_y1,
                mid_y2,
                'mid',
                (180, 180, 180),
                allow_single=allow_single
            )

            if near_error is None and mid_error is None:
                self.track_lost_count += 1

                if self.track_lost_count > self.max_track_lost_count:
                    self.state = 'ACQUIRE'
                    self.acquire_count = 0
                    self.reset_pid_soft()
                    print('[LOST TOO LONG] Switch back to ACQUIRE, keeping locked center.')

                speed, steer = self.directed_lost_search_control()
                speed, steer = self.apply_line_guard(speed, steer, guard)
                self.publish_cmd(speed, steer)

                self.draw_text(
                    vis,
                    f'TRACK LOST -> search locked lane {self.track_lost_count}/{self.max_track_lost_count} steer:{steer:.3f} speed:{speed:.2f}',
                    25,
                    38,
                    (0, 0, 255),
                    0.62,
                    2
                )

            else:
                self.track_lost_count = 0
                self.lost_count = 0

                # Combine near/mid. Pair is trusted more than single.
                if near_error is not None and mid_error is not None:
                    near_w = 0.82
                    mid_w = 0.18

                    if near_mode == 'single':
                        near_w *= 0.45

                    if mid_mode == 'single':
                        mid_w *= 0.45

                    total = near_w + mid_w
                    near_w /= total
                    mid_w /= total

                    error = near_w * near_error + mid_w * mid_error
                    center_now = near_w * near_center + mid_w * mid_center

                elif near_error is not None:
                    error = near_error
                    center_now = near_center

                    if near_mode == 'single':
                        error *= 0.55

                else:
                    # Mid-only single is risky in adjacent hairpins.
                    if mid_mode == 'single':
                        self.track_lost_count += 1
                        speed, steer = self.directed_lost_search_control()
                        speed, steer = self.apply_line_guard(speed, steer, guard)
                        self.publish_cmd(speed, steer)

                        self.draw_text(
                            vis,
                            f'TRACK HOLD: mid-only single rejected steer:{steer:.3f} speed:{speed:.2f}',
                            25,
                            38,
                            (0, 0, 255),
                            0.62,
                            2
                        )

                        out_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                        self.vis_pub.publish(out_msg)
                        return

                    error = 0.55 * mid_error
                    center_now = mid_center

                # Debug curve only. No curve bias.
                if near_center is not None and mid_center is not None:
                    self.curve_debug(vis, h, w, near_center, mid_center)

                # Locked center update only from pair.
                pair_centers = []

                if near_error is not None and near_mode == 'pair':
                    pair_centers.append((0.82, near_center))

                if mid_error is not None and mid_mode == 'pair':
                    pair_centers.append((0.18, mid_center))

                if len(pair_centers) > 0:
                    total_w = sum([p[0] for p in pair_centers])
                    pair_center_now = sum([p[0] * p[1] for p in pair_centers]) / total_w

                    if self.lane_center_est is None:
                        self.lane_center_est = pair_center_now
                    else:
                        delta = pair_center_now - self.lane_center_est
                        delta = float(np.clip(delta, -self.lock_update_max_step, self.lock_update_max_step))
                        self.lane_center_est = self.lane_center_est + 0.10 * delta

                speed, steer, d_error = self.compute_pid_control(error)

                single_used = (near_mode == 'single') or (mid_mode == 'single')
                pair_used = (near_mode == 'pair') or (mid_mode == 'pair')

                if single_used and not pair_used and allow_single:
                    speed = max(speed, self.safe_single_speed_floor)

                # Curve speed limit by center shift only, not bias.
                if near_center is not None and mid_center is not None:
                    curve_strength = abs(float(mid_center - near_center))
                    if curve_strength > 125:
                        speed = min(speed, 1.05)
                    elif curve_strength > 75:
                        speed = min(speed, 1.45)

                speed, steer = self.apply_line_guard(speed, steer, guard)
                self.publish_cmd(speed, steer)

                self.draw_text(
                    vis,
                    f'TRACK PID | err:{error:.1f} d:{d_error:.1f} steer:{steer:.3f} speed:{speed:.2f} single:{allow_single}',
                    25,
                    38,
                    (0, 255, 0),
                    0.62,
                    2
                )

        else:
            self.state = 'ACQUIRE'

        self.draw_text(
            vis,
            f"state:{self.state} mode:{info['mode']} yolo:{info['yolo_count']} area:{info['final_area']} lane_w:{self.lane_width_px:.0f}",
            25,
            68,
            (0, 255, 255),
            0.55,
            2
        )

        if self.lane_center_est is not None:
            cv2.line(vis, (int(self.lane_center_est), 0), (int(self.lane_center_est), h), (255, 180, 0), 1)
            label = 'locked center' if self.had_lane_lock else 'center est'
            self.draw_text(vis, label, int(self.lane_center_est) + 5, 95, (255, 180, 0), 0.5, 1)

        if self.last_left_x is not None:
            cv2.line(vis, (int(self.last_left_x), 0), (int(self.last_left_x), h), (0, 200, 255), 1)
            self.draw_text(vis, 'left anchor', int(self.last_left_x) + 5, 180, (0, 200, 255), 0.45, 1)

        if self.last_right_x is not None:
            cv2.line(vis, (int(self.last_right_x), 0), (int(self.last_right_x), h), (0, 200, 255), 1)
            self.draw_text(vis, 'right anchor', int(self.last_right_x) + 5, 205, (0, 200, 255), 0.45, 1)

        out_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        self.vis_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)

    node = LaneFollowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.publish_cmd(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
