import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import cv2
import numpy as np
import argparse
import os
import platform
import sys
import time
import queue
import collections
import torch
import threading
from tqdm import tqdm

from sensor_msgs.msg import Image
from std_msgs.msg import String  # Required for gesture publishing
from cv_bridge import CvBridge

# Import your unified message
from body_data.msg import MultiCameraPoints

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.webcam_detector import RealsenseDetectionLoader
from alphapose.utils.writer import DataWriter_rs

class process_3D(Node):
    def __init__(self):
        super().__init__('process_3D')
        self.get_logger().info('Point 3D Unified Node (Dual-Hand Gesture Tracking) started.')

        self.br = CvBridge()

        # Gesture vote buffer — require 3 / 5 frames to agree before publishing.
        self._gesture_buffer = collections.deque(maxlen=5)
        # Leading-edge latch — fire once per gesture show; reset when hand relaxes.
        self._prev_stable_gesture = "NONE"

        # Publishers
        self.unified_publisher = self.create_publisher(MultiCameraPoints, '/multi_camera_points', 10)
        self.gesture_publisher = self.create_publisher(String, '/operator_gesture', 10)
        self.image_publishers = {}

        args, cfg = self.alphapose_parser()
        
        for side in args.active_sides:
            topic_name = {0: 'front', 1: 'right', 2: 'left'}.get(side, 'general')
            self.image_publishers[side] = self.create_publisher(Image, f'{topic_name}_frame_2D', 1)
            self.get_logger().info(f"{topic_name.capitalize()} side camera stream initialized.")

        self.inference_thread = threading.Thread(target=self.alphapose_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()

    def alphapose_parser(self):
        parser = argparse.ArgumentParser(description='AlphaPose 3D Tracker')
        parser.add_argument('--cfg', type=str, default='/home/nimarme/AlphaPose/configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml')
        parser.add_argument('--checkpoint', type=str, default='/home/nimarme/AlphaPose/pretrained_models/multi_domain_fast50_regression_256x192.pth')
        parser.add_argument('--sp', default=False, action='store_true')
        parser.add_argument('--detector', dest='detector', default="yolox-x")
        parser.add_argument('--detbatch', type=int, default=5)
        parser.add_argument('--posebatch', type=int, default=64)
        parser.add_argument('--gpus', type=str, dest='gpus', default="0")
        parser.add_argument('--qsize', type=int, dest='qsize', default=1024)
        parser.add_argument('--flip', default=False, action='store_true')
        parser.add_argument('--device_name', type=str, default=None, nargs='+')
        parser.add_argument('--active_sides', type=int, nargs='+', default=[-1])
        parser.add_argument('--pose_track', dest='pose_track', action='store_true', default=False)
        parser.add_argument('--pose_flow', dest='pose_flow', action='store_true', default=False)
        parser.add_argument('--min_box_area', type=int, default=0)
        parser.add_argument('--format', type=str)
        
        # --- RESTORED ALPHAPOSE INTERNAL ARGS ---
        parser.add_argument('--eval', dest='eval', default=False, action='store_true')
        parser.add_argument('--save_video', dest='save_video', default=False, action='store_true')
        parser.add_argument('--vis_fast', dest='vis_fast', action='store_true', default=False)
        parser.add_argument('--save_img', default=False, action='store_true')
        parser.add_argument('--vis', default=False, action='store_true')
        parser.add_argument('--showbox', default=False, action='store_true')
        parser.add_argument('--profile', default=False, action='store_true')
        # ----------------------------------------

        args, unknown = parser.parse_known_args()
        cfg = update_config(args.cfg)

        if platform.system() == 'Windows':
            args.sp = True

        args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
        args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        args.detbatch = args.detbatch * len(args.gpus)
        args.posebatch = args.posebatch * len(args.gpus)
        args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'
        return args, cfg

    def detect_gesture(self, kpts_2d):
        """
        Evaluates both hands for Thumbs Up / Down.

        Camera coordinate frame: +Y points DOWN.

        THUMBS_UP:
          • Thumb tip clearly ABOVE wrist  (t_y < w_y − gap)
          • Index tip NOT extended upward  (i_y > w_y − gap)
            → index stays at/below wrist level when fingers are curled

        THUMBS_DOWN:
          • Thumb tip clearly BELOW wrist  (t_y > w_y + gap)
          • Index tip ABOVE thumb tip      (i_y < t_y − half_gap)
            → anchored to thumb, not wrist, so it works even when the
              whole fist is held low (curled index can be 40-60 px below
              the wrist but still well above the extended thumb tip).

        Confidence gating falls back to 1.0 if the model omits scores.
        """
        gap      = 35   # px — thumb-to-wrist minimum separation
        half_gap = 17   # px — index-to-thumb minimum separation (THUMBS_DOWN)
        conf_thr = 0.10 # AlphaPose hand keypoints commonly score 0.10–0.30

        def _check_hand(wrist_idx, thumb_idx, index_idx):
            try:
                w_y = kpts_2d[wrist_idx][1].item()
                t_y = kpts_2d[thumb_idx][1].item()
                i_y = kpts_2d[index_idx][1].item()

                # Confidence scores — fall back to 1.0 if the model omits them.
                try:
                    w_c = kpts_2d[wrist_idx][2].item()
                    t_c = kpts_2d[thumb_idx][2].item()
                except (IndexError, AttributeError):
                    w_c = t_c = 1.0

                if w_c < conf_thr or t_c < conf_thr or w_y == 0 or t_y == 0:
                    return "NONE"

                # THUMBS_UP
                if (t_y < w_y - gap) and (i_y > w_y - gap):
                    return "THUMBS_UP"

                # THUMBS_DOWN — index checked relative to thumb tip, not wrist.
                # This is robust even when the whole hand is held low because the
                # curled index is always above the downward-extended thumb.
                if (t_y > w_y + gap) and (i_y < t_y - half_gap):
                    return "THUMBS_DOWN"

            except Exception:
                pass
            return "NONE"

        # Check both hands independently, then resolve.
        r = _check_hand(115, 119, 123)   # right hand
        l = _check_hand(94,  98,  102)   # left  hand

        # Agreement or one is NONE → use the non-NONE result.
        # Conflict (one says UP, other says DOWN) → NONE (ambiguous).
        if r == l:               return r   # both agree (including both NONE)
        if r == "NONE":          return l
        if l == "NONE":          return r
        return "NONE"                        # conflict — discard

    def alphapose_loop(self):
        args, cfg = self.alphapose_parser()
        
        if not args.sp:
            torch.multiprocessing.set_start_method('forkserver', force=True)
            torch.multiprocessing.set_sharing_strategy('file_system')

        det_loader = {}
        for idx in range(len(args.active_sides)):
            side = args.active_sides[idx]
            device = args.device_name[idx]
            det_loader[side] = RealsenseDetectionLoader(get_detector(args), cfg, args, device_name=device)
            det_loader[side].start()

        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        
        if args.tracking:
            tracker = Tracker(tcfg, args)
            
        pose_model.to(args.device)
        pose_model.eval()

        writer = {}
        # Queues to hold depth map and intrinsics synchronized with frames
        depth_info_queues = {} 
        for side in args.active_sides:
            writer[side] = DataWriter_rs(cfg, args, save_video=False, queueSize=2)
            writer[side].start()
            depth_info_queues[side] = queue.Queue(maxsize=10)

        batchSize = args.posebatch if not args.flip else int(args.posebatch / 2)

        try:
            while rclpy.ok():
                with torch.no_grad():
                    front_kpts, right_kpts, left_kpts = [], [], []

                    for side in det_loader.keys():
                        # Unpack 9 items corresponding to RealSense depth loader
                        (inps, orig_img, depth_img, intrinsics, im_name, boxes, scores, ids, cropped_boxes) = det_loader[side].read()
                        
                        if orig_img is None: 
                            continue 
                            
                        if boxes is None or boxes.nelement() == 0:
                            writer[side].save(None, None, None, None, None, orig_img, im_name)
                        else:
                            # SINGLE PERSON FILTER: Keep only the largest bounding box
                            if boxes.size(0) > 1:
                                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                                max_idx = torch.argmax(areas).item()
                                boxes = boxes[max_idx:max_idx+1]
                                inps = inps[max_idx:max_idx+1]
                                scores = scores[max_idx:max_idx+1]
                                if ids is not None:
                                    ids = ids[max_idx:max_idx+1]
                                cropped_boxes = cropped_boxes[max_idx:max_idx+1]

                            # Send to inference
                            inps = inps.to(args.device)
                            hm = pose_model(inps)
                            hm = hm.cpu()
                            
                            # Push current depth profile to sync queue
                            depth_info_queues[side].put((depth_img, intrinsics))
                            writer[side].save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)

                        # Check if inference is ready
                        if writer[side].count_results() > 0:
                            try:
                                keypoints, vis_frame = writer[side].key_points.get_nowait()
                                synced_depth, synced_intrinsics = depth_info_queues[side].get_nowait()
                                
                                # Safely handle tensor shapes
                                kpts_raw = keypoints['keypoints']
                                kpts_2d = kpts_raw[0] if len(kpts_raw.shape) == 3 else kpts_raw
                                
                                kpts_3d = []

                                # --- 3D DEPROJECTION PROCESS ---
                                for kp in kpts_2d:
                                    # Use .item() to safely convert PyTorch scalar to Python int
                                    px, py = int(kp[0].item()), int(kp[1].item())
                                    
                                    # Boundary checks
                                    px = min(max(px, 0), synced_depth.shape[1]-1)
                                    py = min(max(py, 0), synced_depth.shape[0]-1)
                                    
                                    # Extract depth. Use a 5x5 median filter to patch 0-depth holes
                                    window = synced_depth[max(0, py-2):min(synced_depth.shape[0], py+3), max(0, px-2):min(synced_depth.shape[1], px+3)]
                                    valid_depths = window[window > 0]
                                    
                                    if len(valid_depths) > 0:
                                        Z = np.median(valid_depths) / 1000.0  # Convert mm to meters
                                        X = (px - synced_intrinsics['cx']) * Z / synced_intrinsics['fx']
                                        Y = (py - synced_intrinsics['cy']) * Z / synced_intrinsics['fy']
                                        kpts_3d.extend([X, Y, Z])
                                    else:
                                        kpts_3d.extend([0.0, 0.0, 0.0]) # Invalid depth fallback

                                if side == 0:
                                    front_kpts = kpts_3d

                                    # --- GESTURE RECOGNITION (Both Hands, Front Camera Only) ---
                                    gesture = self.detect_gesture(kpts_2d)
                                    self._gesture_buffer.append(gesture)

                                    # Compute stable vote (≥3 of last 5 agree on same gesture).
                                    buf = self._gesture_buffer
                                    if (len(buf) == buf.maxlen
                                            and gesture != "NONE"
                                            and buf.count(gesture) >= 3):
                                        stable = gesture
                                    else:
                                        stable = "NONE"

                                    # Leading-edge latch: fire ONCE per gesture show.
                                    # The same gesture cannot re-fire until the hand
                                    # relaxes (≥2 NONE frames in the buffer).
                                    if stable != "NONE":
                                        if stable != self._prev_stable_gesture:
                                            msg = String()
                                            msg.data = stable
                                            self.gesture_publisher.publish(msg)
                                            self._prev_stable_gesture = stable
                                    else:
                                        # Reset latch once hand relaxes enough.
                                        if buf.count("NONE") >= 2:
                                            self._prev_stable_gesture = "NONE"
                                        
                                elif side == 1: 
                                    right_kpts = kpts_3d
                                elif side == 2: 
                                    left_kpts = kpts_3d

                                # Publish visual frame back to GUI
                                small_vis = cv2.resize(vis_frame, (320, 240))
                                self.image_publishers[side].publish(self.br.cv2_to_imgmsg(small_vis, encoding='bgr8'))
                                
                            except queue.Empty:
                                pass
                                
                    # Publish 3D Points Unified MSG
                    unified_msg = MultiCameraPoints()
                    unified_msg.header.stamp = self.get_clock().now().to_msg()
                    unified_msg.front_points = front_kpts
                    unified_msg.right_points = right_kpts
                    unified_msg.left_points = left_kpts
                    
                    self.unified_publisher.publish(unified_msg)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
        finally:
            for side in det_loader.keys():
                writer[side].stop()
                det_loader[side].terminate()
                writer[side].terminate()

def main(args=None):
    rclpy.init(args=args)
    node = process_3D()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()