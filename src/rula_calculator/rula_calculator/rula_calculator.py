import rclpy
from rclpy.node import Node
import numpy as np
import copy
import argparse
import math

from body_data.msg import BodyMsg, MultiCameraPoints
from std_msgs.msg import String, Int16

# --- TABLES & CONSTANTS ---
tableA_in = np.array([
    [1, 2, 2, 2, 2, 3, 3, 3], [2, 2, 2, 2, 3, 3, 3, 3], [2, 3, 3, 3, 3, 3, 4, 4],
    [2, 3, 3, 3, 3, 4, 4, 4], [3, 3, 3, 3, 3, 4, 4, 4], [3, 3, 4, 4, 4, 4, 5, 5],
    [3, 3, 4, 4, 4, 4, 5, 5], [3, 4, 4, 4, 4, 4, 5, 5], [4, 4, 4, 4, 4, 5, 5, 5],
    [4, 4, 4, 4, 4, 5, 5, 5], [4, 4, 4, 5, 5, 5, 6, 6], [4, 4, 4, 5, 5, 5, 6, 6],
    [5, 5, 5, 5, 5, 6, 6, 7], [5, 6, 6, 6, 6, 7, 7, 7], [6, 6, 6, 7, 7, 7, 7, 8],
    [7, 7, 7, 7, 7, 8, 8, 8], [8, 8, 8, 8, 8, 9, 9, 9], [9, 9, 9, 9, 9, 9, 9, 9]
])

tableB_in = np.array([
    [1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7], [2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7],
    [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7], [5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
    [7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]
])

body_part = {
    "pelvis": 0, "rhip": 1, "rknee": 2, "rankle": 3, "lhip": 4, "lknee": 5, "lankle": 6, 
    "back": 7, "neck": 8, "nose": 9, "head": 10, "lshoulder": 11, "lelbow": 12, 
    "lwrist": 13, "rshoulder": 14, "relbow": 15, "rwrist": 16
}

# --- 3D MATH HELPER FUNCTIONS ---
def halpe2h36m(x):
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    mapping = {0:19, 1:12, 2:14, 3:16, 4:11, 5:13, 6:15, 8:18, 9:0, 10:17, 11:5, 12:7, 13:9, 14:6, 15:8, 16:10}
    for k, v in mapping.items(): y[:,k,:] = x[:,v,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    return y

def points2angle(a, b, c):
    """Calculates true 3D spatial angle A-B-C"""
    AB, BC = a - b, c - b  # Vectors pointing away from vertex b
    norm_AB, norm_BC = np.linalg.norm(AB), np.linalg.norm(BC)
    if norm_AB == 0 or norm_BC == 0: return 0.0
    cosine_angle = np.dot(AB, BC) / (norm_AB * norm_BC)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def point_to_line_distance(A, B, P):
    """Calculates shortest 3D distance (in meters) from Point P to Line segment AB"""
    line_vec = B - A
    norm_line = np.linalg.norm(line_vec)
    if norm_line == 0: return 0.0
    return np.linalg.norm(np.cross(line_vec, A - P)) / norm_line

class Point_Transformer():
    def transform(self, point):
        # Only translate indices from 136 Halpe to 17 H36m.
        # No scaling needed, we are operating in real-world METERS.
        point = np.array(point).reshape(1, -1, 3)
        point = halpe2h36m(point)
        return point.astype(np.float32).reshape(-1, 3)


# --- REFACTORED NODE ---
class rula_calculator(Node):
    def __init__(self):
        super().__init__('rula_calculator')
        self.get_logger().info('3D RULA Spatial Calculator started.')
        
        self.kpts_keeper = Point_Transformer()
        self.args = self.rula_arg_parser()

        self.unified_sub = self.create_subscription(MultiCameraPoints, '/multi_camera_points', self.unified_callback, 10)
        
        self.publisher_score_right = self.create_publisher(Int16, 'right_rula_score', 10)
        self.publisher_score_left = self.create_publisher(Int16, 'left_rula_score', 10)
        self.publisher_full_body_data = self.create_publisher(BodyMsg, 'full_body_data', 10)

    def rula_arg_parser(self):
        parser = argparse.ArgumentParser(description='Rula calculator input')
        parser.add_argument('--arm_support', type=int, default=0, choices=[0, -1])
        parser.add_argument('--muscle_use', type=int, default=0, choices=[0, 1])
        parser.add_argument('--load_score', type=int, default=0, choices=[0, 1, 2, 3])
        parser.add_argument('--leg_support', type=int, default=2, choices=[1, 2])
        args, unknown = parser.parse_known_args()
        return args

    def unified_callback(self, msg):
        front_values, right_values, left_values = None, None, None

        if len(msg.front_points) > 0:
            front_values = self.process_front(msg.front_points)
            
        if len(msg.right_points) > 0:
            right_values = self.process_side(msg.right_points, is_right=True)
            
        if len(msg.left_points) > 0:
            left_values = self.process_side(msg.left_points, is_right=False)

        if front_values is not None:
            if left_values is not None and right_values is not None:
                self.rula_score('both', front=front_values, left=left_values, right=right_values)
            elif left_values is not None:
                self.rula_score('left', front=front_values, left=left_values)
            elif right_values is not None:
                self.rula_score('right', front=front_values, right=right_values)

    def process_front(self, points_array):
        points = self.kpts_keeper.transform(np.array(points_array))

        # Angle relative to spine
        shoulder_angle_r = points2angle(points[body_part["rshoulder"]], points[body_part["neck"]], points[body_part["back"]])
        r_raised = 1 if shoulder_angle_r < 110 else 0
        
        shoulder_angle_l = points2angle(points[body_part["lshoulder"]], points[body_part["neck"]], points[body_part["back"]])
        l_raised = 1 if shoulder_angle_l < 110 else 0

        r_up_abduction_angle = points2angle(points[body_part["relbow"]], points[body_part["rshoulder"]], points[body_part["rhip"]])
        r_up_abduction = 1 if r_up_abduction_angle > 40 else 0
        
        l_up_abduction_angle = points2angle(points[body_part["lelbow"]], points[body_part["lshoulder"]], points[body_part["lhip"]])
        l_up_abduction = 1 if l_up_abduction_angle > 40 else 0

        # Meters evaluation (real world metric). Spine -> wrist distance > 0.25 meters implies abduction.
        r_low_abduction_dis = point_to_line_distance(points[body_part["neck"]], points[body_part["pelvis"]], points[body_part["rwrist"]])
        r_low_abduction = 1 if r_low_abduction_dis > 0.25 else 0
        
        l_low_abduction_dis = point_to_line_distance(points[body_part["neck"]], points[body_part["pelvis"]], points[body_part["lwrist"]])
        l_low_abduction = 1 if l_low_abduction_dis > 0.25 else 0

        # Twist threshold > 0.05 meters (~5 cm deviation from centerline)
        neck_twist = point_to_line_distance(points[body_part["neck"]], points[body_part["head"]], points[body_part["nose"]])
        n_twist = 1 if neck_twist > 0.05 else 0

        neck_bending = points2angle(points[body_part["lshoulder"]], points[body_part["neck"]], points[body_part["head"]])
        n_bending = 0 if 50 < neck_bending < 73 else 1
 
        side_bending = points2angle(points[body_part["neck"]], points[body_part["pelvis"]], points[body_part["rhip"]])
        s_bending = 0 if 72 < side_bending < 82 else 1

        return np.array([r_raised, l_raised, r_up_abduction, l_up_abduction, r_low_abduction, l_low_abduction, n_twist, n_bending, s_bending])

    def process_side(self, points_array, is_right):
        points = self.kpts_keeper.transform(np.array(points_array))
        
        side_prefix = "r" if is_right else "l"
        
        # OpenCV RealSense Frame: +Y is pointing Downwards (gravity)
        # We create absolute vertical reference vectors to calculate realistic spatial flexion.
        vertical_offset = np.array([0, 0.5, 0])  # 0.5 meters perfectly downwards

        shoulder_base = points[body_part[f"{side_prefix}shoulder"]] + vertical_offset
        up_hand_angle = points2angle(points[body_part[f"{side_prefix}elbow"]], points[body_part[f"{side_prefix}shoulder"]], shoulder_base)
        
        low_hand_angle = points2angle(points[body_part[f"{side_prefix}shoulder"]], points[body_part[f"{side_prefix}elbow"]], points[body_part[f"{side_prefix}wrist"]])
        
        # points2angle returns the interior angle at the middle vertex.
        # For neck: angle(head, neck, back) ≈ 180° when upright because the
        # head-to-neck vector points UP and the back-to-neck vector points DOWN
        # — they are nearly anti-parallel.  RULA expects 0° for a neutral neck.
        # Converting to forward-flexion: flexion = 180° − raw_angle.
        # Identical geometry applies to trunk: angle(neck, pelvis, trunk_base)
        # ≈ 180° when upright, so same correction is applied.
        raw_head   = points2angle(points[body_part["head"]], points[body_part["neck"]], points[body_part["back"]])
        head_angle = max(180.0 - raw_head, 0.0)   # 0° = neutral, +° = forward flex

        trunk_base = points[body_part["pelvis"]] + vertical_offset
        raw_trunk  = points2angle(points[body_part["neck"]], points[body_part["pelvis"]], trunk_base)
        trunk      = max(180.0 - raw_trunk, 0.0)  # 0° = upright, +° = forward lean

        if math.isnan(up_hand_angle): up_hand_angle = 0
        if math.isnan(low_hand_angle): low_hand_angle = 0
        if math.isnan(head_angle): head_angle = 0
        if math.isnan(trunk): trunk = 0

        return np.array([up_hand_angle, low_hand_angle, head_angle, trunk])

    def rula_calculation(self, side_values, front_values):
        up_hand_angle, low_hand_angle, head_angle, trunk = side_values
        raised, up_abduction, low_abduction, n_twist, n_bending, s_bending = front_values

        if 20 < up_hand_angle < 45: up_score = 2
        elif 45 < up_hand_angle < 90: up_score = 3
        elif up_hand_angle >= 90: up_score = 4
        else: up_score = 1
        
        up_score += raised + up_abduction + self.args.arm_support

        lower_score = 1 if 60 < low_hand_angle < 100 else 2
        lower_score += low_abduction

        up_final = tableA_in[((up_score - 1) * 3 + lower_score) - 1, 0]
        up_final += self.args.muscle_use + self.args.load_score

        # head_angle / trunk are now forward-flexion degrees (0° = neutral).
        # Boundary at 0 is inclusive so a perfectly upright posture scores 1.
        if head_angle < 10:  neck_score = 1   # 0–9°   neutral / slight flex
        elif head_angle < 20: neck_score = 2  # 10–19°
        else:                 neck_score = 3  # ≥20°  (backward extension → also 3+)

        neck_score += n_twist + n_bending

        if trunk < 10:   trunk_score = 1   # 0–9°   upright
        elif trunk < 20: trunk_score = 2   # 10–19°
        elif trunk < 60: trunk_score = 3   # 20–59°
        else:            trunk_score = 4   # ≥60°
        
        trunk_score += s_bending

        trunk_neck_score = tableB_in[neck_score - 1, (trunk_score * 2 + self.args.leg_support) - 1]
        trunk_neck_score += self.args.muscle_use + self.args.load_score

        final_score = tableB_in[min(up_final, 6) - 1, min(trunk_neck_score, 11) - 1]
        
        return final_score, up_score, lower_score, up_final, neck_score, trunk_score, trunk_neck_score

    def rula_score(self, side, front, left=None, right=None):
        msg = BodyMsg()
        msg.right_shoulder, msg.left_shoulder = int(front[0]), int(front[1])
        msg.right_up_abduction, msg.left_up_abduction = int(front[2]), int(front[3])
        msg.right_low_abduction, msg.left_low_abduction = int(front[4]), int(front[5])
        msg.neck_twist, msg.neck_bending, msg.side_bending = int(front[6]), int(front[7]), int(front[8])

        msg.right, msg.left = False, False

        if side in ['left', 'both']:
            rula = self.rula_calculation(left, [front[1], front[3], front[5], front[6], front[7], front[8]])
            msg.left_arm_up, msg.left_low_angle, msg.neck_angle, msg.trunk_angle = left
            msg.left_rula_score, msg.up_arm_score_left, msg.lower_arm_score_left = int(rula[0]), int(rula[1]), int(rula[2])
            msg.neck_score, msg.trunk_score = int(rula[4]), int(rula[5])
            msg.left = True

        if side in ['right', 'both']:
            rula = self.rula_calculation(right, [front[0], front[2], front[4], front[6], front[7], front[8]])
            msg.right_arm_up, msg.right_low_angle = right[0], right[1]
            msg.right_rula_score, msg.up_arm_score_right, msg.lower_arm_score_right = int(rula[0]), int(rula[1]), int(rula[2])
            msg.right = True

            if side == 'both':
                msg.neck_angle = (msg.neck_angle + right[2]) / 2
                msg.trunk_angle = (msg.trunk_angle + right[3]) / 2
                msg.neck_score = int((msg.neck_score + rula[4]) / 2)
                msg.trunk_score = int((msg.trunk_score + rula[5]) / 2)
            else:
                msg.neck_angle, msg.trunk_angle = right[2], right[3]
                msg.neck_score, msg.trunk_score = int(rula[4]), int(rula[5])

        self.publisher_full_body_data.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = rula_calculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()