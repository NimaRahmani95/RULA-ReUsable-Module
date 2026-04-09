import rclpy
from rclpy.node import Node
import time
import rtde_control
import rtde_receive

from body_data.msg import BodyMsg 
from std_msgs.msg import String

class ProportionalRTDEController(Node):
    def __init__(self):
        super().__init__('pcb_ergonomic_assistant')
        
        # --- ROS 2 Parameters ---
        self.declare_parameter('robot_ip', '192.168.0.100')
        self.declare_parameter('movement_cooldown_sec', 4.0) 
        
        # KINEMATIC LIMITS: Absolute Z-height boundaries (in meters relative to robot base)
        # Prevents the robot from driving the PCB into the table or out of reach
        self.declare_parameter('z_min_limit', 0.20) 
        self.declare_parameter('z_max_limit', 0.65) 
        
        self.robot_ip = self.get_parameter('robot_ip').value
        self.cooldown = self.get_parameter('movement_cooldown_sec').value
        self.z_min_limit = self.get_parameter('z_min_limit').value
        self.z_max_limit = self.get_parameter('z_max_limit').value
        
        # --- PROPORTIONAL CONTROL TUNING ---
        self.safe_upper_arm_max = 45.0  
        self.safe_lower_arm_min = 60.0  
        self.safe_lower_arm_max = 100.0 
        
        self.kp_upper = 0.0020  
        self.kp_lower = 0.0015  
        
        self.min_move_threshold = 0.01 
        self.max_step_size = 0.08      
        # -----------------------------------

        # State Machine Variables
        self.latest_msg = None
        self.state = 'IDLE' 
        self.last_action_time = time.time()
        self.telegraph_start_time = 0.0
        self.pending_z_target = 0.0

        self.get_logger().info(f"Connecting to UR5e at {self.robot_ip}...")
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.get_logger().info("Connected! PCB Desoldering Ergonomic Logic active.")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to UR5e: {e}")
            # For testing without a real robot, comment out `raise e`
            raise e

        # Pub/Sub
        self.subscription = self.create_subscription(BodyMsg, '/full_body_data', self.rula_callback, 10)
        self.gui_notification_pub = self.create_publisher(String, '/gui_notifications', 10)

        # NON-BLOCKING TIMER: Replaces time.sleep()
        # Runs at 10Hz to evaluate state without blocking incoming messages
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def rula_callback(self, msg):
        # Simply store the latest data; the timer handles the logic asynchronously
        self.latest_msg = msg

    def control_loop(self):
        if self.latest_msg is None:
            return

        current_time = time.time()

        # State: IDLE (Monitoring operator posture)
        if self.state == 'IDLE':
            if current_time - self.last_action_time < self.cooldown:
                return # Still in cooldown

            self.evaluate_posture()

        # State: TELEGRAPHING (Warning the user before moving)
        elif self.state == 'TELEGRAPHING':
            # Non-blocking wait of 2.0 seconds
            if current_time - self.telegraph_start_time >= 2.0:
                self.state = 'MOVING'
                self.execute_movement()
                
                # Reset state to IDLE and start cooldown
                self.state = 'IDLE'
                self.last_action_time = time.time()

    def evaluate_posture(self):
        msg = self.latest_msg
        z_offset = 0.0
        active_correction = ""

        # Check if we have any valid arm data at all for this frame
        if not msg.left and not msg.right:
            return

        # 1. Upper Arm Error (Shoulders)
        # Only evaluate valid, visible arms
        valid_upper_angles = []
        if msg.right: valid_upper_angles.append(msg.right_arm_up)
        if msg.left: valid_upper_angles.append(msg.left_arm_up)
        
        if valid_upper_angles:
            upper_arm_angle = max(valid_upper_angles)
            if upper_arm_angle > self.safe_upper_arm_max:
                error = upper_arm_angle - self.safe_upper_arm_max
                z_offset -= (error * self.kp_upper)
                active_correction = "raised shoulders"

        # 2. Lower Arm Error (Evaluate independently)
        l_dev, r_dev = 0.0, 0.0
        l_dir, r_dir = 0, 0
        
        # Only calculate deviations if the camera actually sees the arm
        if msg.left:
            l_dev, l_dir = self.get_lower_arm_deviation(msg.left_low_angle)
        if msg.right:
            r_dev, r_dir = self.get_lower_arm_deviation(msg.right_low_angle)

        max_dev = max(l_dev, r_dev)
        if max_dev > 0:
            if l_dev >= r_dev:
                z_offset += l_dir * (l_dev * self.kp_lower)
                arm_str = "left forearm"
                dir_str = "extended" if l_dir == 1 else "cramped"
            else:
                z_offset += r_dir * (r_dev * self.kp_lower)
                arm_str = "right forearm"
                dir_str = "extended" if r_dir == 1 else "cramped"
            
            if not active_correction:
                active_correction = f"{dir_str} {arm_str}"

        # 3. Check threshold and apply limits
        if abs(z_offset) >= self.min_move_threshold:
            # Clamp step size
            z_offset = max(min(z_offset, self.max_step_size), -self.max_step_size)

            # APPLY KINEMATIC LIMITS
            current_pose = self.rtde_r.getActualTCPPose()
            current_z = current_pose[2]
            
            target_z = current_z + z_offset
            # Clamp the absolute Z position to the safe workspace boundaries
            clamped_target_z = max(min(target_z, self.z_max_limit), self.z_min_limit)
            
            actual_offset = clamped_target_z - current_z

            if abs(actual_offset) < 0.005: 
                self.get_logger().debug("Movement aborted: Reached Z-axis workspace limit.")
                return

            self.pending_z_target = clamped_target_z
            
            direction = "UP" if actual_offset > 0 else "DOWN"
            alert_msg = f"Desoldering Assist: Moving PCB {direction} by {round(abs(actual_offset)*100, 1)}cm to ease {active_correction}..."
            
            gui_msg = String()
            gui_msg.data = alert_msg
            self.gui_notification_pub.publish(gui_msg)
            self.get_logger().info(alert_msg)

            self.telegraph_start_time = time.time()
            self.state = 'TELEGRAPHING'

    def get_lower_arm_deviation(self, angle):
        """ Returns (absolute_deviation, direction_multiplier) """
        if angle > self.safe_lower_arm_max:
            # Extended arm: PCB is too low, move UP (+1)
            return (angle - self.safe_lower_arm_max), 1
        elif angle < self.safe_lower_arm_min:
            # Cramped arm: PCB is too high, move DOWN (-1)
            return (self.safe_lower_arm_min - angle), -1
        return 0.0, 0

    def execute_movement(self):
        try:
            target_pose = self.rtde_r.getActualTCPPose()
            # Apply the pre-calculated, clamped Z target
            target_pose[2] = self.pending_z_target 
            
            self.get_logger().info(f"Moving to new Z height: {round(self.pending_z_target, 3)}m")
            
            # Note: moveL blocks execution natively until the motion completes.
            # Asynchronous `moveL` (async=True) can be used if `ur_rtde` version supports it.
            self.rtde_c.moveL(target_pose, 0.05, 0.2)
        except Exception as e:
            self.get_logger().error(f"RTDE Movement failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ProportionalRTDEController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()