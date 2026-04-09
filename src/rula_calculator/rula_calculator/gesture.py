import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class GestureDebugger(Node):
    def __init__(self):
        super().__init__('gesture_debugger')
        
        # Subscribe to the new gesture topic we created in point_3D.py
        self.subscription = self.create_subscription(
            String, 
            '/operator_gesture', 
            self.gesture_callback, 
            1
        )
        
        self.get_logger().info('🟢 Gesture Debugger Active. Show your left hand to the Front Camera...')
        self.last_gesture = "NONE"

    def gesture_callback(self, msg):
        current_gesture = msg.data
        
        # We only print when the gesture CHANGES to avoid flooding your terminal
        if current_gesture != self.last_gesture:
            if current_gesture == "THUMBS_UP":
                self.get_logger().info('👍 DETECTED: Thumbs UP (Robot would move UP)')
            elif current_gesture == "THUMBS_DOWN":
                self.get_logger().info('👎 DETECTED: Thumbs DOWN (Robot would move DOWN)')
            elif current_gesture == "RESET":
                self.get_logger().info('✋ DETECTED: Open Hand (Robot would RESET to strict RULA)')
            
            self.last_gesture = current_gesture

def main(args=None):
    rclpy.init(args=args)
    node = GestureDebugger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down debugger...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()