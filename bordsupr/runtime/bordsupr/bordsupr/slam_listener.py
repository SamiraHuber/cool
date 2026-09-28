import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_msgs.msg import TFMessage


def _parse_object_id(child_frame_id):
    if not child_frame_id.startswith('object_') or not child_frame_id.endswith('_link'):
        return None

    object_id_str = child_frame_id[len('object_'):-len('_link')]
    if not object_id_str.isdigit():
        return None

    return int(object_id_str)

class SlamListener(Node):
    def __init__(self):
        super().__init__('slam_listener')
        print('STARTED SLAM LISTENER')
        self.sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.callback,
            qos_profile_sensor_data,
        )

    def callback(self, msg):
        for transform in msg.transforms:
            yolo_id = _parse_object_id(transform.child_frame_id)
            if yolo_id is None:
                continue

            stamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
            translation = transform.transform.translation
            frame_id = transform.header.frame_id
            child_frame_id = transform.child_frame_id

            print(
                f'[t={stamp:.3f}] yolo_id={yolo_id} '
                f'frame={frame_id} child_frame={child_frame_id} '
                f'location=({translation.x:.3f}, {translation.y:.3f}, {translation.z:.3f})'
            )

def main():
    rclpy.init()
    node = SlamListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
