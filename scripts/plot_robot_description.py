import numpy as np

import rospy
import tensorflow_kinematics as tk
from arc_utilities.ros_helpers import get_connected_publisher
from urdf_parser_py.xml_reflection import core as urdf_core
from visualization_msgs.msg import Marker


def on_error(message):
    rospy.logdebug(message, logger_name='urdf_parser_py')


urdf_core.on_error = on_error


def plot_point(frame, pos, ns):
    msg = Marker()
    msg.action = Marker.ADD
    msg.type = Marker.SPHERE
    msg.ns = ns
    msg.header.frame_id = frame
    msg.pose.position.x = pos[0]
    msg.pose.position.y = pos[1]
    msg.pose.position.z = pos[2]
    msg.pose.orientation.w = 1
    msg.scale.x = 0.01
    msg.scale.y = 0.01
    msg.scale.z = 0.01
    msg.color.r = 1
    msg.color.a = 1
    pub = get_connected_publisher('/point', Marker, queue_size=10)
    pub.publish(msg)


def main():
    rospy.init_node('scratch')
    np.set_printoptions(precision=3, suppress=True)

    urdf = rospy.get_param('robot_description')

    chain = tk.build_chain_from_urdf(urdf)
    joint_positions = [0] * 20
    ret = chain.forward_kinematics(joint_positions)

    for k, v in ret.items():
        plot_point('world', v.pos().numpy().squeeze(), ns=k)


if __name__ == "__main__":
    main()
