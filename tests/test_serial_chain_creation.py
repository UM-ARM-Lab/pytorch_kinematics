import os
from timeit import default_timer as timer

import torch

import pytorch_kinematics as pk

TEST_DIR = os.path.dirname(__file__)


def test_extract_serial_chain_from_tree():
    urdf = "widowx/wx250s.urdf"
    full_urdf = os.path.join(TEST_DIR, urdf)
    chain = pk.build_chain_from_urdf(open(full_urdf, mode="rb").read())
    # full frames
    full_frame_expected = """
base_link
└── shoulder_link
    └── upper_arm_link
        └── upper_forearm_link
            └── lower_forearm_link
                └── wrist_link
                    └── gripper_link
                        └── ee_arm_link
                            ├── gripper_prop_link
                            └── gripper_bar_link
                                └── fingers_link
                                    ├── left_finger_link
                                    ├── right_finger_link
                                    └── ee_gripper_link
    """
    full_frame = chain.print_tree()
    assert full_frame_expected.strip() == full_frame.strip()

    serial_chain = pk.SerialChain(chain, "ee_gripper_link", "base_link")
    serial_frame_expected = """
base_link
└── shoulder_link
    └── upper_arm_link
        └── upper_forearm_link
            └── lower_forearm_link
                └── wrist_link
                    └── gripper_link
                        └── ee_arm_link
                            └── gripper_bar_link
                                └── fingers_link
                                    └── ee_gripper_link
    """
    serial_frame = serial_chain.print_tree()
    assert serial_frame_expected.strip() == serial_frame.strip()

    # full chain should have DOF = 8, however since we are creating just a serial chain to ee_gripper_link, should be 6
    assert chain.n_joints == 8
    assert serial_chain.n_joints == 6

    serial_chain = pk.SerialChain(chain, "gripper_prop_link", "base_link")
    serial_frame_expected = """
base_link
└── shoulder_link
    └── upper_arm_link
        └── upper_forearm_link
            └── lower_forearm_link
                └── wrist_link
                    └── gripper_link
                        └── ee_arm_link
                            └── gripper_prop_link
    """
    serial_frame = serial_chain.print_tree()
    assert serial_frame_expected.strip() == serial_frame.strip()

    serial_chain = pk.SerialChain(chain, "ee_gripper_link", "gripper_link")
    serial_frame_expected = """
 gripper_link
└── ee_arm_link
    └── gripper_bar_link
        └── fingers_link
            └── ee_gripper_link
        """
    serial_frame = serial_chain.print_tree()
    assert serial_frame_expected.strip() == serial_frame.strip()
    # only gripper_link is the parent frame of a joint in this serial chain
    assert serial_chain.n_joints == 1


if __name__ == "__main__":
    test_extract_serial_chain_from_tree()
