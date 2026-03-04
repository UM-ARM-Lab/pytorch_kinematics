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
    # gripper_link's own joint connects wrist_link -> gripper_link, which is outside
    # this sub-chain, so it should not be counted. All remaining joints are fixed.
    assert serial_chain.n_joints == 0


def test_serial_chain_non_root_start():
    """Test that SerialChain with a non-root root_frame_name excludes the root's own joint (issue #62)."""
    data = ('<robot name="test_robot">'
            '<link name="link1" />'
            '<link name="link2" />'
            '<link name="link3" />'
            '<link name="link4" />'
            '<joint name="joint1" type="revolute">'
            '<origin xyz="1.0 0.0 0.0"/>'
            '<parent link="link1"/>'
            '<child link="link2"/>'
            '</joint>'
            '<joint name="joint2" type="revolute">'
            '<origin xyz="1.0 0.0 0.0"/>'
            '<parent link="link2"/>'
            '<child link="link3"/>'
            '</joint>'
            '<joint name="joint3" type="revolute">'
            '<origin xyz="1.0 0.0 0.0"/>'
            '<parent link="link3"/>'
            '<child link="link4"/>'
            '</joint>'
            '</robot>')
    full_chain = pk.build_chain_from_urdf(data)

    # SerialChain from link2 to link4 should only include joint2 and joint3, not joint1
    serial = pk.SerialChain(full_chain, "link4", "link2")
    assert "joint1" not in serial.get_joint_parameter_names()
    assert serial.get_joint_parameter_names() == ["joint2", "joint3"]

    # FK from a frame to itself should be identity
    identity_chain = pk.SerialChain(full_chain, "link2", "link2")
    tg = identity_chain.forward_kinematics(torch.zeros(0))
    assert torch.allclose(tg.get_matrix(), torch.eye(4).unsqueeze(0), atol=1e-7)

    # FK from link2 to link4 with zeros should give translation of (2, 0, 0) — two joints each offset by (1, 0, 0)
    tg = serial.forward_kinematics(torch.zeros(2))
    pos = tg.get_matrix()[0, :3, 3]
    assert torch.allclose(pos, torch.tensor([2.0, 0.0, 0.0]), atol=1e-7)

    # Compare: full chain FK at zeros, link4 relative to link2
    full_ret = full_chain.forward_kinematics(torch.zeros(3))
    full_link2 = full_ret['link2'].get_matrix()
    full_link4 = full_ret['link4'].get_matrix()
    relative = torch.linalg.inv(full_link2) @ full_link4
    assert torch.allclose(relative, tg.get_matrix(), atol=1e-7)


if __name__ == "__main__":
    test_extract_serial_chain_from_tree()
    test_serial_chain_non_root_start()
