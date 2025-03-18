import os

import pytorch_kinematics as pk

TEST_DIR = os.path.dirname(__file__)

def test_limits():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(), "lbr_iiwa_link_7")
    for joint in chain.get_joints():
        # Default velocity and effort limits for the iiwa arm
        assert joint.velocity_limits == (-10, 10)
        assert joint.effort_limits == (-300, 300)
    chain = pk.build_chain_from_urdf(open(os.path.join(TEST_DIR, "joint_limit_robot.urdf")).read())
    for joint in chain.get_joints():
        # Slice off the "joint" prefix to get just the number of the joint
        num = int(joint.name[5])
        # This robot is defined specifically to test joint limits. It always
        # sets velocity limits equal to the joint number and effort limits
        # equal to the joint number + 4
        assert joint.velocity_limits == (-num, num)
        assert joint.effort_limits == (-(num + 4), num + 4)


if __name__ == "__main__":
    test_limits()
