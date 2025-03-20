import os

import pytorch_kinematics as pk

TEST_DIR = os.path.dirname(__file__)

def test_limits():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(), "lbr_iiwa_link_7")
    iiwa_low_individual = []
    iiwa_high_individual = []
    for joint in chain.get_joints():
        # Default velocity and effort limits for the iiwa arm
        assert joint.velocity_limits == (-10, 10)
        assert joint.effort_limits == (-300, 300)
        iiwa_low_individual.append(joint.limits[0])
        iiwa_high_individual.append(joint.limits[1])
    iiwa_low, iiwa_high = chain.get_joint_limits()
    assert iiwa_low == iiwa_low_individual
    assert iiwa_high == iiwa_high_individual
    chain = pk.build_chain_from_urdf(open(os.path.join(TEST_DIR, "joint_limit_robot.urdf")).read())
    nums = []
    for joint in chain.get_joints():
        # Slice off the "joint" prefix to get just the number of the joint
        num = int(joint.name[5])
        nums.append(num)
        # This robot is defined specifically to test joint limits. For joint
        # number `num`, it sets lower, upper, velocity, and effort limits to
        # `num+8`, `num+9`, `num`, and `num+4` respectively
        assert joint.limits == (num + 8, num + 9)
        assert joint.velocity_limits == (-num, num)
        assert joint.effort_limits == (-(num + 4), num + 4)
    low, high = chain.get_joint_limits()
    v_low, v_high = chain.get_joint_velocity_limits()
    e_low, e_high = chain.get_joint_effort_limits()
    assert low == [x + 8 for x in nums]
    assert high == [x + 9 for x in nums]
    assert v_low == [-x for x in nums]
    assert v_high == [x for x in nums]
    assert e_low == [-(x + 4) for x in nums]
    assert e_high == [x + 4 for x in nums]


def test_empty_limits():
    chain = pk.build_chain_from_urdf(open(os.path.join(TEST_DIR, "joint_no_limit_robot.urdf")).read())
    nums = []
    for joint in chain.get_joints():
        # Slice off the "joint" prefix to get just the number of the joint
        num = int(joint.name[5])
        nums.append(num)
        # This robot is defined specifically to test joint limits. For joint
        # number `num`, it sets velocity, and effort limits to
        # `num`, and `num+4` respectively, and leaves the lower and upper
        # limits undefined
        assert joint.limits == (0, 0)
        assert joint.velocity_limits == (-num, num)
        assert joint.effort_limits == (-(num + 4), num + 4)
    low, high = chain.get_joint_limits()
    v_low, v_high = chain.get_joint_velocity_limits()
    e_low, e_high = chain.get_joint_effort_limits()
    assert low == [0] * len(nums)
    assert high == [0] * len(nums)
    assert v_low == [-x for x in nums]
    assert v_high == [x for x in nums]
    assert e_low == [-(x + 4) for x in nums]
    assert e_high == [x + 4 for x in nums]


if __name__ == "__main__":
    test_limits()
    test_empty_limits()
