import math
import numpy as np

import tensorflow as tf
import tensorflow_kinematics as pk


def quat_pos_from_transform3d(tg):
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot


def quaternion_equality(a, b):
    # negative of a quaternion is the same rotation
    return np.allclose(a, b) or np.allclose(a, -b)


def test_fkik():
    data = '<robot name="test_robot">' \
           '<link name="link1" />' \
           '<link name="link2" />' \
           '<link name="link3" />' \
           '<joint name="joint1" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link1"/>' \
           '<child link="link2"/>' \
           '</joint>' \
           '<joint name="joint2" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link2"/>' \
           '<child link="link3"/>' \
           '</joint>' \
           '</robot>'
    chain = pk.build_serial_chain_from_urdf(data, 'link3')
    th1 = tf.constant([0.42553542, 0.17529176])
    tg = chain.forward_kinematics(th1)
    pos, rot = quat_pos_from_transform3d(tg)
    assert np.allclose(pos, tf.constant([[1.91081784, 0.41280851, 0.0000]]))
    assert quaternion_equality(rot, tf.constant([[0.95521418, 0.0000, 0.0000, 0.2959153]]))
    print(tg)
    # TODO implement and test inverse kinematics
    # th2 = chain.inverse_kinematics(tg)
    # self.assertTrue(np.allclose(th1, th2, atol=1.0e-6))
    # test batch kinematics
    N = 20
    th_batch = np.random.rand(N, 2)
    tg_batch = chain.forward_kinematics(th_batch)
    m = tg_batch.get_matrix()
    for i in range(N):
        tg = chain.forward_kinematics(th_batch[i])
        assert np.allclose(tg.get_matrix().view(4, 4), m[i])

    # check that gradients are passed through
    th2 = tf.constant([0.42553542, 0.17529176], requires_grad=True)
    tg = chain.forward_kinematics(th2)
    pos, rot = quat_pos_from_transform3d(tg)
    # note that since we are using existing operations we are not checking grad calculation correctness
    assert th2.grad is None
    pos.norm().backward()
    assert th2.grad is not None


def test_urdf():
    chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
    print(chain)
    print(chain.get_joint_parameter_names())
    th = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]
    ret = chain.forward_kinematics(th, end_only=False)
    tg = ret['lbr_iiwa_link_7']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, tf.constant([7.07106781e-01, 0, -7.07106781e-01, 0]))
    assert np.allclose(pos, tf.constant([-6.60827561e-01, 0, 3.74142136e-01]))

    N = 1000
    dtype = tf.float64

    th_batch = np.random.rand(N, len(chain.get_joint_parameter_names()))

    import time
    start = time.time()
    tg_batch = chain.forward_kinematics(th_batch)
    m = tg_batch.get_matrix()
    elapsed = time.time() - start
    print("elapsed {}s for N={} when parallel".format(elapsed, N))

    start = time.time()
    elapsed = 0
    for i in range(N):
        tg = chain.forward_kinematics(th_batch[i])
        elapsed += time.time() - start
        start = time.time()
        assert np.allclose(tg.get_matrix().view(4, 4), m[i])
    print("elapsed {}s for N={} when serial".format(elapsed, N))


# test robot with prismatic and fixed joints
def test_fk_simple_arm():
    chain = pk.build_chain_from_sdf(open("simple_arm.sdf").read())
    # print(chain)
    # print(chain.get_joint_parameter_names())
    ret = chain.forward_kinematics({'arm_elbow_pan_joint': math.pi / 2.0, 'arm_wrist_lift_joint': -0.5})
    tg = ret['arm_wrist_roll']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, tf.constant([0.70710678, 0., 0., 0.70710678]))
    assert np.allclose(pos, tf.constant([1.05, 0.55, 0.5]))

    N = 100
    ret = chain.forward_kinematics(
        {'arm_elbow_pan_joint': np.random.rand(N, 1), 'arm_wrist_lift_joint': np.random.rand(N, 1)})
    tg = ret['arm_wrist_roll']
    assert list(tg.get_matrix().shape) == [N, 4, 4]


def test_cuda():
    d = "cuda"
    dtype = tf.float64
    chain = pk.build_chain_from_sdf(open("simple_arm.sdf").read())

    ret = chain.forward_kinematics({'arm_elbow_pan_joint': math.pi / 2.0, 'arm_wrist_lift_joint': -0.5})
    tg = ret['arm_wrist_roll']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, tf.constant([0.70710678, 0., 0., 0.70710678], dtype=dtype))
    assert np.allclose(pos, tf.constant([1.05, 0.55, 0.5], dtype=dtype))

    data = '<robot name="test_robot">' \
           '<link name="link1" />' \
           '<link name="link2" />' \
           '<link name="link3" />' \
           '<joint name="joint1" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link1"/>' \
           '<child link="link2"/>' \
           '</joint>' \
           '<joint name="joint2" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link2"/>' \
           '<child link="link3"/>' \
           '</joint>' \
           '</robot>'
    chain = pk.build_serial_chain_from_urdf(data, 'link3')
    N = 20
    th_batch = np.random.rand(N, 2)
    tg_batch = chain.forward_kinematics(th_batch)
    m = tg_batch.get_matrix()
    for i in range(N):
        tg = chain.forward_kinematics(th_batch[i])
        assert np.allclose(tg.get_matrix().view(4, 4), m[i])


# test more complex robot and the MJCF parser
def test_fk_mjcf():
    chain = pk.build_chain_from_mjcf(open("ant.xml").read())
    print(chain)
    print(chain.get_joint_parameter_names())
    th = {'hip_1': 1.0, 'ankle_1': 1}
    ret = chain.forward_kinematics(th)
    tg = ret['aux_1_child']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, tf.constant([0.87758256, 0., 0., 0.47942554]))
    assert np.allclose(pos, tf.constant([0.2, 0.2, 0.75]))
    tg = ret['front_left_foot_child']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, tf.constant([0.77015115, -0.4600326, 0.13497724, 0.42073549]))
    assert np.allclose(pos, tf.constant([0.13976626, 0.47635466, 0.75]))
    print(ret)


def test_fk_mjcf_humanoid():
    chain = pk.build_chain_from_mjcf(open("humanoid.xml").read())
    print(chain)
    print(chain.get_joint_parameter_names())
    th = {'left_knee': 0.0, 'right_knee': 0.0}
    ret = chain.forward_kinematics(th)
    print(ret)


if __name__ == "__main__":
    test_fkik()
    test_fk_simple_arm()
    test_fk_mjcf()
    test_cuda()
    test_urdf()
    # test_fk_mjcf_humanoid()
