import torch
import tensorflow_kinematics.transforms as tf


def test_transform():
    N = 20
    mats = tf.random_rotations(N, dtype=torch.float64, device="cpu", requires_grad=True)
    assert list(mats.shape) == [N, 3, 3]
    # test batch conversions
    quat = tf.matrix_to_quaternion(mats)
    assert list(quat.shape) == [N, 4]
    mats_recovered = tf.quaternion_to_matrix(quat)
    assert torch.allclose(mats, mats_recovered)

    quat_identity = tf.quaternion_multiply(quat, tf.quaternion_invert(quat))
    assert torch.allclose(tf.quaternion_to_matrix(quat_identity), torch.eye(3, dtype=torch.float64).repeat(N, 1, 1))
