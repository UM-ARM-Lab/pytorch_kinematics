import pytest
import torch

import pytorch_kinematics as pk


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distance_close_rows_are_independent(dtype):
    angles = torch.tensor([0.0, 0.01, 0.7, 2.0], dtype=dtype)
    zeros = torch.zeros_like(angles)
    q1 = torch.stack((torch.ones_like(angles), zeros, zeros, zeros), -1)
    q2 = torch.stack(((angles / 2).cos(), zeros, zeros, (angles / 2).sin()), -1)
    batched = pk.quaternion_angular_distance(q1, q2)
    separate = torch.stack([pk.quaternion_angular_distance(a, b) for a, b in zip(q1, q2)])
    torch.testing.assert_close(batched, separate)
    torch.testing.assert_close(batched, torch.tensor([0.0, 0.0, 0.7, 2.0], dtype=dtype))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_distance_endpoint_gradients(sign):
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    q2 = sign * q1.detach()
    result = pk.quaternion_angular_distance(q1, q2)
    torch.testing.assert_close(result, torch.zeros_like(result))
    gradient = torch.autograd.grad(result.sum(), q1)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(q1))


def test_distance_angle_parameter_gradient():
    angles = torch.tensor([0.3, 1.7, 2.9], dtype=torch.float64, requires_grad=True)
    zeros = torch.zeros_like(angles)
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q2 = torch.stack(((angles / 2).cos(), zeros, zeros, (angles / 2).sin()), -1)
    result = pk.quaternion_angular_distance(q1, q2)
    torch.testing.assert_close(result, angles)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), angles)[0], torch.ones_like(angles))


def test_distance_pi_and_multidimensional_batch():
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q2 = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    actual = pk.quaternion_angular_distance(q1, q2)
    torch.testing.assert_close(actual, torch.tensor([[torch.pi, 0.0]], dtype=torch.float64))


def test_distance_matches_scipy_rotation_reference():
    from scipy.spatial.transform import Rotation

    a = Rotation.random(8, random_state=17)
    b = Rotation.random(8, random_state=21)
    q1 = torch.tensor(a.as_quat()[:, [3, 0, 1, 2]], dtype=torch.float64)
    q2 = torch.tensor(b.as_quat()[:, [3, 0, 1, 2]], dtype=torch.float64)
    expected = torch.tensor((a.inv() * b).magnitude(), dtype=torch.float64)
    torch.testing.assert_close(pk.quaternion_angular_distance(q1, q2), expected)
