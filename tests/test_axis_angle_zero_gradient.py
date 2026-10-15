import pytest
import torch

import pytorch_kinematics as pk


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_axis_angle_to_quaternion_differential_at_zero(dtype):
    vector = torch.zeros(3, dtype=dtype, requires_grad=True)
    result = pk.axis_angle_to_quaternion(vector)
    torch.testing.assert_close(result, vector.new_tensor([1., 0., 0., 0.]))
    jacobian = torch.autograd.functional.jacobian(pk.axis_angle_to_quaternion, vector)
    expected = torch.cat((torch.zeros(1, 3, dtype=dtype), torch.eye(3, dtype=dtype) / 2))
    torch.testing.assert_close(jacobian, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quaternion_to_axis_angle_differential_at_identity(dtype):
    quaternion = torch.tensor([1., 0., 0., 0.], dtype=dtype, requires_grad=True)
    result = pk.quaternion_to_axis_angle(quaternion)
    torch.testing.assert_close(result, torch.zeros(3, dtype=dtype))
    jacobian = torch.autograd.functional.jacobian(pk.quaternion_to_axis_angle, quaternion)
    expected = torch.cat((torch.zeros(3, 1, dtype=dtype), 2 * torch.eye(3, dtype=dtype)), -1)
    torch.testing.assert_close(jacobian, expected)


def test_axis_angle_mixed_batch_round_trip_and_gradient():
    vectors = torch.tensor([[0., 0., 0.], [1e-8, 0., 0.], [.3, -.8, .4]],
                           dtype=torch.float64, requires_grad=True)
    result = pk.quaternion_to_axis_angle(pk.axis_angle_to_quaternion(vectors))
    torch.testing.assert_close(result, vectors)
    gradient = torch.autograd.grad(result.sum(), vectors)[0]
    torch.testing.assert_close(gradient, torch.ones_like(vectors))


def test_conversion_general_rotation_gradcheck():
    vector = torch.tensor([.2, -.6, 1.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pk.axis_angle_to_quaternion, (vector,))
    quaternion = pk.axis_angle_to_quaternion(vector).detach().requires_grad_()
    assert torch.autograd.gradcheck(pk.quaternion_to_axis_angle, (quaternion,))
