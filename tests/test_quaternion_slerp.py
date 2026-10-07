import math

import pytest
import torch

from pytorch_kinematics.transforms.math import quaternion_slerp


def _z_rotation(angle, dtype):
    angle = torch.as_tensor(angle, dtype=dtype) / 2
    return torch.stack((angle.cos(), torch.zeros_like(angle), torch.zeros_like(angle), angle.sin()), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.0])
def test_slerp_mixed_batch_has_independent_shortest_paths(dtype, t):
    # Identity, antipodal identity, tiny and large rotations share a batch.
    angles = torch.tensor([0.0, 0.0, 1e-4, math.pi / 2], dtype=dtype)
    starts = _z_rotation(torch.zeros_like(angles), dtype)
    ends = _z_rotation(angles, dtype)
    ends[1] *= -1
    ends[3] *= -1
    starts_before, ends_before = starts.clone(), ends.clone()

    actual = quaternion_slerp(starts, ends, t)
    expected = _z_rotation(angles * t, dtype)

    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(4, dtype=dtype))
    torch.testing.assert_close(starts, starts_before, atol=0, rtol=0)
    torch.testing.assert_close(ends, ends_before, atol=0, rtol=0)
    for i in range(4):
        torch.testing.assert_close(actual[i], quaternion_slerp(starts[i], ends[i], t))


def test_slerp_normalizes_the_nearby_linear_branch():
    starts = _z_rotation([0.0], torch.float64)
    ends = _z_rotation([0.002], torch.float64)
    actual = quaternion_slerp(starts, ends, 0.5)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(1, dtype=torch.float64), atol=1e-15, rtol=0)
    torch.testing.assert_close(actual, _z_rotation([0.001], torch.float64), atol=1e-15, rtol=0)


def test_slerp_supports_per_quaternion_times_and_multiple_batch_dimensions():
    start_angles = torch.tensor([[0.0, 0.3, -0.2], [0.7, -0.5, 0.4]], dtype=torch.float64)
    end_angles = torch.tensor([[0.0, 0.3001, 1.0], [0.7001, 0.8, -1.0]], dtype=torch.float64)
    times = torch.tensor([[0.0, 0.25, 0.5], [0.75, 1.0, 0.3]], dtype=torch.float64)
    actual = quaternion_slerp(_z_rotation(start_angles, torch.float64), _z_rotation(end_angles, torch.float64), times)
    expected = _z_rotation(start_angles + times * (end_angles - start_angles), torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_slerp_has_finite_gradients_for_mixed_leaf_inputs(dtype):
    starts = _z_rotation([0.0, 0.0, 0.3], dtype).requires_grad_()
    ends = _z_rotation([0.0, 1e-4, 1.2], dtype)
    ends[0] *= -1
    ends = ends.requires_grad_()
    time = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    actual = quaternion_slerp(starts, ends, time)
    grads = torch.autograd.grad(actual.sum(), (starts, ends, time))
    for grad in grads:
        assert torch.isfinite(grad).all()


def test_slerp_gradient_matches_finite_differences_on_both_branches():
    start_angles = torch.tensor([0.0, 0.3], dtype=torch.float64, requires_grad=True)
    end_angles = torch.tensor([1e-4, 1.2], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0.4, 0.7], dtype=torch.float64, requires_grad=True)

    def interpolate(a, b, t):
        return quaternion_slerp(_z_rotation(a, torch.float64), _z_rotation(b, torch.float64), t)

    assert torch.autograd.gradcheck(interpolate, (start_angles, end_angles, times), atol=1e-6, rtol=1e-4)


def test_slerp_matches_great_circle_oracle_for_general_quaternions():
    generator = torch.Generator().manual_seed(5)
    starts = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    starts = starts / starts.norm(dim=-1, keepdim=True)
    tangent = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    tangent = tangent - (starts * tangent).sum(dim=-1, keepdim=True) * starts
    tangent = tangent / tangent.norm(dim=-1, keepdim=True)
    angles = torch.tensor([0.0, 1e-4, 0.4, 0.9, 0.0, 1e-4, 0.4, 0.9], dtype=torch.float64)
    ends = starts * angles.cos()[:, None] + tangent * angles.sin()[:, None]
    ends[4:] *= -1
    ends_before = ends.clone()
    times = torch.linspace(0.1, 0.9, 8, dtype=torch.float64)

    actual = quaternion_slerp(starts, ends, times)
    expected = starts * (times * angles).cos()[:, None] + tangent * (times * angles).sin()[:, None]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(8, dtype=torch.float64), atol=1e-15, rtol=0)
    torch.testing.assert_close(ends, ends_before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.0])
def test_slerp_low_precision_identity_limits_are_finite(dtype, t):
    angles = torch.tensor([0.0, 0.0, 1e-4, math.pi / 2], dtype=dtype)
    starts = _z_rotation(torch.zeros_like(angles), dtype)
    ends = _z_rotation(angles, dtype)
    ends[1] *= -1
    actual = quaternion_slerp(starts, ends, t)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[:2], starts[:2], atol=0, rtol=0)
    tol = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, _z_rotation(angles * t, dtype), atol=tol, rtol=tol)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(4, dtype=dtype), atol=tol, rtol=tol)


@pytest.mark.parametrize("angles", [[1e-4, 2e-4], [0.0, math.pi / 2]])
def test_slerp_column_times_preserve_paired_shape_and_values(angles):
    angles = torch.tensor(angles, dtype=torch.float64)
    starts = _z_rotation(torch.zeros_like(angles), torch.float64)
    ends = _z_rotation(angles, torch.float64)
    times = torch.tensor([[0.2], [0.8]], dtype=torch.float64)
    actual = quaternion_slerp(starts, ends, times)
    assert actual.shape == starts.shape
    expected = _z_rotation(angles * times[:, 0], torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    vector_times = quaternion_slerp(starts, ends, times[:, 0])
    torch.testing.assert_close(actual, vector_times, atol=0, rtol=0)


def test_slerp_column_times_broadcast_over_leading_batch_dimensions():
    angles = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64)
    starts = _z_rotation(torch.zeros_like(angles), torch.float64)
    ends = _z_rotation(angles, torch.float64)
    times = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64)
    actual = quaternion_slerp(starts, ends, times)
    assert actual.shape == starts.shape
    expected = quaternion_slerp(starts, ends, times[:, 0])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_slerp_per_pair_and_column_times_with_singleton_batch_dimension():
    angles = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    starts = _z_rotation(torch.zeros_like(angles), torch.float64)
    ends = _z_rotation(angles, torch.float64)
    times = torch.tensor([[0.2], [0.8]], dtype=torch.float64)
    per_pair = quaternion_slerp(starts, ends, times)
    column = quaternion_slerp(starts, ends, times[..., None])
    assert per_pair.shape == starts.shape
    assert column.shape == starts.shape
    torch.testing.assert_close(per_pair, column, atol=0, rtol=0)


def test_slerp_single_pair_with_vector_and_column_times():
    starts = _z_rotation(0.0, torch.float64)
    ends = _z_rotation(0.1, torch.float64)
    times = torch.tensor([0.2, 0.8], dtype=torch.float64)
    vector = quaternion_slerp(starts, ends, times)
    column = quaternion_slerp(starts, ends, times[:, None])
    assert vector.shape == (2, 4)
    assert column.shape == (2, 4)
    torch.testing.assert_close(vector, column, atol=0, rtol=0)


def test_slerp_column_time_gradients_are_pairwise_and_match_finite_differences():
    start_angles = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
    end_angles = torch.tensor([0.1001, -0.1998], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([[0.2], [0.8]], dtype=torch.float64, requires_grad=True)

    def interpolate(a, b, t):
        return quaternion_slerp(_z_rotation(a, torch.float64), _z_rotation(b, torch.float64), t)

    assert interpolate(start_angles, end_angles, times).shape == (2, 4)
    assert torch.autograd.gradcheck(interpolate, (start_angles, end_angles, times), atol=1e-6, rtol=1e-4)
    jacobian = torch.autograd.functional.jacobian(lambda t: interpolate(start_angles, end_angles, t), times)
    assert jacobian.shape == (2, 4, 2, 1)
    torch.testing.assert_close(jacobian[0, :, 1, 0], torch.zeros(4, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(jacobian[1, :, 0, 0], torch.zeros(4, dtype=torch.float64), atol=0, rtol=0)


def test_slerp_column_broadcasting_preserves_existing_axis_assignment():
    angles = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4) * 1e-4
    starts = _z_rotation(torch.zeros_like(angles), torch.float64)
    ends = _z_rotation(angles, torch.float64)
    times = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=torch.float64)
    actual = quaternion_slerp(starts, ends, times)
    # Existing column broadcasting assigns these four times to the inner axis.
    expected = quaternion_slerp(starts, ends, times[None])
    assert actual.shape == starts.shape
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("quaternion_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("time_dtype", [torch.float32, torch.float64])
def test_slerp_tensor_column_times_preserve_dtype_promotion_and_precision(quaternion_dtype, time_dtype):
    starts = _z_rotation([0.0, 0.0], quaternion_dtype)
    ends = _z_rotation([1e-4, 2e-4], quaternion_dtype)
    times = torch.tensor([[0.234567891234], [0.876543219876]], dtype=time_dtype)
    reference = starts + times * (ends - starts)
    reference = reference / reference.norm(dim=-1, keepdim=True)
    actual = quaternion_slerp(starts, ends, times)
    assert actual.dtype == reference.dtype
    assert actual.shape == starts.shape
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_slerp_python_scalar_times_retain_endpoint_precision(dtype):
    starts = _z_rotation([0.0, 0.0], dtype)
    ends = _z_rotation([1e-4, 2e-4], dtype)
    time = 0.234567891234
    reference_time = torch.tensor(time, dtype=dtype)
    reference = starts + reference_time * (ends - starts)
    reference = reference / reference.norm(dim=-1, keepdim=True)
    actual = quaternion_slerp(starts, ends, time)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_slerp_zero_dimensional_tensor_time_preserves_scalar_promotion(dtype):
    starts = _z_rotation([0.0, 0.0], dtype)
    ends = _z_rotation([1e-4, 2e-4], dtype)
    time = torch.tensor(0.234567891234, dtype=torch.float64, requires_grad=True)
    reference = starts + time * (ends - starts)
    reference = reference / reference.norm(dim=-1, keepdim=True)
    actual = quaternion_slerp(starts, ends, time)
    assert actual.dtype == reference.dtype
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    (grad,) = torch.autograd.grad(actual.sum(), time)
    assert grad.dtype == time.dtype
    assert torch.isfinite(grad)


@pytest.mark.parametrize("time_dtype", [torch.int64, torch.bool])
def test_slerp_nonfloating_tensor_times_preserve_endpoint_dtype(time_dtype):
    starts = _z_rotation([0.0, 0.0], torch.float64)
    ends = _z_rotation([1e-4, 2e-4], torch.float64)
    times = torch.tensor([[0], [1]], dtype=time_dtype)
    actual = quaternion_slerp(starts, ends, times)
    reference = starts + times * (ends - starts)
    reference = reference / reference.norm(dim=-1, keepdim=True)
    assert actual.dtype == reference.dtype
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_slerp_mixed_dtype_column_time_gradients_match_promoted_reference():
    starts = _z_rotation([0.0, 0.0], torch.float32).requires_grad_()
    ends = _z_rotation([1e-4, 2e-4], torch.float32).requires_grad_()
    times = torch.tensor([[0.234567891234], [0.876543219876]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[0.3, -0.2, 0.7, -0.5], [0.9, 0.4, 0.2, -0.1]], dtype=torch.float64)
    actual = quaternion_slerp(starts, ends, times)
    assert actual.dtype == torch.float64
    actual_grads = torch.autograd.grad((actual * weights).sum(), (starts, ends, times))
    reference = starts + times * (ends - starts)
    reference = reference / reference.norm(dim=-1, keepdim=True)
    reference_grads = torch.autograd.grad((reference * weights).sum(), (starts, ends, times))
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert actual_grad.dtype == reference_grad.dtype
        torch.testing.assert_close(actual_grad, reference_grad, atol=0, rtol=0)
    assert torch.autograd.gradcheck(
        lambda t: quaternion_slerp(starts.detach(), ends.detach(), t), (times,), atol=1e-9, rtol=1e-4
    )
