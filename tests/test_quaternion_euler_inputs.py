import pytest
import torch

from pytorch_kinematics.transforms import quaternion_from_euler, quaternion_to_matrix


SEQUENCES = ["xyz", "xyx", "xzy", "xzx", "yzx", "yzy",
             "yxz", "yxy", "zxy", "zxz", "zyx", "zyz"]
CONVENTIONS = [frame + sequence for frame in ("s", "r") for sequence in SEQUENCES]


def axis_rotation(angle, axis):
    cosine, sine = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    rows = {
        "x": ((one, zero, zero), (zero, cosine, -sine), (zero, sine, cosine)),
        "y": ((cosine, zero, sine), (zero, one, zero), (-sine, zero, cosine)),
        "z": ((cosine, -sine, zero), (sine, cosine, zero), (zero, zero, one)),
    }[axis]
    return torch.stack([torch.stack(row, -1) for row in rows], -2)


def matrix_reference(angles, convention):
    rotations = [axis_rotation(angles[..., i], axis)
                 for i, axis in enumerate(convention[1:])]
    if convention[0] == "s":
        return rotations[2] @ rotations[1] @ rotations[0]
    return rotations[0] @ rotations[1] @ rotations[2]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("convention", CONVENTIONS)
def test_euler_conversion_preserves_angles_and_matches_axis_composition(dtype, convention):
    for batch in (False, True):
        angles = torch.tensor([[0.31, -0.47, 0.83], [-0.62, 0.35, -0.21]], dtype=dtype)
        if not batch:
            angles = angles[0]
        original = angles.clone()
        expected = matrix_reference(original, convention)
        actual = quaternion_to_matrix(quaternion_from_euler(angles, convention))
        torch.testing.assert_close(angles, original, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected)
        repeated = quaternion_to_matrix(quaternion_from_euler(angles, convention))
        torch.testing.assert_close(repeated, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("convention", CONVENTIONS)
def test_euler_conversion_gradients_match_axis_composition(dtype, convention):
    angles = torch.tensor([[0.31, -0.47, 0.83], [-0.62, 0.35, -0.21]],
                          dtype=dtype, requires_grad=True)
    reference_angles = angles.detach().clone().requires_grad_(True)
    coefficients = torch.linspace(-1, 0.9, 9, dtype=dtype).reshape(3, 3)
    actual = quaternion_to_matrix(quaternion_from_euler(angles, convention))
    expected = matrix_reference(reference_angles, convention)
    actual_gradient = torch.autograd.grad((actual * coefficients).sum(), angles)[0]
    expected_gradient = torch.autograd.grad((expected * coefficients).sum(), reference_angles)[0]
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual_gradient, expected_gradient)
