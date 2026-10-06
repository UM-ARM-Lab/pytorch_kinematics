import numpy as np
import pytest
import torch

import pytorch_kinematics as pk


URDF = """<robot name="polar">
<link name="base"/><link name="rotated"/><link name="tool"/>
<joint name="yaw" type="revolute">
  <parent link="base"/><child link="rotated"/><axis xyz="0 0 1"/>
  <limit lower="-3" upper="3" effort="10" velocity="10"/>
</joint>
<joint name="slide" type="prismatic">
  <parent link="rotated"/><child link="tool"/><origin xyz="0.5 0 0"/>
  <axis xyz="1 0 0"/><limit lower="-0.4" upper="1" effort="10" velocity="10"/>
</joint></robot>"""


def values(kind, dtype, requires_grad=False):
    yaw = torch.tensor([0.3, -0.4, 0.8] if kind.endswith("batch") else 0.3, dtype=dtype)
    slide = torch.tensor([0.2, 0.1, -0.1] if kind.endswith("batch") else 0.2, dtype=dtype)
    if kind == "float":
        mapped = {"slide": slide.item(), "yaw": yaw.item()}
    elif kind.startswith("numpy"):
        mapped = {"slide": slide.numpy(), "yaw": yaw.numpy()}
    else:
        mapped = {"slide": slide.requires_grad_(requires_grad),
                  "yaw": yaw.requires_grad_(requires_grad)}
    return mapped, yaw.reshape(-1), slide.reshape(-1)


def polar_reference(yaw, slide, offset=None):
    dx, dy, dz = (0.0, 0.0, 0.0) if offset is None else offset
    c, s = yaw.cos(), yaw.sin()
    radius = 0.5 + slide + dx
    px, py = radius * c - dy * s, radius * s + dy * c
    position = torch.stack((px, py, torch.full_like(yaw, dz)), -1)
    matrix = torch.eye(4, dtype=yaw.dtype).expand(len(yaw), 4, 4).clone()
    matrix[:, 0, 0], matrix[:, 0, 1] = c, -s
    matrix[:, 1, 0], matrix[:, 1, 1] = s, c
    matrix[:, :3, 3] = position
    jacobian = torch.zeros(len(yaw), 6, 2, dtype=yaw.dtype)
    jacobian[:, 0, 0], jacobian[:, 1, 0] = -py, px
    jacobian[:, 0, 1], jacobian[:, 1, 1] = c, s
    jacobian[:, 5, 0] = 1
    return matrix, jacobian


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["float", "numpy_scalar", "tensor_scalar", "numpy_batch", "tensor_batch"])
def test_named_joint_forward_kinematics_matches_physical_polar_arm(dtype, kind):
    chain = pk.build_serial_chain_from_urdf(URDF, "tool").to(dtype=dtype)
    mapped, yaw, slide = values(kind, dtype)
    expected, _ = polar_reference(yaw, slide)
    torch.testing.assert_close(chain.forward_kinematics(mapped).get_matrix(), expected)
    all_frames = chain.forward_kinematics(mapped, end_only=False)
    torch.testing.assert_close(all_frames["tool"].get_matrix(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["float", "numpy_scalar", "tensor_scalar", "numpy_batch", "tensor_batch"])
@pytest.mark.parametrize("offset", [None, (0.1, 0.05, 0.2)])
def test_named_joint_jacobian_and_returned_pose_match_physical_arm(dtype, kind, offset):
    chain = pk.build_serial_chain_from_urdf(URDF, "tool").to(dtype=dtype)
    mapped, yaw, slide = values(kind, dtype)
    expected_pose, expected_jacobian = polar_reference(yaw, slide, offset)
    actual_jacobian, actual_pose = chain.jacobian(mapped, locations=offset, ret_eef_pose=True)
    torch.testing.assert_close(actual_jacobian, expected_jacobian)
    torch.testing.assert_close(actual_pose, expected_pose)
    if offset is None:
        torch.testing.assert_close(chain.jacobian(mapped), expected_jacobian)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["tensor_scalar", "tensor_batch"])
@pytest.mark.parametrize("offset", [None, (0.1, 0.05, 0.2)])
def test_named_tensor_jacobian_preserves_parameter_gradients(dtype, kind, offset):
    chain = pk.build_serial_chain_from_urdf(URDF, "tool").to(dtype=dtype)
    mapped, yaw, slide = values(kind, dtype, requires_grad=True)
    actual = chain.jacobian(mapped, locations=offset)
    _, expected = polar_reference(yaw, slide, offset)
    actual_grad = torch.autograd.grad(actual.sum(), (mapped["yaw"], mapped["slide"]), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (mapped["yaw"], mapped["slide"]))
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["list", "numpy", "tensor"])
def test_unnamed_inputs_keep_original_jacobian_behavior(dtype, kind):
    chain = pk.build_serial_chain_from_urdf(URDF, "tool").to(dtype=dtype)
    configuration = torch.tensor([0.3, 0.2], dtype=dtype)
    source = {"list": configuration.tolist(), "numpy": configuration.numpy(),
              "tensor": configuration}[kind]
    _, expected = polar_reference(configuration[:1], configuration[1:])
    torch.testing.assert_close(chain.jacobian(source), expected)
