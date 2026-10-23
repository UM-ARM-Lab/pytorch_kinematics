import math

import pytest
import torch

import pytorch_kinematics as pk
from pytorch_kinematics.frame import Joint


def _urdf(joint_type, axis, yaw=0.0):
    axis_xml = '' if axis is None else f'<axis xyz="{axis}"/>'
    return f'''<robot name="joint_axis">
      <link name="base"/><link name="tip"/>
      <joint name="joint" type="{joint_type}">
        <parent link="base"/><child link="tip"/>
        <origin xyz="0 0 0" rpy="0 0 {yaw}"/>
        {axis_xml}
        <limit lower="-2" upper="2" effort="1" velocity="1"/>
      </joint>
    </robot>'''


@pytest.mark.parametrize('joint_type', ['prismatic', 'revolute', 'continuous'])
@pytest.mark.parametrize('axis', [None, '1 0 0', '0 0 1'])
@pytest.mark.parametrize('yaw', [0.0, math.pi / 2])
def test_urdf_joint_axis_fk_jacobian_and_gradients(joint_type, axis, yaw):
    chain = pk.build_serial_chain_from_urdf(_urdf(joint_type, axis, yaw), 'tip')
    chain = chain.to(dtype=torch.float64, device='cpu')
    q = torch.tensor([[0.0], [0.3], [-0.7]], dtype=torch.float64, requires_grad=True)
    actual = chain.forward_kinematics(q).get_matrix()

    # URDF axes live in the joint frame; omitted axes mean X, even after an origin rotation.
    origin = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.0],
                           [math.sin(yaw), math.cos(yaw), 0.0],
                           [0.0, 0.0, 1.0]], dtype=torch.float64)
    local_axis = torch.tensor([0.0, 0.0, 1.0] if axis == '0 0 1' else [1.0, 0.0, 0.0],
                              dtype=torch.float64)
    world_axis = origin @ local_axis
    expected = torch.eye(4, dtype=torch.float64).repeat(len(q), 1, 1)
    jacobian = torch.zeros(len(q), 6, 1, dtype=torch.float64)
    if joint_type == 'prismatic':
        expected[:, :3, :3] = origin
        expected[:, :3, 3] = q * world_axis
        jacobian[:, :3, 0] = world_axis
    else:
        c, s = q[:, 0].cos(), q[:, 0].sin()
        rotation = torch.eye(3, dtype=torch.float64).repeat(len(q), 1, 1)
        i, j = (0, 1) if axis == '0 0 1' else (1, 2)
        rotation[:, i, i] = c
        rotation[:, i, j] = -s
        rotation[:, j, i] = s
        rotation[:, j, j] = c
        expected[:, :3, :3] = origin @ rotation
        jacobian[:, 3:, 0] = world_axis
    # Origin transforms are parsed in float32 before the chain is promoted to float64.
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(chain.jacobian(q), jacobian, atol=2e-7, rtol=2e-7)
    weights = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    actual_gradient = torch.autograd.grad((actual * weights).sum(), q)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), q)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=2e-7)


@pytest.mark.parametrize('axis,reachable', [(None, False), ('1 0 0', False), ('0 0 1', True)])
def test_omitted_prismatic_axis_does_not_make_z_target_reachable(axis, reachable):
    chain = pk.build_serial_chain_from_urdf(_urdf('prismatic', axis), 'tip')
    chain = chain.to(dtype=torch.float64, device='cpu')
    solver = pk.PseudoInverseIK(
        chain, retry_configs=torch.zeros(1, 1, dtype=torch.float64), max_iterations=10,
        lr=1.0, pos_tolerance=1e-5, rot_tolerance=1e-5,
        early_stopping_any_converged=True, early_stopping_no_improvement=None,
        enforce_joint_limits=False, num_limit_refinement_iterations=0, use_compile=False,
    )
    goal = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    goal[0, 2, 3] = 0.7
    result = solver.solve(pk.Transform3d(matrix=goal, dtype=torch.float64, device='cpu'))
    assert bool(result.converged[0, 0]) is reachable
    assert torch.isfinite(result.solutions).all()
    if reachable:
        torch.testing.assert_close(result.solutions[0, 0, 0], torch.tensor(0.7, dtype=torch.float64),
                                   atol=1e-5, rtol=0)
    else:
        assert result.err_pos[0, 0] >= 0.7 - 1e-10


def test_generic_joint_default_remains_z():
    torch.testing.assert_close(Joint(joint_type='revolute').axis, torch.tensor([0.0, 0.0, 1.0]))


def test_fixed_urdf_joint_keeps_its_existing_axis_metadata():
    chain = pk.build_chain_from_urdf(_urdf('fixed', None))
    torch.testing.assert_close(chain.find_joint('joint').axis, torch.tensor([0.0, 0.0, 1.0]))
