import math

import pytest
import torch

import pytorch_kinematics as pk
from pytorch_kinematics.ik import _ik_step_kernel_svd


@pytest.mark.parametrize("dof", [3, 7])
def test_svd_step_matches_weighted_normal_equations(dof):
    dtype = torch.float64
    generator = torch.Generator().manual_seed(3)
    jacobian = torch.randn(2, 6, dof, generator=generator, dtype=dtype)
    poses = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    target_pos = torch.tensor([[0.2, -0.1, 0.3]], dtype=dtype)
    target_quat = torch.tensor([[math.cos(0.2), math.sin(0.2), 0.0, 0.0]], dtype=dtype)
    weights = torch.tensor([1.0, 2.0, 0.3, 0.5, 0.2, 1.2], dtype=dtype)
    reg = 0.07 * torch.eye(6, dtype=dtype)
    lm_damping = 0.1
    expected_dx = torch.tensor([0.2, -0.1, 0.3, 0.4, 0.0, 0.0], dtype=dtype).reshape(1, 6, 1).repeat(2, 1, 1)
    weighted_jacobian = weights.reshape(1, 6, 1) * jacobian
    weighted_dx = weights.reshape(1, 6, 1) * expected_dx
    damping = reg[0, 0] + lm_damping * weighted_dx.square().sum(dim=1, keepdim=True)
    jt = weighted_jacobian.transpose(1, 2)
    expected_dq = torch.linalg.solve(jt @ weighted_jacobian + damping * torch.eye(dof, dtype=dtype), jt @ weighted_dx)

    dq, dx = _ik_step_kernel_svd(poses, target_pos, target_quat, jacobian, reg, 2, lm_damping, weights)

    torch.testing.assert_close(dx, expected_dx)
    torch.testing.assert_close(dq, expected_dq)
    if dof == 3:
        compiled = torch.compile(_ik_step_kernel_svd, fullgraph=True)
        compiled_dq, compiled_dx = compiled(poses, target_pos, target_quat, jacobian, reg, 2, lm_damping, weights)
        torch.testing.assert_close(compiled_dx, expected_dx)
        torch.testing.assert_close(compiled_dq, expected_dq)


def test_svd_ik_reachable_three_dof_targets():
    urdf = """<robot name="three_axis_arm">
        <link name="base"/><link name="yaw"/><link name="pitch"/>
        <link name="roll"/><link name="tool"/>
        <joint name="yaw_joint" type="revolute">
            <parent link="base"/><child link="yaw"/><axis xyz="0 0 1"/>
            <limit lower="-3" upper="3" effort="1" velocity="1"/>
        </joint>
        <joint name="pitch_joint" type="revolute">
            <parent link="yaw"/><child link="pitch"/><origin xyz="0.3 0 0"/>
            <axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="1" velocity="1"/>
        </joint>
        <joint name="roll_joint" type="revolute">
            <parent link="pitch"/><child link="roll"/><origin xyz="0.2 0 0"/>
            <axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="1" velocity="1"/>
        </joint>
        <joint name="tool_joint" type="fixed">
            <parent link="roll"/><child link="tool"/><origin xyz="0.1 0.05 0.1"/>
        </joint>
    </robot>"""
    chain = pk.build_serial_chain_from_urdf(urdf, "tool").to(dtype=torch.float64)
    goal_q = torch.tensor([[0.2, -0.3, 0.4], [-0.1, 0.3, -0.2]], dtype=chain.dtype)
    goal_tf = chain.forward_kinematics(goal_q)
    solver = pk.PseudoInverseIKWithSVD(
        chain,
        retry_configs=torch.zeros(1, 3, dtype=chain.dtype),
        max_iterations=50,
        early_stopping_any_converged=True,
        early_stopping_no_improvement=None,
        lr=0.8,
    )

    solution = solver.solve(goal_tf)

    assert solution.converged.all()
    assert torch.isfinite(solution.solutions).all()
    torch.testing.assert_close(
        chain.forward_kinematics(solution.solutions[:, 0]).get_matrix(), goal_tf.get_matrix(), atol=1e-6, rtol=1e-6
    )
