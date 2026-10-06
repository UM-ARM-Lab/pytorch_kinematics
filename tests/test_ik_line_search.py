import pytest
import torch

import pytorch_kinematics as pk


SLIDE_URDF = """<robot name="linear_slide">
    <link name="base"/><link name="tool"/>
    <joint name="slide" type="prismatic">
        <parent link="base"/><child link="tool"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" effort="10" velocity="1"/>
    </joint>
</robot>"""


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tensor_fk", [False, True])
def test_backtracking_accepts_exact_steps_at_different_error_scales(dtype, tensor_fk):
    chain = pk.build_serial_chain_from_urdf(SLIDE_URDF, "tool").to(dtype=dtype)
    # For this prismatic chain, x(q) = q: all three steps reach their goal exactly.
    target_q = torch.tensor([[0.0002], [0.009], [0.5]], dtype=dtype)
    targets = chain.forward_kinematics(target_q).get_matrix()
    q = torch.zeros_like(target_q)
    initial = chain.forward_kinematics(q).get_matrix()
    dx = torch.zeros(3, 6, 1, dtype=dtype)
    dx[:, :3, 0] = targets[:, :3, 3] - initial[:, :3, 3]
    target_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype).expand(3, -1)
    kwargs = {}
    if tensor_fk:
        kwargs = {"fk_fn": chain.forward_kinematics_tensor,
                  "eef_frame_idx": chain._serial_eef_frame_idx}

    lr, improvement = pk.BacktrackingLineSearch().do_line_search(
        chain, q, target_q, targets[:, :3, 3], target_wxyz, dx, **kwargs,
    )

    torch.testing.assert_close(lr, torch.ones_like(lr))
    torch.testing.assert_close(improvement, target_q[:, 0])
    achieved = chain.forward_kinematics(q + lr.unsqueeze(-1) * target_q).get_matrix()
    torch.testing.assert_close(achieved, targets)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("distance", [0.009, 0.005, -0.007])
def test_backtracking_ik_converges_to_reachable_nearby_goal(dtype, distance):
    chain = pk.build_serial_chain_from_urdf(SLIDE_URDF, "tool").to(dtype=dtype)
    target_q = torch.tensor([[distance]], dtype=dtype)
    target = chain.forward_kinematics(target_q)
    solver = pk.PseudoInverseIK(
        chain, retry_configs=torch.zeros(1, 1, dtype=dtype),
        line_search=pk.BacktrackingLineSearch(),
        # This supported option isolates the IK iteration from the separate
        # unbacktracked limit-refinement pass. The target remains within limits.
        enforce_joint_limits=False,
        early_stopping_any_converged=True,
    )

    result = solver.solve(target)

    assert result.converged.all()
    achieved_x = chain.forward_kinematics(result.solutions[:, 0]).get_matrix()[:, 0, 3]
    torch.testing.assert_close(achieved_x, target_q[:, 0], rtol=0.0, atol=solver.pos_tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nearby_goal_is_already_reachable_without_backtracking(dtype):
    chain = pk.build_serial_chain_from_urdf(SLIDE_URDF, "tool").to(dtype=dtype)
    target = chain.forward_kinematics(torch.tensor([[0.009]], dtype=dtype))
    solver = pk.PseudoInverseIK(
        chain, retry_configs=torch.zeros(1, 1, dtype=dtype),
        enforce_joint_limits=False, early_stopping_any_converged=True,
    )
    assert solver.solve(target).converged.all()
