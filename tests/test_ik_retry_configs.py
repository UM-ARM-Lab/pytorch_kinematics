import pytest
import torch

import pytorch_kinematics as pk


def slide_chain(dof, dtype):
    links = '<link name="base"/>'
    joints = ''
    axes = ['1 0 0', '0 1 0']
    for i in range(dof):
        links += f'<link name="link_{i}"/>'
        parent = 'base' if i == 0 else f'link_{i - 1}'
        joints += f'''<joint name="slide_{i}" type="prismatic">
            <parent link="{parent}"/><child link="link_{i}"/><axis xyz="{axes[i]}"/>
            <limit lower="-1" upper="1" effort="10" velocity="1"/>
        </joint>'''
    return pk.build_serial_chain_from_urdf(
        f'<robot name="linear_stages">{links}{joints}</robot>', f'link_{dof - 1}',
    ).to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_targets", [1, 3])
@pytest.mark.parametrize("dof,num_retries", [(1, 1), (1, 3), (1, 5), (2, 2), (2, 3)])
def test_per_target_retry_configs_preserve_individual_warm_starts(dtype, num_targets, dof, num_retries):
    chain = slide_chain(dof, dtype)
    goal_q = torch.linspace(-0.4, 0.6, num_targets * dof, dtype=dtype).reshape(num_targets, dof)
    seeds = goal_q.unsqueeze(1).repeat(1, num_retries, 1)
    target = chain.forward_kinematics(goal_q)
    solver = pk.PseudoInverseIK(
        chain, retry_configs=seeds, max_iterations=1,
        early_stopping_any_converged=True, enforce_joint_limits=False,
    )

    solution = solver.solve(target)

    assert solution.solutions.shape == seeds.shape
    assert solution.converged.all()
    torch.testing.assert_close(solution.solutions, seeds)
    poses = chain.forward_kinematics(solution.solutions.reshape(-1, dof)).get_matrix()
    expected = target.get_matrix().unsqueeze(1).repeat(1, num_retries, 1, 1).reshape(-1, 4, 4)
    torch.testing.assert_close(poses, expected)
    # The independent physical model is two orthogonal prismatic translations.
    torch.testing.assert_close(poses[:, :dof, 3], seeds.reshape(-1, dof))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_two_dimensional_retry_configs_still_solve_batched_targets(dtype):
    chain = slide_chain(1, dtype)
    targets = chain.forward_kinematics(torch.tensor([[0.1], [0.2], [-0.3]], dtype=dtype))
    solver = pk.PseudoInverseIK(
        chain, retry_configs=torch.zeros(2, 1, dtype=dtype),
        max_iterations=15, early_stopping_any_converged=True, enforce_joint_limits=False,
    )
    assert solver.solve(targets).converged.all()


@pytest.mark.parametrize("shape", [(1,), (2, 2, 1, 1), (2, 3, 2), (3, 2)])
def test_invalid_retry_config_shape_reports_value_error(shape):
    chain = slide_chain(1, torch.float64)
    with pytest.raises(ValueError, match="initial_configs must have shape"):
        pk.PseudoInverseIK(chain, retry_configs=torch.zeros(shape, dtype=chain.dtype))
