import os
import math

import torch
import pytorch_kinematics as pk
import pytorch_seed

try:
    import pybullet_data
    URDF_PATH = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
except ImportError:
    URDF_PATH = os.path.join(os.path.dirname(__file__), "kuka_iiwa.urdf")


def test_ik_solutions_within_joint_limits():
    pytorch_seed.seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chain = pk.build_serial_chain_from_urdf(open(URDF_PATH).read(), "lbr_iiwa_link_7")
    chain = chain.to(device=device)

    lim = torch.tensor(chain.get_joint_limits(), device=device)
    M = 100
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]
    goal_tf = chain.forward_kinematics(goal_q)

    num_retries = 10
    ik = pk.PseudoInverseIK(chain, max_iterations=50, num_retries=num_retries,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            debug=False, lr=0.2,
                            enforce_joint_limits=True)

    sol = ik.solve(goal_tf)
    print("IK solved %d / %d goals" % (sol.converged_any.sum(), M))

    # All solutions must respect joint limits
    q_all = sol.solutions
    eps = 1e-6
    low = chain.low
    high = chain.high
    assert (q_all >= low - eps).all(), \
        f"Solutions below lower limits: max violation = {(low - q_all).clamp(min=0).max()}"
    assert (q_all <= high + eps).all(), \
        f"Solutions above upper limits: max violation = {(q_all - high).clamp(min=0).max()}"

    # Convergence rate should be high since these goals are reachable
    converged_rate = sol.converged_any.sum().item() / M
    print(f"Convergence rate: {converged_rate:.1%}")
    assert converged_rate > 0.8, f"Convergence rate too low: {converged_rate:.1%}"


def test_enforce_joint_limits_flag():
    """Test that enforce_joint_limits=False disables limit enforcement."""
    pytorch_seed.seed(2)
    chain = pk.build_serial_chain_from_urdf(open(URDF_PATH).read(), "lbr_iiwa_link_7")

    lim = torch.tensor(chain.get_joint_limits())
    M = 50
    goal_q = torch.rand(M, lim.shape[1]) * (lim[1] - lim[0]) + lim[0]
    goal_tf = chain.forward_kinematics(goal_q)

    ik = pk.PseudoInverseIK(chain, max_iterations=30, num_retries=10,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            debug=False, lr=0.2,
                            enforce_joint_limits=False)

    sol = ik.solve(goal_tf)
    # With enforce_joint_limits=False, some solutions may exceed limits
    q_all = sol.solutions
    has_violation = ((q_all < chain.low).any() | (q_all > chain.high).any()).item()
    # We don't assert violations must exist (they might not), but just check it ran without error
    print(f"enforce_joint_limits=False: violations present = {has_violation}")


def test_ik_mixed_continuous_and_bounded_joints():
    """Limit enforcement must preserve solved poses with continuous joints."""
    urdf = """<robot name="turntable_slide">
        <link name="base"/><link name="turntable"/><link name="tool"/>
        <joint name="spin" type="continuous">
            <parent link="base"/><child link="turntable"/><axis xyz="0 0 1"/>
        </joint>
        <joint name="slide" type="prismatic">
            <parent link="turntable"/><child link="tool"/><axis xyz="1 0 0"/>
            <limit lower="0" upper="1" effort="10" velocity="1"/>
        </joint>
    </robot>"""
    chain = pk.build_serial_chain_from_urdf(urdf, "tool").to(dtype=torch.float64)
    goal_q = torch.tensor([[0.5, 0.25]], dtype=chain.dtype)
    goal_tf = chain.forward_kinematics(goal_q)
    retry_configs = goal_q.repeat(2, 1)
    retry_configs[1, 0] += 4 * math.pi
    ik = pk.PseudoInverseIK(chain, retry_configs=retry_configs,
                            early_stopping_any_converged=True)

    sol = ik.solve(goal_tf)

    assert torch.isfinite(sol.solutions).all()
    assert sol.converged.all()
    assert (sol.solutions >= chain.low).all()
    assert (sol.solutions <= chain.high).all()
    torch.testing.assert_close(chain.forward_kinematics(sol.solutions[0]).get_matrix(),
                               goal_tf.get_matrix().expand(2, -1, -1))


def test_revolute_wrapping_with_one_sided_limits():
    """Periodic wrapping respects finite bounds without clamping the pose."""
    urdf = """<robot name="partial_limits">
        <link name="base"/><link name="lower"/><link name="upper"/>
        <link name="spin"/><link name="tool"/>
        <joint name="lower_only" type="revolute">
            <parent link="base"/><child link="lower"/><axis xyz="0 0 1"/>
            <limit lower="0" upper="inf" effort="10" velocity="1"/>
        </joint>
        <joint name="upper_only" type="revolute">
            <parent link="lower"/><child link="upper"/><axis xyz="0 1 0"/>
            <limit lower="-inf" upper="0" effort="10" velocity="1"/>
        </joint>
        <joint name="continuous" type="continuous">
            <parent link="upper"/><child link="spin"/><axis xyz="1 0 0"/>
        </joint>
        <joint name="slide" type="prismatic">
            <parent link="spin"/><child link="tool"/><axis xyz="1 0 0"/>
            <limit lower="0" upper="1" effort="10" velocity="1"/>
        </joint>
    </robot>"""
    chain = pk.build_serial_chain_from_urdf(urdf, "tool").to(dtype=torch.float64)
    q = torch.tensor([[-2 * math.pi + 0.3, 4 * math.pi - 0.4, 6 * math.pi + 0.2, 0.25],
                      [4 * math.pi + 0.3, -2 * math.pi - 0.4, -6 * math.pi - 0.2, 0.4]],
                     dtype=chain.dtype)
    ik = pk.PseudoInverseIK(chain, retry_configs=q)

    wrapped = ik._wrap_revolute_joints(q)

    expected = q.clone()
    expected[0, :2] = torch.tensor([0.3, -0.4], dtype=chain.dtype)
    torch.testing.assert_close(wrapped, expected)
    torch.testing.assert_close(chain.forward_kinematics(wrapped).get_matrix(),
                               chain.forward_kinematics(q).get_matrix())
    assert (wrapped >= chain.low).all()
    assert (wrapped <= chain.high).all()


if __name__ == "__main__":
    test_ik_solutions_within_joint_limits()
    test_enforce_joint_limits_flag()
