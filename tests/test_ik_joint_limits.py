import os

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


if __name__ == "__main__":
    test_ik_solutions_within_joint_limits()
    test_enforce_joint_limits_flag()
