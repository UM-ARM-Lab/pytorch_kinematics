"""Benchmark IK performance: convergence rate, timing, and joint limit compliance."""
import os
import sys
from timeit import default_timer as timer

import torch
import pytorch_kinematics as pk
import pytorch_seed


def benchmark(device="cpu"):
    pytorch_seed.seed(2)

    # Try pybullet_data kuka first, fall back to local test URDF
    try:
        import pybullet_data
        full_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
    except ImportError:
        full_urdf = os.path.join(os.path.dirname(__file__), "kuka_iiwa.urdf")
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    chain = chain.to(device=device)

    lim = torch.tensor(chain.get_joint_limits(), device=device)
    M = 500
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]
    goal_tf = chain.forward_kinematics(goal_q)

    num_retries = 10
    ik = pk.PseudoInverseIK(chain, max_iterations=30, num_retries=num_retries,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            debug=False,
                            lr=0.2)

    # Warmup
    _ = ik.solve(goal_tf)

    # Timed run
    N_RUNS = 3
    times = []
    for _ in range(N_RUNS):
        pytorch_seed.seed(2)
        ik.initial_config = ik.sample_configs(num_retries)
        t0 = timer()
        sol = ik.solve(goal_tf)
        t1 = timer()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)

    q_all = sol.solutions
    low, high = chain.low, chain.high
    below = (low - q_all).clamp(min=0).max().item()
    above = (q_all - high).clamp(min=0).max().item()

    print(f"Goals solved:       {sol.converged_any.sum().item()} / {M}")
    print(f"Convergence rate:   {sol.converged_any.sum().item() / M:.1%}")
    print(f"Total converged:    {sol.converged.sum().item()} / {sol.converged.numel()}")
    print(f"Iterations:         {sol.iterations}")
    print(f"Avg time ({N_RUNS} runs): {avg_time:.4f}s")
    print(f"Max violation below: {below:.6f}")
    print(f"Max violation above: {above:.6f}")


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"
    print(f"=== Device: {device} ===")
    benchmark(device=device)
