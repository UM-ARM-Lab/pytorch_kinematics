import os

import torch
from torch.distributions.uniform import Uniform

import pytorch_kinematics as pk
from pytorch_kinematics import cfg


def ik(chain: pk.Chain, target_positions, n_organisms_per_problem=100, max_steps=1000, solved_threshold=1e-3,
       verbose: int = 0):
    n_ik_problems = target_positions.shape[0]

    population_size = n_ik_problems * n_organisms_per_problem

    target_positions_repeated = target_positions.repeat_interleave(n_organisms_per_problem, dim=0)
    target_positions_repeated = target_positions_repeated.to(device=chain.device, dtype=chain.dtype)

    n_joints = len(chain.get_joint_parameter_names())
    th = torch.rand(population_size, n_joints, device=chain.device, dtype=chain.dtype, requires_grad=True)
    opt = torch.optim.RMSprop(params=[th], lr=0.03)
    for i in range(max_steps):
        opt.zero_grad()

        ee_transform = chain.forward_kinematics(th).get_matrix()

        population_loss = (ee_transform[..., :3, 3] - target_positions_repeated).norm(dim=1)

        population_loss_matrix = population_loss.reshape(n_ik_problems, n_organisms_per_problem)
        min_loss_per_ik_problem, min_loss_per_ik_problem_idx = population_loss_matrix.min(dim=1)
        population_solved = population_loss < solved_threshold
        solved = min_loss_per_ik_problem < solved_threshold
        if verbose >= 1:
            percent_solved = torch.count_nonzero(solved).detach().cpu().numpy() / n_ik_problems
            print(f"{i=} {percent_solved:.1%}")
        if torch.all(solved):
            th_matrix = th.reshape(n_ik_problems, n_organisms_per_problem, -1)
            batch_indices = torch.arange(n_ik_problems)
            th_solutions = th_matrix[batch_indices, min_loss_per_ik_problem_idx]
            return th_solutions

        loss_masked = population_loss * ~population_solved  # if we've found a solution, set loss to zero so we don't make it worse
        loss_masked = loss_masked.mean(dim=0)

        loss_masked.backward()

        opt.step()

    print("IK Failed!")
    return None


def solve_random_position_ik():
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    n_ik_problems = 1000
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(cfg.TEST_DIR, "kuka_iiwa.urdf")).read(),
                                            "lbr_iiwa_link_7")
    chain = chain.to(dtype=dtype, device=d)

    target_positions_distribution = Uniform(torch.tensor([0.4, 0.4, 0.7]), torch.tensor([0.45, 0.45, 0.75]))
    target_positions = target_positions_distribution.sample([n_ik_problems])
    target_positions = target_positions.to(dtype=dtype, device=d)

    from time import perf_counter
    t0 = perf_counter()
    joint_positions = ik(chain, target_positions)
    print(f"{perf_counter() - t0:.4}")


def main():
    for _ in range(3):
        solve_random_position_ik()


if __name__ == '__main__':
    main()
