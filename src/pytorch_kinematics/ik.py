from pytorch_kinematics.chain import SerialChain
from pytorch_kinematics.transforms import Transform3d
from pytorch_kinematics.transforms import rotation_conversions
from typing import NamedTuple, Union, Optional, Callable
import typing
import torch
import inspect
from matplotlib import pyplot as plt, cm as cm


class IKSolution:
    def __init__(self, dof, num_problems, num_retries, pos_tolerance, rot_tolerance, device="cpu"):
        self.device = device
        self.num_problems = num_problems
        self.num_retries = num_retries
        self.dof = dof
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance

        M = num_problems
        # N x DOF tensor of joint angles; if converged[i] is False, then solutions[i] is undefined
        self.solutions = torch.zeros((M, self.num_retries, self.dof), device=self.device)
        self.remaining = torch.ones(M, dtype=torch.bool, device=self.device)

        # M is the total number of problems
        # N is the total number of attempts
        # M x N tensor of position and rotation errors
        self.err_pos = torch.zeros((M, self.num_retries), device=self.device)
        self.err_rot = torch.zeros_like(self.err_pos)
        # M x N boolean values indicating whether the solution converged (a solution could be found)
        self.converged_pos = torch.zeros((M, self.num_retries), dtype=torch.bool, device=self.device)
        self.converged_rot = torch.zeros_like(self.converged_pos)
        self.converged = torch.zeros_like(self.converged_pos)

        # M whether any position and rotation converged for that problem
        self.converged_pos_any = torch.zeros_like(self.remaining)
        self.converged_rot_any = torch.zeros_like(self.remaining)
        self.converged_any = torch.zeros_like(self.remaining)

    def update_remaining_with_keep_mask(self, keep: torch.tensor):
        _r = self.remaining.clone()
        self.remaining[_r] = _r[_r] & keep
        this_masked = _r ^ self.remaining
        return this_masked

    def update(self, q: torch.tensor, err: torch.tensor,
               use_remaining=False, keep_mask=None):
        err = err.reshape(-1, self.num_retries, 6)
        err_pos = err[..., :3].norm(dim=-1)
        err_rot = err[..., 3:].norm(dim=-1)
        converged_pos = err_pos < self.pos_tolerance
        converged_rot = err_rot < self.rot_tolerance
        converged = converged_pos & converged_rot
        converged_any = converged.any(dim=1)

        if keep_mask is None:
            keep_mask = ~converged_any

        # stop considering problems where any converged
        qq = q.reshape(-1, self.num_retries, self.dof)

        if use_remaining:
            this_converged = self.remaining
            keep_mask[:] = False
        else:
            # those that have converged are no longer remaining
            this_converged = self.update_remaining_with_keep_mask(keep_mask)

        done = ~keep_mask
        self.solutions[this_converged] = qq[done]
        self.err_pos[this_converged] = err_pos[done]
        self.err_rot[this_converged] = err_rot[done]
        self.converged_pos[this_converged] = converged_pos[done]
        self.converged_rot[this_converged] = converged_rot[done]
        self.converged[this_converged] = converged[done]
        self.converged_any[this_converged] = converged_any[done]

        return converged_any


# helper config sampling method
def gaussian_around_config(config: torch.Tensor, std: float) -> Callable[[int], torch.Tensor]:
    def config_sampling_method(num_configs):
        return torch.randn(num_configs, config.shape[0], dtype=config.dtype, device=config.device) * std + config

    return config_sampling_method


class LineSearch:
    def do_line_search(self, chain, q, dq, target_pos, target_rot_rpy, initial_dx):
        raise NotImplementedError()


class BacktrackingLineSearch(LineSearch):
    def __init__(self, max_lr=2.0, decrease_factor=0.5, max_iterations=5, sufficient_decrease=0.01):
        self.initial_lr = max_lr
        self.decrease_factor = decrease_factor
        self.max_iterations = max_iterations
        self.sufficient_decrease = sufficient_decrease

    def do_line_search(self, chain, q, dq, target_pos, target_rot_rpy, initial_dx):
        N = target_pos.shape[0]
        NM = q.shape[0]
        M = NM // N
        lr = torch.ones(NM, device=q.device) * self.initial_lr
        err = initial_dx.squeeze().norm(dim=-1)
        for i in range(self.max_iterations):
            # try stepping with this learning rate
            q_new = q + lr.unsqueeze(1) * dq
            # evaluate the error
            m = chain.forward_kinematics(q_new).get_matrix()
            m = m.view(-1, M, 4, 4)
            dx, pos_diff, rot_diff = delta_pose(m, target_pos, target_rot_rpy)
            err_new = dx.squeeze().norm(dim=-1)
            # check if it's better
            improvement = err - err_new
            improved = improvement > self.sufficient_decrease
            # if it's better, we're done for those
            # TODO mask out the ones that are better and stop considering them
            # if it's not better, reduce the learning rate
            lr[~improved] *= self.decrease_factor

        improvement = improvement.reshape(-1, M)
        improvement = improvement.mean(dim=1)
        return lr, improvement


class InverseKinematics:
    """Jacobian follower based inverse kinematics solver"""

    def __init__(self, serial_chain: SerialChain,
                 pos_tolerance: float = 1e-3, rot_tolerance: float = 1e-2,
                 initial_configs: Optional[torch.Tensor] = None, num_retries: Optional[int] = None,
                 joint_limits: Optional[torch.Tensor] = None,
                 config_sampling_method: Union[str, Callable[[int], torch.Tensor]] = "uniform",
                 max_iterations: int = 50,
                 lr: float = 0.5, line_search: Optional[LineSearch] = None,
                 regularlization: float = 1e-9,
                 debug=False,
                 early_stopping_any_converged=False,
                 early_stopping_no_improvement=True,
                 optimizer_method: Union[str, typing.Type[torch.optim.Optimizer]] = "sgd"
                 ):
        """
        :param serial_chain:
        :param pos_tolerance:
        :param rot_tolerance:
        :param initial_configs:
        :param num_retries:
        :param joint_limits: (DOF, 2) tensor of joint limits (min, max) for each joint in radians
        :param config_sampling_method:
        """
        self.chain = serial_chain
        self.dtype = serial_chain.dtype
        self.device = serial_chain.device
        joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(joint_names)
        self.debug = debug
        self.early_stopping_any_converged = early_stopping_any_converged
        self.early_stopping_no_improvement = early_stopping_no_improvement
        self.past_improvement = None

        self.max_iterations = max_iterations
        self.lr = lr
        self.regularlization = regularlization
        self.optimizer_method = optimizer_method
        self.line_search = line_search

        self.err = None
        self.err_prev = None

        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance
        self.initial_config = initial_configs
        if initial_configs is None and num_retries is None:
            raise ValueError("either initial_configs or num_retries must be specified")

        # sample initial configs instead
        self.config_sampling_method = config_sampling_method
        self.joint_limits = joint_limits
        if initial_configs is None:
            self.initial_config = self.sample_configs(num_retries)
        else:
            if initial_configs.shape[1] != self.dof:
                raise ValueError("initial_configs must have shape (N, %d)" % self.dof)
        # could give a batch of initial configs
        self.num_retries = self.initial_config.shape[-2]

    def sample_configs(self, num_configs: int) -> torch.Tensor:
        if self.config_sampling_method == "uniform":
            # bound by joint_limits
            if self.joint_limits is None:
                raise ValueError("joint_limits must be specified if config_sampling_method is uniform")
            return torch.rand(num_configs, self.dof, device=self.device) * (
                    self.joint_limits[:, 1] - self.joint_limits[:, 0]) + self.joint_limits[:, 0]
        elif self.config_sampling_method == "gaussian":
            return torch.randn(num_configs, self.dof, device=self.device)
        elif callable(self.config_sampling_method):
            return self.config_sampling_method(num_configs)
        else:
            raise ValueError("invalid config_sampling_method %s" % self.config_sampling_method)

    def solve(self, target_poses: Transform3d) -> IKSolution:
        """
        Solve IK for the given target poses in robot frame
        :param target_poses: (N, 4, 4) tensor, goal pose in robot frame
        :return: IKSolution solutions
        """
        raise NotImplementedError()


def delta_pose(m: torch.tensor, target_pos, target_rot_rpy):
    """
    Determine the error in position and rotation between the given poses and the target poses

    :param m: (N x M x 4 x 4) tensor of homogenous transforms
    :param target_pos:
    :param target_rot_rpy:
    :return: (N*M, 6, 1) tensor of delta pose (dx, dy, dz, droll, dpitch, dyaw)
    """
    pos_diff = target_pos.unsqueeze(1) - m[:, :, :3, 3]
    pos_diff = pos_diff.view(-1, 3, 1)
    rot_diff = target_rot_rpy.unsqueeze(1) - rotation_conversions.matrix_to_euler_angles(m[:, :, :3, :3],
                                                                                         "XYZ")
    rot_diff = rot_diff.view(-1, 3, 1)

    dx = torch.cat((pos_diff, rot_diff), dim=1)
    return dx, pos_diff, rot_diff


def apply_mask(mask, *args):
    return [a[mask] for a in args]


class PseudoInverseIK(InverseKinematics):
    def solve(self, target_poses: Transform3d) -> IKSolution:
        target = target_poses.get_matrix()

        M = target.shape[0]

        target_pos = target[:, :3, 3]
        # jacobian gives angular rotation about x,y,z axis of the base frame
        # convert target rot to desired rotation about x,y,z
        target_rot_rpy = rotation_conversions.matrix_to_euler_angles(target[:, :3, :3], "XYZ")

        sol = IKSolution(self.dof, M, self.num_retries, self.pos_tolerance, self.rot_tolerance, device=self.device)

        q = self.initial_config
        if q.numel() == M * self.dof * self.num_retries:
            q = q.reshape(-1, self.dof)
        elif q.numel() == self.dof * self.num_retries:
            # repeat and manually flatten it
            q = self.initial_config.repeat(M, 1)
        else:
            raise ValueError(
                f"initial_config must have shape ({M}, {self.num_retries}, {self.dof}) or ({self.num_retries}, {self.dof})")
        # for logging, let's keep track of the joint angles at each iteration
        if self.debug:
            pos_errors = []
            rot_errors = []

        optimizer = None
        if inspect.isclass(self.optimizer_method) and issubclass(self.optimizer_method, torch.optim.Optimizer):
            q.requires_grad = True
            optimizer = torch.optim.Adam([q], lr=self.lr)
        for i in range(self.max_iterations):
            with torch.no_grad():
                # early termination if we're out of problems to solve
                if q.numel() == 0:
                    break
                # compute forward kinematics
                # fk = self.chain.forward_kinematics(q)
                # N x 6 x DOF
                J, m = self.chain.jacobian(q, ret_eef_pose=True)
                # unflatten to broadcast with goal
                m = m.view(-1, self.num_retries, 4, 4)
                dx, pos_diff, rot_diff = delta_pose(m, target_pos, target_rot_rpy)

                tmpA = J @ J.transpose(1, 2) + self.regularlization * torch.eye(6, device=self.device, dtype=self.dtype)
                A = torch.linalg.solve(tmpA, dx)
                dq = J.transpose(1, 2) @ A
                dq = dq.squeeze(2)

            improvement = None
            if optimizer is not None:
                q.grad = -dq
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    if self.line_search is not None:
                        lr, improvement = self.line_search.do_line_search(self.chain, q, dq, target_pos, target_rot_rpy,
                                                                          dx)
                        lr = lr.unsqueeze(1)
                    else:
                        lr = self.lr
                    q = q + lr * dq

            with torch.no_grad():
                self.err_prev = self.err
                self.err_all = dx.squeeze()
                self.err = self.err_all.norm(dim=-1)

                def apply_mask_to_all(mask):
                    nonlocal q, target_pos, target_rot_rpy, improvement
                    q, target_pos, target_rot_rpy = apply_mask(mask,
                                                               q.reshape(-1,
                                                                         self.num_retries,
                                                                         self.dof),
                                                               target_pos,
                                                               target_rot_rpy,
                                                               )
                    q = q.reshape(-1, self.dof)
                    if improvement is not None:
                        improvement = improvement[mask]
                    if self.err_prev is not None:
                        self.err_prev = self.err_prev.reshape(-1, self.num_retries)
                        self.err_prev = self.err_prev[mask]
                        self.err_prev = self.err_prev.reshape(-1)
                        if self.past_improvement is not None:
                            self.past_improvement = self.past_improvement[mask]
                    self.err = self.err.reshape(-1, self.num_retries)
                    self.err = self.err[mask]
                    self.err = self.err.reshape(-1)
                    self.err_all = self.err_all.reshape(-1, self.num_retries, 6)
                    self.err_all = self.err_all[mask]
                    self.err_all = self.err_all.reshape(-1, 6)

                if improvement is None and self.err_prev is not None:
                    improvement = self.err_prev - self.err
                    improvement = improvement.reshape(-1, self.num_retries)
                    improvement = improvement.mean(dim=1)

                if self.early_stopping_any_converged:
                    # stop considering problems where any converged
                    converged_any = sol.update(q, self.err_all, use_remaining=False)
                    apply_mask_to_all(~converged_any)

                if self.early_stopping_no_improvement:
                    if self.past_improvement is not None:
                        # average improvement of this and before
                        avg_improvement = (improvement + self.past_improvement) / 2
                        enough_improvement = avg_improvement > 0.0
                        # stop working on those that we can't improve
                        sol.update(q, self.err_all, use_remaining=False, keep_mask=enough_improvement)
                        apply_mask_to_all(enough_improvement)
                    self.past_improvement = improvement

                if self.debug:
                    pos_errors.append(pos_diff.reshape(-1, 3).norm(dim=1))
                    rot_errors.append(rot_diff.reshape(-1, 3).norm(dim=1))

        if self.debug:
            # errors
            fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
            pos_e = torch.stack(pos_errors, dim=0).cpu()
            rot_e = torch.stack(rot_errors, dim=0).cpu()
            ax[0].set_ylim(0, 1.)
            ax[1].set_ylim(0, rot_e.max().item() * 1.1)
            ax[0].set_xlim(0, self.max_iterations - 1)
            ax[1].set_xlim(0, self.max_iterations - 1)
            # draw at most 50 lines
            draw_max = min(50, pos_e.shape[1])
            for b in range(draw_max):
                c = (b + 1) / draw_max
                ax[0].plot(pos_e[:, b], c=cm.GnBu(c))
                ax[1].plot(rot_e[:, b], c=cm.GnBu(c))
            # label these axis
            ax[0].set_ylabel("position error")
            ax[1].set_xlabel("iteration")
            ax[1].set_ylabel("rotation error")
            plt.show()

        if i == self.max_iterations - 1:
            sol.update(q, self.err_all, use_remaining=True)
        return sol
