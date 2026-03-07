from pytorch_kinematics.chain import SerialChain
from pytorch_kinematics.transforms import Transform3d
from pytorch_kinematics.transforms import rotation_conversions
from typing import NamedTuple, Union, Optional, Callable
import typing
import math
import torch
import inspect
from matplotlib import pyplot as plt, cm as cm

# Check if torch.compile is available (PyTorch 2.0+)
_TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and torch.__version__ >= '2.0'


def _compute_dq_kernel(J: torch.Tensor, dx: torch.Tensor, reg_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute joint velocity using damped least squares.

    This function is designed to be compatible with torch.compile for JIT optimization.

    Args:
        J: Jacobian matrix (N, 6, DOF)
        dx: Pose error (N, 6, 1)
        reg_matrix: Regularization matrix (6, 6)

    Returns:
        dq: Joint velocity (N, DOF, 1)
    """
    # JJ^T + lambda^2*I
    tmpA = J @ J.transpose(1, 2) + reg_matrix
    # Solve (JJ^T + lambda^2*I) A = dx
    A = torch.linalg.solve(tmpA, dx)
    # dq = J^T @ A
    return J.transpose(1, 2) @ A


def _ik_step_kernel(m_flat: torch.Tensor, target_pos: torch.Tensor, target_wxyz: torch.Tensor,
                    J: torch.Tensor, reg_matrix: torch.Tensor, num_retries: int):
    """
    Fused IK step: delta_pose + damped least squares. Compatible with torch.compile(fullgraph=True).

    Args:
        m_flat: End-effector poses (N*M, 4, 4) where N=num_problems, M=num_retries
        target_pos: Target positions (N, 3)
        target_wxyz: Target orientations as wxyz quaternions (N, 4)
        J: Jacobian matrix (N*M, 6, DOF)
        reg_matrix: Regularization matrix (6, 6)
        num_retries: Number of retries per problem (M)

    Returns:
        dq: Joint velocity (N*M, DOF, 1)
        dx: Pose error (N*M, 6, 1)
    """
    NM = m_flat.shape[0]
    M = num_retries
    N = NM // M
    m = m_flat.view(N, M, 4, 4)

    # delta_pose
    pos_diff = (target_pos.unsqueeze(1) - m[:, :, :3, 3]).view(-1, 3, 1)
    cur_wxyz = rotation_conversions.matrix_to_quaternion(m[:, :, :3, :3])
    diff_wxyz = rotation_conversions.quaternion_multiply(
        target_wxyz.unsqueeze(1),
        rotation_conversions.quaternion_invert(cur_wxyz))
    diff_axis_angle = rotation_conversions.quaternion_to_axis_angle(diff_wxyz)
    rot_diff = diff_axis_angle.view(-1, 3, 1)
    dx = torch.cat((pos_diff, rot_diff), dim=1)

    # DLS
    tmpA = J @ J.transpose(1, 2) + reg_matrix
    A = torch.linalg.solve(tmpA, dx)
    dq = J.transpose(1, 2) @ A
    return dq, dx


class IKSolution:
    def __init__(self, dof, num_problems, num_retries, pos_tolerance, rot_tolerance, device="cpu", dtype=None):
        self.iterations = 0
        self.device = device
        self.num_problems = num_problems
        self.num_retries = num_retries
        self.dof = dof
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance

        M = num_problems
        # N x DOF tensor of joint angles; if converged[i] is False, then solutions[i] is undefined
        self.solutions = torch.zeros((M, self.num_retries, self.dof), device=self.device, dtype=dtype)
        self.remaining = torch.ones(M, dtype=torch.bool, device=self.device)

        # M is the total number of problems
        # N is the total number of attempts
        # M x N tensor of position and rotation errors
        self.err_pos = torch.zeros((M, self.num_retries), device=self.device, dtype=dtype)
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
        self.remaining = self.remaining & keep
        return self.remaining

    def update(self, q: torch.tensor, err: torch.tensor, use_keep_mask=True, keep_mask=None):
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

        if use_keep_mask:
            # those that have converged are no longer remaining
            self.update_remaining_with_keep_mask(keep_mask)

        # sticky convergence: only overwrite retries that haven't already converged
        # this prevents overwriting a good solution if the solver overshoots on the next step
        already_converged = self.converged
        update_mask = ~already_converged
        self.solutions[update_mask] = qq[update_mask]
        self.err_pos[update_mask] = err_pos[update_mask]
        self.err_rot[update_mask] = err_rot[update_mask]
        self.converged_pos = self.converged_pos | converged_pos
        self.converged_rot = self.converged_rot | converged_rot
        self.converged = self.converged | converged
        self.converged_any = self.converged_any | converged_any

        return converged_any


# helper config sampling method
def gaussian_around_config(config: torch.Tensor, std: float) -> Callable[[int], torch.Tensor]:
    def config_sampling_method(num_configs):
        return torch.randn(num_configs, config.shape[0], dtype=config.dtype, device=config.device) * std + config

    return config_sampling_method


class LineSearch:
    def do_line_search(self, chain, q, dq, target_pos, target_wxyz, initial_dx, problem_remaining=None):
        raise NotImplementedError()


class BacktrackingLineSearch(LineSearch):
    def __init__(self, max_lr=1.0, decrease_factor=0.5, max_iterations=5, sufficient_decrease=0.01):
        self.initial_lr = max_lr
        self.decrease_factor = decrease_factor
        self.max_iterations = max_iterations
        self.sufficient_decrease = sufficient_decrease

    def do_line_search(self, chain, q, dq, target_pos, target_wxyz, initial_dx, problem_remaining=None):
        N = target_pos.shape[0]
        NM = q.shape[0]
        M = NM // N
        lr = torch.ones(NM, device=q.device) * self.initial_lr
        err = initial_dx.squeeze().norm(dim=-1)
        if problem_remaining is None:
            problem_remaining = torch.ones(N, dtype=torch.bool, device=q.device)
        remaining = torch.ones((N, M), dtype=torch.bool, device=q.device)
        # don't care about the ones that are no longer remaining
        remaining[~problem_remaining] = False
        remaining = remaining.reshape(-1)
        for i in range(self.max_iterations):
            if not remaining.any():
                break
            # try stepping with this learning rate
            q_new = q + lr.unsqueeze(1) * dq
            # evaluate the error
            m = chain.forward_kinematics(q_new).get_matrix()
            m = m.view(-1, M, 4, 4)
            dx, pos_diff, rot_diff = delta_pose(m, target_pos, target_wxyz)
            err_new = dx.squeeze().norm(dim=-1)
            # check if it's better
            improvement = err - err_new
            improved = improvement > self.sufficient_decrease
            # if it's better, we're done for those
            # if it's not better, reduce the learning rate
            lr[~improved] *= self.decrease_factor
            remaining = remaining & ~improved

        improvement = improvement.reshape(-1, M)
        improvement = improvement.mean(dim=1)
        return lr, improvement


class InverseKinematics:
    """Jacobian follower based inverse kinematics solver"""

    def __init__(self, serial_chain: SerialChain,
                 pos_tolerance: float = 1e-3, rot_tolerance: float = 1e-2,
                 retry_configs: Optional[torch.Tensor] = None, num_retries: Optional[int] = None,
                 joint_limits: Optional[torch.Tensor] = None,
                 config_sampling_method: Union[str, Callable[[int], torch.Tensor]] = "uniform",
                 max_iterations: int = 50,
                 lr: float = 0.2, line_search: Optional[LineSearch] = None,
                 regularlization: float = 1e-9,
                 debug=False,
                 early_stopping_any_converged=False,
                 early_stopping_no_improvement="any", early_stopping_no_improvement_patience=2,
                 optimizer_method: Union[str, typing.Type[torch.optim.Optimizer]] = "sgd",
                 enforce_joint_limits: bool = True,
                 num_limit_refinement_iterations: int = 10
                 ):
        """
        :param serial_chain:
        :param pos_tolerance: position tolerance in meters
        :param rot_tolerance: rotation tolerance in radians
        :param retry_configs: (M, DOF) tensor of initial configs to try for each problem; leave as None to sample
        :param num_retries: number, M, of random initial configs to try for that problem; implemented with batching
        :param joint_limits: (DOF, 2) tensor of joint limits (min, max) for each joint in radians
        :param config_sampling_method: either "uniform" or "gaussian" or a function that takes in the number of configs
        :param max_iterations: maximum number of iterations to run
        :param lr: learning rate
        :param line_search: LineSearch object to use for line search
        :param regularlization: regularization term to add to the Jacobian
        :param debug: whether to print debug information
        :param early_stopping_any_converged: whether to stop when any of the retries for a problem converged
        :param early_stopping_no_improvement: {None, "all", "any", ratio} whether to stop when no improvement is made
        (consecutive iterations no improvement in minimum error - number of consecutive iterations is the patience).
        None means no early stopping from this, "all" means stop when all retries for that problem makes no improvement,
        "any" means stop when any of the retries for that problem makes no improvement, and ratio means stop when
        the ratio (between 0 and 1) of the number of retries that is making improvement falls below the ratio.
        So "all" is equivalent to ratio=0.999, and "any" is equivalent to ratio=0.001
        :param early_stopping_no_improvement_patience: number of consecutive iterations with no improvement before
        considering it no improvement
        :param optimizer_method: either a string or a torch.optim.Optimizer class
        :param enforce_joint_limits: whether to enforce joint limits on the solution. Uses the chain's joint limits
        (chain.low/chain.high). After solving, revolute joints are wrapped by multiples of 2*pi, then any remaining
        violations are resolved via clamped refinement iterations. Set to False to disable.
        :param num_limit_refinement_iterations: number of clamped IK refinement iterations for enforcing joint limits
        """
        self.chain = serial_chain
        self.dtype = serial_chain.dtype
        self.device = serial_chain.device
        joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(joint_names)
        self.debug = debug
        self.enforce_joint_limits = enforce_joint_limits
        self.num_limit_refinement_iterations = num_limit_refinement_iterations
        # precompute which joints are revolute (for 2*pi wrapping)
        joints = self.chain.get_joints(exclude_fixed=True)
        self._revolute_mask = torch.tensor([j.joint_type == 'revolute' for j in joints],
                                           dtype=torch.bool, device=self.device)
        self.early_stopping_any_converged = early_stopping_any_converged
        self.early_stopping_no_improvement = early_stopping_no_improvement
        self.early_stopping_no_improvement_patience = early_stopping_no_improvement_patience

        self.max_iterations = max_iterations
        self.lr = lr
        self.regularlization = regularlization
        self.optimizer_method = optimizer_method
        self.line_search = line_search

        self.err = None
        self.err_all = None
        self.err_min = None
        self.no_improve_counter = None

        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance
        self.initial_config = retry_configs
        if retry_configs is None and num_retries is None:
            raise ValueError("either initial_configs or num_retries must be specified")

        # sample initial configs instead
        self.config_sampling_method = config_sampling_method
        self.joint_limits = joint_limits
        if retry_configs is None:
            self.initial_config = self.sample_configs(num_retries)
        else:
            if retry_configs.shape[1] != self.dof:
                raise ValueError("initial_configs must have shape (N, %d)" % self.dof)
        # could give a batch of initial configs
        self.num_retries = self.initial_config.shape[-2]

    def clear(self):
        self.err = None
        self.err_all = None
        self.err_min = None
        self.no_improve_counter = None

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


def delta_pose(m: torch.tensor, target_pos, target_wxyz, out: torch.Tensor = None):
    """
    Determine the error in position and rotation between the given poses and the target poses

    :param m: (N x M x 4 x 4) tensor of homogenous transforms
    :param target_pos:
    :param target_wxyz: target orientation represented in unit quaternion
    :param out: optional pre-allocated output buffer (N*M, 6, 1) to reduce memory allocation
    :return: (N*M, 6, 1) tensor of delta pose (dx, dy, dz, droll, dpitch, dyaw)
    """
    pos_diff = target_pos.unsqueeze(1) - m[:, :, :3, 3]
    pos_diff = pos_diff.view(-1, 3, 1)
    cur_wxyz = rotation_conversions.matrix_to_quaternion(m[:, :, :3, :3])

    # quaternion that rotates from the current orientation to the desired orientation
    # inverse for unit quaternion is the conjugate
    diff_wxyz = rotation_conversions.quaternion_multiply(target_wxyz.unsqueeze(1),
                                                         rotation_conversions.quaternion_invert(cur_wxyz))
    # angular velocity vector needed to correct the orientation
    # if time is considered, should divide by \delta t, but doing it iteratively we can choose delta t to be 1
    diff_axis_angle = rotation_conversions.quaternion_to_axis_angle(diff_wxyz)

    rot_diff = diff_axis_angle.view(-1, 3, 1)

    # Use pre-allocated buffer if provided
    if out is not None:
        out[:, :3] = pos_diff
        out[:, 3:] = rot_diff
        dx = out
    else:
        dx = torch.cat((pos_diff, rot_diff), dim=1)
    return dx, pos_diff, rot_diff


def apply_mask(mask, *args):
    return [a[mask] for a in args]


class PseudoInverseIK(InverseKinematics):
    def __init__(self, *args, use_compile: bool = False, **kwargs):
        """
        Initialize PseudoInverseIK solver.

        Args:
            *args: Arguments passed to InverseKinematics.
            use_compile: If True and PyTorch 2.0+ is available, use torch.compile
                for JIT compilation of the compute_dq kernel. This can provide
                performance improvements after a warmup period. Default: False.
            **kwargs: Keyword arguments passed to InverseKinematics.
        """
        super().__init__(*args, **kwargs)
        # Pre-compute regularization matrix once
        self._reg_matrix = self.regularlization * torch.eye(6, device=self.device, dtype=self.dtype)

        # Set up compute kernels (potentially compiled)
        self._use_compile = use_compile and _TORCH_COMPILE_AVAILABLE
        if self._use_compile:
            self._compute_dq_fn = torch.compile(_compute_dq_kernel)
            self._ik_step_fn = torch.compile(_ik_step_kernel)
            self._jacobian_fn = torch.compile(self.chain.jacobian_tensor)
            self._fk_fn = torch.compile(self.chain.forward_kinematics_tensor)
        else:
            self._compute_dq_fn = _compute_dq_kernel
            self._ik_step_fn = _ik_step_kernel
            self._jacobian_fn = self.chain.jacobian_tensor
            self._fk_fn = self.chain.forward_kinematics_tensor
        self._eef_frame_idx = self.chain._serial_eef_frame_idx

    def compute_dq(self, J, dx):
        """Compute joint velocity using damped least squares."""
        return self._compute_dq_fn(J, dx, self._reg_matrix)

    def solve(self, target_poses: Transform3d) -> IKSolution:
        self.clear()

        target = target_poses.get_matrix()

        M = target.shape[0]

        target_pos = target[:, :3, 3]
        # jacobian gives angular rotation about x,y,z axis of the base frame
        # convert target rot to desired rotation about x,y,z
        target_wxyz = rotation_conversions.matrix_to_quaternion(target[:, :3, :3])

        sol = IKSolution(self.dof, M, self.num_retries, self.pos_tolerance, self.rot_tolerance,
                         device=self.device, dtype=self.dtype)

        q = self.initial_config
        if q.numel() == M * self.dof * self.num_retries:
            q = q.reshape(-1, self.dof)
        elif q.numel() == self.dof * self.num_retries:
            # repeat and manually flatten it
            q = self.initial_config.repeat(M, 1)
        elif q.numel() == self.dof:
            q = q.unsqueeze(0).repeat(M * self.num_retries, 1)
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
                if not sol.remaining.any():
                    break
                sol.iterations += 1
                # compute Jacobian, end-effector pose, and pose error in one FK pass
                J, m = self._jacobian_fn(q, True)
                dq, dx = self._ik_step_fn(m, target_pos, target_wxyz, J,
                                          self._reg_matrix, self.num_retries)
                dq = dq.squeeze(2)

                # convergence check using error at current q (before stepping)
                self.err_all = dx.squeeze(2)
                self.err = self.err_all.reshape(-1, self.num_retries, 6).norm(dim=-1)
                sol.update(q, self.err_all, use_keep_mask=self.early_stopping_any_converged)

                if self.early_stopping_no_improvement is not None:
                    if self.no_improve_counter is None:
                        self.no_improve_counter = torch.zeros_like(self.err)
                        self.err_min = self.err.clone()
                    else:
                        improved = self.err < self.err_min
                        self.err_min[improved] = self.err[improved]

                        self.no_improve_counter[improved] = 0
                        self.no_improve_counter[~improved] += 1

                        # those that haven't improved
                        could_improve = self.no_improve_counter <= self.early_stopping_no_improvement_patience
                        # consider problems, and only throw out those whose all retries cannot be improved
                        could_improve = could_improve.reshape(-1, self.num_retries)
                        if self.early_stopping_no_improvement == "all":
                            could_improve = could_improve.all(dim=1)
                        elif self.early_stopping_no_improvement == "any":
                            could_improve = could_improve.any(dim=1)
                        elif isinstance(self.early_stopping_no_improvement, float):
                            ratio_improved = could_improve.sum(dim=1) / self.num_retries
                            could_improve = ratio_improved > self.early_stopping_no_improvement
                        sol.update_remaining_with_keep_mask(could_improve)

                if self.debug:
                    pos_err = dx[:, :3, 0].reshape(-1, 3).norm(dim=1)
                    rot_err = dx[:, 3:, 0].reshape(-1, 3).norm(dim=1)
                    pos_errors.append(pos_err)
                    rot_errors.append(rot_err)

            improvement = None
            if optimizer is not None:
                q.grad = -dq
                optimizer.step()
                optimizer.zero_grad()
            else:
                with torch.no_grad():
                    if self.line_search is not None:
                        lr, improvement = self.line_search.do_line_search(self.chain, q, dq, target_pos, target_wxyz,
                                                                          dx, problem_remaining=sol.remaining)
                        lr = lr.unsqueeze(1)
                    else:
                        lr = self.lr
                    # Use in-place addition to reduce memory allocation
                    q = q.add(dq, alpha=lr) if isinstance(lr, float) else q.add(lr * dq)

        # Final convergence check for the last stepped q
        with torch.no_grad():
            all_tf_final = self._fk_fn(q)
            m_final = all_tf_final[self._eef_frame_idx].view(-1, self.num_retries, 4, 4)
            dx_final, _, _ = delta_pose(m_final, target_pos, target_wxyz)
            self.err_all = dx_final.squeeze()
            self.err = self.err_all.reshape(-1, self.num_retries, 6).norm(dim=-1)
            sol.update(q, self.err_all, use_keep_mask=self.early_stopping_any_converged)

        if self.debug:
            # errors
            fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
            pos_e = torch.stack(pos_errors, dim=0).cpu()
            rot_e = torch.stack(rot_errors, dim=0).cpu()
            ax[0].set_ylim(0, 1.)
            # ignore nan
            ignore = torch.isnan(rot_e)
            axis_max = rot_e[~ignore].max().item()
            ax[1].set_ylim(0, axis_max * 1.1)
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

        if self.enforce_joint_limits and self._has_finite_limits():
            self._enforce_limits(sol, target_pos, target_wxyz)

        return sol

    def _has_finite_limits(self):
        """Check if the chain has any finite joint limits."""
        return torch.isfinite(self.chain.low).any() or torch.isfinite(self.chain.high).any()

    def _wrap_revolute_joints(self, q):
        """Wrap revolute joint values by multiples of 2*pi to bring them closer to the valid range.

        For revolute joints, q and q + 2*n*pi produce identical FK, so we can freely shift
        by multiples of 2*pi without affecting the solution. Prismatic joints are left unchanged.
        """
        low = self.chain.low
        high = self.chain.high
        center = (low + high) / 2
        # compute the number of 2*pi shifts needed to bring q closest to the center of the limits
        n = torch.round((center - q) / (2 * math.pi))
        # only apply to revolute joints
        q_wrapped = q.clone()
        q_wrapped[..., self._revolute_mask] = q[..., self._revolute_mask] + n[..., self._revolute_mask] * 2 * math.pi
        return q_wrapped

    def _enforce_limits(self, sol, target_pos, target_wxyz):
        """Enforce joint limits on IK solutions via wrapping and clamped refinement.

        1. Wrap revolute joints by 2*pi to bring them into the valid range (preserves FK exactly).
        2. Clamp any remaining violations and run refinement iterations to recover convergence.
        3. Update the solution with the refined joint values.
        """
        q = sol.solutions.reshape(-1, self.dof)

        # Step 1: wrap revolute joints by 2*pi
        q = self._wrap_revolute_joints(q)

        # Step 2: clamp to limits
        q = torch.clamp(q, self.chain.low, self.chain.high)

        # Step 3: clamped refinement iterations to recover from any error introduced by clamping
        for _ in range(self.num_limit_refinement_iterations):
            J, m_flat = self._jacobian_fn(q, True)
            m = m_flat.view(-1, self.num_retries, 4, 4)
            dx, _, _ = delta_pose(m, target_pos, target_wxyz)
            dq = self.compute_dq(J, dx).squeeze(2)
            q = q + self.lr * dq
            q = torch.clamp(q, self.chain.low, self.chain.high)

        # Reset convergence so all retries are re-evaluated with the limit-enforced solutions
        sol.converged[:] = False
        sol.converged_any[:] = False
        sol.converged_pos[:] = False
        sol.converged_rot[:] = False

        # Recompute error and update solution
        all_tf_new = self._fk_fn(q)
        m_new = all_tf_new[self._eef_frame_idx]  # (N*M, 4, 4)
        m_new = m_new.view(-1, self.num_retries, 4, 4)
        dx_new, _, _ = delta_pose(m_new, target_pos, target_wxyz)
        sol.update(q, dx_new.squeeze(), use_keep_mask=False)


class PseudoInverseIKWithSVD(PseudoInverseIK):
    # generally slower, but allows for selective damping if needed
    def compute_dq(self, J, dx):
        # reg = self.regularlization * torch.eye(6, device=self.device, dtype=self.dtype)
        U, D, Vh = torch.linalg.svd(J)
        m = D.shape[1]

        # tmpA = U @ (D @ D.transpose(1, 2) + reg) @ U.transpose(1, 2)
        # singular_val = torch.diagonal(D)

        denom = D ** 2 + self.regularlization
        prod = D / denom
        # J^T (JJ^T + lambda^2I)^-1 = V @ (D @ D^T + lambda^2I)^-1 @ U^T = sum_i (d_i / (d_i^2 + lambda^2) v_i @ u_i^T)
        # should be equivalent to damped least squares
        inverted = torch.diag_embed(prod)

        # drop columns from V
        Vh = Vh[:, :m, :]
        total = Vh.transpose(1, 2) @ inverted @ U.transpose(1, 2)

        # dq = J^T (JJ^T + lambda^2I)^-1 dx
        dq = total @ dx
        return dq
