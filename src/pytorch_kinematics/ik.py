from pytorch_kinematics.chain import SerialChain
from pytorch_kinematics.transforms import Transform3d
from pytorch_kinematics.transforms import rotation_conversions
from typing import Union, Optional, Callable
import math
import torch
from matplotlib import pyplot as plt, cm as cm

# Check if torch.compile is available (PyTorch 2.0+)
_TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and torch.__version__ >= '2.0'


def _ik_step_kernel(m_flat: torch.Tensor, target_pos: torch.Tensor, target_wxyz: torch.Tensor,
                    J: torch.Tensor, reg_matrix: torch.Tensor, num_retries: int,
                    lm_damping: float = 0.0, task_weight: Optional[torch.Tensor] = None):
    """
    Fused IK step: delta_pose + damped least squares. Compatible with torch.compile(fullgraph=True).

    Args:
        m_flat: End-effector poses (N*M, 4, 4) where N=num_problems, M=num_retries
        target_pos: Target positions (N, 3)
        target_wxyz: Target orientations as wxyz quaternions (N, 4)
        J: Jacobian matrix (N*M, 6, DOF)
        reg_matrix: Regularization matrix (6, 6)
        num_retries: Number of retries per problem (M)
        lm_damping: Levenberg-Marquardt damping factor. When > 0, adds error-proportional
            regularization: mu = lm_damping * ||dx||^2. Large errors get more damping (stable),
            small errors get less (fast convergence). Inspired by mink's task-level LM damping.

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

    # Apply per-coordinate weighting: W @ J, W @ dx
    if task_weight is not None:
        w = task_weight.reshape(1, 6, 1)  # (1, 6, 1)
        J_w = w * J       # (NM, 6, DOF) weighted Jacobian
        dx_w = w * dx     # (NM, 6, 1) weighted error
    else:
        J_w = J
        dx_w = dx

    # DLS with adaptive Levenberg-Marquardt damping:
    # reg = lambda^2 * I + lm_damping * ||dx_w||^2 * I
    if lm_damping > 0.0:
        mu = lm_damping * (dx_w * dx_w).sum(dim=1, keepdim=True)  # (NM, 1, 1)
        tmpA = J_w @ J_w.transpose(1, 2) + reg_matrix + mu * torch.eye(6, device=J.device, dtype=J.dtype)
    else:
        tmpA = J_w @ J_w.transpose(1, 2) + reg_matrix
    A = torch.linalg.solve(tmpA, dx_w)
    dq = J_w.transpose(1, 2) @ A
    return dq, dx


def _ik_step_kernel_svd(m_flat: torch.Tensor, target_pos: torch.Tensor, target_wxyz: torch.Tensor,
                        J: torch.Tensor, reg_matrix: torch.Tensor, num_retries: int,
                        lm_damping: float = 0.0, task_weight: Optional[torch.Tensor] = None):
    """
    IK step using SVD-based damped least squares. Generally slower than the Cholesky-based
    kernel, but exposes singular values for selective damping if needed.

    The DLS pseudoinverse is: J^T (JJ^T + lambda^2 I)^{-1}
    Via SVD (J = U D V^T): sum_i (d_i / (d_i^2 + lambda^2)) v_i u_i^T

    Args:
        m_flat: End-effector poses (N*M, 4, 4)
        target_pos: Target positions (N, 3)
        target_wxyz: Target orientations as wxyz quaternions (N, 4)
        J: Jacobian matrix (N*M, 6, DOF)
        reg_matrix: Regularization matrix (6, 6) — only the diagonal (scalar) is used
        num_retries: Number of retries per problem (M)

    Returns:
        dq: Joint velocity (N*M, DOF, 1)
        dx: Pose error (N*M, 6, 1)
    """
    NM = m_flat.shape[0]
    M = num_retries
    N = NM // M
    m = m_flat.view(N, M, 4, 4)

    # delta_pose (same as _ik_step_kernel)
    pos_diff = (target_pos.unsqueeze(1) - m[:, :, :3, 3]).view(-1, 3, 1)
    cur_wxyz = rotation_conversions.matrix_to_quaternion(m[:, :, :3, :3])
    diff_wxyz = rotation_conversions.quaternion_multiply(
        target_wxyz.unsqueeze(1),
        rotation_conversions.quaternion_invert(cur_wxyz))
    diff_axis_angle = rotation_conversions.quaternion_to_axis_angle(diff_wxyz)
    rot_diff = diff_axis_angle.view(-1, 3, 1)
    dx = torch.cat((pos_diff, rot_diff), dim=1)

    # Apply per-coordinate weighting
    if task_weight is not None:
        w = task_weight.reshape(1, 6, 1)
        J_w = w * J
        dx_w = w * dx
    else:
        J_w = J
        dx_w = dx

    # SVD-based DLS
    regularization = reg_matrix[0, 0]  # scalar lambda^2
    if lm_damping > 0.0:
        mu = lm_damping * (dx_w * dx_w).sum(dim=1, keepdim=True).squeeze(2)  # (NM, 1)
        regularization = regularization + mu
    U, D, Vh = torch.linalg.svd(J_w)
    m_sv = D.shape[1]
    denom = D ** 2 + regularization
    prod = D / denom
    inverted = torch.diag_embed(prod)
    Vh = Vh[:, :m_sv, :]
    total = Vh.transpose(1, 2) @ inverted @ U.transpose(1, 2)
    dq = total @ dx_w
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
        # use torch.where instead of boolean indexing for compile compatibility
        not_converged = ~self.converged
        not_converged_dof = not_converged.unsqueeze(-1)  # (M, num_retries, 1) for DOF broadcast
        self.solutions = torch.where(not_converged_dof, qq, self.solutions)
        self.err_pos = torch.where(not_converged, err_pos, self.err_pos)
        self.err_rot = torch.where(not_converged, err_rot, self.err_rot)
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
    def do_line_search(self, chain, q, dq, target_pos, target_wxyz, initial_dx, problem_remaining=None,
                       fk_fn=None, eef_frame_idx=None):
        raise NotImplementedError()


class BacktrackingLineSearch(LineSearch):
    def __init__(self, max_lr=1.0, decrease_factor=0.5, max_iterations=5, sufficient_decrease=0.01):
        self.initial_lr = max_lr
        self.decrease_factor = decrease_factor
        self.max_iterations = max_iterations
        self.sufficient_decrease = sufficient_decrease

    def do_line_search(self, chain, q, dq, target_pos, target_wxyz, initial_dx, problem_remaining=None,
                       fk_fn=None, eef_frame_idx=None):
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
        improvement = torch.zeros(NM, device=q.device, dtype=q.dtype)
        for i in range(self.max_iterations):
            if not remaining.any():
                break
            # try stepping with this learning rate
            q_new = q + lr.unsqueeze(1) * dq
            # evaluate the error using tensor API when available
            if fk_fn is not None and eef_frame_idx is not None:
                all_tf = fk_fn(q_new)
                m = all_tf[eef_frame_idx].view(-1, M, 4, 4)
            else:
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
                 lr: float = 1.0, line_search: Optional[LineSearch] = None,
                 regularlization: float = 1e-9, lm_damping: float = 0.1,
                 position_weight: float = 1.0, orientation_weight: float = 1.0,
                 debug=False,
                 early_stopping_any_converged=False,
                 early_stopping_no_improvement="any", early_stopping_no_improvement_patience=2,
                 enforce_joint_limits: bool = True,
                 num_limit_refinement_iterations: int = 10,
                 clamp_to_limits: bool = False
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
        :param position_weight: weight for position error in DLS objective. Higher values prioritize
        position accuracy. Default 1.0.
        :param orientation_weight: weight for orientation error in DLS objective. Higher values prioritize
        orientation accuracy. Default 1.0.
        :param lm_damping: Levenberg-Marquardt damping factor. Adds error-proportional regularization
        mu = lm_damping * ||error||^2 to the DLS solve, so large errors get more damping (stable) and
        small errors get less (fast convergence). Default 0.1.
        :param enforce_joint_limits: whether to enforce joint limits on the solution. Uses the chain's joint limits
        (chain.low/chain.high). After solving, revolute joints are wrapped by multiples of 2*pi, then any remaining
        violations are resolved via clamped refinement iterations. Set to False to disable.
        :param num_limit_refinement_iterations: number of clamped IK refinement iterations for enforcing joint limits
        :param clamp_to_limits: if True, clamp joint values to chain limits after each IK step (projected gradient).
        Uses chain.low/chain.high. Requires the chain to have finite joint limits. Default False.
        """
        self.chain = serial_chain
        self.dtype = serial_chain.dtype
        self.device = serial_chain.device
        joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(joint_names)
        self.debug = debug
        self.enforce_joint_limits = enforce_joint_limits
        self.num_limit_refinement_iterations = num_limit_refinement_iterations
        self.clamp_to_limits = clamp_to_limits and torch.isfinite(serial_chain.low).any()
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
        self.lm_damping = lm_damping
        self.line_search = line_search
        # Per-coordinate weighting: scales rows of Jacobian and error
        self._task_weight = torch.tensor(
            [position_weight] * 3 + [orientation_weight] * 3,
            device=self.device, dtype=self.dtype
        )  # (6,)

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


class PseudoInverseIK(InverseKinematics):
    def __init__(self, *args, use_compile: bool = False, **kwargs):
        """
        Initialize PseudoInverseIK solver.

        Args:
            *args: Arguments passed to InverseKinematics.
            use_compile: If True and PyTorch 2.0+ is available, use torch.compile
                for JIT compilation of FK, Jacobian, and IK step kernels. This can
                provide performance improvements after a warmup period. Default: False.
            **kwargs: Keyword arguments passed to InverseKinematics.
        """
        super().__init__(*args, **kwargs)
        # Pre-compute regularization matrix once
        self._reg_matrix = self.regularlization * torch.eye(6, device=self.device, dtype=self.dtype)

        # Set up compute kernels (potentially compiled)
        self._use_compile = use_compile and _TORCH_COMPILE_AVAILABLE
        if self._use_compile:
            self._ik_step_fn = torch.compile(_ik_step_kernel)
            self._jacobian_fn = torch.compile(self.chain.jacobian_tensor)
            self._fk_fn = torch.compile(self.chain.forward_kinematics_tensor)
        else:
            self._ik_step_fn = _ik_step_kernel
            self._jacobian_fn = self.chain.jacobian_tensor
            self._fk_fn = self.chain.forward_kinematics_tensor
        self._eef_frame_idx = self.chain._serial_eef_frame_idx

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

        for i in range(self.max_iterations):
            with torch.no_grad():
                # early termination if we're out of problems to solve
                if not sol.remaining.any():
                    break
                sol.iterations += 1
                # compute Jacobian, end-effector pose, and pose error in one FK pass
                J, m = self._jacobian_fn(q, True)
                dq, dx = self._ik_step_fn(m, target_pos, target_wxyz, J,
                                          self._reg_matrix, self.num_retries, self.lm_damping, self._task_weight)
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
                        self.err_min = torch.where(improved, self.err, self.err_min)

                        self.no_improve_counter = torch.where(improved,
                                                              torch.zeros_like(self.no_improve_counter),
                                                              self.no_improve_counter + 1)

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

            with torch.no_grad():
                if self.line_search is not None:
                    lr, _ = self.line_search.do_line_search(self.chain, q, dq, target_pos, target_wxyz,
                                                            dx, problem_remaining=sol.remaining,
                                                            fk_fn=self._fk_fn,
                                                            eef_frame_idx=self._eef_frame_idx)
                    lr = lr.unsqueeze(1)
                else:
                    lr = self.lr
                q = q.add(dq, alpha=lr) if isinstance(lr, float) else q.add(lr * dq)
                if self.clamp_to_limits:
                    q = torch.clamp(q, self.chain.low, self.chain.high)

        # Final convergence check for the last stepped q
        with torch.no_grad():
            J_final, m_final = self._jacobian_fn(q, True)
            _, dx_final = self._ik_step_fn(m_final, target_pos, target_wxyz, J_final,
                                           self._reg_matrix, self.num_retries, self.lm_damping, self._task_weight)
            self.err_all = dx_final.squeeze(2)
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
            dq, _ = self._ik_step_fn(m_flat, target_pos, target_wxyz, J,
                                     self._reg_matrix, self.num_retries, self.lm_damping, self._task_weight)
            q = q + self.lr * dq.squeeze(2)
            q = torch.clamp(q, self.chain.low, self.chain.high)

        # Reset convergence so all retries are re-evaluated with the limit-enforced solutions
        sol.converged[:] = False
        sol.converged_any[:] = False
        sol.converged_pos[:] = False
        sol.converged_rot[:] = False

        # Recompute error and update solution
        J_new, m_new = self._jacobian_fn(q, True)
        _, dx_new = self._ik_step_fn(m_new, target_pos, target_wxyz, J_new,
                                     self._reg_matrix, self.num_retries, self.lm_damping, self._task_weight)
        sol.update(q, dx_new.squeeze(2), use_keep_mask=False)


class PseudoInverseIKWithSVD(PseudoInverseIK):
    """SVD-based damped least squares IK solver.

    About 2x slower per iteration than the default Cholesky-based solver (DLS),
    but converges slightly better near singularities (e.g. 99-100% vs 97-98%
    on near-singular targets). Exposes singular values which enables selective
    damping if subclassed further.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the step kernel with the SVD variant
        if self._use_compile:
            self._ik_step_fn = torch.compile(_ik_step_kernel_svd)
        else:
            self._ik_step_fn = _ik_step_kernel_svd
