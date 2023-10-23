from pytorch_kinematics.chain import SerialChain
from pytorch_kinematics.transforms import Transform3d
from pytorch_kinematics.transforms import rotation_conversions
from typing import NamedTuple, Union, Optional, Callable
import typing
import torch
import inspect
from matplotlib import pyplot as plt, cm as cm


class IKSolution(NamedTuple):
    # N is the total number of attempts
    err_pos: torch.Tensor
    err_rot: torch.Tensor
    # N boolean values indicating whether the solution converged (a solution could be found)
    converged_pos: torch.Tensor
    converged_rot: torch.Tensor
    # whether both position and rotation converged
    converged: torch.Tensor
    # N x DOF tensor of joint angles; if converged[i] is False, then solutions[i] is undefined
    solutions: torch.Tensor


# helper config sampling method
def gaussian_around_config(config: torch.Tensor, std: float) -> Callable[[int], torch.Tensor]:
    def config_sampling_method(num_configs):
        return torch.randn(num_configs, config.shape[0], dtype=config.dtype, device=config.device) * std + config

    return config_sampling_method


class InverseKinematics:
    """Jacobian follower based inverse kinematics solver"""

    def __init__(self, serial_chain: SerialChain,
                 pos_tolerance: float = 1e-3, rot_tolerance: float = 1e-2,
                 initial_configs: Optional[torch.Tensor] = None, num_retries: Optional[int] = None,
                 joint_limits: Optional[torch.Tensor] = None,
                 config_sampling_method: Union[str, Callable[[int], torch.Tensor]] = "uniform",
                 max_iterations: int = 100, lr: float = 0.5,
                 debug=False,
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

        self.max_iterations = max_iterations
        self.lr = lr
        self.regularlization = 1e-9
        self.optimizer_method = optimizer_method

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
        :param target_poses: (N, 4, 4) tensor
        :return: (N, DOF) tensor of joint angles
        """
        raise NotImplementedError()


class PseudoInverseIK(InverseKinematics):
    def solve(self, target_poses: Transform3d) -> IKSolution:
        target = target_poses.get_matrix()

        target_pos = target[:, :3, 3]
        # jacobian gives angular rotation about x,y,z axis of the base frame
        # convert target rot to desired rotation about x,y,z
        target_rot_rpy = rotation_conversions.matrix_to_euler_angles(target[:, :3, :3], "XYZ")

        q = self.initial_config
        # for logging, let's keep track of the joint angles at each iteration
        if self.debug:
            qs = [q]
            pos_errors = []
            rot_errors = []

        optimizer = None
        if inspect.isclass(self.optimizer_method) and issubclass(self.optimizer_method, torch.optim.Optimizer):
            q.requires_grad = True
            optimizer = torch.optim.Adam([q], lr=self.lr)
        for i in range(self.max_iterations):
            with torch.no_grad():
                # TODO early termination when below tolerance
                # compute forward kinematics
                # fk = self.chain.forward_kinematics(q)
                # N x 6 x DOF
                J, m = self.chain.jacobian(q, ret_eef_pose=True)

                pos_diff = target_pos - m[:, :3, 3]
                pos_diff = pos_diff.view(-1, 3, 1)
                rot_diff = target_rot_rpy - rotation_conversions.matrix_to_euler_angles(m[:, :3, :3], "XYZ")
                rot_diff = rot_diff.view(-1, 3, 1)

                dx = torch.cat((pos_diff, rot_diff), dim=1)
                tmpA = J @ J.transpose(1, 2) + self.regularlization * torch.eye(6, device=self.device, dtype=self.dtype)
                A = torch.linalg.solve(tmpA, dx)
                dq = J.transpose(1, 2) @ A
                dq = dq.squeeze(2)

            if optimizer is not None:
                q.grad = -dq
                optimizer.step()
                optimizer.zero_grad()
            else:
                q = q + self.lr * dq
            if self.debug:
                qs.append(q)
                pos_errors.append(pos_diff.reshape(-1, 3).norm(dim=1))
                rot_errors.append(rot_diff.norm(dim=(1, 2)))

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

        # check convergence
        # fk = self.chain.forward_kinematics(q)
        # pose_diff = target - fk.get_matrix()
        err_pos = pos_diff.squeeze(2).norm(dim=1)
        err_rot = rot_diff.squeeze(2).norm(dim=1)

        converged_pos = err_pos < self.pos_tolerance
        converged_rot = err_rot < self.rot_tolerance

        return IKSolution(converged_pos=converged_pos, converged_rot=converged_rot,
                          err_pos=err_pos, err_rot=err_rot,
                          converged=converged_pos & converged_rot, solutions=q)
