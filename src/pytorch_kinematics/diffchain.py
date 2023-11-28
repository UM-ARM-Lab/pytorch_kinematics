# Author: Jonathan Külz
# Date: 23.11.23
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from pytorch_kinematics import SerialChain, Transform3d, build_chain_from_urdf
from pytorch_kinematics.transforms.parameterized_transform import (CONVENTION_IMPLEMENTATIONS, MDHTransform,
                                                                   ParameterConvention, ParameterizedTransform)


class DiffChain(SerialChain):
    """
    A serial kinematic chain, but with all joint offsets created from differentiable parameters.
    """

    def __init__(self,
                 chain: SerialChain,
                 end_frame_name: str,
                 root_frame_name: str = "",
                 batch_size: int = 1,
                 parameter_convention: Union[str, ParameterConvention] = ParameterConvention.MDH,
                 **kwargs):
        """Initialize the DiffChain by providing a serial chain."""
        super().__init__(chain, end_frame_name, root_frame_name, **kwargs)
        self.batch_size = batch_size
        parameter_batch_size = (batch_size, self.n_joints)
        self._convention: ParameterConvention = ParameterConvention(parameter_convention)
        self._num_parameters = CONVENTION_IMPLEMENTATIONS[self._convention].get_num_parameters()
        self.parameterized_offsets: ParameterizedTransform = \
            CONVENTION_IMPLEMENTATIONS[self._convention](default_batch_size=parameter_batch_size)
        self.__setup()

    @classmethod
    def from_serial_chain(cls, c: SerialChain):
        """Creates a DiffChain from a SerialChain."""
        return cls(c, c._serial_frames[-1].name, c._serial_frames[0].name)

    @property
    def parameters(self) -> torch.Tensor:
        """Returns the underlying differentiable parameters"""
        return self.parameterized_offsets.parameters

    def _get_jnt_transform(self, th) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override the serial chain method: The joint transform is contained in the joint offset already!"""
        T = torch.eye(4, dtype=self.dtype, device=self.device).expand(self.batch_size, self.n_joints, 4, 4)
        return T, T

    def get_joint_offset(self, chain_idx) -> Optional[torch.Tensor]:
        """Returns the parameterized joint offset"""
        joint_idx = self.joint_indices[chain_idx]
        if joint_idx == -1:  # This is not a parameterized, moving joint
            return self.joint_offsets[chain_idx]
        return self.parameterized_offsets[:, joint_idx, :, :].get_matrix()

    def forward_kinematics(self, th, end_only: bool = True) -> Transform3d:
        """Overrides the SerialChain method to ensure using the parameters of this chain."""
        th = th.reshape((self.batch_size, self.n_joints))
        self.set_joint_parameters(th)
        return super().forward_kinematics(th, end_only)

    def set_joint_parameters(self, th: torch.Tensor):
        """
        Overrides the parameters that describe the current joint configuration.
        """
        types = np.repeat(np.array([joint.joint_type for joint in self.get_joints()]).reshape(1, -1), self.batch_size, axis=0)
        self.parameterized_offsets.update_joint_parameters(th, types)

    def __setup(self):
        """Ensure that all parameters are differentiable -- create a single leaf node containing all parameters."""
        offset_matrix = torch.empty((self.n_joints, 4, 4))
        for i, frame in enumerate(self._serial_frames):
            if frame.joint.joint_type == 'fixed':
                continue
            joint_idx = self.joint_indices[i]
            offset_matrix[joint_idx, :, :] = frame.joint.offset.get_matrix()

        offset_matrix = torch.stack([offset_matrix] * self.batch_size, dim=0)
        self.parameterized_offsets = MDHTransform.from_homogeneous(offset_matrix,
                                                                   dtype=self.dtype,
                                                                   device=self.device,
                                                                   requires_grad=True)


def build_diff_chain_from_urdf(data, end_link_name, root_link_name="", **kwargs):
    """Pipes pk function to build a DiffChain from URDF data."""
    urdf_chain = build_chain_from_urdf(data)
    return DiffChain(urdf_chain, end_link_name, "" if root_link_name == "" else root_link_name, **kwargs)
