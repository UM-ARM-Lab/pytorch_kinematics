from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from typing import Collection, Iterable, Optional, Tuple, Union

import numpy as np
import torch

from .transform3d import Transform3d
from .parameter_conversions import mdh_to_homogeneous, homogeneous_to_mdh


class ParameterConvention(Enum):
    """A parameter convention for a kinematic chain."""

    MDH = 1  # Modified Denavit-Hartenberg convention after Craig


class ParameterizedTransform(Transform3d, ABC):
    """
    A 3d transform which is made from a set of (differentiable) parameters.

    The ParameterizedTransform class supports two levels of batching: The first level is the batch dimension of the
    parameters for a single robot, the second level is the batch dimension of multiple robots.
    """

    convention: ParameterConvention  # Implement this in subclasses
    parameter_names: Tuple[str]  # Implement this in subclasses

    def __init__(
            self,
            parameters: Optional[torch.Tensor] = None,
            dtype: torch.dtype = torch.float32,
            device: str = 'cpu',
            requires_grad: bool = True,
            matrix: Optional[torch.Tensor] = None,
            default_batch_size: Union[Tuple[int], Tuple[int, int]] = (1, 1)
    ):
        """Initialize a ParameterizedTransform."""
        super().__init__(dtype=dtype, device=device, matrix=matrix)
        assert len(default_batch_size) in (1, 2), "default_batch_size must be a tuple of length 1 or 2"

        if parameters is None:
            parameters = torch.zeros(*default_batch_size, self.get_num_parameters())
        while parameters.ndim < 1 + len(default_batch_size):
            parameters = parameters.unsqueeze(0)

        self.default_batch_size: Union[Tuple[int], Tuple[int, int]] = default_batch_size
        self.requires_grad: bool = requires_grad
        self.parameters: torch.Tensor = parameters.to(self.device, self.dtype)
        self._matrix = None

    @abstractmethod
    def get_matrix(self) -> torch.Tensor:
        """Returns the matrix representation of the transform."""

    @abstractmethod
    def update_joint_parameters(self, th: torch.Tensor, joint_types: np.array):
        """Updates the parameters of the parameters according to joint configuration th."""

    def clone(self) -> ParameterizedTransform:
        """
        Deep copy of ParameterizedTransform object. All internal tensors are cloned individually.

        Returns:
            new ParameterizedTransform object.
        """
        other = self.__class__(dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        other._matrix = self.get_matrix()
        other.parameters = self._parameters.detach().clone()
        if self._lu is not None:
            other._lu = [elem.clone() for elem in self._lu]
        return other

    def stack(self, *others, dim=0):
        """
        Stacks multiple ParameterizedTransform objects together.

        Args:
            others: ParameterizedTransform objects to stack.
            dim: Dimension along which to stack. Can be 0 for batch and 1 for joints.

        Returns:
            new ParameterizedTransform object.
        """
        other = self.__class__(dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        transforms = [self] + list(others)
        parameters = torch.cat([t._parameters for t in transforms], dim=dim).to(self.device, dtype=self.dtype)
        other._parameters = parameters
        return other

    def to(self, device, copy: bool = False, dtype=None):
        """Makes sure to also set parameters to the correct device."""
        other = super().to(device, copy, dtype)
        if other is self and other._parameters.device == device and (dtype is None or dtype == other._parameters.dtype):
            return self
        other.parameters = self._parameters.detach().to(device, dtype)
        return other

    def toTransform3d(self):
        """Returns a Transform3d object with the same matrix as this ParameterizedTransform."""
        return Transform3d(matrix=self.get_matrix(), dtype=self.dtype, device=self.device)

    @property
    def num_batch_levels(self) -> int:
        """Returns the number of batch levels."""
        return len(self.default_batch_size)

    @property
    def parameters(self) -> torch.Tensor:
        """Returns the joint parameters"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        """Sets the joint parameters"""
        if parameters.requires_grad != self.requires_grad:
            # This check allows to set non-leaf parameters and is necessary for example for __getitem__
            parameters.requires_grad = self.requires_grad
        self._parameters = parameters

    @classmethod
    def get_num_parameters(cls) -> int:
        """Returns the number of parameters"""
        return len(cls.parameter_names)

    def __getitem__(self, item):
        if isinstance(item, Iterable):
            item = [i if not isinstance(i, int) else slice(i, i + 1) for i in item]  # Make sure to not lose dimensions
        return self.__class__(parameters=self.parameters[item], dtype=self.dtype, device=self.device,
                              default_batch_size=self.default_batch_size)

    def __repr__(self) -> str:
        """Returns a string representation of the transform."""
        info = ', '.join([f'{name}={self.parameters[..., i]}' for i, name in enumerate(self.parameter_names)])
        return f"{self.__class__}({info})".replace('\n       ', '')


class MDHTransform(ParameterizedTransform):
    """A transformation derived from Denavit-Hartenberg parameters."""

    convention: ParameterConvention = ParameterConvention.MDH
    parameter_names: Tuple[str, str, str, str] = ('alpha', 'a', 'd', 'theta')

    @property
    def theta(self):
        """Returns the joint angle."""
        return self.parameters[..., 3]

    @property
    def d(self):
        """Returns the joint offset."""
        return self.parameters[..., 2]

    @property
    def a(self):
        """Returns the link length."""
        return self.parameters[..., 1]

    @property
    def alpha(self):
        """Returns the link twist."""
        return self.parameters[..., 0]

    def get_matrix(self) -> torch.Tensor:
        """Returns the matrix representation of the transform. Redos the computation on every call"""
        b = self.parameters.shape[0]
        if self.num_batch_levels == 1:
            self._matrix = mdh_to_homogeneous(self.parameters).view(b, 4, 4)
        else:
            self._matrix = mdh_to_homogeneous(self.parameters).view(b, -1, 4, 4)
        return self._matrix

    def update_joint_parameters(self, th: torch.Tensor, joint_types: np.array):
        """Updates the parameters of the parameters according to joint configuration th."""
        is_revolute = joint_types == 'revolute'
        is_prismatic = joint_types == 'prismatic'
        assert np.all(np.logical_xor(is_revolute, is_prismatic))
        self.parameters[is_revolute.nonzero()][:, 3] = th[is_revolute.nonzero()]
        self.parameters[is_prismatic.nonzero()][:, 2] = th[is_prismatic.nonzero()]

    @classmethod
    def from_homogeneous(cls,
                         homogeneous: torch.Tensor,
                         dtype: torch.dtype = torch.float32,
                         device: str = 'cpu',
                         requires_grad: bool = True,
                         ):
        """Creates a MDHTransform from a homogeneous transformation matrix."""
        parameters = homogeneous_to_mdh(homogeneous)
        return cls(parameters, dtype, device, requires_grad)


CONVENTION_IMPLEMENTATIONS = {
    ParameterConvention.MDH: MDHTransform,
}
