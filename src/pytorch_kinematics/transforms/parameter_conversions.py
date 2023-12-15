#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 23.11.23
import torch


def mdh_to_homogeneous(mdh_parameters: torch.Tensor) -> torch.Tensor:
    """
    Converts a set of MDH parameters to a homogeneous transformation matrix.

    Follows Craig, Introduction to Robotics, 2005, p. 75.
    :param mdh_parameters: The MDH parameters, ordered as alpha, a, d, theta.
    :return: The homogeneous transformation matrix.
    """
    alpha = mdh_parameters[..., 0]
    a = mdh_parameters[..., 1]
    d = mdh_parameters[..., 2]
    theta = mdh_parameters[..., 3]

    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)
    zeros = torch.zeros_like(theta)
    return torch.stack([
        torch.stack([ct, -st, zeros, a], dim=-1),
        torch.stack([st * ca, ct * ca, -sa, -d * sa], dim=-1),
        torch.stack([st * sa, ct * sa, ca, d * ca], dim=-1),
        torch.stack([zeros, zeros, zeros, torch.ones_like(theta)],
                    dim=-1)
    ], dim=-2)


def homogeneous_to_mdh(T: torch.Tensor) -> torch.Tensor:
    """
    Converts a homogeneous transformation matrix to a set of MDH parameters.

    Attention, this method is expensive due to an internal sanity check.
    Follows Craig, Introduction to Robotics, 2005, p. 75.
    :param T: The homogeneous transformation matrix.
    :return: The MDH parameters.
    """
    a = T[..., 0, 3]
    theta = torch.atan2(-T[..., 0, 1], T[..., 0, 0])
    alpha = torch.atan2(-T[..., 1, 2], T[..., 2, 2])
    d = torch.empty_like(a)
    use_cos = torch.isclose(torch.sin(alpha), torch.zeros_like(alpha))
    d[~use_cos] = -T[~use_cos][:, 1, 3] / torch.sin(alpha[~use_cos])
    d[use_cos] = T[use_cos][:, 2, 3] / torch.cos(alpha[use_cos])

    parameters = torch.stack([alpha, a, d, theta], dim=-1)
    if not torch.allclose(mdh_to_homogeneous(parameters), T, atol=1e-3):
        raise ValueError('The given transformation is not MDH.')

    return parameters
