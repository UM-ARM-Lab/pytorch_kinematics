from typing import Optional

import torch

from pytorch_kinematics.transforms.rotation_conversions import (
    axis_and_angle_to_matrix_33,
)


def sample_perturbations(
    T: torch.Tensor,
    num_perturbations: int,
    radian_sigma: float,
    translation_sigma: float,
    axis_of_rotation: Optional[torch.Tensor] = None,
    translation_perpendicular_to_axis_of_rotation: bool = True,
) -> torch.Tensor:
    """
    Sample perturbations around the given transform. The translation and rotation are sampled independently from
    0-mean Gaussians. Rotational perturbations are sampled via axis-angle with random directions unless an axis is given.

    Parameters
    ----------
    T : torch.Tensor
        Input transform of shape (..., 4, 4). Only the last two dims are used.
    num_perturbations : int
        Number of perturbations to sample.
    radian_sigma : float
        Stddev of Gaussian angular perturbation (radians).
    translation_sigma : float
        Stddev of Gaussian translation perturbation (meters).
    axis_of_rotation : torch.Tensor, optional
        If supplied, perturb around this axis (shape (3,) or (N,3)).
    translation_perpendicular_to_axis_of_rotation : bool
        If True, translation perturbations are forced perpendicular to axis_of_rotation.

    Returns
    -------
    torch.Tensor
        Perturbed transforms of shape (num_perturbations, 4, 4).
    """
    dtype = T.dtype
    device = T.device
    perturbed = torch.eye(4, dtype=dtype, device=device).repeat(num_perturbations, 1, 1)

    # Gaussian translation perturbations
    delta_t = torch.randn((num_perturbations, 3), dtype=dtype, device=device) * translation_sigma
    # consider sampling from the Bingham distribution
    theta = torch.randn(num_perturbations, dtype=dtype, device=device) * radian_sigma
    if axis_of_rotation is not None:
        axis_angle = axis_of_rotation
        # sample translation perturbation perpendicular to the axis of rotation
        # remove the component of delta_t along the axis_of_rotation
        if translation_perpendicular_to_axis_of_rotation:
            delta_t -= (delta_t * axis_of_rotation).sum(dim=1, keepdim=True) * axis_of_rotation
    else:
        axis_angle = torch.randn((num_perturbations, 3), dtype=dtype, device=device)
        # normalize to unit length
        axis_angle = axis_angle / axis_angle.norm(dim=1, keepdim=True)

    delta_R = axis_and_angle_to_matrix_33(axis_angle, theta)
    perturbed[:, :3, :3] = delta_R @ T[..., :3, :3]
    perturbed[:, :3, 3] = T[..., :3, 3]

    perturbed[:, :3, 3] += delta_t

    return perturbed
