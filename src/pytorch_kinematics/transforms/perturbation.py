import torch
from pytorch_kinematics.transforms.rotation_conversions import axis_angle_to_matrix


def sample_perturbations(T, num_perturbations, radian_sigma, translation_sigma):
    """
    Sample perturbations around the given transform. The translation and rotation are sampled independently from
    0 mean gaussians. The angular perturbations' directions are uniformly sampled from the unit sphere while its
    magnitude is sampled from a gaussian.
    :param T: given transform to perturb around
    :param num_perturbations: number of perturbations to sample
    :param radian_sigma: standard deviation of the gaussian angular perturbation in radians
    :param translation_sigma: standard deviation of the gaussian translation perturbation in meters / T units
    :return: perturbed transforms; may not include the original transform
    """
    dtype = T.dtype
    device = T.device
    perturbed = torch.eye(4, dtype=dtype, device=device).repeat(num_perturbations, 1, 1)

    delta_R = torch.randn((num_perturbations, 3), dtype=dtype, device=device) * radian_sigma
    delta_R = axis_angle_to_matrix(delta_R)
    perturbed[:, :3, :3] = delta_R @ T[..., :3, :3]
    perturbed[:, :3, 3] = T[..., :3, 3]

    delta_t = torch.randn((num_perturbations, 3), dtype=dtype, device=device) * translation_sigma
    perturbed[:, :3, 3] += delta_t

    return perturbed
