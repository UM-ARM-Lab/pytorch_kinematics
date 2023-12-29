import torch
from .so3 import _so3_exp_map, so3_log_map, hat, so3_rotation_angle


def compute_V_matrix(rot_angles, skews, skews_square):
    """V matrix used for both se(3) exp and log maps"""
    dtype = rot_angles.dtype
    device = rot_angles.device

    rot_angles_square = rot_angles * rot_angles
    fac1 = (1 - rot_angles.cos()) / rot_angles_square
    fac2 = (rot_angles - rot_angles.sin()) / (rot_angles_square * rot_angles)

    V = (
            torch.eye(3, dtype=dtype, device=device)[None]
            + fac1[:, None, None] * skews
            + fac2[:, None, None] * skews_square
    )
    return V


def se3_exp_map(v: torch.Tensor, eps: float = 0.0001):
    """
    Exponential map from se(3) (actually R^6, so it's the Exp map, skipping intermediate representation)
    to SE(3) for a batch of 6D vectors.
    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    v = [t, omega]^T where t is translation and omega is rotation vector
    see page 44 of https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    N, dim = v.shape
    if dim != 6:
        raise ValueError("Input tensor shape has to be Nx6.")

    t = v[:, :3]
    # omega is log_rot in so3_exp_map
    omega = v[:, 3:]

    exp_omega_x, rot_angles, skews, skews_square = _so3_exp_map(omega, eps=eps)

    exp_v = torch.eye(4, dtype=v.dtype, device=v.device).repeat(N, 1, 1)
    exp_v[:, :3, :3] = exp_omega_x

    V = compute_V_matrix(rot_angles, skews, skews_square)

    Vt = V @ t[:, :, None]
    exp_v[:, :3, 3] = Vt.squeeze(-1)

    return exp_v


def se3_log_map(T, eps: float = 0.0001):
    """
    Convert a batch of 4x4 transformation matrices `T`
    to a batch of 6-dimensional coordinates of se(3) representations.

    Args:
        T: batch of transformation matrices of shape `(minibatch, 4, 4)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of logarithms of input transformation matrices
        of shape `(minibatch, 6)`.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    omega = so3_log_map(R, eps=eps)

    nrms = (omega * omega).sum(1)
    rot_angles = torch.clamp(nrms, eps).sqrt()

    # consider using the alternative
    # rot_angles = so3_rotation_angle(R)

    skews = hat(omega)
    skews_square = torch.bmm(skews, skews)

    V = compute_V_matrix(rot_angles, skews, skews_square)

    t_prime = V.inverse() @ t[:, :, None]
    v = torch.cat([t_prime.squeeze(-1), omega], dim=-1)
    return v


def adjoint(R, t):
    """
    Compute the adjoint representation Ad of SE(3) X [1], separated into its batch of 3x3 rotation matrices
    and 3D position vectors.
    The adjoint is used to linearly map tangent vectors between different points on the manifold.
    For example, E at identity = Ad(X) * E at X for any X in SE(3).

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        t: Batch of 3D position vectors of shape `(minibatch, 3)`.

    Returns:
        Batch of 6x6 matrices of shape `(minibatch, 6, 6)` where each matrix
        is of the form:
            `[ R    [t]xR ]
             [ 0   R ]`

    Raises:
        ValueError if `R` or `t` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Adjoint_representation
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input rotation matrices have to be 3x3.")

    N, dim = t.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    adj = t.new_zeros(N, 6, 6)

    adj[:, :3, :3] = R
    adj[:, :3, 3:] = hat(t) @ R
    adj[:, 3:, 3:] = R

    return adj
