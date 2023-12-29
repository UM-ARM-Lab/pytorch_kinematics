# follows The Probabilistic Robot Kinematics Model and its Application to Sensor Fusion
# uncertanties of poses are represented as 6x6 covariance matrices in their tangent space
# locally perturbed around mean value
from .se3 import adjoint
from .transform3d import Transform3d


def perturbations_global(R, t, perturbations):
    """Get perturbations expressed globally at the origin given perturbations expressed at X (R,t)
    pose perturbations are 6D vectors in the tangent space of SE(3) at X"""
    # the rotation component doesn't
    ad = adjoint(R, t)
    a = ad @ perturbations.transpose(-1, -2)
    a = a.squeeze().transpose(-1, -2)
    return a

    # import torch
    # b = []
    # for i in range(perturbations.shape[0]):
    #     b.append(ad @ perturbations[i])
    # b = torch.stack(b, dim=0)
    #
    # return perturbations @ ad


def sigma_global(R, t, sigma):
    """Get covariance (sigma) expressed globally at the origin given sigma expressed at X (R,t)"""
    ad = adjoint(R, t)
    return ad @ sigma @ ad.transpose(-1, -2)


def compose_uncertainty(X2: Transform3d, sigma1, sigma2):
    """Compose uncertainty of two poses. Note that this is not commutative. It is for pose X1 followed by X2.

    This is the simplest approximation and could be problematic for high perturbation values.

    To get the covariance at the end effector, sequentially compose the uncertainty of each link.
    """
    R2, t2 = X2.inverse().get_RT()
    ad = adjoint(R2, t2)
    sigma_12 = ad @ sigma1 @ ad.transpose(-1, -2) + sigma2
    return sigma_12
