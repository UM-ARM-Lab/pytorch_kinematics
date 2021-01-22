import scipy.optimize as sco
import torch


def inverse_kinematics(serial_chain, pose, initial_state=None):
    # TODO implement IK (not implemented correctly)
    ndim = len(serial_chain.get_joint_parameter_names())
    if initial_state is None:
        x0 = torch.zeros(ndim, dtype=serial_chain.dtype, device=serial_chain.device)
    else:
        x0 = initial_state

    def object_fn(x):
        tf = serial_chain.forward_kinematics(x)
        obj = torch.square(
            torch.lstsq(tf.get_matrix(), pose.matrix())[0] - torch.eye(4, dtype=x0.dtype, device=x0.device)).sum()
        return obj

    ret = sco.minimize(object_fn, x0, method='BFGS')
    return ret.x
