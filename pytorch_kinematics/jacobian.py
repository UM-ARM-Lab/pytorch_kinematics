import torch
from pytorch_kinematics import transforms


def calc_jacobian(serial_chain, th, tool=None):
    """
    Return robot Jacobian J in base frame (N,6,DOF) where dot{x} = J dot{q}
    The first 3 rows relate the translational velocities and the
    last 3 rows relate the angular velocities.

    tool is the transformation wrt the end effector; default is identity. If specified, will have to
    specify for each of the N inputs
    """
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=serial_chain.dtype, device=serial_chain.device)
    if len(th.shape) <= 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    ndof = th.shape[1]

    # j_fl = torch.zeros((N, 6, ndof), dtype=serial_chain.dtype, device=serial_chain.device)
    j_fl = []
    if tool is None:
        cur_transform = transforms.Transform3d(device=serial_chain.device,
                                               dtype=serial_chain.dtype).get_matrix().repeat(N, 1, 1)
    else:
        if tool.dtype != serial_chain.dtype or tool.device != serial_chain.device:
            tool = tool.to(device=serial_chain.device, copy=True, dtype=serial_chain.dtype)
        cur_transform = tool.get_matrix()

    cnt = 0
    for f in reversed(serial_chain._serial_frames):
        if f.joint.joint_type == "revolute":
            cnt += 1
            d = torch.stack([-cur_transform[:, 0, 0] * cur_transform[:, 1, 3]
                             + cur_transform[:, 1, 0] * cur_transform[:, 0, 3],
                             -cur_transform[:, 0, 1] * cur_transform[:, 1, 3]
                             + cur_transform[:, 1, 1] * cur_transform[:, 0, 3],
                             -cur_transform[:, 0, 2] * cur_transform[:, 1, 3]
                             + cur_transform[:, 1, 2] * cur_transform[:, 0, 3]]).transpose(0, 1)
            delta = cur_transform[:, 2, 0:3]
            # j_fl[:, :, -cnt] = torch.cat((d, delta), dim=-1)
            j_fl.append(torch.cat((d, delta), dim=-1))

        elif f.joint.joint_type == "prismatic":
            raise ValueError('Not supported')
            # cnt += 1
            # j_fl[:, :3, -cnt] = f.joint.axis.repeat(N, 1) @ cur_transform[:, :3, :3]
            # j_fl[:, :3, -cnt] = f.joint.axis.repeat(N, 1) @ cur_transform[:, :3, :3]

        cur_frame_transform = f.get_transform(th[:, -cnt].view(N, 1)).get_matrix()
        cur_transform = cur_frame_transform @ cur_transform

    j_fl = torch.stack(j_fl, dim=2)
    j_fl = torch.flip(j_fl, dims=(2,))
    # currently j_fl is Jacobian in flange (end-effector) frame, convert to base/world frame
    pose = serial_chain.forward_kinematics(th)
    rotation = pose[:, :3, :3]
    zeros = torch.zeros_like(rotation)
    # j_tr = torch.zeros((N, 6, 6), dtype=serial_chain.dtype, device=serial_chain.device)
    j_tr_top = torch.cat((rotation, zeros), dim=2)
    j_tr_bot = torch.cat((zeros, rotation), dim=2)
    j_tr = torch.cat((j_tr_top, j_tr_bot), dim=1)
    # j_tr[:, :3, :3] = rotation
    # j_tr[:, 3:, 3:] = rotation
    j_w = j_tr @ j_fl
    return j_w


def calc_jacobian_and_hessian(serial_chain, th):
    """
        Calculates robot jacobian and kinematic hessian in the base frame

        Returns:
            J: torch.tensor of shape (N, 6, DOF) representing robot jacobian
            H: torch.tensor of shape (N, 6, DOF, DOF) - kinematic Hessian. The kinematic hessian is the partial
               derivative of the robot jacobian

    """
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=serial_chain.dtype, device=serial_chain.device)
    if len(th.shape) <= 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    ndof = th.shape[1]

    J = calc_jacobian(serial_chain, th)

    H = torch.zeros(N, 6, ndof, ndof, device=serial_chain.device, dtype=serial_chain.dtype)

    for j in range(ndof):
        for i in range(j, ndof):
            H[:, :3, j, i] = torch.cross(J[:, 3:, j], J[:, :3, i])
            H[:, 3:, j, i] = torch.cross(J[:, 3:, j], J[:, 3:, i])

            if i != j:
                H[:, :3, i, j] = H[:, :3, j, i]

    return J, H
