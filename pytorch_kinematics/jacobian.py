import torch
from pytorch_kinematics import transforms


def calc_jacobian(serial_chain, th, tool=transforms.Transform3d()):
    """
    Return robot Jacobian J in base frame (N,6,DOF) where dot{x} = J dot{q}
    The first 3 rows relate the translational velocities and the
    last 3 rows relate the angular velocities.
    """
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=serial_chain.dtype, device=serial_chain.device)
    if len(th.shape) <= 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    ndof = th.shape[1]
    if tool.dtype != serial_chain.dtype or tool.device != serial_chain.device:
        tool = tool.to(device=serial_chain.device, copy=True, dtype=serial_chain.dtype)

    j_fl = torch.zeros((N, 6, ndof), dtype=serial_chain.dtype, device=serial_chain.device)
    cur_transform = tool.get_matrix().repeat(N, 1, 1)

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
            j_fl[:, :, -cnt] = torch.cat((d, delta), dim=-1)
        # TODO handle prismatic joints
        cur_frame_transform = f.get_transform(th[:, -cnt].view(N, 1)).get_matrix()
        cur_transform = cur_frame_transform @ cur_transform

    # currently j_fl is Jacobian in flange (end-effector) frame, convert to base/world frame
    pose = serial_chain.forward_kinematics(th).get_matrix()
    rotation = pose[:, :3, :3]
    j_tr = torch.zeros((N, 6, 6), dtype=serial_chain.dtype, device=serial_chain.device)
    j_tr[:, :3, :3] = rotation
    j_tr[:, 3:, 3:] = rotation
    j_w = j_tr @ j_fl
    return j_w
