import torch

from pytorch_kinematics import transforms


def calc_jacobian(serial_chain, th, tool=None, ret_eef_pose=False):
    """
    Return robot Jacobian J in base frame (N,6,DOF) where dot{x} = J dot{q}
    The first 3 rows relate the translational velocities and the
    last 3 rows relate the angular velocities.

    tool is the transformation wrt the end effector; default is identity. If specified, will have to
    specify for each of the N inputs

    FIXME: this code assumes the joint frame and the child link frame are the same
    """
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=serial_chain.dtype, device=serial_chain.device)
    if len(th.shape) <= 1:
        N = 1
        th = th.reshape(1, -1)
    else:
        N = th.shape[0]
    ndof = th.shape[1]

    j_eef = torch.zeros((N, 6, ndof), dtype=serial_chain.dtype, device=serial_chain.device)

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
            # cur_transform transforms a point in eef frame into a point in joint frame, i.e. p_joint = curr_transform @ p_eef
            axis_in_eef = cur_transform[:, :3, :3].transpose(1, 2) @ f.joint.axis
            eef2joint_pos_in_joint = cur_transform[:, :3, 3].unsqueeze(2)
            joint2eef_rot = cur_transform[:, :3, :3].transpose(1, 2)  # transpose of rotation is inverse
            eef2joint_pos_in_eef = joint2eef_rot @ eef2joint_pos_in_joint
            position_jacobian = torch.cross(axis_in_eef, eef2joint_pos_in_eef.squeeze(2), dim=1)
            j_eef[:, :, -cnt] = torch.cat((position_jacobian, axis_in_eef), dim=-1)
        elif f.joint.joint_type == "prismatic":
            cnt += 1
            j_eef[:, :3, -cnt] = f.joint.axis.repeat(N, 1) @ cur_transform[:, :3, :3]
        cur_frame_transform = f.get_transform(th[:, -cnt]).get_matrix()
        cur_transform = cur_frame_transform @ cur_transform

    # currently j_eef is Jacobian in end-effector frame, convert to base/world frame
    pose = serial_chain.forward_kinematics(th).get_matrix()
    rotation = pose[:, :3, :3]
    j_tr = torch.zeros((N, 6, 6), dtype=serial_chain.dtype, device=serial_chain.device)
    j_tr[:, :3, :3] = rotation
    j_tr[:, 3:, 3:] = rotation
    j_w = j_tr @ j_eef
    if ret_eef_pose:
        return j_w, pose
    return j_w
