import torch
from pytorch_kinematics import transforms


def calc_jacobian(serial_chain, th, tool=transforms.Transform3d()):
    # TODO handle batch input
    ndof = len(th)
    j_fl = torch.zeros((6, ndof), dtype=tool.dtype, device=tool.device)
    cur_transform = tool.get_matrix()

    cnt = 0
    for f in reversed(serial_chain._serial_frames):
        if f.joint.joint_type == "revolute":
            cnt += 1
            d = torch.tensor([-cur_transform[0, 0] * cur_transform[1, 3]
                              + cur_transform[1, 0] * cur_transform[0, 3],
                              -cur_transform[0, 1] * cur_transform[1, 3]
                              + cur_transform[1, 1] * cur_transform[0, 3],
                              -cur_transform[0, 2] * cur_transform[1, 3]
                              + cur_transform[1, 2] * cur_transform[0, 3]], dtype=tool.dtype, device=tool.device)
            delta = cur_transform[2, 0:3]
            j_fl[:, -cnt] = torch.cat((d, delta), dim=-1)
        cur_frame_transform = f.get_transform(th[-cnt]).get_matrix()
        cur_transform = torch.dot(cur_frame_transform, cur_transform)

    pose = serial_chain.forward_kinematics(th).get_matrix()
    rotation = pose[:3, :3]
    j_tr = torch.zeros((6, 6), dtype=tool.dtype, device=tool.device)
    j_tr[:3, :3] = rotation
    j_tr[3:, 3:] = rotation
    j_w = torch.dot(j_tr, j_fl)
    return j_w
