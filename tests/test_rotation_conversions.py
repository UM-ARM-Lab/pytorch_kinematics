import timeit

import torch

from pytorch_kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix_directly, axis_angle_to_matrix, \
    pos_rot_to_matrix, matrix_to_pos_rot, random_rotations


def test_axis_angle_to_matrix_perf():
    number = 1_000
    N = 1_000

    dt1 = timeit.timeit(lambda: axis_angle_to_matrix(torch.randn([N, 3])), number=number)
    print(f'{dt1:.5f}')

    dt2 = timeit.timeit(
        lambda: axis_and_angle_to_matrix_directly(axis=torch.tensor([1.0, 0, 0]), theta=torch.randn([N, 1])),
        number=number)
    print(f'{dt2:.5f}')

    assert dt1 > dt2


def test_pos_rot_conversion():
    N = 1000
    R = random_rotations(N)
    t = torch.randn((N, 3), dtype=R.dtype, device=R.device)
    T = torch.eye(4, dtype=R.dtype, device=R.device).repeat(N, 1, 1)
    T[:, :3, 3] = t
    T[:, :3, :3] = R
    pos, rot = matrix_to_pos_rot(T)
    TT = pos_rot_to_matrix(pos, rot)
    assert torch.allclose(T, TT, atol=1e-6)


test_pos_rot_conversion()
