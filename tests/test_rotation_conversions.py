import timeit

import torch

from pytorch_kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix_33, axis_angle_to_matrix, \
    pos_rot_to_matrix, matrix_to_pos_rot, random_rotations


def test_axis_angle_to_matrix_perf():
    number = 100
    N = 1_000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    axis_angle = torch.randn([N, 3], device=device, dtype=torch.float64)
    axis_1d = torch.tensor([1., 0, 0], device=device, dtype=torch.float64)  # in the FK code this is NOT batched!
    theta = axis_angle.norm(dim=1, keepdim=True)

    dt1 = timeit.timeit(lambda: axis_angle_to_matrix(axis_angle), number=number)
    print(f'Old method: {dt1:.5f}')

    dt2 = timeit.timeit(lambda: axis_and_angle_to_matrix_33(axis=axis_1d, theta=theta), number=number)
    print(f'New method: {dt2:.5f}')


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


if __name__ == '__main__':
    test_axis_angle_to_matrix_perf()
    test_pos_rot_conversion()
