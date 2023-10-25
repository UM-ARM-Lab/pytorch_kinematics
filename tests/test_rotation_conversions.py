import timeit

import torch

from pytorch_kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix, axis_angle_to_matrix, \
    pos_rot_to_matrix, matrix_to_pos_rot, random_rotations
import zpk_cpp


def test_axis_angle_to_matrix_perf():
    number = 100
    N = 1_000

    axis_angle = torch.randn([N, 3], device='cuda', dtype=torch.float64)
    axis_1d = torch.tensor([1., 0, 0], device='cuda', dtype=torch.float64)  # in the FK code this is NOT batched!
    theta = axis_angle.norm(dim=1, keepdim=False)

    dt1 = timeit.timeit(lambda: axis_angle_to_matrix(axis_angle), number=number)
    print(f'Old method: {dt1:.5f}')

    dt2 = timeit.timeit(lambda: axis_and_angle_to_matrix(axis=axis_1d, theta=theta), number=number)
    print(f'New method: {dt2:.5f}')


def test_axis_angle_to_matrix_perf_zpk():
    # Seems the C++ version is not faster than then python version for this case.

    # now test perf for a higher dim version, which has batches (B) of joints (N)
    number = 100
    N = 10
    B = 10_000
    axis = torch.randn([B, N, 3], device='cuda', dtype=torch.float64)
    axis = axis / axis.norm(dim=2, keepdim=True)
    theta = torch.randn([B, N], device='cuda', dtype=torch.float64)

    dt1 = timeit.timeit(lambda: axis_and_angle_to_matrix(axis, theta), number=number)
    print(f'Py: {dt1:.5f}')

    dt2 = timeit.timeit(lambda: zpk_cpp.axis_and_angle_to_matrix(axis, theta), number=number)
    print(f'Cpp: {dt2:.5f}')

    a1 = axis_and_angle_to_matrix(axis, theta)
    a2 = zpk_cpp.axis_and_angle_to_matrix(axis, theta)
    torch.testing.assert_allclose(a1, a2[..., :3, :3])


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
    test_axis_angle_to_matrix_perf_zpk()
    test_axis_angle_to_matrix_perf()
    test_pos_rot_conversion()
