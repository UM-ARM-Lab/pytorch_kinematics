import timeit

import torch

from pytorch_kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix_directly, axis_angle_to_matrix


def test_axis_angle_to_matrix_perf():
    number = 1_000
    N = 1_000

    dt1 = timeit.timeit(lambda: axis_angle_to_matrix(torch.randn([N, 3])), number=number)
    print(f'{dt1:.5f}')

    dt2 = timeit.timeit(lambda: axis_and_angle_to_matrix_directly(axis=torch.tensor([1.0, 0, 0]), theta=torch.randn([N, 1])),
                        number=number)
    print(f'{dt2:.5f}')

    assert dt1 > dt2
