import torch
from pytorch_mppi.mppi import handle_batch_input


@handle_batch_input(n=2)
def add_2d(a, b):
    assert a.ndim == 2
    assert b.ndim == 2
    return a + b


@handle_batch_input(n=3)
def add_3d(a, b):
    assert a.ndim == 3
    assert b.ndim == 3
    return a + b


def test_batch_wrapper_2d():
    a_2d = torch.tensor([[0.1, 0.2, 0.3]])
    b_2d = torch.tensor([[0.5, -0.2, 0.3]])
    a_3d = torch.tile(a_2d, [1, 1, 1])
    b_3d = torch.tile(b_2d, [1, 1, 1])
    a_4d = torch.tile(a_3d, [2, 1, 1])
    b_4d = torch.tile(b_3d, [2, 1, 1])
    expected_sum_2d = torch.tensor([[0.6, 0.0, 0.6]])
    expected_sum_3d = torch.tensor([[[0.6, 0.0, 0.6]]])
    expected_sum_4d = torch.tensor([[[[0.6, 0.0, 0.6]]], [[[0.6, 0.0, 0.6]]]])
    sum_2d = add_2d(a_2d, b_2d)
    sum_3d = add_2d(a_3d, b_3d)
    sum_4d = add_2d(a_4d, b_4d)
    assert torch.allclose(sum_2d, expected_sum_2d)
    assert torch.allclose(sum_3d, expected_sum_3d)
    assert torch.allclose(sum_4d, expected_sum_4d)


def test_batch_wrapper_3d():
    a_3d = torch.tensor([[[0.1, 0.2, 0.3]]])
    b_3d = torch.tensor([[[0.5, -0.2, 0.3]]])
    a_4d = torch.tile(a_3d, [2, 1, 1])
    b_4d = torch.tile(b_3d, [2, 1, 1])
    expected_sum_3d = torch.tensor([[[0.6, 0.0, 0.6]]])
    expected_sum_4d = torch.tensor([[[[0.6, 0.0, 0.6]]], [[[0.6, 0.0, 0.6]]]])
    sum_3d = add_3d(a_3d, b_3d)
    sum_4d = add_3d(a_4d, b_4d)
    assert torch.allclose(sum_3d, expected_sum_3d)
    assert torch.allclose(sum_4d, expected_sum_4d)
