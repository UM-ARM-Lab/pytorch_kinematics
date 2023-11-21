import math
import os
from timeit import default_timer as timer

import torch

import pytorch_kinematics as pk

TEST_DIR = os.path.dirname(__file__)


def test_correctness():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(),
                                            "lbr_iiwa_link_7")
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
    J = chain.jacobian(th)

    J_expected = torch.tensor([[[0, 1.41421356e-02, 0, 2.82842712e-01, 0, 0, 0],
                                [-6.60827561e-01, 0, -4.57275649e-01, 0, 5.72756493e-02, 0, 0],
                                [0, 6.60827561e-01, 0, -3.63842712e-01, 0, 8.10000000e-02, 0],
                                [0, 0, -7.07106781e-01, 0, -7.07106781e-01, 0, -1],
                                [0, 1, 0, -1, 0, 1, 0],
                                [1, 0, 7.07106781e-01, 0, -7.07106781e-01, 0, 0]]])
    assert torch.allclose(J, J_expected, atol=1e-7)

    chain = pk.build_chain_from_sdf(open(os.path.join(TEST_DIR, "simple_arm.sdf")).read())
    chain = pk.SerialChain(chain, "arm_wrist_roll")
    th = torch.tensor([0.8, 0.2, -0.5, -0.3])
    J = chain.jacobian(th)
    torch.allclose(J, torch.tensor([[[0., -1.51017878, -0.46280904, 0.],
                                     [0., 0.37144033, 0.29716627, 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 1., 1., 1.]]]))


def test_jacobian_at_different_loc_than_ee():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(),
                                            "lbr_iiwa_link_7")
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
    loc = torch.tensor([0.1, 0, 0])
    J = chain.jacobian(th, locations=loc)
    J_c1 = torch.tensor([[[-0., 0.11414214, -0., 0.18284271, 0., 0.1, 0.],
                          [-0.66082756, -0., -0.38656497, -0., 0.12798633, -0., 0.1],
                          [-0., 0.66082756, -0., -0.36384271, 0., 0.081, -0.],
                          [-0., -0., -0.70710678, -0., -0.70710678, 0., -1.],
                          [0., 1., 0., -1., 0., 1., 0.],
                          [1., 0., 0.70710678, 0., -0.70710678, -0., 0.]]])

    assert torch.allclose(J, J_c1, atol=1e-7)

    loc = torch.tensor([-0.1, 0.05, 0])
    J = chain.jacobian(th, locations=loc)
    J_c2 = torch.tensor([[[-0.05, -0.08585786, -0.03535534, 0.38284271, 0.03535534, -0.1, -0.],
                          [-0.66082756, -0., -0.52798633, -0., -0.01343503, 0., -0.1],
                          [-0., 0.66082756, -0.03535534, -0.36384271, -0.03535534, 0.081, -0.05],
                          [-0., -0., -0.70710678, -0., -0.70710678, 0., -1.],
                          [0., 1., 0., -1., 0., 1., 0.],
                          [1., 0., 0.70710678, 0., -0.70710678, -0., 0.]]])

    assert torch.allclose(J, J_c2, atol=1e-7)

    # check that batching the location is fine
    th = th.repeat(2, 1)
    loc = torch.tensor([[0.1, 0, 0], [-0.1, 0.05, 0]])
    J = chain.jacobian(th, locations=loc)
    assert torch.allclose(J, torch.cat((J_c1, J_c2)), atol=1e-7)


def test_jacobian_y_joint_axis():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "simple_y_arm.urdf")).read(), "eef")
    th = torch.tensor([0.])
    J = chain.jacobian(th)
    J_c3 = torch.tensor([[[0.], [0.], [-0.3], [0.], [1.], [0.]]])
    assert torch.allclose(J, J_c3, atol=1e-7)


def test_parallel():
    N = 100
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(),
                                            "lbr_iiwa_link_7")
    th = torch.cat(
        (torch.tensor([[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]]), torch.rand(N, 7)))
    J = chain.jacobian(th)
    for i in range(N):
        J_i = chain.jacobian(th[i])
        assert torch.allclose(J[i], J_i)


def test_dtype_device():
    N = 1000
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(),
                                            "lbr_iiwa_link_7")
    chain = chain.to(dtype=dtype, device=d)
    th = torch.rand(N, 7, dtype=dtype, device=d)
    J = chain.jacobian(th)
    assert J.dtype is dtype


def test_gradient():
    N = 10
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(),
                                            "lbr_iiwa_link_7")
    chain = chain.to(dtype=dtype, device=d)
    th = torch.rand(N, 7, dtype=dtype, device=d, requires_grad=True)
    J = chain.jacobian(th)
    assert th.grad is None
    J.norm().backward()
    assert th.grad is not None


def test_jacobian_prismatic():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "prismatic_robot.urdf")).read(), "link4")
    th = torch.zeros(3)
    tg = chain.forward_kinematics(th)
    m = tg.get_matrix()
    pos = m[0, :3, 3]
    assert torch.allclose(pos, torch.tensor([0, 0, 1.]))
    th = torch.tensor([0, 0.1, 0])
    tg = chain.forward_kinematics(th)
    m = tg.get_matrix()
    pos = m[0, :3, 3]
    assert torch.allclose(pos, torch.tensor([0, -0.1, 1.]))
    th = torch.tensor([0.1, 0.1, 0])
    tg = chain.forward_kinematics(th)
    m = tg.get_matrix()
    pos = m[0, :3, 3]
    assert torch.allclose(pos, torch.tensor([0, -0.1, 1.1]))
    th = torch.tensor([0.1, 0.1, 0.1])
    tg = chain.forward_kinematics(th)
    m = tg.get_matrix()
    pos = m[0, :3, 3]
    assert torch.allclose(pos, torch.tensor([0.1, -0.1, 1.1]))

    J = chain.jacobian(th)
    assert torch.allclose(J, torch.tensor([[[0., 0., 1.],
                                            [0., -1., 0.],
                                            [1., 0., 0.],
                                            [0., 0., 0.],
                                            [0., 0., 0.],
                                            [0., 0., 0.]]]))


def test_comparison_to_autograd():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(),
                                            "lbr_iiwa_link_7")
    d = "cuda" if torch.cuda.is_available() else "cpu"
    chain = chain.to(device=d)

    def get_pt(th):
        return chain.forward_kinematics(th).transform_points(
            torch.zeros((1, 3), device=th.device, dtype=th.dtype)).squeeze(1)

    # compare the time taken
    N = 1000
    ths = (torch.tensor([[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]], device=d),
           torch.rand(N - 1, 7, device=d))
    th = torch.cat(ths)

    autograd_start = timer()
    j1 = torch.autograd.functional.jacobian(get_pt, inputs=th, vectorize=True)
    # get_pt will produce N x 3
    # jacobian will compute the jacobian of the N x 3 points with respect to each of the N x DOF inputs
    # so j1 is N x 3 x N x DOF (3 since it only considers the position change)
    # however, we know the ith point only has a non-zero jacobian with the ith input
    j1_ = j1[range(N), :, range(N)]
    pk_start = timer()
    j2 = chain.jacobian(th)
    pk_end = timer()
    # we can only compare the positional parts
    assert torch.allclose(j1_, j2[:, :3], atol=1e-6)
    print(f"for N={N} on {d} autograd:{(pk_start - autograd_start) * 1000}ms")
    print(f"for N={N} on {d} pytorch-kinematics:{(pk_end - pk_start) * 1000}ms")
    # if we have functools (for pytorch>=1.13.0 it comes with installing pytorch)
    try:
        import functorch
        ft_start = timer()
        grad_func = functorch.vmap(functorch.jacrev(get_pt))
        j3 = grad_func(th).squeeze(1)
        ft_end = timer()
        assert torch.allclose(j1_, j3, atol=1e-6)
        assert torch.allclose(j3, j2[:, :3], atol=1e-6)
        print(f"for N={N} on {d} functorch:{(ft_end - ft_start) * 1000}ms")
    except:
        pass


if __name__ == "__main__":
    test_correctness()
    test_parallel()
    test_dtype_device()
    test_gradient()
    test_jacobian_prismatic()
    test_jacobian_at_different_loc_than_ee()
    test_comparison_to_autograd()
