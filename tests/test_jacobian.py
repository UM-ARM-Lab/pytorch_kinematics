import math
import torch
import pytorch_kinematics as pk


def test_correctness():
    chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
    J = chain.jacobian(th)
    assert torch.allclose(J, torch.tensor([[[-2.47727592e-12, 1.41421356e-02, -2.44155677e-12,
                                             2.82842712e-01, 3.93156995e-13, 1.67535985e-14,
                                             0.00000000e+00],
                                            [-6.60827561e-01, -3.23289325e-12, -4.57275649e-01,
                                             -2.63812394e-12, 5.72756493e-02, 1.76135096e-13,
                                             0.00000000e+00],
                                            [4.75722920e-17, 6.60827561e-01, -1.60872783e-12,
                                             -3.63842712e-01, 2.72085294e-13, 8.10000000e-02,
                                             0.00000000e+00],
                                            [-9.71829137e-17, -2.06797669e-13, -7.07106781e-01,
                                             -6.71813437e-12, -7.07106781e-01, 6.86429574e-12,
                                             -1.00000000e+00],
                                            [-3.06245972e-16, 1.00000000e+00, 1.28796020e-12,
                                             -1.00000000e+00, 8.21292608e-12, 1.00000000e+00,
                                             1.96776811e-12],
                                            [1.00000000e+00, 4.89681088e-12, 7.07106781e-01,
                                             2.02812117e-12, -7.07106781e-01, -2.17450736e-12,
                                             2.06834549e-13]]]))

    chain = pk.build_chain_from_sdf(open("simple_arm.sdf").read())
    chain = pk.SerialChain(chain, "arm_wrist_roll_frame")
    th = torch.tensor([0.8, 0.2, -0.5, -0.3])
    J = chain.jacobian(th)
    torch.allclose(J, torch.tensor([[[0., -1.51017878, -0.46280904, 0.],
                                     [0., 0.37144033, 0.29716627, 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 1., 1., 1.]]]))


def test_parallel():
    N = 100
    chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
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

    chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
    chain = chain.to(dtype=dtype, device=d)
    th = torch.rand(N, 7, dtype=dtype, device=d)
    J = chain.jacobian(th)
    assert J.dtype is dtype


def test_gradient():
    N = 10
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
    chain = chain.to(dtype=dtype, device=d)
    th = torch.rand(N, 7, dtype=dtype, device=d, requires_grad=True)
    J = chain.jacobian(th)
    assert th.grad is None
    J.norm().backward()
    assert th.grad is not None


if __name__ == "__main__":
    test_correctness()
    test_parallel()
    test_dtype_device()
    test_gradient()
