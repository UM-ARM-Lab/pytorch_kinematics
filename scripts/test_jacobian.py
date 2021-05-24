import math
import torch
import tensorflow_kinematics as pk


def test_correctness():
    chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
    J = chain.jacobian(th)

    assert torch.allclose(J, torch.tensor([[[0, 1.41421356e-02, 0, 2.82842712e-01, 0, 0, 0],
                                            [-6.60827561e-01, 0, -4.57275649e-01, 0, 5.72756493e-02, 0, 0],
                                            [0, 6.60827561e-01, 0, -3.63842712e-01, 0, 8.10000000e-02, 0],
                                            [0, 0, -7.07106781e-01, 0, -7.07106781e-01, 0, -1],
                                            [0, 1, 0, -1, 0, 1, 0],
                                            [1, 0, 7.07106781e-01, 0, -7.07106781e-01, 0, 0]]]))

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


def test_jacobian_prismatic():
    chain = pk.build_serial_chain_from_urdf(open("prismatic_robot.urdf").read(), "link4")
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


if __name__ == "__main__":
    test_correctness()
    test_parallel()
    test_dtype_device()
    test_gradient()
    test_jacobian_prismatic()
