import math
import torch
import pytorch_kinematics as pk


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


def test_jacobian_at_different_loc_than_ee():
    chain = pk.build_serial_chain_from_urdf(open("kuka_iiwa.urdf").read(), "lbr_iiwa_link_7")
    th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
    loc = torch.tensor([0.1, 0, 0])
    J = chain.jacobian(th, locations=loc)
    J_c1 = torch.tensor([[[-0., 0.11414214, -0., 0.18284271, 0., 0.1, 0.],
                          [-0.66082756, -0., -0.38656497, -0., 0.12798633, -0., 0.1],
                          [-0., 0.66082756, -0., -0.36384271, 0., 0.081, -0.],
                          [-0., -0., -0.70710678, -0., -0.70710678, 0., -1.],
                          [0., 1., 0., -1., 0., 1., 0.],
                          [1., 0., 0.70710678, 0., -0.70710678, -0., 0.]]])

    assert torch.allclose(J, J_c1)

    loc = torch.tensor([-0.1, 0.05, 0])
    J = chain.jacobian(th, locations=loc)
    J_c2 = torch.tensor([[[-0.05, -0.08585786, -0.03535534, 0.38284271, 0.03535534, -0.1, -0.],
                          [-0.66082756, -0., -0.52798633, -0., -0.01343503, 0., -0.1],
                          [-0., 0.66082756, -0.03535534, -0.36384271, -0.03535534, 0.081, -0.05],
                          [-0., -0., -0.70710678, -0., -0.70710678, 0., -1.],
                          [0., 1., 0., -1., 0., 1., 0.],
                          [1., 0., 0.70710678, 0., -0.70710678, -0., 0.]]])

    assert torch.allclose(J, J_c2)

    # check that batching the location is fine
    th = th.repeat(2, 1)
    loc = torch.tensor([[0.1, 0, 0], [-0.1, 0.05, 0]])
    J = chain.jacobian(th, locations=loc)
    assert torch.allclose(J, torch.cat((J_c1, J_c2)))


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
    test_jacobian_at_different_loc_than_ee()
