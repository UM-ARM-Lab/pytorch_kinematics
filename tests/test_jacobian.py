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


def test_jacobian_at_different_links():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(),
                                            "lbr_iiwa_link_7")
    # th = torch.tensor([-0.25203286, 2.47622446, 2.15980668, 0.207287, -1.18388882, 1.04708102, -2.04419997])
    th = torch.tensor([[0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0],
                       [0.2] * 7,
                       [-0.1] * 7])

    link_index = torch.tensor([4, 5, 3])

    loc = torch.tensor([
        [0., 0., 0.],
        [0.1, 0., -0.1],
        [-0.2, 0.1, -0.15]
    ])

    jac_des = torch.tensor([
        [[4.6046e-13, 2.9698e-01, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          -0.0000e+00, -0.0000e+00],
         [-2.9698e-01, -1.3928e-12, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [-1.1352e-17, 2.9698e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [-1.3625e-23, -2.0690e-13, -7.0711e-01, -6.7180e-12, -0.0000e+00,
          -0.0000e+00, -0.0000e+00],
         [1.8953e-16, 1.0000e+00, 1.2880e-12, -1.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [1.0000e+00, 4.8967e-12, 7.0711e-01, 2.0282e-12, 0.0000e+00,
          0.0000e+00, 0.0000e+00]],
        [[4.2741e-02, 4.8503e-01, 6.1423e-02, -7.9103e-02, 5.6115e-02,
          0.0000e+00, 0.0000e+00],
         [-1.4741e-06, 9.8321e-02, -9.6362e-02, -2.8875e-02, -8.2685e-02,
          -0.0000e+00, -0.0000e+00],
         [5.8838e-18, 8.4928e-03, -8.3220e-03, -9.8258e-02, -3.7912e-03,
          0.0000e+00, 0.0000e+00],
         [-3.1225e-17, -1.9867e-01, 1.9471e-01, 3.8554e-01, 1.1645e-02,
          0.0000e+00, 0.0000e+00],
         [9.7145e-17, 9.8007e-01, 3.9470e-02, -9.2185e-01, -3.7912e-02,
          -0.0000e+00, -0.0000e+00],
         [1.0000e+00, 4.8966e-12, 9.8007e-01, -3.9470e-02, 9.9921e-01,
          0.0000e+00, 0.0000e+00]],
        [[-1.3819e-01, 3.5181e-02, -1.3715e-01, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [-1.8062e-01, -3.5299e-03, -1.7621e-01, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [3.7651e-17, 1.9351e-01, -1.1927e-02, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [4.1633e-17, 9.9833e-02, -9.9335e-02, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [2.1337e-16, 9.9500e-01, 9.9667e-03, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00],
         [1.0000e+00, 4.8967e-12, 9.9500e-01, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00]]
    ])

    J = chain.jacobian(th, locations=loc, link_indices=link_index)
    assert torch.allclose(J, jac_des, atol=1e-5)


def get_jacobian_finite_difference(chain, th):
    """
    Get the jacobian of the chain using finite differencing
    :param chain: SerialChain
    :param th: torch.tensor of shape (batch_size, num_dofs)
    :return: torch.tensor of shape (batch_size, 6, num_dofs)

    # TODO this can be extended to non-serial chains with locations and link indices for testing
    """

    # estimate jacobian with finite differencing
    # th should be
    eps = 1e-3
    J = []
    for i in range(th.shape[1]):
        # use central difference scheme
        th_plus = th.clone()
        th_plus[:, i] += eps
        th_minus = th.clone()
        th_minus[:, i] -= eps

        # Do FK
        mat_plus = chain.forward_kinematics(th_plus).get_matrix()
        mat_minus = chain.forward_kinematics(th_minus).get_matrix()

        # get position finite difference
        position_J = (mat_plus[:, :3, 3] - mat_minus[:, :3, 3]) / (2 * eps)

        # get rotation - slightly more involved
        # rotation matrix diff
        diff_ori = mat_plus[:, :3, :3] @ mat_minus[:, :3, :3].permute(0, 2, 1)
        # ensure symmetric
        diff_ori = (diff_ori - diff_ori.permute(0, 2, 1)) / 2

        # get corresponding angular velocity
        omega = torch.stack([diff_ori[:, 2, 1], diff_ori[:, 0, 2], diff_ori[:, 1, 0]], dim=1)
        ori_J = omega / (2 * eps)

        J_col = torch.cat([position_J, ori_J], dim=1)
        J.append(J_col)

    return torch.stack(J, dim=2)


def test_jacobian_partial_th():
    chain_right = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "victor_mallet.urdf")).read(),
                                            'victor_right_arm_link_7')
    chain_left = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "victor_mallet.urdf")).read(),
                                            'victor_left_arm_link_7')
    th = -0.1 * torch.ones(1, 14)

    J_right_test = get_jacobian_finite_difference(chain_right, th)
    J_left_test = get_jacobian_finite_difference(chain_left, th)

    # test with left
    J_right_serial = chain_right.jacobian(th[:, 7:])
    J_left_serial = chain_left.jacobian(th[:, :7])

    # try normal chain and use appropriate link indices
    chain = pk.build_chain_from_urdf(open(os.path.join(TEST_DIR, "victor_mallet.urdf")).read())
    link_indices = torch.tensor([chain.frame_to_idx['victor_left_arm_link_7'],
                                 chain.frame_to_idx['victor_right_arm_link_7']])

    J_non_serial = chain.jacobian(th.expand(2, 14), link_indices=link_indices)

    # using slacker tolerance due to finite differencing
    # test serial
    assert torch.allclose(J_right_serial, J_right_test, atol=1e-4)
    assert torch.allclose(J_left_serial, J_left_test, atol=1e-4)

    # test non serial
    assert torch.allclose(J_non_serial[0], J_left_test.squeeze(0), atol=1e-4)
    assert torch.allclose(J_non_serial[1], J_right_test.squeeze(0), atol=1e-4)


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
    test_jacobian_at_different_links()
    test_comparison_to_autograd()
    test_jacobian_partial_th()
