import torch

import pytorch_kinematics.transforms as tf


def test_transform():
    N = 20
    mats = tf.random_rotations(N, dtype=torch.float64, device="cpu", requires_grad=True)
    assert list(mats.shape) == [N, 3, 3]
    # test batch conversions
    quat = tf.matrix_to_quaternion(mats)
    assert list(quat.shape) == [N, 4]
    mats_recovered = tf.quaternion_to_matrix(quat)
    assert torch.allclose(mats, mats_recovered)

    quat_identity = tf.quaternion_multiply(quat, tf.quaternion_invert(quat))
    assert torch.allclose(tf.quaternion_to_matrix(quat_identity), torch.eye(3, dtype=torch.float64).repeat(N, 1, 1))


def test_translations():
    t = tf.Translate(1, 2, 3)
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(
        1, 3, 3
    )
    points_out = t.transform_points(points)
    points_out_expected = torch.tensor(
        [[2.0, 2.0, 3.0], [1.0, 3.0, 3.0], [1.5, 2.5, 3.0]]
    ).view(1, 3, 3)
    assert torch.allclose(points_out, points_out_expected)

    N = 20
    points = torch.randn((N, N, 3))
    translation = torch.randn((N, 3))
    transforms = tf.Transform3d(pos=translation)
    translated_points = transforms.transform_points(points)
    assert torch.allclose(translated_points, translation.repeat(N, 1, 1).transpose(0, 1) + points)
    returned_points = transforms.inverse().transform_points(translated_points)
    assert torch.allclose(returned_points, points, atol=1e-6)


def test_rotate_axis_angle():
    t = tf.Transform3d().rotate_axis_angle(90.0, axis="Z")
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]).view(
        1, 3, 3
    )
    normals = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    ).view(1, 3, 3)
    points_out = t.transform_points(points)
    normals_out = t.transform_normals(normals)
    points_out_expected = torch.tensor(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]
    ).view(1, 3, 3)
    normals_out_expected = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    ).view(1, 3, 3)
    assert torch.allclose(points_out, points_out_expected)
    assert torch.allclose(normals_out, normals_out_expected)


def test_rotate():
    R = tf.so3_exp_map(torch.randn((1, 3)))
    t = tf.Transform3d().rotate(R)
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(
        1, 3, 3
    )
    normals = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    ).view(1, 3, 3)
    points_out = t.transform_points(points)
    normals_out = t.transform_normals(normals)
    points_out_expected = torch.bmm(points, R.transpose(-1, -2))
    normals_out_expected = torch.bmm(normals, R.transpose(-1, -2))
    assert torch.allclose(points_out, points_out_expected, atol=1e-7)
    assert torch.allclose(normals_out, normals_out_expected, atol=1e-7)
    for i in range(3):
        assert torch.allclose(points_out[0, i], R @ points[0, i], atol=1e-7)
        assert torch.allclose(normals_out[0, i], R @ normals[0, i], atol=1e-7)


def test_transform_combined():
    R = tf.so3_exp_map(torch.randn((1, 3)))
    tr = torch.randn((1, 3))
    t = tf.Transform3d(rot=R, pos=tr)
    N = 10
    points = torch.randn((N, 3))
    normals = torch.randn((N, 3))
    points_out = t.transform_points(points)
    normals_out = t.transform_normals(normals)
    for i in range(N):
        assert torch.allclose(points_out[i], R @ points[i] + tr, atol=1e-7)
        assert torch.allclose(normals_out[i], R @ normals[i], atol=1e-7)


def test_euler():
    euler_angles = torch.tensor([1, 0, 0.5])
    t = tf.Transform3d(rot=euler_angles)
    sxyz_matrix = torch.tensor([[0.87758256, -0.47942554, 0., 0., ],
                                [0.25903472, 0.47415988, -0.84147098, 0.],
                                [0.40342268, 0.73846026, 0.54030231, 0.],
                                [0., 0., 0., 1.]])
    # from tf.transformations import euler_matrix
    # print(euler_matrix(*euler_angles, "rxyz"))
    # print(t.get_matrix())
    assert torch.allclose(sxyz_matrix, t.get_matrix())


def test_quaternions():
    n = 10
    q = tf.random_quaternions(n)
    q_tf = tf.wxyz_to_xyzw(q)
    assert torch.allclose(q, tf.xyzw_to_wxyz(q_tf))


def test_compose():
    import torch
    theta = 1.5707
    a2b = tf.Transform3d(pos=[0.1, 0, 0])  # joint.offset
    b2j = tf.Transform3d(rot=tf.axis_angle_to_quaternion(theta * torch.tensor([0.0, 0, 1])))  # joint.axis
    j2c = tf.Transform3d(pos=[0.1, 0, 0])  # link.offset ?
    a2c = a2b.compose(b2j, j2c)
    m = a2c.get_matrix()
    print(m)
    print(a2c.transform_points(torch.zeros([1, 3])))


def test_so3_exp_map_consistency():
    N = 100
    eps = 1e-4

    omega = torch.randn((N, 3))
    R = tf.so3_exp_map(omega, eps=eps)
    omega_recovered = tf.so3_log_map(R, eps=eps)

    # ignore ones that are close to 0 or pi

    # 3 ways of getting rotation angle
    rot_mag = tf.so3_rotation_angle(R, eps=eps)
    rot_mag_from_omega = torch.linalg.norm(omega, dim=-1)
    mask = (rot_mag_from_omega > 0.005) & (rot_mag_from_omega < 3.1415 - 0.005)

    nrms = (omega * omega).sum(1)
    rot_angles = torch.clamp(nrms, eps).sqrt()

    assert torch.allclose(rot_mag[mask], rot_mag_from_omega[mask], atol=eps * 2)
    assert torch.allclose(rot_mag[mask], rot_angles[mask], atol=eps * 2)

    assert torch.allclose(omega[mask], omega_recovered[mask], atol=eps * 5)

    R_recovered = tf.so3_exp_map(omega_recovered, eps=eps)
    # but with the mask we should get better results
    assert torch.allclose(R[mask], R_recovered[mask], atol=eps * 4)


def test_se3_exp_map_consistency():
    N = 100
    eps = 1e-4

    v = torch.randn((N, 6))
    T = tf.se3_exp_map(v, eps=eps)
    v_recovered = tf.se3_log_map(T, eps=eps)

    omega = v[:, 3:]

    rot_mag_from_omega = torch.linalg.norm(omega, dim=-1)
    mask = (rot_mag_from_omega > 0.005) & (rot_mag_from_omega < 3.1415 - 0.005)

    assert torch.allclose(v[mask], v_recovered[mask], atol=eps * 4)

    T_recovered = tf.se3_exp_map(v_recovered, eps=eps)
    # but with the mask we should get better results
    assert torch.allclose(T[mask], T_recovered[mask], atol=eps * 4)


def test_adjoint():
    N = 100
    eps = 1e-5
    perturbation_mag = 0.2
    # for perturbations around a pose X, the following should be equivalent:
    # for notation, let v = (t, omega) be a perturbation in se3
    # X Exp(v in X frame) = Exp(v in global frame) X
    # where v in global frame is Ad(X) v in X frame
    pose = tf.Transform3d(rot=tf.random_rotations(1), pos=torch.randn((1, 3)))

    perturbations = torch.randn((N, 6)) * perturbation_mag
    X = pose.get_matrix()
    R, t = pose.get_RT()
    perturbations_global = tf.perturbations_global(R, t, perturbations)

    perturbed_pose_in_X = X @ tf.se3_exp_map(perturbations)
    perturbed_pose_global = tf.se3_exp_map(perturbations_global) @ X

    assert torch.allclose(perturbed_pose_in_X, perturbed_pose_global, atol=eps)


if __name__ == "__main__":
    test_compose()
    test_transform()
    test_translations()
    test_rotate_axis_angle()
    test_rotate()
    test_euler()
    test_quaternions()
    test_so3_exp_map_consistency()
    test_se3_exp_map_consistency()
    test_adjoint()
