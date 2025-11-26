import torch

import pytorch_kinematics as pk
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
    assert torch.allclose(
        tf.quaternion_to_matrix(quat_identity),
        torch.eye(3, dtype=torch.float64).repeat(N, 1, 1),
    )


def test_translations():
    t = tf.Translate(1, 2, 3)
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(1, 3, 3)
    points_out = t.transform_points(points)
    points_out_expected = torch.tensor([[2.0, 2.0, 3.0], [1.0, 3.0, 3.0], [1.5, 2.5, 3.0]]).view(1, 3, 3)
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
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]).view(1, 3, 3)
    normals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).view(1, 3, 3)
    points_out = t.transform_points(points)
    normals_out = t.transform_normals(normals)
    points_out_expected = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]).view(1, 3, 3)
    normals_out_expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]).view(1, 3, 3)
    assert torch.allclose(points_out, points_out_expected)
    assert torch.allclose(normals_out, normals_out_expected)


def test_rotate():
    R = tf.so3_exp_map(torch.randn((1, 3)))
    t = tf.Transform3d().rotate(R)
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(1, 3, 3)
    normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]).view(1, 3, 3)
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
    sxyz_matrix = torch.tensor(
        [
            [
                0.87758256,
                -0.47942554,
                0.0,
                0.0,
            ],
            [0.25903472, 0.47415988, -0.84147098, 0.0],
            [0.40342268, 0.73846026, 0.54030231, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(sxyz_matrix, t.get_matrix())


def test_quaternions():
    import pytorch_seed

    pytorch_seed.seed(0)

    n = 10
    q = tf.random_quaternions(n)
    q_tf = tf.wxyz_to_xyzw(q)
    assert torch.allclose(q, tf.xyzw_to_wxyz(q_tf))

    qq = pk.standardize_quaternion(q)
    assert torch.allclose(qq.norm(dim=-1), torch.ones(n))

    # random quaternions should already be unit quaternions
    assert torch.allclose(q, qq)

    # distances to themselves should be zero
    d = pk.quaternion_angular_distance(q, q)
    assert torch.allclose(d, torch.zeros(n))
    # q = -q
    d = pk.quaternion_angular_distance(q, -q)
    assert torch.allclose(d, torch.zeros(n))

    axis = torch.tensor([0.0, 0.5, 0.5])
    axis = axis / axis.norm()
    magnitudes = torch.tensor([2.32, 1.56, -0.52, 0.1])
    n = len(magnitudes)
    aa_1 = axis.repeat(n, 1)
    aa_2 = axis * magnitudes[:, None]
    q1 = pk.axis_angle_to_quaternion(aa_1)
    q2 = pk.axis_angle_to_quaternion(aa_2)
    d = pk.quaternion_angular_distance(q1, q2)
    expected_d = (magnitudes - 1).abs()
    assert torch.allclose(d, expected_d, atol=1e-4)


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


def test_quaternion_slerp():
    q = tf.random_quaternions(20)
    q1 = q[:10]
    q2 = q[10:]
    t = torch.rand(10)
    q_interp = pk.quaternion_slerp(q1, q2, t)
    # check the distance between them is consistent
    full_dist = pk.quaternion_angular_distance(q1, q2)
    interp_dist = pk.quaternion_angular_distance(q1, q_interp)
    # print(f"full_dist: {full_dist} interp_dist: {interp_dist} t: {t}")
    assert torch.allclose(full_dist * t, interp_dist, atol=1e-5)


if __name__ == "__main__":
    test_compose()
    test_transform()
    test_translations()
    test_rotate_axis_angle()
    test_rotate()
    test_euler()
    test_quaternions()
    test_quaternion_slerp()
