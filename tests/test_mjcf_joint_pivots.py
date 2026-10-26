"""Compare imported MJCF body frames with the simulator's kinematics."""
import numpy as np
import pytest
import torch

import pytorch_kinematics as pk

mujoco = pytest.importorskip("mujoco")


def make_model(joint_type, pivot, rotated):
    orientation = "0.1 -0.2 0.3" if rotated else "0 0 0"
    return f'''<mujoco><compiler angle="radian"/><worldbody>
      <body name="link" pos="0.5 0.25 0.1" euler="{orientation}">
        <joint name="joint" type="{joint_type}" axis="0 0 1"
               pos="{pivot}" range="-2 2"/>
        <geom type="sphere" size="0.05" mass="1"/>
        <site name="tip" pos="0.4 0 0"/>
      </body></worldbody></mujoco>'''


def reference(model, configurations, name):
    data = mujoco.MjData(model)
    poses, jacobians = [], []
    for configuration in configurations:
        data.qpos[:] = configuration
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        pose = np.eye(4)
        if name == "tip":
            index = model.site(name).id
            pose[:3, 3] = data.site_xpos[index]
            pose[:3, :3] = data.site_xmat[index].reshape(3, 3)
            mujoco.mj_jacSite(model, data, jacp, jacr, index)
        else:
            index = model.body(name).id
            pose[:3, 3] = data.xpos[index]
            pose[:3, :3] = data.xmat[index].reshape(3, 3)
            mujoco.mj_jacBody(model, data, jacp, jacr, index)
        poses.append(pose)
        jacobians.append(np.vstack((jacp, jacr)))
    return np.stack(poses), np.stack(jacobians)


@pytest.fixture(params=["hinge", "slide"])
def joint_type(request):
    return request.param


@pytest.fixture(params=["0 0 0", "0.2 -0.1 0"])
def pivot(request):
    return request.param


@pytest.fixture(params=[False, True])
def rotated(request):
    return request.param


@pytest.mark.parametrize("name", ["link", "tip"])
def test_mjcf_pivot_poses_and_jacobians(joint_type, pivot, rotated, name):
    xml = make_model(joint_type, pivot, rotated)
    model = mujoco.MjModel.from_xml_string(xml)
    chain = pk.build_chain_from_mjcf(xml)
    serial = pk.SerialChain(chain, name).to(dtype=torch.float64)
    chain.to(dtype=torch.float64)
    configurations = [[0.0], [0.7], [-0.4]]
    expected_poses, expected_jacobians = reference(model, configurations, name)
    q = torch.tensor(configurations, dtype=torch.float64)
    poses = chain.forward_kinematics(q)[name].get_matrix()
    np.testing.assert_allclose(poses.numpy(), expected_poses, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(serial.forward_kinematics(q).get_matrix().numpy(),
                               expected_poses, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(serial.jacobian(q).numpy(), expected_jacobians,
                               atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("analytical_grad", [False, True])
@pytest.mark.parametrize("name", ["link", "tip"])
def test_mjcf_pivot_gradients(joint_type, pivot, rotated, analytical_grad, name):
    xml = make_model(joint_type, pivot, rotated)
    model = mujoco.MjModel.from_xml_string(xml)
    chain = pk.build_chain_from_mjcf(xml).to(dtype=torch.float64)
    configurations = [[0.0], [0.7], [-0.4]]
    _, expected_jacobians = reference(model, configurations, name)
    q = torch.tensor(configurations, dtype=torch.float64, requires_grad=True)
    poses = chain.forward_kinematics_tensor(q, analytical_grad=analytical_grad)
    positions = poses[chain.frame_to_idx[name], :, :3, 3]
    weights = torch.tensor([0.3, -0.5, 0.7], dtype=torch.float64)
    gradient, = torch.autograd.grad((positions * weights).sum(), q)
    expected = np.einsum("i,bij->bj", weights.numpy(), expected_jacobians[:, :3])
    np.testing.assert_allclose(gradient.numpy(), expected, atol=2e-7, rtol=2e-7)


BRANCHED_XML = '''<mujoco><compiler angle="radian"/><worldbody>
  <body name="base" pos="0.5 -0.25 0.1" euler="0.1 -0.2 0.3">
    <body name="link" pos="0.2 0.1 0.3" euler="-0.2 0.3 0.1">
      <joint name="shoulder" axis="1 2 -1" pos="0.2 -0.1 0.05" range="-2 2"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="forearm" pos="0.4 -0.2 0.1" euler="0.4 0.2 -0.1">
        <joint name="elbow" axis="-1 1 2" pos="-0.1 0.2 0.15" range="-2 2"/>
        <geom type="sphere" size="0.05" mass="1"/>
        <site name="tip" pos="0.3 -0.1 0.2"/>
      </body>
    </body>
    <body name="branch" pos="0.1 0.2 -0.1">
      <joint name="slider" type="slide" axis="1 -1 2" pos="0.3 0.2 -0.1" range="-2 2"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </body></worldbody></mujoco>'''


def configurations_for(model, chain, q):
    configurations = np.zeros((len(q), model.nq))
    for column, name in enumerate(chain.get_joint_parameter_names()):
        configurations[:, model.joint(name).qposadr[0]] = q[:, column].detach().numpy()
    return configurations


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mjcf_branched_pivots_and_tool_points(dtype):
    model = mujoco.MjModel.from_xml_string(BRANCHED_XML)
    chain = pk.build_chain_from_mjcf(BRANCHED_XML).to(dtype=dtype, device="cpu")
    q = torch.tensor([[0.0, 0.0, 0.0], [0.3, -0.5, 0.2], [-0.6, 0.4, -0.1]], dtype=dtype)
    configurations = configurations_for(model, chain, q)
    poses = chain.forward_kinematics(q)
    # Body and site names/topology stay unchanged: no synthetic pivot frames.
    assert set(poses) == {"world", "base", "link", "forearm", "branch", "tip"}
    for name in ["base", "link", "forearm", "branch", "tip"]:
        expected, _ = reference(model, configurations, name)
        np.testing.assert_allclose(poses[name].get_matrix().numpy(), expected, atol=4e-7, rtol=4e-7)

    for name in ["link", "forearm", "branch", "tip"]:
        serial = pk.SerialChain(chain, name, dtype=dtype)
        columns = [chain.get_joint_parameter_names().index(n) for n in serial.get_joint_parameter_names()]
        model_columns = [model.joint(n).dofadr[0] for n in serial.get_joint_parameter_names()]
        expected, jac = reference(model, configurations, name)
        actual_jac, actual_pose = serial.jacobian(q[:, columns], ret_eef_pose=True)
        np.testing.assert_allclose(actual_pose.numpy(), expected, atol=4e-7, rtol=4e-7)
        np.testing.assert_allclose(actual_jac.numpy(), jac[:, :, model_columns], atol=4e-7, rtol=4e-7)

    serial = pk.SerialChain(chain, "forearm", dtype=dtype)
    columns = [chain.get_joint_parameter_names().index(n) for n in serial.get_joint_parameter_names()]
    model_columns = [model.joint(n).dofadr[0] for n in serial.get_joint_parameter_names()]
    location = torch.tensor([0.1, -0.2, 0.3], dtype=dtype)
    tool_jac, tool_pose = serial.jacobian(q[:, columns], locations=location, ret_eef_pose=True)
    data = mujoco.MjData(model)
    for i, configuration in enumerate(configurations):
        data.qpos[:] = configuration
        mujoco.mj_forward(model, data)
        index = model.body("forearm").id
        point = data.xpos[index] + data.xmat[index].reshape(3, 3) @ location.numpy()
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, point, index)
        np.testing.assert_allclose(tool_pose[i, :3, 3].numpy(), point, atol=4e-7, rtol=4e-7)
        np.testing.assert_allclose(tool_jac[i].numpy(), np.vstack((jacp, jacr))[:, model_columns],
                                   atol=4e-7, rtol=4e-7)


@pytest.mark.parametrize("analytical_grad", [False, True])
def test_mjcf_branched_pivot_gradcheck(analytical_grad):
    chain = pk.build_chain_from_mjcf(BRANCHED_XML).to(dtype=torch.float64, device="cpu")
    q = torch.tensor([[0.3, -0.5, 0.2], [-0.6, 0.4, -0.1]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda angles: chain.forward_kinematics_tensor(angles, analytical_grad=analytical_grad),
        (q,), atol=1e-6, rtol=1e-4)


def test_frame_transform_about_pivot():
    pivot = torch.tensor([0.25, -0.125, 0.5], dtype=torch.float64)
    joint = pk.Joint("hinge", joint_type="revolute", pivot=pivot)
    frame = pk.Frame("body", joint=joint).to(dtype=torch.float64, device="cpu")
    q = torch.tensor([0.0, torch.pi / 2], dtype=torch.float64)
    transforms = frame.get_transform(q).get_matrix()
    expected = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    expected[1, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    expected[:, :3, 3] = pivot - (expected[:, :3, :3] @ pivot)
    torch.testing.assert_close(transforms, expected)
    assert frame.joint.pivot.dtype == torch.float64


def test_mjcf_pivot_serial_subchain():
    model = mujoco.MjModel.from_xml_string(BRANCHED_XML)
    chain = pk.build_chain_from_mjcf(BRANCHED_XML)
    serial = pk.SerialChain(chain, "tip", root_frame_name="link").to(dtype=torch.float64)
    assert serial.get_joint_parameter_names() == ["elbow"]
    qs = [[0.0], [0.4], [-0.6]]
    configurations = np.zeros((3, model.nq))
    configurations[:, model.joint("shoulder").qposadr[0]] = 0.7
    configurations[:, model.joint("elbow").qposadr[0]] = np.asarray(qs)[:, 0]
    base_poses, _ = reference(model, configurations, "link")
    tip_poses, _ = reference(model, configurations, "tip")
    expected = np.linalg.solve(base_poses, tip_poses)
    actual = serial.forward_kinematics(torch.tensor(qs, dtype=torch.float64)).get_matrix()
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7, rtol=3e-7)


def test_mjcf_pivot_ik_reaches_simulator_pose(joint_type):
    xml = make_model(joint_type, "0.2 -0.1 0", True)
    model = mujoco.MjModel.from_xml_string(xml)
    chain = pk.build_serial_chain_from_mjcf(xml, "tip").to(dtype=torch.float64)
    target, _ = reference(model, [[0.7]], "tip")
    solver = pk.PseudoInverseIK(
        chain, pos_tolerance=1e-6, rot_tolerance=1e-6,
        retry_configs=torch.zeros((1, 1), dtype=torch.float64), max_iterations=50,
        lr=1.0, early_stopping_any_converged=True, early_stopping_no_improvement=None,
        enforce_joint_limits=False, clamp_to_limits=True,
        num_limit_refinement_iterations=0, use_compile=False)
    result = solver.solve(pk.Transform3d(matrix=torch.from_numpy(target)))
    assert result.converged.all()
    achieved, _ = reference(model, result.solutions.detach().numpy().reshape(-1, 1), "tip")
    np.testing.assert_allclose(achieved, target, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("pivot", ["0 0 0", "0.2 -0.1 0"])
def test_mjcf_pivot_compiled_fk_jacobian_backward(pivot):
    chain = pk.build_serial_chain_from_mjcf(make_model("hinge", pivot, True), "tip").to(dtype=torch.float64)
    q = torch.tensor([[0.0], [0.7], [-0.4]], dtype=torch.float64, requires_grad=True)
    expected_fk = chain.forward_kinematics_tensor(q)
    expected_jac = chain.jacobian_tensor(q)
    expected_grad, = torch.autograd.grad(expected_fk.square().sum(), q)
    compiled_fk = torch.compile(chain.forward_kinematics_tensor, fullgraph=True)(q)
    compiled_jac = torch.compile(chain.jacobian_tensor, fullgraph=True)(q)
    compiled_grad, = torch.autograd.grad(compiled_fk.square().sum(), q)
    torch.testing.assert_close(compiled_fk, expected_fk)
    torch.testing.assert_close(compiled_jac, expected_jac)
    torch.testing.assert_close(compiled_grad, expected_grad)
