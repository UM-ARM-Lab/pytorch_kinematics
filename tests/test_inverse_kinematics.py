import os
from timeit import default_timer as timer

import numpy as np
import pybullet as p
import pybullet_data
import pytorch_seed
import torch

import pytorch_kinematics as pk


visualize = False


def _make_robot_translucent(robot_id, alpha=0.4):
    def make_transparent(link):
        link_id = link[1]
        rgba = list(link[7])
        rgba[3] = alpha
        p.changeVisualShape(robot_id, link_id, rgbaColor=rgba)

    visual_data = p.getVisualShapeData(robot_id)
    for link in visual_data:
        make_transparent(link)


def create_test_chain(robot="kuka_iiwa", device="cpu"):
    if robot == "kuka_iiwa":
        urdf = "kuka_iiwa/model.urdf"
        search_path = pybullet_data.getDataPath()
        full_urdf = os.path.join(search_path, urdf)
        chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
        chain = chain.to(device=device)
    elif robot == "widowx":
        urdf = "widowx/wx250s.urdf"
        full_urdf = urdf
        chain = pk.build_serial_chain_from_urdf(open(full_urdf, "rb").read(), "ee_gripper_link")
        chain = chain.to(device=device)
    else:
        raise NotImplementedError(f"Robot {robot} not implemented")
    return chain, urdf


def test_jacobian_follower(robot="kuka_iiwa"):
    pytorch_seed.seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search_path = pybullet_data.getDataPath()
    chain, urdf = create_test_chain(robot=robot, device=device)

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # world frame goal
    M = 1000
    # generate random goal joint angles (so these are all achievable)
    # use the joint limits to generate random joint angles
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    goal_q = torch.rand(M, lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

    # transform to world frame for visualization
    goal_tf = rob_tf.compose(goal_in_rob_frame_tf)
    goal = goal_tf.get_matrix()
    goal_pos = goal[..., :3, 3]
    goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    num_retries = 10
    ik = pk.PseudoInverseIK(
        chain,
        max_iterations=30,
        num_retries=num_retries,
        joint_limits=lim.T,
        early_stopping_any_converged=True,
        early_stopping_no_improvement="all",
        # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
        debug=False,
        lr=0.2,
    )

    # do IK
    timer_start = timer()
    sol = ik.solve(goal_in_rob_frame_tf)
    timer_end = timer()
    print("IK took %f seconds" % (timer_end - timer_start))
    print("IK converged number: %d / %d" % (sol.converged.sum(), sol.converged.numel()))
    print("IK took %d iterations" % sol.iterations)
    print("IK solved %d / %d goals" % (sol.converged_any.sum(), M))

    # check that solving again produces the same solutions
    sol_again = ik.solve(goal_in_rob_frame_tf)
    assert torch.allclose(sol.solutions, sol_again.solutions)
    assert torch.allclose(sol.converged, sol_again.converged)

    # visualize everything
    if visualize:
        p.connect(p.GUI)
        p.setRealTimeSimulation(False)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(search_path)

        yaw = 90
        pitch = -65
        # dist = 1.
        dist = 2.4
        target = np.array([2.0, 1.5, 0])
        p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

        plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])

        # make 1 per retry with positional offsets
        robots = []
        num_robots = 16
        # 4x4 grid position offset
        offset = 1.0
        m = rob_tf.get_matrix()
        pos = m[0, :3, 3]
        rot = m[0, :3, :3]
        quat = pk.matrix_to_quaternion(rot)
        pos = pos.cpu().numpy()
        rot = pk.wxyz_to_xyzw(quat).cpu().numpy()

        for i in range(num_robots):
            this_offset = np.array([i % 4 * offset, i // 4 * offset, 0])
            armId = p.loadURDF(
                urdf,
                basePosition=pos + this_offset,
                baseOrientation=rot,
                useFixedBase=True,
            )
            # _make_robot_translucent(armId, alpha=0.6)
            robots.append({"id": armId, "offset": this_offset, "pos": pos})

        show_max_num_retries_per_goal = 10

        goals = []
        # draw cone to indicate pose instead of sphere
        visId = p.createVisualShape(
            p.GEOM_MESH,
            fileName="meshes/cone.obj",
            meshScale=1.0,
            rgbaColor=[0.0, 1.0, 0.0, 0.5],
        )
        for _ in range(num_robots):
            goals.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId))

        try:
            import window_recorder

            with window_recorder.WindowRecorder(save_dir="."):
                # batch over goals with num_robots
                for j in range(0, M, num_robots):
                    this_selection = slice(j, j + num_robots)
                    r = goal_rot[this_selection]
                    xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(r, "XYZ")))

                    solutions = sol.solutions[this_selection, :, :]
                    converged = sol.converged[this_selection, :]

                    # print how many retries converged for this one
                    print("Goal %d to %d converged %d / %d" % (j, j + num_robots, converged.sum(), converged.numel()))

                    # outer loop over retries, inner loop over goals (for each robot shown in parallel)
                    for ii in range(num_retries):
                        if ii > show_max_num_retries_per_goal:
                            break
                        for jj in range(num_robots):
                            p.resetBasePositionAndOrientation(
                                goals[jj],
                                goal_pos[j + jj].cpu().numpy() + robots[jj]["offset"],
                                xyzw[jj].cpu().numpy(),
                            )
                            armId = robots[jj]["id"]
                            q = solutions[jj, ii, :]
                            for dof in range(q.shape[0]):
                                p.resetJointState(armId, dof, q[dof])

                        input("Press enter to continue")
        except ImportError:
            print("pip install window_recorder")

        while True:
            p.stepSimulation()


def test_ik_in_place_no_err(robot="kuka_iiwa"):
    pytorch_seed.seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    chain, urdf = create_test_chain(robot=robot, device=device)
    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # goal equal to current configuration
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    cur_q = torch.rand(lim.shape[1], device=device) * (lim[1] - lim[0]) + lim[0]
    M = 1
    goal_q = cur_q.unsqueeze(0).repeat(M, 1)

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

    # transform to world frame for visualization
    goal_tf = rob_tf.compose(goal_in_rob_frame_tf)
    goal = goal_tf.get_matrix()
    goal[..., :3, 3]
    pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    ik = pk.PseudoInverseIK(
        chain,
        max_iterations=30,
        num_retries=10,
        joint_limits=lim.T,
        early_stopping_any_converged=True,
        early_stopping_no_improvement="all",
        retry_configs=cur_q.reshape(1, -1),
        # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
        debug=False,
        lr=0.2,
    )

    # do IK
    sol = ik.solve(goal_in_rob_frame_tf)
    assert sol.converged.sum() == M
    assert torch.allclose(sol.solutions[0][0], cur_q)
    assert torch.allclose(sol.err_pos[0], torch.zeros(1, device=device), atol=1e-6)
    assert torch.allclose(sol.err_rot[0], torch.zeros(1, device=device), atol=1e-6)


if __name__ == "__main__":
    print("Testing kuka_iiwa IK")
    test_jacobian_follower(robot="kuka_iiwa")
    test_ik_in_place_no_err(robot="kuka_iiwa")
    print("Testing widowx IK")
    test_jacobian_follower(robot="widowx")
    test_ik_in_place_no_err(robot="widowx")
