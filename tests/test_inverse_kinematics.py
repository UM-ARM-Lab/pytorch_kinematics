import os
from timeit import default_timer as timer

import torch

import pytorch_kinematics as pk
import pytorch_seed

import pybullet as p
import pybullet_data

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


def test_jacobian_follower():
    pytorch_seed.seed(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    urdf = "kuka_iiwa/model.urdf"
    search_path = pybullet_data.getDataPath()
    full_urdf = os.path.join(search_path, urdf)
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    chain = chain.to(device=device)

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # world frame goal
    M = 1000
    # generate random goal joint angles (so these are all achievable)
    # use the joint limits to generate random joint angles
    lim = torch.tensor(chain.get_joint_limits(), device=device)
    goal_q = torch.rand(M, 7, device=device) * (lim[1] - lim[0]) + lim[0]

    # get ee pose (in robot frame)
    goal_in_rob_frame_tf = chain.forward_kinematics(goal_q)

    # transform to world frame for visualization
    goal_tf = rob_tf.compose(goal_in_rob_frame_tf)
    goal = goal_tf.get_matrix()
    goal_pos = goal[..., :3, 3]
    goal_rot = pk.matrix_to_euler_angles(goal[..., :3, :3], "XYZ")

    ik = pk.PseudoInverseIK(chain, max_iterations=30, num_retries=10,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            # line_search=pk.BacktrackingLineSearch(max_lr=0.2),
                            debug=False,
                            lr=0.2)

    # do IK
    timer_start = timer()
    sol = ik.solve(goal_in_rob_frame_tf)
    timer_end = timer()
    print("IK took %f seconds" % (timer_end - timer_start))
    print("IK converged number: %d / %d" % (sol.converged.sum(), sol.converged.numel()))
    print("IK took %d iterations" % sol.iterations)
    print("IK solved %d / %d goals" % (sol.converged_any.sum(), M))

    # visualize everything
    if visualize:
        p.connect(p.GUI)
        p.setRealTimeSimulation(False)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(search_path)

        yaw = 90
        pitch = -55
        dist = 1.
        target = goal_pos[0].cpu().numpy()
        p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        m = rob_tf.get_matrix()
        pos = m[0, :3, 3]
        rot = m[0, :3, :3]
        quat = pk.matrix_to_quaternion(rot)
        pos = pos.cpu().numpy()
        rot = pk.wxyz_to_xyzw(quat).cpu().numpy()
        armId = p.loadURDF(urdf, basePosition=pos, baseOrientation=rot, useFixedBase=True)

        _make_robot_translucent(armId, alpha=0.6)
        # p.resetBasePositionAndOrientation(armId, [0, 0, 0], [0, 0, 0, 1])
        # draw goal
        # place a translucent sphere at the goal
        show_max_num_retries_per_goal = 10
        for goal_num in range(M):
            # draw cone to indicate pose instead of sphere
            visId = p.createVisualShape(p.GEOM_MESH, fileName="meshes/cone.obj", meshScale=1.0,
                                        rgbaColor=[0., 1., 0., 0.5])
            # visId = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0., 1., 0., 0.5])
            r = goal_rot[goal_num]
            xyzw = pk.wxyz_to_xyzw(pk.matrix_to_quaternion(pk.euler_angles_to_matrix(r, "XYZ")))
            goalId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId,
                                       basePosition=goal_pos[goal_num].cpu().numpy(),
                                       baseOrientation=xyzw.cpu().numpy())

            solutions = sol.solutions[goal_num]
            # sort based on if they converged
            converged = sol.converged[goal_num]
            idx = torch.argsort(converged.to(int), descending=True)
            solutions = solutions[idx]

            # print how many retries converged for this one
            print("Goal %d converged %d / %d" % (goal_num, converged.sum(), converged.numel()))

            for i, q in enumerate(solutions):
                if i > show_max_num_retries_per_goal:
                    break
                for dof in range(q.shape[0]):
                    p.resetJointState(armId, dof, q[dof])
                input("Press enter to continue")

            p.removeBody(goalId)

        while True:
            p.stepSimulation()


if __name__ == "__main__":
    test_jacobian_follower()
