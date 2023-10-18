import math
import os
from timeit import default_timer as timer

import torch

import pytorch_kinematics as pk

import pybullet as p
import pybullet_data

TEST_DIR = os.path.dirname(__file__)

visualize = True


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    urdf = "kuka_iiwa/model.urdf"
    search_path = pybullet_data.getDataPath()
    full_urdf = os.path.join(search_path, urdf)
    chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "lbr_iiwa_link_7")
    chain = chain.to(device=device)

    joints_high = torch.tensor([170, 120, 170, 120, 170, 120, 175], device=device)
    joint_limits = torch.stack((-joints_high, joints_high), dim=-1) * math.pi / 180.0
    ik = pk.PseudoInverseIK(chain, max_iterations=100, num_retries=10000, joint_limits=joint_limits)

    # robot frame
    pos = torch.tensor([0.0, 0.0, 0.0], device=device)
    rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    rob_tf = pk.Transform3d(pos=pos, rot=rot, device=device)

    # world frame goal
    goal_pos = torch.tensor([0.0, 0.5, 0.5], device=device)
    goal_rot = torch.tensor([0.0, 0.0, 0.0], device=device)
    goal_tf = pk.Transform3d(pos=goal_pos, rot=goal_rot, device=device)

    # transform to robot frame
    goal_in_rob_frame_tf = rob_tf.inverse().compose(goal_tf)

    # do IK
    timer_start = timer()
    sol = ik.solve(goal_in_rob_frame_tf)
    timer_end = timer()
    print("IK took %f seconds" % (timer_end - timer_start))
    print("IK converged number: %d / %d" % (sol.converged.sum(), sol.converged.shape[0]))

    # visualize everything
    if visualize:
        p.connect(p.GUI)
        p.setRealTimeSimulation(False)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(search_path)
        p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
        m = rob_tf.get_matrix()
        pos = m[0, :3, 3]
        rot = m[0, :3, :3]
        quat = pk.matrix_to_quaternion(rot)
        armId = p.loadURDF(urdf, basePosition=pos.cpu().numpy(), baseOrientation=pk.wxyz_to_xyzw(quat).cpu().numpy(),
                           useFixedBase=True)
        _make_robot_translucent(armId, alpha=0.6)
        # p.resetBasePositionAndOrientation(armId, [0, 0, 0], [0, 0, 0, 1])
        # draw goal
        # place a translucent sphere at the goal
        visId = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0., 1., 0., 0.5])
        goalId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visId, basePosition=goal_pos.cpu().numpy())

        yaw = 90
        pitch = -55
        dist = 1.
        target = goal_pos.cpu().numpy()
        p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

        for i, q in enumerate(sol.solutions):
            input("Press enter to continue")
            for dof in range(q.shape[0]):
                p.resetJointState(armId, dof, q[dof])

        while True:
            p.stepSimulation()


if __name__ == "__main__":
    test_jacobian_follower()