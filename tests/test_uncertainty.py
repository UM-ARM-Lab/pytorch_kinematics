import os
import math
import pybullet as p
import pybullet_data
import pytorch_kinematics as pk
import torch

TEST_DIR = os.path.dirname(__file__)


def test_planar_rotating_arm():
    cov_mag = 10 * math.pi / 180
    num_samples = 100

    urdf = os.path.join(TEST_DIR, "sequential_planar_arm.urdf")
    chain = pk.build_chain_from_urdf(open(urdf).read())

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    planeId = p.loadURDF("plane.urdf", [0, 0, 0.0], useFixedBase=True)
    armId = p.loadURDF(urdf, [0, 0, 0.05], useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=2.1, cameraYaw=0, cameraPitch=-85,
                                 cameraTargetPosition=[0.98, 0.2, -0.55])

    pose_vis_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 0.3])
    debug_objs = []

    no_cov = torch.zeros(6, 6)
    yaw_cov = no_cov.clone()
    yaw_cov[5, 5] = cov_mag

    def end_effector_perturbed_poses(joint_values, joint_covs):
        for i in range(4):
            p.resetJointState(armId, i, joint_values[i])

        # these are the mean pose estimates
        ret = chain.forward_kinematics(joint_values, return_relative_tsf=True)
        # convert to list
        eef = ret['eef']
        ret = [ret['link0_rel'], ret['link1_rel'], ret['link2_rel'], ret['link3_rel'], ret['eef_rel']]
        # determine covariance at end effector given:

        epsilon = 1e-6 * torch.eye(6)
        cov = None
        for i in range(5):
            if cov is None:
                cov = joint_covs[i]
            else:
                cov = pk.compose_uncertainty(ret[i], cov, joint_covs[i])
        cov = cov[0]

        # take samples from the covariance (0 mean)
        perturbations_in_X = torch.distributions.MultivariateNormal(torch.zeros(6), cov + epsilon).sample(
            (num_samples,))

        X = eef.get_matrix()
        perturbed_pose = X @ pk.se3_exp_map(perturbations_in_X)

        return perturbed_pose

    scenarios = [
        ([0., 0.4, 0.5, 0.3], [no_cov, yaw_cov, no_cov, no_cov, no_cov]),
        ([0., 0., 0., 0.], [no_cov, yaw_cov, yaw_cov, yaw_cov, no_cov]),
        ([0., 0.4, 0.5, 0.3], [no_cov, yaw_cov, yaw_cov, yaw_cov, no_cov]),
        ([0., 0.8, 0.8, 0.9], [no_cov, yaw_cov, yaw_cov, yaw_cov, no_cov]),
    ]

    for joint_values, joint_covs in scenarios:
        perturbed_pose = end_effector_perturbed_poses(joint_values, joint_covs)

        perturbed_eef_positions = perturbed_pose[:, :3, 3]
        if debug_objs:
            for i, pos in enumerate(perturbed_eef_positions):
                p.resetBasePositionAndOrientation(debug_objs[i], pos, [0, 0, 0, 1])
        else:
            for pos in perturbed_eef_positions:
                debug_objs.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=pose_vis_id, basePosition=pos))
        input("Press enter to continue")

    while True:
        p.stepSimulation()


def test_panda_arm():
    cov_mag = 1 * math.pi / 180
    num_samples = 100

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    planeId = p.loadURDF("plane.urdf", [0, 0, 0.0], useFixedBase=True)

    data_path = pybullet_data.getDataPath()
    robot_name = "franka_panda/panda.urdf"
    urdf = os.path.join(data_path, robot_name)
    print("Location of Panda's URDF: ", urdf)
    chain = pk.build_chain_from_urdf(open(urdf).read())

    armId = p.loadURDF(robot_name, useFixedBase=True)
    # p.resetDebugVisualizerCamera(cameraDistance=2.1, cameraYaw=0, cameraPitch=-85,
    #                              cameraTargetPosition=[0.98, 0.2, -0.55])

    pose_vis_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 0.3])
    debug_objs = []

    no_cov = torch.zeros(6, 6)
    # equally uncertain about all rotation axis let's say
    equi_rot_cov = no_cov.clone()
    equi_rot_cov[5, 5] = cov_mag
    equi_rot_cov[4, 4] = cov_mag
    equi_rot_cov[3, 3] = cov_mag

    def end_effector_perturbed_poses(joint_values, joint_covs):
        # joint_values = [0, 0.2, 0., -1.3, 0.1, 1.7, 0., 0., 0.]
        for i in range(len(joint_values)):
            p.resetJointState(armId, i, joint_values[i])

        # these are the mean pose estimates
        eef_name = "panda_grasptarget"
        eef_idx = chain.get_frame_indices(eef_name)
        ret = chain.forward_kinematics(joint_values, frame_indices=eef_idx, return_relative_tsf=True)
        eef = ret[eef_name]
        # convert to list
        parents = chain.parents_indices[eef_idx]
        parents = [chain.idx_to_frame[i.item()] for i in parents]
        rel_tsf = [ret[parent_frame + '_rel'] for parent_frame in parents]
        # determine covariance at end effector given:

        epsilon = 1e-6 * torch.eye(6)
        cov = None
        for i, rel_tsf in enumerate(rel_tsf):
            joint_cov = no_cov
            if parents[i] in joint_covs:
                joint_cov = joint_covs[parents[i]]

            if cov is None:
                cov = joint_cov
            else:
                cov = pk.compose_uncertainty(rel_tsf, cov, joint_cov)
        cov = cov[0]

        # take samples from the covariance (0 mean)
        perturbations_in_X = torch.distributions.MultivariateNormal(torch.zeros(6), cov + epsilon).sample(
            (num_samples,))

        X = eef.get_matrix()
        perturbed_pose = X @ pk.se3_exp_map(perturbations_in_X)

        return perturbed_pose

    scenarios = [
        # ([0, 0.4, 0., -1.3, 0.1, 1.7, 0., 0., 0.], {'panda_link1': equi_rot_cov * 3}),
        # try 1 degrees for joints 0-1 and 0.5 degrees for rest
        ([0, 0.4, 0., -1.3, 0.1, 1.7, 0., 0., 0.],
         {'panda_link0': equi_rot_cov, 'panda_link1': equi_rot_cov, 'panda_link2': equi_rot_cov * 0.5,
          'panda_link3': equi_rot_cov * 0.5, 'panda_link4': equi_rot_cov * 0.5, 'panda_link5': equi_rot_cov * 0.5,
          'panda_link6': equi_rot_cov * 0.5, 'panda_link7': equi_rot_cov * 0.5}),
        # ([0., 0., 0., 0.], [no_cov, equi_rot_cov, equi_rot_cov, equi_rot_cov, no_cov]),
        # ([0., 0.4, 0.5, 0.3], [no_cov, equi_rot_cov, equi_rot_cov, equi_rot_cov, no_cov]),
        # ([0., 0.8, 0.8, 0.9], [no_cov, equi_rot_cov, equi_rot_cov, equi_rot_cov, no_cov]),
    ]

    for joint_values, joint_covs in scenarios:
        perturbed_pose = end_effector_perturbed_poses(joint_values, joint_covs)

        perturbed_eef_positions = perturbed_pose[:, :3, 3]
        if debug_objs:
            for i, pos in enumerate(perturbed_eef_positions):
                p.resetBasePositionAndOrientation(debug_objs[i], pos, [0, 0, 0, 1])
        else:
            for pos in perturbed_eef_positions:
                debug_objs.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=pose_vis_id, basePosition=pos))
        input("Press enter to continue")

    while True:
        p.stepSimulation()


if __name__ == "__main__":
    # test_planar_rotating_arm()
    test_panda_arm()
