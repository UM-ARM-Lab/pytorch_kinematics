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


if __name__ == "__main__":
    test_planar_rotating_arm()
