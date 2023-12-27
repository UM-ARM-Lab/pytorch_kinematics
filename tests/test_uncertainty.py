import os
import pybullet as p
import pybullet_data

TEST_DIR = os.path.dirname(__file__)


def test_planar_rotating_arm():
    urdf = os.path.join(TEST_DIR, "sequential_planar_arm.urdf")

    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    planeId = p.loadURDF("plane.urdf", [0, 0, 0.0], useFixedBase=True)
    armId = p.loadURDF(urdf, [0, 0, 0.05], useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=2.1, cameraYaw=-6.4, cameraPitch=-77.4,
                                 cameraTargetPosition=[0.98, -0.13, -0.55])

    test_joint_vals = [0.1, 0.3, -0.2, 0.2]
    for i in range(4):
        p.resetJointState(armId, i, test_joint_vals[i])

    while True:
        p.stepSimulation()

    pass


if __name__ == "__main__":
    test_planar_rotating_arm()
