import os
import pathlib

import numpy as np

import pytorch_kinematics as pk

# Find all files named "scene*.xml" in the "mujoco_menagerie" directory
_MENAGERIE_ROOT = pathlib.Path(__file__).parent / 'mujoco_menagerie'
_XMLS_AND_BODIES = {
    # 'agility_cassie/scene.xml':            'cassie-pelvis',  # not supported because it has a ball joint
    'anybotics_anymal_b/scene.xml':        'base',
    'anybotics_anymal_c/scene.xml':        'base',
    'franka_emika_panda/scene.xml':        'link0',
    'google_barkour_v0/scene.xml':         'chassis',
    'google_barkour_v0/scene_barkour.xml': 'chassis',
    # 'hello_robot_stretch/scene.xml':       'base_link', # not supported because it has composite joints
    'kuka_iiwa_14/scene.xml':              'base',
    'rethink_robotics_sawyer/scene.xml':   'base',
    'robotiq_2f85/scene.xml':              'base_mount',
    'robotis_op3/scene.xml':               'body_link',
    'shadow_hand/scene_left.xml':          'lh_forearm',
    'shadow_hand/scene_right.xml':         'rh_forearm',
    'ufactory_xarm7/scene.xml':            'link_base',
    'unitree_a1/scene.xml':                'trunk',
    'unitree_go1/scene.xml':               'trunk',
    'universal_robots_ur5e/scene.xml':     'base',
    'wonik_allegro/scene_left.xml':        'palm',
    'wonik_allegro/scene_right.xml':       'palm',
}


def test_menagerie():
    for xml_filename, body in _XMLS_AND_BODIES.items():
        xml_filename = _MENAGERIE_ROOT / xml_filename
        xml_dir = xml_filename.parent
        # Menagerie files assume the current working directory is the directory of the scene.xml
        os.chdir(xml_dir)
        with xml_filename.open('r') as f:
            xml = f.read()
        chain = pk.build_chain_from_mjcf(xml, body)
        print(xml_filename)
        print("=" * 32)
        print(f"\t {chain.get_frame_names()}")
        print(f"\t {chain.get_joint_parameter_names()}")
        th = np.zeros(len(chain.get_joint_parameter_names()))
        fk_dict = chain.forward_kinematics(th)


if __name__ == '__main__':
    test_menagerie()
