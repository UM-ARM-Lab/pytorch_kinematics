import numpy as np

import pytorch_kinematics as pk


def test_shadow_hand():
    with open('right_hand.xml') as f:
        xml = f.read()
    chain = pk.build_chain_from_mjcf(xml)
    print(chain.get_frame_names())
    print(chain.get_joint_parameter_names())
    th = np.zeros(len(chain.get_joint_parameter_names()))
    fk_dict = chain.forward_kinematics(th, end_only=True)
    print(fk_dict['rh_lfproximal'])
    print(fk_dict['rh_lfdistal'])
