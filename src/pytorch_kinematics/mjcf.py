from dm_control import mjcf

import pytorch_kinematics.transforms as tf
from . import chain
from . import frame

JOINT_TYPE_MAP = {'hinge': 'revolute', "slide": "prismatic"}


def geoms_to_visuals(geom):
    visuals = []
    for g in geom:
        if g.type == 'capsule':
            param = (g.size[0], g.fromto)
        elif g.type == 'sphere':
            param = g.size[0]
        elif g.type == 'mesh':
            param = None
        else:
            raise ValueError('Invalid geometry type %s.' % g.type)
        visuals.append(frame.Visual(offset=tf.Transform3d(rot=g.quat, pos=g.pos), geom_type=g.type, geom_param=param))
    return visuals


def _build_chain_recurse(parent_frame, parent_body):
    parent_frame.link.visuals = geoms_to_visuals(parent_body.geom)
    for b in parent_body.body:
        n_joints = len(b.joint)
        if n_joints > 1:
            raise ValueError("composite joints not supported (could implement this if needed)")
        if n_joints == 1:
            joint = b.joint[0]
            child_joint = frame.Joint(joint.name, tf.Transform3d(pos=joint.pos), axis=joint.axis,
                                      joint_type=JOINT_TYPE_MAP[joint.type])
        else:
            child_joint = frame.Joint(b.name + "_imaginary_fixed_joint")
        child_link = frame.Link(b.name, offset=tf.Transform3d(rot=b.quat, pos=b.pos))
        child_frame = frame.Frame(name=b.name, link=child_link, joint=child_joint)
        parent_frame.children = parent_frame.children + (child_frame,)
        _build_chain_recurse(child_frame, b)

    for site in parent_body.site:
        site_link = frame.Link(site.name, offset=tf.Transform3d(rot=site.quat, pos=site.pos))
        site_frame = frame.Frame(name=site.name, link=site_link)
        parent_frame.children = parent_frame.children + (site_frame,)


def build_chain_from_mjcf(data):
    """
    Build a Chain object from MJCF data.

    Parameters
    ----------
    data : str
        MJCF string data.

    Returns
    -------
    chain.Chain
        Chain object created from MJCF.
    """
    model = mjcf.from_xml_string(data)
    root_body = model.worldbody.body[0]
    root_frame = frame.Frame(root_body.name + "_frame",
                             link=frame.Link(root_body.name,
                                             offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos)),
                             joint=frame.Joint())
    _build_chain_recurse(root_frame, root_body)
    return chain.Chain(root_frame)


def build_serial_chain_from_mjcf(data, end_link_name, root_link_name=""):
    """
    Build a SerialChain object from MJCF data.

    Parameters
    ----------
    data : str
        MJCF string data.
    end_link_name : str
        The name of the link that is the end effector.
    root_link_name : str, optional
        The name of the root link.

    Returns
    -------
    chain.SerialChain
        SerialChain object created from MJCF.
    """
    mjcf_chain = build_chain_from_mjcf(data)
    return chain.SerialChain(mjcf_chain, end_link_name,
                             "" if root_link_name == "" else root_link_name)
