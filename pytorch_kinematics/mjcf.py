import numpy as np
from dm_control import mjcf

import pytorch_kinematics.transforms as tf
from . import chain
from . import frame

JOINT_TYPE_MAP = {'hinge': 'revolute'}


def geoms_to_visuals(geom, base=None):
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
        if base is None:
            visuals.append(
                frame.Visual(offset=tf.Transform3d(rot=g.quat, pos=g.pos), geom_type=g.type, geom_param=param))
        else:
            if g.pos is None and g.quat is None:
                visual = frame.Visual(offset=base, geom_type=g.type, geom_param=param)
            else:
                visual = frame.Visual(offset=base.compose(tf.Transform3d(rot=g.quat, pos=g.pos)), geom_type=g.type,
                                      geom_param=param)
            visuals.append(visual)
    return visuals


def body_to_link(body, base=None):
    if base is None:
        return frame.Link(body.name, offset=tf.Transform3d(rot=body.quat, pos=body.pos))
    else:
        return frame.Link(body.name, offset=tf.Transform3d(rot=body.quat, pos=body.pos))


def joint_to_joint(joint, base=None):
    if base is None:
        return frame.Joint(joint.name, offset=tf.Transform3d(pos=joint.pos), joint_type=JOINT_TYPE_MAP[joint.type],
                           axis=joint.axis)
    elif np.allclose(joint.pos, np.zeros(3)):
        return frame.Joint(joint.name, offset=base, joint_type=JOINT_TYPE_MAP[joint.type], axis=joint.axis)
    else:
        return frame.Joint(joint.name, offset=base.compose(tf.Transform3d(pos=joint.pos)),
                           joint_type=JOINT_TYPE_MAP[joint.type], axis=joint.axis)


def add_composite_joint(root_frame, joints, base=None):
    if len(joints) > 0:
        child_frame = frame.Frame(
            name=joints[0].name,
            link=frame.Link(name=root_frame.link.name + '_child'),
            joint=joint_to_joint(joints[0], base))
        root_frame.children = root_frame.children + (child_frame,)
        ret, offset = add_composite_joint(root_frame.children[-1], joints[1:])
        if root_frame.joint.offset is None:
            return ret, offset
        else:
            return ret, root_frame.joint.offset.compose(offset)
    else:
        return root_frame, root_frame.joint.offset


def _build_chain_recurse(root_frame, root_body):
    base = root_frame.link.offset
    cur_frame, cur_base = add_composite_joint(root_frame, root_body.joint, base)
    # if len(root_body.joint) > 0:
    #     cur_frame.link.visuals = geoms_to_visuals(root_body.geom, base)
    # else:
    #     cur_frame.link.visuals = geoms_to_visuals(root_body.geom)
    for b in root_body.body:
        cur_frame.children = cur_frame.children + (frame.Frame(),)
        next_frame = cur_frame.children[-1]
        next_frame.name = b.name + "_frame"
        next_frame.link = frame.Link(b.name, offset=tf.Transform3d(rot=b.quat, pos=b.pos))
        _build_chain_recurse(next_frame, b)
    for site in root_body.site:
        cur_frame.children = cur_frame.children + (frame.Frame(),)
        next_frame = cur_frame.children[-1]
        next_frame.name = site.name + "_frame"
        next_frame.link = frame.Link(site.name, offset=tf.Transform3d(rot=site.quat, pos=site.pos))


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
                             link=body_to_link(root_body),
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
    return chain.SerialChain(mjcf_chain, end_link_name + "_frame",
                             "" if root_link_name == "" else root_link_name + "_frame")
