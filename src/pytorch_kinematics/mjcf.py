from typing import Union

import mujoco
from mujoco._structs import _MjModelBodyViews as MjModelBodyViews

import pytorch_kinematics.transforms as tf
from . import chain
from . import frame

# Converts from MuJoCo joint types to pytorch_kinematics joint types
JOINT_TYPE_MAP = {
    mujoco.mjtJoint.mjJNT_HINGE: 'revolute',
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic"
}


def body_to_geoms(m: mujoco.MjModel, body: MjModelBodyViews):
    # Find all geoms which have body as parent
    visuals = []
    for geom_id in range(m.ngeom):
        geom = m.geom(geom_id)
        if geom.bodyid == body.id:
            visuals.append(frame.Visual(offset=tf.Transform3d(rot=geom.quat, pos=geom.pos), geom_type=geom.type,
                                        geom_param=geom.size))
    return visuals


def _build_chain_recurse(m, parent_frame, parent_body):
    parent_frame.link.visuals = body_to_geoms(m, parent_body)
    # iterate through all bodies that are children of parent_body
    for body_id in range(m.nbody):
        body = m.body(body_id)
        if body.parentid == parent_body.id and body_id != parent_body.id:
            n_joints = body.jntnum
            if n_joints > 1:
                raise ValueError("composite joints not supported (could implement this if needed)")
            if n_joints == 1:
                # Find the joint for this body, again assuming there's only one joint per body.
                joint = m.joint(body.jntadr[0])
                joint_offset = tf.Transform3d(pos=joint.pos)
                child_joint = frame.Joint(joint.name, offset=joint_offset, axis=joint.axis,
                                          joint_type=JOINT_TYPE_MAP[joint.type[0]],
                                          limits=(joint.range[0], joint.range[1]))
            else:
                child_joint = frame.Joint(body.name + "_fixed_joint")
            child_link = frame.Link(body.name, offset=tf.Transform3d(rot=body.quat, pos=body.pos))
            child_frame = frame.Frame(name=body.name, link=child_link, joint=child_joint)
            parent_frame.children = parent_frame.children + [child_frame, ]
            _build_chain_recurse(m, child_frame, body)

    # iterate through all sites that are children of parent_body
    for site_id in range(m.nsite):
        site = m.site(site_id)
        if site.bodyid == parent_body.id:
            site_link = frame.Link(site.name, offset=tf.Transform3d(rot=site.quat, pos=site.pos))
            site_frame = frame.Frame(name=site.name, link=site_link)
            parent_frame.children = parent_frame.children + [site_frame, ]


def build_chain_from_mjcf(data, body: Union[None, str, int] = None):
    """
    Build a Chain object from MJCF data.

    Parameters
    ----------
    data : str
        MJCF string data.
    body : str or int, optional
        The name or index of the body to use as the root of the chain. If None, body idx=0 is used.

    Returns
    -------
    chain.Chain
        Chain object created from MJCF.
    """
    m = mujoco.MjModel.from_xml_string(data)
    if body is None:
        root_body = m.body(0)
    else:
        root_body = m.body(body)
    root_frame = frame.Frame(root_body.name,
                             link=frame.Link(root_body.name,
                                             offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos)),
                             joint=frame.Joint())
    _build_chain_recurse(m, root_frame, root_body)
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
    serial_chain = chain.SerialChain(mjcf_chain, end_link_name, "" if root_link_name == "" else root_link_name)
    return serial_chain
