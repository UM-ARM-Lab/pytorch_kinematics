from typing import Dict, List, Optional, Union

import mujoco
from mujoco._structs import _MjModelBodyViews as MjModelBodyViews

import pytorch_kinematics.transforms as tf

from . import chain, frame


# Converts from MuJoCo joint types to pytorch_kinematics joint types
JOINT_TYPE_MAP: Dict[int, str] = {
    mujoco.mjtJoint.mjJNT_HINGE: "revolute",
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
}


def body_to_geoms(m: mujoco.MjModel, body: MjModelBodyViews) -> List[frame.Visual]:
    """
    Collect Visual objects for all geoms attached to the given MuJoCo body.
    """
    visuals: List[frame.Visual] = []
    for geom_id in range(m.ngeom):
        geom = m.geom(geom_id)
        if geom.bodyid == body.id:
            visuals.append(
                frame.Visual(
                    offset=tf.Transform3d(rot=geom.quat, pos=geom.pos),
                    geom_type=geom.type,
                    geom_param=geom.size,
                )
            )
    return visuals


def _build_chain_recurse(
    m: mujoco.MjModel,
    parent_frame: frame.Frame,
    parent_body: MjModelBodyViews,
) -> None:
    """
    Recursively attach children frames/links/joints to a pytorch_kinematics Frame
    based on MuJoCo's body tree.
    """
    parent_frame.link.visuals = body_to_geoms(m, parent_body)

    # iterate through all bodies that are children of parent_body
    for body_id in range(m.nbody):
        body = m.body(body_id)
        if body.parentid == parent_body.id and body_id != parent_body.id:
            n_joints = body.jntnum
            if n_joints > 1:
                raise ValueError("composite joints not supported")

            if n_joints == 1:
                # single joint case
                joint = m.joint(body.jntadr[0])
                joint_offset = tf.Transform3d(pos=joint.pos)
                child_joint = frame.Joint(
                    name=joint.name,
                    offset=joint_offset,
                    axis=joint.axis,
                    joint_type=JOINT_TYPE_MAP[joint.type[0]],
                    limits=(joint.range[0], joint.range[1]),
                )
            else:
                # fixed joint
                child_joint = frame.Joint(body.name + "_fixed_joint")

            child_link = frame.Link(
                body.name,
                offset=tf.Transform3d(rot=body.quat, pos=body.pos),
            )
            child_frame = frame.Frame(
                name=body.name,
                link=child_link,
                joint=child_joint,
            )

            parent_frame.children = [*parent_frame.children, child_frame]
            _build_chain_recurse(m, child_frame, body)

    # iterate through all MuJoCo sites attached to this body
    for site_id in range(m.nsite):
        site = m.site(site_id)
        if site.bodyid == parent_body.id:
            site_link = frame.Link(
                site.name,
                offset=tf.Transform3d(rot=site.quat, pos=site.pos),
            )
            site_frame = frame.Frame(
                name=site.name,
                link=site_link,
                joint=frame.Joint(),  # sites are fixed
            )
            parent_frame.children = [*parent_frame.children, site_frame]


def build_chain_from_mjcf(
    data: str,
    body: Union[None, str, int] = None,
    assets: Optional[Dict[str, bytes]] = None,
) -> chain.Chain:
    """
    Build a pytorch-kinematics Chain object from MJCF XML string.

    Parameters
    ----------
    data : str
        MJCF string data.
    body : str or int, optional
        The name or index of the body to use as root of the chain.
        If None, body idx=0 is used (MuJoCo worldbody root).
    assets : dict of name → file bytes, optional
        MJCF asset dictionary.

    Returns
    -------
    chain.Chain
        The constructed robot chain.
    """
    m = mujoco.MjModel.from_xml_string(data, assets=assets)

    # Select root body
    root_body = m.body(0) if body is None else m.body(body)

    root_frame = frame.Frame(
        name=root_body.name,
        link=frame.Link(
            root_body.name,
            offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos),
        ),
        joint=frame.Joint(),
    )

    _build_chain_recurse(m, root_frame, root_body)
    return chain.Chain(root_frame)


def build_serial_chain_from_mjcf(
    data: str,
    end_link_name: str,
    root_link_name: str = "",
) -> chain.SerialChain:
    """
    Build a SerialChain from MJCF XML.

    Parameters
    ----------
    data : str
        MJCF robot description (XML).
    end_link_name : str
        Name of the end-effector link.
    root_link_name : str, optional
        Name of the root link. Default MJCF root if empty.

    Returns
    -------
    chain.SerialChain
        The resulting SerialChain.
    """
    mjcf_chain = build_chain_from_mjcf(data)
    serial_chain = chain.SerialChain(
        mjcf_chain,
        end_link_name,
        "" if root_link_name == "" else root_link_name,
    )
    return serial_chain
