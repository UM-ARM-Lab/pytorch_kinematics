import math
from typing import Any, Dict, List, Optional, Sequence

import torch

import pytorch_kinematics.transforms as tf

from . import chain, frame
from .urdf_parser_py.sdf import SDF, Box, Cylinder, Mesh, Sphere


JOINT_TYPE_MAP: Dict[str, str] = {
    "revolute": "revolute",
    "prismatic": "prismatic",
    "fixed": "fixed",
}


def _convert_transform(pose: Optional[Sequence[float]]) -> tf.Transform3d:
    if pose is None:
        return tf.Transform3d()
    else:
        # pose: [x, y, z, roll, pitch, yaw] (assumed)
        return tf.Transform3d(
            rot=tf.euler_angles_to_matrix(torch.tensor(pose[3:], dtype=torch.float32), "ZYX"),
            pos=pose[:3],
        )


def _convert_visuals(visuals: Sequence[Any]) -> List[frame.Visual]:
    vlist: List[frame.Visual] = []
    for v in visuals:
        v_tf = _convert_transform(v.pose)
        if isinstance(v.geometry, Mesh):
            g_type = "mesh"
            g_param = v.geometry.filename
        elif isinstance(v.geometry, Cylinder):
            g_type = "cylinder"
            v_tf = v_tf.compose(
                tf.Transform3d(
                    rot=tf.euler_angles_to_matrix(
                        torch.tensor([0.5 * math.pi, 0.0, 0.0], dtype=torch.float32),
                        "ZYX",
                    )
                )
            )
            g_param = (v.geometry.radius, v.geometry.length)
        elif isinstance(v.geometry, Box):
            g_type = "box"
            g_param = v.geometry.size
        elif isinstance(v.geometry, Sphere):
            g_type = "sphere"
            g_param = v.geometry.radius
        else:
            g_type = None
            g_param = None
        vlist.append(frame.Visual(v_tf, g_type, g_param))
    return vlist


def _build_chain_recurse(
    root_frame: frame.Frame,
    lmap: Dict[str, Any],
    joints: Sequence[Any],
) -> List[frame.Frame]:
    children: List[frame.Frame] = []
    for j in joints:
        if j.parent == root_frame.link.name:
            child_frame = frame.Frame(j.child)
            link_p = lmap[j.parent]
            link_c = lmap[j.child]
            t_p = _convert_transform(link_p.pose)
            t_c = _convert_transform(link_c.pose)
            try:
                limits = (j.axis.limit.lower, j.axis.limit.upper)
            except AttributeError:
                limits = None
            child_frame.joint = frame.Joint(
                j.name,
                offset=t_p.inverse().compose(t_c),
                joint_type=JOINT_TYPE_MAP[j.type],
                axis=j.axis.xyz,
                limits=limits,
            )
            child_frame.link = frame.Link(
                link_c.name,
                offset=tf.Transform3d(),
                visuals=_convert_visuals(link_c.visuals),
            )
            child_frame.children = _build_chain_recurse(child_frame, lmap, joints)
            children.append(child_frame)
    return children


def build_chain_from_sdf(data: str) -> chain.Chain:
    """
    Build a Chain object from SDF data.

    Parameters
    ----------
    data : str
        SDF string data.

    Returns
    -------
    chain.Chain
        Chain object created from SDF.
    """
    sdf: SDF = SDF.from_xml_string(data)
    robot = sdf.model
    lmap: Dict[str, Any] = robot.link_map
    joints: Sequence[Any] = robot.joints
    n_joints = len(joints)
    has_root: List[bool] = [True for _ in range(len(joints))]
    for i in range(n_joints):
        for j in range(i + 1, n_joints):
            if joints[i].parent == joints[j].child:
                has_root[i] = False
            elif joints[j].parent == joints[i].child:
                has_root[j] = False
    for i in range(n_joints):
        if has_root[i]:
            root_link = lmap[joints[i].parent]
            break
    else:
        # fallback if no root is found; should not usually happen
        root_link = lmap[joints[0].parent]

    root_frame = frame.Frame(root_link.name)
    root_frame.joint = frame.Joint(offset=_convert_transform(root_link.pose))
    root_frame.link = frame.Link(
        root_link.name,
        tf.Transform3d(),
        _convert_visuals(root_link.visuals),
    )
    root_frame.children = _build_chain_recurse(root_frame, lmap, joints)
    return chain.Chain(root_frame)


def build_serial_chain_from_sdf(
    data: str,
    end_link_name: str,
    root_link_name: str = "",
) -> chain.SerialChain:
    mjcf_chain = build_chain_from_sdf(data)
    serial_chain = chain.SerialChain(
        mjcf_chain,
        end_link_name,
        "" if root_link_name == "" else root_link_name,
    )
    return serial_chain
