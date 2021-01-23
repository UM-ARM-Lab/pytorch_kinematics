import torch
import math
from .urdf_parser_py.sdf import SDF, Mesh, Cylinder, Box, Sphere
from . import frame
from . import chain
import pytorch_kinematics.transforms as tf

JOINT_TYPE_MAP = {'revolute': 'revolute',
                  'prismatic': 'prismatic',
                  'fixed': 'fixed'}


def _convert_transform(pose):
    if pose is None:
        return tf.Transform3d()
    else:
        return tf.Transform3d(rot=tf.euler_angles_to_matrix(torch.tensor(pose[3:]), "ZYX"), pos=pose[:3])


def _convert_visuals(visuals):
    vlist = []
    for v in visuals:
        v_tf = _convert_transform(v.pose)
        if isinstance(v.geometry, Mesh):
            g_type = "mesh"
            g_param = v.geometry.filename
        elif isinstance(v.geometry, Cylinder):
            g_type = "cylinder"
            v_tf = v_tf.compose(
                tf.Transform3d(rot=tf.euler_angles_to_matrix(torch.tensor([0.5 * math.pi, 0, 0]), "ZYX")))
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


def _build_chain_recurse(root_frame, lmap, joints):
    children = []
    for j in joints:
        if j.parent == root_frame.link.name:
            child_frame = frame.Frame(j.child + "_frame")
            link_p = lmap[j.parent]
            link_c = lmap[j.child]
            t_p = _convert_transform(link_p.pose)
            t_c = _convert_transform(link_c.pose)
            child_frame.joint = frame.Joint(j.name, offset=t_p.inverse().compose(t_c),
                                            joint_type=JOINT_TYPE_MAP[j.type], axis=j.axis.xyz)
            child_frame.link = frame.Link(link_c.name, offset=tf.Transform3d(),
                                          visuals=_convert_visuals(link_c.visuals))
            child_frame.children = _build_chain_recurse(child_frame, lmap, joints)
            children.append(child_frame)
    return children


def build_chain_from_sdf(data):
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
    sdf = SDF.from_xml_string(data)
    robot = sdf.model
    lmap = robot.link_map
    joints = robot.joints
    n_joints = len(joints)
    has_root = [True for _ in range(len(joints))]
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
    root_frame = frame.Frame(root_link.name + "_frame")
    root_frame.joint = frame.Joint(offset=_convert_transform(root_link.pose))
    root_frame.link = frame.Link(root_link.name, tf.Transform3d(),
                                 _convert_visuals(root_link.visuals))
    root_frame.children = _build_chain_recurse(root_frame, lmap, joints)
    return chain.Chain(root_frame)
