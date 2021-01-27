from . import frame
from . import chain
from . import mjcf_parser
import pytorch_kinematics.transforms as tf

JOINT_TYPE_MAP = {'hinge': 'revolute'}


def geoms_to_visuals(geom, base=tf.Transform3d()):
    visuals = []
    for g in geom:
        if g.type == 'capsule':
            param = (g.size[0], g.fromto)
        elif g.type == 'sphere':
            param = g.size[0]
        else:
            raise ValueError('Invalid geometry type %s.' % g.type)
        visuals.append(frame.Visual(offset=base.compose(tf.Transform3d(rot=g.quat, pos=g.pos)),
                                    geom_type=g.type,
                                    geom_param=param))
    return visuals


def body_to_link(body, base=tf.Transform3d()):
    return frame.Link(body.name,
                      offset=base.compose(tf.Transform3d(rot=body.quat, pos=body.pos)))


def joint_to_joint(joint, base=tf.Transform3d()):
    return frame.Joint(joint.name,
                       offset=base.compose(tf.Transform3d(pos=joint.pos)),
                       joint_type=JOINT_TYPE_MAP[joint.type],
                       axis=joint.axis)


def add_composite_joint(root_frame, joints, base=tf.Transform3d()):
    if len(joints) > 0:
        root_frame.children = root_frame.children + (frame.Frame(link=frame.Link(name=root_frame.link.name + '_child'),
                                                                 joint=joint_to_joint(joints[0], base)),)
        ret, offset = add_composite_joint(root_frame.children[-1], joints[1:])
        return ret, root_frame.joint.offset.compose(offset)
    else:
        return root_frame, root_frame.joint.offset


def _build_chain_recurse(root_frame, root_body):
    base = root_frame.link.offset
    cur_frame, cur_base = add_composite_joint(root_frame, root_body.joint, base)
    jbase = cur_base.inverse().compose(base)
    if len(root_body.joint) > 0:
        cur_frame.link.visuals = geoms_to_visuals(root_body.geom, jbase)
    else:
        cur_frame.link.visuals = geoms_to_visuals(root_body.geom)
    for b in root_body.body:
        cur_frame.children = cur_frame.children + (frame.Frame(),)
        next_frame = cur_frame.children[-1]
        next_frame.name = b.name + "_frame"
        next_frame.link = body_to_link(b, jbase)
        _build_chain_recurse(next_frame, b)


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
    model = mjcf_parser.from_xml_string(data)
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
