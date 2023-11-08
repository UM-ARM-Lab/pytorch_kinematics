import torch

import pytorch_kinematics.transforms as tf
from pytorch_kinematics.transforms import axis_and_angle_to_matrix_33


class Visual(object):
    TYPES = ['box', 'cylinder', 'sphere', 'capsule', 'mesh']

    def __init__(self, offset=None, geom_type=None, geom_param=None):
        if offset is None:
            self.offset = None
        else:
            self.offset = offset
        self.geom_type = geom_type
        self.geom_param = geom_param

    def __repr__(self):
        return "Visual(offset={0}, geom_type='{1}', geom_param={2})".format(self.offset,
                                                                            self.geom_type,
                                                                            self.geom_param)


class Link(object):
    def __init__(self, name=None, offset=None, visuals=()):
        if offset is None:
            self.offset = None
        else:
            self.offset = offset
        self.name = name
        self.visuals = visuals

    def to(self, *args, **kwargs):
        if self.offset is not None:
            self.offset = self.offset.to(*args, **kwargs)
        return self

    def __repr__(self):
        return "Link(name='{0}', offset={1}, visuals={2})".format(self.name,
                                                                  self.offset,
                                                                  self.visuals)


class Joint(object):
    TYPES = ['fixed', 'revolute', 'prismatic']

    def __init__(self, name=None, offset=None, joint_type='fixed', axis=(0.0, 0.0, 1.0),
                 dtype=torch.float32, device="cpu", limits=None):
        if offset is None:
            self.offset = None
        else:
            self.offset = offset
        self.name = name
        if joint_type not in self.TYPES:
            raise RuntimeError("joint specified as {} type not, but we only support {}".format(joint_type, self.TYPES))
        self.joint_type = joint_type
        if axis is None:
            self.axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        else:
            if torch.is_tensor(axis):
                self.axis = axis.clone().detach().to(dtype=dtype, device=device)
            else:
                self.axis = torch.tensor(axis, dtype=dtype, device=device)
        # normalize axis to have norm 1 (needed for correct representation scaling with theta)
        self.axis = self.axis / self.axis.norm()

        self.limits = limits

    def to(self, *args, **kwargs):
        self.axis = self.axis.to(*args, **kwargs)
        if self.offset is not None:
            self.offset = self.offset.to(*args, **kwargs)
        return self

    def clamp(self, joint_position):
        if self.limits is None:
            return joint_position
        else:
            return torch.clamp(joint_position, self.limits[0], self.limits[1])

    def __repr__(self):
        return "Joint(name='{0}', offset={1}, joint_type='{2}', axis={3})".format(self.name,
                                                                                  self.offset,
                                                                                  self.joint_type,
                                                                                  self.axis)


class Frame(object):
    def __init__(self, name=None, link=None, joint=None, children=None):
        self.name = 'None' if name is None else name
        self.link = link if link is not None else Link()
        self.joint = joint if joint is not None else Joint()
        if children is None:
            self.children = []

    def __str__(self, level=0):
        ret = " \t" * level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def to(self, *args, **kwargs):
        self.joint = self.joint.to(*args, **kwargs)
        self.link = self.link.to(*args, **kwargs)
        self.children = [c.to(*args, **kwargs) for c in self.children]
        return self

    def add_child(self, child):
        self.children.append(child)

    def is_end(self):
        return (len(self.children) == 0)

    def get_transform(self, theta):
        dtype = self.joint.axis.dtype
        d = self.joint.axis.device
        if self.joint.joint_type == 'revolute':
            rot = axis_and_angle_to_matrix_33(self.joint.axis, theta)
            t = tf.Transform3d(rot=rot, dtype=dtype, device=d)
        elif self.joint.joint_type == 'prismatic':
            t = tf.Transform3d(pos=theta * self.joint.axis, dtype=dtype, device=d)
        elif self.joint.joint_type == 'fixed':
            t = tf.Transform3d(default_batch_size=theta.shape[0], dtype=dtype, device=d)
        else:
            raise ValueError("Unsupported joint type %s." % self.joint.joint_type)
        if self.joint.offset is None:
            return t
        else:
            return self.joint.offset.compose(t)
