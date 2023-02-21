import torch
import typing
import pytorch_kinematics.transforms as tf
from pytorch_kinematics.frame import Frame, Link
from . import jacobian


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) <= 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    return th, N


class Chain(object):
    """Robot model that may be constructed from different descriptions via their respective parsers.
    Fundamentally, a robot is modelled as a chain (not necessarily serial) of frames, with each frame
    having a physical link and a number of child frames each connected via some joint."""

    def __init__(self, root_frame, dtype=torch.float32, device="cpu"):
        self._root = root_frame
        self.dtype = dtype
        self.device = device

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)
        return self

    def __str__(self):
        return str(self._root)

    @staticmethod
    def _find_frame_recursive(name, frame: Frame) -> typing.Optional[Frame]:
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name) -> typing.Optional[Frame]:
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name, frame) -> typing.Optional[Link]:
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name) -> typing.Optional[Link]:
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    @staticmethod
    def _get_joint_parameter_names(frame: Frame, exclude_fixed=True) -> typing.Sequence[str]:
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.joint.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parameter_names(child, exclude_fixed))
        return joint_names

    @staticmethod
    def _get_frame_names(frame: Frame, exclude_fixed=True) -> typing.Sequence[str]:
        names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            names.append(frame.name)
        for child in frame.children:
            names.extend(Chain._get_frame_names(child, exclude_fixed))
        return names

    def get_joint_parameter_names(self, exclude_fixed=True):
        names = self._get_joint_parameter_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    def get_frame_names(self, exclude_fixed=True):
        names = self._get_frame_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    def add_frame(self, frame, parent_name):
        frame = self.find_frame(parent_name)
        if not frame is None:
            frame.add_child(frame)

    @staticmethod
    def _forward_kinematics(root, th_dict, world=None):
        if world is None:
            world = tf.Transform3d()
        link_transforms = {}

        N = next(iter(th_dict.values())).shape[0]
        zeros = torch.zeros([N, 1], dtype=world.dtype, device=world.device)
        th = th_dict.get(root.joint.name, zeros)

        if root.link.offset is not None:
            trans = world.compose(root.link.offset)
        else:
            trans = world

        joint_trans = root.get_transform(th.view(N, 1))
        trans = trans.compose(joint_trans)
        link_transforms[root.link.name] = trans

        for child in root.children:
            link_transforms.update(Chain._forward_kinematics(child, th_dict, trans))
        return link_transforms

    def forward_kinematics(self, th: torch.tensor, world: typing.Optional[tf.Transform3d] = None, end_only=True):
        """
        Return Transform3D wrappers around 4x4 homogenous transform matrices (called H)
        that map points in link frame to world frame (via left multiplication Hx). Specifically,
        (world)H(link), so it expects points in link coordinates on the right and results in world
        frame coordinates.
        :param th: B x J joint values, where B is any number of batch dimensions (including 0) and
        J is the number of joints
        :param world: Transform3D representing the (world)B(base) transform; if omitted, the base frame
        is assumed to lie at the world origin.
        :param end_only: Whether only the end-effector (for serial chains) transform should be returned.
        This parameter does nothing for non-serial chains, but are there for API consistency.
        :return: If end_only, then a single Transform3D mapping end effector to world. Else, a dictionary
        from link name to Transform3D mapping that link to the world frame.
        """
        if world is None:
            world = tf.Transform3d()
        if not isinstance(th, dict):
            th, _ = ensure_2d_tensor(th, self.dtype, self.device)
            jn = self.get_joint_parameter_names()
            assert len(jn) == th.shape[-1]
            th_dict = dict((j, th[..., i]) for i, j in enumerate(jn))
        else:
            th_dict = {k: ensure_2d_tensor(v, self.dtype, self.device)[0] for k, v in th.items()}

        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        return self._forward_kinematics(self._root, th_dict, world)


class SerialChain(Chain):
    """A serial Chain specialization with no branches and clearly defined end effector.
    Note that serial chains can be generated from subsets of a Chain."""

    def __init__(self, chain, end_frame_name, root_frame_name="", **kwargs):
        if root_frame_name == "":
            super(SerialChain, self).__init__(chain._root, **kwargs)
        else:
            super(SerialChain, self).__init__(chain.find_frame(root_frame_name), **kwargs)
            if self._root is None:
                raise ValueError("Invalid root frame name %s." % root_frame_name)
        self._serial_frames = [self._root] + self._generate_serial_chain_recurse(self._root, end_frame_name)
        if self._serial_frames is None:
            raise ValueError("Invalid end frame name %s." % end_frame_name)

    @staticmethod
    def _generate_serial_chain_recurse(root_frame, end_frame_name):
        for child in root_frame.children:
            if child.name == end_frame_name:
                return [child]
            else:
                frames = SerialChain._generate_serial_chain_recurse(child, end_frame_name)
                if not frames is None:
                    return [child] + frames
        return None

    def get_joint_parameter_names(self, exclude_fixed=True):
        names = []
        for f in self._serial_frames:
            if exclude_fixed and f.joint.joint_type == 'fixed':
                continue
            names.append(f.joint.name)
        return names

    def forward_kinematics(self, th, world=None, end_only=True):
        if world is None:
            world = tf.Transform3d()
        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        th, N = ensure_2d_tensor(th, self.dtype, self.device)
        zeros = torch.zeros([N, 1], dtype=world.dtype, device=world.device)

        theta_idx = 0
        link_transforms = {}
        trans = tf.Transform3d(matrix=world.get_matrix().repeat(N, 1, 1))
        for f in self._serial_frames:
            if f.link.offset is not None:
                trans = trans.compose(f.link.offset)

            if f.joint.joint_type == "fixed":  # If fixed
                trans = trans.compose(f.get_transform(zeros))
            else:
                trans = trans.compose(f.get_transform(th[:, theta_idx].view(N, 1)))
                theta_idx += 1

            link_transforms[f.link.name] = trans

        return link_transforms[self._serial_frames[-1].link.name] if end_only else link_transforms

    def jacobian(self, th, locations=None):
        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        return jacobian.calc_jacobian(self, th, tool=locations)
