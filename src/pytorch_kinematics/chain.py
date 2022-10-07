from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import torch
import pytorch_kinematics.transforms as tf
from pytorch_kinematics.frame import Frame, Link
from pytorch_kinematics import jacobian
import zpk_cpp


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) <= 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    return th, N


def transform_direction(pose, v):
    v_ = torch.cat([v, torch.zeros_like(v[..., 0:1])], dim=-1).unsqueeze(-1)
    new_v = (pose @ v_)[..., :3, 0]
    return new_v


class Chain(object):
    """Robot model that may be constructed from different descriptions via their respective parsers.
    Fundamentally, a robot is modelled as a chain (not necessarily serial) of frames, with each frame
    having a physical link and a number of child frames each connected via some joint."""

    def __init__(self, root_frame, dtype=torch.float32, device="cpu"):
        self._root = root_frame
        self.dtype = dtype
        self.device = device

        self.identity = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
        low, high = self.get_joint_limits()
        self.low = torch.tensor(low, device=self.device, dtype=self.dtype)
        self.high = torch.tensor(high, device=self.device, dtype=self.dtype)
        self.sphere_indices = None
        self.precompute_fk_info()

    def precompute_fk_info(self):
        self.parent_indices = []
        self.joint_indices = []
        n = len(self.get_joint_parameter_names())
        self.axes = torch.zeros([n, 3], dtype=self.dtype, device=self.device)
        self.is_fixed = []
        self.link_offsets = []
        self.joint_offsets = []
        queue = []
        queue.insert(-1, (self._root, -1, 0))
        idx = 0
        self.frame_to_idx = {}
        self.joint_weights = torch.zeros([n], dtype=self.dtype, device=self.device)
        while len(queue) > 0:
            root, parent_idx, depth = queue.pop(0)
            self.frame_to_idx[root.name.strip("\n")] = idx
            self.parent_indices.append(parent_idx)
            self.is_fixed.append(root.joint.joint_type == 'fixed')

            if root.link.offset is None:
                self.link_offsets.append(None)
            else:
                self.link_offsets.append(root.link.offset.get_matrix())

            if root.joint.offset is None:
                self.joint_offsets.append(None)
            else:
                self.joint_offsets.append(root.joint.offset.get_matrix())

            if self.is_fixed[-1]:
                self.joint_indices.append(-1)
            else:
                jnt_idx = self.get_joint_parameter_names().index(root.joint.name)
                self.axes[jnt_idx] = root.joint.axis
                self.joint_indices.append(jnt_idx)
                self.joint_weights[jnt_idx] = depth

            for child in root.children:
                queue.append((child, idx, depth + 1))

            idx += 1
        self.joint_indices = torch.tensor(self.joint_indices)
        self.joint_weights = self.joint_weights / self.joint_weights.norm()

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)

        self.identity = self.identity.to(device=self.device, dtype=self.dtype)
        self.axes = self.axes.to(dtype=self.dtype, device=self.device)
        self.link_offsets = [l if l is None else l.to(dtype=self.dtype, device=self.device) for l in self.link_offsets]
        self.joint_offsets = [j if j is None else j.to(dtype=self.dtype, device=self.device) for j in
                              self.joint_offsets]
        self.joint_weights = [j if j is None else j.to(dtype=self.dtype, device=self.device) for j in
                              self.joint_weights]
        self.low = self.low.to(dtype=self.dtype, device=self.device)
        self.high = self.high.to(dtype=self.dtype, device=self.device)

        return self

    def __str__(self):
        return str(self._root)

    @staticmethod
    def _find_frame_recursive(name, frame: Frame) -> Optional[Frame]:
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name) -> Optional[Frame]:
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name, frame) -> Optional[Link]:
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    @staticmethod
    def _get_joints(frame, exclude_fixed=True):
        joints = []
        if exclude_fixed and frame.joint.joint_type != "fixed":
            joints.append(frame.joint)
        for child in frame.children:
            joints.extend(Chain._get_joints(child))
        return joints

    def get_joints(self, exclude_fixed=True):
        joints = self._get_joints(self._root, exclude_fixed=exclude_fixed)
        return joints

    @staticmethod
    def _find_joint_recursive(name, frame):
        for child in frame.children:
            if child.joint.name == name:
                return child.joint
            ret = Chain._find_joint_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name) -> Optional[Link]:
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    def find_joint(self, name):
        if self._root.joint.name == name:
            return self._root.joint
        return self._find_joint_recursive(name, self._root)

    @staticmethod
    def _get_joint_parent_frame_names(frame, exclude_fixed=True):
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parent_frame_names(child, exclude_fixed))
        return joint_names

    def get_joint_parent_frame_names(self, exclude_fixed=True):
        names = self._get_joint_parent_frame_names(self._root, exclude_fixed)
        return sorted(set(names), key=names.index)

    @staticmethod
    def _get_joint_parameter_names(frame: Frame, exclude_fixed=True) -> Sequence[str]:
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.joint.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parameter_names(child, exclude_fixed))
        return joint_names

    @staticmethod
    def _get_frame_names(frame: Frame, exclude_fixed=True) -> Sequence[str]:
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
    def _get_links(frame):
        links = [frame.link]
        for child in frame.children:
            links.extend(Chain._get_links(child))
        return links

    def get_links(self):
        links = self._get_links(self._root)
        return links

    @staticmethod
    def _get_link_names(frame):
        link_names = [frame.link.name]
        for child in frame.children:
            link_names.extend(Chain._get_link_names(child))
        return link_names

    def get_link_names(self):
        names = self._get_link_names(self._root)
        return sorted(set(names), key=names.index)

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

    def forward_kinematics(self, th: torch.tensor, world: Optional[tf.Transform3d] = None, end_only=True):
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
            world = tf.Transform3d(dtype=self.dtype, device=self.device)

        th_dict = self.ensure_dict_of_2d_tensors(th)

        if world.dtype != self.dtype or world.device != self.device:
            world = world.to(dtype=self.dtype, device=self.device, copy=True)
        return self._forward_kinematics(self._root, th_dict, world)

    @lru_cache
    def get_link_indices(self, *tool_names):
        return torch.tensor([self.frame_to_idx[n + '_frame'] for n in tool_names], dtype=torch.long,
                            device=self.device)

    def forward_kinematics_fast(self, th, link_indices):
        """
        Instead of a tree, we can use a flat data structure with indexes to represent the parent
        then instead of recursion we can just iterate in order and use parent pointers. This
        reduces function call overhead and moves some of the indexing work to the constructor.
        """
        if isinstance(th, np.ndarray):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)

        th = torch.atleast_2d(th)

        b, n = th.shape

        axes_expanded = self.axes.unsqueeze(0).repeat(b, 1, 1)

        tool_transforms = zpk_cpp.fk(
            link_indices,
            axes_expanded,
            th,
            self.parent_indices,
            self.is_fixed,
            self.joint_indices,
            self.joint_offsets,
            self.link_offsets
        )

        return tool_transforms

    def ensure_dict_of_2d_tensors(self, th):
        if not isinstance(th, dict):
            th, _ = ensure_2d_tensor(th, self.dtype, self.device)
            jn = self.get_joint_parameter_names()
            assert len(jn) == th.shape[-1]
            th_dict = dict((j, th[..., i]) for i, j in enumerate(jn))
        else:
            th_dict = {k: ensure_2d_tensor(v, self.dtype, self.device)[0] for k, v in th.items()}
        return th_dict

    def clamp(self, th):
        th_dict = self.ensure_dict_of_2d_tensors(th)

        out_th_dict = {}
        for joint_name, joint_position in th_dict.items():
            joint = self.find_joint(joint_name)
            joint_position_clamped = joint.clamp(joint_position)
            out_th_dict[joint_name] = joint_position_clamped

        return self.match_input_type(out_th_dict, th)

    @staticmethod
    def match_input_type(th_dict, th):
        if isinstance(th, dict):
            return th_dict
        else:
            return torch.stack([v for v in th_dict.values()], dim=-1)

    def get_joint_limits(self):
        low = []
        high = []
        for joint_name in self.get_joint_parameter_names(exclude_fixed=True):
            joint = self.find_joint(joint_name)
            low.append(joint.limits[0])
            high.append(joint.limits[1])

        return low, high

    @staticmethod
    def _get_joints_and_child_links(frame):
        joint = frame.joint

        me_and_my_children = [frame.link]
        for child in frame.children:
            recursive_child_links = yield from Chain._get_joints_and_child_links(child)
            me_and_my_children.extend(recursive_child_links)

        if joint is not None and joint.joint_type != 'fixed':
            yield joint, me_and_my_children

        return me_and_my_children

    def get_joints_and_child_links(self):
        yield from Chain._get_joints_and_child_links(self._root)

    def expected_torques(self, th):
        gravity = torch.tensor([0, 0, -9.8], dtype=self.dtype, device=self.device)
        poses_dict = self.forward_kinematics(th)
        torques_dict = {}
        for joint, child_links in self.get_joints_and_child_links():
            # NOTE: assumes joint has not offset from joint_link
            joint_link = child_links[0]
            # print(joint.name, [l.name for l in child_links])
            child_masses = []
            child_positions = []
            for link in child_links:
                child_masses.append(link.mass)
                child_positions.append(poses_dict[link.name].get_matrix()[:, :3, 3])
            child_masses = torch.tensor(child_masses, dtype=self.dtype, device=self.device)
            child_positions = torch.stack(child_positions, dim=-1)
            avg_position = (child_masses[None, None] * child_positions).sum(dim=-1)  # add batch_dims
            total_mass = torch.sum(child_masses)
            joint_pose = poses_dict[joint_link.name].get_matrix()
            joint_position = joint_pose[:, :3, 3]
            axis_joint_frame = joint.axis[None]  # add batch dims
            axis_world_frame = transform_direction(joint_pose, axis_joint_frame)
            # print(avg_position, total_mass, axis_world_frame, joint_position)
            joint_to_avg_pos = avg_position - joint_position
            # NOTE: A@B.T is the same as batched dot product
            lever_vector = joint_to_avg_pos - axis_world_frame @ joint_to_avg_pos.T * axis_world_frame
            force_vector = (total_mass * gravity)[None]
            torque_world_frame = torch.cross(lever_vector, force_vector)
            torque_joint_frame = transform_direction(torch.linalg.pinv(joint_pose), torque_world_frame)
            # print(torque_joint_frame)
            torques_dict[joint.name] = torque_joint_frame

        torques = [torques_dict[n] for n in self.get_joint_parameter_names(True)]
        return torques


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

    def clamp(self, th: torch.tensor):
        return th
