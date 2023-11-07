from functools import lru_cache
from pytorch_kinematics.transforms.rotation_conversions import tensor_axis_and_angle_to_matrix
from pytorch_kinematics.transforms.rotation_conversions import tensor_axis_and_d_to_pris_matrix
from typing import Optional, Sequence

import numpy as np
import torch

import pytorch_kinematics.transforms as tf
from pytorch_kinematics.frame import Frame, Link, Joint


def get_th_size(th):
    """

    Args:
        th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched

    Returns: The number of joints in the input

    """
    if isinstance(th, torch.Tensor) or isinstance(th, np.ndarray):
        return th.shape[-1]
    elif isinstance(th, list) or isinstance(th, dict):
        return len(th)
    else:
        raise NotImplementedError(f"Unsupported type {type(th)}")


def ensure_2d_tensor(th, dtype, device):
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=dtype, device=device)
    if len(th.shape) <= 1:
        N = 1
        th = th.reshape(1, -1)
    else:
        N = th.shape[0]
    return th, N


def get_dict_elem_shape(th_dict):
    elem = th_dict[list(th_dict.keys())[0]]
    if isinstance(elem, np.ndarray):
        return elem.shape
    elif isinstance(elem, torch.Tensor):
        return elem.shape
    else:
        return ()


class Chain:
    """
    Robot model that may be constructed from different descriptions via their respective parsers.
    Fundamentally, a robot is modelled as a chain (not necessarily serial) of frames, with each frame
    having a physical link and a number of child frames each connected via some joint.
    """

    def __init__(self, root_frame, dtype=torch.float32, device="cpu"):
        self._root = root_frame
        self.dtype = dtype
        self.device = device

        self.identity = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)

        low, high = self.get_joint_limits()
        self.low = torch.tensor(low, device=self.device, dtype=self.dtype)
        self.high = torch.tensor(high, device=self.device, dtype=self.dtype)

        # As we traverse the kinematic tree, each frame is assigned an index.
        # We use this index to build a flat representation of the tree.
        # parents_indices and joint_indices all use this indexing scheme.
        # The root frame will be index 0 and the first frame of the root frame's children will be index 1,
        # then the child of that frame will be index 2, etc. In other words, it's a depth-first ordering.
        self.parents_indices = []  # list of indices from 0 (root) to the given frame
        self.joint_indices = []
        self.n_joints = len(self.get_joint_parameter_names())
        self.axes = torch.zeros([self.n_joints, 3], dtype=self.dtype, device=self.device)
        self.link_offsets = []
        self.joint_offsets = []
        self.joint_type_indices = []
        queue = []
        queue.insert(-1, (self._root, -1, 0))  # the root has no parent so we use -1.
        idx = 0
        self.frame_to_idx = {}
        self.idx_to_frame = {}
        while len(queue) > 0:
            root, parent_idx, depth = queue.pop(0)
            name_strip = root.name.strip("\n")
            self.frame_to_idx[name_strip] = idx
            self.idx_to_frame[idx] = name_strip
            if parent_idx == -1:
                self.parents_indices.append([idx])
            else:
                self.parents_indices.append(self.parents_indices[parent_idx] + [idx])

            is_fixed = root.joint.joint_type == 'fixed'

            if root.link.offset is None:
                self.link_offsets.append(None)
            else:
                self.link_offsets.append(root.link.offset.get_matrix())

            if root.joint.offset is None:
                self.joint_offsets.append(None)
            else:
                self.joint_offsets.append(root.joint.offset.get_matrix())

            if is_fixed:
                self.joint_indices.append(-1)
            else:
                jnt_idx = self.get_joint_parameter_names().index(root.joint.name)
                self.axes[jnt_idx] = root.joint.axis
                self.joint_indices.append(jnt_idx)

            # these are integers so that we can use them as indices into tensors
            # FIXME: how do we know the order of these types in C++?
            self.joint_type_indices.append(Joint.TYPES.index(root.joint.joint_type))

            for child in root.children:
                queue.append((child, idx, depth + 1))

            idx += 1
        self.joint_type_indices = torch.tensor(self.joint_type_indices)
        self.joint_indices = torch.tensor(self.joint_indices)
        # We need to use a dict because torch.compile doesn't list lists of tensors

        # pad parent indices with -1 so they are all the same length

        # also get parent indices excluding fixed joints - useful for jacobian calcualtion
        self.non_fixed_parents_indices = []
        for parent_indices in self.parents_indices:
            non_fixed_parent_idx = []
            for parent_idx in parent_indices:
                if self.find_frame(self.idx_to_frame[parent_idx]).joint.joint_type != 'fixed':
                    non_fixed_parent_idx.append(parent_idx)
            self.non_fixed_parents_indices.append(non_fixed_parent_idx)

        # convert to tensors
        self.parents_indices = [torch.tensor(p, dtype=torch.long, device=self.device) for p in self.parents_indices]

        # pad non fixed parent indices with -1 so they are all fixed
        max_len = max([len(p) for p in self.non_fixed_parents_indices])
        self.non_fixed_parents_indices = [torch.tensor(p + [-1] * (max_len - len(p)),
                                                       dtype=torch.long,
                                                       device=self.device) for p in self.non_fixed_parents_indices]

        self.non_fixed_parents_indices = torch.stack(self.non_fixed_parents_indices, dim=0)
        self.max_kinematic_tree_depth = self.non_fixed_parents_indices.shape[1]

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)

        self.identity = self.identity.to(device=self.device, dtype=self.dtype)
        self.parents_indices = [p.to(dtype=torch.long, device=self.device) for p in self.parents_indices]
        self.non_fixed_parents_indices = self.non_fixed_parents_indices.to(dtype=torch.long, device=self.device)
        self.joint_type_indices = self.joint_type_indices.to(dtype=torch.long, device=self.device)
        self.joint_indices = self.joint_indices.to(dtype=torch.long, device=self.device)
        self.axes = self.axes.to(dtype=self.dtype, device=self.device)
        self.link_offsets = [l if l is None else l.to(dtype=self.dtype, device=self.device) for l in self.link_offsets]
        self.joint_offsets = [j if j is None else j.to(dtype=self.dtype, device=self.device) for j in
                              self.joint_offsets]
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

    @lru_cache()
    def get_joint_parameter_names(self, exclude_fixed=True):
        names = []
        for f in self.get_joints(exclude_fixed=exclude_fixed):
            if exclude_fixed and f.joint.joint_type == 'fixed':
                continue
            names.append(f.joint.name)
        return names

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

    @lru_cache
    def get_frame_indices(self, *frame_names):
        return torch.tensor([self.frame_to_idx[n] for n in frame_names], dtype=torch.long,
                            device=self.device)

    def forward_kinematics(self, th, frame_indices: Optional = None):
        if frame_indices is None:
            frame_indices = self.get_all_frame_indices()

        th = self.ensure_tensor(th)
        th = torch.atleast_2d(th)

        b = th.shape[0]
        axes_expanded = self.axes.unsqueeze(0).repeat(b, 1, 1)

        # compute all joint transforms at once first
        # in order to handle multiple joint types without branching, we create all possible transforms
        # for all joint types and then select the appropriate one for each joint.
        rev_jnt_transform = tensor_axis_and_angle_to_matrix(axes_expanded, th)
        pris_jnt_transform = tensor_axis_and_d_to_pris_matrix(axes_expanded, th)

        frame_transforms = {}
        b = th.shape[0]
        for frame_idx in frame_indices:
            frame_transform = torch.eye(4).to(th).unsqueeze(0).repeat(b, 1, 1)

            # iterate down the list and compose the transform
            for chain_idx in self.parents_indices[frame_idx.item()]:
                if chain_idx.item() in frame_transforms:
                    frame_transform = frame_transforms[chain_idx.item()]
                else:
                    link_offset_i = self.link_offsets[chain_idx]
                    if link_offset_i is not None:
                        frame_transform = frame_transform @ link_offset_i

                    joint_offset_i = self.joint_offsets[chain_idx]
                    if joint_offset_i is not None:
                        frame_transform = frame_transform @ joint_offset_i

                    jnt_idx = self.joint_indices[chain_idx]
                    jnt_type = self.joint_type_indices[chain_idx]
                    if jnt_type == 0:
                        pass
                    elif jnt_type == 1:
                        jnt_transform_i = rev_jnt_transform[:, jnt_idx]
                        frame_transform = frame_transform @ jnt_transform_i
                    elif jnt_type == 2:
                        jnt_transform_i = pris_jnt_transform[:, jnt_idx]
                        frame_transform = frame_transform @ jnt_transform_i

            frame_transforms[frame_idx.item()] = frame_transform

        frame_names_and_transform3ds = {self.idx_to_frame[frame_idx]: tf.Transform3d(matrix=transform) for
                                        frame_idx, transform in frame_transforms.items()}

        return frame_names_and_transform3ds

    def ensure_tensor(self, th):
        """
        Converts a number of possible types into a tensor. The order of the tensor is determined by the order
        of self.get_joint_parameter_names().
        """
        if isinstance(th, np.ndarray):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, list):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
        elif isinstance(th, dict):
            # convert dict to a flat, complete, tensor of all joints values. Missing joints are filled with zeros.
            th_dict = th
            elem_shape = get_dict_elem_shape(th_dict)
            th = torch.ones([*elem_shape, self.n_joints], device=self.device, dtype=self.dtype) * torch.nan
            joint_names = self.get_joint_parameter_names()
            for joint_name, joint_position in th_dict.items():
                jnt_idx = joint_names.index(joint_name)
                th[..., jnt_idx] = joint_position
            if torch.any(torch.isnan(th)):
                msg = "Missing values for the following joints:\n"
                for joint_name, th_i in zip(self.get_joint_parameter_names(), th):
                    msg += joint_name + "\n"
                raise ValueError(msg)
        return th

    def get_all_frame_indices(self):
        frame_indices = self.get_frame_indices(*self.get_frame_names(exclude_fixed=False))
        return frame_indices

    def ensure_dict_of_2d_tensors(self, th):
        if not isinstance(th, dict):
            th, _ = ensure_2d_tensor(th, self.dtype, self.device)
            assert self.n_joints == th.shape[-1]
            th_dict = dict((j, th[..., i]) for i, j in enumerate(self.get_joint_parameter_names()))
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
            if joint.limits is None:
                low.append(-np.pi)
                high.append(np.pi)
            else:
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

    def jacobian(self, th, locations=None, link_indices=None):
        if link_indices is None:
            msg = "Cannot compute Jacobian for non-serial chain unless frame_indices are specified." \
                  " The Jacobian is can be defined for any links, but we need to know which links you want."
            raise ValueError(msg)
        if locations is not None:
            locations = tf.Transform3d(pos=locations, device=self.device)

        return self.calc_jacobian(th, tool=locations, link_indices=link_indices)

    def jacobian_and_hessian(self, th, locations=None, link_indices=None):
        if link_indices is None:
            msg = "Cannot compute Jacobian for non-serial chain unless frame_indices are specified." \
                  " The Jacobian is can be defined for any links, but we need to know which links you want."
            raise ValueError(msg)
        if locations is not None:
            locations = tf.Transform3d(pos=locations, device=self.device)

        return self.calc_jacobian_and_hessian(th, tool=locations, link_indices=link_indices)

    def calc_jacobian(self, th, tool=None, link_indices=None):
        """
        Return robot Jacobian J in base frame (N,6,DOF) where dot{x} = J dot{q}
        The first 3 rows relate the translational velocities and the
        last 3 rows relate the angular velocities.

        tool is the transformation wrt the end effector; default is identity. If specified, will have to
        specify for each of the N inputs

        FIXME: this code assumes the joint frame and the child link frame are the same
        """
        if not torch.is_tensor(th):
            th = torch.tensor(th, dtype=self.dtype, device=self.device)
        if len(th.shape) <= 1:
            N = 1
            th = th.reshape(1, -1)
        else:
            N = th.shape[0]
        ndof = th.shape[1]
        N_range = torch.arange(0, N).to(dtype=torch.long, device=self.device)

        if tool is None:
            cur_transform = tf.Transform3d(device=self.device,
                                           dtype=self.dtype).get_matrix().repeat(N, 1, 1)
        else:
            if tool.dtype != self.dtype or tool.device != self.device:
                tool = tool.to(device=self.device, copy=True, dtype=self.dtype)
            cur_transform = tool.get_matrix()

        if isinstance(self, SerialChain):
            fk_dict = self.forward_kinematics(th, end_only=False)
        else:
            fk_dict = self.forward_kinematics(th)

        # assemble transforms into a single multi-batch tensor - should be in the order of the frames
        T = []
        for frame_idx, frame_name in self.idx_to_frame.items():
            T.append(fk_dict[frame_name].get_matrix())
        T = torch.stack(T, dim=0)  # num_links x N x 4 x 4

        # retrieve desired link-transform
        ee_transform = T[link_indices, N_range] @ cur_transform
        tool_world = ee_transform[:, :3, 3]

        # compute jacobian in world frame
        jacobian = torch.zeros((N, 6, ndof), dtype=self.dtype, device=self.device)
        # TODO exclude fixed joints? saves for loop time
        # construct a parent indices tensor which ommits fixed joints
        # rather than incrementing count, some how we need to extract
        for d in range(self.max_kinematic_tree_depth):
            # Retrieve frame information
            frame_idx = self.non_fixed_parents_indices[link_indices, d]
            transform = T[frame_idx, N_range]
            joint_idx = self.joint_indices[frame_idx]
            joint_axes = self.axes[joint_idx].expand(N, 3).unsqueeze(-1)
            joint_type = self.joint_type_indices[frame_idx].unsqueeze(-1)

            old_jacobian_col = jacobian[N_range, :, joint_idx].clone()
            jacobian_col = jacobian[N_range, :, joint_idx].clone()
            joint_axes_world = (transform[:, :3, :3] @ joint_axes).squeeze(-1)

            # compute jacobian as if revolute joint
            if torch.any(joint_type == Joint.TYPES.index('revolute')):
                position_jacobian = torch.cross(joint_axes_world, tool_world - transform[:, :3, 3], dim=-1)
                jac_column_revolute = torch.cat((position_jacobian, joint_axes_world), dim=-1)
                # Replace jacobian column with correct type
                jacobian_col = torch.where(joint_type == Joint.TYPES.index('revolute'),
                                           jac_column_revolute,
                                           jacobian_col
                                           )

            # compute jacobian as if prismatic joint
            if torch.any(joint_type == Joint.TYPES.index('prismatic')):
                jacobian_column_prismatic = torch.cat((joint_axes_world,
                                                       torch.zeros_like(joint_axes_world)), dim=-1)
                jacobian_col = torch.where(joint_type == Joint.TYPES.index('prismatic'),
                                           jacobian_column_prismatic,
                                           jacobian_col
                                           )

            # update jacobian but only if we are not at the desired frame_idx
            jacobian[N_range, :, joint_idx] = torch.where(frame_idx.unsqueeze(-1) > -1,
                                                          jacobian_col,
                                                          old_jacobian_col)

            # break if have calculated for all desired links
            if torch.all(frame_idx < 0):
                break

        return jacobian

    def calc_jacobian_and_hessian(self, th, tool=None, link_indices=None):
        """
            Calculates robot jacobian and kinematic hessian in the base frame

            Returns:
                J: torch.tensor of shape (N, 6, DOF) representing robot jacobian
                H: torch.tensor of shape (N, 6, DOF, DOF) - kinematic Hessian. The kinematic hessian is the partial
                   derivative of the robot jacobian

        """
        if not torch.is_tensor(th):
            th = torch.tensor(th, dtype=self.dtype, device=self.device)
        if len(th.shape) <= 1:
            N = 1
            th = th.view(1, -1)
        else:
            N = th.shape[0]
        ndof = th.shape[1]

        J = self.calc_jacobian(th, tool, link_indices)

        H = torch.zeros(N, 6, ndof, ndof, device=self.device, dtype=self.dtype)
        # TODO can this be vectorized?
        for j in range(ndof):
            for i in range(j, ndof):
                H[:, :3, i, j] = torch.cross(J[:, 3:, j], J[:, :3, i])
                H[:, 3:, i, j] = torch.cross(J[:, 3:, j], J[:, 3:, i])
                if i != j:
                    H[:, :3, j, i] = H[:, :3, i, j]
        return J, H

    @staticmethod
    def _generate_serial_chain_recurse(root_frame, end_frame_name):
        for child in root_frame.children:
            if child.name == end_frame_name:
                return [child]
            else:
                frames = Chain._generate_serial_chain_recurse(child, end_frame_name)
                if not frames is None:
                    return [child] + frames
        return None


class SerialChain(Chain):
    """
    A serial Chain specialization with no branches and clearly defined end effector.
    Serial chains can be generated from subsets of a Chain.
    """

    def __init__(self, chain, end_frame_name, root_frame_name="", **kwargs):
        if root_frame_name == "":
            super().__init__(chain._root, **kwargs)
        else:
            super().__init__(chain.find_frame(root_frame_name), **kwargs)
            if self._root is None:
                raise ValueError("Invalid root frame name %s." % root_frame_name)
        self._serial_frames = [self._root] + self._generate_serial_chain_recurse(self._root, end_frame_name)
        if self._serial_frames is None:
            raise ValueError("Invalid end frame name %s." % end_frame_name)
        # self.convert_serial_inputs_to_chain_inputs = torch.vmap(self._convert_serial_inputs_to_chain_inputs, in_dims=(None, 1))

    def jacobian(self, th, locations=None, link_indices=None):
        if locations is not None:
            locations = tf.Transform3d(pos=locations)

        if not torch.is_tensor(th):
            th = torch.tensor(th, dtype=self.dtype, device=self.device)
        if len(th.shape) <= 1:
            N = 1
            th = th.reshape(1, -1)
        else:
            N = th.shape[0]

        # if link indices is None we assume the end-effector
        if link_indices is None:
            link_indices = self.get_frame_indices(self._serial_frames[-1].name).expand(N)

        _, th = self._convert_serial_inputs_to_chain_inputs(False, th)

        return self.calc_jacobian(th, tool=locations, link_indices=link_indices)

    def forward_kinematics(self, th, end_only: bool = True):
        """ Like the base class, except `th` only needs to contain the joints in the SerialChain, not all joints. """
        frame_indices, th = self._convert_serial_inputs_to_chain_inputs(end_only, th)

        mat = super().forward_kinematics(th, frame_indices)

        if end_only:
            return mat[self._serial_frames[-1].name]
        else:
            return mat

    def _convert_serial_inputs_to_chain_inputs(self, end_only, th):
        if not torch.is_tensor(th):
            th = torch.tensor(th, dtype=self.dtype, device=self.device)
        if len(th.shape) <= 1:
            th = th.reshape(1, -1)
        else:
            N = th.shape[0]
        th_size = get_th_size(th)
        if end_only:
            frame_indices = self.get_frame_indices(self._serial_frames[-1].name)
        else:
            # pass through default behavior for frame indices being None, which is currently
            # to return all frames.
            frame_indices = None
        if th_size < self.n_joints:
            # if th is only a partial list of joints, assume it's a list of joints for only the serial chain.
            partial_th = th
            nonfixed_serial_frames = list(filter(lambda f: f.joint.joint_type != 'fixed', self._serial_frames))
            if len(nonfixed_serial_frames) != th_size:
                raise ValueError(f'Expected {len(nonfixed_serial_frames)} joint values, got {th_size}.')
            th = torch.zeros(N, self.n_joints, device=self.device, dtype=self.dtype)
            for i, frame, in enumerate(nonfixed_serial_frames):
                partial_th_i = partial_th[:, i]
                k = self.frame_to_idx[frame.name]
                jnt_idx = self.joint_indices[k]
                if frame.joint.joint_type != 'fixed':
                    th[:, jnt_idx] = partial_th_i
        return frame_indices, th
