from functools import lru_cache
from typing import Optional, Sequence

import copy
import numpy as np
import torch

import pytorch_kinematics.transforms as tf
from pytorch_kinematics.frame import Frame, Link, Joint
from pytorch_kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix_44, axis_and_d_to_pris_matrix


def _fk_impl(th, static_offsets, joint_indices_clamped, joint_type_indices,
             direct_parent_idx, bfs_levels, axes,
             has_revolute, has_prismatic, num_frames):
    """Batched forward kinematics: joint angles → world-frame transforms.

    Computes the 4x4 homogeneous world-frame transform for every frame in the
    kinematic tree, for B joint configurations in parallel.

    Algorithm:
      1. Convert joint values to per-joint 4x4 transforms (rotation or translation
         along the joint axis). Fixed joints get identity.
      2. Compute local transforms: static_offset @ joint_transform for each frame.
         The static offset encodes the fixed geometric relationship between a frame
         and its parent (link offset @ joint offset from the URDF/MJCF).
      3. Accumulate world transforms by traversing the tree level-by-level (BFS).
         Root frames copy their local transform directly. Each subsequent level
         composes: world_T[f] = world_T[parent[f]] @ local_T[f].
         Frames at the same depth are independent and computed in one batched matmul.

    Args:
        th: (B, n_joints) joint angles/positions
        static_offsets: (num_frames, 4, 4) pre-multiplied link_offset @ joint_offset
        joint_indices_clamped: (num_frames,) which joint drives each frame (0 for fixed)
        joint_type_indices: (num_frames,) 0=fixed, 1=revolute, 2=prismatic
        direct_parent_idx: (num_frames,) parent frame index (-1 for roots)
        bfs_levels: list of tensors, bfs_levels[d] = frame indices at depth d
        axes: (n_joints, 3) joint rotation/translation axes
        has_revolute: bool, whether any revolute joints exist
        has_prismatic: bool, whether any prismatic joints exist
        num_frames: int, total number of frames in the kinematic tree

    Returns:
        T_world_link: (num_frames, B, 4, 4) transforms from each link frame to world.
            Right-multiplying by a point in link coordinates gives world coordinates:
            p_world = T_world_link[f] @ p_link.
    """
    B = th.shape[0]
    jidx = joint_indices_clamped
    eye4 = torch.eye(4, device=th.device, dtype=th.dtype)

    # Step 1: Joint values → per-joint 4x4 transforms
    if th.shape[1] > 0:
        axes_expanded = axes.unsqueeze(0).expand(B, -1, -1)
        if has_revolute:
            rev_transforms = axis_and_angle_to_matrix_44(axes_expanded, th)
            rev_per_frame = rev_transforms[:, jidx].permute(1, 0, 2, 3)
        if has_prismatic:
            pris_transforms = axis_and_d_to_pris_matrix(axes_expanded, th)
            pris_per_frame = pris_transforms[:, jidx].permute(1, 0, 2, 3)

    # Assign joint transforms by type (fixed → identity, revolute → rotation, prismatic → translation)
    if has_revolute and not has_prismatic:
        is_rev = (joint_type_indices == 1).reshape(-1, 1, 1, 1)
        joint_transforms = torch.where(is_rev, rev_per_frame,
                                        eye4.reshape(1, 1, 4, 4).expand(num_frames, B, 4, 4))
    elif has_prismatic and not has_revolute:
        is_pris = (joint_type_indices == 2).reshape(-1, 1, 1, 1)
        joint_transforms = torch.where(is_pris, pris_per_frame,
                                        eye4.reshape(1, 1, 4, 4).expand(num_frames, B, 4, 4))
    elif has_revolute and has_prismatic:
        is_rev = (joint_type_indices == 1).reshape(-1, 1, 1, 1)
        is_pris = (joint_type_indices == 2).reshape(-1, 1, 1, 1)
        joint_transforms = eye4.reshape(1, 1, 4, 4).expand(num_frames, B, 4, 4)
        joint_transforms = torch.where(is_rev, rev_per_frame, joint_transforms)
        joint_transforms = torch.where(is_pris, pris_per_frame, joint_transforms)
    else:
        joint_transforms = eye4.reshape(1, 1, 4, 4).expand(num_frames, B, 4, 4)

    # Step 2: Local transforms = static geometric offset @ joint-dependent transform
    local_transforms = static_offsets.unsqueeze(1) @ joint_transforms

    # Step 3: BFS accumulation — T_world_link[f] = T_world_link[parent[f]] @ local_T[f]
    # Frames at the same depth have independent parents, so each level is one batched matmul.
    T_world_link = torch.empty(num_frames, B, 4, 4, device=th.device, dtype=th.dtype)
    T_world_link[bfs_levels[0]] = local_transforms[bfs_levels[0]]
    for d in range(1, len(bfs_levels)):
        level_indices = bfs_levels[d]
        parents = direct_parent_idx[level_indices]
        T_world_link[level_indices] = T_world_link[parents] @ local_transforms[level_indices]

    return T_world_link


class _FKAnalyticalBackward(torch.autograd.Function):
    """Attach analytical geometric Jacobian as the backward for FK.

    Forward passes through the pre-computed T_world_link unchanged. Backward
    uses the geometric Jacobian to analytically compute d(loss)/d(joint_angles)
    from d(loss)/d(T_world_link), avoiding the expensive graph replay of
    standard autograd (~9x faster on GPU).

    All inputs to apply() are plain tensors, making this compatible with
    torch.compile (no list[Tensor], bool, or int args that dynamo can't trace).

    Notation — indices iterate over:
      L = number of links (frames), B = batch of configs, J = number of DOFs.

    The backward formula for each DOF j, summing over descendant links l:
      revolute:  d(loss)/d(q_j) = z_j · Σ_l mask[j,l] * (τ_l + (t_l - o_j) × ∂L/∂t_l)
      prismatic: d(loss)/d(q_j) = z_j · Σ_l mask[j,l] * ∂L/∂t_l

    where:
      T_world_link[l] has rotation R_l and translation t_l (link l's world-frame pose),
      z_j = world-frame axis of DOF j,  o_j = world-frame origin of DOF j's link,
      ∂L/∂R_l, ∂L/∂t_l = upstream gradients for link l's rotation and translation,
      τ_l = axial_vector(R_l @ (∂L/∂R_l)^T) captures the rotation gradient contribution,
      mask[j,l] = 1 if DOF j is an ancestor of link l (i.e., moving joint j moves link l).
    """

    @staticmethod
    def forward(ctx, th, T_world_link, dof_frame_indices, dof_ancestor_mask, dof_is_revolute, axes):
        # No FK computation — just save what backward needs and pass through.
        ctx.save_for_backward(T_world_link, dof_frame_indices, dof_ancestor_mask,
                              dof_is_revolute, axes)
        return T_world_link

    @staticmethod
    def backward(ctx, grad_output):
        T_world_link, dof_frame_indices, dof_ancestor_mask, dof_is_revolute, axes = \
            ctx.saved_tensors

        # Per-link rotation and translation from FK output
        R_link = T_world_link[:, :, :3, :3]   # (L, B, 3, 3)
        t_link = T_world_link[:, :, :3, 3]    # (L, B, 3)

        # Upstream gradients for each link's rotation and translation
        grad_R_link = grad_output[:, :, :3, :3]  # (L, B, 3, 3)
        grad_t_link = grad_output[:, :, :3, 3]   # (L, B, 3)

        # τ_l: rotation gradient contribution per link — axial vector of R_l @ (∂L/∂R_l)^T
        # For skew-symmetric M, axial_vector extracts the rotation axis.
        M = R_link @ grad_R_link.transpose(-1, -2)  # (L, B, 3, 3)
        tau_link = torch.stack([M[:,:,1,2] - M[:,:,2,1],
                                M[:,:,2,0] - M[:,:,0,2],
                                M[:,:,0,1] - M[:,:,1,0]], dim=-1)  # (L, B, 3)

        # Per-DOF world-frame axis z_j and origin o_j
        T_dof = T_world_link[dof_frame_indices]  # (J, B, 4, 4)
        z_dof = (T_dof[:, :, :3, :3] @ axes.unsqueeze(-1).unsqueeze(1)).squeeze(-1)  # (J, B, 3)
        o_dof = T_dof[:, :, :3, 3]  # (J, B, 3)

        # Weighted sums over descendant links via ancestor mask: (J, L) @ (L, B*3)
        L, B, _ = tau_link.shape
        sum_tau = (dof_ancestor_mask @ tau_link.reshape(L, B * 3)).reshape(-1, B, 3)
        sum_grad_t = (dof_ancestor_mask @ grad_t_link.reshape(L, B * 3)).reshape(-1, B, 3)
        cross_t_grad = torch.cross(t_link, grad_t_link, dim=-1)
        sum_cross = (dof_ancestor_mask @ cross_t_grad.reshape(L, B * 3)).reshape(-1, B, 3)

        # Revolute: z_j · (sum_tau + sum_cross - o_j × sum_grad_t)
        rev_grad = (z_dof * (sum_tau + sum_cross - torch.cross(o_dof, sum_grad_t, dim=-1))).sum(dim=-1)
        # Prismatic: z_j · sum_grad_t
        pris_grad = (z_dof * sum_grad_t).sum(dim=-1)  # (J, B)

        grad_th = torch.where(dof_is_revolute.unsqueeze(-1), rev_grad, pris_grad).T  # (B, J)

        # Grads for: th, T_world_link, dof_frame_indices, dof_ancestor_mask, dof_is_revolute, axes
        return (grad_th,) + (None,) * 5


def get_n_joints(th):
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


def get_batch_size(th):
    if isinstance(th, torch.Tensor) or isinstance(th, np.ndarray):
        return th.shape[0]
    elif isinstance(th, dict):
        elem_shape = get_dict_elem_shape(th)
        return elem_shape[0]
    elif isinstance(th, list):
        # Lists cannot be batched. We don't allow lists of lists.
        return 1
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
        direct_parents = []
        frame_depths = []
        while len(queue) > 0:
            root, parent_idx, depth = queue.pop(0)
            direct_parents.append(parent_idx)
            frame_depths.append(depth)
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
        self.parents_indices = [torch.tensor(p, dtype=torch.long, device=self.device) for p in self.parents_indices]

        # Precomputed structures for torch.compile-compatible FK kernel
        self._num_frames = idx
        self._direct_parent_idx = torch.tensor(direct_parents, dtype=torch.long, device=self.device)

        # BFS levels: group frame indices by depth for level-by-level traversal
        max_depth = max(frame_depths) if frame_depths else 0
        self._bfs_levels = []
        for d in range(max_depth + 1):
            level_frames = [i for i, fd in enumerate(frame_depths) if fd == d]
            self._bfs_levels.append(torch.tensor(level_frames, dtype=torch.long, device=self.device))

        # Static offsets: pre-multiply link_offset @ joint_offset per frame, identity where None
        eye4 = torch.eye(4, dtype=self.dtype, device=self.device)
        static_offsets = []
        for i in range(self._num_frames):
            lo = self.link_offsets[i] if self.link_offsets[i] is not None else eye4.unsqueeze(0)
            jo = self.joint_offsets[i] if self.joint_offsets[i] is not None else eye4.unsqueeze(0)
            static_offsets.append((lo @ jo).squeeze(0))
        self._static_offsets = torch.stack(static_offsets, dim=0)  # (num_frames, 4, 4)

        # Clamped joint indices for safe tensor indexing (fixed joints: -1 → 0)
        self._joint_indices_clamped = self.joint_indices.clamp(min=0)

        # Joint type flags for skipping unnecessary computation in FK
        self._has_revolute = bool((self.joint_type_indices == 1).any())
        self._has_prismatic = bool((self.joint_type_indices == 2).any())

        # Analytical backward data: DOF-to-frame mapping and ancestor masks
        self._dof_frame_indices = torch.zeros(self.n_joints, dtype=torch.long, device=self.device)
        for fi in range(self._num_frames):
            ji = self.joint_indices[fi].item()
            if ji >= 0:
                self._dof_frame_indices[ji] = fi

        # (n_joints, num_frames) float mask: 1.0 if DOF j is an ancestor of frame f
        ancestor_mask = torch.zeros(self.n_joints, self._num_frames, dtype=self.dtype, device=self.device)
        for f in range(self._num_frames):
            ancestor_frames = set(int(x) for x in self.parents_indices[f])
            for j in range(self.n_joints):
                if self._dof_frame_indices[j].item() in ancestor_frames:
                    ancestor_mask[j, f] = 1.0
        self._dof_ancestor_mask = ancestor_mask

        jt = self.joint_type_indices[self._dof_frame_indices]
        self._dof_is_revolute = (jt == 1)  # (n_joints,)

    def to(self, dtype=None, device=None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        self._root = self._root.to(dtype=self.dtype, device=self.device)

        self.identity = self.identity.to(device=self.device, dtype=self.dtype)
        self.parents_indices = [p.to(dtype=torch.long, device=self.device) for p in self.parents_indices]
        self.joint_type_indices = self.joint_type_indices.to(dtype=torch.long, device=self.device)
        self.joint_indices = self.joint_indices.to(dtype=torch.long, device=self.device)
        self.axes = self.axes.to(dtype=self.dtype, device=self.device)
        self.link_offsets = [l if l is None else l.to(dtype=self.dtype, device=self.device) for l in self.link_offsets]
        self.joint_offsets = [j if j is None else j.to(dtype=self.dtype, device=self.device) for j in
                              self.joint_offsets]
        self.low = self.low.to(dtype=self.dtype, device=self.device)
        self.high = self.high.to(dtype=self.dtype, device=self.device)

        self._direct_parent_idx = self._direct_parent_idx.to(device=self.device)
        self._bfs_levels = [l.to(device=self.device) for l in self._bfs_levels]
        self._static_offsets = self._static_offsets.to(dtype=self.dtype, device=self.device)
        self._joint_indices_clamped = self._joint_indices_clamped.to(device=self.device)

        self._dof_frame_indices = self._dof_frame_indices.to(device=self.device)
        self._dof_ancestor_mask = self._dof_ancestor_mask.to(dtype=self.dtype, device=self.device)
        self._dof_is_revolute = self._dof_is_revolute.to(device=self.device)

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
        for j in self.get_joints(exclude_fixed=exclude_fixed):
            if exclude_fixed and j.joint_type == 'fixed':
                continue
            names.append(j.name)
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
    def _get_frame_names(frame: Frame, exclude_fixed=True) -> Sequence[str]:
        names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            names.append(frame.name)
        for child in frame.children:
            names.extend(Chain._get_frame_names(child, exclude_fixed))
        return names

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
        return torch.tensor([self.frame_to_idx[n] for n in frame_names], dtype=torch.long, device=self.device)

    def print_tree(self, do_print=True):
        tree = str(self._root)
        if do_print:
            print(tree)
        return tree

    def forward_kinematics_tensor(self, th, analytical_grad=True):
        """
        Compute forward kinematics for a batch of joint configurations.

        When th.requires_grad is True, backward uses an analytical geometric
        Jacobian by default (~9x faster than autograd on GPU). Set
        analytical_grad=False to use standard autograd instead (needed for
        higher-order gradients or differentiating w.r.t. chain parameters).

        Args:
            th: (B, n_joints) joint angle tensor
            analytical_grad: if True (default), use the analytical geometric
                Jacobian for backward. If False, use standard autograd (supports
                create_graph=True and gradients w.r.t. chain parameters).

        Returns: (num_frames, B, 4, 4) tensor of all frame transforms
        """
        if th.requires_grad and analytical_grad:
            # Compute FK on detached th (no autograd graph needed for forward)
            T_world_link = _fk_impl(
                th.detach(), self._static_offsets, self._joint_indices_clamped,
                self.joint_type_indices, self._direct_parent_idx,
                self._bfs_levels, self.axes,
                self._has_revolute, self._has_prismatic, self._num_frames)
            # Attach analytical backward: connects T_world_link to th in the
            # autograd graph so that backprop uses the geometric Jacobian
            # instead of replaying the forward ops. No FK happens here.
            return _FKAnalyticalBackward.apply(
                th, T_world_link, self._dof_frame_indices, self._dof_ancestor_mask,
                self._dof_is_revolute, self.axes)
        return _fk_impl(th, self._static_offsets, self._joint_indices_clamped,
                        self.joint_type_indices, self._direct_parent_idx,
                        self._bfs_levels, self.axes,
                        self._has_revolute, self._has_prismatic, self._num_frames)

    def forward_kinematics(self, th, frame_indices: Optional = None):
        """
        Compute forward kinematics for the given joint values.

        Args:
            th: A dict, list, numpy array, or torch tensor of joints values. Possibly batched.
            frame_indices: A list of frame indices to compute transforms for. If None, all frames are computed.
                Use `get_frame_indices` to convert from frame names to frame indices.

        Returns:
            A dict of Transform3d objects for each frame.

        """
        if frame_indices is None:
            frame_indices = self.get_all_frame_indices()

        th = self.ensure_tensor(th)
        th = torch.atleast_2d(th)

        all_transforms = self.forward_kinematics_tensor(th)

        return {self.idx_to_frame[fi.item()]: tf.Transform3d(matrix=all_transforms[fi])
                for fi in frame_indices}

    def ensure_tensor(self, th):
        """
        Converts a number of possible types into a tensor. The order of the tensor is determined by the order
        of self.get_joint_parameter_names(). th must contain all joints in the entire chain.
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

    def clamp(self, th):
        """

        Args:
            th: Joint configuration

        Returns: Always a tensor in the order of self.get_joint_parameter_names(), possibly batched.

        """
        th = self.ensure_tensor(th)
        return torch.clamp(th, self.low, self.high)

    def get_joint_limits(self):
        return self._get_joint_limits("limits")

    def get_joint_velocity_limits(self):
        return self._get_joint_limits("velocity_limits")

    def get_joint_effort_limits(self):
        return self._get_joint_limits("effort_limits")

    def _get_joint_limits(self, param_name):
        low = []
        high = []
        for joint in self.get_joints():
            val = getattr(joint, param_name)
            if val is None:
                # NOTE: This changes the previous default behavior of returning
                # +/- np.pi for joint limits to be more natural for both
                # revolute and prismatic joints
                low.append(-np.inf)
                high.append(np.inf)
            else:
                low.append(val[0])
                high.append(val[1])
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


class SerialChain(Chain):
    """
    A serial Chain specialization with no branches and clearly defined end effector.
    Serial chains can be generated from subsets of a Chain.
    """

    def __init__(self, chain, end_frame_name, root_frame_name="", **kwargs):
        root_frame = chain._root if root_frame_name == "" else chain.find_frame(root_frame_name)
        if root_frame is None:
            raise ValueError("Invalid root frame name %s." % root_frame_name)
        chain = Chain(root_frame, **kwargs)

        # make a copy of those frames that includes only the chain up to the end effector
        end_frame_idx = chain.get_frame_indices(end_frame_name)
        ancestors = chain.parents_indices[end_frame_idx]

        frames = []
        # first pass create copies of the ancestor nodes
        for idx in ancestors:
            this_frame_name = chain.idx_to_frame[idx.item()]
            this_frame = copy.deepcopy(chain.find_frame(this_frame_name))
            if idx == end_frame_idx:
                this_frame.children = []
            frames.append(this_frame)
        # second pass assign correct children (only the next one in the frame list)
        for i in range(len(ancestors) - 1):
            frames[i].children = [frames[i + 1]]

        # The root frame's joint and link offset describe the transform from its parent
        # to itself, which is outside this serial chain. Reset them so the root acts as
        # the origin of the new coordinate system.
        frames[0].joint = Joint()
        frames[0].link.offset = None

        self._serial_frames = frames
        super().__init__(frames[0], **kwargs)

        # Precompute data for compilable Jacobian (jacobian_tensor)
        dof_frame_indices = []
        dof_axes = []
        dof_types = []  # 1=revolute, 2=prismatic
        for f in self._serial_frames:
            if f.joint.joint_type != 'fixed':
                fi = self.frame_to_idx[f.name]
                dof_frame_indices.append(fi)
                dof_axes.append(f.joint.axis)
                jtype = 1 if f.joint.joint_type == 'revolute' else 2
                dof_types.append(jtype)

        self._serial_dof_frame_indices = torch.tensor(dof_frame_indices, dtype=torch.long, device=self.device)
        if dof_axes:
            self._serial_dof_axes = torch.stack(dof_axes, dim=0).to(device=self.device, dtype=self.dtype)  # (ndof, 3)
        else:
            self._serial_dof_axes = torch.zeros(0, 3, device=self.device, dtype=self.dtype)
        self._serial_dof_types = torch.tensor(dof_types, dtype=torch.long, device=self.device)  # (ndof,)
        self._serial_eef_frame_idx = self.frame_to_idx[self._serial_frames[-1].name]

    def to(self, dtype=None, device=None):
        super().to(dtype=dtype, device=device)
        self._serial_dof_frame_indices = self._serial_dof_frame_indices.to(device=self.device)
        self._serial_dof_axes = self._serial_dof_axes.to(dtype=self.dtype, device=self.device)
        self._serial_dof_types = self._serial_dof_types.to(device=self.device)
        return self

    def jacobian_tensor(self, th, ret_eef_pose=False, all_transforms=None):
        """
        Compilable Jacobian kernel. Computes the geometric Jacobian in the base frame.
        Compatible with torch.compile(fullgraph=True).

        Args:
            th: (B, n_joints) joint angle tensor
            ret_eef_pose: if True, also return the (B, 4, 4) end-effector pose matrix.
            all_transforms: optional pre-computed (num_frames, B, 4, 4) FK transforms.
                If provided, skips the internal FK computation. Useful to avoid redundant FK calls.

        Returns: (B, 6, ndof) geometric Jacobian, and optionally (B, 4, 4) EEF pose
        """
        if all_transforms is None:
            all_transforms = self.forward_kinematics_tensor(th)  # (num_frames, B, 4, 4)

        p_ee = all_transforms[self._serial_eef_frame_idx, :, :3, 3]  # (B, 3)

        # DOF frame transforms
        T_dof = all_transforms[self._serial_dof_frame_indices]  # (ndof, B, 4, 4)
        R_dof = T_dof[:, :, :3, :3]  # (ndof, B, 3, 3)
        p_dof = T_dof[:, :, :3, 3]   # (ndof, B, 3)

        # Joint axes in base frame: z_i = R_i @ axis_i
        axes = self._serial_dof_axes  # (ndof, 3)
        z = torch.einsum('nbij,nj->nbi', R_dof, axes)  # (ndof, B, 3)

        # Position difference: p_ee - p_i
        dp = p_ee.unsqueeze(0) - p_dof  # (ndof, B, 3)

        # Revolute: J_v = z x dp, J_w = z
        # Prismatic: J_v = z, J_w = 0
        cross = torch.cross(z, dp, dim=2)  # (ndof, B, 3)

        is_rev = (self._serial_dof_types == 1).reshape(-1, 1, 1)  # (ndof, 1, 1)

        J_v = torch.where(is_rev, cross, z)  # (ndof, B, 3)
        J_w = torch.where(is_rev, z, torch.zeros_like(z))  # (ndof, B, 3)

        # Stack to (B, 6, ndof)
        J = torch.cat([J_v, J_w], dim=2)  # (ndof, B, 6)
        J = J.permute(1, 2, 0)  # (B, 6, ndof)

        if ret_eef_pose:
            return J, all_transforms[self._serial_eef_frame_idx]
        return J

    def jacobian(self, th, locations=None, ret_eef_pose=False):
        """
        Compute the geometric Jacobian in the base frame.

        Args:
            th: Joint angles as a dict, list, numpy array, or torch tensor. Possibly batched.
            locations: (B, 3) or (3,) tool offset position relative to the end effector.
            ret_eef_pose: if True, also return the (B, 4, 4) end-effector pose matrix.

        Returns: (B, 6, ndof) Jacobian, and optionally (B, 4, 4) EEF pose
        """
        if not torch.is_tensor(th):
            th = torch.tensor(th, dtype=self.dtype, device=self.device)
        if len(th.shape) <= 1:
            th = th.reshape(1, -1)

        if locations is None and not ret_eef_pose:
            return self.jacobian_tensor(th)

        if locations is not None:
            # Compute FK once and pass to jacobian_tensor to avoid redundant FK
            all_transforms = self.forward_kinematics_tensor(th)
            J = self.jacobian_tensor(th, all_transforms=all_transforms)
            T_ee = all_transforms[self._serial_eef_frame_idx]  # (B, 4, 4)

            if isinstance(locations, tf.Transform3d):
                tool = locations
            else:
                tool = tf.Transform3d(pos=locations)
            if tool.dtype != self.dtype or tool.device != self.device:
                tool = tool.to(device=self.device, copy=True, dtype=self.dtype)
            tool_matrix = tool.get_matrix()
            T_ee_tool = T_ee @ tool_matrix

            # Tool offset only changes p_ee, affecting linear velocity of revolute joints:
            #   J_v_new[i] = z_i x (p_ee_tool - p_i) = J_v_old[i] + z_i x (p_ee_tool - p_ee)
            # Prismatic joints are unaffected (J_v = z, independent of p_ee).
            delta_p = T_ee_tool[:, :3, 3] - T_ee[:, :3, 3]  # (B, 3)

            T_dof = all_transforms[self._serial_dof_frame_indices]
            R_dof = T_dof[:, :, :3, :3]
            z = torch.einsum('nbij,nj->nbi', R_dof, self._serial_dof_axes)  # (ndof, B, 3)

            correction = torch.cross(z, delta_p.unsqueeze(0).expand_as(z), dim=2)  # (ndof, B, 3)
            is_rev = (self._serial_dof_types == 1).reshape(-1, 1, 1)
            correction = torch.where(is_rev, correction, torch.zeros_like(correction))
            J = J.clone()
            J[:, :3, :] = J[:, :3, :] + correction.permute(1, 2, 0)

            T_ee = T_ee_tool
            if ret_eef_pose:
                return J, T_ee
            return J

        # ret_eef_pose=True, no locations: get both from single FK pass
        J, T_ee = self.jacobian_tensor(th, ret_eef_pose=True)
        return J, T_ee

    def forward_kinematics(self, th, end_only: bool = True):
        """ Like the base class, except `th` only needs to contain the joints in the SerialChain, not all joints. """
        frame_indices, th = self.convert_serial_inputs_to_chain_inputs(th, end_only)

        mat = super().forward_kinematics(th, frame_indices)

        if end_only:
            return mat[self._serial_frames[-1].name]
        else:
            return mat

    def convert_serial_inputs_to_chain_inputs(self, th, end_only: bool):
        # th = self.ensure_tensor(th)
        th_b = get_batch_size(th)
        th_n_joints = get_n_joints(th)
        if isinstance(th, list):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)

        if end_only:
            frame_indices = self.get_frame_indices(self._serial_frames[-1].name)
        else:
            # pass through default behavior for frame indices being None, which is currently
            # to return all frames.
            frame_indices = None
        if th_n_joints < self.n_joints:
            # if th is only a partial list of joints, assume it's a list of joints for only the serial chain.
            partial_th = th
            nonfixed_serial_frames = list(filter(lambda f: f.joint.joint_type != 'fixed', self._serial_frames))
            if th_n_joints != len(nonfixed_serial_frames):
                raise ValueError(f'Expected {len(nonfixed_serial_frames)} joint values, got {th_n_joints}.')
            th = torch.zeros([th_b, self.n_joints], device=self.device, dtype=self.dtype)
            for i, frame in enumerate(nonfixed_serial_frames):
                joint_name = frame.joint.name
                if isinstance(partial_th, dict):
                    partial_th_i = partial_th[joint_name]
                else:
                    partial_th_i = partial_th[..., i]
                k = self.frame_to_idx[frame.name]
                jnt_idx = self.joint_indices[k]
                if frame.joint.joint_type != 'fixed':
                    th[..., jnt_idx] = partial_th_i
        return frame_indices, th
