import torch

import pytorch_kinematics.transforms as tf
from . import jacobian
from .frame import Joint
from copy import deepcopy


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
    def __init__(self, root_frame, dtype=torch.float32, device="cpu", end_frame_names: list=None):
        self._root = root_frame
        self.dtype = dtype
        self.device = device
        self.end_frame_names = end_frame_names

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
    def _find_frame_recursive(name, frame):
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_frame(self, name):
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name, frame):
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name):
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    @staticmethod
    def _get_joint_parameter_names(frame, exclude_fixed=True):
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.joint.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parameter_names(child, exclude_fixed))
        return joint_names

    def get_joint_parameter_names(self, exclude_fixed=True):
        names = self._get_joint_parameter_names(self._root, exclude_fixed)
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

    def forward_kinematics(self, th, world=None):
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

    def forward_kinematics_end(self, th, world=None):
        position_dict = self.forward_kinematics(th, world)
        ee_pos_dict = {k:v for k,v in position_dict.items() if (k+'_frame') in self.end_frame_names}
        ee_pos_list = [ee_pos_dict[k.rstrip('_frame')].get_matrix() for k in self.end_frame_names]
        ee_pos_list = torch.stack(ee_pos_list, dim=1)
        return ee_pos_list
    def jacobian(self, th, locations=None):
        #TODO: do not support locations / tools now
        assert locations == None
        if not isinstance(th, dict):
            th, _ = ensure_2d_tensor(th, self.dtype, self.device)
            jn = self.get_joint_parameter_names()
            assert len(jn) == th.shape[-1]
            th_dict = dict((j, th[..., i]) for i, j in enumerate(jn))
        else:
            th_dict = {k: ensure_2d_tensor(v, self.dtype, self.device)[0] for k, v in th.items()}
        jac = torch.zeros((th.shape[0],6*(len(self.end_frame_names)), th.shape[-1]), device=self.device, dtype=self.dtype)

        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        
        for i, ee_frame_name in enumerate(self.end_frame_names):
            serial_chain_names = [self._root] + SerialChain._generate_serial_chain_recurse(self._root, ee_frame_name)
            serial_chain_joint_names = [frame.joint.name for frame in serial_chain_names if frame.joint.name in th_dict.keys()]
            serial_chain_th = [th_dict[frame.joint.name] for frame in serial_chain_names if frame.joint.name in th_dict.keys()]
            serial_chain_th = torch.stack(serial_chain_th, dim=1)
            serial_chain = SerialChain(self, ee_frame_name)
            jac_chain = jacobian.calc_jacobian(serial_chain, serial_chain_th, tool=locations)
            index_list = [list(th_dict).index(joint_name) for joint_name in serial_chain_joint_names]
            assert (jac[:, i*6:(i+1)*6, index_list] == 0).all() # make sure the jacobian slots are empty
            jac[:, i*6:(i+1)*6, index_list] = jac_chain
        return jac
    
    def jacobian_and_hessian(self, th, locations=None):
        #TODO: do not support locations / tools now
        assert locations == None
        if not isinstance(th, dict):
            th, _ = ensure_2d_tensor(th, self.dtype, self.device)
            jn = self.get_joint_parameter_names()
            assert len(jn) == th.shape[-1]
            th_dict = dict((j, th[..., i]) for i, j in enumerate(jn))
        else:
            th_dict = {k: ensure_2d_tensor(v, self.dtype, self.device)[0] for k, v in th.items()}
        jac = torch.zeros((th.shape[0],6*(len(self.end_frame_names)), th.shape[-1]), device=self.device, dtype=self.dtype)
        H = torch.zeros((th.shape[0],6*(len(self.end_frame_names)), th.shape[-1], th.shape[-1]), device=self.device, dtype=self.dtype)

        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        for i, ee_frame_name in enumerate(self.end_frame_names):
            serial_chain_names = [self._root] + SerialChain._generate_serial_chain_recurse(self._root, ee_frame_name)
            serial_chain_joint_names = [frame.joint.name for frame in serial_chain_names if frame.joint.name in th_dict.keys()]
            serial_chain_th = [th_dict[frame.joint.name] for frame in serial_chain_names if frame.joint.name in th_dict.keys()]
            serial_chain_th = torch.stack(serial_chain_th, dim=1)
            serial_chain = SerialChain(self, ee_frame_name)
            jac_chain, H_chain = jacobian.calc_jacobian_and_hessian(serial_chain, serial_chain_th, tool=locations)
            index_list = [list(th_dict).index(joint_name) for joint_name in serial_chain_joint_names]
            assert (jac[:, i*6:(i+1)*6, index_list] == 0).all() # make sure the jacobian slots are empty
            assert (H[:, i*6:(i+1)*6, index_list, index_list] == 0).all() # make sure the jacobian slots are empty
            jac[:, i*6:(i+1)*6, index_list] = jac_chain
            x_coor, y_coor = torch.meshgrid(torch.LongTensor(index_list), torch.LongTensor(index_list), indexing='ij')
            H[:, i*6:(i+1)*6, x_coor, y_coor] = H_chain
        return jac, H

    def jacobian_and_hessian_dhessian(self, th, locations=None):
        #TODO: do not support locations / tools now
        assert locations == None
        if not isinstance(th, dict):
            th, _ = ensure_2d_tensor(th, self.dtype, self.device)
            jn = self.get_joint_parameter_names()
            assert len(jn) == th.shape[-1]
            th_dict = dict((j, th[..., i]) for i, j in enumerate(jn))
        else:
            th_dict = {k: ensure_2d_tensor(v, self.dtype, self.device)[0] for k, v in th.items()}
        jac = torch.zeros((th.shape[0],6*(len(self.end_frame_names)), th.shape[-1]), device=self.device, dtype=self.dtype)
        H = torch.zeros((th.shape[0],6*(len(self.end_frame_names)), th.shape[-1], th.shape[-1]), device=self.device, dtype=self.dtype)
        dH = torch.zeros((th.shape[0],6*(len(self.end_frame_names)), th.shape[-1], th.shape[-1], th.shape[-1]), device=self.device, dtype=self.dtype)

        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        for i, ee_frame_name in enumerate(self.end_frame_names):
            serial_chain_names = [self._root] + SerialChain._generate_serial_chain_recurse(self._root, ee_frame_name)
            serial_chain_joint_names = [frame.joint.name for frame in serial_chain_names if frame.joint.name in th_dict.keys()]
            serial_chain_th = [th_dict[frame.joint.name] for frame in serial_chain_names if frame.joint.name in th_dict.keys()]
            serial_chain_th = torch.stack(serial_chain_th, dim=1)
            serial_chain = SerialChain(self, ee_frame_name)
            jac_chain, H_chain, dH_chain = jacobian.calc_jacobian_hessian_dhessian(serial_chain, serial_chain_th, tool=locations)
            index_list = [list(th_dict).index(joint_name) for joint_name in serial_chain_joint_names]
            assert (jac[:, i*6:(i+1)*6, index_list] == 0).all() # make sure the jacobian slots are empty
            assert (H[:, i*6:(i+1)*6, index_list, index_list] == 0).all() # make sure the jacobian slots are empty
            jac[:, i*6:(i+1)*6, index_list] = jac_chain
            x_coor, y_coor = torch.meshgrid(torch.LongTensor(index_list), torch.LongTensor(index_list), indexing='ij')
            H[:, i*6:(i+1)*6, x_coor, y_coor] = H_chain
            x_coor, y_coor, z_coor = torch.meshgrid(torch.LongTensor(index_list), torch.LongTensor(index_list), torch.LongTensor(index_list), indexing='ij')
            dH[:, i*6:(i+1)*6, x_coor, y_coor, z_coor] = dH_chain
        return jac, H, dH


class SerialChain(Chain):
    def __init__(self, chain, end_frame_name, root_frame_name="", **kwargs):
        if root_frame_name == "":
            super(SerialChain, self).__init__(chain._root, **kwargs)
        else:
            # if it does not start from the actual root, then we should not consider the joint 
            # before the first link
            new_root = deepcopy(chain.find_frame(root_frame_name))
            new_root.joint = Joint()
            super(SerialChain, self).__init__(new_root, **kwargs)
            if self._root is None:
                raise ValueError("Invalid root frame name %s." % root_frame_name)
        self._serial_frames = [self._root] + self._generate_serial_chain_recurse(self._root, end_frame_name)
        if self._serial_frames is None:
            raise ValueError("Invalid end frame name %s." % end_frame_name)

    @staticmethod
    def _generate_serial_chain_recurse(root_frame, end_frame_name):
        for child in root_frame.children:
            # print(child.name, end_frame_name)
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

        return link_transforms[self._serial_frames[-1].link.name].get_matrix()  # if end_only else link_transforms

    def jacobian(self, th, locations=None):
        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        return jacobian.calc_jacobian(self, th, tool=locations)

    def jacobian_and_hessian(self, th, locations=None):
        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        return jacobian.calc_jacobian_and_hessian(self, th, tool=locations)

    def jacobian_and_hessian_and_dhessian(self, th, locations=None):
        if locations is not None:
            locations = tf.Transform3d(pos=locations)
        return jacobian.calc_jacobian_hessian_dhessian(self, th, tool=locations)