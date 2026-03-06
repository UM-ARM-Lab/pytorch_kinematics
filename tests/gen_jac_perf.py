""" Generate performance data for Jacobian: old vs new eager vs compiled. """
import timeit
import torch
import numpy as np
import os

import pytorch_kinematics as pk
from pytorch_kinematics import transforms

TEST_DIR = os.path.dirname(__file__)


def old_calc_jacobian(serial_chain, th):
    """Original reverse-accumulation Jacobian (copied from pre-refactor code)."""
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=serial_chain.dtype, device=serial_chain.device)
    if len(th.shape) <= 1:
        N = 1
        th = th.reshape(1, -1)
    else:
        N = th.shape[0]
    ndof = th.shape[1]

    j_eef = torch.zeros((N, 6, ndof), dtype=serial_chain.dtype, device=serial_chain.device)

    cur_transform = transforms.Transform3d(device=serial_chain.device,
                                           dtype=serial_chain.dtype).get_matrix().repeat(N, 1, 1)

    cnt = 0
    for f in reversed(serial_chain._serial_frames):
        if f.joint.joint_type == "revolute":
            cnt += 1
            axis_in_eef = cur_transform[:, :3, :3].transpose(1, 2) @ f.joint.axis
            eef2joint_pos_in_joint = cur_transform[:, :3, 3].unsqueeze(2)
            joint2eef_rot = cur_transform[:, :3, :3].transpose(1, 2)
            eef2joint_pos_in_eef = joint2eef_rot @ eef2joint_pos_in_joint
            position_jacobian = torch.cross(axis_in_eef, eef2joint_pos_in_eef.squeeze(2), dim=1)
            j_eef[:, :, -cnt] = torch.cat((position_jacobian, axis_in_eef), dim=-1)
        elif f.joint.joint_type == "prismatic":
            cnt += 1
            j_eef[:, :3, -cnt] = (f.joint.axis.repeat(N, 1, 1) @ cur_transform[:, :3, :3])[:, 0, :]
        cur_frame_transform = f.get_transform(th[:, -cnt]).get_matrix()
        cur_transform = cur_frame_transform @ cur_transform
        if f.link.offset is not None:
            cur_transform = f.link.offset.get_matrix() @ cur_transform

    pose = cur_transform
    rotation = pose[:, :3, :3]
    j_tr = torch.zeros((N, 6, 6), dtype=serial_chain.dtype, device=serial_chain.device)
    j_tr[:, :3, :3] = rotation
    j_tr[:, 3:, 3:] = rotation
    j_w = j_tr @ j_eef
    return j_w


def bench(fn, number, device):
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = timeit.timeit(fn, number=number)
    if device == 'cuda':
        torch.cuda.synchronize()
    return dt / number


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=220)

    chains = {
        'kuka_iiwa': pk.build_serial_chain_from_urdf(
            open(os.path.join(TEST_DIR, 'kuka_iiwa.urdf')).read(),
            end_link_name='lbr_iiwa_link_7'),
        'val_serial': pk.build_serial_chain_from_mjcf(
            open(os.path.join(TEST_DIR, 'val.xml')).read(),
            end_link_name='left_tool'),
    }

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    dtype = torch.float32
    batch_sizes = [1, 10, 100, 1_000, 10_000, 100_000]
    number = 50
    warmup = 5

    headers = ['method', 'chain', 'device', 'dtype', 'batch_size', 'time']
    data = []

    for name, chain in chains.items():
        for device in devices:
            chain = chain.to(dtype=dtype, device=device)

            print(f"\nCompiling jacobian_tensor for {name} on {device}...", flush=True)
            th_dummy = torch.randn(2, chain.n_joints, dtype=dtype, device=device)
            compiled_jac = torch.compile(chain.jacobian_tensor, fullgraph=True)
            compiled_jac(th_dummy)  # trigger compilation
            print(f"  Done compiling.\n", flush=True)

            for batch_size in batch_sizes:
                th = torch.randn(batch_size, chain.n_joints, dtype=dtype, device=device)

                # Warmup all methods
                for _ in range(warmup):
                    old_calc_jacobian(chain, th)
                    chain.jacobian_tensor(th)
                    compiled_jac(th)

                t_old = bench(lambda: old_calc_jacobian(chain, th), number, device)
                t_new = bench(lambda: chain.jacobian_tensor(th), number, device)
                t_compiled = bench(lambda: compiled_jac(th), number, device)

                data.append(['old', name, device, str(dtype), batch_size, t_old])
                data.append(['new_eager', name, device, str(dtype), batch_size, t_new])
                data.append(['compiled', name, device, str(dtype), batch_size, t_compiled])

                speedup_new = t_old / t_new if t_new > 0 else float('inf')
                speedup_compiled = t_old / t_compiled if t_compiled > 0 else float('inf')
                print(f"  {name:12s} {device:5s} B={batch_size:>6d}  "
                      f"old={t_old*1000:8.3f}ms  "
                      f"new_eager={t_new*1000:8.3f}ms ({speedup_new:.2f}x)  "
                      f"compiled={t_compiled*1000:8.3f}ms ({speedup_compiled:.2f}x)",
                      flush=True)

    import pickle
    with open(os.path.join(TEST_DIR, 'jac_perf.pkl'), 'wb') as f:
        pickle.dump([headers, data], f)


if __name__ == '__main__':
    main()
