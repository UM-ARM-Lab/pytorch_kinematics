import timeit
from time import perf_counter
import torch

import pytorch_kinematics as pk
import numpy as np

N = 10000
number = 100


def test_val_fk_correctness():
    val = pk.build_chain_from_mjcf(open('val.xml').read())
    val = val.to(dtype=torch.float32, device='cuda')

    th = torch.zeros(N, 20, dtype=torch.float32, device='cuda')

    frame_indices = val.get_frame_indices('left_tool', 'right_tool')
    t_py = val.forward_kinematics_py(th, frame_indices)
    t_cpp = val.forward_kinematics(th, frame_indices)
    l_py = t_py['left_tool'].get_matrix()
    l_cpp = t_cpp['left_tool'].get_matrix()
    r_py = t_py['right_tool'].get_matrix()
    r_cpp = t_cpp['right_tool'].get_matrix()

    assert torch.allclose(l_py, l_cpp)
    assert torch.allclose(r_py, r_cpp)


def test_val_fk_perf():
    val = pk.build_serial_chain_from_mjcf(open('val.xml').read(), end_link_name='left_tool')
    val = val.to(dtype=torch.float32, device='cuda')

    th = torch.zeros(N, 20, dtype=torch.float32, device='cuda')

    def _val_old_fk():
        tg = val.forward_kinematics_slow(th, end_only=True)
        m = tg.get_matrix()
        return m

    def _val_new_py_fk():
        tg = val.forward_kinematics_py(th, end_only=True)
        m = tg.get_matrix()
        return m

    def _val_new_cpp_fk():
        tg = val.forward_kinematics(th, end_only=True)
        m = tg.get_matrix()
        return m

    val_old_dt = timeit.timeit(_val_old_fk, number=number)
    print(f'Val FK OLD dt: {val_old_dt / number:.4f}')

    val_new_py_dt = timeit.timeit(_val_new_py_fk, number=number)
    print(f'Val FK NEW dt: {val_new_py_dt / number:.4f}')

    val_new_cpp_dt = timeit.timeit(_val_new_cpp_fk, number=number)
    print(f'Val FK NEW C++ dt: {val_new_cpp_dt / number:.4f}')

    assert val_old_dt > val_new_cpp_dt


def test_kuka_fk_perf():
    kuka = pk.build_serial_chain_from_urdf(open('kuka_iiwa.urdf').read(), end_link_name='lbr_iiwa_link_7')
    kuka = kuka.to(dtype=torch.float32, device='cuda')

    th = torch.zeros(N, 7, dtype=torch.float32, device='cuda')

    def _kuka_old_fk():
        tg = kuka.forward_kinematics_slow(th, end_only=True)
        m = tg.get_matrix()
        return m

    def _kuka_new_py_fk():
        tg = kuka.forward_kinematics_py(th, end_only=True)
        m = tg.get_matrix()
        return m

    def _kuka_new_cpp_fk():
        tg = kuka.forward_kinematics(th, end_only=True)
        m = tg.get_matrix()
        return m

    kuka_old_dt = timeit.timeit(_kuka_old_fk, number=number)
    print(f'Kuka FK OLD dt: {kuka_old_dt / number:.4f}')

    kuka_new_py_dt = timeit.timeit(_kuka_new_py_fk, number=number)
    print(f'Kuka FK NEW dt: {kuka_new_py_dt / number:.4f}')

    kuka_new_cpp_dt = timeit.timeit(_kuka_new_cpp_fk, number=number)
    print(f'Kuka FK NEW C++ dt: {kuka_new_cpp_dt / number:.4f}')

    assert kuka_old_dt > kuka_new_cpp_dt


def main():
    # do an in-depth analysis of multiple models, devices, data types, batch sizes, etc.
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=220)

    chains = [
        pk.build_chain_from_mjcf(open('val.xml').read()),
        pk.build_serial_chain_from_mjcf(open('val.xml').read(), end_link_name='left_tool'),
        pk.build_serial_chain_from_urdf(open('kuka_iiwa.urdf').read(), end_link_name='lbr_iiwa_link_7'),
    ]
    names = ['val', 'val_serial', 'kuka_iiwa']

    devices = ['cpu', 'cuda']
    dtypes = [torch.float32, torch.float64]
    batch_sizes = [1, 10, 100, 1_000, 10_000, 100_000]

    # iterate over all combinations and store in a pandas dataframe
    headers = ['chain', 'device', 'dtype', 'batch_size', 'time']
    data = []

    for chain, name in zip(chains, names):
        for device in devices:
            for dtype in dtypes:
                for batch_size in batch_sizes:
                    chain = chain.to(dtype=dtype, device=device)
                    th = torch.zeros(batch_size, chain.n_joints).to(dtype=dtype, device=device)

                    dt = timeit.timeit(lambda: chain.forward_kinematics(th), number=number)
                    data.append([name, device, dtype, batch_size, dt / number])
                    print(f"{name=} {device=} {dtype=} {batch_size=} {dt / number:.4f}")

    # pickle the data for visualization in jupyter notebook
    import pickle
    with open('fk_perf.pkl', 'wb') as f:
        pickle.dump([headers, data], f)


if __name__ == '__main__':
    main()
