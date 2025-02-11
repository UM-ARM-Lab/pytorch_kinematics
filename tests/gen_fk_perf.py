""" Generate performance data for multiple models, devices, data types, batch sizes, etc. """
import timeit
from time import perf_counter
import torch

import pytorch_kinematics as pk
import numpy as np


def main():
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
    number = 100

    # iterate over all combinations and store in a pandas dataframe
    headers = ['method', 'chain', 'device', 'dtype', 'batch_size', 'time']
    data = []

    def _fk_cpp(th):
        return chain.forward_kinematics(th)

    @torch.compile(backend='eager')
    def _fk_torch_compile(th):
        return chain.forward_kinematics_py(th)

    method_names = ['fk_cpp', 'fk_torch_compile']
    methods = [_fk_cpp, _fk_torch_compile]

    for chain, name in zip(chains, names):
        for device in devices:
            for dtype in dtypes:
                for batch_size in batch_sizes:
                    for method_name, method in zip(method_names, methods):
                        chain = chain.to(dtype=dtype, device=device)
                        th = torch.zeros(batch_size, chain.n_joints).to(dtype=dtype, device=device)

                        dt = timeit.timeit(lambda: method(th), number=number)
                        data.append([name, device, dtype, batch_size, dt / number])
                        print(f"{method_name} {name=} {device=} {dtype=} {batch_size=} {dt / number:.4f}")

    # pickle the data for visualization in jupyter notebook
    import pickle
    with open('fk_perf.pkl', 'wb') as f:
        pickle.dump([headers, data], f)


if __name__ == '__main__':
    main()
