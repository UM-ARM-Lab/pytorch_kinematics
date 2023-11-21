""" Generate performance data for multiple models, devices, data types, batch sizes, etc. """
import timeit
from time import perf_counter
import torch

import pytorch_kinematics as pk
import numpy as np


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=220)

    chains = {
        'val':        pk.build_chain_from_mjcf(open('val.xml').read()),
        'val_serial': pk.build_serial_chain_from_mjcf(open('val.xml').read(), end_link_name='left_tool'),
        'kuka_iiwa':  pk.build_serial_chain_from_urdf(open('kuka_iiwa.urdf').read(), end_link_name='lbr_iiwa_link_7'),
    }

    devices = ['cpu', 'cuda']
    dtypes = [torch.float32, torch.float64]
    batch_sizes = [1, 10, 100, 1_000, 10_000, 100_000]
    number = 100

    # iterate over all combinations and store in a pandas dataframe
    headers = ['method', 'chain', 'device', 'dtype', 'batch_size', 'time']
    data = []

    def _fk(th):
        return chain.forward_kinematics(th)

    for name, chain in chains.items():
        for device in devices:
            for dtype in dtypes:
                for batch_size in batch_sizes:
                    chain = chain.to(dtype=dtype, device=device)
                    th = torch.zeros(batch_size, chain.n_joints).to(dtype=dtype, device=device)

                    dt = timeit.timeit(lambda: _fk(th), number=number)
                    data.append([name, device, dtype, batch_size, dt / number])
                    print(f"{name=} {device=} {dtype=} {batch_size=} {dt / number:.4f}")

    # pickle the data for visualization in jupyter notebook
    import pickle
    with open('fk_perf.pkl', 'wb') as f:
        pickle.dump([headers, data], f)


if __name__ == '__main__':
    main()
