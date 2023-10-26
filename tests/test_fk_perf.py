import timeit
from time import perf_counter
import torch

import pytorch_kinematics as pk
import numpy as np

N = 10000
number = 100


def test_val_fk_perf():
    val = pk.build_serial_chain_from_mjcf(open('val.xml').read(), end_link_name='left_tool')
    val = val.to(dtype=torch.float32, device='cuda')

    th = torch.zeros(N, 20, dtype=torch.float32, device='cuda')
    print(len(val.get_joint_parameter_names()))

    def _val_old_fk():
        tg = val.forward_kinematics_slow(th, end_only=True)
        m = tg.get_matrix()
        return m

    def _val_new_fk():
        tg = val.forward_kinematics(th, end_only=True)
        m = tg.get_matrix()
        return m

    val_old_dt = timeit.timeit(_val_old_fk, number=number)
    print(f'Val FK OLD dt: {val_old_dt / number:.4f}')

    val_new_dt = timeit.timeit(_val_new_fk, number=number)
    print(f'Val FK NEW dt: {val_new_dt / number:.4f}')

    assert val_old_dt > val_new_dt


def test_kuka_fk_perf():
    kuka = pk.build_serial_chain_from_urdf(open('kuka_iiwa.urdf').read(), end_link_name='lbr_iiwa_link_7')
    kuka = kuka.to(dtype=torch.float32, device='cuda')

    th = torch.zeros(N, 7, dtype=torch.float32, device='cuda')

    def _kuka_old_fk():
        tg = kuka.forward_kinematics_slow(th, end_only=True)
        m = tg.get_matrix()
        return m

    def _kuka_new_fk():
        tg = kuka.forward_kinematics(th, end_only=True)
        m = tg.get_matrix()
        return m

    kuka_old_dt = timeit.timeit(_kuka_old_fk, number=number)
    print(f'Kuka FK OLD dt: {kuka_old_dt / number:.4f}')

    kuka_new_dt = timeit.timeit(_kuka_new_fk, number=number)
    print(f'Kuka FK NEW dt: {kuka_new_dt / number:.4f}')

    assert kuka_old_dt > kuka_new_dt


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=220)

    test_val_fk_perf()
    test_kuka_fk_perf()
