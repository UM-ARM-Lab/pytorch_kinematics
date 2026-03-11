def calc_jacobian(serial_chain, th, tool=None, ret_eef_pose=False):
    """
    Return robot Jacobian J in base frame (N,6,DOF) where dot{x} = J dot{q}.
    Delegates to serial_chain.jacobian().
    """
    return serial_chain.jacobian(th, locations=tool, ret_eef_pose=ret_eef_pose)
