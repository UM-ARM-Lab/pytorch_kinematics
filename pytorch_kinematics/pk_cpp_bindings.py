import torch
import pk_cpp


class AxisAndAngleToMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward():
        outputs = pk_cpp.axis_and_angle_to_matrix()
        # ctx.save_for_backward(*)
        return None
