import torch
import zpk_cpp


class FK(torch.autograd.Function):
    @staticmethod
    def forward(ctx ):
        outputs = zpk_cpp.fk()
        ctx.save_for_backward(*)
        return None

    @staticmethod
    def backward(ctx):
        outputs = zpk_cpp.jacobian()
        return None

fk = FK.apply
