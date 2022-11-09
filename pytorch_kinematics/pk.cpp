#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace torch::indexing;

// NOTE: cos is not that precise for float32, you may want to use float64
torch::Tensor axis_and_angle_to_matrix(torch::Tensor axis,
                                       torch::Tensor theta) {
  auto c = theta.cos();
  auto one_minus_c = 1 - c;
  auto s = theta.sin();
  auto kx = axis.index({Ellipsis, 0});
  auto ky = axis.index({Ellipsis, 1});
  auto kz = axis.index({Ellipsis, 2});
  auto r00 = c + kx * kx * one_minus_c;
  auto r01 = kx * ky * one_minus_c - kz * s;
  auto r02 = kx * kz * one_minus_c + ky * s;
  auto r10 = ky * kx * one_minus_c + kz * s;
  auto r11 = c + ky * ky * one_minus_c;
  auto r12 = ky * kz * one_minus_c - kx * s;
  auto r20 = kz * kx * one_minus_c - ky * s;
  auto r21 = kz * ky * one_minus_c + kx * s;
  auto r22 = c + kz * kz * one_minus_c;
  // Do negatives work???
  auto r0 = at::cat({r00, r01, r02}, -1);
  auto r1 = at::cat({r10, r11, r12}, -1);
  auto r2 = at::cat({r20, r21, r22}, -1);
  auto rot = torch::stack({r0, r1, r2}, -2);
  return rot;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("axis_and_angle_to_matrix", &axis_and_angle_to_matrix,
        "axis_and_angle_to_matrix");
}
