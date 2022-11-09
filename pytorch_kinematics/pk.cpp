#include <torch/extension.h>

using namespace torch::indexing;

torch::Tensor axis_and_angle_to_matrix(torch::Tensor axis,
                                       torch::Tensor theta) {
  /**
   * cos is not that precise for float32, you may want to use float64
   * axis is [b, n, 3]
   * theta is [b, n]
   */
  auto b = axis.size(0);
  auto n = axis.size(1);
  auto m = torch::eye(4).to(axis).unsqueeze(0).unsqueeze(0).repeat({b, n, 1, 1});

  auto kx = axis.index({Ellipsis, 0});
  auto ky = axis.index({Ellipsis, 1});
  auto kz = axis.index({Ellipsis, 2});
  auto c = theta.cos();
  auto one_minus_c = 1 - c;
  auto s = theta.sin();
  auto kxs = kx * s;
  auto kys = ky * s;
  auto kzs = kz * s;
  auto kxky = kx * ky;
  auto kxkz = kx * kz;
  auto kykz = ky * kz;
  m.index_put_({Ellipsis, 0, 0}, c + kx * kx * one_minus_c);
  m.index_put_({Ellipsis, 0, 1}, kxky * one_minus_c - kzs);
  m.index_put_({Ellipsis, 0, 2}, kxkz * one_minus_c + kys);
  m.index_put_({Ellipsis, 1, 0}, kxky * one_minus_c + kzs);
  m.index_put_({Ellipsis, 1, 1}, c + ky * ky * one_minus_c);
  m.index_put_({Ellipsis, 1, 2}, kykz * one_minus_c - kxs);
  m.index_put_({Ellipsis, 2, 0}, kxkz *  one_minus_c - kys);
  m.index_put_({Ellipsis, 2, 1}, kykz *  one_minus_c + kxs);
  m.index_put_({Ellipsis, 2, 2}, c + kz * kz * one_minus_c);
  return m;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("axis_and_angle_to_matrix", &axis_and_angle_to_matrix,
        "axis_and_angle_to_matrix");

}
