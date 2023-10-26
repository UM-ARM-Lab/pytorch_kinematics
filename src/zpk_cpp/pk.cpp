#include <optional>
#include <torch/extension.h>
#include <vector>

using namespace torch::indexing;

torch::Tensor axis_and_angle_to_matrix(torch::Tensor axis,
                                       torch::Tensor theta) {
  /**
   * cos is not that precise for float32, you may want to use float64
   * axis is [b, n, 3]
   * theta is [b, n]
   * Return is [b, n, 4, 4]
   */
  auto b = axis.size(0);
  auto n = axis.size(1);
  auto m =
      torch::eye(4).to(axis).unsqueeze(0).unsqueeze(0).repeat({b, n, 1, 1});

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
  m.index_put_({Ellipsis, 2, 0}, kxkz * one_minus_c - kys);
  m.index_put_({Ellipsis, 2, 1}, kykz * one_minus_c + kxs);
  m.index_put_({Ellipsis, 2, 2}, c + kz * kz * one_minus_c);
  return m;
}

std::vector<torch::Tensor>
fk(torch::Tensor link_indices, torch::Tensor axes, torch::Tensor th,
   std::vector<int> parent_indices, std::vector<bool> is_fixed,
   torch::Tensor joint_indices,
   std::vector<std::optional<torch::Tensor>> joint_offsets,
   std::vector<std::optional<torch::Tensor>> link_offsets) {
  std::vector<torch::Tensor> link_transforms;

  auto b = th.size(0);
  // NOTE: assumes revolute joint!
  auto const jnt_transform = axis_and_angle_to_matrix(axes, th);

  for (auto i{0}; i < link_indices.size(0); ++i) {
    auto idx = link_indices.index({i}).item().to<int>();
    auto link_transform = torch::eye(4).to(th).unsqueeze(0).repeat({b, 1, 1});

    while (idx >= 0) {
      auto const joint_offset_i = joint_offsets[idx];
      if (joint_offset_i) {
        link_transform = torch::matmul(*joint_offset_i, link_transform);
      }

      if (!is_fixed[idx]) {
        auto const jnt_idx = joint_indices[idx];
        auto const jnt_transform_i = jnt_transform.index({Slice(), jnt_idx});
        link_transform = torch::matmul(jnt_transform_i, link_transform);
      }

      auto const link_offset_i = link_offsets[idx];
      if (link_offset_i) {
        link_transform = torch::matmul(*link_offset_i, link_transform);
      }

      idx = parent_indices[idx];
    }

     link_transforms.emplace_back(link_transform);
  }
  return link_transforms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("axis_and_angle_to_matrix", &axis_and_angle_to_matrix,
        "axis_and_angle_to_matrix", py::arg("axis"), py::arg("theta"));
  m.def("fk", &fk, "fk");
}
