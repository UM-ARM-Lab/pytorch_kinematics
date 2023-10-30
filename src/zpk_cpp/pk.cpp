#include <torch/extension.h>

#include <optional>
#include <vector>

using namespace torch::indexing;

torch::Tensor axis_and_d_to_pris_matrix(torch::Tensor axis, torch::Tensor d) {
  /**
   * axis is [b, n, 3]
   * d is [b, n]
   *
   * Output is [b, n, 4, 4]
   */
  auto b = axis.size(0);
  auto n = axis.size(1);
  auto mat44 = torch::eye(4).to(axis).repeat({b, n, 1, 1});
  auto pos = axis * d.unsqueeze(-1);
  mat44.index_put_({Ellipsis, Slice(0, 3), 3}, pos);
  return mat44;
}

torch::Tensor axis_and_angle_to_matrix(torch::Tensor axis, torch::Tensor theta) {
  /**
   * cos is not that precise for float32, you may want to use float64
   * axis is [b, n, 3]
   * theta is [b, n]
   * Return is [b, n, 4, 4]
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
  m.index_put_({Ellipsis, 2, 0}, kxkz * one_minus_c - kys);
  m.index_put_({Ellipsis, 2, 1}, kykz * one_minus_c + kxs);
  m.index_put_({Ellipsis, 2, 2}, c + kz * kz * one_minus_c);
  return m;
}

std::map<int64_t, torch::Tensor> fk(torch::Tensor frame_indices, torch::Tensor axes, torch::Tensor th,
                              std::vector<torch::Tensor> parents_indices, torch::Tensor joint_type_indices,
                              torch::Tensor joint_indices, std::vector<std::optional<torch::Tensor>> joint_offsets,
                              std::vector<std::optional<torch::Tensor>> link_offsets) {
  std::map<int64_t, torch::Tensor> frame_transforms;

  auto const b = th.size(0);
  auto const axes_expanded = axes.unsqueeze(0).repeat({b, 1, 1});

  auto const rev_jnt_transform = axis_and_angle_to_matrix(axes_expanded, th);
  auto const pris_jnt_transform = axis_and_d_to_pris_matrix(axes_expanded, th);

  for (int64_t i = 0; i < frame_indices.size(0); i++) {
    auto const frame_idx = frame_indices[i].item<int64_t>();
    auto frame_transform = torch::eye(4).to(th).unsqueeze(0).expand({b, 4, 4});

    auto const parent_indices = parents_indices[frame_idx];
    for (int64_t j = 0; j < parent_indices.size(0); j++) {
      auto const chain_idx_tensor = parent_indices[j];
      int64_t chain_idx_int = chain_idx_tensor.item<int64_t>();
      if (frame_transforms.find(chain_idx_int) != frame_transforms.end()) {
        frame_transform = frame_transforms[chain_idx_int];
      } else {
        auto const link_offset_i = link_offsets[chain_idx_int];
        if (link_offset_i) {
          frame_transform = frame_transform.matmul(*link_offset_i);
        }

        auto const joint_offset_i = joint_offsets[chain_idx_int];
        if (joint_offset_i) {
          frame_transform = frame_transform.matmul(*joint_offset_i);
        }

        auto const jnt_idx = joint_indices.index({chain_idx_tensor}).item<int64_t>();
        auto const jnt_type = joint_type_indices.index({chain_idx_tensor}).item<int64_t>();
        if (jnt_type == 1) {
          auto const jnt_transform_i = rev_jnt_transform.index({Slice(), jnt_idx});
          frame_transform = frame_transform.matmul(jnt_transform_i);
        } else if (jnt_type == 2) {
          auto const jnt_transform_i = pris_jnt_transform.index({Slice(), jnt_idx});
          frame_transform = frame_transform.matmul(jnt_transform_i);
        }
      }
    }
    frame_transforms.emplace(frame_idx, frame_transform);
  }
  return frame_transforms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fk", &fk, "fk"); }
