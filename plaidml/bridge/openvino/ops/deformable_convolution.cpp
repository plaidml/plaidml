// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace edsl;

namespace PlaidMLPlugin {

size_t compute_padding_before(size_t input_size, size_t off_size, size_t filter_size, size_t stride,
                              plaidml::op::AutoPadMode autopad_mode, size_t pad_before, size_t pad_end,
                              size_t dilation) {
  size_t f = (dilation * (filter_size - 1)) + 1;
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    return pad_before;
  }
  if (autopad_mode == plaidml::op::AutoPadMode::VALID) {
    return 0;
  }
  if (autopad_mode == plaidml::op::AutoPadMode::SAME_LOWER || autopad_mode == plaidml::op::AutoPadMode::SAME_UPPER) {
    size_t lower_term = (autopad_mode == plaidml::op::AutoPadMode::SAME_LOWER) ? 1 : 0;
    size_t max = 0;
    if (((off_size - 1) * stride + f - input_size) > 0) {
      max = (off_size - 1) * stride + f - input_size;
    }
    pad_before = (max + lower_term) / 2;
    return pad_before;
  }
  THROW_IE_EXCEPTION << "Unexpected autopadding mode.";
}

size_t compute_output_size(size_t input_size, size_t filter_size, size_t stride, plaidml::op::AutoPadMode autopad_mode,
                           size_t pad_lo, size_t pad_hi, size_t dilation) {
  auto I_eff = input_size;
  auto F_eff = (dilation * (filter_size - 1)) + 1;
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    size_t output_size = ((I_eff + pad_lo + pad_hi - F_eff + stride) / stride);
    return output_size;
  }
  if (autopad_mode == plaidml::op::AutoPadMode::VALID) {
    size_t output_size = ((I_eff - F_eff + stride) / stride);
    return output_size;
  }
  if (autopad_mode == plaidml::op::AutoPadMode::SAME_LOWER || autopad_mode == plaidml::op::AutoPadMode::SAME_UPPER) {
    size_t output_size = ((I_eff + stride - 1) / stride);
    return output_size;
  }
  THROW_IE_EXCEPTION << "Unexpected autopadding mode.";
}

edsl::Tensor extract_tensor(edsl::Tensor I, size_t rank) {
  std::vector<TensorDim> I_dims(rank * (rank + 2) + 2), O_dims(rank + 2);
  std::vector<TensorIndex> I_idxs(rank * (rank + 2) + 2), O_idxs(rank + 2);
  I.bind_dims(I_dims);
  // Connect dimension between input and output.
  for (auto i = 2; i < rank + 4; ++i) {
    O_dims[i - 2] = I_dims[i];
  }
  // Set up batch_size and channel.
  for (auto i = 0; i < 2; ++i) {
    I_idxs[i + 2] = I_idxs[i];
  }
  // Set up index of input.
  for (auto i = 2; i < rank + 4; ++i) {
    for (auto j = 1; j < rank; ++j) {
      I_idxs[j * (rank + 2) + i] = I_idxs[i];
    }
  }
  // Connect index between input and output.
  for (auto i = 2; i < rank + 4; ++i) {
    O_idxs[i - 2] = I_idxs[i];
  }
  edsl::Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return O;
}

void registerDeformableConvolution() {
  registerOp("DeformableConvolution", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::DeformableConvolution>(ctx.layer);
    DType precision = to_plaidml(layer->get_output_element_type(0));
    IE_ASSERT(ctx.operands.size() == 3);
    auto* input_constant_operand = ngraph::as_type<ngraph::op::Constant>(ctx.layer->get_input_node_ptr(2));
    if (input_constant_operand == nullptr) {
      THROW_IE_EXCEPTION << "Filter need to be constant node.";
    }
    auto I = ctx.operands.at(0), OFF = ctx.operands.at(1), F = ctx.operands.at(2);
    auto I_shape = I.compute_shape().sizes(), OFF_shape = OFF.compute_shape().sizes(),
         F_shape = F.compute_shape().sizes();
    auto N = I_shape[0], CI = I_shape[1], CO = F_shape[0], OFF_C = OFF_shape[0];
    auto G = layer->get_group(), DG = layer->get_deformable_group();
    auto rank = I.rank();
    // Get autopad_mode.
    auto autopad_mode = to_plaidml(layer->get_auto_pad());
    // Compute manual_padding.
    std::vector<size_t> manual_padding;
    if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
      for (auto pad : layer->get_pads_begin()) {
        manual_padding.push_back(pad);
      }
      for (auto pad : layer->get_pads_end()) {
        manual_padding.push_back(pad);
      }
      while (manual_padding.size() < 2 * (rank - 2)) {
        manual_padding.push_back(0);
      }
    } else {
      while (manual_padding.size() < 2 * (rank - 2)) {
        manual_padding.push_back(0);
      }
    }
    // Get the strides.
    std::vector<size_t> strides;
    for (auto stride : layer->get_strides()) {
      strides.push_back(stride);
    }
    // Get the dilations.
    std::vector<size_t> dilations;
    for (auto dilation : layer->get_dilations()) {
      dilations.push_back(dilation);
    }
    // Compute input_sizes.
    std::vector<size_t> input_sizes;
    for (auto i = 2; i < rank; ++i) {
      input_sizes.push_back(I_shape[i]);
    }
    // Compute off_sizes.
    std::vector<size_t> off_sizes;
    for (auto i = 2; i < rank; ++i) {
      off_sizes.push_back(OFF_shape[i]);
    }
    // Compute filter_sizes.
    std::vector<size_t> filter_sizes;
    for (auto i = 2; i < rank; ++i) {
      filter_sizes.push_back(F_shape[i]);
    }
    // Compute pad_before.
    std::vector<size_t> pad_befores;
    for (auto i = 0; i < rank - 2; ++i) {
      auto pad_before = compute_padding_before(input_sizes[i], off_sizes[i], filter_sizes[i], strides[i], autopad_mode,
                                               manual_padding[i], manual_padding[i + rank - 2], dilations[i]);
      pad_befores.push_back(pad_before);
    }
    // Compute the shape of output.
    std::vector<size_t> output_sizes;
    for (auto i = 0; i < rank - 2; ++i) {
      auto output_size = compute_output_size(input_sizes[i], filter_sizes[i], strides[i], autopad_mode,
                                             manual_padding[i], manual_padding[i + rank - 2], dilations[i]);
      output_sizes.push_back(output_size);
    }
    // Compute the spatial size of filter;
    auto F_spatial_size = 1;
    for (auto i = 0; i < F_shape.size() - 2; ++i) {
      F_spatial_size *= F_shape[i + 2];
    }
    // Validate the shape of offset.
    for (auto i = 0; i < rank - 2; ++i) {
      if (output_sizes[i] != OFF_shape[i + 2]) {
        THROW_IE_EXCEPTION << "Incorrected shape for DeformableConvolution.";
      }
    }
    if (OFF_shape[1] != (OFF_shape.size() - 2) * DG * F_spatial_size) {
      THROW_IE_EXCEPTION << "Incorrected shape for DeformableConvolution.";
    }
    if (rank == 3) {
      // Define dim of each tensor.
      auto W = I_shape[2], OFF_W = OFF_shape[2], F_W = F_shape[2];
      // Throw exception
      if (CI % G != 0 || CO % G != 0 || CI % DG != 0 || OFF_shape[1] % DG != 0) {
        THROW_IE_EXCEPTION << "Incorrected shape for DeformableConvolution.";
      }
      // split offset along dg axis.
      std::vector<edsl::Tensor> OFF_split_dgs;
      for (auto i = 0; i < DG; ++i) {
        edsl::Tensor OFF_split_dg =
            op::slice(OFF).add_dim(0, N).add_dim(i * OFF_C / DG, (i + 1) * OFF_C / DG).add_dim(0, OFF_W);
        OFF_split_dgs.push_back(OFF_split_dg);
      }
      // Get offsets of height and width.
      std::vector<edsl::Tensor> offset_w_slices;
      for (auto i = 0; i < DG; ++i) {
        edsl::Tensor offset_w_concat_axis_2(0);
        for (auto off_w = 0; off_w < OFF_W; ++off_w) {
          edsl::Tensor single_offset_w =
              op::slice(OFF_split_dgs[i]).add_dim(0, N).add_dim(0, OFF_C / DG).add_dim(off_w, off_w + 1);
          single_offset_w = op::reshape(single_offset_w, make_tuple<int64_t>({N, 1, F_W}));
          if (offset_w_concat_axis_2.rank() != 3) {
            offset_w_concat_axis_2 = single_offset_w;
          } else {
            offset_w_concat_axis_2 = op::concatenate({offset_w_concat_axis_2, single_offset_w}, 2);
          }
        }
        offset_w_slices.push_back(offset_w_concat_axis_2);
      }
      // Define new width.
      auto NEW_W = F_W * OFF_W;
      // Broadcast offset.
      edsl::Tensor offset_w(0);
      for (auto dg = 0; dg < DG; ++dg) {
        edsl::Tensor offset_w_dg = offset_w_slices[dg];
        offset_w_dg = op::broadcast(
            offset_w_dg, {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_W)}, {0, 1, 2});
        if (offset_w.rank() != 4) {
          offset_w = offset_w_dg;
        } else {
          offset_w = op::concatenate({offset_w, offset_w_dg}, 1);
        }
      }
      // Define width index of input.
      TensorShape shape_w(precision, {NEW_W});
      Buffer buffer_w(shape_w);
      std::vector<float> data_w;
      for (auto off_w = 0; off_w < OFF_W; ++off_w) {
        for (auto f_w = 0; f_w < F_W; ++f_w) {
          auto data = off_w * strides[0] + f_w * dilations[0] - pad_befores[0];
          data_w.push_back(static_cast<float>(data));
        }
      }
      buffer_w.copy_from(data_w.data());
      auto input_index_w = Constant(buffer_w, "input_index_w");
      input_index_w = op::reshape(input_index_w, make_tuple<int64_t>({1, 1, NEW_W}));
      // Get deformabled index.
      edsl::Tensor new_index_w = offset_w + input_index_w;
      // Get deformable input tensor.
      edsl::Tensor deform_input = edsl::gather(I, new_index_w).axis(-1);
      deform_input = extract_tensor(deform_input, rank - 2);
      // Compute DeformableConvolution.
      autopad_mode = plaidml::op::AutoPadMode::VALID;
      auto result = op::convolution(deform_input, F)
                        .strides(std::vector<size_t>({(size_t)F_W}))
                        .autopad_mode(autopad_mode)
                        .input_layout(plaidml::op::TensorLayout::NCX)
                        .filter_layout(plaidml::op::TensorLayout::KCX)
                        .autogroup_mode(plaidml::op::AutoGroupMode::EXPLICIT)
                        .group_layout(plaidml::op::GroupLayout::IN_C)
                        .group_layout(plaidml::op::GroupLayout::IN_K)
                        .groups(static_cast<int>(G));
      return edsl::make_tuple(result);
    } else if (rank == 4) {
      // Define dim of each tensor.
      auto H = I_shape[2], W = I_shape[3], OFF_H = OFF_shape[2], OFF_W = OFF_shape[3], F_H = F_shape[2],
           F_W = F_shape[3];
      // Throw exception
      if (CI % G != 0 || CO % G != 0 || CI % DG != 0 || OFF_shape[1] % DG != 0) {
        THROW_IE_EXCEPTION << "Incorrected shape for DeformableConvolution.";
      }
      // split offset along dg axis.
      std::vector<edsl::Tensor> OFF_split_dgs;
      for (auto i = 0; i < DG; ++i) {
        edsl::Tensor OFF_split_dg = op::slice(OFF)
                                        .add_dim(0, N)
                                        .add_dim(i * OFF_C / DG, (i + 1) * OFF_C / DG)
                                        .add_dim(0, OFF_H)
                                        .add_dim(0, OFF_W);
        OFF_split_dgs.push_back(OFF_split_dg);
      }
      // Get offsets of height and width.
      std::vector<edsl::Tensor> offset_h_slices, offset_w_slices;
      for (auto i = 0; i < DG; ++i) {
        edsl::Tensor OFF_split_dg_h =
            op::slice(OFF_split_dgs[i]).add_dim(0, N).add_dim(0, OFF_C / DG - 1, 2).add_dim(0, OFF_H).add_dim(0, OFF_W);
        edsl::Tensor OFF_split_dg_w =
            op::slice(OFF_split_dgs[i]).add_dim(0, N).add_dim(1, OFF_C / DG, 2).add_dim(0, OFF_H).add_dim(0, OFF_W);
        edsl::Tensor offset_h_concat_axis_2(0), offset_w_concat_axis_2(0);
        for (auto off_h = 0; off_h < OFF_H; ++off_h) {
          edsl::Tensor offset_h_concat_axis_3(0), offset_w_concat_axis_3(0);
          for (auto off_w = 0; off_w < OFF_W; ++off_w) {
            edsl::Tensor single_offset_h = op::slice(OFF_split_dg_h)
                                               .add_dim(0, N)
                                               .add_dim(0, (OFF_C / DG) / 2)
                                               .add_dim(off_h, off_h + 1)
                                               .add_dim(off_w, off_w + 1);
            edsl::Tensor single_offset_w = op::slice(OFF_split_dg_w)
                                               .add_dim(0, N)
                                               .add_dim(0, (OFF_C / DG) / 2)
                                               .add_dim(off_h, off_h + 1)
                                               .add_dim(off_w, off_w + 1);
            single_offset_h = op::reshape(single_offset_h, make_tuple<int64_t>({N, 1, F_H, F_W}));
            single_offset_w = op::reshape(single_offset_w, make_tuple<int64_t>({N, 1, F_H, F_W}));
            if (offset_h_concat_axis_3.rank() != 4) {
              offset_h_concat_axis_3 = single_offset_h;
            } else {
              offset_h_concat_axis_3 = op::concatenate({offset_h_concat_axis_3, single_offset_h}, 3);
            }
            if (offset_w_concat_axis_3.rank() != 4) {
              offset_w_concat_axis_3 = single_offset_w;
            } else {
              offset_w_concat_axis_3 = op::concatenate({offset_w_concat_axis_3, single_offset_w}, 3);
            }
          }
          if (offset_h_concat_axis_2.rank() != 4) {
            offset_h_concat_axis_2 = offset_h_concat_axis_3;
          } else {
            offset_h_concat_axis_2 = op::concatenate({offset_h_concat_axis_2, offset_h_concat_axis_3}, 2);
          }
          if (offset_w_concat_axis_2.rank() != 4) {
            offset_w_concat_axis_2 = offset_w_concat_axis_3;
          } else {
            offset_w_concat_axis_2 = op::concatenate({offset_w_concat_axis_2, offset_w_concat_axis_3}, 2);
          }
        }
        offset_h_slices.push_back(offset_h_concat_axis_2);
        offset_w_slices.push_back(offset_w_concat_axis_2);
      }
      // Define new height and width.
      auto NEW_H = F_H * OFF_H, NEW_W = F_W * OFF_W;
      // Broadcast offset.
      edsl::Tensor offset_h(0), offset_w(0);
      for (auto dg = 0; dg < DG; ++dg) {
        edsl::Tensor offset_h_dg = offset_h_slices[dg];
        edsl::Tensor offset_w_dg = offset_w_slices[dg];
        offset_h_dg = op::broadcast(
            offset_h_dg,
            {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
            {0, 1, 2, 3});
        offset_w_dg = op::broadcast(
            offset_w_dg,
            {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
            {0, 1, 2, 3});
        if (offset_h.rank() != 4) {
          offset_h = offset_h_dg;
        } else {
          offset_h = op::concatenate({offset_h, offset_h_dg}, 1);
        }
        if (offset_w.rank() != 4) {
          offset_w = offset_w_dg;
        } else {
          offset_w = op::concatenate({offset_w, offset_w_dg}, 1);
        }
      }
      // Define height index of input.
      TensorShape shape_h(precision, {NEW_H, NEW_W});
      Buffer buffer_h(shape_h);
      std::vector<float> data_h;
      for (auto off_h = 0; off_h < OFF_H; ++off_h) {
        for (auto f_h = 0; f_h < F_H; ++f_h) {
          auto data = off_h * strides[0] + f_h * dilations[0] - pad_befores[0];
          for (auto new_w = 0; new_w < NEW_W; ++new_w) {
            data_h.push_back(static_cast<float>(data));
          }
        }
      }
      buffer_h.copy_from(data_h.data());
      auto input_index_h = Constant(buffer_h, "input_index_h");
      input_index_h = op::reshape(input_index_h, make_tuple<int64_t>({1, 1, NEW_H, NEW_W}));
      // Define width index of input.
      TensorShape shape_w(precision, {NEW_W, NEW_H});
      Buffer buffer_w(shape_w);
      std::vector<float> data_w;
      for (auto off_w = 0; off_w < OFF_W; ++off_w) {
        for (auto f_w = 0; f_w < F_W; ++f_w) {
          auto data = off_w * strides[1] + f_w * dilations[1] - pad_befores[1];
          for (auto new_h = 0; new_h < NEW_H; ++new_h) {
            data_w.push_back(static_cast<float>(data));
          }
        }
      }
      buffer_w.copy_from(data_w.data());
      auto input_index_w = Constant(buffer_w, "input_index_w");
      input_index_w = op::transpose(input_index_w);
      input_index_w = op::reshape(input_index_w, make_tuple<int64_t>({1, 1, NEW_H, NEW_W}));
      // Get deformabled index.
      edsl::Tensor new_index_h = offset_h + input_index_h;
      edsl::Tensor new_index_w = offset_w + input_index_w;
      // Get deformable input tensor.
      edsl::Tensor deform_input = edsl::gather(I, new_index_h).axis(-2);
      deform_input = edsl::gather(deform_input, new_index_w).axis(-1);
      deform_input = extract_tensor(deform_input, rank - 2);
      // Compute DeformableConvolution.
      autopad_mode = plaidml::op::AutoPadMode::VALID;
      auto result = op::convolution(deform_input, F)
                        .strides(std::vector<size_t>({(size_t)F_H, (size_t)F_W}))
                        .autopad_mode(autopad_mode)
                        .input_layout(plaidml::op::TensorLayout::NCX)
                        .filter_layout(plaidml::op::TensorLayout::KCX)
                        .autogroup_mode(plaidml::op::AutoGroupMode::EXPLICIT)
                        .group_layout(plaidml::op::GroupLayout::IN_C)
                        .group_layout(plaidml::op::GroupLayout::IN_K)
                        .groups(static_cast<int>(G));
      return edsl::make_tuple(result);
    } else if (rank == 5) {
      // Define dim of each tensor.
      auto D = I_shape[2], H = I_shape[3], W = I_shape[4], OFF_D = OFF_shape[2], OFF_H = OFF_shape[3],
           OFF_W = OFF_shape[4], F_D = F_shape[2], F_H = F_shape[3], F_W = F_shape[4];
      // Throw exception
      if (CI % G != 0 || CO % G != 0 || CI % DG != 0 || OFF_shape[1] % DG != 0) {
        THROW_IE_EXCEPTION << "Incorrected shape for DeformableConvolution.";
      }
      // split offset along dg axis.
      std::vector<edsl::Tensor> OFF_split_dgs;
      for (auto i = 0; i < DG; ++i) {
        edsl::Tensor OFF_split_dg = op::slice(OFF)
                                        .add_dim(0, N)
                                        .add_dim(i * OFF_C / DG, (i + 1) * OFF_C / DG)
                                        .add_dim(0, OFF_D)
                                        .add_dim(0, OFF_H)
                                        .add_dim(0, OFF_W);
        OFF_split_dgs.push_back(OFF_split_dg);
      }
      // Get offsets of height and width.
      std::vector<edsl::Tensor> offset_h_slices, offset_w_slices, offset_d_slices;
      for (auto i = 0; i < DG; ++i) {
        edsl::Tensor OFF_split_dg_d = op::slice(OFF_split_dgs[i])
                                          .add_dim(0, N)
                                          .add_dim(0, OFF_C / DG - 2, 3)
                                          .add_dim(0, OFF_D)
                                          .add_dim(0, OFF_H)
                                          .add_dim(0, OFF_W);
        edsl::Tensor OFF_split_dg_h = op::slice(OFF_split_dgs[i])
                                          .add_dim(0, N)
                                          .add_dim(1, OFF_C / DG - 1, 3)
                                          .add_dim(0, OFF_D)
                                          .add_dim(0, OFF_H)
                                          .add_dim(0, OFF_W);
        edsl::Tensor OFF_split_dg_w = op::slice(OFF_split_dgs[i])
                                          .add_dim(0, N)
                                          .add_dim(2, OFF_C / DG, 3)
                                          .add_dim(0, OFF_D)
                                          .add_dim(0, OFF_H)
                                          .add_dim(0, OFF_W);
        edsl::Tensor offset_d_concat_axis_2(0), offset_h_concat_axis_2(0), offset_w_concat_axis_2(0);
        for (auto off_d = 0; off_d < OFF_D; ++off_d) {
          edsl::Tensor offset_d_concat_axis_3(0), offset_h_concat_axis_3(0), offset_w_concat_axis_3(0);
          for (auto off_h = 0; off_h < OFF_H; ++off_h) {
            edsl::Tensor offset_d_concat_axis_4(0), offset_h_concat_axis_4(0), offset_w_concat_axis_4(0);
            for (auto off_w = 0; off_w < OFF_W; ++off_w) {
              edsl::Tensor single_offset_d = op::slice(OFF_split_dg_d)
                                                 .add_dim(0, N)
                                                 .add_dim(0, (OFF_C / DG) / 3)
                                                 .add_dim(off_d, off_d + 1)
                                                 .add_dim(off_h, off_h + 1)
                                                 .add_dim(off_w, off_w + 1);
              edsl::Tensor single_offset_h = op::slice(OFF_split_dg_h)
                                                 .add_dim(0, N)
                                                 .add_dim(0, (OFF_C / DG) / 3)
                                                 .add_dim(off_d, off_d + 1)
                                                 .add_dim(off_h, off_h + 1)
                                                 .add_dim(off_w, off_w + 1);
              edsl::Tensor single_offset_w = op::slice(OFF_split_dg_w)
                                                 .add_dim(0, N)
                                                 .add_dim(0, (OFF_C / DG) / 3)
                                                 .add_dim(off_d, off_d + 1)
                                                 .add_dim(off_h, off_h + 1)
                                                 .add_dim(off_w, off_w + 1);
              single_offset_d = op::reshape(single_offset_d, make_tuple<int64_t>({N, 1, F_D, F_H, F_W}));
              single_offset_h = op::reshape(single_offset_h, make_tuple<int64_t>({N, 1, F_D, F_H, F_W}));
              single_offset_w = op::reshape(single_offset_w, make_tuple<int64_t>({N, 1, F_D, F_H, F_W}));
              if (offset_d_concat_axis_4.rank() != 5) {
                offset_d_concat_axis_4 = single_offset_d;
              } else {
                offset_d_concat_axis_4 = op::concatenate({offset_d_concat_axis_4, single_offset_d}, 4);
              }
              if (offset_h_concat_axis_4.rank() != 5) {
                offset_h_concat_axis_4 = single_offset_h;
              } else {
                offset_h_concat_axis_4 = op::concatenate({offset_h_concat_axis_4, single_offset_h}, 4);
              }
              if (offset_w_concat_axis_4.rank() != 5) {
                offset_w_concat_axis_4 = single_offset_w;
              } else {
                offset_w_concat_axis_4 = op::concatenate({offset_w_concat_axis_4, single_offset_w}, 4);
              }
            }
            if (offset_d_concat_axis_3.rank() != 5) {
              offset_d_concat_axis_3 = offset_d_concat_axis_4;
            } else {
              offset_d_concat_axis_3 = op::concatenate({offset_d_concat_axis_3, offset_d_concat_axis_4}, 3);
            }
            if (offset_h_concat_axis_3.rank() != 5) {
              offset_h_concat_axis_3 = offset_h_concat_axis_4;
            } else {
              offset_h_concat_axis_3 = op::concatenate({offset_h_concat_axis_3, offset_h_concat_axis_4}, 3);
            }
            if (offset_w_concat_axis_3.rank() != 5) {
              offset_w_concat_axis_3 = offset_w_concat_axis_4;
            } else {
              offset_w_concat_axis_3 = op::concatenate({offset_w_concat_axis_3, offset_w_concat_axis_4}, 3);
            }
          }
          if (offset_d_concat_axis_2.rank() != 5) {
            offset_d_concat_axis_2 = offset_d_concat_axis_3;
          } else {
            offset_d_concat_axis_2 = op::concatenate({offset_d_concat_axis_2, offset_d_concat_axis_3}, 2);
          }
          if (offset_h_concat_axis_2.rank() != 5) {
            offset_h_concat_axis_2 = offset_h_concat_axis_3;
          } else {
            offset_h_concat_axis_2 = op::concatenate({offset_h_concat_axis_2, offset_h_concat_axis_3}, 2);
          }
          if (offset_w_concat_axis_2.rank() != 5) {
            offset_w_concat_axis_2 = offset_w_concat_axis_3;
          } else {
            offset_w_concat_axis_2 = op::concatenate({offset_w_concat_axis_2, offset_w_concat_axis_3}, 2);
          }
        }
        offset_d_slices.push_back(offset_d_concat_axis_2);
        offset_h_slices.push_back(offset_h_concat_axis_2);
        offset_w_slices.push_back(offset_w_concat_axis_2);
      }
      // Define new depth, height and width.
      auto NEW_D = F_D * OFF_D, NEW_H = F_H * OFF_H, NEW_W = F_W * OFF_W;
      // Broadcast offset.
      edsl::Tensor offset_d(0), offset_h(0), offset_w(0);
      for (auto dg = 0; dg < DG; ++dg) {
        edsl::Tensor offset_d_dg = offset_d_slices[dg];
        edsl::Tensor offset_h_dg = offset_h_slices[dg];
        edsl::Tensor offset_w_dg = offset_w_slices[dg];
        offset_h_dg = op::broadcast(offset_h_dg,
                                    {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_D),
                                     static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                                    {0, 1, 2, 3, 4});
        offset_w_dg = op::broadcast(offset_w_dg,
                                    {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_D),
                                     static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                                    {0, 1, 2, 3, 4});
        offset_d_dg = op::broadcast(offset_d_dg,
                                    {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_D),
                                     static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                                    {0, 1, 2, 3, 4});
        if (offset_d.rank() != 5) {
          offset_d = offset_d_dg;
        } else {
          offset_d = op::concatenate({offset_d, offset_d_dg}, 1);
        }
        if (offset_h.rank() != 5) {
          offset_h = offset_h_dg;
        } else {
          offset_h = op::concatenate({offset_h, offset_h_dg}, 1);
        }
        if (offset_w.rank() != 5) {
          offset_w = offset_w_dg;
        } else {
          offset_w = op::concatenate({offset_w, offset_w_dg}, 1);
        }
      }
      // Define depth index of input.
      TensorShape shape_d(precision, {NEW_D, NEW_H, NEW_W});
      Buffer buffer_d(shape_d);
      std::vector<float> data_d;
      for (auto off_d = 0; off_d < OFF_D; ++off_d) {
        for (auto f_d = 0; f_d < F_D; ++f_d) {
          auto data = off_d * strides[0] + f_d * dilations[0] - pad_befores[0];
          for (auto new_h = 0; new_h < NEW_H; ++new_h) {
            for (auto new_w = 0; new_w < NEW_W; ++new_w) {
              data_d.push_back(static_cast<float>(data));
            }
          }
        }
      }
      buffer_d.copy_from(data_d.data());
      auto input_index_d = Constant(buffer_d, "input_index_d");
      input_index_d = op::reshape(input_index_d, make_tuple<int64_t>({1, 1, NEW_D, NEW_H, NEW_W}));
      // Define height index of input.
      TensorShape shape_h(precision, {NEW_H, NEW_D, NEW_W});
      Buffer buffer_h(shape_h);
      std::vector<float> data_h;
      for (auto off_h = 0; off_h < OFF_H; ++off_h) {
        for (auto f_h = 0; f_h < F_H; ++f_h) {
          auto data = off_h * strides[1] + f_h * dilations[1] - pad_befores[1];
          for (auto new_d = 0; new_d < NEW_D; ++new_d) {
            for (auto new_w = 0; new_w < NEW_W; ++new_w) {
              data_h.push_back(static_cast<float>(data));
            }
          }
        }
      }
      buffer_h.copy_from(data_h.data());
      auto input_index_h = Constant(buffer_h, "input_index_h");
      input_index_h = op::transpose(input_index_h, make_tuple<size_t>({1, 0, 2}));
      input_index_h = op::reshape(input_index_h, make_tuple<int64_t>({1, 1, NEW_D, NEW_H, NEW_W}));
      // Define width index of input.
      TensorShape shape_w(precision, {NEW_W, NEW_D, NEW_H});
      Buffer buffer_w(shape_w);
      std::vector<float> data_w;
      for (auto off_w = 0; off_w < OFF_W; ++off_w) {
        for (auto f_w = 0; f_w < F_W; ++f_w) {
          auto data = off_w * strides[2] + f_w * dilations[2] - pad_befores[2];
          for (auto new_d = 0; new_d < NEW_D; ++new_d) {
            for (auto new_h = 0; new_h < NEW_H; ++new_h) {
              data_w.push_back(static_cast<float>(data));
            }
          }
        }
      }
      buffer_w.copy_from(data_w.data());
      auto input_index_w = Constant(buffer_w, "input_index_w");
      input_index_w = op::transpose(input_index_w, make_tuple<size_t>({1, 2, 0}));
      input_index_w = op::reshape(input_index_w, make_tuple<int64_t>({1, 1, NEW_D, NEW_H, NEW_W}));
      // Get deformabled index.
      edsl::Tensor new_index_d = offset_d + input_index_d;
      edsl::Tensor new_index_h = offset_h + input_index_h;
      edsl::Tensor new_index_w = offset_w + input_index_w;
      // Get deformable input tensor.
      edsl::Tensor deform_input = edsl::gather(I, new_index_d).axis(-3);
      deform_input = edsl::gather(deform_input, new_index_h).axis(-2);
      deform_input = edsl::gather(deform_input, new_index_w).axis(-1);
      deform_input = extract_tensor(deform_input, rank - 2);
      // Compute DeformableConvolution.
      autopad_mode = plaidml::op::AutoPadMode::VALID;
      auto result = op::convolution(deform_input, F)
                        .strides(std::vector<size_t>({(size_t)F_D, (size_t)F_H, (size_t)F_W}))
                        .autopad_mode(autopad_mode)
                        .input_layout(plaidml::op::TensorLayout::NCX)
                        .filter_layout(plaidml::op::TensorLayout::KCX)
                        .autogroup_mode(plaidml::op::AutoGroupMode::EXPLICIT)
                        .group_layout(plaidml::op::GroupLayout::IN_C)
                        .group_layout(plaidml::op::GroupLayout::IN_K)
                        .groups(static_cast<int>(G));
      return edsl::make_tuple(result);
    } else {
      THROW_IE_EXCEPTION << "Higher dimensions are not supported for now.";
    }
  });
}

}  // namespace PlaidMLPlugin
