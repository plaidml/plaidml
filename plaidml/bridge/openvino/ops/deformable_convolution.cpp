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

// Compute pad_before and the size of output
std::pair<size_t, size_t> compute_padding_before_and_output_size(size_t input_size, size_t off_size, size_t filter_size,
                                                                 size_t stride, plaidml::op::AutoPadMode autopad_mode,
                                                                 size_t pad_before, size_t pad_end, size_t dilation) {
  size_t I_eff = input_size;
  size_t F_eff = (dilation * (filter_size - 1)) + 1;
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    size_t output_size = ((I_eff + pad_before + pad_end - F_eff + stride) / stride);
    std::pair<size_t, size_t> result(pad_before, output_size);
    return result;
  }
  if (autopad_mode == plaidml::op::AutoPadMode::VALID) {
    size_t output_size = ((I_eff - F_eff + stride) / stride);
    std::pair<size_t, size_t> result(0, output_size);
    return result;
  }
  if (autopad_mode == plaidml::op::AutoPadMode::SAME_LOWER || autopad_mode == plaidml::op::AutoPadMode::SAME_UPPER) {
    size_t lower_term = (autopad_mode == plaidml::op::AutoPadMode::SAME_LOWER) ? 1 : 0;
    size_t max = 0;
    if (((off_size - 1) * stride + F_eff - input_size) > 0) {
      max = (off_size - 1) * stride + F_eff - input_size;
    }
    pad_before = (max + lower_term) / 2;
    size_t output_size = ((I_eff + stride - 1) / stride);
    std::pair<size_t, size_t> result(pad_before, output_size);
    return result;
  }
  THROW_IE_EXCEPTION << "Unexpected autopadding mode.";
}

// Extract values which is needed.
edsl::Tensor extract_tensor(edsl::Tensor I, size_t rank) {
  std::vector<TensorDim> I_dims(rank * (rank + 2) + 2), O_dims(rank + 2);
  std::vector<TensorIndex> I_idxs(rank * (rank + 2) + 2), O_idxs(rank + 2);
  I.bind_dims(I_dims);
  // Set up the dimensions of output.
  for (auto i = 2; i < rank + 4; ++i) {
    O_dims[i - 2] = I_dims[i];
  }
  // Set up the indexs of batch_size and channel.
  for (auto i = 0; i < 2; ++i) {
    I_idxs[i + 2] = I_idxs[i];
  }
  // Set up the indexs of input.
  for (auto i = 2; i < rank + 4; ++i) {
    for (auto j = 1; j < rank; ++j) {
      I_idxs[j * (rank + 2) + i] = I_idxs[i];
    }
  }
  // Set up the indexs of output.
  for (auto i = 2; i < rank + 4; ++i) {
    O_idxs[i - 2] = I_idxs[i];
  }
  edsl::Tensor O = Contraction(O_dims, O_idxs).assign(I(I_idxs));
  return O;
}

// Compute 1D DeformableConvolution.
edsl::Tensor compute_1d_deformable_convolution(edsl::Tensor I, edsl::Tensor OFF, edsl::Tensor F,
                                               std::vector<int64_t> I_shape, std::vector<int64_t> OFF_shape,
                                               std::vector<int64_t> F_shape, int64_t G, int64_t DG, size_t rank,
                                               std::vector<size_t> strides, std::vector<size_t> dilations,
                                               std::vector<size_t> pad_befores, plaidml::op::AutoPadMode autopad_mode) {
  // Define dim of each tensor.
  auto N = I_shape[0], CI = I_shape[1], CO = F_shape[1], OFF_C = OFF_shape[1], W = I_shape[2], OFF_W = OFF_shape[2],
       F_W = F_shape[2];
  // Throw exception
  if (CI % G != 0 || CO % G != 0 || CI % DG != 0 || OFF_C % DG != 0) {
    THROW_IE_EXCEPTION << "Incorrected shape for DeformableConvolution.";
  }
  // Define new width.
  auto NEW_W = OFF_W * F_W;
  // split offset along dg axis and get offset of width.
  std::vector<edsl::Tensor> offset_w_vec;
  for (auto dg = 0; dg < DG; ++dg) {
    edsl::Tensor OFF_split_dg =
        op::slice(OFF).add_dim(0, N).add_dim(dg * OFF_C / DG, (dg + 1) * OFF_C / DG).add_dim(0, OFF_W);
    edsl::Tensor OFF_split_dg_w = op::reshape(OFF_split_dg, make_tuple<int64_t>({N, 1, F_W, OFF_W}));
    OFF_split_dg_w = op::transpose(OFF_split_dg_w, make_tuple<int64_t>({0, 1, 3, 2}));
    OFF_split_dg_w = op::reshape(OFF_split_dg_w, make_tuple<int64_t>({N, 1, OFF_W * F_W}));
    OFF_split_dg_w = op::broadcast(
        OFF_split_dg_w, {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_W)}, {0, 1, 2});
    offset_w_vec.push_back(OFF_split_dg_w);
  }
  edsl::Tensor offset_w = op::concatenate(offset_w_vec, 1);
  // Define width index of input.
  std::vector<edsl::TensorDim> F_dims(rank);
  F.bind_dims(F_dims);
  edsl::Tensor index_w = edsl::index({F_dims[2]}, static_cast<size_t>(0));
  index_w = index_w * dilations[0];
  std::vector<edsl::Tensor> index_w_vec;
  index_w_vec.push_back(index_w);
  for (auto off_w = 0; off_w < OFF_W - 1; ++off_w) {
    index_w = index_w + strides[0];
    index_w_vec.push_back(index_w);
  }
  index_w = op::concatenate(index_w_vec, 0);
  index_w = index_w - pad_befores[0];
  index_w = op::reshape(index_w, make_tuple<int64_t>({1, 1, NEW_W}));
  // Get deformabled index.
  edsl::Tensor new_index_w = offset_w + index_w;
  // Get deformable input tensor.
  edsl::Tensor deform_input = edsl::gather(I, new_index_w).axis(-1);
  deform_input = extract_tensor(deform_input, rank - 2);
  // Compute DeformableConvolution.
  autopad_mode = plaidml::op::AutoPadMode::VALID;
  edsl::Tensor result = op::convolution(deform_input, F)
                            .strides(std::vector<size_t>({(size_t)F_W}))
                            .autopad_mode(autopad_mode)
                            .input_layout(plaidml::op::TensorLayout::NCX)
                            .filter_layout(plaidml::op::TensorLayout::KCX)
                            .autogroup_mode(plaidml::op::AutoGroupMode::EXPLICIT)
                            .group_layout(plaidml::op::GroupLayout::IN_C)
                            .group_layout(plaidml::op::GroupLayout::IN_K)
                            .groups(static_cast<int>(G));
  return result;
}

// Compute 2D DeformableConvolution.
edsl::Tensor compute_2d_deformable_convolution(edsl::Tensor I, edsl::Tensor OFF, edsl::Tensor F,
                                               std::vector<int64_t> I_shape, std::vector<int64_t> OFF_shape,
                                               std::vector<int64_t> F_shape, int64_t G, int64_t DG, size_t rank,
                                               std::vector<size_t> strides, std::vector<size_t> dilations,
                                               std::vector<size_t> pad_befores, plaidml::op::AutoPadMode autopad_mode) {
  // Define dim of each tensor.
  auto N = I_shape[0], CI = I_shape[1], CO = F_shape[1], OFF_C = OFF_shape[1], H = I_shape[2], W = I_shape[3],
       OFF_H = OFF_shape[2], OFF_W = OFF_shape[3], F_H = F_shape[2], F_W = F_shape[3];
  // Throw exception
  if (CI % G != 0 || CO % G != 0 || CI % DG != 0 || OFF_C % DG != 0) {
    THROW_IE_EXCEPTION << "Incorrected shape for DeformableConvolution.";
  }
  // Define new hight and width.
  auto NEW_H = OFF_H * F_H, NEW_W = OFF_W * F_W;
  // split offset along dg axis and get offset of hight and width.
  std::vector<edsl::Tensor> offset_h_vec, offset_w_vec;
  for (auto dg = 0; dg < DG; ++dg) {
    edsl::Tensor OFF_split_dg = op::slice(OFF)
                                    .add_dim(0, N)
                                    .add_dim(dg * OFF_C / DG, (dg + 1) * OFF_C / DG)
                                    .add_dim(0, OFF_H)
                                    .add_dim(0, OFF_W);
    edsl::Tensor OFF_split_dg_h =
        op::slice(OFF_split_dg).add_dim(0, N).add_dim(0, OFF_C / DG - 1, 2).add_dim(0, OFF_H).add_dim(0, OFF_W);
    edsl::Tensor OFF_split_dg_w =
        op::slice(OFF_split_dg).add_dim(0, N).add_dim(1, OFF_C / DG, 2).add_dim(0, OFF_H).add_dim(0, OFF_W);
    OFF_split_dg_h = op::reshape(OFF_split_dg_h, make_tuple<int64_t>({N, 1, F_H, F_W, OFF_H, OFF_W}));
    OFF_split_dg_h = op::transpose(OFF_split_dg_h, make_tuple<int64_t>({0, 1, 4, 2, 5, 3}));
    OFF_split_dg_h = op::reshape(OFF_split_dg_h, make_tuple<int64_t>({N, 1, OFF_H * F_H, OFF_W * F_W}));
    OFF_split_dg_h = op::broadcast(
        OFF_split_dg_h,
        {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
        {0, 1, 2, 3});
    OFF_split_dg_w = op::reshape(OFF_split_dg_w, make_tuple<int64_t>({N, 1, F_H, F_W, OFF_H, OFF_W}));
    OFF_split_dg_w = op::transpose(OFF_split_dg_w, make_tuple<int64_t>({0, 1, 4, 2, 5, 3}));
    OFF_split_dg_w = op::reshape(OFF_split_dg_w, make_tuple<int64_t>({N, 1, OFF_H * F_H, OFF_W * F_W}));
    OFF_split_dg_w = op::broadcast(
        OFF_split_dg_w,
        {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
        {0, 1, 2, 3});
    offset_h_vec.push_back(OFF_split_dg_h);
    offset_w_vec.push_back(OFF_split_dg_w);
  }
  edsl::Tensor offset_h = op::concatenate(offset_h_vec, 1);
  edsl::Tensor offset_w = op::concatenate(offset_w_vec, 1);
  // Define height index and width index of input.
  std::vector<edsl::TensorDim> F_dims(rank);
  F.bind_dims(F_dims);
  edsl::Tensor index_h = edsl::index({F_dims[2]}, static_cast<size_t>(0));
  edsl::Tensor index_w = edsl::index({F_dims[3]}, static_cast<size_t>(0));
  index_h = index_h * dilations[0];
  index_w = index_w * dilations[1];
  std::vector<edsl::Tensor> index_h_vec, index_w_vec;
  index_h_vec.push_back(index_h);
  index_w_vec.push_back(index_w);
  for (auto off_h = 0; off_h < OFF_H - 1; ++off_h) {
    index_h = index_h + strides[0];
    index_h_vec.push_back(index_h);
  }
  for (auto off_w = 0; off_w < OFF_W - 1; ++off_w) {
    index_w = index_w + strides[1];
    index_w_vec.push_back(index_w);
  }
  index_h = op::concatenate(index_h_vec, 0);
  index_w = op::concatenate(index_w_vec, 0);
  index_h = index_h - pad_befores[0];
  index_w = index_w - pad_befores[1];
  index_h = op::reshape(index_h, make_tuple<int64_t>({1, 1, NEW_H, 1}));
  index_w = op::reshape(index_w, make_tuple<int64_t>({1, 1, 1, NEW_W}));
  index_h = op::broadcast(index_h, {1, 1, static_cast<int>(NEW_H), static_cast<int>(NEW_W)}, {0, 1, 2, 3});
  index_w = op::broadcast(index_w, {1, 1, static_cast<int>(NEW_H), static_cast<int>(NEW_W)}, {0, 1, 2, 3});
  // Get deformabled index.
  edsl::Tensor new_index_h = offset_h + index_h;
  edsl::Tensor new_index_w = offset_w + index_w;
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
  return result;
}

// Compute 3D DeformableConvolution.
edsl::Tensor compute_3d_deformable_convolution(edsl::Tensor I, edsl::Tensor OFF, edsl::Tensor F,
                                               std::vector<int64_t> I_shape, std::vector<int64_t> OFF_shape,
                                               std::vector<int64_t> F_shape, int64_t G, int64_t DG, size_t rank,
                                               std::vector<size_t> strides, std::vector<size_t> dilations,
                                               std::vector<size_t> pad_befores, plaidml::op::AutoPadMode autopad_mode) {
  // Define dim of each tensor.
  auto N = I_shape[0], CI = I_shape[1], CO = F_shape[1], OFF_C = OFF_shape[1], D = I_shape[2], H = I_shape[3],
       W = I_shape[4], OFF_D = OFF_shape[2], OFF_H = OFF_shape[3], OFF_W = OFF_shape[4], F_D = F_shape[2],
       F_H = F_shape[3], F_W = F_shape[4];
  // Throw exception
  if (CI % G != 0 || CO % G != 0 || CI % DG != 0 || OFF_C % DG != 0) {
    THROW_IE_EXCEPTION << "Incorrected shape for DeformableConvolution.";
  }
  // Define new depth, height and width.
  auto NEW_D = F_D * OFF_D, NEW_H = F_H * OFF_H, NEW_W = F_W * OFF_W;
  // split offset along dg axis and get offset of depth, height and width.
  std::vector<edsl::Tensor> offset_d_vec, offset_h_vec, offset_w_vec;
  for (auto dg = 0; dg < DG; ++dg) {
    edsl::Tensor OFF_split_dg = op::slice(OFF)
                                    .add_dim(0, N)
                                    .add_dim(dg * OFF_C / DG, (dg + 1) * OFF_C / DG)
                                    .add_dim(0, OFF_D)
                                    .add_dim(0, OFF_H)
                                    .add_dim(0, OFF_W);
    edsl::Tensor OFF_split_dg_d = op::slice(OFF_split_dg)
                                      .add_dim(0, N)
                                      .add_dim(0, OFF_C / DG - 2, 3)
                                      .add_dim(0, OFF_D)
                                      .add_dim(0, OFF_H)
                                      .add_dim(0, OFF_W);
    edsl::Tensor OFF_split_dg_h = op::slice(OFF_split_dg)
                                      .add_dim(0, N)
                                      .add_dim(1, OFF_C / DG - 1, 3)
                                      .add_dim(0, OFF_D)
                                      .add_dim(0, OFF_H)
                                      .add_dim(0, OFF_W);
    edsl::Tensor OFF_split_dg_w = op::slice(OFF_split_dg)
                                      .add_dim(0, N)
                                      .add_dim(2, OFF_C / DG, 3)
                                      .add_dim(0, OFF_D)
                                      .add_dim(0, OFF_H)
                                      .add_dim(0, OFF_W);
    OFF_split_dg_d = op::reshape(OFF_split_dg_d, make_tuple<int64_t>({N, 1, F_D, F_H, F_W, OFF_D, OFF_H, OFF_W}));
    OFF_split_dg_d = op::transpose(OFF_split_dg_d, make_tuple<int64_t>({0, 1, 5, 2, 6, 3, 7, 4}));
    OFF_split_dg_d = op::reshape(OFF_split_dg_d, make_tuple<int64_t>({N, 1, OFF_D * F_D, OFF_H * F_H, OFF_W * F_W}));
    OFF_split_dg_d = op::broadcast(OFF_split_dg_d,
                                   {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_D),
                                    static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                                   {0, 1, 2, 3, 4});
    OFF_split_dg_h = op::reshape(OFF_split_dg_h, make_tuple<int64_t>({N, 1, F_D, F_H, F_W, OFF_D, OFF_H, OFF_W}));
    OFF_split_dg_h = op::transpose(OFF_split_dg_h, make_tuple<int64_t>({0, 1, 5, 2, 6, 3, 7, 4}));
    OFF_split_dg_h = op::reshape(OFF_split_dg_h, make_tuple<int64_t>({N, 1, OFF_D * F_D, OFF_H * F_H, OFF_W * F_W}));
    OFF_split_dg_h = op::broadcast(OFF_split_dg_h,
                                   {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_D),
                                    static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                                   {0, 1, 2, 3, 4});
    OFF_split_dg_w = op::reshape(OFF_split_dg_w, make_tuple<int64_t>({N, 1, F_D, F_H, F_W, OFF_D, OFF_H, OFF_W}));
    OFF_split_dg_w = op::transpose(OFF_split_dg_w, make_tuple<int64_t>({0, 1, 5, 2, 6, 3, 7, 4}));
    OFF_split_dg_w = op::reshape(OFF_split_dg_w, make_tuple<int64_t>({N, 1, OFF_D * F_D, OFF_H * F_H, OFF_W * F_W}));
    OFF_split_dg_w = op::broadcast(OFF_split_dg_w,
                                   {static_cast<int>(N), static_cast<int>(CI / DG), static_cast<int>(NEW_D),
                                    static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                                   {0, 1, 2, 3, 4});
    offset_d_vec.push_back(OFF_split_dg_d);
    offset_h_vec.push_back(OFF_split_dg_h);
    offset_w_vec.push_back(OFF_split_dg_w);
  }
  edsl::Tensor offset_d = op::concatenate(offset_d_vec, 1);
  edsl::Tensor offset_h = op::concatenate(offset_h_vec, 1);
  edsl::Tensor offset_w = op::concatenate(offset_w_vec, 1);
  // Define depth, hight and width of input.
  std::vector<edsl::TensorDim> F_dims(rank);
  F.bind_dims(F_dims);
  edsl::Tensor index_d = edsl::index({F_dims[2]}, static_cast<size_t>(0));
  edsl::Tensor index_h = edsl::index({F_dims[3]}, static_cast<size_t>(0));
  edsl::Tensor index_w = edsl::index({F_dims[4]}, static_cast<size_t>(0));
  index_d = index_d * dilations[0];
  index_h = index_h * dilations[1];
  index_w = index_w * dilations[2];
  std::vector<edsl::Tensor> index_d_vec, index_h_vec, index_w_vec;
  index_d_vec.push_back(index_d);
  index_h_vec.push_back(index_h);
  index_w_vec.push_back(index_w);
  for (auto off_d = 0; off_d < OFF_D - 1; ++off_d) {
    index_d = index_d + strides[0];
    index_d_vec.push_back(index_d);
  }
  for (auto off_h = 0; off_h < OFF_H - 1; ++off_h) {
    index_h = index_h + strides[1];
    index_h_vec.push_back(index_h);
  }
  for (auto off_w = 0; off_w < OFF_W - 1; ++off_w) {
    index_w = index_w + strides[2];
    index_w_vec.push_back(index_w);
  }
  index_d = op::concatenate(index_d_vec, 0);
  index_h = op::concatenate(index_h_vec, 0);
  index_w = op::concatenate(index_w_vec, 0);
  index_d = index_d - pad_befores[0];
  index_h = index_h - pad_befores[1];
  index_w = index_w - pad_befores[2];
  index_d = op::reshape(index_d, make_tuple<int64_t>({1, 1, NEW_D, 1, 1}));
  index_h = op::reshape(index_h, make_tuple<int64_t>({1, 1, 1, NEW_H, 1}));
  index_w = op::reshape(index_w, make_tuple<int64_t>({1, 1, 1, 1, NEW_W}));
  index_d = op::broadcast(index_d, {1, 1, static_cast<int>(NEW_D), static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                          {0, 1, 2, 3, 4});
  index_h = op::broadcast(index_h, {1, 1, static_cast<int>(NEW_D), static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                          {0, 1, 2, 3, 4});
  index_w = op::broadcast(index_w, {1, 1, static_cast<int>(NEW_D), static_cast<int>(NEW_H), static_cast<int>(NEW_W)},
                          {0, 1, 2, 3, 4});
  // Get deformabled index.
  edsl::Tensor new_index_d = offset_d + index_d;
  edsl::Tensor new_index_h = offset_h + index_h;
  edsl::Tensor new_index_w = offset_w + index_w;
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
  return result;
}

void registerDeformableConvolution() {
  registerOp("DeformableConvolution", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::DeformableConvolution>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0), OFF = ctx.operands.at(1), F = ctx.operands.at(2);
    auto I_shape = I.compute_shape().sizes(), OFF_shape = OFF.compute_shape().sizes(),
         F_shape = F.compute_shape().sizes();
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
    auto strides = layer->get_strides();
    // Get the dilations.
    auto dilations = layer->get_dilations();
    // Compute pad_before and the shape of output.
    std::vector<size_t> pad_befores, output_sizes;
    for (auto i = 0; i < rank - 2; ++i) {
      std::pair<size_t, size_t> pad_before_and_output = compute_padding_before_and_output_size(
          I_shape[i + 2], OFF_shape[i + 2], F_shape[i + 2], strides[i], autopad_mode, manual_padding[i],
          manual_padding[i + rank - 2], dilations[i]);
      pad_befores.push_back(pad_before_and_output.first);
      output_sizes.push_back(pad_before_and_output.second);
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
    // Compute DeformableConvolution.
    edsl::Tensor O;
    switch (rank) {
      case 3:
        O = compute_1d_deformable_convolution(I, OFF, F, I_shape, OFF_shape, F_shape, G, DG, rank, strides, dilations,
                                              pad_befores, autopad_mode);
        return edsl::make_tuple(O);
        break;
      case 4:
        O = compute_2d_deformable_convolution(I, OFF, F, I_shape, OFF_shape, F_shape, G, DG, rank, strides, dilations,
                                              pad_befores, autopad_mode);
        return edsl::make_tuple(O);
        break;
      case 5:
        O = compute_3d_deformable_convolution(I, OFF, F, I_shape, OFF_shape, F_shape, G, DG, rank, strides, dilations,
                                              pad_befores, autopad_mode);
        return edsl::make_tuple(O);
        break;
      default:
        THROW_IE_EXCEPTION << "Higher dimension are not supported for now.";
    }
  });
}

}  // namespace PlaidMLPlugin
