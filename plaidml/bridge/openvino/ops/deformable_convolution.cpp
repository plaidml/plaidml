// Copyright (C) 2021 Intel Corporation
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

namespace {

// Compute pad_before and the size of output.
// This function is based on (the functon compute_padding_and_output_size() in plaidml/op/lib/ops.cc.).
// The type of the return variable and parameters are changed from TensorDim to size_t, because TensorDim can't be
// converted to int.
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

// Cast long int to int in vector.
std::vector<int> cast_vector(std::vector<int64_t> vec) {
  std::vector<int> cast_vec(vec.size());
  for (auto i = 0; i < vec.size(); ++i) {
    cast_vec[i] = static_cast<int>(vec[i]);
  }
  return cast_vec;
}

// Compute DeformableConvolution.
edsl::Tensor compute_deformable_convolution(edsl::Tensor I, edsl::Tensor OFF, edsl::Tensor F, std::vector<int> I_shape,
                                            std::vector<int> OFF_shape, std::vector<int> F_shape, int G, int DG,
                                            size_t rank, std::vector<size_t> strides, std::vector<size_t> dilations,
                                            std::vector<size_t> pad_befores) {
  // Define dim of each tensor.
  int N = I_shape[0];
  int CI = I_shape[1];
  int CO = F_shape[1];
  int OFF_C = OFF_shape[1];
  // Throw exception
  if (CI % G != 0 || CO % G != 0 || CI % DG != 0 || OFF_C % DG != 0) {
    THROW_IE_EXCEPTION << "Incorrect shape for DeformableConvolution.";
  }
  // Define new dimention.
  std::vector<int> NEW_DIM;
  for (auto i = 2; i < rank; ++i) {
    NEW_DIM.push_back(F_shape[i] * OFF_shape[i]);
  }
  // Set up the value of the reshape to split the dimension.
  // For example in 2D, after this operation, the shape of offset will be {N, DG, F_H, F_W, 2, OFF_H, OFF_W}.
  std::vector<int> OFF_reshape_dims;
  OFF_reshape_dims.push_back(N);
  OFF_reshape_dims.push_back(DG);
  OFF_reshape_dims.insert(OFF_reshape_dims.end(), F_shape.begin() + 2, F_shape.end());
  OFF_reshape_dims.push_back(rank - 2);
  OFF_reshape_dims.insert(OFF_reshape_dims.end(), OFF_shape.begin() + 2, OFF_shape.end());
  edsl::Tensor offset = op::reshape(OFF, make_tuple<int>(OFF_reshape_dims));
  // Set up the value of transpose.
  // For example in 2D, after this operation, the shape of offset will be {N, DG, OFF_H, F_H, OFF_W, F_W, 2}.
  std::vector<int> OFF_transpose_dims;
  OFF_transpose_dims.push_back(0);
  OFF_transpose_dims.push_back(1);
  for (auto i = 0; i < rank - 2; ++i) {
    OFF_transpose_dims.push_back(rank + 1 + i);
    OFF_transpose_dims.push_back(2 + i);
  }
  OFF_transpose_dims.push_back(rank);
  offset = op::transpose(offset, make_tuple<int>(OFF_transpose_dims));
  // Set up the value of the reshape to multiply the dimension.
  // For example in 2D, after this operation, the shape of offset will be {N, DG, 1, OFF_H*F_H, OFF_W*F_W, 2}.
  std::vector<int> OFF_multiply_dims;
  OFF_multiply_dims.push_back(N);
  OFF_multiply_dims.push_back(DG);
  OFF_multiply_dims.push_back(1);
  OFF_multiply_dims.insert(OFF_multiply_dims.end(), NEW_DIM.begin(), NEW_DIM.end());
  OFF_multiply_dims.push_back(rank - 2);
  offset = op::reshape(offset, make_tuple<int>(OFF_multiply_dims));
  // Set up the value of the broadcast to broadcast the channel dimension.
  // For example in 2D, after this operation, the shape of offset will be {N, DG, CI/DG, OFF_H*F_H, OFF_W*F_W, 2}.
  std::vector<int> OFF_broadcast_dims, OFF_broadcast_axes;
  OFF_broadcast_dims.push_back(N);
  OFF_broadcast_dims.push_back(DG);
  OFF_broadcast_dims.push_back(CI / DG);
  OFF_broadcast_dims.insert(OFF_broadcast_dims.end(), NEW_DIM.begin(), NEW_DIM.end());
  OFF_broadcast_dims.push_back(rank - 2);
  for (auto i = 0; i < rank + 2; ++i) {
    OFF_broadcast_axes.push_back(i);
  }
  offset = op::broadcast(offset, OFF_broadcast_dims, OFF_broadcast_axes);
  // Set up the value of the reshape to multiply the channel dimension.
  // For example in 2D, after this operation, the shape of offset will be {N, CI, OFF_H*F_H, OFF_W*F_W, 2}.
  std::vector<int> OFF_reshape_channel;
  OFF_reshape_channel.push_back(N);
  OFF_reshape_channel.push_back(CI);
  OFF_reshape_channel.insert(OFF_reshape_channel.end(), NEW_DIM.begin(), NEW_DIM.end());
  OFF_reshape_channel.push_back(rank - 2);
  offset = op::reshape(offset, make_tuple<int>(OFF_reshape_channel));
  // Define the index of input.
  std::vector<edsl::Tensor> index_vec(rank - 2);
  for (auto i = 0; i < rank - 2; ++i) {
    edsl::Tensor index_vec_0 =
        edsl::index({edsl::TensorDim(OFF_shape[i + 2]), edsl::TensorDim(F_shape[i + 2])}, static_cast<size_t>(0));
    edsl::Tensor index_vec_1 =
        edsl::index({edsl::TensorDim(OFF_shape[i + 2]), edsl::TensorDim(F_shape[i + 2])}, static_cast<size_t>(1));
    index_vec[i] = index_vec_0 * strides[i] + index_vec_1 * dilations[i] - pad_befores[i];
    // Set up the value of reshape to reshape index.
    std::vector<int> index_reshape_dims(rank + 1, 1);
    index_reshape_dims[i + 2] = NEW_DIM[i];
    index_vec[i] = op::reshape(index_vec[i], make_tuple<int>(index_reshape_dims));
    // Set up the value of broadcast to broadcast the index.
    // For example in 2D, after this operation, the shape of index_vec[i] will be {1, 1, OFF_H*F_H, OFF_W*F_W, 1}.
    std::vector<int> index_broadcast_dims(rank + 1, 1);
    for (auto j = 2; j < rank; ++j) {
      index_broadcast_dims[j] = NEW_DIM[j - 2];
    }
    std::vector<int> index_broadcast_axes;
    for (auto j = 0; j < rank + 1; ++j) {
      index_broadcast_axes.push_back(j);
    }
    index_vec[i] = op::broadcast(index_vec[i], index_broadcast_dims, index_broadcast_axes);
  }
  edsl::Tensor index = op::concatenate(index_vec, -1);  // The shape of index is {1, 1, NEW_DIM, rank-2}
  // Get deformed index.
  edsl::Tensor new_index = offset + index;
  // Get deformed input tensor with gatherND and interplate method.
  std::vector<edsl::Tensor> new_index_vec(rank - 2);
  std::vector<edsl::Tensor> new_index_vec_ceil(rank - 2);
  std::vector<edsl::Tensor> new_index_vec_floor(rank - 2);
  // Set up the value of reshape to reshape the new_index_ceil and new_index_floor tensor.
  std::vector<int> new_index_reshape_dims;
  new_index_reshape_dims.push_back(N);
  new_index_reshape_dims.push_back(CI);
  new_index_reshape_dims.insert(new_index_reshape_dims.end(), NEW_DIM.begin(), NEW_DIM.end());
  new_index_reshape_dims.push_back(1);
  auto index_slice = op::slice(new_index).add_dim(0, N).add_dim(0, CI);
  for (auto i = 0; i < rank - 2; ++i) {
    index_slice = op::slice(index_slice).add_dim(0, NEW_DIM[i]);
  }
  for (auto i = 0; i < rank - 2; ++i) {
    new_index_vec[i] = op::slice(index_slice).add_dim(i);
    new_index_vec_ceil[i] = edsl::ceil(new_index_vec[i]);
    new_index_vec_floor[i] = edsl::floor(new_index_vec[i]);
    new_index_vec_ceil[i] = op::reshape(new_index_vec_ceil[i], make_tuple<int>(new_index_reshape_dims));
    new_index_vec_floor[i] = op::reshape(new_index_vec_floor[i], make_tuple<int>(new_index_reshape_dims));
  }
  // Coord_num is the number of tensor required for multi-linear interpolation.
  int coord_num = static_cast<int>(std::pow(2.0, rank - 2));
  std::vector<edsl::Tensor> new_index_coord_vec(coord_num);
  std::vector<edsl::Tensor> deform_input_vec(coord_num);
  for (auto i = 0; i < coord_num; ++i) {
    std::vector<edsl::Tensor> concat_vec(rank - 2);
    for (auto j = 0; j < rank - 2; ++j) {
      int flag = (i / static_cast<int>(std::pow(2.0, rank - 2 - j - 1))) % 2;
      if (flag == 0) {
        concat_vec[j] = new_index_vec_floor[j];
      } else {
        concat_vec[j] = new_index_vec_ceil[j];
      }
    }
    new_index_coord_vec[i] = op::concatenate(concat_vec, rank);
    new_index_coord_vec[i] = edsl::cast(new_index_coord_vec[i], DType::INT32);
    deform_input_vec[i] = edsl::gather(I, new_index_coord_vec[i]).mode(edsl::GatherMode::ND).batchDims(2);
  }
  // Change the shape of new_index_ceil tensor and new_index_floor tensor to original shape.
  new_index_reshape_dims.erase(new_index_reshape_dims.end() - 1);
  for (auto i = 0; i < rank - 2; ++i) {
    new_index_vec_ceil[i] = op::reshape(new_index_vec_ceil[i], make_tuple<int>(new_index_reshape_dims));
    new_index_vec_floor[i] = op::reshape(new_index_vec_floor[i], make_tuple<int>(new_index_reshape_dims));
  }
  // Get deformed input tensor with multi-linear interpolation.
  auto ONE = cast(Tensor{1.0}, DType::FLOAT32);
  for (auto i = 0; i < rank - 2; ++i) {
    for (auto j = 0; j < static_cast<int>(std::pow(2.0, rank - 2 - i - 1)); ++j) {
      deform_input_vec[j] = deform_input_vec[j] * (new_index_vec_ceil[i] - new_index_vec[i]) +
                            deform_input_vec[j + static_cast<int>(std::pow(2.0, rank - 2 - i - 1))] *
                                (ONE - new_index_vec_ceil[i] + new_index_vec[i]);
    }
  }
  edsl::Tensor deform_input = deform_input_vec[0];
  // Set up the new strides.
  std::vector<int> new_strides(F_shape.begin() + 2, F_shape.end());
  // Compute DeformableConvolution.
  edsl::Tensor result = op::convolution(deform_input, F)
                            .strides(new_strides)
                            .autopad_mode(plaidml::op::AutoPadMode::VALID)
                            .input_layout(plaidml::op::TensorLayout::NCX)
                            .filter_layout(plaidml::op::TensorLayout::KCX)
                            .autogroup_mode(plaidml::op::AutoGroupMode::EXPLICIT)
                            .group_layout(plaidml::op::GroupLayout::IN_C)
                            .group_layout(plaidml::op::GroupLayout::IN_K)
                            .groups(G);
  return result;
}

}  // namespace

namespace PlaidMLPlugin {

void registerDeformableConvolution() {
  registerOp("DeformableConvolution", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::DeformableConvolution>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0);
    auto OFF = ctx.operands.at(1);
    auto F = ctx.operands.at(2);
    auto I_shape = I.compute_shape().sizes();
    auto OFF_shape = OFF.compute_shape().sizes();
    auto F_shape = F.compute_shape().sizes();
    std::vector<int> I_shape_cast = cast_vector(I_shape);
    std::vector<int> OFF_shape_cast = cast_vector(OFF_shape);
    std::vector<int> F_shape_cast = cast_vector(F_shape);
    int G = layer->get_group();
    int DG = layer->get_deformable_group();
    auto rank = I.rank();
    // Get autopad_mode.
    auto autopad_mode = to_plaidml(layer->get_auto_pad());
    // Compute manual_padding.
    std::vector<size_t> manual_padding;
    if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
      auto pads_begin = layer->get_pads_begin();
      auto pads_end = layer->get_pads_end();
      manual_padding.insert(manual_padding.end(), pads_begin.begin(), pads_begin.end());
      manual_padding.insert(manual_padding.end(), pads_end.begin(), pads_end.end());
    }
    while (manual_padding.size() < 2 * (rank - 2)) {
      manual_padding.push_back(0);
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
        THROW_IE_EXCEPTION << "Incorrect shape for DeformableConvolution.";
      }
    }
    if (OFF_shape[1] != (rank - 2) * DG * F_spatial_size) {
      THROW_IE_EXCEPTION << "Incorrect shape for DeformableConvolution.";
    }
    // Compute DeformableConvolution.
    edsl::Tensor O = compute_deformable_convolution(I, OFF, F, I_shape_cast, OFF_shape_cast, F_shape_cast, G, DG, rank,
                                                    strides, dilations, pad_befores);
    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
