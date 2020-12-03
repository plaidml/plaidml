// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace {

template <typename T>
edsl::Tensor create_kernel_tensor(const int64_t& filter_row, const int64_t& filter_col,
                                  const edsl::Tensor& input_tensor) {
  auto input_type = input_tensor.dtype();
  auto input_shape = input_tensor.compute_shape().sizes();

  // Ops need Tensor order have to be "NCHW".
  auto input_channel = input_shape[1];

  // build kernel Tensor dims.
  auto depths = filter_row * filter_col * input_channel;
  std::vector<int64_t> kernel_dims{
      /*output channels*/ depths,
      /*input channels*/ input_channel,
      /*filter width*/ filter_row,
      /*filter height*/ filter_col,
  };

  // kernel tensor element size.
  size_t kernel_sum = 1;
  for (auto dim : kernel_dims) {
    kernel_sum *= dim;
  }

  // build kernel Tensor buffer.
  std::vector<T> data(kernel_sum, 0);
  int64_t channel_index = 0;
  for (int64_t depth = 0; depth < depths; depth++) {
    auto index = depth * kernel_dims[1] * kernel_dims[2] * kernel_dims[3] +
                 channel_index * kernel_dims[2] * kernel_dims[3] +
                 (depth / input_channel) / filter_col * kernel_dims[3] + (depth / input_channel) % filter_col;
    data[index] = 1;
    if (++channel_index == input_channel) {
      channel_index = 0;
    }
  }

  // build one-zero kernel Tensor.
  TensorShape shape(input_type, kernel_dims);
  Buffer buffer(shape);
  buffer.copy_from(data.data());
  return edsl::Constant(buffer, "Kernel");
}

}  // namespace

namespace PlaidMLPlugin {

void registerExtractImagePatches() {
  registerOp("ExtractImagePatches", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset3::ExtractImagePatches>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 1);

    auto input_tensor = ctx.operands.at(0);
    auto filter_row = layer->get_sizes()[0];
    auto filter_col = layer->get_sizes()[1];

    edsl::Tensor kernel_tensor;
    switch (input_tensor.dtype()) {
      case DType::FLOAT32:
        kernel_tensor = create_kernel_tensor<float>(filter_row, filter_col, input_tensor);
        break;
      case DType::INT32:
        kernel_tensor = create_kernel_tensor<int>(filter_row, filter_col, input_tensor);
        break;
    }

    std::vector<size_t> strides;
    for (auto stride : layer->get_strides()) {
      strides.push_back(stride);
    }

    std::vector<size_t> dilations;
    for (auto dilation : layer->get_rates()) {
      dilations.push_back(dilation);
    }

    auto autopad_mode = to_plaidml(layer->get_auto_pad());
    if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
      THROW_IE_EXCEPTION << "only valid or auto_pad(same_upper or same_lower) "
                            "PadType is accepted";
    }

    auto result = op::convolution(input_tensor, kernel_tensor)
                      .strides(strides)
                      .dilations(dilations)
                      .autopad_mode(autopad_mode)
                      .input_layout(plaidml::op::TensorLayout::NCX)
                      .filter_layout(plaidml::op::TensorLayout::KCX);
    return edsl::make_tuple(result);
  });
}

}  // namespace PlaidMLPlugin
