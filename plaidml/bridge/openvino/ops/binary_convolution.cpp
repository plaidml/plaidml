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

namespace PlaidMLPlugin {

static OpRegistration reg("binaryconvolution", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset4::BinaryConvolution>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto I = ctx.operands.at(0);
  auto F = ctx.operands.at(1);
  std::vector<int> strides;
  for (auto stride : layer->get_strides()) {
    strides.push_back(stride);
  }
  std::vector<int> dilations;
  for (auto dilation : layer->get_dilations()) {
    dilations.push_back(dilation);
  }

  auto mode = layer->get_mode();
  edsl::Tensor input_tensor, filter_tensor;
  IE_ASSERT(mode == ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT);
  auto one = cast(edsl::Tensor(1), DType::INT32);
  auto minus_one = cast(edsl::Tensor(-1), DType::INT32);
  input_tensor = edsl::select(I == 0, minus_one, one);
  filter_tensor = edsl::select(F == 0, minus_one, one);

  auto autopad_mode = to_plaidml(layer->get_auto_pad());
  auto pad_value = layer->get_pad_value();
  auto result = op::convolution(input_tensor, filter_tensor)
                    .strides(strides)
                    .dilations(dilations)
                    .autopad_mode(autopad_mode)
                    .input_layout(plaidml::op::TensorLayout::NCX)
                    .filter_layout(plaidml::op::TensorLayout::KCX);
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    int padding = static_cast<int>(pad_value);
    result.manual_padding({padding});
  }
  return edsl::make_tuple(result);
});

}  // namespace PlaidMLPlugin
