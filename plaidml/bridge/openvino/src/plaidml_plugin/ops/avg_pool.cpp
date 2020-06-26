// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("avgpool", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::AvgPool*>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  std::vector<int> strides;
  for (auto stride : layer->get_strides()) {
    strides.push_back(stride);
  }
  std::vector<int> kernel;
  for (auto k : layer->get_kernel()) {
    kernel.push_back(k);
  }
  auto pool_type = plaidml::op::PoolMode::AVG;
  auto input_layout = plaidml::op::TensorLayout::NCX;
  auto autopad_mode = to_plaidml(layer->get_auto_pad());
  auto include_padding_in_avg = to_plaidml(layer->get_exclude_pad();
  auto use_ceil_for_output_shape = to_plaidml(layer->get_rounding_type());
  std::vector<int> padding;
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    for (auto pad : layer->get_pads_begin()) {
      padding.push_back(pad);
    }
    for (auto pad : layer->get_pads_end()) {
      padding.push_back(pad);
    }
  }
  return edsl::make_tuple(op::pool(I, pool_type, kernel, strides, autopad_mode, padding, input_layout,
                                   include_padding_in_avg, use_ceil_for_output_shape));
});

}  // namespace PlaidMLPlugin
