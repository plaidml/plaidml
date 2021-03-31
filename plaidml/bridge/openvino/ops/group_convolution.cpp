// Copyright (C) 2021 Intel Corporation
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

void registerGroupConvolution() {
  registerOp("GroupConvolution", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::GroupConvolution>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto F = ctx.operands.at(1);
    auto autopad_mode = to_plaidml(layer->get_auto_pad());
    auto result = op::convolution(I, F)
                      .strides(layer->get_strides())
                      .dilations(layer->get_dilations())
                      .autopad_mode(autopad_mode)
                      .autogroup_mode(plaidml::op::AutoGroupMode::AUTO)
                      .input_layout(plaidml::op::TensorLayout::NCX)
                      .filter_layout(plaidml::op::TensorLayout::GKCX)
                      .group_layout(plaidml::op::GroupLayout::SEPARATE);
    if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
      std::vector<int> padding;
      for (auto pad : layer->get_pads_begin()) {
        padding.push_back(pad);
      }
      for (auto pad : layer->get_pads_end()) {
        padding.push_back(pad);
      }
      result.manual_padding(padding);
    }
    return edsl::make_tuple(result);
  });
}

}  // namespace PlaidMLPlugin
