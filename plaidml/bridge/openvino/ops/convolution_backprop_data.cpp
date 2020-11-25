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

void registerConvolutionBackpropData() {
  registerOp("ConvolutionBackpropData", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::ConvolutionBackpropData>(ctx.layer);
    IE_ASSERT(ctx.operands.size() >= 2);
    IE_ASSERT(ctx.operands.size() <= 3);
    auto I = ctx.operands.at(0);
    auto F = ctx.operands.at(1);
    auto autopad_mode = to_plaidml(layer->get_auto_pad());
    auto result = op::convolution(I, F)
                      .deriv_mode(plaidml::op::ConvDerivMode::DATA)
                      .strides(layer->get_strides())
                      .dilations(layer->get_dilations())
                      .autopad_mode(autopad_mode)
                      .input_layout(plaidml::op::TensorLayout::NCX)
                      .filter_layout(plaidml::op::TensorLayout::KCX);
    if (ctx.operands.size() == 3) {
      auto result_shape = layer->get_output_shape();
      if (!result_shape.is_static()) {
        THROW_IE_EXCEPTION << "Dynamic conv backprop output_shape not supported";
      }
      result.result_shape(result_shape.to_shape());
    } else {
      result.infer_result_shape(true);
    }
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
