// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("convolution", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::Convolution*>(ctx.layer);
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
  // TODO : padding
  // TODO: auto padding
  return edsl::make_tuple(op::convolution(I, F).strides(strides).dilations(dilations));
});

}  // namespace PlaidMLPlugin
