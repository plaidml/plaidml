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

static OpRegistration reg("pad", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::Pad*>(ctx.layer);
  IE_ASSERT((ctx.operands.size() == 3) || (ctx.operands.size() == 4));

  auto I = ctx.operands.at(0);
  std::vector<int> lo_pads;
  auto lopadset = get_axis_vector_from_constant_operand(1, ctx.layer);
  for (auto p : lopadset) {
    lo_pads.push_back(p);
  }
  std::vector<int> hi_pads;
  auto hipadset = get_axis_vector_from_constant_operand(2, ctx.layer);
  for (auto p : hipadset) {
    hi_pads.push_back(p);
  }
  float padval = 0.;  // OV tests currently only use default constant pad of 0
  if (ctx.operands.size() == 4) {
    auto padvalset = get_axis_set_from_constant_operand(3, ctx.layer);
    padval = *padvalset.begin();
  }

  auto autopad_mode = to_plaidml(layer->get_pad_mode());

  return edsl::make_tuple(op::explicit_padding(I, lo_pads, hi_pads, autopad_mode, padval));
});

}  // namespace PlaidMLPlugin
