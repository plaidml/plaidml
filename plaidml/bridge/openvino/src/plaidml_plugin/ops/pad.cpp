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

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto ngraph_const =
      std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(operand_idx).get_node_shared_ptr());
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION << "Dynamic padding not currently supported by PlaidML plugin; all of pads_begin, pads_end, "
                          "and pads_value must be Constants";
  }
}

}  // namespace

namespace PlaidMLPlugin {

static OpRegistration reg("pad", [](const Context& ctx) {
  auto* layer = dynamic_cast<ngraph::opset1::Pad*>(ctx.layer);
  IE_ASSERT((ctx.operands.size() == 3) || (ctx.operands.size() == 4));

  auto I = ctx.operands.at(0);
  auto lo_pads = cast_constant_operand<int>(1, layer);
  auto hi_pads = cast_constant_operand<int>(2, layer);

  auto autopad_mode = to_plaidml(layer->get_pad_mode());

  auto op = op::explicit_padding(I, lo_pads, hi_pads).mode(autopad_mode);

  if (ctx.operands.size() == 4) {
    op.padval(edsl::Value(ctx.operands.at(0)));
  }

  return edsl::make_tuple(op);
});

}  // namespace PlaidMLPlugin
