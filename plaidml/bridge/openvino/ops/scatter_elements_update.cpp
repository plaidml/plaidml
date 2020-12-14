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

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION << "Dynamic slicing not currently supported by PlaidML plugin; all of axes must be Constants.";
  }
}

}  // namespace

namespace PlaidMLPlugin {

void registerScatterElementsUpdate() {
  registerOp("ScatterElementsUpdate", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::ScatterElementsUpdate>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 4);
    auto data = ctx.operands.at(0);
    auto indices = ctx.operands.at(1);
    auto updates = ctx.operands.at(2);
    auto axis = cast_constant_operand<int32_t>(3, layer)[0];

    auto O = edsl::scatter(data, indices, updates).axis(axis).mode(edsl::ScatterMode::UPDATE_ELT);
    return edsl::make_tuple(O);
  });
}

}  // namespace PlaidMLPlugin
