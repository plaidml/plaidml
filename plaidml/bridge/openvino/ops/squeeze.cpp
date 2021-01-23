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

namespace {

// TODO: Remove and replace use with get_axis_set_from_constant_operand once upstream fixed for negatives
template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION << " input [1] is Unsupported inputType; ";
  }
}

}  // namespace

void registerSqueeze() {
  registerOp("squeeze", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto axes = cast_constant_operand<int64_t>(1, ctx.layer);
    return edsl::make_tuple(op::squeeze(I, axes));
  });
}

}  // namespace PlaidMLPlugin
