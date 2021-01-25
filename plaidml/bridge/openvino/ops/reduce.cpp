// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerReduceOps() {
  registerOp("ReduceLogicalAnd", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset1::ReduceLogicalAnd>(ctx.layer);
    I = edsl::cast(I, DType::FLOAT32);
    return edsl::make_tuple(op::all(I, edsl::make_tuple(axes), layer->get_keep_dims()));
  });

  registerOp("ReduceLogicalOr", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset1::ReduceLogicalOr>(ctx.layer);
    I = edsl::cast(I, DType::FLOAT32);
    return edsl::make_tuple(op::any(I, edsl::make_tuple(axes), layer->get_keep_dims()));
  });

  registerOp("ReduceMax", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset1::ReduceMax>(ctx.layer);
    return edsl::make_tuple(op::max(I, edsl::make_tuple(axes), layer->get_keep_dims()));
  });

  registerOp("ReduceMean", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset1::ReduceMean>(ctx.layer);
    return edsl::make_tuple(op::mean(I, edsl::make_tuple(axes), layer->get_keep_dims()));
  });

  registerOp("ReduceMin", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset1::ReduceMin>(ctx.layer);
    return edsl::make_tuple(op::min(I, edsl::make_tuple(axes), layer->get_keep_dims()));
  });

  registerOp("ReduceProd", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset1::ReduceProd>(ctx.layer);
    return edsl::make_tuple(op::prod(I, edsl::make_tuple(axes), layer->get_keep_dims()));
  });

  registerOp("ReduceSum", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset1::ReduceSum>(ctx.layer);
    return edsl::make_tuple(op::sum(I, edsl::make_tuple(axes), layer->get_keep_dims()));
  });

  registerOp("ReduceL1", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset4::ReduceL1>(ctx.layer);
    return edsl::make_tuple(op::sum(op::abs(I), edsl::make_tuple(axes), layer->get_keep_dims()));
  });

  registerOp("ReduceL2", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    std::vector<int64_t> axes = cast_constant_operand<int64_t>(1, ctx.layer);
    auto* layer = ngraph::as_type<ngraph::opset4::ReduceL2>(ctx.layer);
    return edsl::make_tuple(edsl::sqrt(op::sum(I * I, edsl::make_tuple(axes), layer->get_keep_dims())));
  });
}

}  // namespace PlaidMLPlugin
