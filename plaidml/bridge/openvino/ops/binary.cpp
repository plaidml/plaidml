// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static void registerBinaryOp(const std::string& name, const std::string& op) {
  registerOp(name, [op](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    return edsl::make_tuple(edsl::intrinsicCall(op, ctx.operands));
  });
}

template <typename T>
static void registerBinaryOpCall(const std::string& name, T op) {
  registerOp(name, [op](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto X = ctx.operands.at(0);
    auto Y = ctx.operands.at(1);
    return edsl::make_tuple(op(X, Y));
  });
}

void registerBinaryOps() {
  registerBinaryOp("Add", "add");
  registerBinaryOp("Divide", "div");
  registerBinaryOp("Equal", "cmp_eq");
  registerBinaryOp("GreaterEqual", "cmp_ge");
  registerBinaryOp("Greater", "cmp_gt");
  registerBinaryOp("LessEqual", "cmp_le");
  registerBinaryOp("Less", "cmp_lt");
  registerBinaryOp("LogicalAnd", "logical_and");
  registerBinaryOp("LogicalOr", "logical_or");
  registerBinaryOpCall("Maximum", op::maximum);
  registerBinaryOpCall("Minimum", op::minimum);
  registerBinaryOp("Mod", "mod");
  registerBinaryOp("Multiply", "mul");
  registerBinaryOp("NotEqual", "cmp_ne");
  registerBinaryOp("Power", "pow");
  registerBinaryOp("Subtract", "sub");
}

}  // namespace PlaidMLPlugin
