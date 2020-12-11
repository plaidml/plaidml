// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static void registerUnaryOp(const std::string& name, const std::string& op) {
  registerOp(name, [op](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 1);
    return edsl::make_tuple(edsl::intrinsicCall(op, ctx.operands));
  });
}

template <typename T>
static void registerUnaryOpCall(const std::string& name, T op) {
  registerOp(name, [op](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 1);
    edsl::Tensor I = ctx.operands.at(0);
    return edsl::make_tuple(op(I));
  });
}

void registerUnaryOps() {
  registerUnaryOpCall("abs", op::abs);
  registerUnaryOp("acos", "acos");
  registerUnaryOp("acosh", "acosh");
  registerUnaryOp("asin", "asin");
  registerUnaryOp("asinh", "asinh");
  registerUnaryOp("atan", "atan");
  registerUnaryOp("atanh", "atanh");
  registerUnaryOp("ceiling", "ceil");
  registerUnaryOp("cos", "cos");
  registerUnaryOp("cosh", "cosh");
  registerUnaryOp("erf", "erf");
  registerUnaryOp("exp", "exp");
  registerUnaryOp("floor", "floor");
  registerUnaryOp("LogicalNot", "logical_not");
  registerUnaryOp("Negative", "neg");
  registerUnaryOpCall("sigmoid", op::sigmoid);
  registerUnaryOp("sin", "sin");
  registerUnaryOp("sinh", "sinh");
  registerUnaryOp("sqrt", "sqrt");
  registerUnaryOp("tan", "tan");
  registerUnaryOp("tanh", "tanh");
}

}  // namespace PlaidMLPlugin
