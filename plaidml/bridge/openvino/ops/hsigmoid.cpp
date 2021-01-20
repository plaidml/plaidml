// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerHSigmoid() {
  registerOp("HSigmoid", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 1);
    auto I = ctx.operands.at(0);
    auto zero = edsl::cast(edsl::Tensor{0}, I.dtype());
    auto six = edsl::cast(edsl::Tensor{6}, I.dtype());
    return edsl::make_tuple(op::minimum(op::maximum(I + 3, zero), six) / 6);
  });
}

}  // namespace PlaidMLPlugin
